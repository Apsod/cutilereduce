from __future__ import annotations
from dataclasses import dataclass
import linecache
import math
import uuid

import cuda.tile as ct
import torch

from .base import Phase
from .buffer import BufferDep, BufferRole, ConcreteBuffer
from .grid import ConcreteGrid
from .spec import ConcreteSpec

def ceil_pow2(x: int):
    return 1 << (x-1).bit_length()

def floor_pow2(x: int):
    return 1 << (x.bit_length() - 1)

@dataclass(frozen=True)
class TileCtx:
    tile: ct.Tile
    shape: tuple[int,...]
    index: tuple[int,...]

    def indices_along(self, i):
        s = self.tile.shape[i]
        return ct.arange(s, dtype=ct.int32) + self.index[i] * s

    def start_along(self, i):
        return self.tile.shape[i] * self.index[i]

    def stop_along(self, i):
        return self.tile.shape[i] * (self.index[i] + 1)

    def mask_along(self, i):
        return self.indices_along(i) < self.shape[i]

@dataclass(frozen=True)
class View:
    view: ct.TiledView
    buffer_shape: tuple[int,...]
    grid_index: tuple[int,...]

    def load_tilectx(self, tid):
        ixs = retile(tid, self.grid_index)
        return TileCtx(self.view.load(ixs), self.buffer_shape, ixs)

    def tid2vid(self, tid):
        return retile(tid, self.grid_index)

    def load(self, tid):
        return self.view.load(self.tid2vid(tid))

    def store(self, tid, tile):
        return self.view.store(self.tid2vid(tid), tile)

    def atomic_add(self, tid, tile):
        return self.view.atomic_store_add(self.tid2vid(tid), tile)

@ct.function
def retile(original, index):
    ret = ()
    for i in ct.static_iter(index):
        ret += (original[i],)
    return ret

@ct.function
def add_at(original, index, value):
    return (*original[:index], original[index] + value, original[index+1:])

@ct.function
def set_at(original, index, value):
    return (*original[:index], value, original[index+1:])

def make_loads(num, *, tilectx=False):
    if tilectx:
        @ct.function
        def _loads(tid, views):
            tiles = ()
            for i in ct.static_iter(range(num)):
                tiles += (views[i].load_tilectx(tid),)
            return tiles
    else:
        @ct.function
        def _loads(tid, views):
            tiles = ()
            for i in ct.static_iter(range(num)):
                tiles += (views[i].load(tid),)
            return tiles
    return _loads


def make_stores(num):
    @ct.function
    def _stores(tid, views, tiles):
        for i in ct.static_iter(range(num)):
            views[i].store(tid, tiles[i])
    return _stores

def make_pads(num):
    @ct.function
    def _pads(tiles):
        ret = ()
        for i in ct.static_iter(range(num)):
            ret += (tiles[i][None],)
        return ret
    return _pads

def make_views(buffer_specs):
    @ct.function
    def _views(buffers):
        views = ()
        for i, grid_index, tile_shape in ct.static_iter(
                (i, b.grid_index, b.tile_shape)
                for i, b
                in enumerate(buffer_specs)
                ):
            buffer = buffers[i]
            view = buffer.tiled_view(tile_shape)
            views += (View(view, buffer.shape, grid_index),)
        return views
    return _views

def mk_bwd_kernel(spec, map_finalize, embed):
    assert spec.phase == Phase.bwd
    grid = spec.grid
    group_dim = grid.group_dim
    gix = group_dim.grid_index

    group_tiles = group_dim.num_tiles
    group_size_base = group_tiles // spec.groups
    group_remainder = group_tiles % spec.groups

    batch_buffers = tuple(b for b in spec.read_buffers if not b.is_grouped)
    group_buffers = tuple(b for b in spec.read_buffers if b.is_grouped)

    grad_buffers = tuple(b for b in spec.grad_buffers)

    load_order = tuple(b.program_index for b in batch_buffers + group_buffers)

def mk_fwd_kernel(spec, map_reduce, combine):
    assert spec.phase == Phase.fwd
    grid = spec.grid
    group_dim = grid.group_dim
    gix = group_dim.grid_index

    groups = spec.groups

    group_tiles = group_dim.num_tiles
    group_size_base = group_tiles // groups
    group_remainder = group_tiles % groups

    ratio = int(spec.eval('residency_bytes') / spec.eval('output_bytes'))
    combine_tile = floor_pow2(ratio)
    combine_tile = min(combine_tile, ceil_pow2(groups))
    combine_steps = ct.cdiv(groups, combine_tile)

    identity = tuple(b.default for b in spec.output_buffers)

    batch_buffer_index, batch_buffer_specs = zip(*(
        (i,b) 
        for (i,b) 
        in enumerate(spec.input_buffers) 
        if not b.is_grouped))

    group_buffer_index, group_buffer_specs = zip(*(
        (i,b) 
        for (i,b) 
        in enumerate(spec.input_buffers) 
        if b.is_grouped))
    
    intermediate_specs = tuple(b.make_derived(gix, groups, 1) for b in spec.output_buffers)
    combine_specs = tuple(b.make_derived(gix, groups, combine_tile) for b in spec.output_buffers)
    
    num_output_buffers = len(spec.output)
    num_batch_buffers = len(batch_buffer_specs)
    num_group_buffers = len(group_buffer_specs)

    load_order = batch_buffer_index + group_buffer_index

    load_batch = make_loads(num_batch_buffers, tilectx=True)
    view_batch = make_views(batch_buffer_specs)

    load_group = make_loads(num_group_buffers, tilectx=True)
    view_group = make_views(group_buffer_specs)

    load_output = make_loads(num_output_buffers)
    store_output = make_stores(num_output_buffers)
    pad_output = make_pads(num_output_buffers)

    view_intermediate = make_views(intermediate_specs)
    view_combine = make_views(combine_specs)
    
    @ct.function
    def get_last(tiles):
        lasts = ()
        for i, tile_shape, index in ct.static_iter(
                (b.program_index, b.dims.tile_shape, (0,)*len(b.dims))
                for b
                in spec.output_buffers
                ):
            index = (combine_tile-1, *index)
            shape = (1, *tile_shape)
            lasts += (ct.extract(tiles[i], index, shape),)
        return lasts

    @ct.function
    def init():
        pid = ct.bid(0)
        tid = ()
        for s in ct.static_iter(grid.task_grid):
            lid = pid % s
            tid += (lid,)
            pid = pid // s

        gid = tid[gix]
        offset = (gid * group_size_base + ct.minimum(gid, group_remainder))
        size = (group_size_base + (gid < group_remainder))

        return tid, offset, size

    @ct.function
    def increment_and_check_write_lock(lock, pid):
        ixs = ct.static_eval(tuple(d.grid_index for d in spec.grid.batch))
        written = ct.atomic_add(lock, retile(pid, ixs), 1)
        return written == groups - 1

    @ct.function
    def load_map_reduce(tid, btiles, group):
        gtiles = load_group(tid, group)
        return map_reduce(*retile(btiles + gtiles, load_order))

    @ct.function
    def load_scan(tid, intermediate):
        return ct.scan(load_output(tid, intermediate), 0, combine, identity)

    @ct.function
    def fwd(lock, batch_buffers, group_buffers, output_buffers):
        pid, goffset, gsize = init()
        
        batch = view_batch(batch_buffers)
        group = view_group(group_buffers)
        
        tid = set_at(pid, gix, goffset)

        btiles = load_batch(tid, batch)
        acc = load_map_reduce(tid, btiles, group)
        
        for i in range(1, gsize):
            acc = combine(
                *acc, 
                *load_map_reduce(add_at(tid, gix, i), btiles, group)
                )

        output = view_intermediate(output_buffers)
        store_output(pid, output, pad_output(acc))
        done = increment_and_check_write_lock(lock, pid)

        if done:
            tid = set_at(pid, gix, 0)
            intermediate = view_combine(output_buffers)
            local = load_scan(tid, intermediate)
            store_output(tid, intermediate, local) # IF NOT COMM
            for i in range(1, combine_steps):
                boundary = get_last(local)
                tid = set_at(pid, gix, i)
                local = load_scan(tid, intermediate)
                local = combine(*boundary, *local)
                store_output(tid, intermediate, local) # IF NOT COMM
            #store_output(tid, intermediate, local) # IF COMM
    
    lock = 'lock'
    input_args = tuple(f'in_{i}' for i in range(len(spec.input)))
    batch_args = tuple(f'in_{i}' for i in batch_buffer_index)
    group_args = tuple(f'in_{i}' for i in group_buffer_index)
    output_args = tuple(f'out_{i}' for i in range(len(spec.output)))

    def csv(*args):
        return ','.join(args)
    def tuplify(*args):
        return ('(' + ','.join(args) + ',)')
    ns = {'ct': ct, 'fwd': fwd}

    source = (
            "@ct.kernel\n"
            f"def kernel({csv(lock, *input_args, *output_args)}):\n"
            f"  fwd(lock, {tuplify(*batch_args)}, {tuplify(*group_args)}, {tuplify(*output_args)})\n"
            )

    filename = f"<cutilereduce_kernel_{uuid.uuid4().hex}>"
    linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
    )

    code = compile(source, filename, "exec")
    exec(code, ns)
    return ns['kernel'], combine_specs

def mk_fwd(spec, map_reduce, combine, to_semantic=None, to_output=None):
    kernel, out_spec = mk_fwd_kernel(spec, map_reduce, combine)
    def fwd(*inputs):
        outputs = tuple(b.empty('cuda') for b in out_spec)
        lock_shape = tuple(b.num_programs for b in spec.grid.batch)
        lock = torch.zeros(lock_shape, device='cuda', dtype=torch.int32)
        args = (lock, *inputs, *outputs)
        launch_grid = (spec.grid.tasks, 1, 1)
        ct.launch(torch.cuda.current_stream(), launch_grid, kernel, args)
        outputs = tuple(b[-1] for b in outputs)
        if to_semantic is not None:
            outputs = to_semantic(*outputs)
        if to_output is not None:
            outputs = to_output(*outputs)
        return outputs
    return fwd
