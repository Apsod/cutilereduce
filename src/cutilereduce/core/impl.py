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


    num_output_buffers = len(spec.output)

    ratio = int(spec.eval('residency_bytes') / spec.eval('output_bytes'))
    combine_tile = floor_pow2(ratio)
    combine_tile = min(combine_tile, ceil_pow2(groups))
    combine_steps = ct.cdiv(groups, combine_tile)

    identity = tuple(b.default for b in spec.output_buffers)

    batch_buffer_specs = tuple(b for b in spec.read_buffers if not b.is_grouped)
    group_buffer_specs = tuple(b for b in spec.read_buffers if b.is_grouped)

    num_batch_buffers = len(batch_buffer_specs)
    num_group_buffers = len(group_buffer_specs)


    load_order = tuple(b.program_index for b in batch_buffer_specs + group_buffer_specs)
    
    @ct.function
    def split_input(buffers):
        batch = ()
        group = ()
        for i, grid_index, tile_shape, total_shape in ct.static_iter(
                (b.program_index, b.dims.grid_index, b.dims.tile_shape, b.dims.shape)
                for b
                in batch_buffer_specs
                ):
            view = buffers[i].tiled_view(tile_shape)
            batch += (View(view, total_shape, grid_index),)
        for i, grid_index, tile_shape, total_shape in ct.static_iter(
                (b.program_index, b.dims.grid_index, b.dims.tile_shape, b.dims.shape)
                for b
                in group_buffer_specs
                ):
            view = buffers[i].tiled_view(tile_shape)
            group += (View(view, total_shape, grid_index),)
        return batch, group

    @ct.function
    def ctx_loads(tid, views, num):
        tiles = ()
        for i in ct.static_iter(range(num)):
            tiles += (views[i].load_tilectx(tid),)
        return tiles

    @ct.function
    def loads(tid, views, num):
        tiles = ()
        for i in ct.static_iter(range(num)):
            tiles += (views[i].load(tid),)
        return tiles

    @ct.function
    def stores(tid, views, tiles, num):
        for i in ct.static_iter(range(num)):
            views[i].store(tid, tiles[i])


    @ct.function
    def store_intermediate(buffers, tid, tiles):
        for i, ixs, shape in ct.static_iter(
                (b.program_index, b.dims.grid_index, b.dims.tile_shape)
                for b
                in spec.write_buffers
                ):
            shape = (1, *shape)
            ixs = (gix, *ixs)
            index = retile(tid, ixs)
            view = buffers[i].tiled_view(shape)
            view.store(index, tiles[i])

    @ct.function
    def mk_intermediate_views(buffers, gtile):
        views = ()
        for i, grid_index, tile_shape, total_shape in ct.static_iter(
                (b.program_index, b.dims.grid_index, b.dims.tile_shape, b.dims.shape)
                for b 
                in spec.output_buffers
            ):
            shape = (gtile, *tile_shape)
            grid_index = (gix, *grid_index)
            view = buffers[i].tiled_view(shape)
            views += (View(view, total_shape, grid_index),)
        return views

    @ct.function
    def load_intermediates(tid, views, num):
        tiles = ()
        for i, id in ct.static_iter(enumerate(identity)):
            tile = views[i].load_tilectx(tid)
            tiles += (views[i].load(tid),)
        return tiles

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
    def increment_and_check_written_lock(lock, pid):
        ixs = ct.static_eval(tuple(d.grid_index for d in spec.grid.batch))
        written = ct.atomic_add(lock, retile(pid, ixs), 1)
        return written == groups - 1


    @ct.function
    def fwd(lock, input_buffers, output_buffers):
        pid, goffset, gsize = init()
        
        batch, group = split_input(input_buffers)
        
        tid = set_at(pid, gix, goffset)
        
        btiles = ctx_loads(tid, batch, num_batch_buffers)
        gtiles = ctx_loads(tid, group, num_group_buffers)

        tiles = retile(btiles + gtiles, load_order)
        acc = map_reduce(*tiles)
        
        for i in range(1, gsize):
            gtiles = ctx_loads(add_at(tid, gix, i), group, num_group_buffers)
            tiles = retile(btiles + gtiles, load_order)
            acc = combine(*acc, *map_reduce(*tiles))

        output = mk_intermediate_views(output_buffers, 1)
        stores(pid, output, acc, num_output_buffers)
        done = increment_and_check_written_lock(lock, pid)

        if done:
            # Last worker to finish runs final scan.
            tid = set_at(pid, gix, 0)
            intermediate = mk_intermediate_views(output_buffers, combine_tile)
            local = ct.scan(loads(tid, intermediate, num_output_buffers), 0, combine, identity)
            stores(tid, intermediate, local, num_output_buffers)
            for i in range(1, combine_steps):
                boundary = get_last(local)
                tid = set_at(pid, gix, i)
                local = ct.scan(loads(tid, intermediate, num_output_buffers), 0, combine, identity)
                local = combine(*boundary, *local)
                stores(tid, intermediate, local, num_output_buffers)
    
    lock = 'lock'
    input_args = tuple(f'in_{i}' for i in range(len(spec.input)))
    batch_args = tuple(f'in_{b.program_index}' for b in batch_buffer_specs)
    group_args = tuple(f'in_{b.program_index}' for b in group_buffer_specs)
    output_args = tuple(f'out_{i}' for i in range(len(spec.output)))

    def csv(*args):
        return ','.join(args)
    def tuplify(*args):
        return ('(' + ','.join(args) + ')')
    ns = {'ct': ct, 'fwd': fwd}

    source = (
            "@ct.kernel\n"
            f"def kernel({csv(lock, *input_args, *output_args)}):\n"
            f"  fwd(lock, {tuplify(*input_args)}, {tuplify(*output_args)})\n"
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
    return ns['kernel']

def mk_fwd(spec, map_reduce, combine, to_semantic=None, to_output=None):
    kernel = mk_fwd_kernel(spec, map_reduce, combine)
    def fwd(*inputs):
        outputs = tuple(b.init_buffer('cuda', extra=spec.groups) for b in spec.output.values())
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
