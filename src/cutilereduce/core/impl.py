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

def csv(*args):
    return ','.join(args)
def tuplify(*args):
    return ('(' + ','.join(args) + ',)')

@dataclass(frozen=True)
class View:
    view: ct.TiledView
    grid_index: tuple[int,...]

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

def make_set_gix(grid):
    @ct.function
    def _set_gix(original, value):
        ret = ()
        for i in ct.static_iter(range(grid.group_dim.grid_index)):
            ret += (original[i],)
        ret += (value,)
        for i in ct.static_iter(range(grid.group_dim.grid_index+1, len(grid.task_grid))):
            ret += (original[i],)
        return ret
    return _set_gix

def make_views(buffer_specs):
    @ct.function
    def _views(buffers):
        views = ()
        for i, grid_index, tile_shape, padding_mode in ct.static_iter(
                (i, b.grid_index, b.tile_shape, b.padding_mode)
                for i, b
                in enumerate(buffer_specs)
                ):
            buffer = buffers[i]
            view = buffer.tiled_view(tile_shape, padding_mode=padding_mode)
            views += (View(view, grid_index),)
        return views
    return _views

def make_loads(buffer_specs):
    num = len(buffer_specs)

    @ct.function
    def _loads(tid, views):
        tiles = ()
        for i in ct.static_iter(range(num)):
            tiles += (views[i].load(tid),)
        return tiles

    return _loads

def make_fused_loads(buffer_specs):
    @ct.function
    def _fused_loads(tid, buffers):
        tiles = ()
        for i, grid_index, tile_shape, padding_mode in ct.static_iter(
                (i, b.grid_index, b.tile_shape, b.padding_mode)
                for i, b
                in enumerate(buffer_specs)
                ):
            buffer = buffers[i]
            view = buffer.tiled_view(tile_shape, padding_mode=padding_mode)
            tiles += (View(view, grid_index).load(tid),)
        return tiles
    return _fused_loads

def make_stores(buffer_specs):
    num = len(buffer_specs)

    @ct.function
    def _stores(tid, views, tiles):
        for i in ct.static_iter(range(num)):
            views[i].store(tid, tiles[i])
    return _stores

def make_fused_stores(buffer_specs):
    @ct.function
    def _fused_stores(tid, buffers, tiles):
        for i, grid_index, tile_shape in ct.static_iter(
                (i, b.grid_index, b.tile_shape)
                for i, b
                in enumerate(buffer_specs)
                ):
            buffer = buffers[i]
            view = buffer.tiled_view(tile_shape)
            View(view, grid_index).store(tid, tiles[i])
    return _fused_stores

def make_pads(buffer_specs):
    num = len(buffer_specs)

    @ct.function
    def _pads(tiles):
        ret = ()
        for i in ct.static_iter(range(num)):
            ret += (tiles[i][None],)
        return ret
    return _pads


class Bundle:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return tuple(getattr(self, k) for k in key)
        else:
            return getattr(self, key)


def make_tid_info(grid):
    
    dims = grid.dims

    print(dims)

    @dataclass(frozen=True)
    class TidInfo:
        tid: tuple[int,...]

        def shape(self, dim, *more):
            s = ct.static_eval(dims.get(dim).tile_exp)
            if more:
                ret = (s,)
                for s in ct.static_iter(
                        dims.get(d).tile_exp
                        for d
                        in more
                        ):
                    ret += (s,)
                return ret
            else:
                return s

        def offset(self, dim):
            i = ct.static_eval(dims.get(dim).grid_index)
            return self.tid[i] * ct.static_eval(dims.get(dim).tile_exp)

        def indices(self, dim):
            i = ct.static_eval(dims.get(dim).grid_index)
            s = ct.static_eval(dims.get(dim).tile_exp)
            return self.tid[i] * s + ct.arange(s, dtype=ct.int32)

        def mask(self, dim):
            i = ct.static_eval(dims.get(dim).grid_index)
            s = ct.static_eval(dims.get(dim).tile_exp)
            b = ct.static_eval(dims.get(dim).total_var)
            return (self.tid[i] * s - b) + ct.arange(s, dtype=ct.int32) < 0 

    return TidInfo

def make_buffer_helper(buffer_specs):
    loads = make_loads(buffer_specs)
    stores = make_stores(buffer_specs)
    views = make_views(buffer_specs)
    pads = make_pads(buffer_specs)
    fused_loads = make_fused_loads(buffer_specs)
    fused_stores = make_fused_stores(buffer_specs)
    return Bundle(load=loads, store=stores, view=views, pad=pads, fused_load=fused_loads, fused_store=fused_stores)

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

def mk_fwd_no_group_kernel(spec, map_reduce, combine):
    assert spec.phase == Phase.fwd
    assert spec.groups == 1
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


    load_order = batch_buffer_index + group_buffer_index

    set_gix = make_set_gix(spec.grid)
    tid_info = make_tid_info(spec.grid)

    load_batch = make_buffer_helper(batch_buffer_specs)['fused_load']
    load_group, view_group = make_buffer_helper(group_buffer_specs)['load', 'view']
    store_output = make_buffer_helper(spec.output_buffers)['fused_store']
    gsize = spec.grid.group_dim.num_tiles
    gix = spec.grid.group_dim.grid_index

    @ct.function
    def load_map_reduce(tid, batch_tiles, group_view):
        group_tiles = load_group(tid, group_view)
        return map_reduce(tid_info(tid), *retile(batch_tiles + group_tiles, load_order))

    @ct.function
    def init():
        pid = ct.bid(0)
        tid = ()
        for s in ct.static_iter(spec.grid.task_grid):
            lid = pid % s
            tid += (lid,)
            pid = pid // s
        return tid

    @ct.function
    def fwd(batch_buffers, group_buffers, output_buffers):
        tid = init()
        
        batch_tiles = load_batch(tid, batch_buffers)
        group_view = view_group(group_buffers)
        
        acc = load_map_reduce(tid, batch_tiles, group_view)
        
        for i in range(1, gsize):
            acc = combine(
                *acc, 
                *load_map_reduce(set_gix(tid, i), batch_tiles, group_view)
                )

        store_output(tid, output_buffers, acc)

    ns = {'ct': ct, 'fwd': fwd}

    input_args = tuple(f'in_{i}' for i in range(len(spec.input)))
    batch_args = tuple(f'in_{i}' for i in batch_buffer_index)
    group_args = tuple(f'in_{i}' for i in group_buffer_index)
    output_args = tuple(f'out_{i}' for i in range(len(spec.output)))

    source = (
            "@ct.kernel\n"
            f"def kernel({csv(*input_args, *output_args)}):\n"
            f"  fwd({tuplify(*batch_args)}, {tuplify(*group_args)}, {tuplify(*output_args)})\n"
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

def mk_fwd_no_group(spec, map_reduce, combine, to_semantic=None, to_output=None):
    kernel = mk_fwd_no_group_kernel(spec, map_reduce, combine)
    def fwd(*inputs):
        outputs = tuple(b.empty('cuda') for b in spec.output_buffers)
        args = (*inputs, *outputs)
        launch_grid = (spec.grid.tasks, 1, 1)
        ct.launch(torch.cuda.current_stream(), launch_grid, kernel, args)
        if to_semantic is not None:
            outputs = to_semantic(*outputs)
        if to_output is not None:
            outputs = to_output(*outputs)
        return outputs
    return fwd

def mk_fwd(spec, map_reduce, combine, to_semantic, to_output):
    raise NotImplementedError()

#def mk_fwd_kernel(spec, map_reduce, combine):
#    assert spec.phase == Phase.fwd
#    grid = spec.grid
#    group_dim = grid.group_dim
#    gix = group_dim.grid_index
#
#    groups = spec.groups
#
#    group_tiles = group_dim.num_tiles
#    group_size_base = group_tiles // groups
#    group_remainder = group_tiles % groups
#
#    ratio = int(spec.eval('residency_bytes') / spec.eval('output_bytes'))
#    combine_tile = floor_pow2(ratio)
#    combine_tile = min(combine_tile, ceil_pow2(groups))
#    combine_steps = ct.cdiv(groups, combine_tile)
#
#    identity = tuple(b.default for b in spec.output_buffers)
#
#    batch_buffer_index, batch_buffer_specs = zip(*(
#        (i,b) 
#        for (i,b) 
#        in enumerate(spec.input_buffers) 
#        if not b.is_grouped))
#
#    group_buffer_index, group_buffer_specs = zip(*(
#        (i,b) 
#        for (i,b) 
#        in enumerate(spec.input_buffers) 
#        if b.is_grouped))
#
#    output_specs = spec.output_buffers    
#    intermediate_specs = tuple(b.make_derived(gix, groups, 1) for b in spec.output_buffers)
#    combine_specs = tuple(b.make_derived(gix, groups, combine_tile) for b in spec.output_buffers)
#
#    load_order = batch_buffer_index + group_buffer_index
#
#    load_batch, view_batch = make_buffer_helper(batch_buffer_specs)['load', 'view']
#    load_group, view_group = make_buffer_helper(group_buffer_specs)['load', 'view']
#    store_output = make_buffer_helper(output_specs)['store']
#
#    load_output, store_output, pad_output = make_buffer_helper(output_specs)['load', 'store', 'pad']
#
#    view_intermediate = make_buffer_helper(intermediate_specs)['view']
#    view_combine = make_buffer_helper(combine_specs)['view']
#    
#    @ct.function
#    def get_last(tiles):
#        lasts = ()
#        for i, tile_shape, index in ct.static_iter(
#                (b.program_index, b.dims.tile_shape, (0,)*len(b.dims))
#                for b
#                in spec.output_buffers
#                ):
#            index = (combine_tile-1, *index)
#            shape = (1, *tile_shape)
#            lasts += (ct.extract(tiles[i], index, shape),)
#        return lasts
#
#    @ct.function
#    def init():
#        pid = ct.bid(0)
#        tid = ()
#        for s in ct.static_iter(grid.task_grid):
#            lid = pid % s
#            tid += (lid,)
#            pid = pid // s
#
#        gid = tid[gix]
#        offset = (gid * group_size_base + ct.minimum(gid, group_remainder))
#        size = (group_size_base + (gid < group_remainder))
#
#        return tid, offset, size
#
#    @ct.function
#    def increment_and_check_write_lock(lock, pid):
#        ixs = ct.static_eval(tuple(d.grid_index for d in spec.grid.batch))
#        written = ct.atomic_add(lock, retile(pid, ixs), 1)
#        return written == groups - 1
#
#    @ct.function
#    def load_map_reduce(tid, btiles, group):
#        gtiles = load_group(tid, group)
#        return map_reduce(*retile(btiles + gtiles, load_order))
#
#    @ct.function
#    def load_scan(tid, intermediate):
#        return ct.scan(load_output(tid, intermediate), 0, combine, identity)
#
#    @ct.function
#    def fwd(lock, batch_buffers, group_buffers, output_buffers):
#        pid, goffset, gsize = init()
#        
#        batch = view_batch(batch_buffers)
#        group = view_group(group_buffers)
#        
#        tid = set_at(pid, gix, goffset)
#
#        btiles = load_batch(tid, batch)
#        acc = load_map_reduce(tid, btiles, group)
#        
#        for i in range(1, gsize):
#            acc = combine(
#                *acc, 
#                *load_map_reduce(add_at(tid, gix, i), btiles, group)
#                )
#
#        output = view_intermediate(output_buffers)
#        store_output(pid, output, pad_output(acc))
#        done = increment_and_check_write_lock(lock, pid)
#
#        if done:
#            tid = set_at(pid, gix, 0)
#            intermediate = view_combine(output_buffers)
#            local = load_scan(tid, intermediate)
#            store_output(tid, intermediate, local) # IF NOT COMM
#            for i in range(1, combine_steps):
#                boundary = get_last(local)
#                tid = set_at(pid, gix, i)
#                local = load_scan(tid, intermediate)
#                local = combine(*boundary, *local)
#                store_output(tid, intermediate, local) # IF NOT COMM
#            #store_output(tid, intermediate, local) # IF COMM
#    
#    lock = 'lock'
#    input_args = tuple(f'in_{i}' for i in range(len(spec.input)))
#    batch_args = tuple(f'in_{i}' for i in batch_buffer_index)
#    group_args = tuple(f'in_{i}' for i in group_buffer_index)
#    output_args = tuple(f'out_{i}' for i in range(len(spec.output)))
#    ns = {'ct': ct, 'fwd': fwd}
#
#    source = (
#            "@ct.kernel\n"
#            f"def kernel({csv(lock, *input_args, *output_args)}):\n"
#            f"  fwd(lock, {tuplify(*batch_args)}, {tuplify(*group_args)}, {tuplify(*output_args)})\n"
#            )
#
#    filename = f"<cutilereduce_kernel_{uuid.uuid4().hex}>"
#    linecache.cache[filename] = (
#            len(source),
#            None,
#            source.splitlines(keepends=True),
#            filename,
#    )
#
#    code = compile(source, filename, "exec")
#    exec(code, ns)
#    return ns['kernel'], combine_specs
#
#def mk_fwd(spec, map_reduce, combine, to_semantic=None, to_output=None):
#    kernel, out_spec = mk_fwd_kernel(spec, map_reduce, combine)
#    def fwd(*inputs):
#        outputs = tuple(b.empty('cuda') for b in out_spec)
#        lock_shape = tuple(b.num_programs for b in spec.grid.batch)
#        lock = torch.zeros(lock_shape, device='cuda', dtype=torch.int32)
#        args = (lock, *inputs, *outputs)
#        launch_grid = (spec.grid.tasks, 1, 1)
#        ct.launch(torch.cuda.current_stream(), launch_grid, kernel, args)
#        outputs = tuple(b[-1] for b in outputs)
#        if to_semantic is not None:
#            outputs = to_semantic(*outputs)
#        if to_output is not None:
#            outputs = to_output(*outputs)
#        return outputs
#    return fwd
