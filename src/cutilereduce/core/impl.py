from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Self
from copy import replace
import math

from .buffer import Buffer

import cuda.tile as ct
import torch

from .base import Phase
from cutilereduce.util.tune import exhaustive_search

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
    grid_index: ct.Constant[tuple[int,...]]

    def tid2vid(self, tid):
        return retile(tid, self.grid_index)

    def load(self, tid):
        return self.view.load(self.tid2vid(tid))

    def store(self, tid, tile):
        return self.view.store(self.tid2vid(tid), tile)

    def atomic_add(self, tid, tile):
        return self.view.atomic_store_add(self.tid2vid(tid), tile)

@ct.function
def retile(original, index: ct.Constant[tuple[int,...]]):
    ret = ()
    for i in ct.static_iter(index):
        ret += (original[i],)
    return ret

def inverse_p(p):
    ret = [None] * len(p)
    for i, j in enumerate(p):
        ret[j] = i
    return tuple(ret)

def ctmap(fun, xs):

    def _ctmap(*args):
        ret = ()
        for x in ct.static_iter(xs):
            ret += (fun(*args, x),)
        return ret
    return _ctmap

def ctzipmap(fun, xs, *, nzips=1):

    def _ctzipmap(*args):
        ret = ()
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i, x in ct.static_iter(enumerate(xs)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            ret += (fun(*static, *current, x),)
        return ret
    return _ctzipmap

def ctzipmaprange(fun, *, nzips=1, num=1):

    def _ctzipmaprange(*args):
        ret = ()
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i in ct.static_iter(range(num)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            ret += (fun(*static, *current),)
        return ret
    return _ctzipmaprange

def ctdo(fun, xs):

    def _ctdo(*args):
        for x in ct.static_iter(xs):
            fun(*args, x)
    return _ctdo

def ctzipdo(fun, xs, *, nzips=1):

    def _ctzipdo(*args):
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i, x in ct.static_iter(enumerate(xs)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            fun(*static, *current, x)
    return _ctzipdo

def ctzipdorange(fun, *, nzips=1, num=1):

    def _ctzipdorange(*args):
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i in ct.static_iter(range(num)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            fun(*static, *current)
    return _ctzipdorange

def ctiter(fun, *, nmut=-0, num=1):

    def _ctiter(*args):
        static = args[:-nmut]
        muts = args[-nmut:]
        for _ in ct.static_iter(range(num)):
            muts = fun(*static, *muts)
        return muts

    return _ctiter

def ctrange(fun, *, nmut=1, num=1):

    def _ctrange(*args):
        static = args[:-nmut]
        muts = args[-nmut:]
        for i in ct.static_iter(range(num)):
            muts = fun(i, *static, *muts)
        return muts

    return _ctrange

@dataclass(frozen=True)
class BufferInfo:
    grid_index: ct.Constant[tuple[int,...]]
    tile_shape: ct.Constant[tuple[int,...]]
    padding_mode: ct.Constant[ct.PaddingMode]
    default: ct.Constant[Any]
    dtype: ct.Constant[ct.DType]
    multiplicity: ct.Constant[int]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return tuple(getattr(self, k) for k in key)
        else:
            return getattr(self, key)


    @staticmethod
    def make(b: Buffer) -> Self:
        return BufferInfo(
                grid_index = b.grid_index,
                tile_shape = b.tile_shape,
                padding_mode = b.padding_mode,
                default = b.default,
                dtype = b.dtype,
                multiplicity = math.ceil(b.residual_multiplicity),
                )

    @staticmethod
    def get(infos, *attributes):
        tuple(info[*attributes] for info in infos)



def make_buffer_helper(buffer_specs):
    num = len(buffer_specs)
    infos = tuple(BufferInfo.make(b) for b in buffer_specs)

    def _view_load(tid, view, info):
        return view.load(retile(tid, info.grid_index))

    def _view_store_add(tid, view, tile, info):
        if info.multiplicity == 1:
            view.store(retile(tid, info.grid_index), tile)
        else:
            view.atomic_store_add(retile(tid, info.grid_index), tile)

    @dataclass(frozen=True)
    class Views:
        views: tuple(ct.TiledView)

        def load(self, tid):
            return ctzipmap(_view_load, infos)(tid, self.views)

        def store_add(self, tid, tiles):
            ctzipdo(_view_store_add, infos, nzips=2)(tid, self.views, tiles)

    def _mk_view(buffer, info):
        return buffer.tiled_view(info.tile_shape, padding_mode=info.padding_mode)

    def _mk_views(buffers):
        return Views(ctzipmap(_mk_view, infos)(buffers))

    def _load(tid, buffer, info):
        return ct.load(buffer, retile(tid, info.grid_index), info.tile_shape, padding_mode=info.padding_mode)
    
    def _store(tid, buffer, tile, info):
        ct.store(buffer, retile(tid, info.grid_index), tile)

    def _init(info):
        return ct.full(info.tile_shape, info.default, info.dtype)

    def _add(a, b):
        return a + b

    def _pad(tile):
        return ct.expand_dim(tile, 0)

    return Bundle(
            view = _mk_views,
            pad = ctzipmaprange(_pad, num=num),
            load = ctzipmap(_load, infos),
            store = ctzipdo(_store, infos, nzips=2),
            add = ctzipmaprange(_add, num=num, nzips=2),
            init = ctmap(_init, infos)
    )

class Bundle:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return tuple(getattr(self, k) for k in key)
        else:
            return getattr(self, key)

def make_grid_helper(spec):
    grid = spec.grid
    gix = grid.group_dim.grid_index
    groups = grid.group_dim.num_programs
    base = grid.group_dim.num_tiles // groups
    rem = grid.group_dim.num_tiles % groups
    size = len(grid.task_grid)
    layout = spec.heuristic_layout

    layout_order = tuple(d.grid_index for d in layout)
    inv_layout_order = inverse_p(layout_order)

    assert base >= 1

    def get(dim, attribute, *more):
        if more:
            return tuple(
                    getattr(grid.dims.get(dim), attr) 
                    for attr 
                    in (attribute, *more)
                    )
        else: 
            return getattr(grid.dims.get(dim), attribute)

    @dataclass(frozen=True)
    class TidInfo:
        tid: tuple[int,...]

        def shape(self, dim, *more):
            s = ct.static_eval(get(dim, 'tile_exp'))
            if more:
                ret = (s,)
                for s in ct.static_iter(
                        get(dim, 'tile_exp')
                        for dim
                        in more
                        ):
                    ret += (s,)
                return ret
            else:
                return s

        def offset(self, dim):
            i, s = ct.static_eval(get(dim, 'grid_index', 'tile_exp'))
            return self.tid[i] * s

        def indices(self, dim):
            i, s = ct.static_eval(get(dim, 'grid_index', 'tile_exp'))
            return self.tid[i] * s + ct.arange(s, dtype=ct.int32)

        def mask(self, dim, *more):
            i, s, b = ct.static_eval(get(dim, 'grid_index', 'tile_exp', 'total_var'))
            ret = ((self.tid[i] * s - b) + ct.arange(s, dtype=ct.int32)) < 0
            if more:
                for i, s, b in ct.static_iter(
                        get(dim, 'grid_index', 'tile_exp', 'total_var')
                        for dim
                        in more):
                    ret = ct.expand_dims(ret, -1)
                    ret &= ((self.tid[i] * s - b) + ct.arange(s, dtype=ct.int32)) < 0
            return ret

    @ct.function
    def _init():
        pid = ct.bid(0)
        tid = ()
        for s in ct.static_iter(d.num_programs for d in layout):
            lid = pid % s
            tid += (lid,)
            pid = pid // s
        return retile(tid, inv_layout_order)

    @ct.function
    def _group_offset_and_extra(tid):
        gid = tid[gix]
        offset = base * gid + ct.minimum(gid, rem)
        extra = (gid < rem)
        return offset, extra

    @ct.function
    def _set_gix(original, value):
        ret = ()
        for i in ct.static_iter(range(gix)):
            ret += (original[i],)
        ret += (value,)
        for i in ct.static_iter(range(gix+1, size)):
            ret += (original[i],)
        return ret

    return Bundle(
            group_size = base, 
            gix = gix,
            init = _init, 
            offset_and_extra = _group_offset_and_extra, 
            set_gix = _set_gix, 
            tid_info = TidInfo
            )

def mk_fwd_no_group_kernel(spec, map_reduce, combine, to_semantic):
    assert spec.phase == Phase.fwd
    assert spec.groups == 1
    gsize, init, set_gix, tid_info = make_grid_helper(spec)['group_size', 'init', 'set_gix', 'tid_info']

    load_order = tuple(b.program_index for b in spec.batch_read_buffers + spec.fold_read_buffers)
    inv_load_order = inverse_p(load_order)

    load_batch = make_buffer_helper(spec.batch_input_buffers)['load']
    view_fold = make_buffer_helper(spec.fold_input_buffers)['view']
    store_output = make_buffer_helper(spec.output_buffers)['store']
    init_execution = make_buffer_helper(spec.execution_buffers)['init']


    @ct.function
    def load_map_reduce(tid, batch_tiles, fold_view):
        fold_tiles = fold_view.load(tid)
        return map_reduce(tid_info(tid), *retile(batch_tiles + fold_tiles, inv_load_order))

    @ct.kernel
    def fwd(batch_buffers, fold_buffers, output_buffers):
        tid = init()
        
        batch_tiles = load_batch(tid, batch_buffers)
        fold_view = view_fold(fold_buffers)
        acc = init_execution()
        for i in range(gsize):
            acc = combine(
                    *acc, 
                    *load_map_reduce(set_gix(tid, i), batch_tiles, fold_view)
                    )

        sem = to_semantic(*acc)

        store_output(tid, output_buffers, sem)

    return fwd

def mk_bwd_kernel(spec, map_finalize, embed):
    assert spec.phase == Phase.bwd
    
    gsize, init, set_gix, offset_and_extra, tid_info = make_grid_helper(spec)['group_size', 'init', 'set_gix', 'offset_and_extra', 'tid_info']

    load_order = tuple(spec.input_buffers.index(b) for b in  spec.batch_input_buffers + spec.fold_input_buffers)
    inv_load_order = inverse_p(load_order)

    grad_batch_buffer_index = tuple(spec.grad_buffers.index(b) for b in spec.batch_grad_buffers)
    grad_fold_buffer_index = tuple(spec.grad_buffers.index(b) for b in spec.fold_grad_buffers)
    grad_order = grad_batch_buffer_index + grad_fold_buffer_index 
    inv_grad_order = inverse_p(grad_order)

    output_batch_buffer_index = tuple(spec.output_buffers.index(b) for b in spec.batch_output_buffers)
    output_fold_buffer_index = tuple(spec.output_buffers.index(b) for b in spec.fold_output_buffers)
    out_order = output_batch_buffer_index + output_fold_buffer_index 
    inv_out_order = inverse_p(out_order)

    load_batch = make_buffer_helper(spec.batch_input_buffers)['load']
    view_batch_grad, init_batch_grad, store_batch_grad = make_buffer_helper(spec.batch_grad_buffers)['view', 'init', 'store']
    load_batch_output = make_buffer_helper(spec.batch_output_buffers)['load']
    view_fold_output = make_buffer_helper(spec.fold_output_buffers)['view']
    view_fold = make_buffer_helper(spec.fold_input_buffers)['view']
    view_fold_grad, init_fold_grad = make_buffer_helper(spec.fold_grad_buffers)['view', 'init']

    loop_embed = len(spec.fold_output_buffers) > 0

    @ct.function
    def load_map_finalize(tid, g_embedded, batch_tiles, fold_view, batch_grads, fold_grads):
        fold_tiles = fold_view.load(tid)
        grads = map_finalize(
                tid_info(tid), 
                *retile(batch_tiles + fold_tiles, inv_load_order), 
                *retile(batch_grads + fold_grads, inv_grad_order),
                *g_embedded)
        return retile(grads, grad_batch_buffer_index), retile(grads, grad_fold_buffer_index)

    @ct.function
    def load_embed(tid, batch_out_tiles, fold_out_view, batch_out_grad_tiles, fold_out_grad_view):
        fold_out_tiles = fold_out_view.load(tid)
        out_tile = retile(batch_out_tiles + fold_out_tiles, inv_out_order)
        fold_out_grad_tiles = fold_out_grad_view.load(tid)
        grad_tile = retile(batch_out_grad_tiles + fold_out_grad_tiles, inv_out_order)
        return embed(*out_tile, *grad_tile)

    @ct.kernel
    def bwd(batch_buffers, fold_buffers, 
            batch_grad_buffers, fold_grad_buffers, 
            batch_output_buffers, fold_output_buffers, 
            batch_output_grad_buffers, fold_output_grad_buffers):
        tid = init()
        
        batch_tiles = load_batch(tid, batch_buffers)
        batch_out_tiles = load_batch_output(tid, batch_output_buffers)
        batch_out_grad_tiles = load_batch_output(tid, batch_output_grad_buffers)

        fold_view = view_fold(fold_buffers)
        fold_grad_view = view_fold_grad(fold_grad_buffers)
        fold_out_view = view_fold_output(fold_output_buffers)
        fold_out_grad_view = view_fold_output(fold_output_grad_buffers)

        batch_grads = init_batch_grad()

        offset, extra = offset_and_extra(tid)

        if not loop_embed:
            g_embedded = load_embed(tid, batch_out_tiles, fold_out_view, batch_out_grad_tiles, fold_out_grad_view)

        for i in range(gsize):
            i_tid = set_gix(tid, i+offset)
            if loop_embed:
                g_embedded = load_embed(i_tid, batch_out_tiles, fold_out_view, batch_out_grad_tiles, fold_out_grad_view)
            fold_grads = init_fold_grad()
            batch_grads, fold_grads = load_map_finalize(i_tid, g_embedded, batch_tiles, fold_view, batch_grads, fold_grads)
            fold_grad_view.store_add(i_tid, fold_grads)
        
        if extra:
            i_tid = set_gix(tid, gsize+offset)
            if loop_embed:
                g_embedded = load_embed(i_tid, batch_out_tiles, fold_out_view, batch_out_grad_tiles, fold_out_grad_view)
            fold_grads = init_fold_grad()
            batch_grads, fold_grads = load_map_finalize(i_tid, g_embedded, batch_tiles, fold_view, batch_grads, fold_grads)
            fold_grad_view.store_add(i_tid, fold_grads)

        view_batch_grad(batch_grad_buffers).store_add(tid, batch_grads)

    return bwd

def mk_autograd(
        fwd_spec,
        bwd_spec,
        map_reduce,
        combine,
        to_semantic,
        to_output,
        map_finalize,
        embed,
        mock_input=None,
        mock_output=None,):

    def batch_fold_split_input(spec, inputs):
        batch = []
        fold = []
        for b, array in zip(spec.input_buffers, inputs):
            if b.is_grouped:
                fold.append(array)
            else:
                batch.append(array)
        return tuple(batch), tuple(fold)

    def batch_fold_split_output(spec, output):
        batch = []
        fold = []
        for b, array in zip(spec.output_buffers, output):
            if b.is_grouped:
                fold.append(array)
            else:
                batch.append(array)
        return tuple(batch), tuple(fold)

    def mk_ret_batch_fold_grads(spec):
        batch = []
        fold =  []
        ret = []
        for b in spec.input_buffers:
            if b.req_grad:
                arr = b.grad_buffer.zeros(device='cuda')
                if b.is_grouped:
                    fold.append(arr)
                else:
                    batch.append(arr)
                ret.append(arr)
            else:
                ret.append(None)
        return tuple(ret), tuple(batch), tuple(fold)

    def grid_fn(spec):
        return (spec.grid.tasks, 1, 1)
    
    def kernel_fn(spec):
        match spec.phase:
            case Phase.fwd:
                return mk_fwd_no_group_kernel(spec, map_reduce, combine, to_semantic)
            case Phase.bwd:
                return mk_bwd_kernel(spec, map_finalize, embed)

    def args_fn(spec):
        match spec.phase:
            case Phase.fwd:
                batch, fold = batch_fold_split_input(spec, mock_input)
                output = tuple(b.empty('cuda') for b in spec.output_buffers)
                return (batch, fold, output)
            case Phase.bwd:
                batch_in, fold_in = batch_fold_split_input(spec, mock_input)
                _, batch_grad, fold_grad = mk_ret_batch_fold_grads(spec)
                batch_out, fold_out = batch_fold_split_output(spec, mock_output)
                batch_out_grad, fold_out_grad = batch_fold_split_output(spec, mock_grad_output)
                return (batch_in, fold_in, batch_grad, fold_grad, batch_out, fold_out, batch_out_grad, fold_out_grad)

    def tune(spec):
        if isinstance(spec, list):
            measurements = exhaustive_search(
                    spec,
                    torch.cuda.current_stream(),
                    grid_fn = grid_fn,
                    kernel_fn = kernel_fn,
                    args_fn = args_fn,
                    )
            spec = measurements.best.config
            print(f'failures: {len(measurements.failures)}')
            print(f'successes: {len(measurements.successes)}')
            print(replace(measurements.best, config = 'see below'))
            print(f'layout: {" ".join(str(d) for d in spec.heuristic_layout)}')
        print(spec.grid.pretty)
        return spec
    
    fwd_spec = tune(fwd_spec)
    fwd_kernel = kernel_fn(fwd_spec) #mk_fwd_no_group_kernel(fwd_spec, map_reduce, combine, to_semantic)

    def fwd_f(*inputs):
        batch, fold = batch_fold_split_input(fwd_spec, inputs)
        output = tuple(b.empty('cuda') for b in fwd_spec.output_buffers)
        stream = torch.cuda.current_stream()
        grid = grid_fn(fwd_spec)
        args = (batch, fold, output)
        ct.launch(stream, grid, fwd_kernel, args)
        return output

    mock_output = fwd_f(*mock_input)
    mock_grad_output = tuple(b.new_empty(b.shape).normal_() for b in mock_output)

    bwd_spec = tune(bwd_spec)
    bwd_kernel = kernel_fn(bwd_spec)

    num_inputs = len(fwd_spec.input_buffers)

    class CutileReduceFn(torch.autograd.Function):
        @staticmethod
        def forward(*inputs):
            return fwd_f(*inputs)

        @staticmethod
        def setup_context(ctx, inputs, outputs):
            ctx.save_for_backward(*inputs, *outputs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            saved = ctx.saved_tensors
            inputs = saved[:num_inputs]
            outputs = saved[num_inputs:]

            batch_in, fold_in = batch_fold_split_input(bwd_spec, inputs)
            grad_storage, batch_grads, fold_grads = mk_ret_batch_fold_grads(bwd_spec)
            batch_out, fold_out = batch_fold_split_output(bwd_spec, outputs)
            batch_out_grad, fold_out_grad = batch_fold_split_output(bwd_spec, grad_outputs)

            stream = torch.cuda.current_stream()
            launch_grid = grid_fn(bwd_spec)
            args = (batch_in, fold_in, batch_grads, fold_grads, batch_out, fold_out, batch_out_grad, fold_out_grad)

            ct.launch(stream, launch_grid, bwd_kernel, args)
            return grad_storage

    def fwd(*inputs):
        outputs = CutileReduceFn.apply(*inputs)
        return to_output(*outputs)

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
