from dataclasses import dataclass

import cuda.tile as ct
import polars as pl
import torch

from cutilereduce.core import *


l40s = {
        PEAK_FLOPS: 362e12,
        BANDWIDTH: 864e9,
        MIN_MMA_EFFICIENCY: 0.5,
        SM_COUNT: 142,
        MAX_PROGRAMS_PER_SM: 32,
        SMEM_PER_SM: 100*1024,
        }

spec=dict(
        input = dict(
            ctx = Buffer.make('b d', ct.bfloat16, req_grad=True),
            trg = Buffer.make('v d', ct.bfloat16, req_grad=True),
            targets = Buffer.make('b', ct.int32),
        ),
        output = dict(
            m = Buffer.make('b', ct.float32, default=float('-inf')),
            e = Buffer.make('b', ct.float32, default=0),
            u = Buffer.make('b', ct.float32, default=0),
        ),
        intermediate = [
            Buffer.make('b v', ct.float32),
            ],
        work = Work.make(
            forward=[
                ': b : v : d',
                ],
            recompute=[
                ': b : v : d',
                ],
            ),
        batch = Dims.parse('b'),
        fold = Dims.parse('v'),
    )

xentropy = Spec.make(
        **spec,
        )

sizes = dict(
        b = 1024,
        v = 1024*32,
        d = 128,
        )

e = Estimator.make(
        xentropy,
        sizes=sizes,
        symbols=l40s,
        )

import pathlib

cached = pathlib.Path('spec.parquet')

if False:#cached.exists():
    res = pl.read_parquet(cached)
else:
    res = Sweep.default.run_all(e)
    res.write_parquet('spec.parquet')

print(res)

for (phase,), df in res.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break


spec = xentropy.concretize(config)

print(spec.eval('estimated_time'))

for k, v in spec.input.items():
    print(k, v.base)

meta = ProgramView.from_spec(spec)

@ct.function
def map_reduce(tiles):
    ctx, trg, targets = tiles
    B = ctx.shape[0]
    V = trg.shape[0]
    logits = ct.zeros((B, V), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits)
    m = ct.max(logits, 1)
    e = ct.sum(ct.exp(logits - m[:, None]), 1)
    v = ct.zeros((B,), ct.float32)
    return m, e, v

@ct.function
def combine(a, b):
    am, ae, av = a
    bm, be, bv = b
    
    key = am > bm
    
    hi_m = ct.where(key, am, bm)
    hi_e = ct.where(key, ae, be)
    hi_v = ct.where(key, av, bv)

    skip = hi_m == float('-inf')

    lo_m = ct.where(key, bm, am)
    lo_e = ct.where(key, be, ae)
    lo_v = ct.where(key, bv, av)

    scaling = ct.exp(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = ct.where(skip, hi_v, hi_v + lo_v * scaling)

    return m, e, v

def test(spec):

    meta = ProgramView.from_spec(spec)

    grid = meta.grid

    ctx, trg, targets = [b.init_buffer('cuda') for b in spec.input.values()]
    m, e, v = [b.init_buffer('cuda') for b in spec.output.values()]

    with torch.no_grad():
        ctx.normal_()
        trg.normal_()

    @ct.function
    def init_tid():
        pid = ct.bid(0)
        quot, rem = ct.static_eval(grid.grouping_info)
        tid = ()

        for s in ct.static_iter(grid.shape[:grid.group_dim]):
            lid = pid % s
            tid += (lid,)
            pid = pid // s

        s = ct.static_eval(grid.shape[grid.group_dim])
        lid = pid % s
        pid = pid // s
        tid += (lid * quot + ct.minimum(lid, rem),)
        size = quot + (lid < rem)

        for s in ct.static_iter(grid.shape[grid.group_dim+1:]):
            lid = pid % s
            tid += (lid,)
            pid = pid // s
        return tid, size

    @ct.function
    def increment_group(original):
        ret = ()
        for i, d in ct.static_iter(
                (i, 1 if i == grid.group_dim else 0)
                for i
                in range(len(grid.shape))
                ):
            ret += (original[i]+d,)
        return ret

    @ct.function
    def retile(original, index):
        ret = ()
        for i in ct.static_iter(index):
            ret += (original[i],)
        return ret

    @ct.function
    def mk_views(buffers):
        views = ()
        for i, shape in ct.static_iter(
            (i, bv.tile_shape(grid)) 
            for i, bv 
            in enumerate(meta.buffers)
            ):
            view = buffers[i].tiled_view(shape)
            views += (view,)
        return views

    @ct.function
    def batch_loads(views, tid):
        tiles = ()
        for i, index in ct.static_iter(
                (i, v.buffer_dims)
                for i, v
                in enumerate(meta.reads)
                if not v.is_grouped(grid)
                ):
            tiles += (views[i].load(retile(tid, index)),)
        return tiles

    @ct.function
    def group_loads(views, tid):
        tiles = ()
        for i, index in ct.static_iter(
                (i, v.buffer_dims)
                for i, v
                in enumerate(meta.reads)
                if v.is_grouped(grid)
                ):
            tiles += (views[i].load(retile(tid, index)),)
        return tiles

    @ct.function
    def store(views, tid, tiles):
        for i, index in ct.static_iter(
                (i, v.buffer_dims)
                for i, v
                in enumerate(meta.writes)
                if v.is_write(spec.phase)
                ):
            views[i].store(retile(tid, index), tiles[i])

    @ct.function
    def init(buffers):
        tid, size = init_tid()
        views = mk_views(buffers)

        inputs = retile(views, ct.static_eval(meta.reads_index))
        outputs = retile(views, ct.static_eval(meta.writes_index))
        load_order = ct.static_eval(meta.load_order)

        return tid, size, views, load_order, inputs, outputs

    @ct.function
    def fwd(buffers):
        tid, size, views, load_order, inputs, outputs = init(buffers)

        btiles = batch_loads(inputs, tid)
        gtiles = group_loads(inputs, tid)

        tiles = retile(btiles + gtiles, load_order)
        acc = map_reduce(tiles)

        for _ in range(size):
            tid = increment_group(tid)
            gtiles = group_loads(views, tid)
            tiles = retile(btiles + gtiles, load_order)
            acc = combine(acc, map_reduce(tiles))

        store(outputs, tid, acc)


    @ct.kernel
    def kernel(ctx, trg, targets, m, e, v):
        buffers = (ctx, trg, targets, m, e, v)
        fwd(buffers)

    print(spec.grid.dims)
    launch_grid = (meta.tasks, 1, 1)
    args = (ctx, trg, targets, m, e, v)
    ct.launch(torch.cuda.current_stream(), launch_grid, kernel, args)
    print(m, e, v)

    print((ctx @ trg.t()).logsumexp(1))
    print((ctx.to(torch.float32) @ trg.t().to(torch.float32)).logsumexp(1))
    print(m + e.log())

test(spec)
