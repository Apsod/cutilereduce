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
            ctx = Buffer.make('b d', ct.bfloat16, req_grad=True, default=0),
            trg = Buffer.make('v d', ct.bfloat16, req_grad=True, default=0),
            targets = Buffer.make('b', ct.int32, default=-100),
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
        b = 1024*32,
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

if False:# cached.exists():
    res = pl.read_parquet(cached)
else:
    res = Sweep.default.run_all(e)
    res.write_parquet('spec.parquet')

print(res.select(pl.selectors.starts_with('cfg:').name.replace('^cfg:', ''), 'estimated_time'))

for (phase,), df in res.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break


spec = xentropy.concretize(config)

@ct.function
def map_reduce(ctx, trg, targets):
    ixs = trg.indices_along(0)
    ctx = ctx.tile
    trg = trg.tile
    targets = targets.tile

    B = ctx.shape[0]
    V = trg.shape[0]

    logits = ct.zeros((B, V), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits)

    m = ct.max(logits, 1)
    e = ct.sum(ct.exp(logits - m[:, None]), 1)
    
    hits = targets[:, None] == ixs[None, :]
    v = ct.sum(hits * logits, 1)
    return m, e, v

@ct.function
def combine(a, b):
    am, ae, av = a
    bm, be, bv = b
    
    key = am > bm
    
    hi_m = ct.where(key, am, bm)
    hi_e = ct.where(key, ae, be)
    lo_m = ct.where(key, bm, am)
    lo_e = ct.where(key, be, ae)

    skip = hi_m == float('-inf')

    scaling = ct.exp(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = av + bv

    return m, e, v

def test(spec):
    ctx, trg, targets = [b.init_buffer('cuda') for b in spec.input.values()]
    fwd = mk_fwd(spec, map_reduce, combine)
    with torch.no_grad():
        ctx.normal_()
        trg.normal_()
        targets.random_(0, trg.shape[0])

    def f(ctx, trg, targets):
        m, e, v = [b.init_buffer('cuda') for b in spec.output.values()]
        args = (ctx, trg, targets, m, e, v)
        launch_grid = (spec.grid.tasks, 1, 1)
        ct.launch(torch.cuda.current_stream(), launch_grid, fwd, args)
        return m + e.log() - v
    
    print(f(ctx, trg, targets))
    print(f(ctx, trg, targets))
    print(f(ctx, trg, targets))
    print(f(ctx, trg, targets))
    print(f(ctx, trg, targets))
    print(torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none'))
    print(torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none'))
    print(torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none'))
    print(torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none'))
    print(torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none'))

test(spec)
