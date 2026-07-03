from dataclasses import dataclass

import cuda.tile as ct
import polars as pl
import torch
import sympy

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

if cached.exists():
    res = pl.read_parquet(cached)
else:
    res = Sweep.default.run_all(e)
    res.write_parquet('spec.parquet')

print(res.select(pl.selectors.starts_with('cfg:').name.replace('^cfg:', ''), 'estimated_time', 'excess_storage_ratio'))

for (phase,), df in res.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break


spec = xentropy.concretize(config)

@ct.function
def embed(z, mu, g_z, g_mu):
    return z, gz - mu * g, g

@ct.function
def map(ctx, trg, targets):
    ixs = trg.indices_along(0)
    vmask = trg.mask_along(0)
    ctx = ctx.tile
    trg = trg.tile
    targets = targets.tile

    B = ctx.shape[0]
    V = trg.shape[0]

    logits = ct.zeros((B, V), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits)
    logits = ct.where(vmask[None,:], logits, float('-inf'))
    hits = (targets[:, None] == ixs[None, :]) & vmask[None, :]
    return logits, hits


@ct.function
def map_finalize(ctx, trg, targets, z, w, s):
    logits, hits = map(ctx, trg, targets)

    scale = ct.exp(logits - z[:, None])

    g_l = scale * (w[:, None]  + s[:, None] * logits - hits)

    g_ctx = ct.zeros((B, D), ct.float32)
    g_ctx = torch.mma(g_l, trg, g_ctx)

    g_trg = ct.zeros((V, D), ct.float32)
    g_trg = torch.mma(g_l.transpose(), ctx, g_trg)

    return g_ctx, g_trg

@ct.function
def map_reduce(ctx, trg, targets):
    logits, hits = map(ctx, trg, targets)

    m = ct.max(logits, 1)
    e = ct.sum(ct.exp(logits - m[:, None]), 1)
    v = ct.sum(hits * logits, 1)

    return m, e, v

@ct.function
def combine(am, ae, av, bm, be, bv):
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

def to_semantic(m, e, v):
    return m + e.log(), v

def to_output(z, v):
    return z - v

def test(spec):
    ctx, trg, targets = [b.empty('cuda') for b in spec.input.values()]
    f = mk_fwd(spec, map_reduce, combine, to_semantic, to_output)
    with torch.no_grad():
        ctx.normal_()
        trg.normal_()
        targets.random_(0, trg.shape[0])
    
    import time

    dur = -time.perf_counter()
    for _ in range(2):
        f(ctx, trg, targets)
    dur += time.perf_counter()
    print('compile:', dur)
    dur = -time.perf_counter()
    for _ in range(20):
        a = f(ctx, trg, targets)
    dur += time.perf_counter()
    print(dur)
    for _ in range(2):
        torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none')
    dur = -time.perf_counter()
    for _ in range(20):
        b = torch.nn.functional.cross_entropy(ctx.to(torch.float32) @ trg.to(torch.float32).t(), targets.to(torch.long), reduction='none')
    dur += time.perf_counter()
    print(dur)

    print(a)
    print(b)
    

test(spec)
