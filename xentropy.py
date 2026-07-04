from dataclasses import dataclass

import cuda.tile as ct
import polars as pl
import torch
import torch.utils.benchmark as benchmark
import sympy

from cutilereduce.core import *
from cutilereduce.util.spec import l40s

import pathlib

xentropy = Spec.make(
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

sweep = Sweep.default.run_all(e)
print(sweep.select(pl.selectors.starts_with('cfg:').name.replace('^cfg:', ''), 'estimated_time', 'excess_storage_ratio'))

for (phase,), df in sweep.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break

spec = xentropy.concretize(config)

@ct.function
def embed(z, mu, g_z, g_mu):
    return z, gz - mu * g_mu, g_mu

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
    f = mk_fwd_no_group(spec, map_reduce, combine, to_semantic, to_output)
    with torch.no_grad():
        ctx.normal_()
        trg.normal_()
        targets.random_(0, trg.shape[0])
    
    import time

    print(f(ctx, trg, targets))
    print(torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none"))

    pytorch = benchmark.Timer(
            stmt='torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")',
            setup='torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")',
            globals = {'ctx': ctx, 'trg': trg, 'targets': targets},
            label = 'xentropy', sub_label='pytorch', description='',
            num_threads=1
            )

    cutile = benchmark.Timer(
            stmt='f(ctx, trg, targets)',
            setup='f(ctx, trg, targets)',
            globals = {'f': f, 'ctx': ctx, 'trg': trg, 'targets': targets},
            label = 'xentropy', sub_label='cutile', description='',
            num_threads=1
    )

    results = []
    results.append(cutile.blocked_autorange(min_run_time=5))
    results.append(pytorch.blocked_autorange(min_run_time=5))
    comparison = benchmark.Compare(results)
    comparison.print()

test(spec)
