from dataclasses import dataclass

import cuda.tile as ct
import polars as pl
import torch
import torch.utils.benchmark as benchmark
import sympy

from cutilereduce.core import *
from cutilereduce.util.spec import l40s

import pathlib

attention = Spec.make(
        input = dict(
            q = Buffer.make('l g h dqk', ct.bfloat16, req_grad=True, default=0),
            k = Buffer.make('r h dqk', ct.bfloat16, req_grad=True, default=0),
            v = Buffer.make('r h dv', ct.bfloat16, req_grad=True, default=0),
        ),
        output = dict(
            m = Buffer.make('l g h', ct.float32, default=float('-inf')),
            e = Buffer.make('l g h', ct.float32, default=0),
            u = Buffer.make('l g h dv', ct.float32, default=0),
        ),
        intermediate = [
            Buffer.make('l r g h', ct.float32),
            ],
        work = Work.make(
            forward=[
                'h : l g : r : dqk',
                ],
            recompute=[
                'h : l g : dv : r',
                ],
            ),
        batch = Dims.parse('l g h'),
        fold = Dims.parse('r'),
        )

sizes = dict(
        l = 1024,
        r = 1024,
        h = 8,
        g = 4,
        dqk = 128,
        dv = 128,
        )

e = Estimator.make(
        attention,
        sizes=sizes,
        symbols=l40s,
        )

sweep = Sweep.default.run_all(e)
print(sweep.select(pl.selectors.starts_with('cfg:').name.replace('^cfg:', ''), 'estimated_time', 'excess_storage_ratio'))

for (phase,), df in sweep.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break

spec = attention.concretize(config)

@ct.function
def embed(z, mu, g_z, g_mu):
    return z, gz - mu * g_mu, g_mu

@ct.function
def map_reduce(query, key, value):
    rmask = key.mask_along(0)
    q = query.tile # l g h dqk
    k = key.tile   # r h dqk
    v = value.tile # r h dv

    (L, G, H, _) = q.shape
    (R, _, DV) = v.shape

    k = k.permute((1, 2, 0))
    q = q.reshape((L*G, H, -1)).transpose(0, 1)
    logits = ct.zeros((H, L*G, R), ct.float32)
    logits = ct.mma(q, k, logits)
    logits = ct.where(rmask[None,None,:], logits, float('-inf'))

    v = v.transpose(0, 1)
    v = ct.where(rmask[None, :, None], v, 0.0)

    m = ct.max(logits, 2) # H LG 
    logits = ct.exp(logits - m[:, :, None])
    e = ct.sum(logits, 2)
    u = ct.zeros((H, L*G, DV), ct.float32)
    u = ct.mma(logits.astype(ct.bfloat16), v, u)
    return (
            m.transpose(0, 1).reshape((L, G, H)), 
            e.transpose(0, 1).reshape((L, G, H)),
            u.transpose(0, 1).reshape((L, G, H, DV)),
            )

@ct.function
def combine(am, ae, av, bm, be, bv):
    key = am > bm
    
    hi_m = ct.where(key, am, bm)
    hi_e = ct.where(key, ae, be)
    hi_v = ct.where(key[:, :, None], av, bv)
    lo_m = ct.where(key, bm, am)
    lo_e = ct.where(key, be, ae)
    lo_v = ct.where(key[:, :, None], bv, av)

    skip = hi_m == float('-inf')

    scaling = ct.exp(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = ct.where(skip[:, :, None], hi_v, hi_v + lo_v * scaling[:, :, None])

    return m, e, v

def to_semantic(m, e, v):
    return m + e.log(), v / e.log()[:, :, :, None]

def to_output(z, v):
    return v

def test(spec):
    query, key, value = [b.empty('cuda') for b in spec.input.values()]
    f = mk_fwd_no_group(spec, map_reduce, combine, to_semantic, to_output)
    with torch.no_grad():
        query.normal_()
        key.normal_()
        value.normal_()
    
    import time

    print(f(query, key, value))

    def naive(query, key, value):
        att = torch.einsum('lghd,rhd->lghr', query, key)
        att = att.softmax(dim=3)
        return torch.einsum('lghr,rhd->lghd', att, value)

    print(naive(query, key, value))

    pytorch = benchmark.Timer(
            stmt='naive(query, key, value)',
            setup='naive(query, key, value)',
            globals = {'naive': naive, 'query': query, 'key': key, 'value': value},
            label = 'attention', sub_label='pytorch', description='',
            num_threads=1
            )

    cutile = benchmark.Timer(
            stmt='f(query, key, value)',
            setup='f(query, key, value)',
            globals = {'f': f, 'query': query, 'key': key, 'value': value},
            label = 'attention', sub_label='cutile', description='',
            num_threads=1
    )

    results = []
    results.append(cutile.blocked_autorange(min_run_time=5))
    results.append(pytorch.blocked_autorange(min_run_time=5))
    comparison = benchmark.Compare(results)
    comparison.print()

test(spec)
