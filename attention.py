import math

import cuda.tile as ct
import polars as pl
import torch
import torch.utils.benchmark as benchmark

from cutilereduce.core import *
from cutilereduce.util.spec import *


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
        l = 1024*4,
        r = 1024*4,
        h = 4,
        g = 8,
        dqk = 128,
        dv = 128,
        )

e = Estimator.make(
        attention,
        sizes=sizes,
        symbols=l4,
        )

sweep = Sweep.default.run_all(e)
print(sweep.select(pl.selectors.starts_with('cfg:').name.replace('^cfg:', '')))

for (phase,), df in sweep.group_by('cfg:phase'):
    if phase == 'forward':
        config = next(e.result2cfg(df))
        break

spec = attention.concretize(config)

@ct.function
def embed(z, mu, g_z, g_mu):
    return z, gz - mu * g_mu, g_mu

LOG2E = math.log2(math.e)
LN2 = math.log(2)

@ct.function
def map_reduce(tile, query, key, value):
    """
    query: l g h dqk
    key: r h dqk
    value: r h dv
    """
    L, G, H, R, DV = tile.shape('l', 'g', 'h', 'r', 'dv')
    rmask = tile.mask('r')

    key = key.permute((1, 2, 0))
    query = query.reshape((L*G, H, -1)).transpose(0, 1)
    logits = ct.zeros((H, L*G, R), ct.float32)
    logits = ct.mma(query, key, logits)
    logits = ct.where(rmask[None,None,:], logits, float('-inf'))
    logits = logits * LOG2E

    value = value.transpose(0, 1)

    m = ct.max(logits, 2) # H LG 
    logits = ct.exp2(logits - m[:, :, None]) # H LG R
    e = ct.sum(logits, 2) # H LG
    u = ct.zeros((H, L*G, DV), ct.float32) # H LG DV
    u = ct.mma(logits.astype(ct.bfloat16), value, u)
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

    scaling = ct.exp2(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = ct.where(skip[:, :, :, None], hi_v, hi_v + lo_v * scaling[:, :, :, None])

    return m, e, v

def to_semantic(m, e, v):
    return (m + e.log2()) * LN2, v / e[:, :, :, None]

def to_output(z, v):
    return v

def test(spec):
    query, key, value = [b.empty('cuda') for b in spec.input.values()]
    f = mk_fwd_no_group(spec, map_reduce, combine, to_semantic, to_output)
    with torch.no_grad():
        query.normal_()
        key.normal_()
        value.normal_()
    

    a = f(query, key, value)

    def naive(query, key, value):
        query, key, value = (x.to(torch.float32) for x in (query, key, value))
        att = torch.einsum('lghd,rhd->lghr', query, key)
        att = att.softmax(dim=3)
        return torch.einsum('lghr,rhd->lghd', att, value)

    h = sizes['h']
    g = sizes['g']
    l = sizes['l']
    r = sizes['r']
    dqk = sizes['dqk']
    dv = sizes['dv']
    
    def sdpa(query, key, value):
        q_sdpa = query.permute(2, 1, 0, 3).reshape(1, h * g, l, dqk)
        k_sdpa = key.permute(1, 0, 2).reshape(1, h, r, dqk)
        v_sdpa = value.permute(1, 0, 2).reshape(1, h, r, dv)

        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            enable_gqa=True,
            scale=1.0,
            dropout_p=0.0,
        )

        # SDPA layout -> kernel layout
        return out_sdpa.reshape(1, h, g, l, dv)[0].permute(2, 1, 0, 3)
    b = naive(query, key, value)
    c = sdpa(query, key, value)

    print((a - c).abs().mean())
    print((a - b).abs().mean())
    print((b - c).abs().mean())

    pytorch_naive = benchmark.Timer(
            stmt='naive(query, key, value)',
            setup='naive(query, key, value)',
            globals = {'naive': naive, 'query': query, 'key': key, 'value': value},
            label = 'attention', sub_label='pytorch naive', description='',
            num_threads=1
            )

    pytorch_sdpa = benchmark.Timer(
            stmt='sdpa(query, key, value)',
            setup='sdpa(query, key, value)',
            globals = {'sdpa': sdpa, 'query': query, 'key': key, 'value': value},
            label = 'attention', sub_label='pytorch sdpa', description='',
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
    results.append(pytorch_naive.blocked_autorange(min_run_time=5))
    results.append(pytorch_sdpa.blocked_autorange(min_run_time=5))
    comparison = benchmark.Compare(results)
    comparison.print()

test(spec)
