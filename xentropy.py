
import cuda.tile as ct
import polars as pl
import torch
import torch.utils.benchmark as benchmark
import math

from cutilereduce.core import *
from cutilereduce.util.spec import *


xentropy = Spec.make(
        input = dict(
            ctx = Buffer.make('b d', ct.bfloat16, req_grad=True, default=0),
            trg = Buffer.make('v d', ct.bfloat16, req_grad=True, default=0),
            targets = Buffer.make('b', ct.int32, default=-100),
        ),
        execution = dict(
            m = Buffer.make('b', ct.float32, default=float('-inf')),
            e = Buffer.make('b', ct.float32, default=0),
            u = Buffer.make('b', ct.float32, default=0),
        ),
        output = dict(
            z = Buffer.make('b', ct.float32, default=float('-inf')),
            l = Buffer.make('b', ct.float32, default=0),
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
        b = 1024,
        v = 1024,
        d = 128,
        )

e = Estimator.make(
        xentropy,
        sizes=sizes,
        symbols=l4,
        )

sweep = Sweep.default.run_all(e)

for (phase,), df in sweep.group_by('cfg:phase'):
    print(phase)
    print(df)
    if phase == 'forward':
        fwd_conf = next(e.result2cfg(df))
    elif phase == 'backward':
        bwd_conf = next(e.result2cfg(df))

fwd_spec = xentropy.concretize(fwd_conf)
bwd_spec = xentropy.concretize(bwd_conf)

LOG2E = math.log2(math.e)
LN2 = math.log(2)

@ct.function
def map(tile, ctx, trg, targets):
    ixs = tile.indices('v')
    mask = tile.mask('v')

    B = ctx.shape[0]
    V = trg.shape[0]

    logits = ct.zeros((B, V), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits)
    logits = logits * LOG2E
    logits = ct.where(mask[None,:], logits, float('-inf'))
    hits = (targets[:, None] == ixs[None, :]) & mask[None, :]
    return logits, hits

@ct.function
def embed(z, mu, g_z, g_mu):
    return z, g_z - mu * g_mu, g_mu

@ct.function
def map_finalize(tile, ctx, trg, targets, z, w, s):
    logits, hits = map(tile, ctx, trg, targets)

    scale = ct.exp2(logits - z[:, None])

    g_l = (scale * (w[:, None]  + s[:, None] * logits - hits)).astype(ct.bfloat16)

    g_ctx = ct.zeros(ctx.shape, ct.float32)
    g_ctx = ct.mma(g_l, trg, g_ctx)

    g_trg = ct.zeros(trg.shape, ct.float32)
    g_trg = ct.mma(g_l.transpose(), ctx, g_trg)

    return g_ctx, g_trg

@ct.function
def map_reduce(tile, ctx, trg, targets):
    logits, hits = map(tile, ctx, trg, targets)

    m = ct.max(logits, 1)
    e = ct.sum(ct.exp2(logits - m[:, None]), 1)
    v = ct.sum(ct.where(hits, logits, 0.0), 1)

    return m, e, v

@ct.function
def combine(am, ae, av, bm, be, bv):
    key = am > bm
    
    hi_m = ct.where(key, am, bm)
    hi_e = ct.where(key, ae, be)
    lo_m = ct.where(key, bm, am)
    lo_e = ct.where(key, be, ae)

    skip = hi_m == float('-inf')

    scaling = ct.exp2(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = av + bv

    return m, e, v

@ct.function
def to_semantic(m, e, v):
    return (m + ct.log2(e)) * LN2, v * LN2

def to_output(z, v):
    return z - v

ctx, trg, targets = [b.empty('cuda') for b in fwd_spec.input.values()]
f = mk_autograd_no_group(
        fwd_spec,
        bwd_spec,
        map_reduce,
        combine,
        to_semantic,
        to_output,
        map_finalize,
        embed,
        )
with torch.no_grad():
    ctx.normal_()
    trg.normal_()

targets.random_(0, trg.shape[0])



out = f(ctx, trg, targets)

def f2(ctx, trg):
    return f(ctx, trg, targets)

def g(ctx, trg):
    return torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")

out_cutile, grad_cutile = torch.func.vjp(f2, ctx, trg)
out_pytorch, grad_pytorch = torch.func.vjp(g, ctx, trg)

print(out_cutile)
print(out_pytorch)
print((out_cutile - out_pytorch).abs().mean())

mock = out_cutile.new_zeros(out_cutile.shape)
mock.normal_()


print(mock)

for a in grad_pytorch(mock):
    print(a)

#for b in grad_cutile(mock):
#    print(b)

(f(ctx, trg, targets) * mock).sum().backward()

print(ctx.grad)
print(trg.grad)

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

