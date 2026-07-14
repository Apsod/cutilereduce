import math

import cuda.tile as ct
import torch
import torch.utils.benchmark as benchmark

from cutilereduce.core import Spec, Buffer, Dims, Work, Estimator, Sweep, mk_autograd, MatMul
from cutilereduce.util.spec import l4
from cutilereduce.util.runner import run_grad


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
        grad_accumulator = dict(
            z = Buffer.make('b', ct.float32, default=float('-inf')),
            g_z = Buffer.make('b', ct.float32, default=0),
            g_l = Buffer.make('b', ct.float32, default=0),
        ),
        intermediate = [
            Buffer.make('b v', ct.float32),
            ],
        work = Work(
            forward=[
                MatMul.make(M='b', N='v', K='d'),
                ],
            recompute=[
                MatMul.make(M='b', N='v', K='d'),
                ],
            ),
        batch = Dims.parse('b'),
        fold = Dims.parse('v'),
        )

sizes = dict(
        b = 1024*16,
        v = 1024*16,
        d = 128,
        )

e = Estimator.make(
        xentropy,
        sizes=sizes,
        symbols=l4,
        )

sweep = Sweep.default.run_all(e)

for (phase,), df in sweep.group_by('cfg:phase'):
    if phase == 'forward':
        fwd_conf = next(e.result2cfg(df))
    elif phase == 'backward':
        bwd_conf = next(e.result2cfg(df))

print(fwd_conf)
print(bwd_conf)

fwd_spec = xentropy.concretize(fwd_conf)
bwd_spec = xentropy.concretize(bwd_conf)

LOG2E = math.log2(math.e)
LN2 = math.log(2)

@ct.function
def map(tile, ctx, trg, targets):
    ixs = tile.indices('v')
    mask = tile.mask('v')
    B, V = tile.shape('b', 'v')

    logits = ct.zeros((B, V), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits)
    logits = logits * LOG2E
    logits = ct.where(mask[None,:], logits, float('-inf'))
    hits = (targets[:, None] == ixs[None, :]) & mask[None, :]
    return logits, hits

@ct.function
def embed(z, mu, g_z, g_l):
    return z * LOG2E, g_z, g_l

@ct.function
def map_finalize(tile, ctx, trg, targets, g_ctx, g_trg, z, g_z, g_l):
    logits, hits = map(tile, ctx, trg, targets)

    scale = ct.exp2(logits - z[:, None])

    g_logits = (scale * g_z[:, None] + hits * g_l[:, None]).astype(ct.bfloat16)

    g_ctx = ct.mma(g_logits, trg, g_ctx)
    g_trg = ct.mma(g_logits.transpose(), ctx, g_trg)

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
f = mk_autograd(
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
    ctx.normal_(std=ctx.shape[1]**-.5)
    trg.normal_(std=trg.shape[1]**-.5)

targets.random_(0, trg.shape[0])

out_cutile = f(ctx, trg, targets)

def pytorch_xentropy(ctx, trg, targets):
    return torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")

out_pytorch = pytorch_xentropy(ctx, trg, targets)

print(out_cutile)
print(out_pytorch)
print((out_cutile - out_pytorch).abs().mean())

mock = out_cutile.new_zeros(out_cutile.shape)
mock.normal_()

def cutile_pass(ctx, trg, targets):
    (f(ctx, trg, targets) * mock).sum().backward()

def pytorch_pass(ctx, trg, targets):
    (pytorch_xentropy(ctx, trg, targets) * mock).sum().backward()

cutile_pass(ctx, trg, targets)

check = run_grad(ctx, trg, targets, cutile=f, pytorch=pytorch_xentropy)

print('forward diff')
for c, p in zip(check['cutile']['fwd'], check['pytorch']['fwd']):
    print((c - p).abs().mean())

print('backward diff')
for c, p in zip(check['cutile']['bwd'], check['pytorch']['bwd']):
    print((c - p).abs().mean())


pytorch_fwd = benchmark.Timer(
        stmt='torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")',
        setup='torch.nn.functional.cross_entropy(ctx @ trg.t(), targets.to(torch.long), reduction="none")',
        globals = {'ctx': ctx, 'trg': trg, 'targets': targets},
        label = 'xentropy fwd', sub_label='pytorch', description='',
        num_threads=1
        )

cutile_fwd = benchmark.Timer(
        stmt='f(ctx, trg, targets)',
        setup='f(ctx, trg, targets)',
        globals = {'f': f, 'ctx': ctx, 'trg': trg, 'targets': targets},
        label = 'xentropy fwd', sub_label='cutile', description='',
        num_threads=1
)

pytorch_fwd_bwd = benchmark.Timer(
        stmt='pytorch_pass(ctx, trg, targets)',
        setup='pytorch_pass(ctx, trg, targets)',
        globals = {'pytorch_pass': pytorch_pass, 'ctx': ctx, 'trg': trg, 'targets': targets},
        label = 'xentropy fwd-bwd', sub_label='pytorch', description='',
        num_threads=1
        )

cutile_fwd_bwd = benchmark.Timer(
        stmt='cutile_pass(ctx, trg, targets)',
        setup='cutile_pass(ctx, trg, targets)',
        globals = {'cutile_pass': cutile_pass, 'ctx': ctx, 'trg': trg, 'targets': targets},
        label = 'xentropy fwd-bwd', sub_label='cutile', description='',
        num_threads=1
)


results = []
results.append(cutile_fwd.blocked_autorange(min_run_time=5))
results.append(pytorch_fwd.blocked_autorange(min_run_time=5))
results.append(cutile_fwd_bwd.blocked_autorange(min_run_time=5))
results.append(pytorch_fwd_bwd.blocked_autorange(min_run_time=5))
comparison = benchmark.Compare(results)
comparison.print()

