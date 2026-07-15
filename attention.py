import math

import cuda.tile as ct
import polars as pl
import torch
import torch.utils.benchmark as benchmark

from cutilereduce.core import Spec, Buffer, Dims, Work, Estimator, Sweep, MatMul, mk_autograd
from cutilereduce.util.spec import l4
from cutilereduce.util.runner import run_grad


attention = Spec.make(
        input = dict(
            q = Buffer.make('h l g dqk', ct.bfloat16, req_grad=True, default=0),
            k = Buffer.make('h r dqk', ct.bfloat16, req_grad=True, default=0),
            v = Buffer.make('h r dv', ct.bfloat16, req_grad=True, default=0),
        ),
        execution = dict(
            m = Buffer.make('h l g', ct.float32, default=float('-inf')),
            e = Buffer.make('h l g', ct.float32, default=0),
            u = Buffer.make('h l g dv', ct.float32, default=0),
        ),
        output = dict(
            z = Buffer.make('h l g', ct.float32, default=float('-inf')),
            mu = Buffer.make('h l g dv', ct.float32, default=0),
        ),
        grad_accumulator = dict(
            z = Buffer.make('h l g', ct.float32, default=float('-inf')),
            g_z = Buffer.make('h l g', ct.float32, default=0),
            g_mu = Buffer.make('h l g dv', ct.float16, default=0),
        ),
        intermediate = [
            Buffer.make('h l r g', ct.float32),
            ],
        work = Work(
            forward=[
                MatMul.make(B='h', M='l g', N='r', K='dqk'),
                MatMul.make(B='h', M='l g', N='dv', K='r')
                ],
            backward=[
                MatMul.make(B='h', M='l g', N='r', K='dqk'),  # recompute logits
                MatMul.make(B='h', M='l g', N='r', K='dv'),   # dLogits
                MatMul.make(B='h', M='r', N='dv', K='l g'),   # dV
                MatMul.make(B='h', M='l g', N='dqk', K='r'),  # dQ
                MatMul.make(B='h', M='r', N='dqk', K='l g'),  # dK
                ],
            ),
        batch = Dims.parse('h l g'),
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


@ct.function
def embed(z, mu, g_z, g_mu):
    """
    z, g_z: h l g
    mu, g_mu: h l g dv 
    return: h l g, h l g, h l g dv
    """
    return (
            z,
            g_z - ct.sum(mu * g_mu, 3),
            g_mu.astype(ct.bfloat16),
            )

LOG2E = math.log2(math.e)
LN2 = math.log(2)

@ct.function
def map(tile, query, key):
    """
    tile: metadata
    query: h l g dqk
    key: h r dqk
    value: h r dv
    returns: h (l g) r-shaped logits
    """
    L, G, H, R, DQK = tile.shape('l', 'g', 'h', 'r', 'dqk')
    rmask = tile.mask('r')
    lgmask = (tile.mask('l')[:, None] & tile.mask('g')[None, :]).reshape((L*G,))

    key = key.transpose(1, 2)
    query = query.reshape((H, L*G, DQK))
    logits = ct.zeros((H, L*G, R), ct.float32)
    logits = ct.mma(query, key, logits)
    logits = ct.where(rmask[None,None,:] & lgmask[None, :, None], logits, float('-inf'))
    logits = logits
    return logits

@ct.function
def map_finalize(tile, query, key, value, g_query, g_key, g_value, z, g_z, g_mu):
    """
    tile: metadata
    [g_]query: h l g dqk
    [g_]key: h r dqk
    [g_]value: h r dv
    z: h l g
    g_z: h l g
    g_mu: h l g dv
    """

    logits = map(tile, query, key)

    L, G, H, R, DV, DQK = tile.shape('l', 'g', 'h', 'r', 'dv', 'dqk')
    
    # h (l g) r
    scale = ct.exp(logits - z.reshape((H, L*G, 1)))

    g_mu = g_mu.reshape((H, L*G, DV))

    # h r dv
    g_value = ct.mma(
            scale.transpose(1, 2).astype(ct.bfloat16),
            g_mu.astype(ct.bfloat16),
            g_value
            )
    
    # h (l g) r
    #g_logits = ct.zeros((H, L*G, R), ct.float32) + g_z.reshape((H, L*G, 1))
    g_logits = ct.broadcast_to(g_z.reshape((H, L*G, 1)), (H, L*G, R))
    g_logits = ct.mma(
            g_mu.astype(ct.bfloat16),
            value.transpose(1, 2), 
            g_logits)
    #g_logits = g_logits + g_z[:, :, None]
    g_logits = (scale * g_logits).astype(ct.bfloat16)

    g_query = ct.mma(
            g_logits, 
            key, 
            g_query.reshape((H, L*G, DQK))
            ).reshape((H, L, G, DQK))
    g_key = ct.mma(
            g_logits.transpose(1, 2), 
            query.reshape((H, L*G, DQK)), 
            g_key
            )

    return g_query, g_key, g_value

@ct.function
def map_reduce(tile, query, key, value):
    """
    query: h l g dqk
    key: h r dqk
    value: h r dv
    """

    L, G, H, R, DV, DQK = tile.shape('l', 'g', 'h', 'r', 'dv', 'dqk')
    
    # logits: h (l g) r
    logits = map(tile, query, key) * LOG2E

    m = ct.max(logits, 2) # H LG 
    logits = ct.exp2(logits - m[:, :, None]) # H LG R

    e = ct.sum(logits, 2) # H LG

    u = ct.zeros((H, L*G, DV), ct.float32) # H LG DV
    u = ct.mma(logits.astype(ct.bfloat16), value, u)

    return (
            m.reshape((H, L, G)), 
            e.reshape((H, L, G)),
            u.reshape((H, L, G, DV)),
            )

@ct.function
def map_reduce_combine(tile, query, key, value, acc_m, acc_e, acc_u):
    """
    query: h l g dqk
    key: h r dqk
    value: h r dv
    """

    L, G, H, R, DV, DQK = tile.shape('l', 'g', 'h', 'r', 'dv', 'dqk')
    
    # logits: h (l g) r
    logits = map(tile, query, key) * LOG2E

    loc_m = ct.max(logits, 2).reshape((H, L, G)) # H L G 
    key = acc_m > loc_m

    acc_m = tl.where(key, acc_m, loc_m)

    scaling = tl.where(key, 1, ct.exp2(loc_m - acc_m))
    
    logits = ct.exp2(logits - acc_m.reshape((H, L*G, 1)))

    acc_e = acc_e * scaling + ct.sum(logits, 2).reshape((H, L, G)) # H LG

    acc_u = scaling[:,:,:, None] * acc_u
    acc_u = ct.mma(logits.astype(ct.bfloat16), value, acc_u.reshape((H, L*G, DV))).reshape((H, L, G, DV))

    return (
            acc_m,
            acc_e,
            acc_u,
            )

@ct.function
def combine(am, ae, au, bm, be, bu):
    key = am > bm
    
    hi_m = ct.where(key, am, bm)
    hi_e = ct.where(key, ae, be)
    hi_u = ct.where(key[:, :, :, None], au, bu)

    lo_m = ct.where(key, bm, am)
    lo_e = ct.where(key, be, ae)
    lo_u = ct.where(key[:, :, :, None], bu, au)

    skip = hi_m == float('-inf')

    scaling = ct.exp2(lo_m - hi_m)

    m = hi_m
    e = ct.where(skip, hi_e, hi_e + lo_e * scaling)
    v = ct.where(skip[:, :, :, None], hi_u, hi_u + lo_u * scaling[:, :, :, None])

    return m, e, v

@ct.function
def to_semantic(m, e, v):
    return (m + ct.log2(e)) * LN2, v / e[:, :, :, None]

def to_output(z, v):
    return v

sweeper = Sweep.default
#sweeper = sweeper.add_filters(
#        pl.when(pl.col('cfg:phase') == 'backward').then(pl.col('cfg:g') == sizes['g']).otherwise(True),
#        pl.when(pl.col('cfg:phase') == 'backward').then(pl.col('cfg:group') == 'l').otherwise(True),
#        )

sweep = sweeper.run_all(e)

for (phase,), df in sweep.group_by('cfg:phase'):
    if phase == 'forward':
        print(df.select(pl.selectors.starts_with('cfg:').name.map(lambda c: c.removeprefix('cfg:')), 'estimated_time'))
        fwd_confs = list(e.result2cfg(df))
    elif phase == 'backward':
        print(df.select(pl.selectors.starts_with('cfg:').name.map(lambda c: c.removeprefix('cfg:')), 'estimated_time'))
        bwd_confs = list(e.result2cfg(df))

fwd_specs = [attention.concretize(conf) for conf in fwd_confs]
bwd_specs = [attention.concretize(conf) for conf in bwd_confs]

query, key, value = [b.empty('cuda') for b in fwd_specs[0].input.values()]

with torch.no_grad():
    query.normal_()
    key.normal_()
    value.normal_()

f = mk_autograd(
        fwd_specs,
        bwd_specs,
        map_reduce,
        combine,
        to_semantic,
        to_output,
        map_finalize,
        embed,
        mock_input=(query, key, value)
        )
    
a = f(query, key, value)

def naive(query, key, value):
    query, key, value = (x.to(torch.float32) for x in (query, key, value))
    att = torch.einsum('hlgd,hrd->hlgr', query, key)
    att = att.softmax(dim=3)
    return torch.einsum('hlgr,hrd->hlgd', att, value)

h = sizes['h']
g = sizes['g']
l = sizes['l']
r = sizes['r']
dqk = sizes['dqk']
dv = sizes['dv']
    
def sdpa(query, key, value):

    q_sdpa = query.transpose(1, 2).reshape(1, h * g, l, dqk)
    k_sdpa = key.reshape(1, h, r, dqk)
    v_sdpa = value.reshape(1, h, r, dv)

    out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        enable_gqa=True,
        scale=1.0,
        dropout_p=0.0,
    )

    # SDPA layout -> kernel layout
    return out_sdpa.reshape(1, h, g, l, dv)[0].transpose(1, 2)
b = naive(query, key, value)
c = sdpa(query, key, value)

mock = a.new_zeros(a.shape)
mock.normal_()

def cutile_pass(query, key, value):
    (f(query, key, value) * mock).sum().backward()

def pytorch_pass(ctx, trg, targets):
    (sdpa(ctx, trg, targets) * mock).sum().backward()


print('cutile - sdpa ', (a - c).abs().mean().item())
print('cutile - naive', (a - b).abs().mean().item())
print('naive - sdpa  ', (b - c).abs().mean().item())


check = run_grad(query, key, value, cutile=f, pytorch=sdpa)

print('forward diff')
for c, p in zip(check['cutile']['fwd'], check['pytorch']['fwd']):
    print((c - p).abs().mean())

print('backward diff')
for c, p in zip(check['cutile']['bwd'], check['pytorch']['bwd']):
    print((c - p).abs().mean())

args = {'query': query, 'key': key, 'value': value}

pytorch_naive = benchmark.Timer(
        stmt='naive(query, key, value)',
        setup='naive(query, key, value)',
        globals = {'naive': naive, **args},
        label = 'attention', sub_label='pytorch naive', description='',
        num_threads=1
        )

pytorch_sdpa = benchmark.Timer(
        stmt='sdpa(query, key, value)',
        setup='sdpa(query, key, value)',
        globals = {'sdpa': sdpa, **args},
        label = 'attention', sub_label='pytorch sdpa', description='',
        num_threads=1
        )

cutile = benchmark.Timer(
        stmt='f(query, key, value)',
        setup='f(query, key, value)',
        globals = {'f': f, **args},
        label = 'attention', sub_label='cutile', description='',
        num_threads=1
)

pytorch_fwd_bwd = benchmark.Timer(
        stmt='pytorch_pass(query, key, value)',
        setup='pytorch_pass(query, key, value)',
        globals = {'pytorch_pass': pytorch_pass, **args},
        label = 'attention fwd-bwd', sub_label='pytorch', description='',
        num_threads=1
        )

cutile_fwd_bwd = benchmark.Timer(
        stmt='cutile_pass(query, key, value)',
        setup='cutile_pass(query, key, value)',
        globals = {'cutile_pass': cutile_pass, **args},
        label = 'attention fwd-bwd', sub_label='cutile', description='',
        num_threads=1
)

results = []
results.append(cutile.blocked_autorange(min_run_time=5))
results.append(pytorch_naive.blocked_autorange(min_run_time=5))
results.append(pytorch_sdpa.blocked_autorange(min_run_time=5))
results.append(cutile_fwd_bwd.blocked_autorange(min_run_time=5))
results.append(pytorch_fwd_bwd.blocked_autorange(min_run_time=5))
comparison = benchmark.Compare(results)
comparison.print()
