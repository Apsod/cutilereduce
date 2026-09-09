from .cli import main as example_main
import math

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import (
    fold_functions,
    make_fold_spec,
    matmul,
    workmodel,
)


LOG2E = math.log2(math.e)
LN2 = math.log(2)


def xentropy_spec():
    return make_fold_spec(
        input={
            "ctx": buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
            "trg": buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
            "targets": buffer_spec("b", ct.int32, default=-100),
        },
        execution={
            "m": buffer_spec("b", ct.float32, default=float("-inf")),
            "e": buffer_spec("b", ct.float32, default=0),
            "u": buffer_spec("b", ct.float32, default=0),
        },
        semantic={
            "z": buffer_spec("b", ct.float32, default=float("-inf")),
            "l": buffer_spec("b", ct.float32, default=0),
        },
        map_fold_intermediate={
            "logits": buffer_spec("b v", ct.float32),
        },
        map_finalize_intermediate={
            "logits": buffer_spec("b v", ct.float32),
        },
        batch="b",
        fold="v",
        map_fold_work=workmodel(matmul(M="b", N="v", K="d")),
        map_finalize_work=workmodel(
            matmul(M="b", N="v", K="d"),
            matmul(M="b", N="d", K="v"),
            matmul(M="v", N="d", K="b"),
        ),
    )


@ct.function
def xentropy_map(tid, ctx, trg, targets):
    indices = tid.indices("v")
    mask = tid.mask("v")
    batch, fold = tid.shape("b", "v")

    logits = ct.zeros((batch, fold), ct.float32)
    logits = ct.mma(ctx, trg.transpose(), logits) * LOG2E
    logits = ct.where(mask[None, :], logits, float("-inf"))
    hits = (targets[:, None] == indices[None, :]) & mask[None, :]
    return logits, hits


@ct.function
def map_fold(tid, ctx, trg, targets):
    logits, hits = xentropy_map(tid, ctx, trg, targets)
    maximum = ct.max(logits, axis=1)
    exponential_sum = ct.sum(ct.exp2(logits - maximum[:, None]), axis=1)
    target_logit = ct.sum(ct.where(hits, logits, 0.0), axis=1)
    return maximum, exponential_sum, target_logit


@ct.function
def combine(am, ae, av, bm, be, bv):
    left_is_high = am > bm
    high_m = ct.where(left_is_high, am, bm)
    high_e = ct.where(left_is_high, ae, be)
    low_m = ct.where(left_is_high, bm, am)
    low_e = ct.where(left_is_high, be, ae)
    skip = high_m == float("-inf")
    scaling = ct.exp2(low_m - high_m)
    return high_m, ct.where(skip, high_e, high_e + low_e * scaling), av + bv


@ct.function
def to_semantic(maximum, exponential_sum, target_logit):
    return (
        (maximum + ct.log2(exponential_sum)) * LN2,
        target_logit * LN2,
    )


def to_output(logsumexp, target_logit):
    return logsumexp - target_logit


@ct.function
def embed(logsumexp, target_logit, g_logsumexp, g_target_logit):
    return logsumexp * LOG2E, g_logsumexp, g_target_logit


@ct.function
def map_finalize(tid, ctx, trg, targets, g_ctx, g_trg, z, g_z, g_l):
    logits, hits = xentropy_map(tid, ctx, trg, targets)
    scale = ct.exp2(logits - z[:, None])
    g_logits = (scale * g_z[:, None] + hits * g_l[:, None]).astype(ct.bfloat16)
    g_ctx = ct.mma(g_logits, trg, g_ctx)
    g_trg = ct.mma(g_logits.transpose(), ctx, g_trg)
    return g_ctx, g_trg


FUNCTIONS = fold_functions(
    map_fold,
    combine,
    to_semantic,
    to_output,
    embed=embed,
    map_finalize=map_finalize,
)


INITIALIZERS = {
    'ctx': lambda t, s: t.normal_().mul_(s["d"] ** -0.5),
    'trg': lambda t, s: t.normal_().mul_(s["d"] ** -0.5),
    'targets': lambda t, s: t.random_(s["v"]),
}

def reference(ctx, trg, targets):
    return torch.nn.functional.cross_entropy(
        ctx @ trg.t(), targets.to(torch.long), reduction="none",
    )


def main():
    example_main(
        'xentropy', xentropy_spec(), FUNCTIONS, {'b': 16384, 'v': 16384, 'd': 128},
        aliases={'b': 'batch', 'v': 'fold', 'd': 'inner'}, benchmark_seconds=1.0,
        reference=reference, references={'PyTorch': reference},
        initializers=INITIALIZERS,
    )


if __name__ == "__main__":
    main()
