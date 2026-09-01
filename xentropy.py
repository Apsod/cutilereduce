import argparse
import math

import cuda.tile as ct
import torch

from cutilereduce.core import MatMulWork, WorkModel
from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import (
    FoldOperator,
    fold_functions,
    make_fold_spec,
)
from cutilereduce.util.runner import (
    benchmark_implementations,
    validate_precision_matrix,
)
from cutilereduce.util.spec import rtx5080


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
        output={
            "z": buffer_spec("b", ct.float32, default=float("-inf")),
            "l": buffer_spec("b", ct.float32, default=0),
        },
        map_intermediate={
            "logits": buffer_spec("b v", ct.float32),
        },
        finalize_intermediate={
            "logits": buffer_spec("b v", ct.float32),
            "scale": buffer_spec("b v", ct.float32),
            "g_logits": buffer_spec("b v", ct.bfloat16),
        },
        batch="b",
        fold="v",
        map_fold_work=WorkModel.make(MatMulWork.make(M="b", N="v", K="d")),
        backward_work=WorkModel.make(
            MatMulWork.make(M="b", N="v", K="d"),
            MatMulWork.make(M="b", N="d", K="v"),
            MatMulWork.make(M="v", N="d", K="b"),
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
def map_reduce(tid, ctx, trg, targets):
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
def finalize(tid, ctx, trg, targets, g_ctx, g_trg, z, g_z, g_l):
    logits, hits = xentropy_map(tid, ctx, trg, targets)
    scale = ct.exp2(logits - z[:, None])
    g_logits = (scale * g_z[:, None] + hits * g_l[:, None]).astype(ct.bfloat16)
    g_ctx = ct.mma(g_logits, trg, g_ctx)
    g_trg = ct.mma(g_logits.transpose(), ctx, g_trg)
    return g_ctx, g_trg


FUNCTIONS = fold_functions(
    map_reduce,
    combine,
    to_semantic,
    to_output,
    embed=embed,
    finalize=finalize,
)


def make_inputs(sizes):
    batch, fold, inner = (sizes[name] for name in ("b", "v", "d"))
    ctx = torch.randn(batch, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    trg = torch.randn(fold, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    targets = torch.randint(fold, (batch,), device="cuda", dtype=torch.int32)
    ctx.requires_grad_()
    trg.requires_grad_()
    return ctx, trg, targets


def reference(ctx, trg, targets):
    return torch.nn.functional.cross_entropy(
        ctx @ trg.t(), targets.to(torch.long), reduction="none",
    )


def validate(operator, plan, sizes, *, accuracy_matrix=False):
    print("PyTorch correctness validation", flush=True)
    reference_dtypes = {
        "PyTorch BF16": torch.bfloat16,
        "PyTorch FP32": torch.float32,
    }
    if accuracy_matrix:
        reference_dtypes["PyTorch FP64"] = torch.float64
    validate_precision_matrix(
        operator.build(plan),
        reference,
        make_inputs(sizes),
        input_names=("ctx", "trg"),
        reference_dtypes=reference_dtypes,
        pairwise=accuracy_matrix,
    )


def benchmark_full(operator, plan, sizes, min_run_time, *, torch_compile=False):
    print("end-to-end timing comparison", flush=True)
    implementations = {
        "CuTile eager": operator.build(plan),
        "PyTorch": reference,
    }
    if torch_compile:
        implementations["CuTile torch.compile"] = operator.build(
            plan, torch_compile=True,
        )
    benchmark_implementations(
        "xentropy",
        make_inputs(sizes),
        implementations,
        min_run_time=min_run_time,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16 * 1024)
    parser.add_argument("--fold", type=int, default=16 * 1024)
    parser.add_argument("--inner", type=int, default=128)
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="complete configurations retained per admissible loop axis",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="subprocess timeout; disabled by default because CUDA IPC may be unavailable",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--accuracy-matrix", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="also benchmark torch.compile(mode='reduce-overhead')",
    )
    parser.add_argument(
        "--quiet-tuning",
        action="store_true",
        help="suppress interactive empirical-tuning progress",
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=1.0,
        help="minimum run time per end-to-end benchmark; set to 0 to disable",
    )
    args = parser.parse_args()
    if args.candidates <= 0:
        parser.error("--candidates must be positive")
    sizes = {"b": args.batch, "v": args.fold, "d": args.inner}
    torch.manual_seed(args.seed)
    spec = xentropy_spec()
    operator = FoldOperator(spec, FUNCTIONS)
    plan = operator.tune(
        sizes,
        args.candidates,
        args.timeout,
        hardware=rtx5080,
        quiet=args.quiet_tuning,
    )
    if args.benchmark_seconds > 0:
        benchmark_full(
            operator, plan, sizes, args.benchmark_seconds,
            torch_compile=args.torch_compile,
        )
    if args.validate:
        validate(
            operator, plan, sizes,
            accuracy_matrix=args.accuracy_matrix,
        )


if __name__ == "__main__":
    main()
