import argparse
import math

import cuda.tile as ct
import torch
import torch.utils.benchmark as torch_benchmark

from cutilereduce.core import MatMulWork, WorkModel
from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import (
    fold_functions,
    make_fold_spec,
    mk_fold_autograd,
    tune_commutative_fold_plan,
)
from cutilereduce.util.runner import run_grad
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
        intermediate={
            "logits": buffer_spec("b v", ct.float32),
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


def validate(plan, sizes):
    print("PyTorch correctness validation", flush=True)
    batch, fold, inner = (sizes[name] for name in ("b", "v", "d"))
    ctx = torch.randn(batch, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    trg = torch.randn(fold, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    targets = torch.randint(fold, (batch,), device="cuda", dtype=torch.int32)

    ctx.requires_grad_()
    trg.requires_grad_()
    function = mk_fold_autograd(plan, FUNCTIONS)

    @torch.compile()
    def cutile(current_ctx, current_trg, current_targets):
        loss, = function(current_ctx, current_trg, current_targets)
        return loss

    @torch.compile()
    def reference(current_ctx, current_trg, current_targets):
        return torch.nn.functional.cross_entropy(
            current_ctx @ current_trg.t(),
            current_targets.to(torch.long),
            reduction="none",
        )

    checked = run_grad(ctx, trg, targets, cutile=cutile, pytorch=reference)
    cutile_forward, = checked["cutile"]["fwd"]
    pytorch_forward, = checked["pytorch"]["fwd"]
    print(f"forward mean absolute error: {(cutile_forward - pytorch_forward).abs().mean().item():.6g}")
    for name, cutile_grad, pytorch_grad in zip(
            ("ctx", "trg"),
            checked["cutile"]["bwd"],
            checked["pytorch"]["bwd"],
            strict=True,
            ):
        print(f"{name} grad mean absolute error: {(cutile_grad - pytorch_grad).abs().mean().item():.6g}")


def benchmark_full(plan, sizes, min_run_time, *, torch_compile=False):
    print("end-to-end timing comparison", flush=True)
    batch, fold, inner = (sizes[name] for name in ("b", "v", "d"))
    ctx = torch.randn(batch, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    trg = torch.randn(fold, inner, device="cuda", dtype=torch.bfloat16) * inner**-0.5
    targets = torch.randint(fold, (batch,), device="cuda", dtype=torch.int32)
    ctx.requires_grad_()
    trg.requires_grad_()
    eager_function = mk_fold_autograd(plan, FUNCTIONS)
    compiled_function = (
        torch.compile(eager_function, fullgraph=True, mode="reduce-overhead")
        if torch_compile else None
    )
    mock = torch.randn(batch, device="cuda", dtype=torch.float32)

    def cutile_forward(function):
        with torch.no_grad():
            return function(ctx, trg, targets)[0]

    def cutile_train_forward(function):
        return function(ctx, trg, targets)[0]

    def pytorch_forward():
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(
                ctx @ trg.t(),
                targets.to(torch.long),
                reduction="none",
            )

    def pytorch_train_forward():
        return torch.nn.functional.cross_entropy(
            ctx @ trg.t(), targets.to(torch.long), reduction="none",
        )

    def cutile_forward_backward(function):
        ctx.grad = None
        trg.grad = None
        (cutile_train_forward(function) * mock).sum().backward()

    def pytorch_forward_backward():
        ctx.grad = None
        trg.grad = None
        (pytorch_train_forward() * mock).sum().backward()

    cases = [
        ("forward", "CuTile eager", lambda: cutile_forward(eager_function)),
        ("forward", "PyTorch", pytorch_forward),
        ("forward + backward", "CuTile eager", lambda: cutile_forward_backward(eager_function)),
        ("forward + backward", "PyTorch", pytorch_forward_backward),
    ]
    if compiled_function is not None:
        cutile_forward_backward(compiled_function)
        cases.extend((
            ("forward", "CuTile torch.compile", lambda: cutile_forward(compiled_function)),
            ("forward + backward", "CuTile torch.compile", lambda: cutile_forward_backward(compiled_function)),
        ))
    measurements = []
    for label, implementation, function_to_time in cases:
        timer = torch_benchmark.Timer(
            stmt="function_to_time()",
            globals={"function_to_time": function_to_time},
            label=f"xentropy {label}",
            sub_label=implementation,
            description="selected configuration",
            num_threads=1,
        )
        measurements.append(timer.blocked_autorange(min_run_time=min_run_time))
    torch_benchmark.Compare(measurements).print()


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
    spec = xentropy_spec()
    plan = tune_commutative_fold_plan(
        spec,
        sizes,
        args.candidates,
        args.timeout,
        functions=FUNCTIONS,
        hardware=rtx5080,
        quiet_tuning=args.quiet_tuning,
    )
    if args.benchmark_seconds > 0:
        benchmark_full(
            plan, sizes, args.benchmark_seconds,
            torch_compile=args.torch_compile,
        )
    if args.validate:
        validate(plan, sizes)


if __name__ == "__main__":
    main()
