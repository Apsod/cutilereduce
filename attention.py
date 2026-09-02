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
    benchmark_memory,
    validate_precision_matrix,
)
from cutilereduce.util.spec import rtx5080


LOG2E = math.log2(math.e)
LN2 = math.log(2)


def attention_spec():
    return make_fold_spec(
        input={
            "query": buffer_spec("h l g dqk", ct.bfloat16, req_grad=True, default=0),
            "key": buffer_spec("h r dqk", ct.bfloat16, req_grad=True, default=0),
            "value": buffer_spec("h r dv", ct.bfloat16, req_grad=True, default=0),
        },
        execution={
            "m": buffer_spec("h l g", ct.float32, default=float("-inf")),
            "e": buffer_spec("h l g", ct.float32, default=0),
            "u": buffer_spec("h l g dv", ct.float32, default=0),
        },
        output={
            "z": buffer_spec("h l g", ct.float32, default=float("-inf")),
            "mu": buffer_spec("h l g dv", ct.float32, default=0),
        },
        map_intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
        },
        finalize_intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
            "scale": buffer_spec("h l g r", ct.float32),
            "g_logits": buffer_spec("h l g r", ct.bfloat16),
        },
        batch="h l g",
        fold="r",
        map_fold_work=WorkModel.make(
            MatMulWork.make(B="h", M="l g", N="r", K="dqk"),
            MatMulWork.make(B="h", M="l g", N="dv", K="r"),
        ),
        backward_work=WorkModel.make(
            MatMulWork.make(B="h", M="l g", N="r", K="dqk"),
            MatMulWork.make(B="h", M="l g", N="r", K="dv"),
            MatMulWork.make(B="h", M="r", N="dv", K="l g"),
            MatMulWork.make(B="h", M="l g", N="dqk", K="r"),
            MatMulWork.make(B="h", M="r", N="dqk", K="l g"),
        ),
    )


@ct.function
def attention_map(tid, query, key):
    length, group, head, right, dqk = tid.shape("l", "g", "h", "r", "dqk")
    right_mask = tid.mask("r")
    query_mask = (
        tid.mask("l")[:, None] & tid.mask("g")[None, :]
    ).reshape((length * group,))
    query = query.reshape((head, length * group, dqk))
    logits = ct.zeros((head, length * group, right), ct.float32)
    logits = ct.mma(query, key.transpose(1, 2), logits)
    return ct.where(
        right_mask[None, None, :] & query_mask[None, :, None],
        logits,
        float("-inf"),
    )


@ct.function
def map_reduce(tid, query, key, value):
    length, group, head, _, dv, _ = tid.shape("l", "g", "h", "r", "dv", "dqk")
    logits = attention_map(tid, query, key) * LOG2E
    maximum = ct.max(logits, axis=2)
    probabilities = ct.exp2(logits - maximum[:, :, None])
    exponential_sum = ct.sum(probabilities, axis=2)
    numerator = ct.zeros((head, length * group, dv), ct.float32)
    numerator = ct.mma(probabilities.astype(ct.bfloat16), value, numerator)
    return (
        maximum.reshape((head, length, group)),
        exponential_sum.reshape((head, length, group)),
        numerator.reshape((head, length, group, dv)),
    )


@ct.function
def map_reduce_combine(tid, query, key, value, acc):
    acc_m, acc_e, acc_u = acc
    length, group, head, _, dv, _ = tid.shape(
        "l", "g", "h", "r", "dv", "dqk"
    )
    logits = attention_map(tid, query, key) * LOG2E
    local_m = ct.max(logits, axis=2)
    local_weights = ct.exp2(logits - local_m[:, :, None])
    local_e = ct.sum(local_weights, axis=2)

    acc_m = acc_m.reshape((head, length * group))
    acc_e = acc_e.reshape((head, length * group))
    acc_u = acc_u.reshape((head, length * group, dv))
    acc_is_high = acc_m > local_m
    high_m = ct.where(acc_is_high, acc_m, local_m)
    low_m = ct.where(acc_is_high, local_m, acc_m)
    skip = high_m == float("-inf")
    scaling = ct.exp2(ct.where(skip, 0.0, low_m - high_m))
    acc_scale = ct.where(acc_is_high, 1.0, scaling)
    local_scale = ct.where(acc_is_high, scaling, 1.0)

    numerator = acc_u * acc_scale[:, :, None]
    numerator = ct.mma(
        (local_weights * local_scale[:, :, None]).astype(ct.bfloat16),
        value,
        numerator,
    )
    return (
        high_m.reshape((head, length, group)),
        ct.where(
            skip,
            acc_e + local_e,
            acc_e * acc_scale + local_e * local_scale,
        ).reshape((head, length, group)),
        ct.where(
            skip[:, :, None],
            acc_u,
            numerator,
        ).reshape((head, length, group, dv)),
    )


@ct.function
def combine(am, ae, au, bm, be, bu):
    left_is_high = am > bm
    high_m = ct.where(left_is_high, am, bm)
    high_e = ct.where(left_is_high, ae, be)
    high_u = ct.where(left_is_high[:, :, :, None], au, bu)
    low_m = ct.where(left_is_high, bm, am)
    low_e = ct.where(left_is_high, be, ae)
    low_u = ct.where(left_is_high[:, :, :, None], bu, au)
    skip = high_m == float("-inf")
    scaling = ct.exp2(low_m - high_m)
    return (
        high_m,
        ct.where(skip, high_e, high_e + low_e * scaling),
        ct.where(
            skip[:, :, :, None],
            high_u,
            high_u + low_u * scaling[:, :, :, None],
        ),
    )


@ct.function
def to_semantic(maximum, exponential_sum, numerator):
    return (
        (maximum + ct.log2(exponential_sum)) * LN2,
        numerator / exponential_sum[:, :, :, None],
    )


def to_output(logsumexp, mean):
    del logsumexp
    return mean


@ct.function
def embed(logsumexp, mean, g_logsumexp, g_mean):
    head, length, group = logsumexp.shape
    dv = mean.shape[3]
    return (
        logsumexp.reshape((head, length * group)),
        (g_logsumexp - ct.sum(mean * g_mean, axis=3)).reshape(
            (head, length * group)
        ),
        g_mean.astype(ct.bfloat16).reshape((head, length * group, dv)),
    )


@ct.function
def finalize(
        tid,
        query,
        key,
        value,
        g_query,
        g_key,
        g_value,
        logsumexp,
        g_logsumexp,
        g_mean,
        ):
    logits = attention_map(tid, query, key)
    length, group, head, right, dv, dqk = tid.shape(
        "l", "g", "h", "r", "dv", "dqk"
    )
    scale = ct.exp(logits - logsumexp[:, :, None])
    g_value = ct.mma(scale.transpose(1, 2).astype(ct.bfloat16), g_mean, g_value)
    g_logits = ct.broadcast_to(
        g_logsumexp[:, :, None],
        (head, length * group, right),
    )
    g_logits = ct.mma(g_mean, value.transpose(1, 2), g_logits)
    g_logits = (scale * g_logits).astype(ct.bfloat16)
    g_query = ct.mma(
        g_logits,
        key,
        g_query.reshape((head, length * group, dqk)),
    ).reshape((head, length, group, dqk))
    g_key = ct.mma(
        g_logits.transpose(1, 2),
        query.reshape((head, length * group, dqk)),
        g_key,
    )
    return g_query, g_key, g_value


FUNCTIONS = fold_functions(
    map_reduce,
    combine,
    to_semantic,
    to_output,
    map_reduce_combine=map_reduce_combine,
    embed=embed,
    finalize=finalize,
)


def sdpa(query, key, value):
    _, heads, length, group, dqk = (1, *query.shape)
    right, dv = key.shape[1], value.shape[-1]
    query = query.transpose(1, 2).reshape(1, heads * group, length, dqk)
    key = key.reshape(1, heads, right, dqk)
    value = value.reshape(1, heads, right, dv)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        enable_gqa=True,
        scale=1.0,
        dropout_p=0.0,
    )
    return output.reshape(1, heads, group, length, dv)[0].transpose(1, 2)


def make_inputs(sizes):
    query = torch.randn(
        sizes["h"], sizes["l"], sizes["g"], sizes["dqk"],
        device="cuda", dtype=torch.bfloat16,
    )
    key = torch.randn(
        sizes["h"], sizes["r"], sizes["dqk"],
        device="cuda", dtype=torch.bfloat16,
    )
    value = torch.randn(
        sizes["h"], sizes["r"], sizes["dv"],
        device="cuda", dtype=torch.bfloat16,
    )
    for tensor in (query, key, value):
        tensor.requires_grad_()
    with torch.no_grad():
        query.mul_(sizes["dqk"] ** -0.5)
    return query, key, value


def validate(operator, plan, sizes, *, accuracy_matrix=False):
    print("SDPA correctness validation", flush=True)
    reference_dtypes = {
        "PyTorch BF16": torch.bfloat16,
        "PyTorch FP32": torch.float32,
    }
    if accuracy_matrix:
        reference_dtypes["PyTorch FP64"] = torch.float64
    validate_precision_matrix(
        operator.build(plan),
        sdpa,
        make_inputs(sizes),
        input_names=("query", "key", "value"),
        reference_dtypes=reference_dtypes,
        pairwise=accuracy_matrix,
    )


def benchmark_full(
        operator, plan, sizes, min_run_time, *,
        torch_compile=False,
        measure_memory=False,
        ):
    implementations = {
        "CuTile eager": operator.build(plan),
        "PyTorch SDPA": sdpa,
    }
    if torch_compile:
        implementations["CuTile torch.compile"] = operator.build(
            plan, torch_compile=True,
        )
    inputs = make_inputs(sizes)
    if min_run_time > 0:
        print("end-to-end timing comparison", flush=True)
        benchmark_implementations(
            "attention",
            inputs,
            implementations,
            min_run_time=min_run_time,
        )
    if measure_memory:
        benchmark_memory("attention", inputs, implementations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--right", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--dqk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--quiet-tuning", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--accuracy-matrix", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float, default=1.0)
    parser.add_argument("--benchmark-memory", action="store_true")
    args = parser.parse_args()
    if args.candidates <= 0:
        parser.error("--candidates must be positive")
    sizes = {
        "h": args.heads,
        "l": args.length,
        "g": args.groups,
        "r": args.right,
        "dqk": args.dqk,
        "dv": args.dv,
    }
    torch.manual_seed(args.seed)
    operator = FoldOperator(attention_spec(), FUNCTIONS)
    plan = operator.tune(
        sizes,
        args.candidates,
        args.timeout,
        hardware=rtx5080,
        quiet=args.quiet_tuning,
    )
    if args.benchmark_seconds > 0 or args.benchmark_memory:
        benchmark_full(
            operator, plan, sizes, args.benchmark_seconds,
            torch_compile=args.torch_compile,
            measure_memory=args.benchmark_memory,
        )
    if args.validate:
        validate(
            operator, plan, sizes,
            accuracy_matrix=args.accuracy_matrix,
        )


if __name__ == "__main__":
    main()
