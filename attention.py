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
        intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
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
    return (
        logsumexp,
        g_logsumexp - ct.sum(mean * g_mean, axis=3),
        g_mean.astype(ct.bfloat16),
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
    scale = ct.exp(logits - logsumexp.reshape((head, length * group, 1)))
    g_mean = g_mean.reshape((head, length * group, dv))
    g_value = ct.mma(scale.transpose(1, 2).astype(ct.bfloat16), g_mean, g_value)
    g_logits = ct.broadcast_to(
        g_logsumexp.reshape((head, length * group, 1)),
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


def validate(plan, sizes):
    print("SDPA correctness validation", flush=True)
    inputs = make_inputs(sizes)
    function = mk_fold_autograd(plan, FUNCTIONS)
    checked = run_grad(
        *inputs,
        cutile=lambda *current: function(*current)[0],
        pytorch=sdpa,
    )
    for cutile, pytorch in zip(
            checked["cutile"]["fwd"], checked["pytorch"]["fwd"], strict=True,
            ):
        print(f"forward mean absolute error: {(cutile - pytorch).abs().mean().item():.6g}")
    for name, cutile, pytorch in zip(
            ("query", "key", "value"),
            checked["cutile"]["bwd"],
            checked["pytorch"]["bwd"],
            strict=True,
            ):
        print(f"{name} grad mean absolute error: {(cutile - pytorch).abs().mean().item():.6g}")


def benchmark_full(plan, sizes, min_run_time, *, torch_compile=False):
    print("end-to-end timing comparison", flush=True)
    query, key, value = make_inputs(sizes)
    eager_function = mk_fold_autograd(plan, FUNCTIONS)
    compiled_function = (
        torch.compile(eager_function, fullgraph=True, mode="reduce-overhead")
        if torch_compile else None
    )
    mock = torch.randn(
        sizes["h"], sizes["l"], sizes["g"], sizes["dv"], device="cuda"
    )

    def cutile_forward(function):
        with torch.no_grad():
            return function(query, key, value)[0]

    def cutile_train_forward(function):
        return function(query, key, value)[0]

    def sdpa_forward():
        with torch.no_grad():
            return sdpa(query, key, value)

    def sdpa_train_forward():
        return sdpa(query, key, value)

    def backward(forward):
        for tensor in (query, key, value):
            tensor.grad = None
        (forward() * mock).sum().backward()

    cases = [
        ("forward", "CuTile eager", lambda: cutile_forward(eager_function)),
        ("forward", "PyTorch SDPA", sdpa_forward),
        ("forward + backward", "CuTile eager", lambda: backward(lambda: cutile_train_forward(eager_function))),
        ("forward + backward", "PyTorch SDPA", lambda: backward(sdpa_train_forward)),
    ]
    if compiled_function is not None:
        backward(lambda: cutile_train_forward(compiled_function))
        cases.extend((
            ("forward", "CuTile torch.compile", lambda: cutile_forward(compiled_function)),
            ("forward + backward", "CuTile torch.compile", lambda: backward(lambda: cutile_train_forward(compiled_function))),
        ))
    measurements = []
    for label, implementation, function_to_time in cases:
        measurements.append(torch_benchmark.Timer(
            stmt="function_to_time()",
            globals={"function_to_time": function_to_time},
            label=f"attention {label}",
            sub_label=implementation,
            description="selected configuration",
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time))
    torch_benchmark.Compare(measurements).print()


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
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float, default=1.0)
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
    plan = tune_commutative_fold_plan(
        attention_spec(),
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
