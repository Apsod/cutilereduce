import argparse
import math

import cuda.tile as ct
import torch
import torch.utils.benchmark as torch_benchmark

from cutilereduce.core import MatMulWork, WorkModel
from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import (
    AlgebraKind,
    StageSchedule,
    fold_functions,
    make_fold_spec,
    mk_fold_autograd,
    mk_fold_forward,
    tune_general_fold_plan,
)
from cutilereduce.fold.general import partial_fold_plan
from cutilereduce.util.spec import rtx5080


LOG2E = math.log2(math.e)
LN2 = math.log(2)


def affine_attention_spec():
    return make_fold_spec(
        input={
            "query": buffer_spec("h l g dqk", ct.bfloat16, req_grad=True, default=0),
            "key": buffer_spec("h r dqk", ct.bfloat16, req_grad=True, default=0),
            "value": buffer_spec("h r dv", ct.bfloat16, req_grad=True, default=0),
            "bias_query": buffer_spec("h l g db", ct.bfloat16, req_grad=True, default=0),
            "bias_key": buffer_spec("h r db", ct.bfloat16, req_grad=True, default=0),
        },
        execution={
            "alpha": buffer_spec("h l g", ct.float32, default=0),
            "m": buffer_spec("h l g", ct.float32, default=float("-inf")),
            "e": buffer_spec("h l g", ct.float32, default=0),
            "u": buffer_spec("h l g dv", ct.float32, default=0),
        },
        output={
            "alpha": buffer_spec("h l g", ct.float32, default=0),
            "z": buffer_spec("h l g", ct.float32, default=float("-inf")),
            "mu": buffer_spec("h l g dv", ct.float32, default=0),
        },
        # Logical tile-local values used by the map/finalize callbacks. These
        # are heuristic residency declarations, not materialized buffers.
        intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
            "bias_logits": buffer_spec("h l g r", ct.float32),
        },
        batch="h l g",
        fold="r",
        map_fold_work=WorkModel.make(
            MatMulWork.make(B="h", M="l g", N="r", K="dqk"),
            MatMulWork.make(B="h", M="l g", N="r", K="db"),
            MatMulWork.make(B="h", M="l g", N="dv", K="r"),
        ),
        backward_work=WorkModel.make(
            MatMulWork.make(B="h", M="l g", N="r", K="dqk"),
            MatMulWork.make(B="h", M="l g", N="r", K="db"),
            MatMulWork.make(B="h", M="l g", N="dqk", K="r"),
            MatMulWork.make(B="h", M="r", N="dqk", K="l g"),
            MatMulWork.make(B="h", M="r", N="dv", K="l g"),
            MatMulWork.make(B="h", M="l g", N="db", K="r"),
            MatMulWork.make(B="h", M="r", N="db", K="l g"),
        ),
        algebra=AlgebraKind.general,
    )


@ct.function
def combine(a_alpha, a_m, a_e, a_u, b_alpha, b_m, b_e, b_u):
    shifted_a_m = a_m + b_alpha
    m = ct.maximum(shifted_a_m, b_m)
    skip = m == float("-inf")
    a_scale = ct.exp2(shifted_a_m - m)
    b_scale = ct.exp2(b_m - m)
    e = ct.where(skip, a_e + b_e, a_e * a_scale + b_e * b_scale)
    u = ct.where(
        skip[..., None],
        a_u + b_u,
        a_u * a_scale[..., None] + b_u * b_scale[..., None],
    )
    return a_alpha + b_alpha, m, e, u


@ct.function
def affine_attention_map(tid, query, key, bias_query, bias_key):
    length, group, head, right, dqk, db = tid.shape(
        "l", "g", "h", "r", "dqk", "db"
    )
    query_mask = (
        tid.mask("l")[:, None] & tid.mask("g")[None, :]
    ).reshape((length * group,))
    factor_mask = query_mask[None, :, None] & tid.mask("r")[None, None, :]

    query = query.reshape((head, length * group, dqk))
    logits = ct.zeros((head, length * group, right), ct.float32)
    logits = ct.mma(query, key.transpose(1, 2), logits)

    bias_query = bias_query.reshape((head, length * group, db))
    bias_logits = ct.zeros((head, length * group, right), ct.float32)
    bias_logits = ct.mma(bias_query, bias_key.transpose(1, 2), bias_logits)
    alpha = -ct.exp(bias_logits)

    alpha = ct.where(factor_mask, alpha, 0.0)
    logits = ct.where(factor_mask, logits, float("-inf"))
    return logits, alpha


@ct.function
def map_reduce(tid, query, key, value, bias_query, bias_key):
    length, group, head, _, _, dv, _ = tid.shape(
        "l", "g", "h", "r", "dqk", "dv", "db"
    )
    logits, alpha = affine_attention_map(
        tid, query, key, bias_query, bias_key,
    )
    logits = logits * LOG2E
    alpha = alpha * LOG2E
    suffix_after = ct.cumsum(alpha, axis=2, reverse=True) - alpha
    adjusted_logits = logits + suffix_after
    maximum = ct.max(adjusted_logits, axis=2)
    weights = ct.exp2(adjusted_logits - maximum[:, :, None])
    exponential_sum = ct.sum(weights, axis=2)
    numerator = ct.zeros((head, length * group, dv), ct.float32)
    numerator = ct.mma(weights.astype(ct.bfloat16), value, numerator)
    return (
        ct.sum(alpha, axis=2).reshape((head, length, group)),
        maximum.reshape((head, length, group)),
        exponential_sum.reshape((head, length, group)),
        numerator.reshape((head, length, group, dv)),
    )


@ct.function
def to_semantic(alpha, maximum, exponential_sum, numerator):
    return (
        alpha * LN2,
        (maximum + ct.log2(exponential_sum)) * LN2,
        numerator / exponential_sum[..., None],
    )


def to_output(alpha, z, mu):
    del alpha, z
    return mu


@ct.function
def embed(alpha, z, mu, g_alpha, g_z, g_mu):
    return (
        alpha,
        z,
        g_z - ct.sum(mu * g_mu, axis=3),
        g_mu.astype(ct.bfloat16),
    )


@ct.function
def finalize(
        tid,
        query, key, value, bias_query, bias_key,
        g_query, g_key, g_value, g_bias_query, g_bias_key,
        total_alpha, total_z, accumulator_w, accumulator_s,
        prefix_alpha, prefix_m, prefix_e, prefix_u,
        ):
    length, group, head, right, dqk, dv, db = tid.shape(
        "l", "g", "h", "r", "dqk", "dv", "db"
    )
    query_flat = query.reshape((head, length * group, dqk))
    bias_query_flat = bias_query.reshape((head, length * group, db))
    logits, alpha = affine_attention_map(
        tid, query, key, bias_query, bias_key,
    )

    prefix_alpha_flat = prefix_alpha.reshape((head, length * group)) * LN2
    total_alpha_flat = total_alpha.reshape((head, length * group))
    total_z_flat = total_z.reshape((head, length * group))
    inclusive_alpha = prefix_alpha_flat[:, :, None] + ct.cumsum(alpha, axis=2)
    scores = logits + total_alpha_flat[:, :, None] - inclusive_alpha
    probability = ct.exp(scores - total_z_flat[:, :, None])

    prefix_z = ((prefix_m + ct.log2(prefix_e)) * LN2).reshape(
        (head, length * group, 1)
    )
    prefix_mass = ct.exp(
        prefix_z
        + total_alpha_flat[:, :, None]
        - prefix_alpha_flat[:, :, None]
        - total_z_flat[:, :, None]
    )
    prefix_mass = ct.where(prefix_e.reshape((head, length * group, 1)) == 0, 0.0, prefix_mass)
    prefix_mu = ct.where(
        prefix_e[..., None] == 0,
        0.0,
        prefix_u / prefix_e[..., None],
    ).reshape((head, length * group, dv))
    previous_mass = prefix_mass + ct.cumsum(probability, axis=2) - probability
    weighted_value = probability[..., None] * value[:, None, :, :]
    previous_value = (
        prefix_mass[..., None] * prefix_mu[:, :, None, :]
        + ct.cumsum(weighted_value, axis=2)
        - weighted_value
    )

    accumulator_s = accumulator_s.reshape((head, length * group, dv))
    accumulator_s_float = accumulator_s.astype(ct.float32)
    accumulator_w = accumulator_w.reshape((head, length * group, 1))
    prior_grad = (
        previous_mass * accumulator_w
        + ct.sum(
            accumulator_s_float[:, :, None, :] * previous_value,
            axis=3,
        )
    )
    g_score = probability * (
        accumulator_w
        + ct.sum(
            accumulator_s_float[:, :, None, :] * value[:, None, :, :],
            axis=3,
        )
    )
    g_score_bf16 = g_score.astype(ct.bfloat16)
    g_bias_score = (prior_grad * alpha).astype(ct.bfloat16)

    g_value = ct.mma(
        probability.transpose(1, 2).astype(ct.bfloat16),
        accumulator_s,
        g_value,
    )
    g_query = ct.mma(g_score_bf16, key, g_query.reshape((head, length * group, dqk)))
    g_key = ct.mma(g_score_bf16.transpose(1, 2), query_flat, g_key)
    g_bias_query = ct.mma(
        g_bias_score,
        bias_key,
        g_bias_query.reshape((head, length * group, db)),
    )
    g_bias_key = ct.mma(g_bias_score.transpose(1, 2), bias_query_flat, g_bias_key)

    alpha_execution = alpha * LOG2E
    logits_execution = logits * LOG2E
    suffix_after = ct.cumsum(alpha_execution, axis=2, reverse=True) - alpha_execution
    adjusted_logits = logits_execution + suffix_after
    local_m = ct.max(adjusted_logits, axis=2)
    local_weights = ct.exp2(adjusted_logits - local_m[:, :, None])
    local_e = ct.sum(local_weights, axis=2)
    local_u = ct.zeros((head, length * group, dv), ct.float32)
    local_u = ct.mma(local_weights.astype(ct.bfloat16), value, local_u)
    next_state = combine(
        prefix_alpha,
        prefix_m,
        prefix_e,
        prefix_u,
        ct.sum(alpha_execution, axis=2).reshape((head, length, group)),
        local_m.reshape((head, length, group)),
        local_e.reshape((head, length, group)),
        local_u.reshape((head, length, group, dv)),
    )
    return (
        (
            g_query.reshape((head, length, group, dqk)),
            g_key,
            g_value,
            g_bias_query.reshape((head, length, group, db)),
            g_bias_key,
        ),
        next_state,
    )


FUNCTIONS = fold_functions(
    map_reduce,
    combine,
    to_semantic,
    to_output,
    embed=embed,
    finalize=finalize,
)


def make_plan(spec, sizes, *, fold_tile, partitions, checkpointed_backward=True):
    partition_axis = spec.fold.partition_axis
    partial_schedule = StageSchedule.make(
        spec,
        extents=sizes,
        tiles={
            "h": 1,
            "l": min(8, sizes["l"]),
            "g": sizes["g"],
            "r": min(fold_tile, sizes["r"]),
            "dqk": sizes["dqk"],
            "dv": sizes["dv"],
            "db": sizes["db"],
        },
        programs={partition_axis: partitions},
        loop=spec.fold,
    )
    scan_schedule = StageSchedule.make(
        spec,
        extents={**sizes, partition_axis: partitions},
        tiles={
            "h": 1,
            "l": min(8, sizes["l"]),
            "g": sizes["g"],
            "dqk": sizes["dqk"],
            "dv": sizes["dv"],
            "db": sizes["db"],
            partition_axis: 1,
        },
        programs={partition_axis: 1},
        loop=partition_axis,
    )
    return partial_fold_plan(
        spec,
        partial_schedule,
        scan_schedule,
        backward_schedule=partial_schedule,
        checkpointed_backward=checkpointed_backward,
    )


def reference(query, key, value, bias_query, bias_key):
    logits = torch.einsum("hlgd,hrd->hlgr", query.float(), key.float())
    bias_logits = torch.einsum(
        "hlgd,hrd->hlgr", bias_query.float(), bias_key.float()
    )
    alpha = -torch.exp(bias_logits)
    suffix_after = alpha.flip(-1).cumsum(-1).flip(-1) - alpha
    weights = torch.softmax(logits + suffix_after, dim=-1)
    return torch.einsum("hlgr,hrd->hlgd", weights, value.float())


def make_inputs(sizes):
    shapes = (
        (sizes["h"], sizes["l"], sizes["g"], sizes["dqk"]),
        (sizes["h"], sizes["r"], sizes["dqk"]),
        (sizes["h"], sizes["r"], sizes["dv"]),
        (sizes["h"], sizes["l"], sizes["g"], sizes["db"]),
        (sizes["h"], sizes["r"], sizes["db"]),
    )
    tensors = tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for shape in shapes
    )
    query, key, value, bias_query, bias_key = tensors
    query.mul_(sizes["dqk"] ** -0.5)
    bias_query.mul_(sizes["db"] ** -0.5)
    for tensor in tensors:
        tensor.requires_grad_()
    return query, key, value, bias_query, bias_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--right", type=int, default=512)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--dqk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--db", type=int, default=32)
    parser.add_argument("--fold-tile", type=int, default=64)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--max-tile", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=0)
    parser.add_argument("--quiet-tuning", action="store_true")
    parser.add_argument("--fixed-plan", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--full-recompute-backward", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.torch_compile and args.forward_only:
        parser.error("--torch-compile currently requires the autograd plan")
    sizes = {
        "h": args.heads,
        "l": args.length,
        "g": args.groups,
        "r": args.right,
        "dqk": args.dqk,
        "dv": args.dv,
        "db": args.db,
    }
    spec = affine_attention_spec()
    if args.fixed_plan:
        plan = make_plan(
            spec,
            sizes,
            fold_tile=args.fold_tile,
            partitions=args.partitions,
            checkpointed_backward=not args.full_recompute_backward,
        )
    else:
        plan = tune_general_fold_plan(
            spec,
            sizes,
            args.candidates,
            args.timeout,
            functions=FUNCTIONS,
            hardware=rtx5080,
            max_tile=args.max_tile,
            max_partition_count=args.partitions,
            quiet_tuning=args.quiet_tuning,
            backward=not args.forward_only,
        )
    eager_function = (
        mk_fold_forward(plan, FUNCTIONS)
        if args.forward_only
        else mk_fold_autograd(plan, FUNCTIONS)
    )
    function = (
        torch.compile(eager_function, fullgraph=True, mode="reduce-overhead")
        if args.torch_compile and not args.forward_only else eager_function
    )
    inputs = make_inputs(sizes)
    cutile, = function(*inputs)
    pytorch = reference(*inputs)
    print(f"forward mean absolute error: {(cutile - pytorch).abs().mean().item():.6g}")
    print("stages:", ", ".join(stage.stage.name for stage in plan.forward))
    if not args.forward_only:
        grad = torch.randn_like(cutile)
        cutile_grads = torch.autograd.grad(cutile, inputs, grad, retain_graph=True)
        pytorch_grads = torch.autograd.grad(pytorch, inputs, grad)
        for name, actual, expected in zip(
                ("query", "key", "value", "bias_query", "bias_key"),
                cutile_grads,
                pytorch_grads,
                strict=True,
                ):
            error = (actual.float() - expected.float()).abs().mean().item()
            print(f"{name} gradient mean absolute error: {error:.6g}")
        print("backward:", ", ".join(stage.stage.name for stage in plan.backward))

    if args.benchmark_seconds > 0:
        cutile_label = "CuTile torch.compile" if args.torch_compile else "CuTile eager"

        def cutile_forward():
            with torch.no_grad():
                return function(*inputs)[0]

        def pytorch_forward():
            with torch.no_grad():
                return reference(*inputs)

        cases = [
            (f"{cutile_label} forward", cutile_forward),
            ("PyTorch forward", pytorch_forward),
        ]
        if not args.forward_only:
            cases.extend((
                (f"{cutile_label} forward + backward", lambda: torch.autograd.grad(
                    function(*inputs)[0], inputs, grad
                )),
                ("PyTorch forward + backward", lambda: torch.autograd.grad(
                    reference(*inputs), inputs, grad
                )),
            ))
        measurements = [
            torch_benchmark.Timer(
                stmt="call()",
                globals={"call": call},
                label="affine LWS attention forward",
                sub_label=name,
                description="",
                num_threads=1,
            ).blocked_autorange(min_run_time=args.benchmark_seconds)
            for name, call in cases
        ]
        torch_benchmark.Compare(measurements).print()


if __name__ == "__main__":
    main()
