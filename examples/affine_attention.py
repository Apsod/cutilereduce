from .cli import main as example_main
import math

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import (
    AlgebraKind,
    fold_functions,
    make_fold_spec,
    matmul,
    workmodel,
)


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
        semantic={
            "alpha": buffer_spec("h l g", ct.float32, default=0),
            "z": buffer_spec("h l g", ct.float32, default=float("-inf")),
            "mu": buffer_spec("h l g dv", ct.float32, default=0),
        },
        # Logical tile-local values used by the map/map_finalize callbacks. These
        # are heuristic residency declarations, not materialized buffers.
        map_fold_intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
            "alpha": buffer_spec("h l g r", ct.float32),
        },
        map_finalize_intermediate={
            "logits": buffer_spec("h l g r", ct.float32),
            "alpha": buffer_spec("h l g r", ct.float32),
            "local_weights": buffer_spec("h l g r", ct.float32),
            "probability": buffer_spec("h l g r", ct.float32),
            "previous_mass": buffer_spec("h l g r", ct.float32),
            "previous_projection": buffer_spec("h l g r", ct.float32),
            "g_score": buffer_spec("h l g r", ct.bfloat16),
            "g_bias_score": buffer_spec("h l g r", ct.bfloat16),
        },
        batch="h l g",
        fold="r",
        map_fold_work=workmodel(
            matmul(B="h", M="l g", N="r", K="dqk"),
            matmul(B="h", M="l g", N="r", K="db"),
            matmul(B="h", M="l g", N="dv", K="r"),
        ),
        map_finalize_work=workmodel(
            matmul(B="h", M="l g", N="r", K="dqk"),
            matmul(B="h", M="l g", N="r", K="db"),
            matmul(B="h", M="l g", N="dqk", K="r"),
            matmul(B="h", M="r", N="dqk", K="l g"),
            matmul(B="h", M="r", N="dv", K="l g"),
            matmul(B="h", M="l g", N="db", K="r"),
            matmul(B="h", M="r", N="db", K="l g"),
        ),
        algebra=AlgebraKind.general,
    )


@ct.function
def combine(a_alpha, a_m, a_e, a_u, b_alpha, b_m, b_e, b_u):
    shifted_a_m = a_m + b_alpha
    left_is_high = shifted_a_m > b_m
    high_m = ct.where(left_is_high, shifted_a_m, b_m)
    low_m = ct.where(left_is_high, b_m, shifted_a_m)
    skip = high_m == float("-inf")
    scaling = ct.exp2(ct.where(skip, 0.0, low_m - high_m))
    a_scale = ct.where(left_is_high, 1.0, scaling)
    b_scale = ct.where(left_is_high, scaling, 1.0)
    return (
        a_alpha + b_alpha,
        high_m,
        ct.where(skip, a_e + b_e, a_e * a_scale + b_e * b_scale),
        ct.where(
            skip[:, :, :, None],
            a_u + b_u,
            a_u * a_scale[:, :, :, None] + b_u * b_scale[:, :, :, None],
        ),
    )


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
    alpha = ct.where(factor_mask, bias_logits * LOG2E, 0.0)
    logits = ct.where(factor_mask, logits * LOG2E, float("-inf"))
    return logits, alpha


@ct.function
def map_fold(tid, query, key, value, bias_query, bias_key):
    length, group, head, _, _, dv, _ = tid.shape(
        "l", "g", "h", "r", "dqk", "dv", "db"
    )
    logits, alpha = affine_attention_map(
        tid, query, key, bias_query, bias_key,
    )
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
    head, length, group = alpha.shape
    dv = mu.shape[3]
    return (
        (alpha * LOG2E).reshape((head, length * group)),
        (z * LOG2E).reshape((head, length * group)),
        (g_z - ct.sum(mu * g_mu, axis=3)).reshape(
            (head, length * group)
        ),
        g_mu.astype(ct.bfloat16).reshape((head, length * group, dv)),
    )


@ct.function
def map_finalize(
        tid,
        query, key, value, bias_query, bias_key,
        g_query, g_key, g_value, g_bias_query, g_bias_key,
        total_alpha, total_z, accumulator_w, accumulator_s,
        prefix_alpha, prefix_m, prefix_e, prefix_u,
        ):
    length, group, head, right, dqk, dv, db = tid.shape(
        "l", "g", "h", "r", "dqk", "dv", "db"
    )
    logits, alpha = affine_attention_map(
        tid, query, key, bias_query, bias_key,
    )

    query_flat = query.reshape((head, length * group, dqk))
    bias_query_flat = bias_query.reshape((head, length * group, db))

    prefix_alpha_flat = prefix_alpha.reshape((head, length * group))
    total_alpha_flat = total_alpha
    total_z_flat = total_z
    inclusive_local_alpha = ct.cumsum(alpha, axis=2)
    local_alpha = ct.sum(alpha, axis=2)
    adjusted_logits = (
        logits + local_alpha[:, :, None] - inclusive_local_alpha
    )
    local_m = ct.max(adjusted_logits, axis=2)
    local_weights = ct.exp2(adjusted_logits - local_m[:, :, None])
    local_e = ct.sum(local_weights, axis=2)
    global_shift = total_alpha_flat - prefix_alpha_flat - local_alpha
    probability_scale = ct.exp2(local_m + global_shift - total_z_flat)
    probability = local_weights * probability_scale[:, :, None]

    prefix_valid = prefix_e != 0
    safe_prefix_e = ct.where(prefix_valid, prefix_e, 1.0)
    prefix_z = ct.where(
        prefix_valid,
        prefix_m + ct.log2(safe_prefix_e),
        float("-inf"),
    ).reshape(
        (head, length * group, 1)
    )
    prefix_mass = ct.exp2(
        prefix_z
        + total_alpha_flat[:, :, None]
        - prefix_alpha_flat[:, :, None]
        - total_z_flat[:, :, None]
    )
    prefix_mass = ct.where(
        prefix_valid.reshape((head, length * group, 1)),
        prefix_mass,
        0.0,
    )
    prefix_mu = (prefix_u / safe_prefix_e[..., None]).reshape(
        (head, length * group, dv)
    )
    previous_mass = prefix_mass + ct.cumsum(probability, axis=2) - probability

    accumulator_s_float = accumulator_s.astype(ct.float32)
    accumulator_w = accumulator_w[:, :, None]
    value_projection = ct.sum(
        accumulator_s_float[:, :, None, :] * value[:, None, :, :],
        axis=3,
    )
    weighted_projection = probability * value_projection
    prefix_projection = ct.sum(
        accumulator_s_float * prefix_mu,
        axis=2,
    )
    previous_projection = (
        prefix_mass * prefix_projection[:, :, None]
        + ct.cumsum(weighted_projection, axis=2)
        - weighted_projection
    )
    prior_grad = (
        previous_mass * accumulator_w
        + previous_projection
    )
    g_score = probability * (accumulator_w + value_projection)
    g_score_bf16 = g_score.astype(ct.bfloat16)
    g_bias_score = prior_grad.astype(ct.bfloat16)

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

    local_u = ct.zeros((head, length * group, dv), ct.float32)
    local_u = ct.mma(local_weights.astype(ct.bfloat16), value, local_u)
    next_state = combine(
        prefix_alpha,
        prefix_m,
        prefix_e,
        prefix_u,
        local_alpha.reshape((head, length, group)),
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
    map_fold,
    combine,
    to_semantic,
    to_output,
    embed=embed,
    map_finalize=map_finalize,
)


def reference(
        query, key, value, bias_query, bias_key, *,
        dtype=None,
        ):
    dtype = dtype or query.dtype
    query = query.to(dtype)
    key = key.to(dtype)
    value = value.to(dtype)
    bias_query = bias_query.to(dtype)
    bias_key = bias_key.to(dtype)
    logits = torch.einsum("hlgd,hrd->hlgr", query, key)
    bias_logits = torch.einsum(
        "hlgd,hrd->hlgr", bias_query, bias_key
    )
    alpha = bias_logits
    suffix_after = alpha.flip(-1).cumsum(-1).flip(-1) - alpha
    weights = torch.softmax(logits + suffix_after, dim=-1)
    return torch.einsum("hlgr,hrd->hlgd", weights, value)


INITIALIZERS = {
    'query': lambda t, s: t.normal_().mul_(s["dqk"] ** -0.25),
    'bias_query': lambda t, s: t.normal_().mul_(s["db"] ** -0.25),
    'key': lambda t, s: t.normal_().mul_(s["dqk"] ** -0.25),
    'bias_key': lambda t, s: t.normal_().mul_(s["db"] ** -0.25),
    'value': lambda t, s: t.normal_(),

}

def main():
    example_main(
        'affine_attention', affine_attention_spec(), FUNCTIONS, {'h': 2, 'l': 4096, 'g': 2, 'r': 4096, 'dqk': 64, 'dv': 64, 'db': 32},
        aliases={'h': 'heads', 'l': 'length', 'g': 'groups', 'r': 'right'}, benchmark_seconds=0.5,
        reference=reference, references={
            "PyTorch FP32": lambda *inputs: reference(*inputs, dtype=torch.float32),
            "PyTorch BF16": lambda *inputs: reference(*inputs, dtype=torch.bfloat16),
        },
        initializers=INITIALIZERS,
    )


if __name__ == "__main__":
    main()
