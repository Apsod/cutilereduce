"""Native CPU fusion preserves broadcasting, numerical domains and ownership."""

import hashlib

import pytest
import torch

from cutilereduce.experimental.ir.core import TensorRegionBuilder
from cutilereduce.experimental import TensorSignature
from cutilereduce.experimental.backends.cpu.regions import _emit_region
from cutilereduce.experimental.backends.reference import _execute_region


def _load(regions, sizes):
    from torch.utils.cpp_extension import load_inline
    source = "\n\n".join(_emit_region(name, region, sizes) for name, region in regions.items())
    return load_inline(
        "cutilereduce_fusion_" + hashlib.sha256(source.encode()).hexdigest()[:16],
        cpp_sources=source, functions=list(regions),
        extra_cflags=["-O2", "-march=native"], with_cuda=False,
    )


def _pointwise_region(dtype="float32"):
    ctx = TensorRegionBuilder(("b", "r"))
    x = ctx.expect("x", TensorSignature(("b", "r"), dtype))
    bias = ctx.expect("bias", TensorSignature(("b",), dtype))
    y = ctx.expect("y", TensorSignature(("b", "r"), dtype))
    exponential = ctx.exp(x - bias)
    masked = ctx.where(ctx.compare("gt", x, bias), exponential, ctx.full(exponential.axes, 0.0, dtype))
    ctx.yield_value("masked", masked)
    ctx.yield_value("sum", ctx.reduce("sum", x + y, ("r",)))
    ctx.yield_value("max", ctx.reduce("max", x + bias, ("r",)))
    ctx.yield_value("dot", ctx.contract(x, y, over=("r",), dtype=dtype))
    ctx.yield_value("probability", ctx.where(ctx.compare("gt", x, bias), ctx.sigmoid(x), ctx.full(x.axes, 0.0, dtype)))
    return ctx.finish_region()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_fusion_strides_tails_nan_and_broadcast(dtype):
    sizes = {"b": 3, "r": 17}
    region = _pointwise_region(dtype)
    module = _load({"fused": region}, sizes)
    source = _emit_region("fused", region, sizes)
    assert "fused reduction" in source
    assert "Vec::blendv" in source
    assert "at::exp(" not in source
    for transpose in (False, True):
        x = (torch.randn(17, 3).T if transpose else torch.randn(3, 17)).to(getattr(torch, dtype))
        x[0, 0] = float('nan')
        inputs = {"x": x, "bias": torch.randn(3, dtype=x.dtype), "y": torch.randn(3, 17, dtype=x.dtype)}
        copies = {key: value.clone() for key, value in inputs.items()}
        expected = _execute_region(region, inputs)
        for _ in range(2):
            actual = module.fused(*inputs.values())
            for result, reference in zip(actual, expected.values(), strict=True):
                torch.testing.assert_close(result, reference, equal_nan=True)
        for key in inputs:
            torch.testing.assert_close(inputs[key], copies[key], equal_nan=True)


def test_last_use_reuse_and_saved_alias_protection():
    ctx = TensorRegionBuilder(("m", "k", "n"))
    lhs = ctx.expect("lhs", TensorSignature(("m", "k")))
    rhs = ctx.expect("rhs", TensorSignature(("k", "n")))
    product = ctx.contract(lhs, rhs, over=("k",))
    ctx.yield_value("result", ctx.exp(product * 0.5 + 1.0))
    private = ctx.finish_region()
    source = _emit_region("private_buffer", private, {"m": 3, "k": 4, "n": 5})
    assert "reuse last-use" in source

    ctx.yield_value("saved", ctx.stop_gradient(product))
    saved = ctx.finish_region()
    module = _load({"private_buffer": private, "saved": saved}, {"m": 3, "k": 4, "n": 5})
    a, b = torch.randn(3, 4), torch.randn(4, 5)
    expected = a @ b
    for _ in range(2):
        torch.testing.assert_close(module.private_buffer(a, b)[0], (expected * 0.5 + 1).exp())
        actual, preserved = module.saved(a, b)
        torch.testing.assert_close(actual, (expected * 0.5 + 1).exp())
        torch.testing.assert_close(preserved, expected)


def test_mixed_dtype_fallback_and_borrowed_inputs():
    ctx = TensorRegionBuilder(("b", "r"))
    x = ctx.expect("x", TensorSignature(("b", "r"), "float32"))
    target = ctx.expect("target", TensorSignature(("b",), "int64"))
    value = ctx.where(ctx.compare("eq", ctx.axis_index("r"), target), x, ctx.full(x.axes, 0.0))
    ctx.yield_value("target_value", ctx.reduce("sum", value, ("r",)))
    ctx.yield_value("all", x)
    region = ctx.finish_region()
    module = _load({"mixed": region}, {"b": 3, "r": 17})
    x = torch.randn(3, 17)
    original = x.clone()
    selected, all_values = module.mixed(x, torch.tensor([0, 9, 16]))
    torch.testing.assert_close(selected, x[torch.arange(3), torch.tensor([0, 9, 16])])
    torch.testing.assert_close(all_values, original)
    torch.testing.assert_close(x, original)
    assert "at::where(" in _emit_region("mixed", region, {"b": 3, "r": 17})


def test_pointwise_result_and_row_sum_share_a_pass():
    ctx = TensorRegionBuilder(("b", "r"))
    x = ctx.expect("x", TensorSignature(("b", "r")))
    weights = ctx.exp(x - 1.0)
    ctx.yield_value("weights", weights)
    ctx.yield_value("normalizer", ctx.reduce("sum", weights, ("r",)))
    region = ctx.finish_region()
    source = _emit_region("weights_and_sum", region, {"b": 3, "r": 17})
    assert source.count("at::parallel_for(") == 1
    assert "sum_out[row] = total" in source
    module = _load({"weights_and_sum": region}, {"b": 3, "r": 17})
    x = torch.randn(3, 17)
    actual_weights, normalizer = module.weights_and_sum(x)
    torch.testing.assert_close(actual_weights, (x - 1).exp())
    torch.testing.assert_close(normalizer, (x - 1).exp().sum(-1))


def test_selected_boolean_masks_remain_simd_predicates():
    ctx = TensorRegionBuilder(("b", "r"))
    x = ctx.expect("x", TensorSignature(("b", "r")))
    positive = ctx.compare("gt", x, 0.0)
    valid = ctx.where(positive, ctx.domain_valid(("b", "r")),
                      ctx.full(("b", "r"), False, "bool"))
    ctx.yield_value("positive", ctx.where(valid, x, ctx.full(x.axes, 0.0)))
    ctx.yield_value("negative", ctx.where(ctx.logical_not(valid), x, ctx.full(x.axes, 0.0)))
    always = ctx.where(ctx.domain_valid(x.axes), ctx.full(x.axes, True, "bool"),
                       ctx.full(x.axes, False, "bool"))
    ctx.yield_value("identity", ctx.where(always, x, ctx.full(x.axes, 0.0)))
    region = ctx.finish_region()
    module = _load({"predicates": region}, {"b": 3, "r": 17})
    x = torch.linspace(-3, 3, 51).reshape(3, 17)
    positive, negative, identity = module.predicates(x)
    torch.testing.assert_close(identity, x)
    torch.testing.assert_close(positive, torch.where(x > 0, x, 0.0))
    torch.testing.assert_close(negative, torch.where(x <= 0, x, 0.0))


def test_compiled_parallel_for_uses_requested_workers():
    import sys
    from torch.utils.cpp_extension import load_inline
    from cutilereduce.experimental.backends.cpu.runtime import _cpu_build_flags
    if not sys.platform.startswith("linux") or not torch.backends.openmp.is_available():
        pytest.skip("requires the Linux OpenMP CPU build")
    source = '''
#include <ATen/Parallel.h>
torch::Tensor worker_ids() {
  auto result = at::empty({512}, at::TensorOptions().dtype(at::kLong));
  auto* data = result.data_ptr<int64_t>();
  at::parallel_for(0, 512, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) data[i] = at::get_thread_num();
  });
  return result;
}
'''
    cflags, ldflags = _cpu_build_flags()
    module = load_inline(
        "cutilereduce_thread_probe_" + hashlib.sha256(source.encode()).hexdigest()[:16],
        cpp_sources=source, functions=["worker_ids"], extra_cflags=cflags,
        extra_ldflags=ldflags, with_cuda=False,
    )
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        assert set(module.worker_ids().tolist()) == {0, 1, 2, 3}
        torch.set_num_threads(1)
        assert set(module.worker_ids().tolist()) == {0}
    finally:
        torch.set_num_threads(previous)


def test_online_attention_parallel_workers_preserve_custom_autograd():
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
    from cutilereduce.experimental.planning.analysis import PlanningProblem, ScheduleCandidate
    from cutilereduce.experimental.backends.cpu.tuning import compile_cpu_candidate, cpu_analytical_target
    from cutilereduce.experimental import FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, StructuredValue, TensorSpec
    sizes = {"h": 1, "l": 8, "g": 1, "r": 17, "dqk": 4, "dv": 5}
    spec = FoldSpec({
        "query": TensorSpec(("h", "l", "g", "dqk")),
        "key": TensorSpec(("h", "r", "dqk")),
        "value": TensorSpec(("h", "r", "dv")),
    }, ("h", "l", "g"), ("r",))
    def factor(ctx, inputs):
        return StructuredValue({"log_weight": ctx.contract(inputs["query"], inputs["key"], over=("dqk",)),
                                "value": inputs["value"]})
    op = FoldOperator(spec, LogWeightedSum(("dv",)),
                      FoldMaps(factor, lambda ctx, state: StructuredValue({"value": state["value"]})))
    def reference(q, k, v):
        return torch.bmm(torch.bmm(q.squeeze(2), k.transpose(1, 2)).softmax(-1), v).unsqueeze(2)
    target = CpuTarget("parallel_test", 4, 32, 64)
    problem = PlanningProblem(op.lower(), cpu_analytical_target(target), sizes)
    candidate = ScheduleCandidate("online", tuple(sorted({
        "tile.h": 1, "tile.g": 1, "tile.l": 2, "tile.r": 8,
        "backward_tile.l": 2, "backward_tile.r": 17,
    }.items())), True, "l")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        compiled = compile_cpu_candidate(op, problem, candidate, target=target)
        inputs = (torch.randn(1, 8, 1, 4, requires_grad=True),
                  torch.randn(1, 17, 4, requires_grad=True),
                  torch.randn(1, 17, 5, requires_grad=True))
        cotangent = torch.randn(1, 8, 1, 5)
        for threads in (1, 4):
            torch.set_num_threads(threads)
            with torch.no_grad():
                assert not compiled(*inputs).requires_grad
            actual, expected = compiled(*inputs), reference(*inputs)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            actual_grad = torch.autograd.grad(actual, inputs, cotangent)
            expected_grad = torch.autograd.grad(expected, inputs, cotangent)
            for actual, expected in zip(actual_grad, expected_grad, strict=True):
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    finally:
        torch.set_num_threads(previous)
