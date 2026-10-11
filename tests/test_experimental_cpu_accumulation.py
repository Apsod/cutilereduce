"""CPU accumulation respects schedule residency and contraction semantics."""

from dataclasses import replace
import hashlib

import pytest
import torch

from cutilereduce.experimental import (
    Add,
    CpuScheduleConfig,
    CpuStageConfig,
    CpuTarget,
    FoldMaps,
    FoldOperator,
    FoldSpec,
    StructuredValue,
    TensorSpec,
    build_schedule_space,
    compile_cpu,
    emit_cpp_source,
    schedule_cpu,
)
from cutilereduce.experimental.backends.cpu.regions import _emit_region
from cutilereduce.experimental.ir.core import TensorBuilder
from cutilereduce.experimental.ir.passes import program_to_region


@pytest.mark.parametrize("plan_index", [0, 1, 2])
def test_cpu_online_residency_and_partial_storage(plan_index):
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))}, batch_axes=("b",), fold_axes=("f",)
    )
    program = FoldOperator(
        spec,
        Add(),
        FoldMaps(lambda ctx, inputs: StructuredValue({"value": inputs["x"]})),
    ).lower(backward=False)
    sizes = {"b": 5, "f": 9}
    scheduled = schedule_cpu(
        program,
        build_schedule_space(program).forward[plan_index],
        sizes,
        target=CpuTarget("test", 2, 32, 64),
        config=CpuScheduleConfig(chunk_sizes=(("f", 2),), partition_count=2),
    )
    source = emit_cpp_source(
        program,
        sizes,
        backward=False,
        scheduled_forward=scheduled,
        forward_extents=(1, 2, 4, 5, 9),
    )
    if plan_index == 1:
        assert "std::vector<std::vector<torch::Tensor>> values(count)" in source
    else:
        assert "std::vector<std::vector<torch::Tensor>> values(count)" not in source
        assert "state = run_map_fold_combine(start + offset, local, state)" in source
    if plan_index == 2:
        assert (
            "std::vector<std::vector<torch::Tensor>> partials(partition_count)"
            in source
        )
    if plan_index == 0:
        # A nested online fold must also combine children immediately.
        stage = scheduled.forward[0]
        binding = stage.bindings[0]
        inner = binding.expression
        outer = replace(inner, iteration=replace(inner.iteration, grain=5), body=inner)
        nested = replace(
            scheduled,
            forward=(replace(stage, bindings=(replace(binding, expression=outer),)),),
        )
        nested_source = emit_cpp_source(
            program,
            sizes,
            backward=False,
            scheduled_forward=nested,
            forward_extents=(1, 2, 4, 5, 9),
        )
        assert "state = combine_carriers(state, eval_carrier_" in nested_source
        assert (
            "std::vector<std::vector<torch::Tensor>> values(count)" not in nested_source
        )
        from torch.utils.cpp_extension import load_inline

        name = (
            "cutilereduce_nested_online_"
            + hashlib.sha256(nested_source.encode()).hexdigest()[:16]
        )
        module = load_inline(
            name,
            cpp_sources=nested_source,
            functions=None,
            extra_cflags=["-O1"],
            with_cuda=False,
        )
        inputs = torch.arange(45, dtype=torch.float32).reshape(5, 9)
        torch.testing.assert_close(module.scheduled_forward(inputs)[0], inputs.sum(1))


def test_generated_accumulated_contractions_match_matmul_plus_add():
    from torch.utils.cpp_extension import load_inline

    sizes = {"n": 3, "m": 2, "b": 2, "k": 4}
    ctx = TensorBuilder(tuple(sizes))
    left = ctx.input("left", ("m", "k"), "bfloat16")
    right = ctx.input("right", ("k", "n"), "float32")
    bias = ctx.input("bias", ("n",), "float32")
    accumulator = bias + ctx.full(("n", "m"), 0.0)
    result = ctx.contract_accumulate(
        accumulator, left, right, over=("k",), dtype="float32"
    )
    ctx.materialize("matrix", result)
    batched_left = ctx.input("batched_left", ("m", "b", "k"), "float32")
    batched_right = ctx.input("batched_right", ("n", "k", "b"), "float32")
    batched_bias = ctx.input("batched_bias", ("b",), "float32")
    batched_accumulator = batched_bias + ctx.full(("n", "m", "b"), 0.0)
    batched = ctx.contract_accumulate(
        batched_accumulator, batched_left, batched_right, over=("k",)
    )
    ctx.materialize("batched", batched)
    vector_left = ctx.input("vector_left", ("k",), "float32")
    vector_right = ctx.input("vector_right", ("k",), "float32")
    scalar = ctx.input("scalar", (), "float32")
    ctx.materialize(
        "dot", ctx.contract_accumulate(scalar, vector_left, vector_right, over=("k",))
    )
    source = _emit_region(
        "accumulated", program_to_region(ctx.finish()), sizes, nested_names={}
    )
    assert "at::addmm(" in source
    assert "at::baddbmm(" in source
    name = (
        "cutilereduce_accumulation_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    )
    module = load_inline(
        name,
        cpp_sources=source,
        functions=["accumulated"],
        extra_cflags=["-O1"],
        with_cuda=False,
    )
    torch.manual_seed(37)
    inputs = (
        torch.randn(2, 4).bfloat16(),
        torch.randn(4, 3),
        torch.randn(3),
        torch.randn(2, 2, 4),
        torch.randn(3, 4, 2),
        torch.randn(2),
        torch.randn(4),
        torch.randn(4),
        torch.randn(()),
    )
    matrix, batched, dot = module.accumulated(*inputs)
    torch.testing.assert_close(matrix, (inputs[0].float() @ inputs[1] + inputs[2]).T)
    expected = torch.einsum("mbk,nkb->nmb", inputs[3], inputs[4]) + inputs[5]
    torch.testing.assert_close(batched, expected)
    torch.testing.assert_close(dot, inputs[6] @ inputs[7] + inputs[8])


def test_online_identity_lifting_with_inner_axes_before_batch_axes():
    spec = FoldSpec(
        inputs={"x": TensorSpec(("d", "b", "f"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )
    operator = FoldOperator(
        spec,
        Add(("d",)),
        FoldMaps(lambda ctx, inputs: StructuredValue({"value": inputs["x"]})),
    )
    compiled = compile_cpu(
        operator,
        {"d": 3, "b": 2, "f": 5},
        backward=True,
        schedule=CpuScheduleConfig(
            chunk_sizes=(("f", 2),),
            stages=(CpuStageConfig("map_fold_online", tile_sizes=(("b", 1),)),),
        ),
    )
    inputs = torch.arange(30, dtype=torch.float32).reshape(3, 2, 5).requires_grad_()
    result = compiled(inputs)
    result.sum().backward()
    torch.testing.assert_close(result, inputs.detach().sum(2))
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
