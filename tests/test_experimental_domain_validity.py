"""Physical padding must be removed before map VJPs reduce shared inputs."""

import pytest
import torch

from cutilereduce.experimental import Add, FoldMaps, FoldOperator, FoldSpec, LogSumExp, LogWeightedSum, Product, StructuredValue, TensorSpec
from cutilereduce.experimental.backends.reference import _execute_region
from cutilereduce.experimental.compiler.fold import compile_masked_map_fold_region
from cutilereduce.experimental.ir.passes import program_to_region


def run_padded_backward(operator, inputs, valid):
    program = operator.lower()
    sizes = {
        axis: inputs[name].shape[index]
        for name, tensor in operator.spec.inputs.items()
        for index, axis in enumerate(tensor.axes)
    }
    masked = program_to_region(
        compile_masked_map_fold_region(
            operator.spec, operator.algebra, operator.maps.to_factor
        )
    )
    execution = _execute_region(masked, inputs, domain_valid=valid, axis_sizes=sizes)
    output = _execute_region(program.project_output, execution)
    assert program.embed_output_vjp is not None
    accumulator = _execute_region(
        program.embed_output_vjp,
        {
            **{
                name.replace("execution.", "saved_execution.", 1): value
                for name, value in execution.items()
            },
            **{
                f"grad_output.{name}": torch.ones_like(value)
                for name, value in output.items()
            },
        },
    )
    assert program.map_finalize_map_vjp is not None
    gradients = _execute_region(
        program.map_finalize_map_vjp,
        {**inputs, **accumulator},
        domain_valid=valid,
    )
    return program, output, gradients


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("product", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_padded_add_reduces_only_valid_factor_gradients(scalar, product, empty):
    spec = FoldSpec(
        inputs={
            "x": TensorSpec(() if scalar else ("b",)),
            "offset": TensorSpec(("b", "f"), requires_grad=False),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = Product(first=Add(), second=Add()) if product else Add()

    def to_factor(ctx, inputs):
        del ctx
        value = inputs["x"] + inputs["offset"]
        if product:
            return algebra.value(
                first=StructuredValue({"value": value}),
                second=StructuredValue({"value": value * 2}),
            )
        return StructuredValue({"value": value})

    def to_output(ctx, semantic):
        del ctx
        value = (
            semantic["first"]["value"] + semantic["second"]["value"]
            if product
            else semantic["value"]
        )
        return StructuredValue({"value": value})

    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
    if empty:
        valid.zero_()
    inputs = {
        "x": torch.zeros(()) if scalar else torch.zeros(2),
        "offset": torch.zeros(2, 4),
    }
    _, _, gradients = run_padded_backward(
        FoldOperator(spec, algebra, FoldMaps(to_factor, to_output)), inputs, valid
    )
    expected = valid.sum() if scalar else valid.sum(dim=1)
    expected = expected.float() * (3 if product else 1)
    torch.testing.assert_close(gradients["partial_grad_input.x"], expected)


@pytest.mark.parametrize("attention", [True, False])
def test_padded_contraction_backward_matches_unpadded_reference(attention):
    torch.manual_seed(23)
    inputs = {
        "context": torch.randn(3, 3),
        "targets": torch.randn(4, 3),
    }
    input_specs = {"context": TensorSpec(("b", "d")), "targets": TensorSpec(("f", "d"))}
    if attention:
        inputs["value"] = torch.randn(4, 2)
        input_specs["value"] = TensorSpec(("f", "v"))
        algebra = LogWeightedSum(("v",))
    else:
        inputs["is_target"] = torch.tensor(
            [[True, False, False, False], [False, True, False, False], [False] * 4]
        )
        input_specs["is_target"] = TensorSpec(("b", "f"), "bool")
        algebra = Product(normalizer=LogSumExp(), target=Add())
    spec = FoldSpec(inputs=input_specs, batch_axes=("b",), fold_axes=("f",))

    def to_factor(ctx, values):
        logits = ctx.contract(values["context"], values["targets"], over=("d",))
        if attention:
            return StructuredValue({"log_weight": logits, "value": values["value"]})
        return algebra.value(
            normalizer=StructuredValue({"value": logits}),
            target=StructuredValue(
                {"value": ctx.where(values["is_target"], logits, ctx.constant(0.0))}
            ),
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue(
            {
                "value": semantic["value"]
                if attention
                else semantic["normalizer"]["value"] - semantic["target"]["value"]
            }
        )

    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False], [False] * 4]
    )
    program, output, gradients = run_padded_backward(
        FoldOperator(spec, algebra, FoldMaps(to_factor, to_output)), inputs, valid
    )
    reference_inputs = {
        name: value.detach().clone().requires_grad_(value.dtype != torch.bool)
        for name, value in inputs.items()
    }
    references = []
    for batch, row in enumerate(valid[:2]):
        logits = reference_inputs["context"][batch] @ reference_inputs["targets"][row].T
        if attention:
            references.append(logits.softmax(dim=0) @ reference_inputs["value"][row])
        else:
            references.append(
                logits.logsumexp(dim=0) - logits[inputs["is_target"][batch, row]].sum()
            )
    reference = torch.stack(references)
    reference.sum().backward()
    torch.testing.assert_close(output["value"][:2], reference)
    if attention:
        torch.testing.assert_close(output["value"][2], torch.zeros(2))
    for name, value in reference_inputs.items():
        if value.requires_grad:
            torch.testing.assert_close(
                gradients[f"partial_grad_input.{name}"], value.grad
            )
    for name, tensor in spec.inputs.items():
        if tensor.differentiable:
            assert any(
                node.op == "contract" and set(node.axes) == set(tensor.axes)
                for node in program.map_finalize_map_vjp.nodes
            ), f"gradient for {name} lost its contraction lowering"


def test_explicit_domain_requires_boolean_matching_shape():
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))}, batch_axes=("b",), fold_axes=("f",)
    )
    region = program_to_region(
        compile_masked_map_fold_region(
            spec, Add(), lambda ctx, inputs: StructuredValue({"value": inputs["x"]})
        )
    )
    for invalid in (torch.ones(2, 4), torch.ones(2, 3, dtype=torch.bool)):
        with pytest.raises(ValueError, match="domain validity"):
            _execute_region(region, {"x": torch.zeros(2, 4)}, domain_valid=invalid)
