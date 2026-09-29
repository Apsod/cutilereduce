import pytest

from cutilereduce.experimental import (
    Add,
    Affine,
    ContractionAxes,
    FoldSpec,
    LogSumExp,
    LogWeightedSum,
    Product,
    StructuredValue,
    TensorBuilder,
    TensorSpec,
    compile_fold,
    compile_fold_backward,
    compile_fold_forward,
    compile_map_finalize_vjp_region,
)


def attention_spec():
    return FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk"), "bfloat16"),
            "key": TensorSpec(("h", "r", "dqk"), "bfloat16"),
            "value": TensorSpec(("h", "r", "dv"), "bfloat16"),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )


def attention_map(ctx, inputs):
    scores = ctx.contract(
        inputs["query"], inputs["key"], over=("dqk",), dtype="float32"
    )
    return StructuredValue(
        {"log_weight": scores, "value": ctx.cast(inputs["value"], "float32")}
    )


def test_attention_fold_lowers_weighted_values_to_one_batched_gemm():
    program = compile_fold(
        attention_spec(),
        LogWeightedSum(("dv",)),
        attention_map,
    )

    assert len(program.contractions) == 2
    score, weighted_value = program.contractions
    assert score.attribute("contraction_axes") == ContractionAxes(
        batch=("h",),
        lhs_free=("l", "g"),
        rhs_free=("r",),
        reduce=("dqk",),
    )
    assert weighted_value.axes == ("h", "l", "g", "dv")
    assert weighted_value.attribute("contraction_axes") == ContractionAxes(
        batch=("h",),
        lhs_free=("l", "g"),
        rhs_free=("dv",),
        reduce=("r",),
    )
    assert weighted_value.attribute("implementation") == "batched_gemm"
    assert weighted_value.dtype == "float32"


def test_attention_program_never_materializes_broadcast_value_factors():
    program = compile_fold(
        attention_spec(),
        LogWeightedSum(("dv",)),
        attention_map,
    )

    assert all(
        set(node.axes) != {"h", "l", "g", "r", "dv"}
        for node in program.nodes
    )
    assert {
        node.attribute("name")
        for node in program.materializations
        if node.op == "materialize"
    } == {"log_weight", "value"}
    assert program.output_node("value").axes == ("h", "l", "g", "dv")


def test_reductions_and_materialization_boundaries_are_explicit():
    program = compile_fold(
        attention_spec(),
        LogWeightedSum(("dv",)),
        attention_map,
    )

    assert [(node.op, node.attribute("over")) for node in program.reductions] == [
        ("reduce_max", ("r",)),
        ("reduce_sum", ("r",)),
    ]
    assert all(program.node(node.inputs[0]).op != "materialize" for node in program.nodes if node.inputs)


def test_contraction_rejects_an_axis_absent_from_an_operand():
    ctx = TensorBuilder(("b", "r", "d"))
    lhs = ctx.input("lhs", ("b", "r"), "float32")
    rhs = ctx.input("rhs", ("r", "d"), "float32")

    with pytest.raises(ValueError, match="must occur in both operands"):
        ctx.contract(lhs, rhs, over=("b",))


def test_lws_exposes_phi_and_execution_combine_lowerings():
    ctx = TensorBuilder(("b", "d"))
    algebra = LogWeightedSum(("d",))
    identity = algebra.execution.lower_identity(ctx)
    assert identity["maximum"].axes == ()
    assert identity["normalizer"].axes == ()
    assert identity["numerator"].axes == ("d",)

    left = algebra.lower_phi(
        ctx,
        StructuredValue(
            {
                "log_weight": ctx.input("left_weight", ("b",), "float32"),
                "value": ctx.input("left_value", ("b", "d"), "float32"),
            }
        ),
    )
    right = algebra.lower_phi(
        ctx,
        StructuredValue(
            {
                "log_weight": ctx.input("right_weight", ("b",), "float32"),
                "value": ctx.input("right_value", ("b", "d"), "float32"),
            }
        ),
    )

    combined = algebra.execution.lower_combine(ctx, left, right)

    assert set(combined.fields) == {"maximum", "normalizer", "numerator"}
    assert combined["maximum"].axes == ("b",)
    assert combined["normalizer"].axes == ("b",)
    assert combined["numerator"].axes == ("b", "d")


def test_lws_backward_uses_an_lws_adjoint():
    algebra = LogWeightedSum(("dv",))
    program = compile_map_finalize_vjp_region(
        attention_spec(), algebra, attention_map
    )

    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "adjoint.log_weight",
        "adjoint.scalar",
        "adjoint.value",
    } <= input_names
    assert set(dict(program.outputs)) == {
        "partial_grad_input.query",
        "partial_grad_input.key",
        "partial_grad_input.value",
    }
    assert program.output_node("partial_grad_input.query").axes == (
        "h",
        "l",
        "g",
        "dqk",
    )
    assert program.output_node("partial_grad_input.value").axes == (
        "h",
        "r",
        "dv",
    )
    assert not any(
        set(node.axes) == {"h", "l", "g", "r", "dv"}
        for node in program.nodes
    )
    value_gradient = next(
        node
        for node in program.contractions
        if node.axes == ("h", "r", "dv")
    )
    assert value_gradient.attribute("contraction_axes") == ContractionAxes(
        batch=("h",),
        lhs_free=("r",),
        rhs_free=("dv",),
        reduce=("l", "g"),
    )
    assert value_gradient.attribute("implementation") == "batched_gemm"


def test_lws_output_map_projects_payload_and_derives_semantic_cotangent():
    algebra = LogWeightedSum(("dv",))

    def output_function(ctx, semantic):
        del ctx
        return StructuredValue({"output": semantic["value"]})

    forward = compile_fold_forward(
        attention_spec(),
        algebra,
        attention_map,
        output_function=output_function,
    )
    assert set(dict(forward.outputs)) == {
        "output",
        "saved_execution.maximum",
        "saved_execution.normalizer",
        "saved_execution.numerator",
    }

    backward = compile_fold_backward(
        attention_spec(),
        algebra,
        attention_map,
        output_function=output_function,
    )
    input_names = {
        node.attribute("name") for node in backward.nodes if node.op == "input"
    }
    assert "grad_output.output" in input_names
    assert "grad_output.log_weight" not in input_names
    assert set(dict(backward.outputs)) == {
        "grad_input.query",
        "grad_input.key",
        "grad_input.value",
    }


def test_cross_entropy_product_lowers_lse_and_add_from_one_map_graph():
    spec = FoldSpec(
        inputs={
            "context": TensorSpec(("b", "d"), "float32"),
            "targets": TensorSpec(("v", "d"), "float32"),
            "is_target": TensorSpec(("b", "v"), "bool"),
        },
        batch_axes=("b",),
        fold_axes=("v",),
    )
    algebra = Product(
        normalizer=LogSumExp(),
        target=Add(),
    )

    def map_function(ctx, inputs):
        logits = ctx.contract(
            inputs["context"], inputs["targets"], over=("d",)
        )
        target_logits = ctx.where(
            inputs["is_target"],
            logits,
            ctx.full(logits.axes, 0.0),
        )
        return algebra.value(
            normalizer=StructuredValue({"value": logits}),
            target=StructuredValue({"value": target_logits}),
        )

    program = compile_fold(spec, algebra, map_function)

    assert len(program.contractions) == 1
    assert [node.op for node in program.reductions] == [
        "reduce_max",
        "reduce_sum",
        "reduce_sum",
    ]
    assert dict(program.outputs).keys() == {
        "normalizer.value",
        "target.value",
    }


def test_cross_entropy_backward_composes_finalize_with_map_vjp():
    spec = FoldSpec(
        inputs={
            "context": TensorSpec(("b", "d"), "float32"),
            "targets": TensorSpec(("v", "d"), "float32"),
            "is_target": TensorSpec(("b", "v"), "bool"),
        },
        batch_axes=("b",),
        fold_axes=("v",),
    )
    algebra = Product(normalizer=LogSumExp(), target=Add())

    def map_function(ctx, inputs):
        logits = ctx.contract(
            inputs["context"], inputs["targets"], over=("d",)
        )
        target_logits = ctx.where(
            inputs["is_target"],
            logits,
            ctx.full(logits.axes, 0.0),
        )
        return algebra.value(
            normalizer=StructuredValue({"value": logits}),
            target=StructuredValue({"value": target_logits}),
        )

    forward = compile_fold_forward(spec, algebra, map_function)
    assert set(dict(forward.outputs)) == {
        "normalizer.value",
        "target.value",
        "saved_execution.normalizer.maximum",
        "saved_execution.normalizer.normalizer",
        "saved_execution.target.value",
    }

    program = compile_fold_backward(spec, algebra, map_function)

    assert [node.attribute("implementation") for node in program.contractions] == [
        "gemm",
        "gemm",
        "gemm",
    ]
    assert set(dict(program.outputs)) == {
        "grad_input.context",
        "grad_input.targets",
    }
    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "saved_execution.normalizer.maximum",
        "saved_execution.normalizer.normalizer",
        "grad_output.normalizer.value",
        "grad_output.target.value",
    } <= input_names
    assert "saved_execution.target.value" not in input_names
    assert program.output_node("grad_input.context").axes == ("b", "d")
    assert program.output_node("grad_input.targets").axes == ("v", "d")


def test_cross_entropy_map_finalize_vjp_is_an_independent_region():
    spec = FoldSpec(
        inputs={
            "context": TensorSpec(("b", "d"), "float32"),
            "targets": TensorSpec(("v", "d"), "float32"),
            "is_target": TensorSpec(("b", "v"), "bool"),
        },
        batch_axes=("b",),
        fold_axes=("v",),
    )
    algebra = Product(normalizer=LogSumExp(), target=Add())

    def map_function(ctx, inputs):
        logits = ctx.contract(
            inputs["context"], inputs["targets"], over=("d",)
        )
        return algebra.value(
            normalizer=StructuredValue({"value": logits}),
            target=StructuredValue(
                {
                    "value": ctx.where(
                        inputs["is_target"],
                        logits,
                        ctx.full(logits.axes, 0.0),
                    )
                }
            ),
        )

    program = compile_map_finalize_vjp_region(spec, algebra, map_function)

    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "adjoint.normalizer.log_weight",
        "adjoint.normalizer.value",
        "adjoint.target.value",
    } <= input_names
    assert not any(name.startswith("saved_execution") for name in input_names)
    assert set(dict(program.outputs)) == {
        "partial_grad_input.context",
        "partial_grad_input.targets",
    }
    # The context is shared across the local v-region, so its partial
    # cotangent sums that axis.  Per-v target cotangents retain it.
    assert program.output_node("partial_grad_input.context").axes == ("b", "d")
    assert program.output_node("partial_grad_input.targets").axes == ("v", "d")


def test_affine_fold_is_an_ordered_semidirect_scan():
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f"), "float32"),
            "payload": TensorSpec(("b", "f", "d"), "float32"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = Affine(("d",))

    def map_function(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue({"value": inputs["payload"]}),
        )

    program = compile_fold(spec, algebra, map_function)

    assert len(program.scans) == 1
    scan = program.scans[0]
    assert scan.op == "scan_sum"
    assert scan.attribute("over") == "f"
    assert scan.attribute("exclusive") is True
    assert scan.attribute("reverse") is True
    assert set(dict(program.outputs)) == {
        "acting.value",
        "payload.value",
    }


def test_affine_map_finalize_vjp_uses_explicit_prefix():
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f"), "float32"),
            "payload": TensorSpec(("b", "f", "d"), "float32"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = Affine(("d",))

    def map_function(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue({"value": inputs["payload"]}),
        )

    program = compile_map_finalize_vjp_region(
        spec, algebra, map_function, with_prefix=True
    )

    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "prefix.acting.value",
        "prefix.payload.value",
        "adjoint.acting.total",
        "adjoint.acting.cotangent",
        "adjoint.payload.cotangent",
    } <= input_names
    assert "adjoint.acting.count" not in input_names
    assert set(dict(program.outputs)) == {
        "partial_grad_input.scale",
        "partial_grad_input.payload",
    }
    assert program.output_node("partial_grad_input.scale").axes == ("b", "f")
    assert program.output_node("partial_grad_input.payload").axes == (
        "b",
        "f",
        "d",
    )
