import pytest

from cutilereduce.experimental import (
    Add,
    Affine,
    AffineLWS,
    ContractionAxes,
    ContractionPrecision,
    FinalizeContext,
    FoldMaps,
    FoldFactor,
    FoldOperator,
    FoldSpec,
    LogSumExp,
    LogWeightedSum,
    Monoid,
    Product,
    StructuredValue,
    TensorBuilder,
    TensorField,
    TensorSpec,
    TensorStructure,
    compile_fold,
    compile_fold_backward,
    compile_fold_forward,
    compile_map_finalize_vjp_region,
    discover_contractions,
    lower_fold,
    lower_scan,
    optimize_program,
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


class GenericAddMonoid(Monoid[StructuredValue]):
    commutative = True

    def __init__(self):
        self.structure = TensorStructure({"value": TensorField(dtype="float32")})

    def identity(self, ctx):
        return StructuredValue({"value": ctx.constant(0.0)})

    def combine(self, ctx, left, right):
        del ctx
        return StructuredValue({"value": left["value"] + right["value"]})


def test_generic_fold_and_scan_are_derived_from_combine():
    monoid = GenericAddMonoid()
    ctx = TensorBuilder(("b", "f"))
    values = StructuredValue({"value": ctx.input("values", ("b", "f"), "float32")})
    initial = StructuredValue({"value": ctx.input("initial", ("b",), "float32")})

    folded = lower_fold(ctx, monoid, values, fold_axes=("f",))
    scanned = lower_scan(ctx, monoid, values, scan_axis="f", initial=initial)

    assert folded["value"].node.op == "associative_fold"
    assert scanned.prefixes["value"].node.op == "associative_scan_prefix"
    assert scanned.prefixes["value"].axes == ("b", "f")
    assert scanned.final["value"].node.op == "associative_scan_final"
    assert scanned.final["value"].axes == ("b",)
    region = scanned.final["value"].node.attribute("combine_region")
    assert any(node.op == "add" for node in region.nodes)


def test_axis_index_and_factor_mask_validation():
    ctx = TensorBuilder(("b", "f", "d"))
    assert ctx.axis_index("f").axes == ("f",)
    assert ctx.axis_index("f", "int32").dtype == "int32"
    with pytest.raises(ValueError, match="unknown axis"):
        ctx.axis_index("missing")
    with pytest.raises(ValueError, match="int32 or int64"):
        ctx.axis_index("f", "float32")

    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f", "d"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def invalid_map(builder, inputs):
        return FoldFactor(
            StructuredValue({"value": inputs["x"]}),
            where=builder.compare("greater", builder.axis_index("d"), 0),
        )

    with pytest.raises(ValueError, match="batch and fold axes"):
        compile_fold(spec, Add(("d",)), invalid_map)


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


def test_contraction_precision_is_explicit_and_survives_algebra_lowering():
    tensor_core_bf16 = ContractionPrecision(
        operand_dtype="bfloat16",
        accumulator_dtype="float32",
        allow_operand_rounding=True,
    )
    program = compile_fold(
        attention_spec(),
        LogWeightedSum(("dv",), contraction_precision=tensor_core_bf16),
        attention_map,
    )

    weighted_value = program.contractions[-1]
    assert weighted_value.attribute("precision") == tensor_core_bf16
    assert weighted_value.dtype == "float32"

    ctx = TensorBuilder(("m", "n", "k"))
    lhs = ctx.input("lhs", ("m", "k"), "float32")
    rhs = ctx.input("rhs", ("n", "k"), "float32")
    with pytest.raises(ValueError, match="allow_operand_rounding"):
        ctx.contract(
            lhs,
            rhs,
            over=("k",),
            dtype="float32",
            precision=ContractionPrecision(
                operand_dtype="bfloat16", accumulator_dtype="float32"
            ),
        )


def test_fold_operator_builds_the_logical_planning_contract():
    algebra = LogWeightedSum(("dv",))

    def output_function(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    operator = FoldOperator(
        spec=attention_spec(),
        algebra=algebra,
        maps=FoldMaps(
            to_factor=attention_map,
            to_output=output_function,
        ),
    )
    lowered = operator.lower()

    assert lowered.semantic_commutative
    assert lowered.execution_commutative
    assert not lowered.requires_prefix
    assert lowered.embed_output_vjp is not None
    assert lowered.map_finalize_map_vjp is not None
    assert lowered.map_finalize_map_vjp_accumulate is None
    assert "execution.maximum" in dict(lowered.map_fold.outputs)
    assert "execution.normalizer" in dict(lowered.map_fold.outputs)
    assert "execution.numerator" in dict(lowered.map_fold.outputs)
    assert "value" in dict(lowered.project_output.outputs)
    assert {
        "accumulator.log_weight",
        "accumulator.scalar",
        "accumulator.value",
    } <= set(dict(lowered.embed_output_vjp.outputs))
    assert not lowered.map_fold.materializations
    formatted = lowered.map_finalize_map_vjp.format()
    assert "%0 query: tensor[h, l, g, dqk; bfloat16]" in formatted
    assert "%0 = argument" not in formatted
    assert "yield partial_grad_input.query=" in formatted
    assert {name for name, _ in lowered.execution_combine.arguments} == {
        "left.maximum",
        "left.normalizer",
        "left.numerator",
        "right.maximum",
        "right.normalizer",
        "right.numerator",
    }


def test_affine_operator_exposes_prefix_planning_capability():
    algebra = Affine(("d",))
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f")),
            "payload": TensorSpec(("b", "f", "d")),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def map_function(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue({"value": inputs["payload"]}),
        )

    operator = FoldOperator(
        spec=spec,
        algebra=algebra,
        maps=FoldMaps(to_factor=map_function),
    )
    lowered = operator.lower()

    assert not lowered.semantic_commutative
    assert lowered.requires_prefix
    assert not lowered.execution_commutative
    assert lowered.map_finalize_map_vjp is None
    assert lowered.map_finalize_map_vjp_accumulate is not None
    assert "updated_prefix.acting.value" in dict(
        lowered.map_finalize_map_vjp_accumulate.outputs
    )


def test_commutative_semantic_algebra_rejects_prefix_finalize():
    class InvalidLogWeightedSum(LogWeightedSum):
        finalize_context = FinalizeContext.PREFIX

    with pytest.raises(ValueError, match="commutative semantic monoid"):
        FoldOperator(
            spec=attention_spec(),
            algebra=InvalidLogWeightedSum(("dv",)),
            maps=FoldMaps(to_factor=attention_map),
        )


def test_attention_program_never_materializes_broadcast_value_factors():
    program = compile_fold(
        attention_spec(),
        LogWeightedSum(("dv",)),
        attention_map,
    )

    assert all(set(node.axes) != {"h", "l", "g", "r", "dv"} for node in program.nodes)
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
    assert all(
        program.node(node.inputs[0]).op != "materialize"
        for node in program.nodes
        if node.inputs
    )


def test_contraction_rejects_an_axis_absent_from_an_operand():
    ctx = TensorBuilder(("b", "r", "d"))
    lhs = ctx.input("lhs", ("b", "r"), "float32")
    rhs = ctx.input("rhs", ("r", "d"), "float32")

    with pytest.raises(ValueError, match="must occur in both operands"):
        ctx.contract(lhs, rhs, over=("b",))


def test_contraction_recognition_folds_zero_mask_into_a_contraction_operand():
    ctx = TensorBuilder(("b", "l", "r", "d"))
    probability = ctx.input("probability", ("b", "l", "r"), "float32")
    cotangent = ctx.input("cotangent", ("b", "l", "d"), "float32")
    valid = ctx.input("valid", ("b", "l", "r"), "bool")
    product = probability * cotangent
    masked = ctx.where(valid, product, ctx.full(product.axes, 0.0))
    reduced = ctx.reduce("sum", masked, ("l",))
    ctx.materialize("gradient", reduced)

    program = optimize_program(ctx.finish())
    assert len(program.contractions) == 1
    contraction = program.contractions[0]
    assert contraction.attribute("contraction_axes") == ContractionAxes(
        batch=("b",),
        lhs_free=("r",),
        rhs_free=("d",),
        reduce=("l",),
    )
    masked_operand = program.node(contraction.inputs[0])
    assert masked_operand.op == "where"
    assert masked_operand.axes == ("b", "l", "r")
    assert program.node(contraction.inputs[1]).id == cotangent.id


def test_masked_product_without_single_operand_masking_axes_stays_generic():
    ctx = TensorBuilder(("b", "l", "r", "d"))
    lhs = ctx.input("lhs", ("b", "l", "r"), "float32")
    rhs = ctx.input("rhs", ("b", "l", "d"), "float32")
    valid = ctx.input("valid", ("r", "d"), "bool")
    product = lhs * rhs
    reduced = ctx.reduce(
        "sum",
        ctx.where(valid, product, ctx.full(product.axes, 0.0)),
        ("l",),
    )
    ctx.materialize("result", reduced)

    program = discover_contractions(ctx.finish())
    assert not program.contractions
    assert program.output_node("result").inputs


def test_lws_exposes_phi_and_execution_combine_lowerings():
    ctx = TensorBuilder(("b", "d"))
    algebra = LogWeightedSum(("d",))
    identity = algebra.execution.identity(ctx)
    assert identity["maximum"].axes == ()
    assert identity["normalizer"].axes == ()
    assert identity["numerator"].axes == ("d",)

    left = algebra.to_execution(
        ctx,
        StructuredValue(
            {
                "log_weight": ctx.input("left_weight", ("b",), "float32"),
                "value": ctx.input("left_value", ("b", "d"), "float32"),
            }
        ),
    )
    right = algebra.to_execution(
        ctx,
        StructuredValue(
            {
                "log_weight": ctx.input("right_weight", ("b",), "float32"),
                "value": ctx.input("right_value", ("b", "d"), "float32"),
            }
        ),
    )

    combined = algebra.execution.combine(ctx, left, right)

    assert set(combined.fields) == {"maximum", "normalizer", "numerator"}
    assert combined["maximum"].axes == ("b",)
    assert combined["normalizer"].axes == ("b",)
    assert combined["numerator"].axes == ("b", "d")


def test_lws_backward_uses_an_lws_accumulator():
    algebra = LogWeightedSum(("dv",))
    program = compile_map_finalize_vjp_region(attention_spec(), algebra, attention_map)

    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "accumulator.log_weight",
        "accumulator.scalar",
        "accumulator.value",
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
        set(node.axes) == {"h", "l", "g", "r", "dv"} for node in program.nodes
    )
    value_gradient = next(
        node for node in program.contractions if node.axes == ("h", "r", "dv")
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
    assert algebra.semantic.commutative
    assert algebra.execution.commutative
    assert algebra.accumulator.commutative
    assert algebra.finalize_context is FinalizeContext.NONE

    def map_function(ctx, inputs):
        logits = ctx.contract(inputs["context"], inputs["targets"], over=("d",))
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
        logits = ctx.contract(inputs["context"], inputs["targets"], over=("d",))
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
        logits = ctx.contract(inputs["context"], inputs["targets"], over=("d",))
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
        "accumulator.normalizer.log_weight",
        "accumulator.normalizer.value",
        "accumulator.target.value",
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
    assert not algebra.semantic.commutative
    assert not algebra.execution.commutative
    assert algebra.accumulator.commutative
    assert algebra.finalize_context is FinalizeContext.PREFIX

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


def test_affine_accumulator_combines_distinct_prefix_contributions():
    algebra = Affine(("d",))
    ctx = TensorBuilder({"b": 2, "d": 4})

    def affine_value(prefix: str):
        return algebra.value(
            acting=StructuredValue(
                {"value": ctx.input(f"{prefix}.scale", ("b",), "float32")}
            ),
            payload=StructuredValue(
                {"value": ctx.input(f"{prefix}.payload", ("b", "d"), "float32")}
            ),
        )

    left = algebra.embed(ctx, affine_value("p0"), affine_value("g0"))
    right = algebra.embed(ctx, affine_value("p1"), affine_value("g1"))
    combined = algebra.accumulator.combine(ctx, left, right)

    acting = combined["acting"]
    payload = combined["payload"]
    assert isinstance(acting, StructuredValue)
    assert isinstance(payload, StructuredValue)
    assert payload["log_weight"].axes == ("b",)
    assert acting["value"].axes == ("b",)
    assert payload["value"].axes == ("b", "d")


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

    program = compile_map_finalize_vjp_region(spec, algebra, map_function)

    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "prefix.acting.value",
        "prefix.payload.value",
        "accumulator.acting.value",
        "accumulator.payload.log_weight",
        "accumulator.payload.value",
    } <= input_names
    assert set(dict(program.outputs)) == {
        "partial_grad_input.scale",
        "partial_grad_input.payload",
        "updated_prefix.acting.value",
        "updated_prefix.payload.value",
    }
    assert program.output_node("partial_grad_input.scale").axes == ("b", "f")
    assert program.output_node("partial_grad_input.payload").axes == (
        "b",
        "f",
        "d",
    )
    scan_prefixes = tuple(
        node for node in program.nodes if node.op == "associative_scan_prefix"
    )
    assert {node.axes for node in scan_prefixes} == {
        ("b", "f"),
        ("b", "f", "d"),
    }
    assert program.output_node("updated_prefix.acting.value").axes == ("b",)
    assert program.output_node("updated_prefix.payload.value").axes == (
        "b",
        "d",
    )


def test_affine_lws_fold_reweights_then_normalizes_payload():
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f"), "float32"),
            "log_weight": TensorSpec(("b", "f"), "float32"),
            "value": TensorSpec(("f", "d"), "float32"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = AffineLWS(("d",))

    def map_function(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue(
                {
                    "log_weight": inputs["log_weight"],
                    "value": inputs["value"],
                }
            ),
        )

    def output_function(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["payload"]["value"]})

    program = compile_fold(spec, algebra, map_function, output_function=output_function)

    assert len(program.scans) == 1
    assert any(node.op == "reduce_max" for node in program.nodes)
    assert any(node.op == "contract" for node in program.nodes)
    assert set(dict(program.outputs)) == {"value"}
    assert program.output_node("value").axes == ("b", "d")


def test_affine_lws_projected_backward_uses_lws_accumulator():
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f"), "float32"),
            "log_weight": TensorSpec(("b", "f"), "float32"),
            "value": TensorSpec(("f", "d"), "float32"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = AffineLWS(("d",))

    def map_function(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue(
                {
                    "log_weight": inputs["log_weight"],
                    "value": inputs["value"],
                }
            ),
        )

    def output_function(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["payload"]["value"]})

    program = compile_map_finalize_vjp_region(
        spec, algebra, map_function, output_function=output_function
    )
    input_names = {
        node.attribute("name") for node in program.nodes if node.op == "input"
    }
    assert {
        "accumulator.acting.value",
        "accumulator.payload.log_weight",
        "accumulator.payload.weight",
        "accumulator.payload.value",
        "prefix.acting.value",
        "prefix.payload.maximum",
        "prefix.payload.normalizer",
        "prefix.payload.numerator",
    } <= input_names
    assert set(dict(program.outputs)) == {
        "partial_grad_input.scale",
        "partial_grad_input.log_weight",
        "partial_grad_input.value",
        "updated_prefix.acting.value",
        "updated_prefix.payload.maximum",
        "updated_prefix.payload.normalizer",
        "updated_prefix.payload.numerator",
    }
