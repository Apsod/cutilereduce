import torch

from cutilereduce.experimental import (
    AxisPartition,
    FoldMaps,
    FoldFactor,
    FoldOperator,
    FoldSpec,
    LogWeightedSum,
    LogSumExp,
    Add,
    AffineLWS,
    Product,
    StructuredValue,
    TensorSpec,
    TiledStage,
    execute_tiled_fold_cpu,
    execute_tiled_backward_cpu,
    emit_cutile_backward_source,
    emit_cutile_source,
    discover_tile_mask,
    format_tiled_fold,
    build_schedule_space,
    specialize_cutile_schedule,
    specialize_cutile_backward_schedule,
)


def _specialize_forward(program, sizes, tiles):
    template = build_schedule_space(program).forward[0]
    return specialize_cutile_schedule(program, template, sizes, tiles=tiles)


def test_cpu_mock_executes_single_owner_tiled_attention_fold():
    sizes = {"h": 1, "l": 5, "g": 1, "r": 7, "dqk": 4, "dv": 3}
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk")),
            "key": TensorSpec(("h", "r", "dqk")),
            "value": TensorSpec(("h", "r", "dv")),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        return StructuredValue(
            {
                "log_weight": ctx.contract(
                    inputs["query"], inputs["key"], over=("dqk",)
                ),
                "value": inputs["value"],
            }
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    problem = FoldOperator(
        spec,
        LogWeightedSum(("dv",)),
        FoldMaps(to_factor, to_output),
    ).lower(backward=False)
    specialized = _specialize_forward(problem, sizes, {"l": 2, "r": 3})
    schedule = specialized.implementation.stages[0]
    assert isinstance(schedule, TiledStage)
    assert [dict(op.attributes)["region"] for op in schedule.loop.body] == [
        "map_fold_combine",
    ]
    assert dict(schedule.epilogue[0].attributes)["region"] == "project_output"
    assert [binding.name for binding in schedule.resident_bindings] == ["query"]
    assert [binding.name for binding in schedule.streamed_bindings] == [
        "key",
        "value",
    ]
    inputs = {
        "query": torch.randn(1, 5, 1, 4),
        "key": torch.randn(1, 7, 4),
        "value": torch.randn(1, 7, 3),
    }

    actual = execute_tiled_fold_cpu(problem, schedule, inputs)["value"]
    scores = torch.einsum("hlgd,hrd->hlgr", inputs["query"], inputs["key"])
    expected = torch.einsum("hlgr,hrv->hlgv", scores.softmax(-1), inputs["value"])

    torch.testing.assert_close(actual, expected)
    formatted = format_tiled_fold(schedule)
    assert "parallel owners" in formatted
    assert "serial fold (r[tile=3, programs=1])" in formatted
    assert "@map_fold_combine" in formatted

    source = emit_cutile_source(specialized)
    assert "@ct.kernel" in source
    assert "def single_owner_tiled_fold" in source
    assert source.count("ct.mma(") >= 2
    assert "ct.mma(" in source.split("def map_fold_combine", 1)[1]
    assert "ct.reshape(ct.mma(" in source.split("def map_fold_combine", 1)[1]
    assert "for local_tile in range(loop_size)" in source
    assert "physical_valid =" in source
    assert "physical_valid" not in dict(problem.map_fold_combine.arguments)
    assert any(node.op == "domain_valid" for node in problem.map_fold_combine.nodes)
    domain_conditions = [
        problem.map_fold_combine.nodes[node.inputs[1]]
        for node in problem.map_fold_combine.nodes
        if node.op == "where"
        and problem.map_fold_combine.nodes[node.inputs[0]].op == "domain_valid"
    ]
    assert domain_conditions
    assert all("dv" not in node.axes for node in domain_conditions)
    assert source.count("ct.load(query,") == 1


def test_axis_partition_balances_tiles_and_permits_idle_programs():
    partition = AxisPartition("r", extent=23, tile=4, programs=4)
    assert partition.tile_count == 6
    assert [partition.tile_span(index) for index in range(4)] == [
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 6),
    ]
    assert partition.element_span(3) == (20, 23)

    sparse = AxisPartition("r", extent=5, tile=4, programs=4)
    assert [sparse.tile_span(index) for index in range(4)] == [
        (0, 1),
        (1, 2),
        (2, 2),
        (2, 2),
    ]


def test_factor_mask_uses_logical_indices_and_kills_masked_interactions():
    sizes = {"l": 6, "r": 6, "d": 4}
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("l", "d")),
            "key": TensorSpec(("r", "d")),
            "value": TensorSpec(("r", "d")),
        },
        batch_axes=("l",),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        logits = ctx.contract(inputs["query"], inputs["key"], over=("d",))
        causal = ctx.compare("less_equal", ctx.axis_index("r"), ctx.axis_index("l"))
        return FoldFactor(
            StructuredValue({"log_weight": logits, "value": inputs["value"]}),
            where=causal,
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    problem = FoldOperator(
        spec, LogWeightedSum(("d",)), FoldMaps(to_factor, to_output)
    ).lower(backward=False)
    mask = discover_tile_mask(problem.map_fold)
    assert mask is not None
    assert (mask.operation, mask.left.axis, mask.right.axis) == (
        "less_equal",
        "r",
        "l",
    )
    specialized = _specialize_forward(problem, sizes, {"l": 2, "r": 4})
    stage = specialized.implementation.stages[0]
    inputs = {
        "query": torch.randn(6, 4),
        "key": torch.randn(6, 4),
        "value": torch.randn(6, 4),
    }
    actual = execute_tiled_fold_cpu(problem, stage, inputs)["value"]
    logits = inputs["query"] @ inputs["key"].T
    causal = torch.arange(6)[None, :] <= torch.arange(6)[:, None]
    expected = (
        torch.softmax(logits.masked_fill(~causal, float("-inf")), -1) @ inputs["value"]
    )
    torch.testing.assert_close(actual, expected)

    source = emit_cutile_source(specialized)
    assert "axis_index_l" in source
    assert "axis_index_r" in source
    assert "mask_identity" not in source
    assert "if (loop_tile * 4) <=" in source


def test_all_masked_lws_is_identity_and_has_zero_input_cotangents():
    sizes = {"b": 2, "r": 4, "d": 3}
    spec = FoldSpec(
        inputs={
            "logits": TensorSpec(("b", "r")),
            "value": TensorSpec(("r", "d")),
        },
        batch_axes=("b",),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        return FoldFactor(
            StructuredValue({"log_weight": inputs["logits"], "value": inputs["value"]}),
            where=ctx.compare("equal", ctx.axis_index("r"), -1),
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    problem = FoldOperator(
        spec, LogWeightedSum(("d",)), FoldMaps(to_factor, to_output)
    ).lower()
    specialized = _specialize_forward(problem, sizes, {"b": 2, "r": 2})
    stage = specialized.implementation.stages[0]
    inputs = {
        "logits": torch.randn(2, 4),
        "value": torch.randn(4, 3),
    }
    output = execute_tiled_fold_cpu(problem, stage, inputs)["value"]
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isfinite(output).all()

    backward_template = build_schedule_space(problem).backward[0]
    backward = specialize_cutile_backward_schedule(
        problem, backward_template, sizes, loop_axis="r", tiles={"b": 2, "r": 2}
    ).implementation
    gradients = execute_tiled_backward_cpu(
        problem,
        backward,
        inputs,
        {
            "maximum": torch.full((2,), float("-inf")),
            "normalizer": torch.zeros(2),
            "numerator": torch.zeros(2, 3),
        },
        {"value": torch.randn(2, 3)},
    )
    assert torch.equal(gradients["logits"], torch.zeros_like(inputs["logits"]))
    assert torch.equal(gradients["value"], torch.zeros_like(inputs["value"]))


def test_unmasked_add_identity_value_still_receives_gradient():
    sizes = {"b": 2, "f": 4}
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        del ctx
        return StructuredValue({"value": inputs["x"]})

    problem = FoldOperator(spec, Add(), FoldMaps(to_factor)).lower()
    backward_template = build_schedule_space(problem).backward[0]
    plan = specialize_cutile_backward_schedule(
        problem, backward_template, sizes, loop_axis="f", tiles={"b": 2, "f": 2}
    ).implementation
    inputs = {"x": torch.zeros(2, 4)}
    gradients = execute_tiled_backward_cpu(
        problem,
        plan,
        inputs,
        {"value": torch.zeros(2)},
        {"value": torch.ones(2)},
    )
    assert torch.equal(gradients["x"], torch.ones_like(inputs["x"]))


def test_cutile_backward_emits_embed_local_accumulation_and_relaxed_atomics():
    sizes = {"h": 1, "l": 128, "g": 1, "r": 256, "dqk": 64, "dv": 64}
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk")),
            "key": TensorSpec(("h", "r", "dqk")),
            "value": TensorSpec(("h", "r", "dv")),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        return StructuredValue(
            {
                "log_weight": ctx.contract(
                    inputs["query"], inputs["key"], over=("dqk",)
                ),
                "value": inputs["value"],
            }
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    problem = FoldOperator(
        spec, LogWeightedSum(("dv",)), FoldMaps(to_factor, to_output)
    ).lower()
    backward_region = problem.map_finalize_map_vjp
    assert backward_region is not None
    assert any(
        node.op == "contract" and node.axes == ("h", "r", "dv")
        for node in backward_region.nodes
    )
    backward_template = build_schedule_space(problem).backward[0]
    backward_specialized = specialize_cutile_backward_schedule(
        problem,
        backward_template,
        sizes,
        loop_axis="r",
        loop_programs=4,
        tiles={"l": 32, "r": 32},
    )
    source = emit_cutile_backward_source(backward_specialized)
    plan = backward_specialized.implementation
    finalize = plan.stages[1]
    assert {binding.name for binding in finalize.resident_bindings} == {
        "query",
        "accumulator.log_weight",
        "accumulator.scalar",
        "accumulator.value",
    }
    assert {binding.name for binding in finalize.streamed_bindings} == {
        "key",
        "value",
    }

    assert "def embed_output_vjp_region" in source
    assert "def embed_output_vjp(" in source
    assert "def finalize_map_vjp" in source
    assert source.split("def map_finalize_map_vjp", 1)[1].split(
        "@ct.kernel", 1
    )[0].count("ct.mma(") >= 5
    assert "physical_valid" not in dict(backward_region.arguments)
    assert any(node.op == "domain_valid" for node in backward_region.nodes)
    assert "for local_tile in range(loop_size)" in source
    assert "physical_valid =" in source
    assert "ct.atomic_add(grad_query" in source
    assert "memory_order=ct.MemoryOrder.RELAXED" in source
    assert "ct.atomic_add(grad_key" in source
    assert "ct.atomic_add(grad_value" in source
    assert source.count("ct.load(query,") == 1
    assert source.count("ct.load(accumulator_log_weight,") == 1

    query = torch.randn(1, 128, 1, 64, requires_grad=True)
    key = torch.randn(1, 256, 64, requires_grad=True)
    value = torch.randn(1, 256, 64, requires_grad=True)
    scores = torch.einsum("hlgd,hrd->hlgr", query, key)
    probabilities = scores.softmax(-1)
    output = torch.einsum("hlgr,hrv->hlgv", probabilities, value)
    cotangent = torch.randn_like(output)
    output.backward(cotangent)
    maximum = scores.amax(-1)
    weights = torch.exp(scores - maximum[..., None])
    normalizer = weights.sum(-1)
    numerator = torch.einsum("hlgr,hrv->hlgv", weights, value.detach())
    actual = execute_tiled_backward_cpu(
        problem,
        plan,
        {"query": query.detach(), "key": key.detach(), "value": value.detach()},
        {
            "maximum": maximum.detach(),
            "normalizer": normalizer.detach(),
            "numerator": numerator.detach(),
        },
        {"value": cotangent},
    )
    torch.testing.assert_close(actual["query"], query.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual["key"], key.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual["value"], value.grad, rtol=1e-4, atol=1e-4)


def test_cutile_source_preserves_shared_cross_entropy_contraction():
    sizes = {"b": 16, "v": 32, "d": 8}
    spec = FoldSpec(
        inputs={
            "context": TensorSpec(("b", "d")),
            "targets": TensorSpec(("v", "d")),
            "target": TensorSpec(("b",), "int64"),
        },
        batch_axes=("b",),
        fold_axes=("v",),
    )
    algebra = Product(normalizer=LogSumExp(), target=Add())

    def to_factor(ctx, inputs):
        logits = ctx.contract(inputs["context"], inputs["targets"], over=("d",))
        is_target = ctx.compare("equal", ctx.axis_index("v"), inputs["target"])
        return algebra.value(
            normalizer=StructuredValue({"value": logits}),
            target=StructuredValue(
                {
                    "value": ctx.where(
                        is_target,
                        logits,
                        ctx.full(logits.axes, 0.0),
                    )
                }
            ),
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue(
            {"loss": semantic["normalizer"]["value"] - semantic["target"]["value"]}
        )

    problem = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output)).lower(
        backward=False
    )
    specialized = _specialize_forward(problem, sizes, {"b": 8, "v": 16})
    stage = specialized.implementation.stages[0]
    assert {binding.name for binding in stage.resident_bindings} == {
        "context",
        "target",
    }
    assert {binding.name for binding in stage.streamed_bindings} == {"targets"}
    source = emit_cutile_source(specialized)

    inputs = {
        "context": torch.randn(16, 8),
        "targets": torch.randn(32, 8),
        "target": torch.randint(0, 32, (16,)),
    }
    actual = execute_tiled_fold_cpu(problem, stage, inputs)["loss"]
    logits = inputs["context"] @ inputs["targets"].T
    expected = torch.logsumexp(logits, -1) - logits[torch.arange(16), inputs["target"]]
    torch.testing.assert_close(actual, expected)

    map_fold_source = source.split("def map_fold_combine", 1)[1].split("@ct.function", 1)[0]
    assert map_fold_source.count("ct.mma(") >= 1
    assert "ct.max(" in map_fold_source
    assert "ct.sum(" in map_fold_source
    assert "axis_index_v" in map_fold_source
    assert "is_target" not in source


def test_cutile_source_preserves_affine_lws_suffix_scan():
    sizes = {"b": 4, "r": 16, "d": 8}
    algebra = AffineLWS(("d",))
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "r")),
            "log_weight": TensorSpec(("b", "r")),
            "value": TensorSpec(("b", "r", "d")),
        },
        batch_axes=("b",),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
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

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["payload"]["value"]})

    problem = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output)).lower(
        backward=False
    )
    specialized = _specialize_forward(problem, sizes, {"b": 4, "r": 8})
    source = emit_cutile_source(specialized)
    map_fold_source = source.split("def map_fold_combine", 1)[1].split("@ct.function", 1)[0]

    assert "ct.cumsum(" in map_fold_source
    assert "reverse=True" in map_fold_source
    assert "ct.mma(" in map_fold_source
