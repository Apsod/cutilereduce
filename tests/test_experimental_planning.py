import pytest

from cutilereduce.experimental import (
    Add,
    AtomicAdd,
    Affine,
    ArtifactKind,
    CommitMode,
    CpuLoopKind,
    CpuScheduleConfig,
    CpuStageConfig,
    CpuTarget,
    FoldMaps,
    FoldOperator,
    FoldSpec,
    LocalLifetime,
    LogWeightedSum,
    Parallel,
    StructuredValue,
    TensorSpec,
    SerialLoop,
    Store,
    TiledStage,
    build_schedule_space,
    emit_cpp_source,
    estimate_cpu_plan,
    format_plan,
    format_cpu_schedule,
    schedule_cpu,
    schedule_backward_program,
    specialize_cutile_backward_schedule,
    lower_backward_stage,
    select_cpu_plan,
    format_scheduled_program,
    TargetModel,
)


def _add_problem():
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        del ctx
        return StructuredValue({"value": inputs["x"]})

    return FoldOperator(spec, Add(), FoldMaps(to_factor)).lower()


def _affine_problem():
    algebra = Affine(("d",))
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f")),
            "payload": TensorSpec(("b", "f", "d")),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        del ctx
        return algebra.value(
            acting=StructuredValue({"value": inputs["scale"]}),
            payload=StructuredValue({"value": inputs["payload"]}),
        )

    return FoldOperator(spec, algebra, FoldMaps(to_factor)).lower()


def _attention_problem():
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

    return FoldOperator(
        spec,
        LogWeightedSum(("dv",)),
        FoldMaps(to_factor, to_output),
    ).lower()


def test_commutative_plans_have_independent_backward():
    problem = _add_problem()
    space = build_schedule_space(problem)
    forward = space.forward
    backward = space.backward

    assert [plan.name for plan in forward] == [
        "online",
        "fused_tree",
        "partial_fold",
    ]
    assert len(backward) == 1
    assert backward[0].requirements.prefix_iteration is None
    for plan in forward:
        training = space.compose(plan, backward[0])
        assert training.persistence.artifacts == (ArtifactKind.GLOBAL_EXECUTION,)


def test_prefix_backward_only_composes_with_checkpoint_plan():
    problem = _affine_problem()
    space = build_schedule_space(problem)
    forward = space.forward
    backward = space.backward[0]

    assert backward.requirements.prefix_iteration is not None
    checkpointed = forward[-1]
    training = space.compose(checkpointed, backward)
    assert set(training.persistence.artifacts) == {
        ArtifactKind.GLOBAL_EXECUTION,
        ArtifactKind.PREFIX_EXECUTION,
    }
    with pytest.raises(ValueError, match="prefix_execution"):
        space.compose(forward[0], backward)


def test_plan_formatter_shows_recursive_online_fold():
    formatted = format_plan(build_schedule_space(_add_problem()).forward[0])

    assert "online_fold chunk in chunks(f, f_chunk)" in formatted
    assert "call map_fold(current_region)" in formatted
    assert "persist[global_execution]" in formatted


def test_cpu_schedule_binds_nested_loops_without_nested_parallelism():
    problem = _add_problem()
    partial = build_schedule_space(problem).forward[2]
    target = CpuTarget("test", 4, 32, 64)
    scheduled = schedule_cpu(
        problem,
        partial,
        {"b": 3, "f": 130},
        target=target,
        config=CpuScheduleConfig(chunk_sizes=(("f", 16),), partition_count=4),
    )

    outer = scheduled.forward[0].bindings[0].expression
    assert outer.iteration.loop is CpuLoopKind.PARALLEL
    assert outer.iteration.count == 4
    inner = outer.body
    assert inner.iteration.loop is CpuLoopKind.SERIAL
    assert inner.iteration.grain == 16
    formatted = format_cpu_schedule(scheduled)
    assert "parallel map partition" in formatted
    assert "serial carrier_fold chunk" in formatted


def test_cpu_schedule_exposes_per_stage_tiles_and_scratch_budget():
    problem = _add_problem()
    plan = build_schedule_space(problem).forward[0]
    scheduled = schedule_cpu(
        problem,
        plan,
        {"b": 3, "f": 130},
        target=CpuTarget("test", 4, 32, 64, l2_bytes=512 * 1024),
        config=CpuScheduleConfig(
            scratch_bytes_per_worker=192 * 1024,
            stages=(
                CpuStageConfig(
                    "map_fold_online",
                    tile_sizes=(("b", 2), ("f", 32)),
                    scratch_bytes=96 * 1024,
                ),
            ),
        ),
    )

    stage = scheduled.forward[0]
    assert dict(stage.tile_sizes) == {"b": 2, "f": 32}
    assert stage.scratch_bytes == 96 * 1024
    assert "tiles[b=2, f=32] scratch<=98304 bytes" in format_cpu_schedule(scheduled)

    with pytest.raises(ValueError, match="unknown stages"):
        schedule_cpu(
            problem,
            plan,
            {"b": 3, "f": 130},
            target=CpuTarget("test", 4, 32, 64),
            config=CpuScheduleConfig(stages=(CpuStageConfig("typo"),)),
        )


def test_cpu_schedule_rejects_incomplete_or_nonpositive_sizes():
    problem = _add_problem()
    plan = build_schedule_space(problem).forward[0]
    target = CpuTarget("test", 1, 16, 64)

    with pytest.raises(ValueError, match="exactly"):
        schedule_cpu(problem, plan, {"f": 4}, target=target)
    with pytest.raises(ValueError, match="positive"):
        schedule_cpu(problem, plan, {"b": 1, "f": 0}, target=target)


def test_scheduled_attention_backward_separates_lifetime_and_contention():
    program = schedule_backward_program(
        _attention_problem(),
        {"h": 1, "l": 128, "g": 1, "r": 256, "dqk": 64, "dv": 64},
        loop_axis="l",
        tiles={"l": 32, "r": 64},
    )
    gradients = {gradient.input_name: gradient for gradient in program.gradients}
    stage = lower_backward_stage(program)
    assert isinstance(stage, TiledStage)
    assert stage.parallel_axes == program.program_axes
    assert dict(stage.loop.body[0].attributes)["region"] == program.region

    query = gradients["query"]
    assert query.lifetime is LocalLifetime.STREAMED
    assert query.contention_axes == ("r",)
    assert query.commit_mode is CommitMode.ATOMIC_ADD_RELAXED

    for name in ("key", "value"):
        gradient = gradients[name]
        assert gradient.lifetime is LocalLifetime.PERSISTENT
        assert gradient.contention_axes == ()
        assert gradient.commit_mode is CommitMode.STORE

    formatted = format_scheduled_program(program)
    assert "loop l[tile=32, count=4]" in formatted
    assert "atomic_add relaxed %partial_grad_input.query" in formatted
    assert "accumulate %key, %partial_grad_input.key" in formatted
    assert "store %local.key, @grad_input.key" in formatted

    assert isinstance(program.body, Parallel)
    loop = next(op for op in program.body.body if isinstance(op, SerialLoop))
    assert any(isinstance(op, AtomicAdd) and op.gradient == "query" for op in loop.body)
    assert any(
        isinstance(op, Store) and op.gradient == "key" for op in program.body.body
    )


def test_scheduled_backward_marks_grouped_key_writes_contended():
    program = schedule_backward_program(
        _attention_problem(),
        {"h": 1, "l": 128, "g": 4, "r": 256, "dqk": 64, "dv": 64},
        loop_axis="l",
        tiles={"l": 32, "g": 1, "r": 64},
    )
    gradients = {gradient.input_name: gradient for gradient in program.gradients}

    assert gradients["query"].contention_axes == ("r",)
    assert gradients["key"].contention_axes == ("g",)
    assert gradients["value"].contention_axes == ("g",)
    assert gradients["key"].lifetime is LocalLifetime.PERSISTENT
    assert gradients["key"].commit_mode is CommitMode.ATOMIC_ADD_RELAXED


def test_backward_partitions_and_loops_over_same_logical_axis():
    program = schedule_backward_program(
        _attention_problem(),
        {"h": 1, "l": 128, "g": 1, "r": 256, "dqk": 64, "dv": 64},
        loop_axis="r",
        loop_programs=4,
        tiles={"l": 32, "r": 32},
    )
    stage = lower_backward_stage(program)
    partition = next(value for value in stage.effective_partitions if value.axis == "r")
    assert partition.programs == 4
    assert partition.tile_span(0) == (0, 2)
    assert stage.loop.axis.axis == "r"

    gradients = {gradient.input_name: gradient for gradient in program.gradients}
    assert gradients["query"].lifetime is LocalLifetime.PERSISTENT
    assert gradients["query"].contention_axes == ("r",)
    assert gradients["query"].commit_mode is CommitMode.ATOMIC_ADD_RELAXED
    assert gradients["key"].lifetime is LocalLifetime.STREAMED
    assert gradients["key"].contention_axes == ("l",)
    assert gradients["key"].commit_mode is CommitMode.ATOMIC_ADD_RELAXED

    attention = _attention_problem()
    plan = specialize_cutile_backward_schedule(
        attention,
        build_schedule_space(attention).backward[0],
        {"h": 1, "l": 128, "g": 1, "r": 256, "dqk": 64, "dv": 64},
        loop_axis="r",
        loop_programs=4,
        tiles={"l": 32, "r": 32},
    ).implementation
    assert [stage.name for stage in plan.stages] == [
        "embed_output_vjp",
        "finalize_map_vjp",
    ]
    assert plan.stages[0].loop is None
    assert plan.dependencies[0].buffer == "accumulator"


def test_cpu_source_emits_named_regions_and_named_axis_reduction():
    problem = _add_problem()
    source = emit_cpp_source(problem, {"b": 3, "f": 7}, backward=True)

    assert "std::vector<torch::Tensor> map_fold" in source
    assert "at::sum(" in source
    assert "std::vector<torch::Tensor> project_output" in source
    assert 'm.def("map_fold", &map_fold,' in source


def test_cpu_source_lowers_named_axis_contractions_to_bmm():
    spec = FoldSpec(
        inputs={
            "lhs": TensorSpec(("b", "f", "k"), dtype="bfloat16"),
            "rhs": TensorSpec(("b", "f", "k"), dtype="bfloat16"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        return StructuredValue(
            {
                "value": ctx.contract(
                    ctx.cast(inputs["lhs"], "float32"),
                    ctx.cast(inputs["rhs"], "float32"),
                    over=("k",),
                )
            }
        )

    problem = FoldOperator(spec, Add(), FoldMaps(to_factor)).lower()
    source = emit_cpp_source(
        problem, {"b": 2, "f": 7, "k": 8}, backward=False
    )

    assert "at::bmm(" in source
    assert "at::einsum" not in source
    assert ".to(at::kFloat)" in source


def test_cpu_cost_accounts_for_partial_materialization_and_parallelism():
    problem = _add_problem()
    target = CpuTarget("test", 4, 32, 64)
    config = CpuScheduleConfig(chunk_sizes=(("f", 16),), partition_count=4)
    schedules = build_schedule_space(problem).forward
    online, _, partial = schedules
    online_cost = estimate_cpu_plan(
        problem,
        schedule_cpu(problem, online, {"b": 3, "f": 128}, target=target, config=config),
    )
    partial_cost = estimate_cpu_plan(
        problem,
        schedule_cpu(
            problem, partial, {"b": 3, "f": 128}, target=target, config=config
        ),
    )

    assert partial_cost.parallel_tasks > online_cost.parallel_tasks
    assert partial_cost.materialized_bytes > online_cost.materialized_bytes
    assert online_cost.operations > 0

    choice = select_cpu_plan(
        problem,
        schedules,
        {"b": 3, "f": 128},
        target=target,
    )
    assert choice.plan in schedules
    assert choice.cost.operations > 0


def test_schedule_space_separates_shape_domains_and_target_selection():
    problem = _add_problem()
    space = build_schedule_space(problem)
    shape = {"b": 8, "f": 128}
    online = next(item for item in space.forward if item.name == "online")
    domains = space.analyze_shape(online, shape)
    assert domains["f_chunk"] == range(1, 129)

    target = TargetModel(
        "test",
        frozenset({"tensor_regions"}),
        cost_hook=lambda _program, schedule, _shape: {
            "online": 5.0,
            "fused_tree": 4.0,
            "partial_fold": 1.0,
        }[schedule.name],
    )
    assert space.select_forward(shape, target).name == "partial_fold"
    assert target.tile_domain("f", shape) == range(1, 129)
    assert target.bind_tiles(shape, ("b", "f"), {"f": 32}) == {
        "b": 8,
        "f": 32,
    }
    with pytest.raises(ValueError, match="legal domain"):
        target.bind_tiles(shape, ("f",), {"f": 129})

    restricted = TargetModel(
        "online-only",
        frozenset({"tensor_regions"}),
        supported_schedules=frozenset({"online"}),
    )
    assert space.select_forward(shape, restricted) == online
