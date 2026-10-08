import torch

from cutilereduce.experimental import (
    Add,
    Affine,
    AffineLWS,
    CommitMode,
    CpuScheduleConfig,
    CpuStageConfig,
    FoldMaps,
    FoldOperator,
    FoldSpec,
    LogSumExp,
    LogWeightedSum,
    Product,
    StructuredValue,
    TensorSpec,
    compile_cpu,
    benchmark_cpu,
    benchmark_cpu_training,
    emit_cpp_source,
    build_schedule_space,
    schedule_backward_program,
    scheduled_memory_metrics,
    scheduled_to_ssa,
    liveness,
)


def test_generated_cpu_add_forward_and_backward():
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        del ctx
        return StructuredValue({"value": inputs["x"]})

    compiled = compile_cpu(
        FoldOperator(spec, Add(), FoldMaps(to_factor)),
        {"b": 5, "f": 5},
        backward=True,
        schedule=CpuScheduleConfig(
            chunk_sizes=(("f", 2),),
            stages=(
                CpuStageConfig(
                    "map_fold_online", tile_sizes=(("b", 2), ("f", 5))
                ),
            )
        ),
    )
    value = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    value.requires_grad_()
    output = compiled(value)
    output.sum().backward()

    torch.testing.assert_close(output, value.detach().sum(dim=1))
    torch.testing.assert_close(value.grad, torch.ones_like(value))

    benchmark = benchmark_cpu(
        compiled,
        (value.detach(),),
        reference=lambda x: x.sum(dim=1),
        warmup=1,
        iterations=2,
    )
    assert benchmark.minimum_seconds > 0
    assert benchmark.reference_median_seconds is not None
    assert benchmark.plan_cost.operations > 0
    training = benchmark_cpu_training(compiled, (value,), warmup=1, iterations=2)
    assert training.backward_median_seconds > 0
    assert training.full_median_seconds > 0

    partial = build_schedule_space(compiled.problem).forward[2]
    partitioned = compile_cpu(
        compiled.operator,
        {"b": 5, "f": 5},
        plan=partial,
        schedule=CpuScheduleConfig(chunk_sizes=(("f", 2),), partition_count=2),
    )
    torch.testing.assert_close(partitioned(value.detach()), output.detach())


def test_generated_cpu_lws_attention_forward_and_backward():
    torch.manual_seed(3)
    sizes = {"h": 1, "l": 2, "g": 1, "r": 3, "dqk": 4, "dv": 5}
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk")),
            "key": TensorSpec(("h", "r", "dqk")),
            "value": TensorSpec(("h", "r", "dv")),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )
    algebra = LogWeightedSum(("dv",))

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

    operator = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output))
    source = emit_cpp_source(operator.lower(backward=False), sizes, backward=False)
    map_fold_source = source.split(
        "std::vector<torch::Tensor> project_output", maxsplit=1
    )[0]
    assert map_fold_source.count("at::bmm(") == 2
    assert "at::einsum" not in map_fold_source
    assert "numerator_ptr" not in map_fold_source

    compiled = compile_cpu(
        operator,
        sizes,
        backward=True,
        schedule=CpuScheduleConfig(chunk_sizes=(("r", 2),)),
    )
    assert compiled.backward_program is not None
    commit_modes = {
        gradient.input_name: gradient.commit_mode
        for gradient in compiled.backward_program.gradients
    }
    assert commit_modes == {
        "query": CommitMode.STORE,
        "key": CommitMode.ATOMIC_ADD_RELAXED,
        "value": CommitMode.ATOMIC_ADD_RELAXED,
    }
    actual_inputs = (
        torch.randn(1, 2, 1, 4, requires_grad=True),
        torch.randn(1, 3, 4, requires_grad=True),
        torch.randn(1, 3, 5, requires_grad=True),
    )
    actual = compiled(*actual_inputs)
    actual.square().sum().backward()

    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    scores = torch.einsum("hlgd,hrd->hlgr", reference_inputs[0], reference_inputs[1])
    reference = torch.einsum(
        "hlgr,hrv->hlgv", scores.softmax(dim=-1), reference_inputs[2]
    )
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference)
    for actual_input, reference_input in zip(
        actual_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(actual_input.grad, reference_input.grad)


def test_scheduled_cpu_attention_backward_uses_local_accumulation_and_atomics():
    torch.manual_seed(13)
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

    operator = FoldOperator(
        spec, LogWeightedSum(("dv",)), FoldMaps(to_factor, to_output)
    )
    problem = operator.lower()
    backward_program = schedule_backward_program(
        problem,
        sizes,
        loop_axis="l",
        tiles={"l": 2, "r": 3},
    )
    ssa = scheduled_to_ssa(backward_program)
    assert {carry.result.name for carry in ssa.loop.carries} == {
        "key.final",
        "value.final",
    }
    assert any(op.op == "atomic_add_relaxed" for op in ssa.loop.body)
    live_ranges = liveness(ssa)
    assert all(start <= end for start, end in live_ranges.values())
    source = emit_cpp_source(
        problem,
        sizes,
        backward=True,
        scheduled_backward=backward_program,
    )
    assert "std::atomic_ref<float>" in source
    assert "std::memory_order_relaxed" in source
    assert "local_key.add_" in source
    assert "destination_key.copy_" in source

    compiled = compile_cpu(
        operator,
        sizes,
        backward=True,
        backward_program=backward_program,
    )
    memory = scheduled_memory_metrics(compiled)
    assert memory.program_count == 3
    assert memory.loop_iterations == 3
    assert memory.persistent_bytes_per_program == (3 * 4 * 4 + 3 * 3 * 4)
    assert memory.streamed_bytes_per_program == 2 * 1 * 4 * 4
    assert (
        memory.maximum_local_bytes
        <= (memory.persistent_bytes_per_program + memory.streamed_bytes_per_program)
        * compiled.scheduled.target.worker_count
    )
    assert memory.atomic_updates == 1 * 5 * 1 * 4 * 3
    actual_inputs = (
        torch.randn(1, 5, 1, 4, requires_grad=True),
        torch.randn(1, 7, 4, requires_grad=True),
        torch.randn(1, 7, 3, requires_grad=True),
    )
    actual = compiled(*actual_inputs)
    actual.square().sum().backward()

    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    scores = torch.einsum("hlgd,hrd->hlgr", reference_inputs[0], reference_inputs[1])
    reference = torch.einsum(
        "hlgr,hrv->hlgv", scores.softmax(dim=-1), reference_inputs[2]
    )
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference)
    for actual_input, reference_input in zip(
        actual_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(actual_input.grad, reference_input.grad)


def test_generated_cpu_lws_bfloat16_inputs_accumulate_float32():
    sizes = {"b": 2, "f": 3, "q": 4, "v": 5}
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("b", "q"), "bfloat16"),
            "key": TensorSpec(("b", "f", "q"), "bfloat16"),
            "value": TensorSpec(("b", "f", "v"), "bfloat16"),
        },
        batch_axes=("b",),
        fold_axes=("f",),
    )
    algebra = LogWeightedSum(("v",))

    def to_factor(ctx, inputs):
        return StructuredValue(
            {
                "log_weight": ctx.contract(
                    inputs["query"],
                    inputs["key"],
                    over=("q",),
                    dtype="float32",
                ),
                "value": ctx.cast(inputs["value"], "float32"),
            }
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    compiled = compile_cpu(
        FoldOperator(spec, algebra, FoldMaps(to_factor, to_output)), sizes
    )
    query = torch.randn(2, 4, dtype=torch.bfloat16)
    key = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    value = torch.randn(2, 3, 5, dtype=torch.bfloat16)
    actual = compiled(query, key, value)
    scores = torch.einsum("bq,bfq->bf", query.float(), key.float())
    reference = torch.einsum("bf,bfv->bv", scores.softmax(dim=1), value.float())

    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, reference)


def test_generated_cpu_logsumexp_forward_and_backward():
    spec = FoldSpec(
        inputs={"x": TensorSpec(("b", "f"))},
        batch_axes=("b",),
        fold_axes=("f",),
    )

    def to_factor(ctx, inputs):
        del ctx
        return StructuredValue({"value": inputs["x"]})

    compiled = compile_cpu(
        FoldOperator(spec, LogSumExp(), FoldMaps(to_factor)),
        {"b": 2, "f": 5},
        backward=True,
    )
    actual_input = torch.randn(2, 5, requires_grad=True)
    actual = compiled(actual_input)
    actual.sum().backward()

    reference_input = actual_input.detach().clone().requires_grad_()
    reference = torch.logsumexp(reference_input, dim=1)
    reference.sum().backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)


def test_generated_cpu_affine_checkpointed_backward():
    torch.manual_seed(5)
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

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["payload"]["value"]})

    operator = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output))
    compiled = compile_cpu(
        operator,
        {"b": 2, "f": 4, "d": 3},
        plan=build_schedule_space(operator.lower()).forward[-1],
        backward=True,
        schedule=CpuScheduleConfig(chunk_sizes=(("f", 1),), partition_count=2),
    )
    actual_scale = torch.randn(2, 4, requires_grad=True)
    actual_payload = torch.randn(2, 4, 3, requires_grad=True)
    actual = compiled(actual_scale, actual_payload)
    actual.square().sum().backward()

    reference_scale = actual_scale.detach().clone().requires_grad_()
    reference_payload = actual_payload.detach().clone().requires_grad_()
    reference = torch.zeros(2, 3)
    for index in range(4):
        reference = (
            torch.exp(reference_scale[:, index, None]) * reference
            + reference_payload[:, index]
        )
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_scale.grad, reference_scale.grad)
    torch.testing.assert_close(actual_payload.grad, reference_payload.grad)


def test_generated_cpu_product_cross_entropy_forward_and_backward():
    torch.manual_seed(7)
    spec = FoldSpec(
        inputs={
            "context": TensorSpec(("b", "d")),
            "targets": TensorSpec(("v", "d")),
            "is_target": TensorSpec(("b", "v"), "bool"),
        },
        batch_axes=("b",),
        fold_axes=("v",),
    )
    algebra = Product(normalizer=LogSumExp(), target=Add())

    def to_factor(ctx, inputs):
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

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue(
            {"loss": semantic["normalizer"]["value"] - semantic["target"]["value"]}
        )

    operator = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output))
    sizes = {"b": 2, "v": 4, "d": 3}
    backward_program = schedule_backward_program(
        operator.lower(),
        sizes,
        loop_axis="b",
        tiles={"b": 1, "v": 2},
    )
    compiled = compile_cpu(
        operator,
        sizes,
        backward=True,
        backward_program=backward_program,
        schedule=CpuScheduleConfig(chunk_sizes=(("v", 3),)),
    )
    actual_context = torch.randn(2, 3, requires_grad=True)
    actual_targets = torch.randn(4, 3, requires_grad=True)
    is_target = torch.zeros(2, 4, dtype=torch.bool)
    is_target[0, 1] = True
    is_target[1, 3] = True
    actual = compiled(actual_context, actual_targets, is_target)
    actual.sum().backward()

    reference_context = actual_context.detach().clone().requires_grad_()
    reference_targets = actual_targets.detach().clone().requires_grad_()
    logits = reference_context @ reference_targets.T
    reference = torch.logsumexp(logits, dim=1) - torch.where(
        is_target, logits, 0.0
    ).sum(dim=1)
    reference.sum().backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_context.grad, reference_context.grad)
    torch.testing.assert_close(actual_targets.grad, reference_targets.grad)


def test_generated_cpu_affine_lws_has_finite_empty_prefix_gradient():
    torch.manual_seed(11)
    algebra = AffineLWS(("d",))
    spec = FoldSpec(
        inputs={
            "scale": TensorSpec(("b", "f")),
            "log_weight": TensorSpec(("b", "f")),
            "value": TensorSpec(("b", "f", "d")),
        },
        batch_axes=("b",),
        fold_axes=("f",),
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

    operator = FoldOperator(spec, algebra, FoldMaps(to_factor, to_output))
    compiled = compile_cpu(
        operator,
        {"b": 2, "f": 3, "d": 2},
        plan=build_schedule_space(operator.lower()).forward[-1],
        backward=True,
        schedule=CpuScheduleConfig(chunk_sizes=(("f", 1),), partition_count=2),
    )
    actual_inputs = (
        torch.randn(2, 3, requires_grad=True),
        torch.randn(2, 3, requires_grad=True),
        torch.randn(2, 3, 2, requires_grad=True),
    )
    actual = compiled(*actual_inputs)
    actual.square().sum().backward()

    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )
    suffix_scale = (
        torch.flip(
            torch.cumsum(torch.flip(reference_inputs[0], dims=(1,)), dim=1),
            dims=(1,),
        )
        - reference_inputs[0]
    )
    effective_logits = reference_inputs[1] + suffix_scale
    reference = torch.einsum(
        "bf,bfd->bd", effective_logits.softmax(dim=1), reference_inputs[2]
    )
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference)
    for actual_input, reference_input in zip(
        actual_inputs, reference_inputs, strict=True
    ):
        assert torch.isfinite(actual_input.grad).all()
        torch.testing.assert_close(actual_input.grad, reference_input.grad)
