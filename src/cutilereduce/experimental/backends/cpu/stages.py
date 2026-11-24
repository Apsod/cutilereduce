"""CPU callable stage boundaries: fused forward, embed, scheduled finalize."""

import hashlib

from cutilereduce.experimental.planning.stages import standard_plan
from ..staging import RunnableStage, StageFixture, ComposedFold


def stage_plan(prepared, candidate):
    return standard_plan(
        prepared,
        candidate,
        namespace_key=prepared.stage_namespace,
        embed_tiled=False,
        layout="cpu-strided-tensor-carrier-v1",
    )


def compile_stage(backend, prepared, stage):
    from .runtime import compile_cpu, _region_arguments, _bind_results, _cpu_build_flags

    implementation = backend.realize(prepared, stage.representative)
    registry = getattr(prepared.analytical_target.region_analyzer, "__self__", None)
    sizes = dict(prepared.sizes)
    if stage.slot == "embed":
        from torch.utils.cpp_extension import load_inline
        from .regions import _emit_region, program_to_region
        from cutilereduce.experimental.ir.core import TensorProgram

        region = prepared.program.embed_output_vjp
        nested, helpers = {}, []

        def visit(current):
            for node in current.nodes:
                program = dict(node.attributes).get("combine_region")
                if isinstance(program, TensorProgram) and id(program) not in nested:
                    child = program_to_region(program)
                    visit(child)
                    name = f"stage_combine_{len(nested)}"
                    nested[id(program)] = name
                    helpers.append(
                        _emit_region(name, child, sizes, nested_names=nested)
                    )

        visit(region)
        kwargs = (
            {"implementation": registry.realize(region)} if registry is not None else {}
        )
        source = "\n\n".join(
            (
                *helpers,
                _emit_region(
                    "stage_embed", region, sizes, nested_names=nested, **kwargs
                ),
            )
        )
        flags, link = _cpu_build_flags()
        key = hashlib.sha256((source + repr((flags, link))).encode()).hexdigest()[:20]
        module = load_inline(
            "cutilereduce_embed_" + key,
            cpp_sources=source,
            functions=["stage_embed"],
            extra_cflags=flags,
            extra_ldflags=link,
            with_cuda=False,
        )

        def run(context):
            values = dict(
                zip(prepared.program.spec.inputs, context["inputs"], strict=True)
            )
            values.update(context["carrier"])
            values.update(
                {
                    f"grad_output.{name}": tensor
                    for (name, _), tensor in zip(
                        prepared.program.project_output.results,
                        context["cotangents"],
                        strict=True,
                    )
                }
            )
            accumulator = _bind_results(
                region, list(module.stage_embed(*_region_arguments(region, values)))
            )
            return {"accumulator": accumulator}

        return RunnableStage(stage, run, module)
    config = implementation.metadata
    execution = implementation.execution
    compiled = compile_cpu(
        None,
        sizes,
        plan=implementation.forward.template,
        schedule=config.forward,
        backward_schedule=config.backward,
        backward=stage.slot == "finalize",
        backward_program=config.scheduled_backward
        if stage.slot == "finalize"
        else None,
        target=prepared.hardware,
        _program=prepared.program,
        _forward_scheduled=execution.forward,
        _backward_scheduled=execution.backward,
        _region_realizations=registry,
        _emit_forward=stage.slot == "forward",
    )
    if stage.slot == "forward":

        def run(context):
            outputs, execution = compiled.forward_values(*context["inputs"])
            return {
                "outputs": outputs,
                "execution": execution,
                "carrier": {
                    n: t for n, t in execution.items() if n.startswith("execution.")
                },
            }
    elif stage.slot == "finalize":

        def run(context):
            return {
                "gradients": compiled.finalize_values(
                    context["inputs"],
                    context["execution"],
                    context["cotangents"],
                    context["accumulator"],
                )
            }
    else:
        raise NotImplementedError(f"unknown CPU stage {stage.slot!r}")
    return RunnableStage(stage, run, compiled)


def fixture(prepared, inputs, reference, **kwargs):
    return StageFixture(prepared, inputs, reference, **kwargs)


def assemble(prepared, stages):
    return ComposedFold(prepared, stages)
