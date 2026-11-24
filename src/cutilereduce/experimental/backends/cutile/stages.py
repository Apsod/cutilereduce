"""cuTile executable units are the forward, embed and finalize kernel launches."""

import tempfile

from cutilereduce.experimental.planning.stages import standard_plan
from ..staging import RunnableStage, StageFixture, ComposedFold


def stage_plan(prepared, candidate):
    return standard_plan(
        prepared,
        candidate,
        namespace_key=prepared.stage_namespace,
        embed_tiled=True,
        layout="gpu-contiguous-carrier-v1",
    )


def compile_stage(backend, prepared, stage):
    from .runtime import CompiledCuTileFold, _load_source
    from .codegen import emit_cutile_source, emit_cutile_backward_source

    implementation = backend.realize(prepared, stage.representative)
    forward, backward = implementation.forward, implementation.backward
    directory = tempfile.TemporaryDirectory(prefix="cutilereduce-stage-")
    if stage.slot == "forward":
        module = _load_source(
            emit_cutile_source(forward, save_execution=backward is not None),
            directory.name,
            "_forward",
        )
        compiled = CompiledCuTileFold(
            prepared.program,
            dict(prepared.sizes),
            forward,
            module,
            backward,
            None,
            directory,
        )

        def run(context):
            outputs, execution = compiled.forward_values(context["inputs"])
            return {
                "outputs": outputs,
                "execution": execution,
                "carrier": {
                    n: t for n, t in execution.items() if n.startswith("execution.")
                },
            }
    else:
        module = _load_source(
            emit_cutile_backward_source(backward), directory.name, "_backward"
        )
        compiled = CompiledCuTileFold(
            prepared.program,
            dict(prepared.sizes),
            forward,
            None,
            backward,
            module,
            directory,
        )
        if stage.slot == "embed":

            def run(context):
                return {
                    "accumulator": compiled.embed_values(
                        context["inputs"], context["carrier"], context["cotangents"]
                    )
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
            raise NotImplementedError(f"unknown cuTile stage {stage.slot!r}")
    return RunnableStage(stage, run, compiled)


def fixture(prepared, inputs, reference, **kwargs):
    return StageFixture(prepared, inputs, reference, **kwargs)


def assemble(prepared, stages):
    return ComposedFold(prepared, stages)
