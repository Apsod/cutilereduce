"""Launch adapters for generated online cuTile folds and prefix-free backward."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.ssa import SpecializedSchedule

from cutilereduce.experimental.planning.templates import build_schedule_space
from .schedule import specialize_cutile_schedule, specialize_cutile_backward_schedule
from .codegen import emit_cutile_source, emit_cutile_backward_source


def _load_source(source, directory, suffix):
    name = "cutilereduce_tuned_" + hashlib.sha256(source.encode()).hexdigest()[:20] + suffix
    if name in sys.modules:
        return sys.modules[name]
    path = Path(directory) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generated source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass
class CompiledCuTileFold:
    problem: FoldProgram
    sizes: dict[str, int]
    forward_schedule: SpecializedSchedule
    forward_module: Any
    backward_schedule: SpecializedSchedule | None
    backward_module: Any
    source_directory: tempfile.TemporaryDirectory

    def _allocate(self, region, items, device, *, zero=False):
        import torch
        factory = torch.zeros if zero else torch.empty
        return {name: factory(tuple(self.sizes[a] for a in region.nodes[i].axes),
                              dtype=getattr(torch, region.nodes[i].dtype), device=device)
                for name, i in items}

    def _launch(self, stage, module, arguments, stream):
        import cuda.tile as ct
        grid = math.prod(p.programs for p in stage.grid_partitions)
        ct.launch(stream, (grid, 1, 1), getattr(module, stage.name), arguments)

    def forward_values(self, inputs):
        import torch
        if len(inputs) != len(self.problem.spec.inputs):
            raise ValueError("input arity mismatch")
        if not inputs:
            raise ValueError("cuTile runtime requires at least one input")
        device = inputs[0].device
        for tensor, (name, spec) in zip(inputs, self.problem.spec.inputs.items(), strict=True):
            if tensor.device.type != "cuda" or tensor.device != device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous tensor on the same CUDA device")
            if tensor.dtype != getattr(torch, spec.dtype) or tensor.shape != tuple(self.sizes[a] for a in spec.axes):
                raise ValueError(f"{name} shape or dtype mismatch")
        outputs = self._allocate(self.problem.project_output, self.problem.project_output.results, device)
        saved = (self._allocate(self.problem.map_fold, self.problem.map_fold.results, device)
                 if self.backward_schedule is not None else {})
        stage = self.forward_schedule.implementation.stages[0]
        values = dict(zip(self.problem.spec.inputs, inputs, strict=True))
        args = tuple(values[b.name] for b in stage.bindings) + tuple(outputs.values()) + tuple(saved.values())
        self._launch(stage, self.forward_module, args, torch.cuda.current_stream(device))
        return tuple(outputs.values()), saved

    def _backward_values_context(self, inputs, saved, cotangents):
        if self.backward_schedule is None:
            raise ValueError("fold was compiled without backward")
        values = dict(zip(self.problem.spec.inputs, inputs, strict=True))
        values.update({name.replace("execution.", "saved_execution.", 1): tensor for name, tensor in saved.items()})
        for (name, _), cotangent in zip(self.problem.project_output.results, cotangents, strict=True):
            values[f"grad_output.{name.removeprefix('output.')}"] = cotangent.contiguous()
        return values

    def embed_values(self, inputs, saved, cotangents):
        import torch
        values = self._backward_values_context(inputs, saved, cotangents)
        device = inputs[0].device
        stream = torch.cuda.current_stream(device)
        assert self.backward_schedule is not None
        embed, _ = self.backward_schedule.implementation.stages
        assert self.problem.embed_output_vjp is not None
        accumulator = self._allocate(self.problem.embed_output_vjp, self.problem.embed_output_vjp.results, device)
        args = tuple(values[b.name] for b in embed.bindings) + tuple(accumulator.values())
        self._launch(embed, self.backward_module, args, stream)
        return accumulator

    def backward_values(self, inputs, saved, cotangents):
        accumulator = self.embed_values(inputs, saved, cotangents)
        return self.finalize_values(inputs, saved, cotangents, accumulator)

    def finalize_values(self, inputs, saved, cotangents, accumulator):
        import torch
        values = self._backward_values_context(inputs, saved, cotangents)
        device = inputs[0].device
        stream = torch.cuda.current_stream(device)
        assert self.backward_schedule is not None
        _, finalize = self.backward_schedule.implementation.stages
        values.update(accumulator)
        region = self.problem.map_finalize_map_vjp
        assert region is not None
        gradients = self._allocate(region, region.results, device, zero=True)
        destinations = tuple(dict.fromkeys(dict(op.attributes)["destination"]
                             for op in (*finalize.loop.body, *finalize.commits)
                             if op.op in {"store", "atomic_add_relaxed"}))
        args = tuple(values[b.name] for b in finalize.bindings) + tuple(gradients[name] for name in destinations)
        self._launch(finalize, self.backward_module, args, stream)
        return tuple(gradients.get(f"partial_grad_input.{name}") for name in self.problem.spec.inputs)

    def __call__(self, *inputs):
        if self.backward_schedule is None:
            values, _ = self.forward_values(inputs)
            return values[0] if len(values) == 1 else values
        return _autograd_function().apply(self, *inputs)


def _autograd_function():
    import torch

    class FoldFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, compiled, *inputs):
            outputs, saved = compiled.forward_values(inputs)
            ctx.compiled = compiled
            ctx.saved_names = tuple(saved)
            ctx.input_count = len(inputs)
            ctx.save_for_backward(*inputs, *saved.values())
            return outputs[0] if len(outputs) == 1 else outputs

        @staticmethod
        def backward(ctx, *cotangents):
            values = ctx.saved_tensors
            inputs = values[:ctx.input_count]
            saved = dict(zip(ctx.saved_names, values[ctx.input_count:], strict=True))
            return (None, *ctx.compiled.backward_values(inputs, saved, cotangents))

    return FoldFunction


def realize_cutile_candidate(problem, candidate):
    """Bind concrete schedules without emitting source or loading executable code."""
    from dataclasses import replace
    if problem.target.backend != "cutile":
        raise ValueError("cuTile compilation requires a cuTile planning problem")
    candidate = problem.validate_candidate(candidate)
    p = dict(candidate.parameters)
    sizes = dict(problem.sizes)
    space = build_schedule_space(problem.program)
    template = next(plan for plan in space.forward if plan.name == candidate.family)
    tiles = {a: p[f"tile.{a}"] for a in (*problem.program.spec.batch_axes, *problem.program.spec.fold_axes)}
    forward = specialize_cutile_schedule(problem.program, template, sizes, tiles=tiles)
    backward = None
    if candidate.backward:
        btiles = {a: p.get(f"backward_tile.{a}", t) for a, t in tiles.items()}
        backward = specialize_cutile_backward_schedule(
            problem.program, space.backward[0], sizes, loop_axis=candidate.loop_axis or problem.program.spec.fold_axes[0],
            tiles=btiles, loop_programs=p.get("loop_programs", 1),
        )
        # Embed has its own batch tiling; it does not traverse the fold axis.
        embed, finalize = backward.implementation.stages
        from cutilereduce.experimental.planning.scheduled import ScheduledAxis
        from cutilereduce.experimental.planning.ssa import AxisPartition
        etiles = {a: p.get(f"embed_tile.{a}", tiles[a]) for a in problem.program.spec.batch_axes}
        embed = replace(embed,
                        parallel_axes=tuple(ScheduledAxis(a, t, sizes[a]) for a, t in etiles.items()),
                        partitions=tuple(AxisPartition(a, sizes[a], t, math.ceil(sizes[a] / t)) for a, t in etiles.items()))
        backward = replace(backward, implementation=replace(backward.implementation, stages=(embed, finalize)))
    if candidate.backward:
        from cutilereduce.experimental.planning.ssa import TiledBuffer
        from cutilereduce.experimental.ir.core import TensorSignature
        saved = TiledBuffer("saved_execution", tuple(
            (name, TensorSignature(problem.program.map_fold.nodes[node].axes,
                                   problem.program.map_fold.nodes[node].dtype))
            for name, node in problem.program.map_fold.results))
        forward = replace(forward, implementation=replace(forward.implementation, buffers=(saved,)))
    return forward, backward


def compile_cutile_candidate(operator, problem, candidate):
    """Compile the concrete schedules; GPU compilation remains lazy."""
    from .backend import CuTileBackend
    from .target import CuTileTarget
    backend = CuTileBackend(analytical_target=problem.target)
    hardware = CuTileTarget(**dict(problem.target.hardware_identity))
    prepared = backend.specialize(problem.program, hardware, sizes=dict(problem.sizes))
    return backend.compile(backend.realize(prepared, candidate))


def compile_cutile_realization(program, sizes, forward, backward=None):
    """Emit and load previously realized schedules, without selecting again."""
    backward_source = emit_cutile_backward_source(backward) if backward is not None else None
    directory = tempfile.TemporaryDirectory(prefix="cutilereduce-tuned-")
    save_execution = any(buffer.name == "saved_execution" for buffer in forward.implementation.buffers)
    if save_execution != (backward is not None):
        raise ValueError("saved execution buffers do not match backward realization")
    source = emit_cutile_source(forward, save_execution=save_execution)
    fmodule = _load_source(source, directory.name, "_forward")
    bmodule = _load_source(backward_source, directory.name, "_backward") if backward_source else None
    # cuTile may inspect source lazily; cached modules must retain their files.
    for module in (fmodule, bmodule):
        if module is not None and not hasattr(module, "_generated_directory"):
            module._generated_directory = directory
    return CompiledCuTileFold(program, sizes, forward, fmodule, backward, bmodule, directory)
