"""Compilation, tensor binding, and autograd execution of native CPU schedules."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from cutilereduce.experimental.backends.cpu.schedule import (
    CpuExpr,
    CpuLeaf,
    CpuLoop,
    CpuScheduleConfig,
    CpuScheduledPlan,
    CpuTarget,
)
from cutilereduce.experimental.ir.core import TensorRegion
from cutilereduce.experimental.compiler.operator import FoldOperator, FoldProgram
from cutilereduce.experimental.planning.templates import (
    FoldTopology,
    ForwardScheduleTemplate,
    MapFoldRegion,
    RegionCall,
    build_schedule_space,
)
from cutilereduce.experimental.planning.scheduled import (
    ScheduledBackwardProgram,
    schedule_backward_program,
)
from cutilereduce.experimental.planning.ssa import SpecializedSchedule
from cutilereduce.experimental.planning.target import ExecutionTarget


from .codegen import emit_cpp_source


def _cpu_build_flags(*, scheduled_backward: bool = False):
    """Match ATen's parallel backend when compiling local generated loops."""
    import torch

    cflags = ["-O3", "-march=native"]
    ldflags = []
    if scheduled_backward:
        cflags.append("-std=c++20")
    if sys.platform.startswith("linux") and torch.backends.openmp.is_available():
        # ParallelOpenMP.h only enables its parallel_for template under _OPENMP.
        cflags.append("-fopenmp")
        ldflags.append("-fopenmp")
    return cflags, ldflags


def _region_arguments(region: TensorRegion, values: dict[str, Any]) -> list[Any]:
    arguments = []
    for name, _ in region.arguments:
        candidates = (name, name.replace("saved_execution.", "execution."))
        value = next((values[key] for key in candidates if key in values), None)
        if value is None:
            raise ValueError(f"missing region argument {name!r}")
        arguments.append(value)
    return arguments


def _bind_results(region: TensorRegion, tensors: list[Any]) -> dict[str, Any]:
    if len(tensors) != len(region.results):
        raise ValueError("compiled region returned the wrong number of values")
    return {
        name: tensor for (name, _), tensor in zip(region.results, tensors, strict=True)
    }


@dataclass
class CompiledCpuFold:
    operator: FoldOperator | None
    problem: FoldProgram
    scheduled: CpuScheduledPlan
    modules: dict[int, Any]
    compile_seconds: float
    backward_program: ScheduledBackwardProgram | None = None
    specialized: SpecializedSchedule | None = None

    @property
    def supports_backward(self) -> bool:
        return bool(self.scheduled.backward) and self.problem.embed_output_vjp is not None

    @property
    def module(self):
        fold_axis = self.problem.spec.fold_axes[0]
        return self.modules[dict(self.scheduled.sizes)[fold_axis]]

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> dict[str, Any]:
        import torch

        torch_dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "bool": torch.bool,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        specs = tuple(self.problem.spec.inputs.items())
        if len(inputs) != len(specs):
            raise ValueError(f"expected {len(specs)} inputs, got {len(inputs)}")
        sizes = dict(self.scheduled.sizes)
        values = {}
        for tensor, (name, spec) in zip(inputs, specs, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"input {name!r} is not a torch.Tensor")
            expected_shape = tuple(sizes[axis] for axis in spec.axes)
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"input {name!r} must be a contiguous CPU tensor")
            if tensor.dtype != torch_dtypes[spec.dtype]:
                raise ValueError(
                    f"input {name!r} has dtype {tensor.dtype}, "
                    f"expected {torch_dtypes[spec.dtype]}"
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"input {name!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            values[name] = tensor
        return values

    def forward_values(self, *inputs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        values = self._validate_inputs(inputs)
        if hasattr(self.module, "scheduled_forward"):
            flat = list(self.module.scheduled_forward(*inputs))
            output_count = len(self.problem.project_output.results)
            carrier_count = len(self.problem.map_fold.results)
            outputs = tuple(flat[:output_count])
            carrier_values = flat[output_count : output_count + carrier_count]
            execution = _bind_results(self.problem.map_fold, carrier_values)
            saved = dict(execution)
            prefix_values = flat[output_count + carrier_count :]
            if len(prefix_values) % carrier_count:
                raise ValueError("scheduled forward returned malformed prefixes")
            for index in range(len(prefix_values) // carrier_count):
                offset = index * carrier_count
                for (name, _), value in zip(
                    self.problem.map_fold.results,
                    prefix_values[offset : offset + carrier_count],
                    strict=True,
                ):
                    suffix = name.removeprefix("execution.")
                    saved[f"prefix_execution.{index}.{suffix}"] = value
            return outputs, saved
        fold_axis = self.problem.spec.fold_axes[0]
        fold_extent = dict(self.scheduled.sizes)[fold_axis]
        forward = self.scheduled.forward
        root = forward[0].bindings[0].expression
        if isinstance(root, CpuLoop) and root.operation == "map":
            partials = self._execute_map(root, inputs, 0, fold_extent)
            if not partials:
                raise ValueError("partitioned fold produced no partial carriers")
            combine = forward[1].bindings[0].expression
            combine_expression = forward[1].bindings[0].expression
            topology = (
                combine.topology
                if isinstance(combine, CpuLoop) and combine.topology is not None
                else FoldTopology.ONLINE
            )
            execution = self._combine_many(partials, topology)
            prefixes = (
                self._exclusive_prefixes(partials)
                if isinstance(combine_expression, CpuLoop)
                and combine_expression.operation == "exclusive_scan"
                else []
            )
        else:
            execution = self._execute_carrier(root, inputs, 0, fold_extent)
            prefixes = []
        values.update(execution)
        outputs = _bind_results(
            self.problem.project_output,
            list(
                self.module.project_output(
                    *_region_arguments(self.problem.project_output, values)
                )
            ),
        )
        saved = dict(execution)
        for index, prefix in enumerate(prefixes):
            for name, value in prefix.items():
                suffix = name.removeprefix("execution.")
                saved[f"prefix_execution.{index}.{suffix}"] = value
        return tuple(
            outputs[name] for name, _ in self.problem.project_output.results
        ), saved

    def _exclusive_prefixes(
        self, partials: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        identity = _bind_results(
            self.problem.execution_identity,
            list(self.module.execution_identity()),
        )
        first = partials[0]
        state = {
            name: value.reshape(
                (1,) * (first[name].ndim - value.ndim) + tuple(value.shape)
            )
            .expand_as(first[name])
            .clone()
            for name, value in identity.items()
        }
        prefixes = []
        for partial in partials:
            prefixes.append(state)
            state = self._combine(state, partial)
        return prefixes

    def _slice_inputs(
        self, inputs: tuple[Any, ...], start: int, extent: int
    ) -> dict[str, Any]:
        fold_axis = self.problem.spec.fold_axes[0]
        values = {}
        for tensor, (name, spec) in zip(
            inputs, self.problem.spec.inputs.items(), strict=True
        ):
            if fold_axis in spec.axes:
                dimension = spec.axes.index(fold_axis)
                tensor = tensor.narrow(dimension, start, extent).contiguous()
            values[name] = tensor
        return values

    def _map_fold(
        self, inputs: tuple[Any, ...], start: int, extent: int
    ) -> dict[str, Any]:
        module = self.modules[extent]
        values = self._slice_inputs(inputs, start, extent)
        return _bind_results(
            self.problem.map_fold,
            list(module.map_fold(*_region_arguments(self.problem.map_fold, values))),
        )

    def _combine(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        values = {}
        for name, value in left.items():
            suffix = name.removeprefix("execution.")
            values[f"left.{suffix}"] = value
        for name, value in right.items():
            suffix = name.removeprefix("execution.")
            values[f"right.{suffix}"] = value
        return _bind_results(
            self.problem.execution_combine,
            list(
                self.module.execution_combine(
                    *_region_arguments(self.problem.execution_combine, values)
                )
            ),
        )

    def _combine_many(
        self, values: list[dict[str, Any]], topology: FoldTopology
    ) -> dict[str, Any]:
        if not values:
            raise ValueError("cannot combine an empty carrier collection")
        if topology is FoldTopology.ONLINE:
            state = values[0]
            for value in values[1:]:
                state = self._combine(state, value)
            return state
        level = values
        while len(level) > 1:
            next_level = []
            for index in range(0, len(level), 2):
                if index + 1 == len(level):
                    next_level.append(level[index])
                else:
                    next_level.append(self._combine(level[index], level[index + 1]))
            level = next_level
        return level[0]

    def _execute_carrier(
        self,
        expression: CpuExpr,
        inputs: tuple[Any, ...],
        start: int,
        extent: int,
    ) -> dict[str, Any]:
        if isinstance(expression, CpuLeaf):
            if not isinstance(expression.expression, MapFoldRegion):
                raise TypeError("carrier leaf must invoke map_fold")
            return self._map_fold(inputs, start, extent)
        if (
            not isinstance(expression, CpuLoop)
            or expression.operation != "carrier_fold"
        ):
            raise TypeError("unsupported executable carrier expression")
        ranges = self._ranges(start, extent, expression.iteration.grain)
        if expression.iteration.loop.value == "parallel" and len(ranges) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.scheduled.target.worker_count, len(ranges))
            ) as pool:
                values = list(
                    pool.map(
                        lambda item: self._execute_carrier(
                            expression.body, inputs, *item
                        ),
                        ranges,
                    )
                )
        else:
            values = [
                self._execute_carrier(expression.body, inputs, *item) for item in ranges
            ]
        return self._combine_many(values, expression.topology or FoldTopology.ONLINE)

    def _execute_map(
        self,
        expression: CpuLoop,
        inputs: tuple[Any, ...],
        start: int,
        extent: int,
    ) -> list[dict[str, Any]]:
        ranges = self._ranges(start, extent, expression.iteration.grain)
        if expression.iteration.loop.value != "parallel" or len(ranges) <= 1:
            return [
                self._execute_carrier(expression.body, inputs, *item) for item in ranges
            ]
        with ThreadPoolExecutor(
            max_workers=min(self.scheduled.target.worker_count, len(ranges))
        ) as pool:
            return list(
                pool.map(
                    lambda item: self._execute_carrier(expression.body, inputs, *item),
                    ranges,
                )
            )

    @staticmethod
    def _ranges(start: int, extent: int, grain: int) -> list[tuple[int, int]]:
        result = []
        offset = start
        end = start + extent
        while offset < end:
            local_extent = min(grain, end - offset)
            result.append((offset, local_extent))
            offset += local_extent
        return result

    def backward_values(
        self,
        inputs: tuple[Any, ...],
        execution: dict[str, Any],
        output_cotangents: tuple[Any, ...],
    ) -> tuple[Any | None, ...]:
        if self.problem.embed_output_vjp is None:
            raise ValueError("CPU fold was compiled without backward regions")
        backward_region = (
            self.problem.map_finalize_map_vjp
            or self.problem.map_finalize_map_vjp_accumulate
        )
        if backward_region is None:
            raise ValueError("CPU fold has no map-finalize backward region")
        values = self._validate_inputs(inputs)
        values.update(execution)
        if len(output_cotangents) != len(self.problem.project_output.results):
            raise ValueError("output cotangent count does not match outputs")
        for (name, _), cotangent in zip(
            self.problem.project_output.results,
            output_cotangents,
            strict=True,
        ):
            values[f"grad_output.{name}"] = cotangent.contiguous()
        accumulator = _bind_results(
            self.problem.embed_output_vjp,
            list(
                self.module.embed_output_vjp(
                    *_region_arguments(self.problem.embed_output_vjp, values)
                )
            ),
        )
        values.update(accumulator)
        if self.backward_program is not None:
            tensors = list(
                self.module.scheduled_backward(
                    *_region_arguments(backward_region, values)
                )
            )
            gradients = {
                gradient.result_name: tensor
                for gradient, tensor in zip(
                    self.backward_program.gradients, tensors, strict=True
                )
            }
        elif not self.problem.requires_prefix:
            gradients = self._independent_backward(inputs, values)
        else:
            gradients = self._prefix_backward(inputs, execution, values)
        result_nodes = dict(backward_region.results)
        input_gradients = []
        for name, spec in self.problem.spec.inputs.items():
            result_name = f"partial_grad_input.{name}"
            value = gradients.get(result_name) if spec.differentiable else None
            if value is not None:
                axes = backward_region.nodes[result_nodes[result_name]].axes
                if axes != spec.axes:
                    value = value.permute(tuple(axes.index(axis) for axis in spec.axes))
            input_gradients.append(value)
        return tuple(input_gradients)

    def _merge_input_gradients(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        fold_axis = self.problem.spec.fold_axes[0]
        region = self.problem.map_finalize_map_vjp
        if region is None:
            raise ValueError("independent backward region is unavailable")
        result_nodes = dict(region.results)
        gradients = {}
        for name, spec in self.problem.spec.inputs.items():
            key = f"partial_grad_input.{name}"
            parts = [result[key] for result in results if key in result]
            if not parts:
                continue
            if fold_axis in spec.axes:
                axes = region.nodes[result_nodes[key]].axes
                gradients[key] = torch.cat(parts, dim=axes.index(fold_axis))
            else:
                value = parts[0]
                for part in parts[1:]:
                    value = value + part
                gradients[key] = value
        return gradients

    def _independent_backward(
        self, inputs: tuple[Any, ...], common_values: dict[str, Any]
    ) -> dict[str, Any]:
        region = self.problem.map_finalize_map_vjp
        if region is None:
            raise ValueError("prefix-free backward region is unavailable")
        root = self.scheduled.backward[-1].bindings[0].expression
        fold_extent = dict(self.scheduled.sizes)[self.problem.spec.fold_axes[0]]
        chunks = self._backward_chunks(root, 0, fold_extent)

        def execute(item):
            start, extent = item
            local_values = dict(common_values)
            local_values.update(self._slice_inputs(inputs, start, extent))
            return _bind_results(
                region,
                list(
                    self.modules[extent].map_finalize_map_vjp(
                        *_region_arguments(region, local_values)
                    )
                ),
            )

        if len(chunks) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.scheduled.target.worker_count, len(chunks))
            ) as pool:
                results = list(pool.map(execute, chunks))
        else:
            results = [execute(chunks[0])]
        return self._merge_input_gradients(results)

    def _backward_chunks(
        self, expression: CpuExpr, start: int, extent: int
    ) -> list[tuple[int, int]]:
        if isinstance(expression, CpuLeaf):
            return [(start, extent)]
        if not isinstance(expression, CpuLoop):
            raise TypeError("unsupported backward schedule expression")
        result = []
        for local_start, local_extent in self._ranges(
            start, extent, expression.iteration.grain
        ):
            result.extend(
                self._backward_chunks(expression.body, local_start, local_extent)
            )
        return result

    def _prefix_backward(
        self,
        inputs: tuple[Any, ...],
        saved: dict[str, Any],
        common_values: dict[str, Any],
    ) -> dict[str, Any]:
        region = self.problem.map_finalize_map_vjp_accumulate
        if region is None:
            raise ValueError("prefix backward region is unavailable")
        forward_root = self.scheduled.forward[0].bindings[0].expression
        backward_root = self.scheduled.backward[-1].bindings[0].expression
        if not isinstance(forward_root, CpuLoop) or forward_root.operation != "map":
            raise ValueError("prefix backward requires partitioned forward")
        fold_axis = self.problem.spec.fold_axes[0]
        fold_extent = dict(self.scheduled.sizes)[fold_axis]
        partition_ranges = self._ranges(0, fold_extent, forward_root.iteration.grain)
        fold_gradients: dict[str, list[Any]] = {
            name: [] for name in self.problem.spec.inputs
        }
        reduced_gradients: dict[str, Any] = {}
        for partition_index, (partition_start, partition_extent) in enumerate(
            partition_ranges
        ):
            prefix = {
                name.removeprefix(f"prefix_execution.{partition_index}."): value
                for name, value in saved.items()
                if name.startswith(f"prefix_execution.{partition_index}.")
            }
            prefix = {f"prefix.{name}": value for name, value in prefix.items()}
            body = (
                backward_root.body
                if isinstance(backward_root, CpuLoop)
                else backward_root
            )
            grain = (
                body.iteration.grain if isinstance(body, CpuLoop) else partition_extent
            )
            for chunk_start, chunk_extent in self._ranges(
                partition_start, partition_extent, grain
            ):
                local_values = dict(common_values)
                local_values.update(
                    self._slice_inputs(inputs, chunk_start, chunk_extent)
                )
                local_values.update(prefix)
                result = _bind_results(
                    region,
                    list(
                        self.modules[chunk_extent].map_finalize_map_vjp(
                            *_region_arguments(region, local_values)
                        )
                    ),
                )
                prefix = {
                    name.replace("updated_prefix.", "prefix."): value
                    for name, value in result.items()
                    if name.startswith("updated_prefix.")
                }
                for name, spec in self.problem.spec.inputs.items():
                    key = f"partial_grad_input.{name}"
                    if key not in result:
                        continue
                    if fold_axis in spec.axes:
                        fold_gradients[name].append(result[key])
                    else:
                        previous = reduced_gradients.get(name)
                        reduced_gradients[name] = (
                            result[key] if previous is None else previous + result[key]
                        )
        gradients = {
            f"partial_grad_input.{name}": value
            for name, value in reduced_gradients.items()
        }
        import torch

        for name, parts in fold_gradients.items():
            if not parts:
                continue
            result_name = f"partial_grad_input.{name}"
            dimension = region.nodes[dict(region.results)[result_name]].axes.index(
                fold_axis
            )
            gradients[f"partial_grad_input.{name}"] = torch.cat(parts, dim=dimension)
        return gradients

    def __call__(self, *inputs):
        if self.supports_backward and any(
            getattr(value, "requires_grad", False) for value in inputs
        ):
            return _CompiledFoldFunction.apply(self, *inputs)
        outputs, _ = self.forward_values(*inputs)
        return outputs[0] if len(outputs) == 1 else outputs


class _CompiledFoldFunction:
    """Lazily defined torch.autograd.Function to keep torch import optional."""

    @classmethod
    def apply(cls, compiled: CompiledCpuFold, *inputs):
        import torch

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *flat_inputs):
                outputs, execution = compiled.forward_values(*flat_inputs)
                ctx.input_count = len(flat_inputs)
                ctx.execution_names = tuple(execution)
                ctx.save_for_backward(
                    *flat_inputs,
                    *(execution[name] for name in ctx.execution_names),
                )
                return outputs[0] if len(outputs) == 1 else outputs

            @staticmethod
            def backward(ctx, *output_cotangents):
                saved = ctx.saved_tensors
                flat_inputs = tuple(saved[: ctx.input_count])
                execution = dict(
                    zip(
                        ctx.execution_names,
                        saved[ctx.input_count :],
                        strict=True,
                    )
                )
                return compiled.backward_values(
                    flat_inputs, execution, output_cotangents
                )

        return Function.apply(*inputs)


def compile_cpu(
    operator: FoldOperator | None,
    sizes: dict[str, int],
    *,
    plan: ForwardScheduleTemplate | None = None,
    backward: bool = False,
    verbose: bool = False,
    schedule: CpuScheduleConfig | None = None,
    backward_schedule: CpuScheduleConfig | None = None,
    backward_program: ScheduledBackwardProgram | None = None,
    target: CpuTarget | None = None,
    _region_realizations=None,
    _program=None,
    _forward_scheduled=None,
    _backward_scheduled=None,
) -> CompiledCpuFold:
    """Compile size-specialized symbolic regions into a cached C++ extension."""
    from torch.utils.cpp_extension import load_inline

    if _program is None:
        if operator is None:
            raise ValueError("compilation requires an operator or a lowered program")
        problem = operator.lower(backward=backward)
    else:
        problem = _program
    schedule_space = build_schedule_space(problem)
    if backward_program is not None:
        if not backward:
            raise ValueError("a scheduled backward program requires backward=True")
        if problem.requires_prefix or backward_program.threads_prefix:
            raise NotImplementedError(
                "scheduled CPU backward initially supports prefix-free folds"
            )
        for axis in (*backward_program.program_axes, backward_program.loop_axis):
            if axis.axis not in sizes or sizes[axis.axis] != axis.extent:
                raise ValueError(
                    f"scheduled extent for {axis.axis!r} does not match sizes"
                )
    if plan is None and schedule is None:
        from cutilereduce.experimental.planning.analysis import PlanningProblem
        from cutilereduce.experimental.backends.cpu.tuning import (
            compile_cpu_candidate, cpu_analytical_target, candidate_cpu_configs,
        )
        target = target or CpuTarget.detect()
        planning = PlanningProblem(problem, cpu_analytical_target(target), sizes)
        finalists = planning.sweep(backward=backward).shortlist(1, per_family=0)
        if not finalists:
            raise ValueError("no legal CPU schedule for this program and hardware")
        if backward_program is not None or backward_schedule is not None:
            # An explicitly supplied backward implementation is authoritative.
            # Select forward decisions without replacing that implementation.
            selected, schedule, _ = candidate_cpu_configs(planning, finalists[0].candidate)
            if finalists[0].implementation is not None:
                _forward_scheduled = finalists[0].implementation.execution.forward
        elif finalists[0].implementation is not None:
            from .backend import CpuBackend
            return CpuBackend().compile(finalists[0].implementation)
        else:
            return compile_cpu_candidate(operator, planning, finalists[0].candidate, target=target)
    else:
        selected = plan or schedule_space.forward[0]
    if (
        backward
        and problem.requires_prefix
        and selected.name != "partial_scan_checkpoints"
    ):
        raise ValueError(
            "prefix-dependent backward requires partial_scan_checkpoints forward"
        )
    from cutilereduce.experimental.backends.cpu.schedule import schedule_cpu

    forward_scheduled = _forward_scheduled or schedule_cpu(
        problem,
        selected,
        sizes,
        target=target,
        config=schedule,
    )
    scheduled = forward_scheduled
    if backward:
        backward_plan = schedule_space.backward[0]
        training = schedule_space.compose(selected, backward_plan)
        if problem.requires_prefix:
            forward_partition_count = None
            root = forward_scheduled.forward[0].bindings[0].expression
            if isinstance(root, CpuLoop) and root.operation == "map":
                forward_partition_count = root.iteration.count
            configured_chunks = (
                backward_schedule.chunk_sizes if backward_schedule is not None else ()
            )
            configured_scratch = (
                backward_schedule.scratch_bytes_per_worker
                if backward_schedule is not None
                else None
            )
            configured_stages = (
                backward_schedule.stages if backward_schedule is not None else ()
            )
            backward_schedule = CpuScheduleConfig(
                chunk_sizes=configured_chunks,
                partition_count=forward_partition_count,
                scratch_bytes_per_worker=configured_scratch,
                stages=configured_stages,
            )
        backward_scheduled = _backward_scheduled or schedule_cpu(
            problem,
            backward_plan,
            sizes,
            target=forward_scheduled.target,
            config=backward_schedule,
        )
        scheduled = CpuScheduledPlan(
            training,
            forward_scheduled.sizes,
            forward_scheduled.target,
            forward_scheduled.forward,
            backward_scheduled.backward,
        )
    if backward and backward_program is None and not problem.requires_prefix:
        if len(problem.spec.fold_axes) != 1:
            raise NotImplementedError(
                "automatic fused CPU backward currently requires one fold axis"
            )
        automatic_config = backward_schedule or CpuScheduleConfig()
        loop_axis = problem.spec.fold_axes[0]
        program_axes = problem.spec.batch_axes
        explicit_tiles = dict(automatic_config.chunk_sizes)
        partition_counts = {axis: 1 for axis in program_axes}
        while (
            math.prod(partition_counts.values()) < scheduled.target.worker_count
            and program_axes
        ):
            axis = max(
                program_axes,
                key=lambda candidate: sizes[candidate] / partition_counts[candidate],
            )
            if partition_counts[axis] >= sizes[axis]:
                break
            partition_counts[axis] = min(sizes[axis], partition_counts[axis] * 2)
        tiles = {
            axis: min(
                sizes[axis],
                explicit_tiles.get(
                    axis,
                    math.ceil(sizes[axis] / partition_counts[axis]),
                ),
            )
            for axis in program_axes
        }
        tiles[loop_axis] = automatic_config.chunk_size(loop_axis, sizes[loop_axis])
        backward_program = schedule_backward_program(
            problem,
            sizes,
            loop_axis=loop_axis,
            program_axes=program_axes,
            tiles=tiles,
        )
    fold_axis = problem.spec.fold_axes[0]
    full_extent = sizes[fold_axis]

    def required_extents(
        expression: CpuExpr,
        extent: int,
    ) -> set[int]:
        if isinstance(expression, CpuLeaf):
            requires_specialized_module = isinstance(
                expression.expression, MapFoldRegion
            ) or (
                isinstance(expression.expression, RegionCall)
                and expression.expression.region.startswith("map_finalize")
            )
            return {extent} if requires_specialized_module else set()
        if not isinstance(expression, CpuLoop):
            return set()
        result: set[int] = set()
        offset = 0
        while offset < extent:
            local = min(expression.iteration.grain, extent - offset)
            result.update(required_extents(expression.body, local))
            offset += local
        return result

    extents = {full_extent}
    for stage in (*scheduled.forward, *scheduled.backward):
        for binding in stage.bindings:
            extents.update(required_extents(binding.expression, full_extent))
    modules = {}
    compile_start = time.perf_counter()
    for extent in sorted(extents):
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        source = emit_cpp_source(
            problem,
            local_sizes,
            region_realizations=_region_realizations,
            backward=backward,
            scheduled_backward=(backward_program if extent == full_extent else None),
            scheduled_forward=(scheduled if extent == full_extent else None),
            forward_extents=(tuple(sorted(extents)) if extent == full_extent else ()),
        )
        extra_cflags, extra_ldflags = _cpu_build_flags(scheduled_backward=backward_program is not None)
        key = json.dumps(
            {
                "source": source,
                "cflags": extra_cflags,
                "ldflags": extra_ldflags,
                "sizes": sorted(local_sizes.items()),
                "target": scheduled.target.architecture,
            },
            sort_keys=True,
        ).encode()
        name = "cutilereduce_cpu_" + hashlib.sha256(key).hexdigest()[:16]
        modules[extent] = load_inline(
            name=name,
            cpp_sources=source,
            functions=None,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=False,
            verbose=verbose,
        )
    specialization_template = scheduled.source
    specialized = SpecializedSchedule(
        program=problem,
        template=specialization_template,
        target=ExecutionTarget(
            "cpu",
            frozenset(
                {"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}
            ),
        ),
        shape=scheduled.sizes,
        parameters=tuple(schedule.chunk_sizes) if schedule is not None else (),
        implementation=scheduled,
    )
    return CompiledCpuFold(
        operator,
        problem,
        scheduled,
        modules,
        time.perf_counter() - compile_start,
        backward_program,
        specialized,
    )


__all__ = ["CompiledCpuFold", "compile_cpu"]
