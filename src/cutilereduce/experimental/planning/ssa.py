"""Explicit SSA form shared by target-neutral tiled stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cutilereduce.experimental.ir.core import TensorSignature
from .accounting import _positive, ceildiv
from cutilereduce.experimental.planning.scheduled import (
    CommitMode,
    LocalLifetime,
    ScheduledAxis,
    ScheduledBackwardProgram,
    TileBinding,
)


@dataclass(frozen=True)
class ScheduledValue:
    id: int
    name: str
    signature: TensorSignature


@dataclass(frozen=True)
class ScheduledOperation:
    op: str
    inputs: tuple[ScheduledValue, ...]
    outputs: tuple[ScheduledValue, ...]
    attributes: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class LoopCarry:
    initial: ScheduledValue
    argument: ScheduledValue
    yielded: ScheduledValue
    result: ScheduledValue


@dataclass(frozen=True)
class ScheduledLoop:
    axis: ScheduledAxis
    carries: tuple[LoopCarry, ...]
    body: tuple[ScheduledOperation, ...]


@dataclass(frozen=True)
class AxisPartition:
    """Parallel programs sharing serial tiles of one logical axis."""

    axis: str
    extent: int
    tile: int
    programs: int

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("partition axis name must be non-empty")
        for value in (self.extent, self.tile, self.programs):
            _positive(value, "partition extent/tile/programs")

    @property
    def tile_count(self) -> int:
        return ceildiv(self.extent, self.tile)

    def tile_span(self, program: int) -> tuple[int, int]:
        """Return the balanced half-open tile-index span owned by a program."""
        if not 0 <= program < self.programs:
            raise ValueError(f"program {program} outside [0, {self.programs})")
        base, remainder = divmod(self.tile_count, self.programs)
        start = base * program + min(program, remainder)
        return start, start + base + int(program < remainder)

    def element_span(self, program: int) -> tuple[int, int]:
        start, end = self.tile_span(program)
        element_start = start * self.tile
        return element_start, min(self.extent, end * self.tile)


@dataclass(frozen=True)
class TiledBuffer:
    name: str
    signatures: tuple[tuple[str, TensorSignature], ...]


@dataclass(frozen=True)
class StageDependency:
    producer: str
    consumer: str
    buffer: str


@dataclass(frozen=True)
class TiledStage:
    """One independently launchable tiled stage in explicit SSA form."""

    name: str
    parallel_axes: tuple[ScheduledAxis, ...]
    bindings: tuple[TileBinding, ...]
    parameters: tuple[ScheduledValue, ...]
    initialization: tuple[ScheduledOperation, ...]
    loop: ScheduledLoop | None
    epilogue: tuple[ScheduledOperation, ...]
    commits: tuple[ScheduledOperation, ...]
    partitions: tuple[AxisPartition, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(partition.axis for partition in self.effective_partitions)
        if len(names) != len(set(names)):
            raise ValueError("a tiled stage may partition each logical axis once")
        if self.loop is not None and self.loop.axis.axis not in names:
            raise ValueError("the serial loop axis requires an axis partition")

    @property
    def effective_partitions(self) -> tuple[AxisPartition, ...]:
        if self.partitions:
            return self.partitions
        values = [
            AxisPartition(axis.axis, axis.extent, axis.tile, axis.count)
            for axis in self.parallel_axes
        ]
        if self.loop is not None and self.loop.axis.axis not in {
            value.axis for value in values
        }:
            axis = self.loop.axis
            values.append(AxisPartition(axis.axis, axis.extent, axis.tile, 1))
        return tuple(values)

    @property
    def grid_partitions(self) -> tuple[AxisPartition, ...]:
        return tuple(
            partition
            for partition in self.effective_partitions
            if partition.programs > 1
        )

    @property
    def resident_bindings(self) -> tuple[TileBinding, ...]:
        """Bindings whose tile is invariant across the stage-local loop."""
        if self.loop is None:
            return self.bindings
        axis = self.loop.axis.axis
        return tuple(binding for binding in self.bindings if axis not in binding.axes)

    @property
    def streamed_bindings(self) -> tuple[TileBinding, ...]:
        """Bindings whose tile changes on each stage-local iteration."""
        if self.loop is None:
            return ()
        axis = self.loop.axis.axis
        return tuple(binding for binding in self.bindings if axis in binding.axes)

    @property
    def operations(self) -> tuple[ScheduledOperation, ...]:
        return (
            *self.initialization,
            *(self.loop.body if self.loop is not None else ()),
            *self.epilogue,
            *self.commits,
        )


# Compatibility name for callers which consumed the initial backward-only SSA.
ScheduledSSAProgram = TiledStage


@dataclass(frozen=True)
class TiledPlan:
    stages: tuple[TiledStage, ...]
    buffers: tuple[TiledBuffer, ...] = ()
    dependencies: tuple[StageDependency, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("tiled plan stage names must be unique")
        known_stages = set(names)
        known_buffers = {buffer.name for buffer in self.buffers}
        for dependency in self.dependencies:
            if {dependency.producer, dependency.consumer} - known_stages:
                raise ValueError("stage dependency references an unknown stage")
            if dependency.buffer not in known_buffers:
                raise ValueError("stage dependency references an unknown buffer")


@dataclass(frozen=True)
class SpecializedSchedule:
    """A schedule template bound to one shape and a target implementation.

    The wrapper keeps the originating program, schedule, shape, and lowered
    implementation together so code generation cannot accidentally combine
    unrelated objects.
    """

    program: Any
    template: Any
    target: Any
    shape: tuple[tuple[str, int], ...]
    parameters: tuple[tuple[str, int], ...]
    implementation: Any

    def __post_init__(self) -> None:
        if not getattr(self.target, "name", self.target):
            raise ValueError("specialized schedule target must be non-empty")
        if len(dict(self.shape)) != len(self.shape):
            raise ValueError("specialized schedule shape axes must be unique")
        if any(type(extent) is not int or extent <= 0 for _, extent in self.shape):
            raise ValueError("specialized schedule extents must be positive integers")
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("specialized schedule parameters must be unique")


class _Builder:
    def __init__(self) -> None:
        self.next_id = 0

    def value(self, name: str, axes: tuple[str, ...], dtype: str) -> ScheduledValue:
        value = ScheduledValue(self.next_id, name, TensorSignature(axes, dtype))
        self.next_id += 1
        return value


def lower_backward_stage(program: ScheduledBackwardProgram) -> TiledStage:
    """Turn named accumulation into explicit loop arguments and yielded values."""
    builder = _Builder()
    parameters = tuple(
        builder.value(binding.name, binding.axes, binding.dtype)
        for binding in program.arguments
    )
    initialization = []
    carry_parts = []
    carry_arguments: dict[str, ScheduledValue] = {}
    for gradient in program.gradients:
        if gradient.lifetime is not LocalLifetime.PERSISTENT:
            continue
        initial = builder.value(
            f"{gradient.input_name}.initial", gradient.axes, gradient.dtype
        )
        initialization.append(ScheduledOperation("local_zero", (), (initial,)))
        argument = builder.value(
            f"{gradient.input_name}.iter", gradient.axes, gradient.dtype
        )
        carry_arguments[gradient.input_name] = argument
        carry_parts.append((gradient, initial, argument))

    call_results = {
        gradient.input_name: builder.value(
            gradient.result_name, gradient.axes, gradient.dtype
        )
        for gradient in program.gradients
    }
    body = [
        ScheduledOperation(
            "call_region",
            parameters,
            tuple(call_results[gradient.input_name] for gradient in program.gradients),
            (("region", program.region), ("domain_axes", program.domain_axes)),
        )
    ]
    yielded: dict[str, ScheduledValue] = {}
    for gradient, _, argument in carry_parts:
        value = builder.value(
            f"{gradient.input_name}.next", gradient.axes, gradient.dtype
        )
        body.append(
            ScheduledOperation(
                "add", (argument, call_results[gradient.input_name]), (value,)
            )
        )
        yielded[gradient.input_name] = value
    for gradient in program.gradients:
        if gradient.lifetime is not LocalLifetime.STREAMED:
            continue
        mode = (
            "store"
            if gradient.commit_mode is CommitMode.STORE
            else "atomic_add_relaxed"
        )
        body.append(
            ScheduledOperation(
                mode,
                (call_results[gradient.input_name],),
                (),
                (("destination", gradient.input_name),),
            )
        )

    carries = []
    loop_results: dict[str, ScheduledValue] = {}
    for gradient, initial, argument in carry_parts:
        result = builder.value(
            f"{gradient.input_name}.final", gradient.axes, gradient.dtype
        )
        loop_results[gradient.input_name] = result
        carries.append(
            LoopCarry(initial, argument, yielded[gradient.input_name], result)
        )
    commits = []
    for gradient in program.gradients:
        if gradient.lifetime is not LocalLifetime.PERSISTENT:
            continue
        mode = (
            "store"
            if gradient.commit_mode is CommitMode.STORE
            else "atomic_add_relaxed"
        )
        commits.append(
            ScheduledOperation(
                mode,
                (loop_results[gradient.input_name],),
                (),
                (("destination", gradient.input_name),),
            )
        )
    return TiledStage(
        program.name,
        program.program_axes,
        program.arguments,
        parameters,
        tuple(initialization),
        ScheduledLoop(program.loop_axis, tuple(carries), tuple(body)),
        (),
        tuple(commits),
        tuple(
            AxisPartition(axis.axis, axis.extent, axis.tile, axis.count)
            for axis in program.program_axes
        )
        + (
            AxisPartition(
                program.loop_axis.axis,
                program.loop_axis.extent,
                program.loop_axis.tile,
                program.loop_programs,
            ),
        ),
    )


def scheduled_to_ssa(program: ScheduledBackwardProgram) -> TiledStage:
    """Compatibility spelling for :func:`lower_backward_stage`."""
    return lower_backward_stage(program)


def format_tiled_stage(stage: TiledStage) -> str:
    """Format the common stage structure without target-specific vocabulary."""
    parallel = ", ".join(
        f"{axis.axis}[tile={axis.tile}, programs={axis.programs}]"
        for axis in stage.effective_partitions
        if axis.programs > 1
    )

    def operation(value: ScheduledOperation, indent: str) -> str:
        outputs = ", ".join(f"%{item.name}" for item in value.outputs)
        inputs = ", ".join(f"%{item.name}" for item in value.inputs)
        attributes = " ".join(f"{key}={item}" for key, item in value.attributes)
        assignment = f"{outputs} = " if outputs else ""
        arguments = f"({inputs})" if inputs else ""
        suffix = f" {attributes}" if attributes else ""
        return f"{indent}{assignment}{value.op}{arguments}{suffix}"

    lines = [f"tiled_stage {stage.name} parallel ({parallel}) {{"]
    if stage.loop is not None:
        resident = ", ".join(binding.name for binding in stage.resident_bindings)
        streamed = ", ".join(binding.name for binding in stage.streamed_bindings)
        lines.append(f"  bindings resident ({resident}) streamed ({streamed})")
    lines.extend(operation(value, "  ") for value in stage.initialization)
    if stage.loop is not None:
        carries = ", ".join(f"%{carry.argument.name}" for carry in stage.loop.carries)
        lines.append(
            f"  serial {stage.loop.axis.axis}[tile={stage.loop.axis.tile}, "
            f"within=owned_partition] iter_args({carries}) {{"
        )
        lines.extend(operation(value, "    ") for value in stage.loop.body)
        yielded = ", ".join(f"%{carry.yielded.name}" for carry in stage.loop.carries)
        lines.extend((f"    yield {yielded}", "  }"))
    lines.extend(operation(value, "  ") for value in stage.epilogue)
    lines.extend(operation(value, "  ") for value in stage.commits)
    lines.append("}")
    return "\n".join(lines)


def liveness(program: TiledStage) -> dict[int, tuple[int, int]]:
    """Return inclusive definition/last-use positions for scheduled SSA values."""
    definitions = {value.id: 0 for value in program.parameters}
    last_use = dict(definitions)
    position = 1
    for operation in program.initialization:
        for value in operation.outputs:
            definitions[value.id] = position
            last_use[value.id] = position
        position += 1
    if program.loop is not None:
        for carry in program.loop.carries:
            definitions[carry.argument.id] = position
            last_use[carry.argument.id] = position
            last_use[carry.initial.id] = position
        for operation in program.loop.body:
            for value in operation.inputs:
                last_use[value.id] = position
            for value in operation.outputs:
                definitions[value.id] = position
                last_use[value.id] = position
            position += 1
        for carry in program.loop.carries:
            last_use[carry.yielded.id] = position
            definitions[carry.result.id] = position
            last_use[carry.result.id] = position
        position += 1
    for operation in program.epilogue:
        for value in operation.inputs:
            last_use[value.id] = position
        for value in operation.outputs:
            definitions[value.id] = position
            last_use[value.id] = position
        position += 1
    for operation in program.commits:
        for value in operation.inputs:
            last_use[value.id] = position
        position += 1
    return {
        value_id: (defined, last_use[value_id])
        for value_id, defined in definitions.items()
    }


__all__ = [
    "LoopCarry",
    "AxisPartition",
    "ScheduledLoop",
    "ScheduledOperation",
    "ScheduledSSAProgram",
    "ScheduledValue",
    "SpecializedSchedule",
    "StageDependency",
    "TiledBuffer",
    "TiledPlan",
    "TiledStage",
    "format_tiled_stage",
    "liveness",
    "lower_backward_stage",
    "scheduled_to_ssa",
]
