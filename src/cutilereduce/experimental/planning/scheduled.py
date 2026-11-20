"""Target-neutral scheduled programs below tensor algebra and above codegen.

The tensor IR describes values and contractions.  This module describes the
physical concurrency decisions which tensor operations alone cannot express:
program partitions, a program-local serial loop, resident gradient tiles, and
the way completed tiles are committed to shared output storage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.accounting import contribution_ownership


class LocalLifetime(Enum):
    """Lifetime of a result tile relative to the program-local loop."""

    STREAMED = "streamed"
    PERSISTENT = "persistent"


class CommitMode(Enum):
    """How a completed local tile is committed to shared storage."""

    UNRESOLVED = "unresolved"
    STORE = "store"
    ATOMIC_ADD_RELAXED = "atomic_add_relaxed"


class CommitPoint(Enum):
    """Point at which a local result becomes complete."""

    LOOP_ITERATION = "loop_iteration"
    PROGRAM_END = "program_end"


class ScheduledOp:
    """Base class for visitable scheduled-program operations."""


@dataclass(frozen=True)
class LocalZero(ScheduledOp):
    gradient: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class CallRegion(ScheduledOp):
    region: str
    arguments: tuple[TileBinding, ...]
    results: tuple[str, ...]
    threads_prefix: bool = False


@dataclass(frozen=True)
class Accumulate(ScheduledOp):
    gradient: str
    value: str


@dataclass(frozen=True)
class Store(ScheduledOp):
    gradient: str
    value: str


@dataclass(frozen=True)
class AtomicAdd(ScheduledOp):
    gradient: str
    value: str
    contention_axes: tuple[str, ...]
    memory_order: str = "relaxed"

    def __post_init__(self) -> None:
        if not self.contention_axes:
            raise ValueError("atomic add requires at least one contention axis")
        if self.memory_order != "relaxed":
            raise ValueError(
                "scheduled gradient atomics currently require relaxed order"
            )


@dataclass(frozen=True)
class ScheduledAxis:
    """A disjoint tiling of one logical axis."""

    axis: str
    tile: int
    extent: int

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("scheduled axis name must be non-empty")
        if self.extent <= 0 or self.tile <= 0:
            raise ValueError("scheduled axis extent and tile must be positive")

    @property
    def count(self) -> int:
        return math.ceil(self.extent / self.tile)


@dataclass(frozen=True)
class TileBinding:
    """One region value and the scheduled axes used to slice its tile."""

    name: str
    axes: tuple[str, ...]
    dtype: str
    sliced_axes: tuple[str, ...]


@dataclass(frozen=True)
class SerialLoop(ScheduledOp):
    axis: ScheduledAxis
    body: tuple[ScheduledOp, ...]


@dataclass(frozen=True)
class Parallel(ScheduledOp):
    axes: tuple[ScheduledAxis, ...]
    body: tuple[ScheduledOp, ...]


@dataclass(frozen=True)
class ScheduledGradient:
    """One gradient tile produced by a scheduled backward program."""

    input_name: str
    axes: tuple[str, ...]
    dtype: str
    lifetime: LocalLifetime
    commit_point: CommitPoint
    contention_axes: tuple[str, ...] = ()
    commit_mode: CommitMode = CommitMode.UNRESOLVED

    @property
    def result_name(self) -> str:
        return f"partial_grad_input.{self.input_name}"


@dataclass(frozen=True)
class ScheduledBackwardProgram:
    """A block/thread-program view of one algebraic backward region.

    ``program_axes`` are mutually independent program partitions.  Each
    program executes ``loop_axis`` serially.  Persistent gradients are loop
    carried and committed at program end; streamed gradients are committed as
    each loop tile becomes complete.
    """

    name: str
    region: str
    program_axes: tuple[ScheduledAxis, ...]
    loop_axis: ScheduledAxis
    arguments: tuple[TileBinding, ...]
    gradients: tuple[ScheduledGradient, ...]
    domain_axes: tuple[str, ...] = ()
    threads_prefix: bool = False
    loop_programs: int = 1

    def __post_init__(self) -> None:
        program_names = tuple(axis.axis for axis in self.program_axes)
        if len(set(program_names)) != len(program_names):
            raise ValueError("scheduled program axes must be unique")
        if self.loop_axis.axis in program_names:
            raise ValueError("the local loop axis cannot also be a program axis")
        if self.loop_programs <= 0:
            raise ValueError("loop program count must be positive")

    @property
    def contributions(self):
        """Required reductions before choosing stores, atomics or another implementation."""
        owners = {axis.axis: axis.count for axis in self.program_axes}
        owners[self.loop_axis.axis] = self.loop_programs
        return {g.input_name: contribution_ownership(g.axes, owners, loop_axes=(self.loop_axis.axis,))
                for g in self.gradients}

    @property
    def body(self) -> Parallel:
        """Materialize the declarative schedule as visitable operations."""
        if any(
            gradient.commit_mode is CommitMode.UNRESOLVED for gradient in self.gradients
        ):
            raise ValueError("scheduled writes must be resolved before lowering")
        persistent = tuple(
            gradient
            for gradient in self.gradients
            if gradient.lifetime is LocalLifetime.PERSISTENT
        )
        streamed = tuple(
            gradient
            for gradient in self.gradients
            if gradient.lifetime is LocalLifetime.STREAMED
        )

        def commit(gradient: ScheduledGradient, value: str) -> ScheduledOp:
            if gradient.commit_mode is CommitMode.STORE:
                return Store(gradient.input_name, value)
            if gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED:
                return AtomicAdd(gradient.input_name, value, gradient.contention_axes)
            raise ValueError(f"unsupported commit mode: {gradient.commit_mode}")

        loop_body: tuple[ScheduledOp, ...] = (
            CallRegion(
                self.region,
                self.arguments,
                tuple(gradient.result_name for gradient in self.gradients),
                self.threads_prefix,
            ),
            *(
                Accumulate(gradient.input_name, gradient.result_name)
                for gradient in persistent
            ),
            *(commit(gradient, gradient.result_name) for gradient in streamed),
        )
        body: tuple[ScheduledOp, ...] = (
            *(LocalZero(gradient.input_name, gradient.axes) for gradient in persistent),
            SerialLoop(self.loop_axis, loop_body),
            *(
                commit(gradient, f"local.{gradient.input_name}")
                for gradient in persistent
            ),
        )
        return Parallel(self.program_axes, body)


def _backward_region(problem: FoldProgram):
    region = (
        problem.map_finalize_map_vjp_accumulate
        if problem.requires_prefix
        else problem.map_finalize_map_vjp
    )
    if region is None:
        raise ValueError("fold planning problem has no backward region")
    return region


def build_backward_program(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    loop_axis: str,
    tiles: dict[str, int],
    program_axes: tuple[str, ...] | None = None,
    loop_programs: int = 1,
) -> ScheduledBackwardProgram:
    """Build an unresolved scheduled backward program.

    By default every batch/fold axis except the local loop axis is a program
    axis.  Inner axes remain inside tensor tiles.  Write contention is resolved
    separately by :func:`resolve_writes`.
    """
    problem.spec.check()
    iteration_axes = tuple(
        dict.fromkeys((*problem.spec.batch_axes, *problem.spec.fold_axes))
    )
    if loop_axis not in iteration_axes:
        raise ValueError(
            f"backward loop axis {loop_axis!r} is not a batch or fold axis"
        )
    expected = set(problem.spec.axes)
    if set(sizes) != expected:
        raise ValueError(
            f"sizes must contain exactly {sorted(expected)}, got {sorted(sizes)}"
        )
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all scheduled extents must be positive integers")
    selected = (
        tuple(axis for axis in iteration_axes if axis != loop_axis)
        if program_axes is None
        else program_axes
    )
    if len(set(selected)) != len(selected):
        raise ValueError("program_axes must be unique")
    invalid = set(selected) - set(iteration_axes)
    if invalid:
        raise ValueError(f"program axes are not batch or fold axes: {sorted(invalid)}")
    if loop_axis in selected:
        raise ValueError("the backward loop axis cannot be a program axis")

    def scheduled(axis: str) -> ScheduledAxis:
        tile = tiles.get(axis, sizes[axis])
        return ScheduledAxis(axis, tile, sizes[axis])

    region = _backward_region(problem)
    results = dict(region.results)
    gradients = []
    for name, spec in problem.spec.inputs.items():
        if not spec.differentiable:
            continue
        result_name = f"partial_grad_input.{name}"
        node_id = results.get(result_name)
        if node_id is None:
            continue
        region_name = "map_finalize_map_vjp_accumulate" if problem.requires_prefix else "map_finalize_map_vjp"
        value = problem.region_facts[region_name].result(result_name)
        axes = value.axes
        streamed = value.depends_on((loop_axis,))
        gradients.append(
            ScheduledGradient(
                name,
                axes,
                region.nodes[node_id].dtype,
                LocalLifetime.STREAMED if streamed else LocalLifetime.PERSISTENT,
                CommitPoint.LOOP_ITERATION if streamed else CommitPoint.PROGRAM_END,
            )
        )
    scheduled_axes = (*selected, loop_axis)
    arguments = tuple(
        TileBinding(
            name,
            region.nodes[node_id].axes,
            region.nodes[node_id].dtype,
            tuple(
                axis for axis in scheduled_axes if axis in region.nodes[node_id].axes
            ),
        )
        for name, node_id in region.arguments
    )
    return ScheduledBackwardProgram(
        "map_finalize_map_vjp",
        (
            "map_finalize_map_vjp_accumulate"
            if problem.requires_prefix
            else "map_finalize_map_vjp"
        ),
        tuple(scheduled(axis) for axis in selected),
        scheduled(loop_axis),
        arguments,
        tuple(gradients),
        domain_axes=(*problem.spec.batch_axes, *problem.spec.fold_axes),
        threads_prefix=problem.requires_prefix,
        loop_programs=loop_programs,
    )


def resolve_writes(
    program: ScheduledBackwardProgram, *, shared_commit: CommitMode | None = CommitMode.ATOMIC_ADD_RELAXED,
) -> ScheduledBackwardProgram:
    """Resolve shared commits from program-axis writer multiplicity.

    A program partition contributes to a result without owning a distinct
    destination whenever that program axis is absent from the result axes.
    Only axes with more than one program participate in contention. Pass
    shared_commit=None to retain the reduction requirement without choosing an
    implementation. The existing executors select relaxed atomics explicitly.
    """
    gradients = []
    for gradient in program.gradients:
        ownership = program.contributions[gradient.input_name]
        contention = ownership.owner_axes
        if contention and shared_commit is CommitMode.STORE:
            raise ValueError("shared contributions require a reduction; a plain store would lose contributions")
        mode = (shared_commit or CommitMode.UNRESOLVED) if contention else CommitMode.STORE
        gradients.append(
            replace(
                gradient,
                contention_axes=contention,
                commit_mode=mode,
            )
        )
    return replace(program, gradients=tuple(gradients))


def schedule_backward_program(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    loop_axis: str,
    tiles: dict[str, int],
    program_axes: tuple[str, ...] | None = None,
    loop_programs: int = 1,
) -> ScheduledBackwardProgram:
    """Build a backward program and resolve its shared write policy."""
    return resolve_writes(
        build_backward_program(
            problem,
            sizes,
            loop_axis=loop_axis,
            tiles=tiles,
            program_axes=program_axes,
            loop_programs=loop_programs,
        )
    )


def format_scheduled_program(program: ScheduledBackwardProgram) -> str:
    def operation(op: ScheduledOp, indent: int) -> list[str]:
        prefix = " " * indent
        match op:
            case LocalZero(gradient=gradient, axes=axes):
                return [f"{prefix}%{gradient} = local.zero [{', '.join(axes)}]"]
            case CallRegion(region=region, threads_prefix=threads_prefix):
                suffix = " threads_prefix" if threads_prefix else ""
                return [f"{prefix}call @{region}{suffix}"]
            case Accumulate(gradient=gradient, value=value):
                return [f"{prefix}accumulate %{gradient}, %{value}"]
            case Store(gradient=gradient, value=value):
                return [f"{prefix}store %{value}, @grad_input.{gradient}"]
            case AtomicAdd(
                gradient=gradient,
                value=value,
                contention_axes=contention,
                memory_order=order,
            ):
                axes = ",".join(contention)
                return [
                    f"{prefix}atomic_add {order} %{value}, @grad_input.{gradient} "
                    f"contention={axes}"
                ]
            case SerialLoop(axis=axis, body=body):
                lines = [
                    f"{prefix}loop {axis.axis}[tile={axis.tile}, count={axis.count}] {{"
                ]
                for child in body:
                    lines.extend(operation(child, indent + 2))
                return [*lines, f"{prefix}}}"]
            case Parallel(axes=axes, body=body):
                description = ", ".join(
                    f"{axis.axis}[tile={axis.tile}, count={axis.count}]"
                    for axis in axes
                )
                lines = [f"{prefix}parallel ({description}) {{"]
                for child in body:
                    lines.extend(operation(child, indent + 2))
                return [*lines, f"{prefix}}}"]
            case _:
                raise TypeError(f"unknown scheduled operation: {op!r}")

    lines = [f"scheduled {program.name} {{"]
    lines.extend(operation(program.body, 2))
    lines.append("}")
    return "\n".join(lines)


__all__ = [
    "Accumulate",
    "AtomicAdd",
    "CallRegion",
    "CommitMode",
    "CommitPoint",
    "LocalLifetime",
    "LocalZero",
    "Parallel",
    "ScheduledAxis",
    "ScheduledOp",
    "ScheduledBackwardProgram",
    "ScheduledGradient",
    "build_backward_program",
    "SerialLoop",
    "Store",
    "format_scheduled_program",
    "resolve_writes",
    "schedule_backward_program",
    "TileBinding",
]
