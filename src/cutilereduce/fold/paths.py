"""First-class forward and backward fold plans.

Each path owns its stage sequence, materialized buffers, lowering, and
recognition of the corresponding persisted stage sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import sympy

from cutilereduce.core.axis import Axes, LogicalAxis
from cutilereduce.fold.plan import AlgebraKind, FoldSpec
from cutilereduce.stages import StageSchedule
from cutilereduce.stages.embed import Embed, embed_axes
from cutilereduce.stages.map_fold import MapFold, MapFoldPartial, partial_buffers


def symbol(name):
    return sympy.Symbol(name, integer=True, positive=True)


def sizes_for(spec):
    return {axis.id: symbol(axis.name.upper()) for axis in spec.axes}


def tiles_for(spec, prefix, sizes):
    return {
        axis.id: symbol(f"{prefix}_{axis.name}_tile")
        if axis in (*spec.batch, spec.fold) else sizes[axis.id]
        for axis in spec.axes
    }


def schedule(sizes, tiles, *, programs=None, loop=None, extra=None):
    return StageSchedule(
        extents=sizes | (extra or {}), tiles=tiles,
        programs=programs or {}, loop=loop,
    )


def symbolic_embed(spec, sizes):
    tiles = {
        axis.id: symbol(f"embed_{axis.name}_tile") if axis in spec.batch else sizes[axis.id]
        for axis in embed_axes(spec)
    }
    return Embed(spec, schedule(sizes, tiles)).build()


def stage_kinds(records):
    return tuple(record["kind"] for record in records)


class ForwardPath:
    spec: FoldSpec

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def stages(self):
        raise NotImplementedError

    def lower(self, schedules):
        raise NotImplementedError

    @cached_property
    def template(self):
        from cutilereduce.fold.templates import FoldPlanTemplate
        return FoldPlanTemplate(self.spec, forward_plan=self)

    @property
    def estimated_time(self):
        return self.template.estimated_time

    @property
    def stage_costs(self):
        return self.template.stage_costs

    @property
    def parameters(self):
        return self.template.parameters

    def evaluate(self, values=None, *, hardware=None):
        return self.template.evaluate(values, hardware=hardware)

    def bind(self, values):
        return self.template.bind(values)


class BackwardPath:
    spec: FoldSpec
    required_forward_plan: ClassVar[type[ForwardPath] | None] = None

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def stages(self):
        raise NotImplementedError

    def compatible(self, forward):
        if forward is None:
            return self.required_forward_plan is None
        return forward.spec == self.spec and (
            self.required_forward_plan is None
            or isinstance(forward, self.required_forward_plan)
        )

    def lower(self, schedules, forward_path, forward_stages):
        raise NotImplementedError

    @cached_property
    def template(self):
        from cutilereduce.fold.templates import FoldPlanTemplate
        return FoldPlanTemplate(self.spec, backward_plan=self)

    @property
    def estimated_time(self):
        return self.template.estimated_time

    @property
    def stage_costs(self):
        return self.template.stage_costs

    @property
    def parameters(self):
        return self.template.parameters

    def evaluate(self, values=None, *, hardware=None):
        return self.template.evaluate(values, hardware=hardware)


@dataclass(frozen=True)
class FullForwardPlan(ForwardPath):
    spec: FoldSpec
    name = "full"

    @cached_property
    def stages(self):
        sizes = sizes_for(self.spec)
        return (MapFold(
            self.spec,
            schedule(sizes, tiles_for(self.spec, "full", sizes), loop=self.spec.fold.id),
        ).build(),)

    def lower(self, schedules):
        return (MapFold(self.spec, schedules[0]).build(),)

    @classmethod
    def from_record(cls, spec, records):
        return cls(spec) if stage_kinds(records) == ("map_fold",) else None


@dataclass(frozen=True)
class _PartialForwardPlan(ForwardPath):
    spec: FoldSpec
    name = "partial"

    def combine(self, partial, combine_schedule):
        raise NotImplementedError

    @cached_property
    def stages(self):
        sizes = sizes_for(self.spec)
        partition_axis = self.spec.fold.partition_axis
        count = symbol("partition_count")
        partial = MapFoldPartial.make(
            self.spec,
            schedule(
                sizes, tiles_for(self.spec, "partial", sizes),
                programs={partition_axis.id: count}, loop=self.spec.fold.id,
            ),
        )
        combine_tiles = {
            axis.id: symbol(f"combine_{axis.name}_tile") if axis in self.spec.batch else sizes[axis.id]
            for axis in self.spec.axes if axis != self.spec.fold
        } | {partition_axis.id: 1}
        combine_schedule = schedule(
            sizes, combine_tiles, programs={partition_axis.id: 1},
            loop=partition_axis.id, extra={partition_axis.id: count},
        )
        return (partial.build(), self.combine(partial, combine_schedule))

    def lower(self, schedules):
        partial = MapFoldPartial.make(self.spec, schedules[0])
        return (partial.build(), self.combine(partial, schedules[1]))


@dataclass(frozen=True)
class CommutativePartialForwardPlan(_PartialForwardPlan):
    def combine(self, partial, combine_schedule):
        from cutilereduce.stages.fold import Fold
        return Fold(
            self.spec, combine_schedule, partial.partition_axis, partial.partials,
        ).build()

    @classmethod
    def from_record(cls, spec, records):
        return cls(spec) if spec.algebra == AlgebraKind.commutative and stage_kinds(records) == (
            "map_fold_partial", "fold",
        ) else None


@dataclass(frozen=True)
class GeneralPartialForwardPlan(_PartialForwardPlan):
    def combine(self, partial, combine_schedule):
        from cutilereduce.stages.scan import Scan
        return Scan.make(
            self.spec, combine_schedule, scan_axis=partial.partition_axis,
            inputs=partial.partials, outputs=self.spec.semantic, exclusive=True,
        ).build()

    @classmethod
    def from_record(cls, spec, records):
        return cls(spec) if spec.algebra == AlgebraKind.general and stage_kinds(records) == (
            "map_fold_partial", "scan",
        ) else None


@dataclass(frozen=True)
class CommutativeBackwardPlan(BackwardPath):
    spec: FoldSpec
    loop_axis: LogicalAxis

    @property
    def name(self):
        return "commutative"

    @property
    def loop_axes(self):
        axes = Axes(values=(*self.spec.batch, self.spec.fold))
        return tuple(self.spec.grad_storage.contention_axes(axes))

    def with_loop(self, axis):
        selected = next((candidate for candidate in self.loop_axes
                         if candidate == axis or candidate.name == axis), None)
        if selected is None:
            raise ValueError(f"invalid commutative backward loop axis {axis!r}")
        return CommutativeBackwardPlan(self.spec, selected)

    @cached_property
    def stages(self):
        from cutilereduce.fold.commutative.plan import commutative_backward_stage

        sizes = sizes_for(self.spec)
        group_axis = self.loop_axis.partition_axis
        finalize = commutative_backward_stage(
            self.spec,
            schedule(
                sizes, tiles_for(self.spec, "backward", sizes),
                programs={group_axis.id: symbol("backward_group_count")},
                loop=self.loop_axis.id,
            ),
            partition_axis=group_axis,
        )
        return (symbolic_embed(self.spec, sizes), finalize)

    def lower(self, schedules, forward_path, forward_stages):
        del forward_path, forward_stages
        return (Embed(self.spec, schedules[0]).build(), self.finalize(schedules[1]))

    def finalize(self, finalize_schedule):
        from cutilereduce.fold.commutative.plan import commutative_backward_stage

        group_axis = self.loop_axis.partition_axis
        groups = finalize_schedule.program(group_axis)
        return commutative_backward_stage(
            self.spec, finalize_schedule,
            partition_axis=group_axis if groups > 1 else None,
        )

    @classmethod
    def from_record(cls, spec, records):
        if spec.algebra != AlgebraKind.commutative or stage_kinds(records) != (
            "embed", "recompute_map_finalize_grad_write",
        ):
            return None
        loop = records[1]["loop"]
        if loop is None:
            raise ValueError("commutative backward stage requires a loop axis")
        axis = next((a for a in (*spec.batch, spec.fold)
                     if loop == f"{a.id.tag.value}:{a.name}"), None)
        if axis is None:
            raise ValueError(f"invalid commutative backward loop axis {loop!r}")
        return cls(spec, axis)


@dataclass(frozen=True)
class GeneralFullBackwardPlan(BackwardPath):
    spec: FoldSpec
    name = "full_recompute"

    @cached_property
    def stages(self):
        from cutilereduce.fold.general.plan import full_recompute_backward_stage

        sizes = sizes_for(self.spec)
        finalize = full_recompute_backward_stage(
            self.spec,
            schedule(sizes, tiles_for(self.spec, "backward", sizes), loop=self.spec.fold.id),
        )
        return (symbolic_embed(self.spec, sizes), finalize)

    def lower(self, schedules, forward_path, forward_stages):
        del forward_path, forward_stages
        return (
            Embed(self.spec, schedules[0]).build(),
            self.finalize(schedules[1]),
        )

    def finalize(self, finalize_schedule):
        from cutilereduce.fold.general.plan import full_recompute_backward_stage
        return full_recompute_backward_stage(self.spec, finalize_schedule)

    @classmethod
    def from_record(cls, spec, records):
        return cls(spec) if spec.algebra == AlgebraKind.general and stage_kinds(records) == (
            "embed", "recompute_fold_map_finalize_grad_write",
        ) else None


@dataclass(frozen=True)
class GeneralCheckpointedBackwardPlan(BackwardPath):
    spec: FoldSpec
    name = "checkpointed"
    required_forward_plan: ClassVar[type[ForwardPath] | None] = GeneralPartialForwardPlan

    @cached_property
    def stages(self):
        from cutilereduce.fold.general.plan import prefix_recompute_backward_stage

        sizes = sizes_for(self.spec)
        partition_axis = self.spec.fold.partition_axis
        tiles = tiles_for(self.spec, "backward", sizes) | {
            self.spec.fold.id: symbol(f"partial_{self.spec.fold.name}_tile")
        }
        finalize = prefix_recompute_backward_stage(
            self.spec,
            schedule(
                sizes, tiles, programs={partition_axis.id: symbol("partition_count")},
                loop=self.spec.fold.id,
            ),
            checkpoints=partial_buffers(self.spec, partition_axis),
            partition_axis=partition_axis,
        )
        return (symbolic_embed(self.spec, sizes), finalize)

    def lower(self, schedules, forward_path, forward_stages):
        if not self.compatible(forward_path):
            raise ValueError("checkpointed backward requires partial general forward")
        return (
            Embed(self.spec, schedules[0]).build(),
            self.finalize(schedules[1], forward_stages[1].carriers),
        )

    def finalize(self, finalize_schedule, checkpoints):
        from cutilereduce.fold.general.plan import prefix_recompute_backward_stage
        return prefix_recompute_backward_stage(
            self.spec, finalize_schedule, checkpoints=checkpoints,
            partition_axis=self.spec.fold.partition_axis,
        )

    @classmethod
    def from_record(cls, spec, records):
        return cls(spec) if spec.algebra == AlgebraKind.general and stage_kinds(records) == (
            "embed", "recompute_prefix_fold_map_finalize_grad_write",
        ) else None


def forward_plans(spec):
    partial = (
        CommutativePartialForwardPlan(spec)
        if spec.algebra == AlgebraKind.commutative
        else GeneralPartialForwardPlan(spec)
    )
    return (FullForwardPlan(spec), partial)


def backward_plans(spec):
    if not spec.embedded:
        return ()
    if spec.algebra == AlgebraKind.commutative:
        axes = Axes(values=(*spec.batch, spec.fold))
        eligible = tuple(spec.grad_storage.contention_axes(axes))
        if not eligible:
            return ()
        default = next((axis for axis in eligible if axis == spec.fold), eligible[0])
        return (CommutativeBackwardPlan(spec, default),)
    return (GeneralFullBackwardPlan(spec), GeneralCheckpointedBackwardPlan(spec))


FORWARD_RECORD_TYPES = (FullForwardPlan, CommutativePartialForwardPlan, GeneralPartialForwardPlan)
BACKWARD_RECORD_TYPES = (
    CommutativeBackwardPlan, GeneralFullBackwardPlan, GeneralCheckpointedBackwardPlan,
)
