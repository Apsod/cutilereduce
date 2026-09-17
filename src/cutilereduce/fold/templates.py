"""Composition, cost inspection, and binding for first-class fold plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import sympy

from cutilereduce.core.sweep import DEFAULT_FUNCTIONS, DEFAULT_SYMBOLS
from cutilereduce.core.variables import (
    BANDWIDTH, MAX_PROGRAMS_PER_SM, PEAK_TENSOR_FLOPS, SM_COUNT, SMEM_PER_SM,
)
from cutilereduce.fold.paths import (
    CommutativeBackwardPlan, CommutativePartialForwardPlan,
    GeneralPartialForwardPlan, ForwardPath, BackwardPath,
    backward_plans, forward_plans,
)
from cutilereduce.fold.plan import FoldPlan, FoldSpec
from cutilereduce.stages import StageSchedule
from cutilereduce.stages.embed import embed_axes
from cutilereduce.fold.plan import AlgebraKind


_COST_SYMBOLS = set(DEFAULT_SYMBOLS) | {
    BANDWIDTH, MAX_PROGRAMS_PER_SM, PEAK_TENSOR_FLOPS, SM_COUNT, SMEM_PER_SM,
}


def _normalize_values(spec: FoldSpec, values: Mapping):
    axis_names = {axis.name for axis in spec.axes}
    normalized = {}
    for key, value in values.items():
        name = str(key)
        if name in axis_names:
            name = name.upper()
        if name in normalized and normalized[name] != value:
            raise ValueError(f"conflicting values for plan parameter {name!r}")
        normalized[name] = value
    return normalized


@dataclass(frozen=True)
class FoldPlanTemplate:
    spec: FoldSpec
    forward_plan: ForwardPath | None = None
    backward_plan: BackwardPath | None = None

    def __post_init__(self):
        if self.forward_plan is None and self.backward_plan is None:
            raise ValueError("a plan needs a forward or backward path")
        if self.forward_plan is not None and self.forward_plan.spec != self.spec:
            raise ValueError("forward path belongs to a different spec")
        if self.backward_plan is not None:
            if self.backward_plan.spec != self.spec:
                raise ValueError("backward path belongs to a different spec")
            if self.forward_plan is not None and not self.backward_plan.compatible(self.forward_plan):
                raise ValueError(f"{self.backward_plan.name} backward is incompatible with {self.forward_plan.name} forward")

    @property
    def name(self):
        return "+".join(path.name for path in (self.forward_plan, self.backward_plan) if path is not None)

    @property
    def forward(self):
        return () if self.forward_plan is None else self.forward_plan.stages

    @property
    def backward(self):
        return () if self.backward_plan is None else self.backward_plan.stages

    @property
    def stages(self):
        return (*self.forward, *self.backward)

    @cached_property
    def stage_costs(self):
        return tuple((stage.stage.name, stage.stage.cost) for stage in self.stages)

    @cached_property
    def forward_estimated_time(self):
        return sum(stage.stage.cost.estimated_time for stage in self.forward)

    @cached_property
    def backward_estimated_time(self):
        return sum(stage.stage.cost.estimated_time for stage in self.backward)

    @cached_property
    def estimated_time(self):
        return self.forward_estimated_time + self.backward_estimated_time

    @cached_property
    def parameters(self):
        symbols = set(self.estimated_time.free_symbols)
        for stage in self.stages:
            for axis in stage.domain.compute_axes:
                for value in (axis.extent, axis.tile):
                    symbols.update(getattr(value, "free_symbols", ()))
            for axis in stage.domain.program_axes:
                symbols.update(getattr(axis.programs, "free_symbols", ()))
        return {str(symbol): symbol for symbol in symbols if symbol not in _COST_SYMBOLS}

    def evaluate(self, values: Mapping | None = None, *, hardware: Mapping | None = None):
        """Substitute known values, leaving all other parameters symbolic."""
        expression = self.estimated_time
        for function, replacement in DEFAULT_FUNCTIONS.items():
            expression = expression.replace(function, replacement)
        substitutions = DEFAULT_SYMBOLS | dict(hardware or {})
        substitutions.update({
            self.parameters.get(name, sympy.Symbol(name)): value
            for name, value in _normalize_values(self.spec, values or {}).items()
        })
        return expression.subs(substitutions)

    def bind(self, values: Mapping) -> FoldPlan:
        """Lower fully specified paths to an executable plan."""
        if self.backward_plan and not self.backward_plan.compatible(self.forward_plan):
            raise ValueError("backward plan requires a compatible forward path")
        supplied = _normalize_values(self.spec, values)
        required = set(self.parameters)
        missing = required - supplied.keys()
        if missing:
            raise ValueError(f"missing plan parameters: {sorted(missing)}")
        if any(type(supplied[name]) is not int or supplied[name] <= 0 for name in required):
            raise ValueError("plan sizes, tiles, and program counts must be positive integers")
        if isinstance(self.forward_plan, (CommutativePartialForwardPlan, GeneralPartialForwardPlan)):
            if supplied["partition_count"] < 2:
                raise ValueError("partial plans require at least two partitions")
            fold_tile = supplied[f"partial_{self.spec.fold.name}_tile"]
            fold_extent = supplied[self.spec.fold.name.upper()]
            if supplied["partition_count"] > (fold_extent + fold_tile - 1) // fold_tile:
                raise ValueError("partition count exceeds fold tile count")
        if isinstance(self.backward_plan, CommutativeBackwardPlan):
            loop = self.backward_plan.loop_axis.name
            if supplied["backward_group_count"] > (
                supplied[loop.upper()] + supplied[f"backward_{loop}_tile"] - 1
            ) // supplied[f"backward_{loop}_tile"]:
                raise ValueError("backward group count exceeds loop tile count")

        def concrete(value):
            if isinstance(value, sympy.Expr):
                value = value.subs({symbol: supplied[str(symbol)] for symbol in value.free_symbols})
            return int(value)

        def bound_schedule(stage):
            domain = stage.domain
            return StageSchedule(
                extents={axis.id: concrete(axis.extent) for axis in domain.compute_axes},
                tiles={axis.id: concrete(axis.tile) for axis in domain.compute_axes},
                programs={axis.id: concrete(axis.programs) for axis in domain.program_axes},
                loop=domain.loop,
            )

        forward = () if self.forward_plan is None else self.forward_plan.lower(
            tuple(bound_schedule(stage) for stage in self.forward)
        )
        backward = () if self.backward_plan is None else self.backward_plan.lower(
            tuple(bound_schedule(stage) for stage in self.backward),
            self.forward_plan, forward,
        )
        return FoldPlan.make(self.spec, forward, backward)


def forward_plan_templates(spec: FoldSpec):
    return forward_plans(spec)


def backward_plan_templates(spec: FoldSpec):
    return backward_plans(spec)


def compose_plans(forward, backward):
    """Compose compatible first-class forward and backward paths."""
    return FoldPlanTemplate(forward.spec, forward, backward)


def joint_plan_templates(spec: FoldSpec):
    return tuple(
        compose_plans(forward, backward)
        for forward in forward_plans(spec)
        for backward in backward_plans(spec)
        if backward.compatible(forward)
    )


def plan_fold(
        spec, sizes, *, path="full", tiles, partitions=1, combine_tiles=None,
        backward=True, backward_tiles=None, backward_loop=None,
        backward_groups=1, embed_tiles=None, checkpointed_backward=False,
        ):
    """Convenience lowering for one selected path and concrete tile mapping."""
    if path not in ("full", "partial"):
        raise ValueError(f"unknown fold path: {path!r}")
    if path == "full" and partitions != 1:
        raise ValueError("partitions applies only to the partial path")
    if checkpointed_backward and not (
        spec.algebra == AlgebraKind.general and path == "partial" and backward
    ):
        raise ValueError("checkpointed backward requires a general partial-fold plan")
    if backward and not spec.embedded:
        raise ValueError("backward requires declared embedded buffers")
    if backward:
        if spec.algebra == AlgebraKind.commutative:
            loop = backward_loop or spec.fold.name
            loop_name = loop if isinstance(loop, str) else loop.name
            backward_kind = "commutative"
        else:
            if backward_loop is not None and backward_loop not in (spec.fold, spec.fold.name):
                raise ValueError("general backward plans loop over the fold axis")
            backward_kind = "checkpointed" if checkpointed_backward else "full_recompute"
    else:
        backward_kind = None
    forward_path = next(plan for plan in forward_plans(spec) if plan.name == path)
    if backward_kind is None:
        backward_path = None
    else:
        backward_path = next((plan for plan in backward_plans(spec)
                              if plan.name == backward_kind), None)
        if backward_path is None:
            raise ValueError(f"unsupported backward path: {backward_kind}")
        if backward_kind == "commutative":
            backward_path = backward_path.with_loop(loop_name)
    template = FoldPlanTemplate(spec, forward_path, backward_path)
    params = template.parameters
    values = {axis.name.upper(): sizes[axis.name] for axis in spec.axes}
    forward_tiles = dict(tiles)
    backward_tiles = dict(backward_tiles or tiles)
    combine_tiles = dict(combine_tiles or {})
    embed_tiles = dict(embed_tiles or {})
    for axis in spec.axes:
        name = axis.name
        for prefix, mapping in (
            ("full", forward_tiles), ("partial", forward_tiles),
            ("backward", backward_tiles),
        ):
            key = f"{prefix}_{name}_tile"
            if key in params:
                values[key] = mapping.get(name, sizes[name])
        key = f"combine_{name}_tile"
        if key in params:
            values[key] = combine_tiles.get(name, forward_tiles.get(name, sizes[name]))
    if path == "partial":
        values["partition_count"] = partitions
    if backward_kind == "commutative":
        values["backward_group_count"] = backward_groups
    budget = 128
    for axis in reversed(tuple(embed_axes(spec))):
        if axis not in spec.batch:
            continue
        key = f"embed_{axis.name}_tile"
        if key in params:
            tile = min(1 << (sizes[axis.name] - 1).bit_length(), budget)
            values[key] = embed_tiles.get(axis.name, tile)
            budget = max(1, budget // tile)
    return template.bind(values)


__all__ = [
    "FoldPlanTemplate", "forward_plan_templates", "backward_plan_templates",
    "joint_plan_templates", "compose_plans", "plan_fold",
]
