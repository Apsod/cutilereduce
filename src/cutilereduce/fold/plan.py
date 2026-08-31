from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from cutilereduce.core.axis import Axis, AxisId, Axes, LogicalAxis
from cutilereduce.core.buffer import (
    BufferBundle,
    BufferSpec,
    Input,
    Internal,
    Output,
    bundle_spec,
)
from cutilereduce.core.work import WorkModel
from cutilereduce.stages import BuiltStage, StageSchedule, resolve_axis_id


class AlgebraKind(Enum):
    commutative = "commutative"
    general = "general"


def _axes(spec: str | Axes) -> Axes:
    if isinstance(spec, Axes):
        return spec
    return Axes.make(spec)


@dataclass(frozen=True)
class FoldSpec:
    input: BufferBundle
    execution: BufferBundle
    output: BufferBundle
    batch: Axes
    fold: Axis
    map_fold_work: WorkModel = WorkModel()
    combine_work: WorkModel = WorkModel()
    algebra: AlgebraKind = AlgebraKind.commutative

    @property
    def grad_storage(self) -> BufferBundle:
        return self.input.as_grad()

    @property
    def axes(self) -> Axes:
        ret = self.batch | Axes(values=(self.fold,))
        for bundle in (self.input, self.execution, self.output):
            for buffer in bundle:
                ret = ret | buffer.axes
        return ret

    def check(self) -> None:
        if self.fold in self.batch:
            raise ValueError(f"fold axis is also a batch axis: {self.fold}")
        invalid = tuple(b.id for b in self.output if self.fold in b.axes)
        if invalid:
            raise ValueError(f"fold outputs must not depend on fold axis: {invalid}")

    def axis_id(self, key: str | Axis | AxisId) -> AxisId:
        return resolve_axis_id(self, key)


def make_fold_spec(
        *,
        input: Mapping[str, BufferSpec],
        execution: Mapping[str, BufferSpec],
        output: Mapping[str, BufferSpec],
        batch: str | Axes,
        fold: str | Axis,
        map_fold_work: WorkModel = WorkModel(),
        combine_work: WorkModel = WorkModel(),
        algebra: AlgebraKind = AlgebraKind.commutative,
        ) -> FoldSpec:
    batch_axes = _axes(batch)
    fold_axis = LogicalAxis.make(fold) if isinstance(fold, str) else fold
    spec = FoldSpec(
        input=bundle_spec(Input, **dict(input)),
        execution=bundle_spec(Internal("execution"), **dict(execution)),
        output=bundle_spec(Output, **dict(output)),
        batch=batch_axes,
        fold=fold_axis,
        map_fold_work=map_fold_work,
        combine_work=combine_work,
        algebra=algebra,
    )
    spec.check()
    return spec


FoldSchedule = StageSchedule
FoldStage = BuiltStage


@dataclass(frozen=True)
class FoldPlan:
    spec: FoldSpec
    stages: tuple[FoldStage, ...]

    @classmethod
    def make(cls, spec: FoldSpec, stages: tuple[FoldStage, ...]) -> FoldPlan:
        spec.check()
        return cls(spec=spec, stages=tuple(stages))

    @property
    def forward(self) -> tuple[FoldStage, ...]:
        return self.stages


__all__ = [
    "AlgebraKind",
    "FoldPlan",
    "FoldSchedule",
    "FoldSpec",
    "FoldStage",
    "StageSchedule",
    "make_fold_spec",
]
