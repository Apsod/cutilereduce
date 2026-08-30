from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import sympy

from cutilereduce.core.kernel_stage import KernelStageCost
from cutilereduce.fold.plan import FoldPlan, FoldStage


@dataclass(frozen=True)
class FoldStageEstimate:
    stage: FoldStage
    cost: KernelStageCost

    def value(self, name: str, substitutions: Mapping[object, object] | None = None):
        value = getattr(self.cost, name)
        if substitutions and isinstance(value, sympy.Expr):
            return value.subs(dict(substitutions))
        return value


@dataclass(frozen=True)
class FoldForwardEstimate:
    stages: tuple[FoldStageEstimate, ...]

    def value(self, name: str, substitutions: Mapping[object, object] | None = None):
        return sum(stage.value(name, substitutions) for stage in self.stages)


def fold_stage_kernel_cost(stage: FoldStage) -> KernelStageCost:
    return stage.stage.cost


def estimate_fold_stage(stage: FoldStage) -> FoldStageEstimate:
    return FoldStageEstimate(stage=stage, cost=fold_stage_kernel_cost(stage))


def estimate_fold_forward(plan: FoldPlan) -> FoldForwardEstimate:
    return FoldForwardEstimate(stages=tuple(estimate_fold_stage(stage) for stage in plan.forward))
