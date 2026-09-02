from __future__ import annotations

from dataclasses import dataclass

import torch

from cutilereduce.fold.commutative import tune_commutative_fold_plan
from cutilereduce.fold.commutative.plan import (
    full_fold_plan as commutative_full_fold_plan,
    partial_fold_plan as commutative_partial_fold_plan,
)
from cutilereduce.fold.general import tune_general_fold_plan
from cutilereduce.fold.general.plan import (
    full_fold_plan as general_full_fold_plan,
    partial_fold_plan as general_partial_fold_plan,
)
from cutilereduce.fold.impl import FoldFunctions, mk_fold_autograd, mk_fold_forward
from cutilereduce.fold.plan import AlgebraKind, FoldPlan, FoldSpec, StageSchedule
from cutilereduce.fold.persistence import load_fold_plan, save_fold_plan


@dataclass(frozen=True)
class FoldOperator:
    """Host-side convenience wrapper for planning and building a fold."""

    spec: FoldSpec
    functions: FoldFunctions

    def _schedule(
            self,
            sizes,
            tiles,
            *,
            programs=None,
            loop=None,
            extra_extents=None,
            ):
        complete_tiles = {
            axis.name: sizes[axis.name]
            for axis in self.spec.axes
        }
        complete_tiles.update(tiles)
        return StageSchedule.make(
            self.spec,
            extents=dict(sizes) | dict(extra_extents or {}),
            tiles=complete_tiles,
            programs=programs,
            loop=loop,
        )

    def plan(
            self,
            sizes,
            *,
            path="full",
            tiles,
            partitions=1,
            combine_tiles=None,
            backward=True,
            backward_tiles=None,
            backward_loop=None,
            checkpointed_backward=False,
            ) -> FoldPlan:
        """Construct a deterministic plan without analytical or empirical tuning."""
        if path not in ("full", "partial"):
            raise ValueError(f"unknown fold path: {path!r}")
        if partitions <= 0:
            raise ValueError("partitions must be positive")
        if path == "full" and partitions != 1:
            raise ValueError("partitions applies only to the partial path")
        if checkpointed_backward and not (
                self.spec.algebra == AlgebraKind.general and path == "partial"
                ):
            raise ValueError(
                "checkpointed backward requires a general partial-fold plan"
            )

        partition_axis = self.spec.fold.partition_axis
        forward_programs = (
            {partition_axis: partitions} if path == "partial" else None
        )
        forward_schedule = self._schedule(
            sizes,
            tiles,
            programs=forward_programs,
            loop=self.spec.fold,
        )
        backward_schedule = None
        if backward:
            backward_programs = (
                {partition_axis: partitions}
                if checkpointed_backward else None
            )
            backward_schedule = self._schedule(
                sizes,
                backward_tiles or tiles,
                programs=backward_programs,
                loop=backward_loop or self.spec.fold,
            )

        if path == "full":
            constructor = (
                commutative_full_fold_plan
                if self.spec.algebra == AlgebraKind.commutative
                else general_full_fold_plan
            )
            return constructor(
                self.spec,
                forward_schedule,
                backward_schedule=backward_schedule,
            )

        combine_tile_overrides = {
            axis.name: forward_schedule.tile(axis)
            for axis in self.spec.batch
        }
        combine_tile_overrides.update(combine_tiles or {})
        combine_tile_overrides[partition_axis] = 1
        combine_schedule = self._schedule(
            sizes,
            combine_tile_overrides,
            programs={partition_axis: 1},
            loop=partition_axis,
            extra_extents={partition_axis: partitions},
        )
        if self.spec.algebra == AlgebraKind.commutative:
            return commutative_partial_fold_plan(
                self.spec,
                forward_schedule,
                combine_schedule,
                backward_schedule=backward_schedule,
            )
        if self.spec.algebra == AlgebraKind.general:
            return general_partial_fold_plan(
                self.spec,
                forward_schedule,
                combine_schedule,
                backward_schedule=backward_schedule,
                checkpointed_backward=checkpointed_backward,
            )
        raise ValueError(f"unsupported fold algebra: {self.spec.algebra}")

    def tune(
            self,
            sizes,
            candidate_count=20,
            timeout=0,
            *,
            hardware,
            backward=True,
            max_tile=128,
            max_partition_count=4,
            quiet=False,
            ) -> FoldPlan:
        if self.spec.algebra == AlgebraKind.commutative:
            if not backward:
                raise ValueError(
                    "forward-only commutative tuning is not currently supported"
                )
            return tune_commutative_fold_plan(
                self.spec,
                sizes,
                candidate_count,
                timeout,
                functions=self.functions,
                hardware=hardware,
                quiet_tuning=quiet,
            )
        if self.spec.algebra == AlgebraKind.general:
            return tune_general_fold_plan(
                self.spec,
                sizes,
                candidate_count,
                timeout,
                functions=self.functions,
                hardware=hardware,
                max_tile=max_tile,
                max_partition_count=max_partition_count,
                quiet_tuning=quiet,
                backward=backward,
            )
        raise ValueError(f"unsupported fold algebra: {self.spec.algebra}")

    def build(
            self,
            plan: FoldPlan,
            *,
            backward=True,
            torch_compile=False,
            device="cuda",
            ):
        function = (
            mk_fold_autograd(plan, self.functions, device=device)
            if backward else mk_fold_forward(plan, self.functions, device=device)
        )
        if torch_compile:
            if not backward:
                raise ValueError("torch.compile currently requires an autograd fold")
            function = torch.compile(
                function,
                fullgraph=True,
                mode="reduce-overhead",
            )
        return function

    def save_plan(self, plan: FoldPlan, path, *, metadata=None) -> None:
        if plan.spec is not self.spec and plan.spec != self.spec:
            raise ValueError("plan was created for a different fold specification")
        save_fold_plan(plan, path, metadata=metadata)

    def load_plan(self, path, sizes=None) -> FoldPlan:
        plan = load_fold_plan(self.spec, path)
        if sizes is not None:
            expected = {self.spec.axis_id(name): size for name, size in sizes.items()}
            for stage in (*plan.forward, *plan.backward):
                for axis in stage.domain.compute_axes:
                    if axis.id in expected and axis.extent != expected[axis.id]:
                        raise ValueError(
                            f"persisted extent for {axis.name!r} is {axis.extent}, "
                            f"but the requested size is {expected[axis.id]}"
                        )
        return plan


__all__ = ["FoldOperator"]
