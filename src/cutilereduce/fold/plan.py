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
    map_intermediate: BufferBundle
    map_finalize_intermediate: BufferBundle
    batch: Axes
    fold: Axis
    map_fold_work: WorkModel = WorkModel()
    backward_work: WorkModel | None = None
    combine_work: WorkModel = WorkModel()
    algebra: AlgebraKind = AlgebraKind.commutative

    @property
    def grad_storage(self) -> BufferBundle:
        return self.input.as_grad()

    @property
    def axes(self) -> Axes:
        ret = self.batch | Axes(values=(self.fold,))
        for bundle in (
            self.input,
            self.execution,
            self.output,
            self.map_intermediate,
            self.map_finalize_intermediate,
        ):
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

    def mk_inputs(self, sizes, *, device="cuda", **initializers):
        """Return leaf tensors in spec order, with spec dtypes and grad flags.

        Each initializer receives ``(tensor, sizes)`` and may fill the tensor
        in place (returning None) or return a replacement of the same shape,
        dtype and device. Floating inputs default to normal random values;
        integer inputs require an initializer to choose their valid range.
        Buffer defaults describe kernel padding, not input distributions.
        """
        import operator
        from types import MappingProxyType

        import torch

        sizes = dict(sizes)
        expected = {axis.name for axis in self.axes}
        if sizes.keys() != expected:
            raise ValueError(f"sizes must contain exactly {sorted(expected)}")
        for name, value in sizes.items():
            if isinstance(value, bool):
                raise ValueError(f"size {name!r} must be a positive integer")
            sizes[name] = operator.index(value)
            if sizes[name] <= 0:
                raise ValueError(f"size {name!r} must be positive")
        unknown = initializers.keys() - {buffer.id.name for buffer in self.input}
        if unknown:
            raise ValueError(f"unknown input initializers: {sorted(unknown)}")
        sizes = MappingProxyType(sizes)
        inputs = []
        with torch.no_grad():
            for buffer in self.input:
                tensor = torch.empty(
                    tuple(sizes[axis.name] for axis in buffer.axes),
                    dtype=buffer.torch_dtype, device=device,
                )
                initializer = initializers.get(buffer.id.name)
                if initializer is None:
                    if not tensor.is_floating_point():
                        raise ValueError(f"input {buffer.id.name!r} requires an initializer")
                    tensor.normal_()
                else:
                    result = initializer(tensor, sizes)
                    if result is not None:
                        if not isinstance(result, torch.Tensor) or (
                            result.shape != tensor.shape
                            or result.dtype != tensor.dtype
                            or result.device != tensor.device
                        ):
                            raise ValueError(f"initializer for {buffer.id.name!r} changed shape, dtype or device")
                        tensor = result
                inputs.append(tensor.detach().requires_grad_(buffer.req_grad))
        return tuple(inputs)


def make_fold_spec(
        *,
        input: Mapping[str, BufferSpec],
        execution: Mapping[str, BufferSpec],
        output: Mapping[str, BufferSpec],
        map_intermediate: Mapping[str, BufferSpec] | None = None,
        map_finalize_intermediate: Mapping[str, BufferSpec] | None = None,
        batch: str | Axes,
        fold: str | Axis,
        map_fold_work: WorkModel = WorkModel(),
        backward_work: WorkModel | None = None,
        combine_work: WorkModel = WorkModel(),
        algebra: AlgebraKind = AlgebraKind.commutative,
        ) -> FoldSpec:
    batch_axes = _axes(batch)
    fold_axis = LogicalAxis.make(fold) if isinstance(fold, str) else fold
    spec = FoldSpec(
        input=bundle_spec(Input, **dict(input)),
        execution=bundle_spec(Internal("execution"), **dict(execution)),
        output=bundle_spec(Output, **dict(output)),
        map_intermediate=bundle_spec(
            Internal("map_intermediate"), **dict(map_intermediate or {})
        ),
        map_finalize_intermediate=bundle_spec(
            Internal("map_finalize_intermediate"),
            **dict(map_finalize_intermediate or {}),
        ),
        batch=batch_axes,
        fold=fold_axis,
        map_fold_work=map_fold_work,
        backward_work=backward_work,
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
    backward_stages: tuple[FoldStage, ...] = ()

    @classmethod
    def make(
            cls,
            spec: FoldSpec,
            stages: tuple[FoldStage, ...],
            backward_stages: tuple[FoldStage, ...] = (),
            ) -> FoldPlan:
        spec.check()
        return cls(spec=spec, stages=tuple(stages), backward_stages=tuple(backward_stages))

    @property
    def forward(self) -> tuple[FoldStage, ...]:
        return self.stages

    @property
    def backward(self) -> tuple[FoldStage, ...]:
        return self.backward_stages


__all__ = [
    "AlgebraKind",
    "FoldPlan",
    "FoldSchedule",
    "FoldSpec",
    "FoldStage",
    "StageSchedule",
    "make_fold_spec",
]
