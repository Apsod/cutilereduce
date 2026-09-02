from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cuda.tile as ct

from cutilereduce.core.axis import AxisId, AxisType
from cutilereduce.fold.commutative.plan import (
    full_fold_plan as commutative_full_fold_plan,
    partial_fold_plan as commutative_partial_fold_plan,
)
from cutilereduce.fold.general.plan import (
    full_fold_plan as general_full_fold_plan,
    partial_fold_plan as general_partial_fold_plan,
)
from cutilereduce.fold.plan import AlgebraKind, FoldPlan, FoldSpec
from cutilereduce.stages import StageSchedule


FORMAT = "cutilereduce.fold-plan"
VERSION = 1


def _axis_key(axis: AxisId) -> str:
    return f"{axis.tag.value}:{axis.name}"


def _axis_from_key(value: str) -> AxisId:
    try:
        tag, name = value.split(":", 1)
        return AxisId(AxisType(tag), name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid persisted axis {value!r}") from exc


def _integer(value, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"persisted plans require resolved integer {field}, got {value!r}")
    return value


def _stage_record(stage) -> dict:
    domain = stage.domain
    return {
        "kind": stage.stage.name,
        "extents": {
            _axis_key(axis.id): _integer(axis.extent, "extent")
            for axis in domain.compute_axes
        },
        "tiles": {
            _axis_key(axis.id): _integer(axis.tile, "tile")
            for axis in domain.compute_axes
        },
        "programs": {
            _axis_key(axis.id): _integer(axis.programs, "program count")
            for axis in domain.program_axes
        },
        "loop": None if domain.loop is None else _axis_key(domain.loop),
    }


def _package_version() -> str:
    try:
        return version("cutilereduce")
    except PackageNotFoundError:
        return "unknown"


def plan_record(plan: FoldPlan, *, metadata=None) -> dict:
    """Return the portable, JSON-compatible scheduling part of a fold plan."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "producer": {
            "cutilereduce": _package_version(),
            "cuda_tile": ct.__version__,
        },
        "metadata": dict(metadata or {}),
        "algebra": plan.spec.algebra.value,
        "axes": sorted(_axis_key(axis.id) for axis in plan.spec.axes),
        "forward": [_stage_record(stage) for stage in plan.forward],
        "backward": [_stage_record(stage) for stage in plan.backward],
    }


def save_fold_plan(plan: FoldPlan, path, *, metadata=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_record(plan, metadata=metadata), indent=2) + "\n")


def _schedule(record: dict) -> StageSchedule:
    required = {"kind", "extents", "tiles", "programs", "loop"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"persisted stage is missing fields: {sorted(missing)}")
    return StageSchedule(
        extents={_axis_from_key(k): _integer(v, "extent") for k, v in record["extents"].items()},
        tiles={_axis_from_key(k): _integer(v, "tile") for k, v in record["tiles"].items()},
        programs={_axis_from_key(k): _integer(v, "program count") for k, v in record["programs"].items()},
        loop=None if record["loop"] is None else _axis_from_key(record["loop"]),
    )


def _check_header(spec: FoldSpec, record: dict) -> None:
    if record.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT!r} file")
    if record.get("version") != VERSION:
        raise ValueError(
            f"unsupported fold-plan version {record.get('version')!r}; expected {VERSION}"
        )
    if record.get("algebra") != spec.algebra.value:
        raise ValueError(
            f"plan algebra {record.get('algebra')!r} does not match {spec.algebra.value!r}"
        )
    axes = sorted(_axis_key(axis.id) for axis in spec.axes)
    if record.get("axes") != axes:
        raise ValueError("persisted plan axes do not match the fold specification")


def fold_plan_from_record(spec: FoldSpec, record: dict) -> FoldPlan:
    """Rebuild a plan against ``spec``; kernels remain owned/cached by CuTile."""
    _check_header(spec, record)
    forward = record.get("forward")
    backward = record.get("backward")
    if not isinstance(forward, list) or not isinstance(backward, list):
        raise ValueError("persisted forward/backward stages must be lists")
    if len(backward) > 1:
        raise ValueError("only single-stage fold backward plans are currently supported")
    backward_schedule = None if not backward else _schedule(backward[0])
    backward_kind = None if not backward else backward[0]["kind"]

    kinds = tuple(stage["kind"] for stage in forward)
    if kinds == ("map_fold",):
        constructor = (
            commutative_full_fold_plan
            if spec.algebra == AlgebraKind.commutative
            else general_full_fold_plan
        )
        expected_backward = (
            "recompute_finalize_grad_write"
            if spec.algebra == AlgebraKind.commutative
            else "recompute_fold_finalize_grad_write"
        )
        if backward_kind not in (None, expected_backward):
            raise ValueError(f"invalid backward stage {backward_kind!r} for full fold")
        return constructor(spec, _schedule(forward[0]), backward_schedule=backward_schedule)

    expected_second = "fold" if spec.algebra == AlgebraKind.commutative else "scan"
    if kinds != ("map_fold_partial", expected_second):
        raise ValueError(f"unsupported persisted forward stage sequence: {kinds}")
    if spec.algebra == AlgebraKind.commutative:
        if backward_kind not in (None, "recompute_finalize_grad_write"):
            raise ValueError(f"invalid backward stage {backward_kind!r} for commutative fold")
        return commutative_partial_fold_plan(
            spec,
            _schedule(forward[0]),
            _schedule(forward[1]),
            backward_schedule=backward_schedule,
        )

    checkpointed = backward_kind == "recompute_prefix_fold_finalize_grad_write"
    if backward_kind not in (
        None,
        "recompute_fold_finalize_grad_write",
        "recompute_prefix_fold_finalize_grad_write",
    ):
        raise ValueError(f"invalid backward stage {backward_kind!r} for general fold")
    return general_partial_fold_plan(
        spec,
        _schedule(forward[0]),
        _schedule(forward[1]),
        backward_schedule=backward_schedule,
        checkpointed_backward=checkpointed,
    )


def load_fold_plan(spec: FoldSpec, path) -> FoldPlan:
    return fold_plan_from_record(spec, json.loads(Path(path).read_text()))


__all__ = [
    "fold_plan_from_record",
    "load_fold_plan",
    "plan_record",
    "save_fold_plan",
]
