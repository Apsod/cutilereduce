from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cuda.tile as ct
import sympy

from cutilereduce.core.axis import AxisId, AxisType
from cutilereduce.fold.plan import FoldPlan, FoldSpec
from cutilereduce.stages import StageSchedule


FORMAT = "cutilereduce.fold-plan"
VERSION = 3


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
    from cutilereduce.fold.paths import BACKWARD_RECORD_TYPES, FORWARD_RECORD_TYPES
    from cutilereduce.fold.templates import FoldPlanTemplate

    _check_header(spec, record)
    forward = record.get("forward")
    backward = record.get("backward")
    if not isinstance(forward, list) or not isinstance(backward, list):
        raise ValueError("persisted forward/backward stages must be lists")
    matches = tuple(path for cls in FORWARD_RECORD_TYPES
                    if (path := cls.from_record(spec, forward)) is not None)
    if len(matches) != 1:
        raise ValueError(f"unsupported persisted forward stage sequence: {tuple(s['kind'] for s in forward)}")
    forward_path, = matches
    if backward:
        matches = tuple(path for cls in BACKWARD_RECORD_TYPES
                        if (path := cls.from_record(spec, backward)) is not None)
        if len(matches) != 1:
            raise ValueError(f"unsupported persisted backward stage sequence: {tuple(s['kind'] for s in backward)}")
        backward_path, = matches
    else:
        backward_path = None
    template = FoldPlanTemplate(spec, forward_path, backward_path)
    values = {}
    for symbolic, saved in zip(template.stages, (*forward, *backward), strict=True):
        if symbolic.stage.name != saved["kind"]:
            raise ValueError(f"persisted stage {saved['kind']!r} does not match plan topology")
        _schedule(saved)  # Validate the recorded axes and integer values.
        for axis in symbolic.domain.compute_axes:
            key = _axis_key(axis.id)
            for symbolic_value, persisted in (
                (axis.extent, saved["extents"][key]),
                (axis.tile, saved["tiles"][key]),
            ):
                if isinstance(symbolic_value, sympy.Symbol):
                    values[str(symbolic_value)] = _integer(persisted, "schedule value")
        for axis in symbolic.domain.program_axes:
            if isinstance(axis.programs, sympy.Symbol):
                values[str(axis.programs)] = _integer(
                    saved["programs"].get(_axis_key(axis.id), 1), "program count",
                )
    if backward_path is not None and backward_path.name == "commutative":
        values.setdefault("backward_group_count", 1)
    plan = template.bind(values)
    if [ _stage_record(stage) for stage in plan.forward ] != forward or [
        _stage_record(stage) for stage in plan.backward
    ] != backward:
        raise ValueError("persisted stage schedules do not match the reconstructed plan")
    return plan


def load_fold_plan(spec: FoldSpec, path, sizes=None) -> FoldPlan:
    plan = fold_plan_from_record(spec, json.loads(Path(path).read_text()))
    if sizes is not None:
        expected = {spec.axis_id(name): size for name, size in sizes.items()}
        for stage in (*plan.forward, *plan.backward):
            for axis in stage.domain.compute_axes:
                if axis.id in expected and axis.extent != expected[axis.id]:
                    raise ValueError(
                        f"persisted extent for {axis.name!r} is {axis.extent}, "
                        f"but the requested size is {expected[axis.id]}"
                    )
    return plan


__all__ = [
    "fold_plan_from_record",
    "load_fold_plan",
    "plan_record",
    "save_fold_plan",
]
