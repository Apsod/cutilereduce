"""Frozen pre-migration estimates for explicit schedules; no native compilation."""

import json
from pathlib import Path

from cutilereduce.experimental import Add, Affine, FoldMaps, FoldOperator, FoldSpec, StructuredValue, TensorSpec
from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.backends.cpu.tuning import cpu_analytical_target
from examples.experimental_cpu_single_tile import attention_operator


def cases():
    add = FoldOperator(FoldSpec({"x": TensorSpec(("b", "r"))}, ("b",), ("r",)),
                       Add(), FoldMaps(lambda ctx, x: StructuredValue({"value": x["x"]})))
    algebra = Affine(("d",))
    affine = FoldOperator(FoldSpec({"scale": TensorSpec(("b", "r")),
        "value": TensorSpec(("b", "r", "d"))}, ("b",), ("r",)), algebra,
        FoldMaps(lambda ctx, x: algebra.value(acting=StructuredValue({"value": x["scale"]}),
                                             payload=StructuredValue({"value": x["value"]}))))
    for family in ("online", "fused_tree", "partial_fold"):
        yield family, add, {"b": 5, "r": 9}, family, {
            "tile.b": 2, "tile.r": 4, "backward_tile.b": 3,
            "backward_tile.r": 2, "partition_count": 2 if family == "partial_fold" else 1}, "b"
    yield "attention", attention_operator(), {"h": 2, "l": 12, "g": 3, "r": 10, "dqk": 8, "dv": 6}, "fused_tree", {
        "tile.h": 1, "tile.l": 4, "tile.g": 3, "tile.r": 10,
        "backward_tile.l": 12, "backward_tile.r": 2}, "r"
    yield "affine", affine, {"b": 3, "r": 9, "d": 4}, "partial_scan_checkpoints", {
        "tile.b": 2, "tile.r": 3, "partition_count": 2}, "r"


def snapshots():
    target = cpu_analytical_target(CpuTarget("parity", 2, 32, 64,
                                            l2_bytes=1310720, l3_bytes=12582912))
    result = {}
    for name, op, sizes, family, parameters, loop in cases():
        cost = PlanningProblem(op.lower(), target, sizes).estimate(
            family, parameters=parameters, backward=True, loop_axis=loop)
        result[name] = {
            "saved": float(cost.saved_bytes), "auxiliary": float(cost.auxiliary_peak_bytes),
            "stages": [{"name": s.name, **{key: float(getattr(s, key)) for key in (
                "read_bytes", "write_bytes", "atomic_bytes", "parallel_tasks", "serial_steps",
                "resident_bytes_upper_bound", "allocated_bytes", "estimated_seconds")},
                "work": float(s.work.total)} for s in cost.stages],
        }
    return result


def test_pre_migration_cost_parity():
    expected = json.loads((Path(__file__).parent / "fixtures" / "experimental_cost_parity.json").read_text())
    assert snapshots() == expected
