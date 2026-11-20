"""Scheduled accounting follows SSA instructions rather than candidate formulas."""

from dataclasses import replace

import pytest

from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental.planning.analysis import PlanningProblem, ScheduleCandidate
from cutilereduce.experimental.backends.cpu.backend import CpuBackend
from cutilereduce.experimental.planning.execution_cost import analyze_tiled_stage
from cutilereduce.experimental.planning.ssa import lower_backward_stage
from test_experimental_cost_parity import cases


@pytest.mark.parametrize("case", list(cases())[:4], ids=lambda c: c[0])
def test_realized_backward_accounting_matches_existing_model(case):
    _, op, sizes, family, parameters, loop = case
    hardware = CpuTarget("parity", 2, 32, 64, l2_bytes=1310720, l3_bytes=12582912)
    backend = CpuBackend()
    program = backend.specialize(op.lower(), hardware, sizes=sizes)
    problem = PlanningProblem(program.program, program.analytical_target, sizes)
    candidate = ScheduleCandidate(family, tuple(sorted(parameters.items())), True, loop)
    implementation = backend.realize(program, candidate)
    stage = lower_backward_stage(implementation.metadata.scheduled_backward)
    actual = analyze_tiled_stage(problem, stage)
    expected = problem.evaluate(candidate).cost.stages[-1]
    for metric in ("work", "read_bytes", "write_bytes", "atomic_bytes", "parallel_tasks",
                   "serial_steps", "resident_bytes_upper_bound", "allocated_bytes", "estimated_seconds"):
        assert getattr(actual, metric) == getattr(expected, metric), metric
    assert actual.accounting_domain == expected.accounting_domain
    # Removing explicit accumulation must reduce charged work, independently of
    # candidate parameters. This proves the visitor reads the actual body.
    removed = "add" if any(o.op == "add" for o in stage.loop.body) else "call_region"
    body = tuple(o for o in stage.loop.body if o.op != removed)
    modified = replace(stage, loop=replace(stage.loop, body=body))
    assert float(analyze_tiled_stage(problem, modified).work.total) < float(actual.work.total)


def test_unknown_execution_instruction_is_not_free():
    from cutilereduce.experimental.planning.ssa import ScheduledOperation
    _, op, sizes, family, parameters, loop = next(cases())
    backend = CpuBackend()
    program = backend.specialize(op.lower(), CpuTarget("test", 1, 32, 64), sizes=sizes)
    candidate = ScheduleCandidate(family, tuple(sorted(parameters.items())), True, loop)
    implementation = backend.realize(program, candidate)
    stage = implementation.execution.gradient_stage
    changed = replace(stage, epilogue=(ScheduledOperation("unimplemented_transfer", (), ()),))
    problem = PlanningProblem(program.program, program.analytical_target, sizes)
    with pytest.raises(NotImplementedError, match="unimplemented_transfer"):
        analyze_tiled_stage(problem, changed)


@pytest.mark.parametrize("case", list(cases()), ids=lambda c: c[0])
def test_complete_bound_execution_matches_parity_baseline(case):
    from cutilereduce.experimental.planning.execution_cost import analyze_implementation
    _, op, sizes, family, parameters, loop = case
    hardware = CpuTarget("parity", 2, 32, 64, l2_bytes=1310720, l3_bytes=12582912)
    backend = CpuBackend()
    prepared = backend.specialize(op.lower(), hardware, sizes=sizes)
    problem = PlanningProblem(prepared.program, prepared.analytical_target, sizes)
    candidate = ScheduleCandidate(family, tuple(sorted(parameters.items())), True, loop)
    baseline = problem.evaluate(candidate).cost
    implementation = backend.realize(prepared, candidate)
    actual = analyze_implementation(problem, implementation, baseline)
    assert actual.stages == baseline.stages
    assert actual.saved_bytes == baseline.saved_bytes
    assert actual.auxiliary_peak_bytes == baseline.auxiliary_peak_bytes


@pytest.mark.parametrize("backward", [False, True])
def test_tiled_online_plan_matches_generic_estimator(backward):
    from cutilereduce.experimental.backends.cutile.backend import CuTileBackend
    from cutilereduce.experimental.backends.cutile.analysis import cutile_analytical_target
    from cutilereduce.experimental.planning.execution_cost import analyze_implementation
    _, op, _, _, _, _ = next(cases())
    sizes = {"b": 4, "r": 8}
    target = cutile_analytical_target(workers=2, contraction_flops_per_second=1e12,
        scalar_ops_per_second=1e10, transcendental_ops_per_second=1e9, bytes_per_second=1e10)
    from cutilereduce.experimental.backends.cutile.target import CuTileTarget
    backend = CuTileBackend(analytical_target=target)
    program = backend.specialize(op.lower(), CuTileTarget("test", 2), sizes=sizes)
    problem = PlanningProblem(program.program, target, sizes)
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), backward, "b" if backward else None)
    baseline = problem.evaluate(candidate, realize=False).cost
    implementation = backend.realize(program, candidate)
    actual = analyze_implementation(problem, implementation, baseline)
    assert actual.stages == baseline.stages
    assert actual.saved_bytes == baseline.saved_bytes
    assert actual.auxiliary_peak_bytes == baseline.auxiliary_peak_bytes
