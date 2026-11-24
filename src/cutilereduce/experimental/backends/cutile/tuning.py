"""cuTile convenience entry point using shared analytical/empirical selection."""

from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.driver import AnalysisReport, tune_report
from .backend import CuTileBackend
from .target import CuTileTarget


def tune_cutile(operator, sizes, inputs, *, reference, analytical_target,
                backward=False, candidate_count=20, sweep=None, warmup=3,
                iterations=20, rtol=1e-3, atol=1e-4):
    if analytical_target.backend != "cutile":
        raise ValueError("tune_cutile requires a cuTile analytical target")
    backend = CuTileBackend(analytical_target=analytical_target)
    hardware = CuTileTarget(**dict(analytical_target.hardware_identity))
    prepared = backend.specialize(operator.lower(backward=backward), hardware, sizes=sizes)
    problem = PlanningProblem(prepared.program, prepared.analytical_target, sizes)
    if sweep is None:
        from cutilereduce.experimental.planning.stages import stage_candidates
        sweep = problem.sweep(stage_candidates(problem) if backward else None, backward=backward)
    return tune_report(AnalysisReport(backend, prepared, problem, sweep), inputs,
        reference=reference, execution=backend.execution_adapter(), backward=backward,
        candidate_count=candidate_count, warmup=warmup, iterations=iterations, rtol=rtol, atol=atol)
