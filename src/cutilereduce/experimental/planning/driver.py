"""Shared analytical selection and empirical tuning across backend adapters."""

from dataclasses import dataclass, field, replace

from .analysis import AnalyticalSweep, PlanningProblem
from .backend import Backend, BackendProgram
from .tuning import tune_candidates
from .validation import validate_callable


@dataclass(frozen=True)
class AnalysisReport:
    backend: Backend
    program: BackendProgram
    problem: PlanningProblem
    sweep: AnalyticalSweep
    implementations: dict = field(default_factory=dict, repr=False)

    def shortlist(self, count=20, *, per_family=1):
        return self.sweep.shortlist(count, per_family=per_family)

    def realize(self, candidate):
        key = self.problem.canonical_candidate(candidate).key
        if key not in self.implementations:
            existing = next((e.implementation for e in self.sweep.estimates
                             if e.candidate.key == key and e.implementation is not None), None)
            self.implementations[key] = existing or self.backend.realize(self.program, candidate)
        return self.implementations[key]


def analyze(operator, sizes, *, backend, hardware, backward=False,
            max_candidates=1_000_000, parameter_domains=None, candidates=None):
    """Analyze without compiling or detecting hardware.

    Candidates are realized before ranking. Shared visitors derive estimates
    from bound recursive execution or tiled SSA; selected realizations are
    retained for compilation and empirical tuning.
    """
    if candidates is not None:
        candidates = tuple(candidates)
        if any(c.backward != backward for c in candidates):
            raise ValueError("candidate direction does not match analysis workload")
    program = backend.specialize(operator.lower(backward=backward), hardware, sizes=sizes)
    # The backend supplies realization and a space, not an opaque cost callback.
    # All candidate estimates then follow PlanningProblem.evaluate's IR analysis.
    def realize(problem, candidate):
        return backend.realize(program, candidate)
    target = replace(program.analytical_target, realization_factory=realize)
    problem = PlanningProblem(program.program, target, dict(program.sizes), program.schedule_space)
    sweep = problem.sweep(candidates, backward=backward, max_candidates=max_candidates,
                          parameter_domains=parameter_domains)
    return AnalysisReport(backend, program, problem, sweep)


def compile_report(report, *, candidate=None):
    """Compile an explicit candidate or the analytically preferred realization."""
    if candidate is None:
        selected = report.shortlist(1, per_family=0)
        if not selected:
            raise ValueError("no legal analytical candidate")
        candidate = selected[0].candidate
    canonical = report.problem.canonical_candidate(candidate)
    estimate = next((e for e in report.sweep.estimates if e.candidate.key == canonical.key), None)
    if estimate is None:
        estimate = report.problem.evaluate(canonical)
        if estimate.implementation is not None:
            report.implementations[canonical.key] = estimate.implementation
    if not estimate.accepted:
        raise ValueError(str(estimate.rejection_reasons))
    return report.backend.compile(report.realize(canonical))


def tune_report(report, inputs, *, reference, execution, backward=False,
                candidate_count=20, per_family=1, warmup=3, iterations=20,
                rtol=1e-4, atol=1e-5):
    """Use the same realizations for explicit analytical and empirical selection."""
    if any(e.candidate.backward != backward for e in report.sweep.estimates):
        raise ValueError("report direction does not match tuning workload")
    return tune_candidates(report.problem,
        compile_candidate=lambda candidate: report.backend.compile(report.realize(candidate)),
        validate=lambda f: validate_callable(f, inputs, reference, backward=backward, rtol=rtol, atol=atol),
        measure=lambda f: execution.measure(f, inputs, backward=backward,
                                           warmup=warmup, iterations=iterations),
        sweep=report.sweep, backward=backward, candidate_count=candidate_count,
        per_family=per_family)
