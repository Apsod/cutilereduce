"""User-facing compilation and tuning; backend-specific mechanics stay behind adapters."""

from .planning.driver import analyze, compile_report, tune_report


def compile(operator, sizes, *, backend, hardware, backward=False,
            candidate=None, max_candidates=1_000_000, parameter_domains=None):
    """Analytically select and compile without running a benchmark."""
    if candidate is not None and candidate.backward != backward:
        raise ValueError("candidate direction does not match requested workload")
    report = analyze(operator, sizes, backend=backend, hardware=hardware,
        backward=backward, max_candidates=max_candidates, parameter_domains=parameter_domains,
        candidates=[candidate] if candidate is not None else None)
    return compile_report(report, candidate=candidate)


def tune(operator, sizes, inputs, *, backend, hardware, reference, backward=False,
         candidate_count=20, per_family=1, max_candidates=1_000_000, parameter_domains=None,
         candidates=None, warmup=3, iterations=20, rtol=1e-4, atol=1e-5):
    """Analytical pruning/top-k followed by validation and empirical selection."""
    report = analyze(operator, sizes, backend=backend, hardware=hardware,
        backward=backward, max_candidates=max_candidates, parameter_domains=parameter_domains,
        candidates=candidates, stagewise=True)
    return tune_report(report, inputs, reference=reference, execution=backend.execution_adapter(),
        backward=backward, candidate_count=candidate_count, per_family=per_family,
        warmup=warmup, iterations=iterations, rtol=rtol, atol=atol)


__all__ = ["analyze", "compile", "tune"]
