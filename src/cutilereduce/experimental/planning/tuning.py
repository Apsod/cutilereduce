"""Empirical selection of analytical finalists, with explicit backend adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .analysis import AnalyticalSweep, PlanningProblem, ScheduleCandidate


def program_fingerprint(program):
    """Include semantics, differentiation regions, precision and input contracts."""
    from dataclasses import fields
    from cutilereduce.experimental.ir.core import TensorRegion
    text = repr(program.spec) + repr(program.execution_structure)
    for field in fields(program):
        value = getattr(program, field.name)
        text += value.format() if isinstance(value, TensorRegion) else repr(value)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TuningMeasurement:
    candidate: ScheduleCandidate
    seconds: float | None
    failure: str | None = None


@dataclass
class TuningResult:
    executable: Any
    selected: ScheduleCandidate
    measurements: tuple[TuningMeasurement, ...]
    problem: PlanningProblem

    def save(self, path):
        """Save concrete decisions, not generated code or pickled callables."""
        Path(path).write_text(json.dumps({
            "version": 1, "backend": self.problem.target.backend,
            "fingerprint": program_fingerprint(self.problem.program),
            "sizes": dict(self.problem.sizes),
            "selected": self.selected.to_dict(),
            "measurements": [{"candidate": m.candidate.to_dict(), "seconds": m.seconds,
                              "failure": m.failure} for m in self.measurements],
        }, indent=2) + "\n")


def load_candidate(problem: PlanningProblem, path) -> ScheduleCandidate:
    value = json.loads(Path(path).read_text())
    if value.get("version") != 1 or value.get("backend") != problem.target.backend:
        raise ValueError("saved plan version or backend mismatch")
    if value["sizes"] != dict(problem.sizes) or value["fingerprint"] != program_fingerprint(problem.program):
        raise ValueError("saved plan does not match program and sizes")
    candidate = ScheduleCandidate.from_dict(value["selected"])
    estimate = problem.evaluate(candidate)
    if not estimate.accepted:
        raise ValueError(f"saved plan is no longer legal: {estimate.rejection_reasons}")
    return candidate


def tune_candidates(
    problem: PlanningProblem,
    *,
    compile_candidate: Callable[[ScheduleCandidate], Any],
    validate: Callable[[Any], None],
    measure: Callable[[Any], float],
    sweep: AnalyticalSweep | None = None,
    backward: bool = False,
    candidate_count: int = 20,
    per_family: int = 1,
) -> TuningResult:
    """Compile, validate, then time each finalist; retain all failure records.

    Validation must raise on failure. Measurement owns synchronization and buffer
    reset, since those are execution-target responsibilities. Stage winners are
    never composed without measuring their actual complete execution.
    """
    sweep = problem.sweep(backward=backward) if sweep is None else sweep
    finalists = sweep.shortlist(candidate_count, per_family=per_family)
    measurements = []
    best = None
    best_seconds = math.inf
    for estimate in finalists:
        candidate = estimate.candidate
        try:
            legal = problem.evaluate(candidate)
            if not legal.accepted:
                raise ValueError(str(legal.rejection_reasons))
            executable = compile_candidate(candidate)
            validate(executable)
            seconds = float(measure(executable))
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("benchmark must return finite positive seconds")
        except Exception as error:
            measurements.append(TuningMeasurement(candidate, None, f"{type(error).__name__}: {error}"))
            continue
        measurements.append(TuningMeasurement(candidate, seconds))
        if seconds < best_seconds:
            best = (executable, candidate)
            best_seconds = seconds
    if best is None:
        failures = "; ".join(m.failure or "" for m in measurements)
        raise RuntimeError(f"no valid tuning candidate: {failures or 'empty analytical shortlist'}")
    return TuningResult(best[0], best[1], tuple(measurements), problem)
