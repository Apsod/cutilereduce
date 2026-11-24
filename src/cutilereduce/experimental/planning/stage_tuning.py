"""Deduplicate, prune, measure and compose compatible executable stages."""

from collections import defaultdict
from dataclasses import dataclass
import math

from .stages import StagedPlan, best_compatible, estimated_time
from .analysis import ScheduleCandidate
from .tuning import TuningResult, TuningMeasurement
from .validation import validate_callable


@dataclass(frozen=True)
class StageMeasurement:
    key: str
    slot: str
    candidate: ScheduleCandidate
    estimated_seconds: float
    seconds: float | None
    failure: str | None = None


def assembled_candidate(stages):
    forward = next(s for s in stages if s.slot == "forward")
    p = dict(forward.parameters)
    final = next((s for s in stages if s.slot == "finalize"), None)
    for stage in stages:
        if stage is not forward:
            p.update(stage.parameters)
    # Checkpoint partitioning belongs to forward; loop_programs in prefix plans
    # is implied by that partitioning and must not become an illegal override.
    if (
        "partition_count" in p
        and final is not None
        and any(
            a.name == "execution" and "checkpoints" in repr(a.representation)
            for a in final.requires
        )
    ):
        p.pop("loop_programs", None)
    return ScheduleCandidate(
        forward.representative.family,
        tuple(sorted(p.items())),
        final is not None,
        final.representative.loop_axis if final else None,
    )


def tune_stages(
    report,
    inputs,
    *,
    reference,
    execution,
    backward,
    candidate_count,
    per_family,
    warmup,
    iterations,
    rtol,
    atol,
):
    if candidate_count <= 0 or per_family < 0 or warmup < 0 or iterations <= 0:
        raise ValueError("invalid stage tuning limits")
    backend = report.backend
    unique, plan_shapes = {}, {}
    estimates = {}
    for estimate in report.sweep.estimates:
        if not estimate.accepted:
            continue
        plan = backend.stage_plan(report.program, estimate.candidate).check()
        shape = tuple((s.slot, s.direction) for s in plan.stages)
        plan_shapes.setdefault(shape, plan)
        for stage in plan.stages:
            score = estimated_time(estimate.cost, stage)
            if not math.isfinite(score) or score < 0:
                raise ValueError("stage ranking requires finite bound estimates")
            if stage.key not in unique or score < estimates[stage.key]:
                unique[stage.key], estimates[stage.key] = stage, score
    if not unique:
        raise RuntimeError("no analytically legal executable stages")
    # Keep a shortlist for every interface class, not just a global whole-plan
    # shortlist. A globally poor plan can contain the best individual stage.
    groups = defaultdict(list)
    for stage in unique.values():
        groups[(stage.slot, stage.requires, stage.produces, stage.couplings)].append(
            stage
        )
    shortlisted = {}
    for options in groups.values():
        options.sort(key=lambda s: estimates[s.key])
        chosen = []
        if options[0].direction == "forward":
            for family in dict.fromkeys(s.representative.family for s in options):
                chosen.extend(
                    [s for s in options if s.representative.family == family][
                        :per_family
                    ]
                )
        chosen = chosen[:candidate_count]
        chosen.extend(s for s in options if s not in chosen)
        for stage in chosen[:candidate_count]:
            shortlisted[stage.key] = stage
    fixture = backend.stage_fixture(
        report.program, inputs, reference, backward=backward, rtol=rtol, atol=atol
    )
    runners, times, measurements = {}, {}, []
    for stage in shortlisted.values():
        try:
            context = fixture.context(stage)
            runner = backend.compile_stage(report.program, stage)
            fixture.validate(runner)
            seconds = float(
                execution.measure_stage(
                    lambda: runner.run(context),
                    inputs,
                    warmup=warmup,
                    iterations=iterations,
                )
            )
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("stage timing must be finite and positive")
            runners[stage.key], times[stage.key] = runner, seconds
            measurements.append(
                StageMeasurement(
                    stage.key,
                    stage.slot,
                    stage.representative,
                    estimates[stage.key],
                    seconds,
                )
            )
        except Exception as error:
            measurements.append(
                StageMeasurement(
                    stage.key,
                    stage.slot,
                    stage.representative,
                    estimates[stage.key],
                    None,
                    f"{type(error).__name__}: {error}",
                )
            )
    pools = defaultdict(list)
    for key in times:
        pools[runners[key].spec.slot].append(runners[key].spec)
    # Validate only assembled optima. If a seemingly valid stage combination
    # fails joint validation, retain the failure and try the next compatible path.
    final_measurements = []
    excluded = set()
    while times:
        try:
            chosen = best_compatible(plan_shapes.values(), pools, times, excluded)
        except RuntimeError as error:
            failures = "; ".join(
                m.failure for m in (*measurements, *final_measurements) if m.failure
            )
            raise RuntimeError(f"no valid compatible stage plan: {failures}") from error
        plan = StagedPlan(
            tuple(chosen),
            next(
                p.external
                for p in plan_shapes.values()
                if tuple(s.slot for s in p.stages) == tuple(s.slot for s in chosen)
            ),
        ).check()
        candidate = report.problem.canonical_candidate(assembled_candidate(plan.stages))
        try:
            legal = report.problem.evaluate(candidate)
            if not legal.accepted:
                raise ValueError(
                    f"composed schedule is analytically illegal: {legal.rejection_reasons}"
                )
            compiled = backend.assemble(
                report.program, tuple(runners[s.key] for s in chosen)
            )
            validate_callable(
                compiled, inputs, reference, backward=backward, rtol=rtol, atol=atol
            )
            seconds = float(
                execution.measure(
                    compiled,
                    inputs,
                    backward=backward,
                    warmup=warmup,
                    iterations=iterations,
                )
            )
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("assembled timing must be finite and positive")
            final_measurements.append(TuningMeasurement(candidate, seconds))
            return TuningResult(
                compiled,
                candidate,
                tuple(final_measurements),
                report.problem,
                stage_measurements=tuple(measurements),
                selected_stages=tuple(chosen),
                mode="staged",
                tuning_metadata={
                    "unique_stages": len(unique),
                    "shortlisted_stages": len(shortlisted),
                    "analytical_candidates": len(report.sweep.estimates),
                },
            )
        except Exception as error:
            final_measurements.append(
                TuningMeasurement(candidate, None, f"{type(error).__name__}: {error}")
            )
            # Exclude this combination without invalidating individually valid
            # stages. The compatibility solver supports explicit path exclusions.
            excluded.add(tuple(s.key for s in chosen))
    failures = "; ".join(m.failure for m in measurements if m.failure)
    raise RuntimeError(
        f"no stage survived compilation/validation/measurement: {failures}"
    )
