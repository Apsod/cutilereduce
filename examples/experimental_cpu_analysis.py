"""Rank generated CPU schedules and compare against recorded measurements.

No compiler, tensor allocation, GPU or hardware detection is required. The input
report supplies explicit dimensions and hardware. Measured tile choices are used
only after ranking, to report their analytical ranks; they do not seed the search.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from cutilereduce.experimental import CpuTarget, PlanningProblem, ScheduleCandidate, cpu_analytical_target
from examples.experimental_cpu_single_tile import attention_operator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path,
                        default=Path("benchmarks/experimental_cpu_attention/2048.json"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    recorded = json.loads(args.measurements.read_text())
    hardware = replace(CpuTarget(**recorded["hardware"]), worker_count=args.threads)
    problem = PlanningProblem(attention_operator().lower(), cpu_analytical_target(hardware), recorded["sizes"])
    sweep = problem.sweep(backward=True, max_candidates=args.max_candidates)
    ranked = sweep.shortlist(len(sweep.estimates), per_family=0)
    ranks = {e.candidate.key: i + 1 for i, e in enumerate(ranked)}
    print(f"Generated {len(sweep.estimates)} candidates; accepted={len(ranked)}; truncated={sweep.truncated}")
    print("Model:", json.dumps(dict(problem.target.model_metadata)))
    for i, estimate in enumerate(ranked[:args.shortlist], 1):
        print(i, estimate.candidate.to_dict(), float(estimate.cost.estimated_seconds) * 1000, "predicted ms")
    comparisons = []
    for label, value in recorded["results"][str(args.threads)]["forward_backward"].items():
        if label == "torch_eager" or label not in recorded["candidates"]:
            continue
        candidate = problem.canonical_candidate(ScheduleCandidate.from_dict(recorded["candidates"][label]["candidate"]))
        comparisons.append({"label": label, "measured_ms": value["median_ms"],
                            "analytical_rank": ranks.get(candidate.key),
                            "in_shortlist": ranks.get(candidate.key, len(ranked) + 1) <= args.shortlist})
    comparisons.sort(key=lambda c: c["measured_ms"])
    print("Recorded measurement comparison:", json.dumps(comparisons, indent=2))
    if args.json:
        report = sweep.to_dict()
        report["measurement_comparison"] = comparisons
        report["shortlist"] = [e.candidate.to_dict() for e in ranked[:args.shortlist]]
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
