"""CPU-only symbolic attention costs and analytical schedule sweeps.

uv run -m examples.experimental_planning --backend cutile --length 1024 --json /tmp/sweep.json
No kernels are compiled and no hardware is detected.
"""

import argparse

from cutilereduce.experimental.planning.analysis import BackendModel, PlanningProblem
from examples.experimental_attention import operator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cutile"), default="cutile")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    # Illustrative rates supplied by the caller, not detected/measured peaks.
    from cutilereduce.experimental.backends.cutile.analysis import cutile_analytical_target
    constructor = cutile_analytical_target if args.backend == "cutile" else lambda **kw: BackendModel("cpu", **kw)
    target = constructor( workers=16, contraction_flops_per_second=1e12,
        scalar_ops_per_second=1e11, transcendental_ops_per_second=1e10,
        bytes_per_second=1e11, launch_seconds=5e-6,
        tile_choices=(1, 16, 32, 64),
    )
    program = operator().lower(backward=args.backward)
    symbolic = PlanningProblem(program, target)
    print("Region map_fold contraction FLOPs:", symbolic.regions["map_fold"].work.contraction_flops)
    partial = symbolic.estimate(parameters={"tile.h": 1, "tile.l": 32, "tile.g": 1, "tile.r": 32}, backward=args.backward)
    print("Symbolic schedule work:", partial.work.total)
    sizes = {"h": 4, "l": args.length, "g": 8, "r": args.length, "dqk": 128, "dv": 128}
    problem = PlanningProblem(program, target, sizes)
    sweep = problem.sweep(backward=args.backward, max_candidates=args.max_candidates)
    for estimate in sweep.shortlist(5):
        print(estimate.candidate.to_dict())
        print({"estimated_ms": float(estimate.cost.estimated_seconds) * 1e3,
               "auxiliary_bytes": float(estimate.cost.auxiliary_peak_bytes),
               "resident_bytes_upper_bound": float(estimate.cost.resident_bytes_upper_bound)})
    print(f"{len(sweep.estimates)} candidates; truncated={sweep.truncated}")
    if args.json:
        sweep.save(args.json)


if __name__ == "__main__":
    main()
