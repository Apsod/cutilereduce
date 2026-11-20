"""Validate and benchmark CPU attention tiles against eager PyTorch.

Compile every candidate before timing, then randomize measurement order. Training
variants include looping over queries to accumulate key/value gradients locally.

uv run -m examples.experimental_cpu_tiles --json /tmp/cpu-tiles.json
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics
import time

import torch

from cutilereduce.experimental import (
    CpuTarget, PlanningProblem, ScheduleCandidate, compile_cpu_candidate,
    cpu_analytical_target,
)
from examples.experimental_cpu_single_tile import attention_operator, reference


def candidates(length):
    def make(name, family, query, fold, backward_query=None, backward_fold=None, loop="r"):
        parameters = {"tile.h": 1, "tile.g": 1, "tile.l": query, "tile.r": fold}
        if backward_query is not None:
            parameters["backward_tile.l"] = backward_query
        if backward_fold is not None:
            parameters["backward_tile.r"] = backward_fold
        return name, ScheduleCandidate(family, tuple(sorted(parameters.items())), True, loop)

    yield make("full", "fused_tree", length, length)
    for q in (256, 128, 64, 32):
        if q < length:
            yield make(f"query_{q}", "fused_tree", q, length,
                       backward_query=q, backward_fold=length, loop="l")
    for q in (128, 64, 32):
        if q < length:
            yield make(f"full_forward_backward_query_{q}", "fused_tree", length, length,
                       backward_query=q, backward_fold=length, loop="l")
    if length >= 256:
        yield make("tree_fold_256", "fused_tree", length, 256)
        yield make("online_query_128_fold_256", "online", 128, 256,
                   backward_query=128, backward_fold=length, loop="l")
        yield make("online_query_64_fold_128", "online", 64, 128,
                   backward_query=64, backward_fold=length, loop="l")
        yield make("atomic_query_128", "fused_tree", 128, length, loop="r")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--analytical-shortlist", type=int, default=0,
                        help="also benchmark analytically selected independent training schedules per thread count")
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.length, args.dimension, args.iterations, *args.threads) <= 0 or args.warmup < 0:
        parser.error("dimensions, iterations and threads must be positive; warmup nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    sizes = {"h": 1, "l": args.length, "g": 1, "r": args.length,
             "dqk": args.dimension, "dv": args.dimension}
    op = attention_operator()
    # Keep the same local architecture key as the single-tile experiment.
    target = CpuTarget("single_tile", 1, 32, 64)
    problem = PlanningProblem(op.lower(), cpu_analytical_target(target), sizes)
    inputs = (
        (torch.randn(1, args.length, 1, args.dimension) / args.dimension ** 0.5).requires_grad_(),
        torch.randn(1, args.length, args.dimension, requires_grad=True),
        torch.randn(1, args.length, args.dimension, requires_grad=True),
    )
    cotangent = torch.randn(1, args.length, 1, args.dimension)
    expected = reference(*inputs)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    functions = {"torch_eager": reference}
    records = {}
    chosen = list(candidates(args.length))
    if args.only:
        unknown = set(args.only) - {name for name, _ in chosen}
        if unknown:
            parser.error(f"unknown candidates: {sorted(unknown)}")
        chosen = [(name, candidate) for name, candidate in chosen if name in args.only]
    analytical_searches = {}
    if args.analytical_shortlist:
        hardware = CpuTarget.detect()
        for threads in args.threads:
            analytical = PlanningProblem(op.lower(), cpu_analytical_target(
                replace(hardware, worker_count=threads)), sizes)
            sweep = analytical.sweep(backward=True, max_candidates=args.max_candidates)
            selected = sweep.shortlist(args.analytical_shortlist, per_family=0)
            analytical_searches[str(threads)] = {
                "candidate_count": len(sweep.estimates), "truncated": sweep.truncated,
                "model": dict(sweep.model_metadata),
                "selected": [e.candidate.to_dict() for e in selected],
            }
            print(f"Analytical threads={threads}: {len(sweep.estimates)} candidates; "
                  f"truncated={sweep.truncated}", flush=True)
            chosen.extend((f"analytical_t{threads}_{i}", e.candidate)
                          for i, e in enumerate(selected, 1))
    for name, candidate in chosen:
        print(f"Compiling {name}: {candidate.to_dict()}", flush=True)
        start = time.perf_counter()
        try:
            executable = compile_cpu_candidate(op, problem, candidate, target=target)
            output = executable(*inputs)
            torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-5)
            gradients = torch.autograd.grad(output, inputs, cotangent)
            for actual, wanted in zip(gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=2e-4, atol=2e-5)
        except Exception as error:
            records[name] = {"candidate": candidate.to_dict(), "failure": f"{type(error).__name__}: {error}"}
            print(f"FAILED {name}: {error}", flush=True)
            continue
        functions[name] = executable
        records[name] = {
            "candidate": candidate.to_dict(), "compile_seconds": time.perf_counter() - start,
            "gradient_commits": {g.input_name: g.commit_mode.value for g in executable.backward_program.gradients},
        }
        print(f"Validated {name} ({records[name]['compile_seconds']:.1f}s)", flush=True)
    report = {"sizes": sizes, "dtype": "float32", "iterations": args.iterations,
              "validation": "outputs and random-cotangent input VJPs",
              "hardware": asdict(CpuTarget.detect()), "torch_version": torch.__version__,
              "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(), "candidates": records,
              "analytical_searches": analytical_searches, "results": {}}
    for threads in args.threads:
        torch.set_num_threads(threads)
        rng = random.Random(42)
        # Parallel execution must satisfy the same random-cotangent checks.
        for name, function in functions.items():
            output = function(*inputs)
            torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-5)
            gradients = torch.autograd.grad(output, inputs, cotangent)
            for actual, wanted in zip(gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=2e-4, atol=2e-5)
        timing = {}
        for mode in ("forward", "forward_backward"):
            def run(function):
                if mode == "forward":
                    with torch.no_grad():
                        function(*inputs)
                else:
                    value = function(*inputs)
                    torch.autograd.grad(value, inputs, cotangent)
            for _ in range(args.warmup):
                for function in functions.values():
                    run(function)
            samples = {name: [] for name in functions}
            for _ in range(args.iterations):
                order = list(functions)
                rng.shuffle(order)
                for name in order:
                    start = time.perf_counter()
                    run(functions[name])
                    samples[name].append(time.perf_counter() - start)
            timing[mode] = {}
            for name, values in samples.items():
                timing[mode][name] = {"median_ms": statistics.median(values) * 1e3,
                                      "p25_ms": sorted(values)[len(values) // 4] * 1e3,
                                      "p75_ms": sorted(values)[3 * len(values) // 4] * 1e3}
            baseline = timing[mode]["torch_eager"]["median_ms"]
            print(f"threads={threads} {mode}: eager={baseline:.4f} ms", flush=True)
            for name in sorted(functions, key=lambda n: timing[mode][n]["median_ms"]):
                value = timing[mode][name]["median_ms"]
                timing[mode][name]["speedup_over_eager"] = baseline / value
                print(f"  {name}: {value:.4f} ms ({baseline / value:.3f}x)", flush=True)
        report["results"][str(threads)] = timing
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
