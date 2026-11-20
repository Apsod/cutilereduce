"""Compare one full-problem CPU attention tile against eager PyTorch.

uv run -m examples.experimental_cpu_single_tile --length 512 --dimension 32
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cutilereduce.experimental.planning.scheduled import CommitMode
from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental import FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, StructuredValue, TensorSpec
from cutilereduce.experimental.planning.analysis import PlanningProblem, ScheduleCandidate
from cutilereduce.experimental.backends.cpu.tuning import compile_cpu_candidate, cpu_analytical_target


def attention_operator():
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk")),
            "key": TensorSpec(("h", "r", "dqk")),
            "value": TensorSpec(("h", "r", "dv")),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )

    def factor(ctx, inputs):
        return StructuredValue({
            "log_weight": ctx.contract(inputs["query"], inputs["key"], over=("dqk",)),
            "value": inputs["value"],
        })

    return FoldOperator(
        spec, LogWeightedSum(("dv",)),
        FoldMaps(factor, lambda ctx, state: StructuredValue({"value": state["value"]})),
    )


def reference(query, key, value):
    scores = torch.bmm(query.squeeze(2), key.transpose(1, 2))
    return torch.bmm(scores.softmax(-1), value).unsqueeze(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.length, args.dimension, args.threads, args.iterations) <= 0 or args.warmup < 0:
        parser.error("dimensions, threads and iterations must be positive; warmup nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    sizes = {"h": 1, "l": args.length, "g": 1, "r": args.length,
             "dqk": args.dimension, "dv": args.dimension}
    op = attention_operator()
    target = CpuTarget("single_tile", args.threads, 32, 64)
    problem = PlanningProblem(op.lower(), cpu_analytical_target(target), sizes)
    # A tree with one leaf invokes map_fold once, avoiding the online update of
    # an identity carrier. Every forward/backward axis fits in one tile.
    candidate = ScheduleCandidate(
        "fused_tree", tuple((f"tile.{a}", sizes[a]) for a in ("h", "l", "g", "r")),
        backward=True, loop_axis="r",
    )
    print("Compiling single-tile attention...", flush=True)
    compiled = compile_cpu_candidate(op, problem, candidate, target=target)
    backward = compiled.backward_program
    assert backward is not None
    assert all(axis.count == 1 for axis in (*backward.program_axes, backward.loop_axis))
    assert all(g.commit_mode is CommitMode.STORE for g in backward.gradients)

    inputs = (
        (torch.randn(1, args.length, 1, args.dimension) / args.dimension ** 0.5).requires_grad_(),
        torch.randn(1, args.length, args.dimension, requires_grad=True),
        torch.randn(1, args.length, args.dimension, requires_grad=True),
    )
    cotangent = torch.randn(1, args.length, 1, args.dimension)
    actual, expected = compiled(*inputs), reference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    actual_grads = torch.autograd.grad(actual, inputs, cotangent)
    expected_grads = torch.autograd.grad(expected, inputs, cotangent)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)

    def forward(function):
        with torch.no_grad():
            function(*inputs)

    def training(function):
        output = function(*inputs)
        torch.autograd.grad(output, inputs, cotangent)

    timings = {}
    for name, run in (("forward", forward), ("forward_backward", training)):
        functions = {"experimental": lambda: run(compiled), "torch_eager": lambda: run(reference)}
        for _ in range(args.warmup):
            for function in functions.values():
                function()
        samples = {key: [] for key in functions}
        # Alternate measurement order to reduce systematic warm-cache/order bias.
        for iteration in range(args.iterations):
            order = tuple(functions) if iteration % 2 == 0 else tuple(reversed(functions))
            for key in order:
                start = time.perf_counter()
                functions[key]()
                samples[key].append(time.perf_counter() - start)
        timings[name] = {key: statistics.median(values) * 1e3 for key, values in samples.items()}
        timings[name]["experimental_over_torch"] = timings[name]["experimental"] / timings[name]["torch_eager"]
    result = {"sizes": sizes, "dtype": "float32", "threads": args.threads,
              "iterations": args.iterations, "schedule": candidate.to_dict(),
              "validation": "outputs and random-cotangent input VJPs passed",
              "atomic_gradients": False, "compile_seconds": compiled.compile_seconds,
              "median_ms": timings}
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
