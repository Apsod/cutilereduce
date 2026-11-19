"""Fixed-tile forward attention through the experimental CuTile path.

This intentionally has no tuner and no backward path.  It exercises the
target-neutral fold IR, the LWS execution carrier, and the CuTile realization
with one reproducible tile configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import cuda.tile as ct
import torch

from cutilereduce.experimental import (
    ContractionPrecision,
    FoldMaps,
    FoldOperator,
    FoldSpec,
    LogWeightedSum,
    StructuredValue,
    TensorSpec,
    build_schedule_space,
    emit_cutile_source,
    specialize_cutile_schedule,
)


FIXED_TILES = {"h": 1, "l": 128, "g": 1, "r": 32}


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def operator() -> FoldOperator:
    spec = FoldSpec(
        inputs={
            "query": TensorSpec(("h", "l", "g", "dqk"), "bfloat16"),
            "key": TensorSpec(("h", "r", "dqk"), "bfloat16"),
            "value": TensorSpec(("h", "r", "dv"), "bfloat16"),
        },
        batch_axes=("h", "l", "g"),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        return StructuredValue(
            {
                "log_weight": ctx.contract(
                    inputs["query"],
                    inputs["key"],
                    over=("dqk",),
                    dtype="float32",
                ),
                # The semantic value is FP32. The execution contraction below
                # explicitly authorizes its physical BF16 realization.
                "value": ctx.cast(inputs["value"], "float32"),
            }
        )

    def to_output(ctx, semantic):
        del ctx
        return StructuredValue({"value": semantic["value"]})

    precision = ContractionPrecision(
        operand_dtype="bfloat16",
        accumulator_dtype="float32",
        allow_operand_rounding=True,
    )
    return FoldOperator(
        spec,
        LogWeightedSum(("dv",), contraction_precision=precision),
        FoldMaps(to_factor, to_output),
    )


def build_kernel(sizes: dict[str, int]):
    program = operator().lower(backward=False)
    template = build_schedule_space(program).forward[0]
    specialized = specialize_cutile_schedule(
        program, template, sizes, tiles=FIXED_TILES
    )
    source_path = Path("/tmp/cutilereduce_experimental_attention.py")
    source_path.write_text(emit_cutile_source(specialized))
    module_spec = importlib.util.spec_from_file_location(
        "cutilereduce_experimental_attention", source_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.single_owner_tiled_fold, specialized


def sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    heads, length, groups, width = query.shape
    right = key.shape[1]
    query = query.permute(0, 2, 1, 3).reshape(1, heads * groups, length, width)
    key = key.reshape(1, heads, right, width)
    value = value.reshape(1, heads, right, value.shape[-1])
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, enable_gqa=True, scale=1.0, dropout_p=0.0
    )
    return output.reshape(1, heads, groups, length, value.shape[-1])[0].permute(0, 2, 1, 3)


def elapsed_ms(callable, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", dest="h", type=_positive, default=4)
    parser.add_argument("--length", dest="l", type=_positive, default=4096)
    parser.add_argument("--groups", dest="g", type=_positive, default=8)
    parser.add_argument("--right", dest="r", type=_positive, default=4096)
    parser.add_argument("--dqk", type=_positive, default=128)
    parser.add_argument("--dv", type=_positive, default=128)
    parser.add_argument("--iterations", type=_positive, default=100)
    parser.add_argument("--warmup", type=_positive, default=25)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("experimental CuTile attention requires CUDA")

    sizes = {name: getattr(args, name) for name in ("h", "l", "g", "r", "dqk", "dv")}
    kernel, specialized = build_kernel(sizes)
    torch.manual_seed(args.seed)
    query = torch.randn((args.h, args.l, args.g, args.dqk), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((args.h, args.r, args.dqk), device="cuda", dtype=torch.bfloat16)
    value = torch.randn((args.h, args.r, args.dv), device="cuda", dtype=torch.bfloat16)
    query.mul_(args.dqk ** -0.25)
    key.mul_(args.dqk ** -0.25)
    output = torch.empty((args.h, args.l, args.g, args.dv), device="cuda")
    stage = specialized.implementation.stages[0]
    grid = (math.prod(partition.programs for partition in stage.grid_partitions),)

    def run_experimental() -> None:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (query, key, value, output),
        )

    def run_sdpa() -> None:
        sdpa(query, key, value)

    run_experimental()
    torch.cuda.synchronize()
    if args.validate:
        torch.testing.assert_close(
            output, sdpa(query, key, value), atol=3e-2, rtol=3e-2, check_dtype=False
        )
    for _ in range(args.warmup):
        run_experimental()
        run_sdpa()
    torch.cuda.synchronize()

    print("experimental attention forward (fixed tiles)")
    print("shape: " + ", ".join(f"{name}={sizes[name]}" for name in sizes))
    print("tiles: " + ", ".join(f"{name}={value}" for name, value in FIXED_TILES.items()))
    print(f"Experimental CuTile forward: {elapsed_ms(run_experimental, iterations=args.iterations):.3f} ms")
    print(f"PyTorch SDPA forward: {elapsed_ms(run_sdpa, iterations=args.iterations):.3f} ms")


if __name__ == "__main__":
    main()
