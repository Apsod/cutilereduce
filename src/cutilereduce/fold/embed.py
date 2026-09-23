from __future__ import annotations

from dataclasses import replace
from itertools import product

import polars as pl
from cutilereduce.core import symbolic as expr

from cutilereduce.core.sweep import Sweep
from cutilereduce.fold.tune import stage_key, tune_fold_stages
from cutilereduce.stages.base import StageSchedule, normalize_axis_mapping
from cutilereduce.stages.embed import Embed, embed_axes


def embed_candidates(spec, sizes, *, hardware, candidate_count=20, max_tile=256):
    """Rank independent batch tiles using traffic and residency, without MMA work."""
    sizes = normalize_axis_mapping(spec, sizes)
    axes = embed_axes(spec)
    batch = tuple(axis for axis in axes if axis in spec.batch)
    symbols = {axis.id: expr.Symbol(f"embed_{axis.name}_tile") for axis in batch}
    fixed_tiles = {
        axis.id: 1 << (int(sizes[axis.id]) - 1).bit_length()
        for axis in axes
        if axis not in batch
    }
    schedule = StageSchedule(extents=sizes, tiles=fixed_tiles | symbols, programs={})
    stage = Embed(spec, schedule).build()
    values = [
        tuple(1 << n for n in range(min(int(sizes[axis.id]), max_tile).bit_length()))
        for axis in batch
    ]
    rows = [
        {
            f"cfg:{symbols[axis.id]}": tile
            for axis, tile in zip(batch, tiles, strict=True)
        }
        for tiles in product(*values)
    ]
    if not batch:
        return (
            Embed(
                spec, StageSchedule(extents=sizes, tiles=fixed_tiles, programs={})
            ).build(),
        )
    ranked = replace(Sweep.default, top_k=candidate_count).apply(
        stage.stage,
        pl.DataFrame(rows),
        symbols=hardware,
    )
    return tuple(
        Embed(
            spec,
            StageSchedule(
                extents=sizes,
                tiles=fixed_tiles
                | {axis.id: int(row[f"cfg:{symbols[axis.id]}"]) for axis in batch},
                programs={},
            ),
        ).build()
        for row in ranked.iter_rows(named=True)
    )


def tune_embed_stage(
    spec, sizes, functions, *, hardware, candidate_count, timeout, quiet
):
    candidates = embed_candidates(
        spec, sizes, hardware=hardware, candidate_count=candidate_count
    )
    if not candidates:
        raise RuntimeError("embed sweep produced no candidates")
    print(
        f"empirical embed-stage tuning ({len(candidates)} configurations)", flush=True
    )
    times = tune_fold_stages(candidates, functions, timeout=timeout, quiet=quiet)
    viable = tuple(stage for stage in candidates if stage_key(stage) in times)
    if not viable:
        raise RuntimeError("empirical tuning left no embed stage")
    best = min(viable, key=lambda stage: times[stage_key(stage)])
    tiles = ", ".join(f"{axis.name}={axis.tile}" for axis in best.domain.compute_axes)
    print(f"  backward embed ({tiles}): {times[stage_key(best)]:.1f} us")
    return best
