from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import polars as pl
import sympy

from cutilereduce.core.axis import Axis, AxisId, axis_id
from cutilereduce.core.sweep import Sweep
from cutilereduce.fold.plan import FoldSpec, StageSchedule
from cutilereduce.stages import Fold, MapFold, MapFoldPartial, normalize_axis_mapping
from cutilereduce.util.spec import l4


@dataclass(frozen=True)
class FoldSweepSymbols:
    full_tiles: Mapping[AxisId, sympy.Symbol]
    partial_tiles: Mapping[AxisId, sympy.Symbol]
    combine_tiles: Mapping[AxisId, sympy.Symbol]
    partition_count: sympy.Symbol = sympy.Symbol("partition_count")

    def full_tile(self, axis: Axis | AxisId) -> sympy.Symbol:
        return self.full_tiles[axis_id(axis)]

    def partial_tile(self, axis: Axis | AxisId) -> sympy.Symbol:
        return self.partial_tiles[axis_id(axis)]

    def combine_tile(self, axis: Axis | AxisId) -> sympy.Symbol:
        return self.combine_tiles[axis_id(axis)]


def powers_of_two(stop: int) -> tuple[int, ...]:
    values = []
    x = 1
    while x <= stop:
        values.append(x)
        x *= 2
    return tuple(values)


def _normalize_sizes(spec: FoldSpec, sizes: Mapping[str | Axis | AxisId, int]) -> dict[AxisId, int]:
    return normalize_axis_mapping(spec, sizes)


def _tile_axes(spec: FoldSpec) -> tuple[Axis, ...]:
    return (*spec.batch, spec.fold)


def _make_symbols(spec: FoldSpec, partition_axis: Axis | None = None) -> FoldSweepSymbols:
    partition_axis = partition_axis or spec.fold.partition_axis
    tile_axes = _tile_axes(spec)
    return FoldSweepSymbols(
        full_tiles={axis.id: sympy.Symbol(f"full_{axis.name}_tile") for axis in tile_axes},
        partial_tiles={axis.id: sympy.Symbol(f"partial_{axis.name}_tile") for axis in tile_axes},
        combine_tiles={
            **{axis.id: sympy.Symbol(f"combine_{axis.name}_tile") for axis in spec.batch},
            partition_axis.id: sympy.Symbol(f"combine_{partition_axis.name}_tile"),
        },
    )


def _extent_mapping(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        extra: Mapping[AxisId, int | sympy.Expr] | None = None,
        ) -> Mapping[AxisId, int | sympy.Expr]:
    return MappingProxyType({axis.id: sizes[axis.id] for axis in spec.axes} | dict(extra or {}))


def _full_schedule(spec: FoldSpec, sizes: Mapping[AxisId, int], symbols: FoldSweepSymbols) -> StageSchedule:
    tile_axes = set(axis.id for axis in _tile_axes(spec))
    return StageSchedule(
        extents=_extent_mapping(spec, sizes),
        tiles=MappingProxyType({
            axis.id: symbols.full_tile(axis) if axis.id in tile_axes else sizes[axis.id]
            for axis in spec.axes
        }),
        programs=MappingProxyType({}),
        loop=spec.fold.id,
    )


def _partial_schedule(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        partition_axis: Axis,
        ) -> StageSchedule:
    tile_axes = set(axis.id for axis in _tile_axes(spec))
    return StageSchedule(
        extents=_extent_mapping(spec, sizes),
        tiles=MappingProxyType({
            axis.id: symbols.partial_tile(axis) if axis.id in tile_axes else sizes[axis.id]
            for axis in spec.axes
        }),
        programs=MappingProxyType({partition_axis.id: symbols.partition_count}),
        loop=spec.fold.id,
    )


def _combine_schedule(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        partition_axis: Axis,
        ) -> StageSchedule:
    return StageSchedule(
        extents=_extent_mapping(spec, sizes, {partition_axis.id: symbols.partition_count}),
        tiles=MappingProxyType({
            **{
                axis.id: symbols.combine_tile(axis)
                for axis in spec.batch
            },
            **{
                axis.id: sizes[axis.id]
                for axis in spec.axes
                if axis not in spec.batch and axis != spec.fold
            },
            partition_axis.id: symbols.combine_tile(partition_axis),
        }),
        programs=MappingProxyType({partition_axis.id: 1}),
        loop=partition_axis.id,
    )


def _config_values(extent: int, max_tile: int) -> tuple[int, ...]:
    return powers_of_two(min(extent, max_tile))


def _tile_rows(names: tuple[str, ...], values: tuple[tuple[int, ...], ...]):
    for row in itertools.product(*values):
        yield dict(zip(names, row, strict=True))


def _stage_tile_configs(
        axes: tuple[Axis, ...],
        names: tuple[str, ...],
        sizes: Mapping[AxisId, int],
        max_tile: int,
        ) -> pl.DataFrame:
    values = tuple(_config_values(sizes[axis.id], max_tile) for axis in axes)
    return pl.DataFrame(_tile_rows(names, values))


def generate_commutative_fold_configs(
        spec: FoldSpec,
        *,
        sizes: Mapping[str | Axis | AxisId, int],
        max_tile: int = 256,
        max_partition_count: int = 16,
        ) -> pl.DataFrame:
    normalized_sizes = _normalize_sizes(spec, sizes)
    partition_axis = spec.fold.partition_axis
    symbols = _make_symbols(spec, partition_axis)
    tile_axes = _tile_axes(spec)
    names = tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in tile_axes)
    tile_values = tuple(_config_values(normalized_sizes[axis.id], max_tile) for axis in tile_axes)
    rows = []
    for values in itertools.product(*tile_values, range(1, max_partition_count + 1)):
        *tiles, partition_count = values
        rows.append({
            **dict(zip(names, tiles, strict=True)),
            f"cfg:{symbols.partition_count}": partition_count,
        })
    return pl.DataFrame(rows)


def _single_configs(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        max_tile: int,
        ) -> pl.DataFrame:
    axes = _tile_axes(spec)
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.full_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
    )
    return configs.with_columns(pl.lit(1).alias(f"cfg:{symbols.partition_count}"))


def _partial_configs(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        max_tile: int,
        max_partition_count: int,
        ) -> pl.DataFrame:
    axes = _tile_axes(spec)
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
    )
    counts = pl.DataFrame({f"cfg:{symbols.partition_count}": range(2, max_partition_count + 1)})
    return configs.join(counts, how="cross")


def _combine_configs(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        partition_axis: Axis,
        max_tile: int,
        max_partition_count: int,
        ) -> pl.DataFrame:
    axes = (*spec.batch, partition_axis)
    config_sizes = dict(sizes) | {partition_axis.id: max_partition_count}
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.combine_tile(axis)}" for axis in axes),
        config_sizes,
        max_tile,
    )
    counts = pl.DataFrame({f"cfg:{symbols.partition_count}": range(2, max_partition_count + 1)})
    return configs.join(counts, how="cross")


def _evaluate_single_path(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        configs: pl.DataFrame,
        *,
        hardware: Mapping = l4,
        sweep: Sweep = Sweep.default,
        ):
    stage = MapFold(spec, _full_schedule(spec, sizes, symbols)).build()
    evaluated = sweep.apply(stage.stage, configs, symbols=hardware)
    if evaluated.is_empty():
        return evaluated
    return (
        evaluated.with_columns(
            pl.lit("single").alias("path"),
            pl.col("estimated_time").alias("fold_estimated_time"),
            pl.lit(0.0).alias("combine_estimated_time"),
        )
        .with_columns(pl.col("fold_estimated_time").alias("forward_estimated_time"))
    )


def _evaluate_partial_path(
        spec: FoldSpec,
        sizes: Mapping[AxisId, int],
        symbols: FoldSweepSymbols,
        configs: pl.DataFrame,
        *,
        hardware: Mapping = l4,
        max_tile: int,
        max_partition_count: int,
        sweep: Sweep = Sweep.default,
        ):
    partition_axis = spec.fold.partition_axis
    partial = MapFoldPartial.make(
        spec,
        _partial_schedule(spec, sizes, symbols, partition_axis),
    )
    fold = sweep.apply(partial.build().stage, configs, symbols=hardware)
    if fold.is_empty():
        return fold

    combine_configs = _combine_configs(
        spec,
        sizes,
        symbols,
        partition_axis,
        max_tile,
        max_partition_count,
    )
    combine_stage = Fold(
        spec=spec,
        schedule=_combine_schedule(spec, sizes, symbols, partition_axis),
        partition_axis=partition_axis,
        partials=partial.partials,
    ).build()
    combine = sweep.apply(combine_stage.stage, combine_configs, symbols=hardware)
    if combine.is_empty():
        return combine

    joined = fold.join(combine, on=f"cfg:{symbols.partition_count}", suffix="_combine")
    return (
        joined.with_columns(
            (pl.col("estimated_time") + pl.col("estimated_time_combine")).alias("forward_estimated_time"),
            pl.col("estimated_time").alias("fold_estimated_time"),
            pl.col("estimated_time_combine").alias("combine_estimated_time"),
            pl.lit("partial").alias("path"),
        )
        .sort("forward_estimated_time")
    )


def sweep_commutative_fold(
        spec: FoldSpec,
        *,
        sizes: Mapping[str | Axis | AxisId, int],
        hardware: Mapping = l4,
        max_tile: int = 256,
        max_fold_programs: int | None = None,
        max_partition_count: int | None = None,
        sweep: Sweep = Sweep.default,
        configs: pl.DataFrame | None = None,
        ) -> pl.DataFrame:
    max_partition_count = max_partition_count or max_fold_programs or 16
    normalized_sizes = _normalize_sizes(spec, sizes)
    partition_axis = spec.fold.partition_axis
    symbols = _make_symbols(spec, partition_axis)

    single_configs = _single_configs(spec, normalized_sizes, symbols, max_tile)
    partial_configs = (
        configs
        if configs is not None
        else _partial_configs(spec, normalized_sizes, symbols, max_tile, max_partition_count)
    )
    single = _evaluate_single_path(
        spec,
        normalized_sizes,
        symbols,
        single_configs,
        hardware=hardware,
        sweep=sweep,
    )
    partial = _evaluate_partial_path(
        spec,
        normalized_sizes,
        symbols,
        partial_configs,
        hardware=hardware,
        max_tile=max_tile,
        max_partition_count=max_partition_count,
        sweep=sweep,
    )
    frames = tuple(frame for frame in (single, partial) if not frame.is_empty())
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort("forward_estimated_time")


__all__ = [
    "FoldSweepSymbols",
    "generate_commutative_fold_configs",
    "powers_of_two",
    "sweep_commutative_fold",
]
