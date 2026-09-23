from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import polars as pl
from cutilereduce.core import symbolic as expr

from cutilereduce.core.axis import Axis, AxisId, Axes, LogicalAxis, axis_id
from cutilereduce.core.stage_domain import AxisRole, ComputeAxis
from cutilereduce.core.sweep import (
    Sweep,
    bounded_tile_configs,
    partition_configs_for_tiles,
)
from cutilereduce.fold.plan import FoldSpec, StageSchedule
from cutilereduce.stages import normalize_axis_mapping


@dataclass(frozen=True)
class FoldSweepSymbols:
    full_tiles: Mapping[AxisId, Any]
    partial_tiles: Mapping[AxisId, Any]
    combine_tiles: Mapping[AxisId, Any]
    backward_tiles: Mapping[AxisId, Any]
    partition_count: Any = expr.Symbol("partition_count")

    def full_tile(self, axis: Axis | AxisId) -> Any:
        return self.full_tiles[axis_id(axis)]

    def partial_tile(self, axis: Axis | AxisId) -> Any:
        return self.partial_tiles[axis_id(axis)]

    def combine_tile(self, axis: Axis | AxisId) -> Any:
        return self.combine_tiles[axis_id(axis)]

    def backward_tile(self, axis: Axis | AxisId) -> Any:
        return self.backward_tiles[axis_id(axis)]


def powers_of_two(stop: int) -> tuple[int, ...]:
    values = []
    x = 1
    while x <= stop:
        values.append(x)
        x *= 2
    return tuple(values)


def _normalize_sizes(
    spec: FoldSpec, sizes: Mapping[str | Axis | AxisId, int]
) -> dict[AxisId, int]:
    return normalize_axis_mapping(spec, sizes)


def _logical_axis(axis: Axis) -> LogicalAxis:
    if not isinstance(axis, LogicalAxis):
        raise TypeError(f"expected logical axis: {axis!r}")
    return axis


def _tile_axes(spec: FoldSpec) -> tuple[LogicalAxis, ...]:
    return tuple(_logical_axis(axis) for axis in (*spec.batch, spec.fold))


def _backward_loop_axes(spec: FoldSpec) -> tuple[LogicalAxis, ...]:
    contribution_axes = Axes(values=_tile_axes(spec))
    return tuple(
        _logical_axis(axis)
        for axis in spec.grad_storage.contention_axes(contribution_axes)
    )


def _make_symbols(spec: FoldSpec) -> FoldSweepSymbols:
    tile_axes = _tile_axes(spec)
    full = {
        axis.id: ComputeAxis.make(
            axis,
            AxisRole.Fold if axis == spec.fold else AxisRole.Batch,
            namespace="full",
        ).tile
        for axis in tile_axes
    }
    partial = {
        axis.id: ComputeAxis.make(
            axis,
            AxisRole.Fold if axis == spec.fold else AxisRole.Batch,
            namespace="partial",
        ).tile
        for axis in tile_axes
    }
    backward = {
        axis.id: ComputeAxis.make(
            axis,
            AxisRole.Fold if axis == spec.fold else AxisRole.Batch,
            namespace="backward",
        ).tile
        for axis in tile_axes
    }
    return FoldSweepSymbols(
        full_tiles=full,
        partial_tiles=partial,
        combine_tiles={
            axis.id: ComputeAxis.make(axis, AxisRole.Batch, namespace="combine").tile
            for axis in spec.batch
        },
        backward_tiles=backward,
    )


def _extent_mapping(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    extra: Mapping[AxisId, int | expr.Expr] | None = None,
) -> Mapping[AxisId, int | expr.Expr]:
    return MappingProxyType(
        {axis.id: sizes[axis.id] for axis in spec.axes} | dict(extra or {})
    )


def _config_values(extent: int, max_tile: int | None) -> tuple[int, ...]:
    return powers_of_two(min(extent, max_tile) if max_tile else extent)


def _tile_rows(names: tuple[str, ...], values: tuple[tuple[int, ...], ...]):
    for row in itertools.product(*values):
        yield dict(zip(names, row, strict=True))


def _stage_tile_configs(
    axes: tuple[Axis, ...],
    names: tuple[str, ...],
    sizes: Mapping[AxisId, int],
    max_tile: int,
    *,
    stage=None,
    symbols=None,
) -> pl.DataFrame:
    if stage is not None:
        return bounded_tile_configs(
            stage,
            axes,
            names,
            sizes,
            max_tile,
            symbols=symbols,
        )
    values = tuple(_config_values(sizes[axis.id], max_tile) for axis in axes)
    return pl.DataFrame(_tile_rows(names, values))


def generate_commutative_fold_configs(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    max_tile: int | None = None,
    max_partition_count: int = 16,
) -> pl.DataFrame:
    normalized_sizes = _normalize_sizes(spec, sizes)
    symbols = _make_symbols(spec)
    tile_axes = _tile_axes(spec)
    names = tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in tile_axes)
    tile_values = tuple(
        _config_values(normalized_sizes[axis.id], max_tile) for axis in tile_axes
    )
    rows = []
    for values in itertools.product(*tile_values, range(1, max_partition_count + 1)):
        *tiles, partition_count = values
        rows.append(
            {
                **dict(zip(names, tiles, strict=True)),
                f"cfg:{symbols.partition_count}": partition_count,
            }
        )
    return pl.DataFrame(rows)


def _single_configs(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    symbols: FoldSweepSymbols,
    max_tile: int,
    *,
    stage=None,
    substitutions=None,
) -> pl.DataFrame:
    axes = _tile_axes(spec)
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.full_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
        stage=stage,
        symbols=substitutions,
    )
    return configs.with_columns(
        pl.lit(1, dtype=pl.Int64).alias(f"cfg:{symbols.partition_count}")
    )


def _partial_configs(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    symbols: FoldSweepSymbols,
    max_tile: int,
    max_partition_count: int | None,
    *,
    stage=None,
    substitutions=None,
    max_storage_ratio: float = 4.0,
) -> pl.DataFrame:
    axes = _tile_axes(spec)
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
    )
    if stage is not None:
        return partition_configs_for_tiles(
            stage,
            configs,
            count_name=f"cfg:{symbols.partition_count}",
            fold_tile_name=f"cfg:{symbols.partial_tile(spec.fold)}",
            fold_extent=sizes[spec.fold.id],
            max_storage_ratio=max_storage_ratio,
            symbols=substitutions,
            explicit_cap=max_partition_count,
        )
    counts = pl.DataFrame(
        {f"cfg:{symbols.partition_count}": range(2, max_partition_count + 1)}
    )
    return configs.join(counts, how="cross").filter(
        pl.col(f"cfg:{symbols.partition_count}")
        <= (sizes[spec.fold.id] + pl.col(f"cfg:{symbols.partial_tile(spec.fold)}") - 1)
        // pl.col(f"cfg:{symbols.partial_tile(spec.fold)}")
    )


def _combine_configs(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    symbols: FoldSweepSymbols,
    max_tile: int,
    max_partition_count: int | None,
    *,
    stage=None,
    substitutions=None,
    partition_counts=None,
) -> pl.DataFrame:
    axes = tuple(spec.batch)
    configs = _stage_tile_configs(
        axes,
        tuple(f"cfg:{symbols.combine_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
        stage=stage,
        symbols=substitutions,
    )
    if partition_counts is None:
        if max_partition_count is None:
            return configs
        partition_counts = range(2, max_partition_count + 1)
    counts = pl.DataFrame({f"cfg:{symbols.partition_count}": partition_counts})
    return configs.join(counts, how="cross")


def _evaluate_single_path(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    symbols: FoldSweepSymbols,
    configs: pl.DataFrame,
    *,
    hardware: Mapping,
    sweep: Sweep = Sweep.default,
):
    from cutilereduce.fold.paths import FullForwardPlan, sizes_for

    stage = FullForwardPlan(spec).stages[0]
    size_values = {
        symbol: sizes[axis_id] for axis_id, symbol in sizes_for(spec).items()
    }
    evaluated = sweep.apply(stage.stage, configs, symbols=dict(hardware) | size_values)
    if evaluated.is_empty():
        return evaluated
    return evaluated.with_columns(
        pl.lit("single").alias("path"),
        pl.col("estimated_time").alias("fold_estimated_time"),
        pl.lit(0.0).alias("combine_estimated_time"),
    ).with_columns(pl.col("fold_estimated_time").alias("forward_estimated_time"))


def _evaluate_partial_path(
    spec: FoldSpec,
    sizes: Mapping[AxisId, int],
    symbols: FoldSweepSymbols,
    configs: pl.DataFrame,
    *,
    hardware: Mapping,
    max_tile: int,
    max_partition_count: int,
    max_storage_ratio: float = 4.0,
    sweep: Sweep = Sweep.default,
):
    from cutilereduce.fold.paths import CommutativePartialForwardPlan, sizes_for

    size_values = {
        symbol: sizes[axis_id] for axis_id, symbol in sizes_for(spec).items()
    }
    frontier_sweep = sweep.add_filters(
        pl.col("partial_storage_ratio") <= max_storage_ratio,
    ).with_frontier_keys(f"cfg:{symbols.partition_count}")
    partial_stage, combine_stage = CommutativePartialForwardPlan(spec).stages
    fold = frontier_sweep.apply(
        partial_stage.stage, configs, symbols=dict(hardware) | size_values
    )
    if fold.is_empty():
        return fold

    combine_configs = _combine_configs(
        spec,
        sizes,
        symbols,
        max_tile,
        max_partition_count,
        stage=combine_stage.stage,
        substitutions=dict(hardware) | size_values,
        partition_counts=(
            configs.select(f"cfg:{symbols.partition_count}")
            .unique()
            .to_series()
            .to_list()
            if f"cfg:{symbols.partition_count}" in configs.columns
            else ()
        ),
    )
    combine = frontier_sweep.apply(
        combine_stage.stage, combine_configs, symbols=dict(hardware) | size_values
    )
    if combine.is_empty():
        return combine

    joined = fold.join(combine, on=f"cfg:{symbols.partition_count}", suffix="_combine")
    return joined.with_columns(
        (pl.col("estimated_time") + pl.col("estimated_time_combine")).alias(
            "forward_estimated_time"
        ),
        pl.col("estimated_time").alias("fold_estimated_time"),
        pl.col("estimated_time_combine").alias("combine_estimated_time"),
        pl.lit("partial").alias("path"),
    ).sort("forward_estimated_time")


def sweep_commutative_fold(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    hardware: Mapping,
    max_tile: int | None = None,
    max_fold_programs: int | None = None,
    max_partition_count: int | None = None,
    max_storage_ratio: float = 4.0,
    sweep: Sweep = Sweep.default,
    configs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    max_partition_count = max_partition_count or max_fold_programs
    normalized_sizes = _normalize_sizes(spec, sizes)
    symbols = _make_symbols(spec)

    from cutilereduce.fold.paths import CommutativePartialForwardPlan, FullForwardPlan
    full_stage = FullForwardPlan(spec).stages[0].stage
    partial_stage, _ = CommutativePartialForwardPlan(spec).stages
    single_configs = _single_configs(
        spec,
        normalized_sizes,
        symbols,
        max_tile,
        stage=full_stage,
        substitutions=hardware,
    )
    partial_configs = (
        configs
        if configs is not None
        else _partial_configs(
            spec, normalized_sizes, symbols, max_tile, max_partition_count,
            stage=partial_stage.stage,
            substitutions=hardware,
            max_storage_ratio=max_storage_ratio,
        )
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
        max_storage_ratio=max_storage_ratio,
        sweep=sweep,
    )
    frames = tuple(frame for frame in (single, partial) if not frame.is_empty())
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort("forward_estimated_time")


def sweep_commutative_backward(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    hardware: Mapping,
    max_tile: int | None = None,
    max_group_count: int = 4,
    sweep: Sweep = Sweep.default,
) -> pl.DataFrame:
    normalized_sizes = _normalize_sizes(spec, sizes)
    from cutilereduce.fold.paths import backward_plans, sizes_for, symbol

    size_values = {
        symbol: normalized_sizes[axis_id] for axis_id, symbol in sizes_for(spec).items()
    }
    symbols = _make_symbols(spec)
    axes = _tile_axes(spec)
    frames = []
    for loop_axis in _backward_loop_axes(spec):
        stage = backward_plans(spec)[0].with_loop(loop_axis).stages[1]
        tile_configs = _stage_tile_configs(
            axes,
            tuple(f"cfg:{symbols.backward_tile(axis)}" for axis in axes),
            normalized_sizes,
            max_tile,
            stage=stage.stage,
            symbols=dict(hardware) | size_values,
        )
        counts = pl.DataFrame(
            {
                f"cfg:{symbols.partition_count}": range(1, max_group_count + 1),
            }
        )
        configs = tile_configs.join(counts, how="cross")
        loop_configs = configs.filter(
            pl.col(f"cfg:{symbols.partition_count}")
            <= (
                normalized_sizes[loop_axis.id]
                + pl.col(f"cfg:{symbols.backward_tile(loop_axis)}")
                - 1
            )
            // pl.col(f"cfg:{symbols.backward_tile(loop_axis)}")
        )
        evaluated = sweep.with_frontier_keys(f"cfg:{symbols.partition_count}").apply(
            stage.stage,
            loop_configs,
            symbols=dict(hardware)
            | size_values
            | {
                symbol("backward_group_count"): symbols.partition_count,
            },
        )
        if not evaluated.is_empty():
            frames.append(
                evaluated.with_columns(pl.lit(loop_axis.name).alias("loop_axis"))
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort("estimated_time")


def _config_value(config: Mapping[str, object], symbol: Any) -> int:
    name = f"cfg:{symbol}"
    value = config.get(name)
    if not isinstance(value, int):
        raise TypeError(f"configuration value for {name} is not an integer: {value!r}")
    return value


def commutative_fold_plan_from_config(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    config: Mapping[str, object],
    backward: bool = False,
):
    """Build a concrete full or partial plan from one sweep result row."""
    from cutilereduce.fold.paths import backward_plans, forward_plans
    from cutilereduce.fold.templates import FoldPlanTemplate

    normalized_sizes = _normalize_sizes(spec, sizes)
    symbols = _make_symbols(spec)
    path = {"single": "full", "partial": "partial"}.get(config["path"])
    if path is None:
        raise ValueError(f"unknown fold path: {config['path']!r}")
    values = {axis.name: normalized_sizes[axis.id] for axis in spec.axes}
    tile = symbols.full_tile if path == "full" else symbols.partial_tile
    for axis in _tile_axes(spec):
        values[f"{path}_{axis.name}_tile"] = _config_value(config, tile(axis))
    if path == "partial":
        values["partition_count"] = _config_value(config, symbols.partition_count)
        for axis in spec.batch:
            values[f"combine_{axis.name}_tile"] = _config_value(
                config, symbols.combine_tile(axis)
            )
    if backward:
        loop = spec.fold.name
        for axis in _tile_axes(spec):
            values[f"backward_{axis.name}_tile"] = values[f"{path}_{axis.name}_tile"]
        values["backward_group_count"] = 1
        for axis in spec.batch:
            values[f"embed_{axis.name}_tile"] = normalized_sizes[axis.id]
        kind = "commutative"
    else:
        kind = None
    forward = next(plan for plan in forward_plans(spec) if plan.name == path)
    backward_plan = (
        None
        if kind is None
        else next(plan for plan in backward_plans(spec) if plan.name == kind).with_loop(
            loop
        )
    )
    return FoldPlanTemplate(spec, forward, backward_plan).bind(values)


def commutative_backward_stage_from_config(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    config: Mapping[str, object],
):
    normalized_sizes = _normalize_sizes(spec, sizes)
    symbols = _make_symbols(spec)
    loop_name = config.get("loop_axis")
    loop_axis = next(
        (axis for axis in _tile_axes(spec) if axis.name == loop_name),
        None,
    )
    if loop_axis is None:
        raise ValueError(f"unknown backward loop axis: {loop_name!r}")
    partition_axis = loop_axis.partition_axis
    group_count = _config_value(config, symbols.partition_count)
    tile_axes = set(axis.id for axis in _tile_axes(spec))
    schedule = StageSchedule(
        extents=_extent_mapping(spec, normalized_sizes),
        tiles=MappingProxyType(
            {
                axis.id: _config_value(config, symbols.backward_tile(axis))
                if axis.id in tile_axes
                else normalized_sizes[axis.id]
                for axis in spec.axes
            }
        ),
        programs=MappingProxyType(
            {} if group_count == 1 else {partition_axis.id: group_count}
        ),
        loop=loop_axis.id,
    )
    from cutilereduce.fold.paths import backward_plans

    return backward_plans(spec)[0].with_loop(loop_axis).finalize(schedule)


__all__ = [
    "FoldSweepSymbols",
    "commutative_backward_stage_from_config",
    "commutative_fold_plan_from_config",
    "generate_commutative_fold_configs",
    "powers_of_two",
    "sweep_commutative_backward",
    "sweep_commutative_fold",
]
