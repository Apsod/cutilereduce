from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import polars as pl
from cutilereduce.core import symbolic as expr

from cutilereduce.core.axis import Axis, AxisId, axis_id
from cutilereduce.core.stage_domain import AxisRole, ComputeAxis
from cutilereduce.core.sweep import (
    Sweep,
    bounded_tile_configs,
    partition_configs_for_tiles,
)
from cutilereduce.fold.plan import FoldSpec, StageSchedule
from cutilereduce.stages import normalize_axis_mapping


@dataclass(frozen=True)
class GeneralFoldSweepSymbols:
    full_tiles: Mapping[AxisId, Any]
    partial_tiles: Mapping[AxisId, Any]
    scan_tiles: Mapping[AxisId, Any]
    backward_tiles: Mapping[AxisId, Any]
    partition_count: Any = expr.Symbol("partition_count")

    def full_tile(self, axis: Axis) -> Any:
        return self.full_tiles[axis_id(axis)]

    def partial_tile(self, axis: Axis) -> Any:
        return self.partial_tiles[axis_id(axis)]

    def scan_tile(self, axis: Axis) -> Any:
        return self.scan_tiles[axis_id(axis)]

    def backward_tile(self, axis: Axis) -> Any:
        return self.backward_tiles[axis_id(axis)]


def _powers_of_two(stop: int) -> tuple[int, ...]:
    values = []
    value = 1
    while value <= stop:
        values.append(value)
        value *= 2
    return tuple(values)


def _normalize_sizes(spec, sizes):
    return normalize_axis_mapping(spec, sizes)


def _map_axes(spec) -> tuple[Axis, ...]:
    return (*spec.batch, spec.fold)


def _symbols(spec) -> GeneralFoldSweepSymbols:
    full = {
        axis.id: ComputeAxis.make(axis, AxisRole.Batch, namespace="full").tile
        for axis in _map_axes(spec)
    }
    partial = {
        axis.id: ComputeAxis.make(axis, AxisRole.Batch, namespace="partial").tile
        for axis in _map_axes(spec)
    }
    scan = {
        axis.id: ComputeAxis.make(axis, AxisRole.Batch, namespace="combine").tile
        for axis in spec.batch
    }
    backward = {
        axis.id: ComputeAxis.make(axis, AxisRole.Batch, namespace="backward").tile
        for axis in _map_axes(spec)
    }
    return GeneralFoldSweepSymbols(
        full_tiles=full,
        partial_tiles=partial,
        scan_tiles=scan,
        backward_tiles=backward,
    )


def _extents(spec, sizes, extra=None):
    return MappingProxyType(
        {axis.id: sizes[axis.id] for axis in spec.axes} | dict(extra or {})
    )


def _tile_configs(axes, names, sizes, max_tile=None):
    values = tuple(
        _powers_of_two(min(sizes[axis.id], max_tile) if max_tile else sizes[axis.id])
        for axis in axes
    )
    return pl.DataFrame(
        dict(zip(names, row, strict=True)) for row in itertools.product(*values)
    )


def sweep_general_fold(
    spec: FoldSpec,
    *,
    sizes: Mapping[str | Axis | AxisId, int],
    hardware: Mapping,
    max_tile: int | None = None,
    max_partition_count: int | None = None,
    max_storage_ratio: float = 4.0,
    sweep: Sweep = Sweep.default,
) -> pl.DataFrame:
    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    map_axes = _map_axes(spec)
    count_name = f"cfg:{symbols.partition_count}"

    from cutilereduce.fold.paths import (
        FullForwardPlan,
        GeneralPartialForwardPlan,
        sizes_for,
    )
    full_stage = FullForwardPlan(spec).stages[0]
    partial_stage, scan_stage = GeneralPartialForwardPlan(spec).stages

    size_values = {
        symbol: sizes[axis_id] for axis_id, symbol in sizes_for(spec).items()
    }
    bound_symbols = dict(hardware) | size_values
    full_configs = bounded_tile_configs(
        full_stage.stage,
        map_axes,
        tuple(f"cfg:{symbols.full_tile(axis)}" for axis in map_axes),
        sizes,
        max_tile,
        symbols=bound_symbols,
    ).with_columns(pl.lit(1, dtype=pl.Int64).alias(count_name))
    full = sweep.apply(
        full_stage.stage, full_configs, symbols=bound_symbols
    )
    if not full.is_empty():
        full = full.with_columns(
            pl.lit("full").alias("path"),
            pl.col("estimated_time").alias("forward_estimated_time"),
            pl.col("estimated_time").alias("map_estimated_time"),
            pl.lit(0.0).alias("scan_estimated_time"),
        )

    partial_tiles = bounded_tile_configs(
        partial_stage.stage,
        map_axes,
        tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in map_axes),
        sizes,
        max_tile,
        symbols=bound_symbols,
    )
    partial_configs = partition_configs_for_tiles(
        partial_stage.stage,
        partial_tiles,
        count_name=count_name,
        fold_tile_name=f"cfg:{symbols.partial_tile(spec.fold)}",
        fold_extent=sizes[spec.fold.id],
        max_storage_ratio=max_storage_ratio,
        symbols=bound_symbols,
        explicit_cap=max_partition_count,
    )
    partial_frontier = sweep.add_filters(
        pl.col("partial_storage_ratio") <= max_storage_ratio,
    ).with_frontier_keys(count_name)
    mapped = partial_frontier.apply(
        partial_stage.stage,
        partial_configs,
        symbols=bound_symbols,
    )

    scan_configs = bounded_tile_configs(
        scan_stage.stage,
        tuple(spec.batch),
        tuple(f"cfg:{symbols.scan_tile(axis)}" for axis in spec.batch),
        sizes,
        max_tile,
        symbols=bound_symbols,
    )
    if not partial_configs.is_empty():
        scan_counts = partial_configs.select(count_name).unique()
        scan_configs = scan_configs.join(scan_counts, how="cross")
    # Scan reuses the materialized partial allocation; it does not make the
    # partition-count storage decision, so do not prune it by that ratio again.
    scanned = sweep.with_frontier_keys(count_name).apply(
        scan_stage.stage,
        scan_configs,
        symbols=bound_symbols,
    )

    partial_frame = pl.DataFrame()
    if not mapped.is_empty() and not scanned.is_empty():
        partial_frame = mapped.join(scanned, on=count_name, suffix="_scan")
        partial_frame = partial_frame.with_columns(
            (pl.col("estimated_time") + pl.col("estimated_time_scan")).alias(
                "forward_estimated_time"
            ),
            pl.col("estimated_time").alias("map_estimated_time"),
            pl.col("estimated_time_scan").alias("scan_estimated_time"),
            pl.lit("partial").alias("path"),
        )

    frames = tuple(frame for frame in (full, partial_frame) if not frame.is_empty())
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort("forward_estimated_time")


def _config_value(config, symbol):
    name = f"cfg:{symbol}"
    value = config.get(name)
    if not isinstance(value, int):
        raise TypeError(f"configuration value for {name} is not an integer: {value!r}")
    return value


def general_fold_plan_from_config(spec, *, sizes, config):
    from cutilereduce.fold.paths import forward_plans
    from cutilereduce.fold.templates import FoldPlanTemplate

    sizes = _normalize_sizes(spec, sizes)
    path = config["path"]
    if path not in ("full", "partial"):
        raise ValueError(f"unknown general fold path: {path!r}")
    symbols = _symbols(spec)
    values = {axis.name: sizes[axis.id] for axis in spec.axes}
    tile = symbols.full_tile if path == "full" else symbols.partial_tile
    for axis in _map_axes(spec):
        values[f"{path}_{axis.name}_tile"] = _config_value(config, tile(axis))
    if path == "partial":
        values["partition_count"] = _config_value(config, symbols.partition_count)
        for axis in spec.batch:
            values[f"combine_{axis.name}_tile"] = _config_value(
                config, symbols.scan_tile(axis)
            )
    forward = next(plan for plan in forward_plans(spec) if plan.name == path)
    return FoldPlanTemplate(spec, forward).bind(values)


def sweep_general_backward(
    spec,
    *,
    sizes,
    forward_plan,
    hardware,
    max_tile=None,
    sweep=Sweep.default,
):
    sizes = _normalize_sizes(spec, sizes)
    from cutilereduce.fold.paths import backward_plans, sizes_for, symbol

    size_values = {sym: sizes[axis_id] for axis_id, sym in sizes_for(spec).items()}
    symbols = _symbols(spec)
    axes = _map_axes(spec)
    full_path, checkpoint_path = backward_plans(spec)
    full = full_path.stages[1]
    full_configs = bounded_tile_configs(
        full.stage,
        axes,
        tuple(f"cfg:{symbols.backward_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
        symbols=dict(hardware) | size_values,
    )
    full_frame = sweep.apply(
        full.stage, full_configs, symbols=dict(hardware) | size_values
    )
    frames = []
    if not full_frame.is_empty():
        frames.append(
            full_frame.with_columns(pl.lit("full_recompute").alias("backward_path"))
        )

    if len(forward_plan.forward) > 1:
        partial_stage, scan_stage = forward_plan.forward
        fold_tile = partial_stage.stage.domain.get(spec.fold).tile
        partition_axis = partial_stage.partition_axis
        partition_count = scan_stage.stage.domain.get(partition_axis).extent
        prefix = checkpoint_path.stages[1]
        prefix_values = (
            dict(hardware)
            | size_values
            | {
                symbol(f"partial_{spec.fold.name}_tile"): int(fold_tile),
                symbol("partition_count"): int(partition_count),
            }
        )
        batch_configs = bounded_tile_configs(
            prefix.stage,
            tuple(spec.batch),
            tuple(f"cfg:{symbols.backward_tile(axis)}" for axis in spec.batch),
            sizes,
            max_tile,
            symbols=prefix_values,
        ).with_columns(
            pl.lit(int(fold_tile), dtype=pl.Int64).alias(
                f"cfg:{symbols.backward_tile(spec.fold)}"
            )
        )
        prefix_frame = sweep.apply(prefix.stage, batch_configs, symbols=prefix_values)
        if not prefix_frame.is_empty():
            frames.append(
                prefix_frame.with_columns(pl.lit("checkpointed").alias("backward_path"))
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort("estimated_time")


def general_backward_stage_from_config(spec, *, sizes, forward_plan, config):
    from cutilereduce.fold.paths import backward_plans

    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    tiles = {
        axis.id: _config_value(config, symbols.backward_tile(axis))
        if axis in _map_axes(spec)
        else sizes[axis.id]
        for axis in spec.axes
    }
    path = config["backward_path"]
    if path == "full_recompute":
        schedule = StageSchedule(
            extents=_extents(spec, sizes),
            tiles=MappingProxyType(tiles),
            programs=MappingProxyType({}),
            loop=spec.fold.id,
        )
        return backward_plans(spec)[0].finalize(schedule)
    if path != "checkpointed" or len(forward_plan.forward) <= 1:
        raise ValueError(f"invalid general backward path: {path!r}")
    partial_stage, scan_stage = forward_plan.forward
    partition_axis = partial_stage.partition_axis
    partition_count = int(scan_stage.stage.domain.get(partition_axis).extent)
    schedule = StageSchedule(
        extents=_extents(spec, sizes),
        tiles=MappingProxyType(tiles),
        programs=MappingProxyType({partition_axis.id: partition_count}),
        loop=spec.fold.id,
    )
    return backward_plans(spec)[1].finalize(schedule, scan_stage.carriers)


__all__ = [
    "GeneralFoldSweepSymbols",
    "general_fold_plan_from_config",
    "general_backward_stage_from_config",
    "sweep_general_backward",
    "sweep_general_fold",
]
