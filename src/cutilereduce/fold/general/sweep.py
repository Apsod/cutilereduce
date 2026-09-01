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
from cutilereduce.stages import MapFold, MapFoldPartial, Scan, normalize_axis_mapping
from cutilereduce.util.spec import l4

from .plan import (
    full_fold_plan,
    full_recompute_backward_stage,
    partial_fold_plan,
    prefix_recompute_backward_stage,
)


@dataclass(frozen=True)
class GeneralFoldSweepSymbols:
    full_tiles: Mapping[AxisId, sympy.Symbol]
    partial_tiles: Mapping[AxisId, sympy.Symbol]
    scan_tiles: Mapping[AxisId, sympy.Symbol]
    backward_tiles: Mapping[AxisId, sympy.Symbol]
    partition_count: sympy.Symbol = sympy.Symbol("partition_count")

    def full_tile(self, axis: Axis) -> sympy.Symbol:
        return self.full_tiles[axis_id(axis)]

    def partial_tile(self, axis: Axis) -> sympy.Symbol:
        return self.partial_tiles[axis_id(axis)]

    def scan_tile(self, axis: Axis) -> sympy.Symbol:
        return self.scan_tiles[axis_id(axis)]

    def backward_tile(self, axis: Axis) -> sympy.Symbol:
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
    return GeneralFoldSweepSymbols(
        full_tiles={
            axis.id: sympy.Symbol(f"full_{axis.name}_tile")
            for axis in _map_axes(spec)
        },
        partial_tiles={
            axis.id: sympy.Symbol(f"partial_{axis.name}_tile")
            for axis in _map_axes(spec)
        },
        scan_tiles={
            axis.id: sympy.Symbol(f"scan_{axis.name}_tile")
            for axis in spec.batch
        },
        backward_tiles={
            axis.id: sympy.Symbol(f"backward_{axis.name}_tile")
            for axis in _map_axes(spec)
        },
    )


def _extents(spec, sizes, extra=None):
    return MappingProxyType(
        {axis.id: sizes[axis.id] for axis in spec.axes} | dict(extra or {})
    )


def _map_schedule(spec, sizes, symbols, *, partial):
    tile = symbols.partial_tile if partial else symbols.full_tile
    partition_axis = spec.fold.partition_axis
    return StageSchedule(
        extents=_extents(spec, sizes),
        tiles=MappingProxyType({
            axis.id: tile(axis)
            if axis in _map_axes(spec)
            else sizes[axis.id]
            for axis in spec.axes
        }),
        programs=MappingProxyType(
            {partition_axis.id: symbols.partition_count} if partial else {}
        ),
        loop=spec.fold.id,
    )


def _scan_schedule(spec, sizes, symbols):
    partition_axis = spec.fold.partition_axis
    return StageSchedule(
        extents=_extents(spec, sizes, {partition_axis.id: symbols.partition_count}),
        tiles=MappingProxyType({
            **{
                axis.id: symbols.scan_tile(axis)
                for axis in spec.batch
            },
            **{
                axis.id: sizes[axis.id]
                for axis in spec.axes
                if axis not in spec.batch and axis != spec.fold
            },
            partition_axis.id: 1,
        }),
        programs=MappingProxyType({partition_axis.id: 1}),
        loop=partition_axis.id,
    )


def _tile_configs(axes, names, sizes, max_tile):
    values = tuple(
        _powers_of_two(min(sizes[axis.id], max_tile)) for axis in axes
    )
    return pl.DataFrame(
        dict(zip(names, row, strict=True))
        for row in itertools.product(*values)
    )


def sweep_general_fold(
        spec: FoldSpec,
        *,
        sizes: Mapping[str | Axis | AxisId, int],
        hardware: Mapping = l4,
        max_tile: int = 128,
        max_partition_count: int = 4,
        sweep: Sweep = Sweep.default,
        ) -> pl.DataFrame:
    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    map_axes = _map_axes(spec)
    count_name = f"cfg:{symbols.partition_count}"

    full_configs = _tile_configs(
        map_axes,
        tuple(f"cfg:{symbols.full_tile(axis)}" for axis in map_axes),
        sizes,
        max_tile,
    ).with_columns(pl.lit(1, dtype=pl.Int64).alias(count_name))
    full_stage = MapFold(spec, _map_schedule(spec, sizes, symbols, partial=False)).build()
    full = sweep.apply(full_stage.stage, full_configs, symbols=hardware)
    if not full.is_empty():
        full = full.with_columns(
            pl.lit("full").alias("path"),
            pl.col("estimated_time").alias("forward_estimated_time"),
            pl.col("estimated_time").alias("map_estimated_time"),
            pl.lit(0.0).alias("scan_estimated_time"),
        )

    counts = pl.DataFrame({count_name: range(2, max_partition_count + 1)})
    partial_configs = _tile_configs(
        map_axes,
        tuple(f"cfg:{symbols.partial_tile(axis)}" for axis in map_axes),
        sizes,
        max_tile,
    ).join(counts, how="cross")
    partial = MapFoldPartial.make(
        spec,
        _map_schedule(spec, sizes, symbols, partial=True),
    )
    partial_frontier = sweep.add_filters(
        pl.col("partial_storage_ratio") <= 1,
    ).with_frontier_keys(count_name)
    mapped = partial_frontier.apply(
        partial.build().stage, partial_configs, symbols=hardware,
    )

    scan_configs = _tile_configs(
        tuple(spec.batch),
        tuple(f"cfg:{symbols.scan_tile(axis)}" for axis in spec.batch),
        sizes,
        max_tile,
    ).join(counts, how="cross")
    scan = Scan.make(
        spec,
        _scan_schedule(spec, sizes, symbols),
        scan_axis=spec.fold.partition_axis,
        inputs=partial.partials,
        outputs=spec.output,
    ).build()
    # Scan reuses the materialized partial allocation; it does not make the
    # partition-count storage decision, so do not prune it by that ratio again.
    scanned = sweep.with_frontier_keys(count_name).apply(
        scan.stage, scan_configs, symbols=hardware,
    )

    partial_frame = pl.DataFrame()
    if not mapped.is_empty() and not scanned.is_empty():
        partial_frame = mapped.join(scanned, on=count_name, suffix="_scan")
        partial_frame = partial_frame.with_columns(
            (
                pl.col("estimated_time") + pl.col("estimated_time_scan")
            ).alias("forward_estimated_time"),
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
    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    map_axes = set(axis.id for axis in _map_axes(spec))

    def concrete_map_schedule(*, partial):
        tile = symbols.partial_tile if partial else symbols.full_tile
        partition_axis = spec.fold.partition_axis
        count = _config_value(config, symbols.partition_count)
        return StageSchedule(
            extents=_extents(spec, sizes),
            tiles=MappingProxyType({
                axis.id: _config_value(config, tile(axis))
                if axis.id in map_axes
                else sizes[axis.id]
                for axis in spec.axes
            }),
            programs=MappingProxyType(
                {partition_axis.id: count} if partial else {}
            ),
            loop=spec.fold.id,
        )

    if config["path"] == "full":
        return full_fold_plan(spec, concrete_map_schedule(partial=False))
    if config["path"] != "partial":
        raise ValueError(f"unknown general fold path: {config['path']!r}")

    partition_axis = spec.fold.partition_axis
    count = _config_value(config, symbols.partition_count)
    scan_schedule = StageSchedule(
        extents=_extents(spec, sizes, {partition_axis.id: count}),
        tiles=MappingProxyType({
            **{
                axis.id: _config_value(config, symbols.scan_tile(axis))
                for axis in spec.batch
            },
            **{
                axis.id: sizes[axis.id]
                for axis in spec.axes
                if axis not in spec.batch and axis != spec.fold
            },
            partition_axis.id: 1,
        }),
        programs=MappingProxyType({partition_axis.id: 1}),
        loop=partition_axis.id,
    )
    return partial_fold_plan(
        spec,
        concrete_map_schedule(partial=True),
        scan_schedule,
    )


def _backward_schedule(spec, sizes, symbols, *, fold_tile=None, partition_count=None):
    tile_axes = set(axis.id for axis in _map_axes(spec))
    partition_axis = spec.fold.partition_axis
    return StageSchedule(
        extents=_extents(spec, sizes),
        tiles=MappingProxyType({
            axis.id: (
                (
                    symbols.backward_tile(axis)
                    if fold_tile is None else fold_tile
                ) if axis == spec.fold else symbols.backward_tile(axis)
            ) if axis.id in tile_axes else sizes[axis.id]
            for axis in spec.axes
        }),
        programs=MappingProxyType(
            {} if partition_count is None else {partition_axis.id: partition_count}
        ),
        loop=spec.fold.id,
    )


def sweep_general_backward(
        spec,
        *,
        sizes,
        forward_plan,
        hardware=l4,
        max_tile=128,
        sweep=Sweep.default,
        ):
    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    axes = _map_axes(spec)
    full_configs = _tile_configs(
        axes,
        tuple(f"cfg:{symbols.backward_tile(axis)}" for axis in axes),
        sizes,
        max_tile,
    )
    full = full_recompute_backward_stage(
        spec, _backward_schedule(spec, sizes, symbols)
    )
    full_frame = sweep.apply(full.stage, full_configs, symbols=hardware)
    frames = []
    if not full_frame.is_empty():
        frames.append(full_frame.with_columns(pl.lit("full_recompute").alias("backward_path")))

    if len(forward_plan.forward) > 1:
        partial_stage, scan_stage = forward_plan.forward
        fold_tile = partial_stage.stage.domain.get(spec.fold).tile
        partition_axis = partial_stage.partition_axis
        partition_count = scan_stage.stage.domain.get(partition_axis).extent
        batch_configs = _tile_configs(
            tuple(spec.batch),
            tuple(f"cfg:{symbols.backward_tile(axis)}" for axis in spec.batch),
            sizes,
            max_tile,
        ).with_columns(
            pl.lit(int(fold_tile), dtype=pl.Int64).alias(
                f"cfg:{symbols.backward_tile(spec.fold)}"
            )
        )
        prefix = prefix_recompute_backward_stage(
            spec,
            _backward_schedule(
                spec,
                sizes,
                symbols,
                fold_tile=int(fold_tile),
                partition_count=int(partition_count),
            ),
            checkpoints=scan_stage.carriers,
            partition_axis=partition_axis,
        )
        prefix_frame = sweep.apply(prefix.stage, batch_configs, symbols=hardware)
        if not prefix_frame.is_empty():
            frames.append(prefix_frame.with_columns(pl.lit("checkpointed").alias("backward_path")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal").sort("estimated_time")


def general_backward_stage_from_config(spec, *, sizes, forward_plan, config):
    sizes = _normalize_sizes(spec, sizes)
    symbols = _symbols(spec)
    tiles = {
        axis.id: _config_value(config, symbols.backward_tile(axis))
        if axis in _map_axes(spec) else sizes[axis.id]
        for axis in spec.axes
    }
    path = config["backward_path"]
    if path == "full_recompute":
        schedule = StageSchedule(
            extents=_extents(spec, sizes), tiles=MappingProxyType(tiles),
            programs=MappingProxyType({}), loop=spec.fold.id,
        )
        return full_recompute_backward_stage(spec, schedule)
    if path != "checkpointed" or len(forward_plan.forward) <= 1:
        raise ValueError(f"invalid general backward path: {path!r}")
    partial_stage, scan_stage = forward_plan.forward
    partition_axis = partial_stage.partition_axis
    partition_count = int(scan_stage.stage.domain.get(partition_axis).extent)
    schedule = StageSchedule(
        extents=_extents(spec, sizes), tiles=MappingProxyType(tiles),
        programs=MappingProxyType({partition_axis.id: partition_count}),
        loop=spec.fold.id,
    )
    return prefix_recompute_backward_stage(
        spec, schedule, checkpoints=scan_stage.carriers,
        partition_axis=partition_axis,
    )


__all__ = [
    "GeneralFoldSweepSymbols",
    "general_fold_plan_from_config",
    "general_backward_stage_from_config",
    "sweep_general_backward",
    "sweep_general_fold",
]
