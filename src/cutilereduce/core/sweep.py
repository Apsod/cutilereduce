from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar

import numpy as np
import polars as pl
from . import symbolic as expr
from numbers import Number
from .symbolic import Symbol, multiplicative_product_groups

from .kernel_stage import KernelStage
from .poly import Polynomial, polynomial_upper_bound
from .variables import (
    ATOMIC_ADD,
    MAX_PROGRAMS_PER_SM,
    PIPELINE_EFFICIENCY,
    READ,
    SMEM_PER_SM,
    WRITE,
)
from .measure import Measure


DEFAULT_SYMBOLS = {
    READ: 1,
    WRITE: 1,
    ATOMIC_ADD: 2,
    PIPELINE_EFFICIENCY: 1,
}

DEFAULT_FUNCTIONS = {}


@dataclass(frozen=True)
class StageEstimate:
    stage: KernelStage

    def value(self, name: str, substitutions: Mapping[object, object] | None = None):
        value = getattr(self.stage.cost, name)
        if substitutions and isinstance(value, expr.Expr):
            return value.subs(dict(substitutions))
        return value


def _as_frame(configs: pl.DataFrame | Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    if isinstance(configs, pl.DataFrame):
        return configs
    return pl.DataFrame(
        [{f"cfg:{k}": v for k, v in config.items()} for config in configs]
    )


def _replace_functions(expr, functions: Mapping[Any, Any]):
    for symbol, fn in functions.items():
        expr = expr.replace(symbol, fn)
    return expr


def _substitute_symbols(expression, symbols: Mapping[Any, Any]):
    normalized = {Symbol(k) if isinstance(k, str) else k: v for k, v in symbols.items()}
    return expression.subs(normalized) if expr.is_expr(expression) else expression


def _column_for_symbol(frame: pl.DataFrame, symbol: Any) -> str:
    name = str(symbol)
    candidates = (name, f"cfg:{name}", name.upper(), f"cfg:{name.upper()}")
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(
        f"missing config column for symbol {symbol}; expected one of {candidates}"
    )


@dataclass(frozen=True)
class _PreparedMetric:
    name: str
    expr: Any
    symbols: tuple[Any, ...]
    fn: Any | None

    @classmethod
    def make(
        cls,
        *,
        name: str,
        expression: Any,
        symbols: Mapping[Any, Any],
        functions: Mapping[Any, Any],
    ):
        if expr.is_expr(expression):
            prepared = _substitute_symbols(_replace_functions(expression, functions), symbols)
            if not expr.is_expr(prepared):
                return cls(name=name, expr=prepared, symbols=(), fn=None)
            args = tuple(symbol for symbol in sorted(prepared.free_symbols, key=str))
            return cls(
                name=name,
                expr=prepared,
                symbols=args,
                fn=expr.lambdify(args, prepared, "numpy", cse=True),
            )
        return cls(name=name, expr=expression, symbols=(), fn=None)

    def evaluate(self, frame: pl.DataFrame):
        if self.fn is None:
            return self.expr
        args = [
            frame[_column_for_symbol(frame, symbol)].to_numpy()
            for symbol in self.symbols
        ]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = self.fn(*args)
        if expr.is_expr(result):
            context = {}
            for column in frame.columns:
                name = column.removeprefix("cfg:")
                values = frame[column].to_numpy()
                context[name] = values
                context.setdefault(name.lower(), values)
            return expr.evaluate(self.expr, context)
        return result

    def evaluate_values(self, values: Mapping[str, Any]):
        args = [
            np.asarray(values[str(symbol)])
            for symbol in self.symbols
        ]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = self.fn(*args) if self.fn is not None else self.expr
        if expr.is_expr(result):
            return expr.evaluate_batch(self.expr, values)
        return result


@dataclass(frozen=True)
class StageInstantiation:
    """A stage cost model with fixed substitutions and prepared metrics."""

    stage: KernelStage
    substitutions: Mapping[Any, Any]
    functions: Mapping[Any, Any]
    metrics: tuple[_PreparedMetric, ...]

    @classmethod
    def make(
        cls,
        stage: KernelStage,
        *,
        attributes: Iterable[str],
        symbols: Mapping[Any, Any] | None = None,
        functions: Mapping[Any, Any] | None = None,
    ):
        symbols = DEFAULT_SYMBOLS | dict(symbols or {})
        functions = DEFAULT_FUNCTIONS | dict(functions or {})
        metrics = tuple(
            _PreparedMetric.make(
                name=name,
                expression=getattr(stage.cost, name),
                symbols=symbols,
                functions=functions,
            )
            for name in attributes
        )
        return cls(stage, symbols, functions, metrics)

    def evaluate(self, frame: pl.DataFrame) -> pl.DataFrame:
        frame = _add_derived_measures(self.stage, frame, self.substitutions)
        metric_frame = pl.DataFrame(
            [
                _metric_series(metric.name, metric.evaluate(frame), frame.height)
                for metric in self.metrics
            ]
        )
        return pl.concat((frame, metric_frame), how="horizontal_extend")

    def evaluate_batch(self, values: Mapping[str, Any]):
        return {
            metric.name: metric.evaluate_values(values)
            for metric in self.metrics
        }


def _iter_frame_chunks(frame: pl.DataFrame, chunk_size: int) -> Iterator[pl.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive: {chunk_size}")
    for offset in range(0, frame.height, chunk_size):
        yield frame.slice(offset, chunk_size)


def _powers_of_two(stop: int) -> tuple[int, ...]:
    values = []
    value = 1
    while value <= stop:
        values.append(value)
        value *= 2
    return tuple(values)


def _tile_rows(names, values):
    import itertools

    for row in itertools.product(*values):
        yield dict(zip(names, row, strict=True))


def _numeric(value):
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_scalar_metric(stage, name, row, symbols):
    frame = pl.DataFrame((row,))
    result = StageInstantiation.make(
        stage,
        attributes=(name,),
        symbols=symbols,
    ).evaluate(frame)
    return _numeric(result[name][0])


def bounded_tile_configs(
    stage: KernelStage,
    axes,
    names,
    sizes,
    max_tile: int | None,
    *,
    symbols: Mapping[Any, Any] | None = None,
) -> pl.DataFrame:
    """Generate discrete tile rows, pruning coupled residency violations."""
    substitutions = DEFAULT_SYMBOLS | dict(symbols or {})
    evaluator = StageInstantiation.make(
        stage,
        attributes=("residency_bytes",),
        symbols=substitutions,
    )
    residency = getattr(stage.cost, "residency_bytes", None)
    residency = residency.value if isinstance(residency, expr.Expr) else residency
    tile_names = tuple(str(name).removeprefix("cfg:") for name in names)
    polynomial_caps = {}
    capacity = substitutions.get(SMEM_PER_SM, substitutions.get(str(SMEM_PER_SM)))
    if isinstance(residency, Polynomial) and capacity is not None:
        minimums = {name: 1 for name in tile_names}
        fixed = {
            str(key): value
            for key, value in substitutions.items()
            if isinstance(value, Number)
        } | minimums
        for name in tile_names:
            polynomial_caps[name] = polynomial_upper_bound(
                residency,
                name,
                capacity,
                fixed,
            )
    values = tuple(
        _powers_of_two(
            min(
                sizes[axis.id],
                *(cap for cap in (max_tile, polynomial_caps.get(name)) if cap),
            )
            if max_tile or polynomial_caps.get(name)
            else sizes[axis.id]
        )
        for axis, name in zip(axes, tile_names, strict=True)
    )
    capacity = substitutions.get(SMEM_PER_SM, substitutions.get(str(SMEM_PER_SM)))
    if capacity is None:
        return pl.DataFrame(_tile_rows(names, values))

    rows = []

    def feasible(partial):
        # Unassigned axes receive their minimum tile.  Residency is monotone,
        # so a violation here safely prunes the entire subtree.
        row = dict(partial)
        for name, candidates in zip(names, values, strict=True):
            row.setdefault(name, candidates[0])
        try:
            context = {
                name.removeprefix("cfg:"): np.asarray([value])
                for name, value in row.items()
            }
            residency = _numeric(evaluator.evaluate_batch(context)["residency_bytes"][0])
            return residency <= capacity
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return True

    def visit(index, row):
        if index == len(names):
            rows.append(dict(row))
            return
        name = names[index]
        for value in values[index]:
            candidate = row | {name: value}
            if feasible(candidate):
                visit(index + 1, candidate)

    visit(0, {})
    return pl.DataFrame(rows)


def partition_configs_for_tiles(
    stage: KernelStage,
    tile_configs: pl.DataFrame,
    *,
    count_name: str,
    fold_tile_name: str,
    fold_extent: int,
    max_storage_ratio: float,
    symbols: Mapping[Any, Any] | None = None,
    min_count: int = 2,
    explicit_cap: int | None = None,
) -> pl.DataFrame:
    """Infer valid partition counts independently for each tile row."""
    substitutions = DEFAULT_SYMBOLS | dict(symbols or {})
    evaluator = StageInstantiation.make(
        stage,
        attributes=("partial_storage_ratio",),
        symbols=substitutions,
    )
    storage_bound = None
    try:
        def storage_value(count):
            context = {count_name.removeprefix("cfg:"): np.asarray([count])}
            return float(
                _numeric(evaluator.evaluate_batch(context)["partial_storage_ratio"][0])
            )

        base = storage_value(0)
        slope = storage_value(1) - base
        second = storage_value(2)
        if slope > 0 and math.isclose(second, base + 2 * slope, rel_tol=1e-10, abs_tol=1e-12):
            storage_bound = math.floor((max_storage_ratio - base) / slope)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        storage_bound = None
    rows = []
    for row in tile_configs.to_dicts():
        fold_tile = int(row[fold_tile_name])
        semantic_max = (fold_extent + fold_tile - 1) // fold_tile
        upper = semantic_max if explicit_cap is None else min(semantic_max, explicit_cap)
        if storage_bound is not None:
            upper = min(upper, storage_bound)
        if upper < min_count:
            continue

        if storage_bound is not None:
            rows.extend(
                row | {count_name: count}
                for count in range(min_count, upper + 1)
            )
            continue

        def valid(count):
            context = {
                name.removeprefix("cfg:"): np.asarray([value])
                for name, value in (row | {count_name: count}).items()
            }
            value = _numeric(
                evaluator.evaluate_batch(context)["partial_storage_ratio"][0]
            )
            return value <= max_storage_ratio

        try:
            if not valid(min_count):
                continue
            if valid(upper):
                bound = upper
            else:
                lo, hi = min_count, upper
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if valid(mid):
                        lo = mid
                    else:
                        hi = mid - 1
                bound = lo
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # Keep the semantic bound as a safe fallback when the symbolic
            # storage expression cannot be evaluated during generation. The
            # later sweep filter remains the final exact validity check.
            bound = upper

        rows.extend(row | {count_name: count} for count in range(min_count, bound + 1))
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={**{c: tile_configs[c].dtype for c in tile_configs.columns}, count_name: pl.Int64})


def _metric_series(name: str, value, height: int) -> pl.Series:
    if isinstance(value, np.ndarray):
        return pl.Series(name, value.astype(np.float64))
    if isinstance(value, np.generic):
        return pl.Series(name, [value.item()] * height)
    if isinstance(value, Number):
        return pl.Series(name, [float(value)] * height)
    return pl.Series(name, [value] * height)


def _add_derived_measures(stage: KernelStage, frame: pl.DataFrame, substitutions) -> pl.DataFrame:
    """Add stage-local derived measures to a sweep frame."""
    def context_for(current):
        context = {}
        for column in current.columns:
            name = column.removeprefix("cfg:")
            values = current[column].to_numpy()
            context[name] = values
            context.setdefault(name.lower(), values)
        return context

    def resolve(value, current):
        if isinstance(value, expr.Expr):
            value = value.subs(substitutions)
            if isinstance(value, expr.Expr):
                try:
                    value = expr.evaluate(value, context_for(current))
                except KeyError:
                    return None
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, np.generic):
            return np.full(current.height, value.item())
        if isinstance(value, Number):
            return np.full(current.height, value)
        return value

    columns = {}
    for axis in stage.domain.compute_axes:
        extent = resolve(axis.extent, frame)
        tile = resolve(axis.tile, frame)
        if extent is None or tile is None:
            continue
        num_tiles = np.ceil(np.asarray(extent) / np.asarray(tile)).astype(int)
        num_name = str(stage.domain.measure_symbol(stage.domain.name, axis, Measure.num_tiles))
        columns[num_name] = num_tiles

        programs = resolve(stage.domain.programs_for(axis.id), frame)
        if programs is None:
            continue
        tiles_per_program = np.ceil(
            np.asarray(num_tiles) / np.asarray(programs)
        ).astype(int)
        for measure, values in (
            (Measure.tiles_per_program, tiles_per_program),
            (Measure.span_tiles, tiles_per_program),
            (Measure.span, tiles_per_program * np.asarray(tile)),
        ):
            columns[str(stage.domain.measure_symbol(stage.domain.name, axis, measure))] = values

    if columns:
        frame = frame.with_columns(
            pl.Series(name, values) for name, values in columns.items()
        )

    cost = stage.cost
    residency = resolve(cost.residency_bytes, frame)
    smem = resolve(substitutions.get(SMEM_PER_SM, substitutions.get(str(SMEM_PER_SM))), frame)
    max_programs = resolve(
        substitutions.get(MAX_PROGRAMS_PER_SM, substitutions.get(str(MAX_PROGRAMS_PER_SM))),
        frame,
    )
    if residency is not None and smem is not None and max_programs is not None:
        occupancy = np.minimum(
            np.asarray(max_programs),
            np.asarray(smem) // np.maximum(1, np.asarray(residency)),
        )
        columns[str(cost.resident_programs_per_sm_symbol)] = occupancy
        frame = frame.with_columns(
            pl.Series(str(cost.resident_programs_per_sm_symbol), occupancy)
        )
    return frame


def _filter_root_names(filters: Iterable[pl.Expr]) -> set[str]:
    """Return metric names mentioned by Polars filter expressions."""
    result = set()
    for predicate in filters:
        try:
            result.update(predicate.meta.root_names())
        except (AttributeError, TypeError):
            # A filter-like object need not be a Polars expression.  It is
            # safer to omit it from grouping than to make assumptions about
            # its implementation.
            continue
    return result


def canonicalize_grouped_configs(
    stage: KernelStage,
    frame: pl.DataFrame,
    expressions: Iterable[Any],
    *,
    axis_order: Iterable[Any] = (),
    blocked_symbols: Iterable[str] = (),
) -> pl.DataFrame:
    """Collapse tile configurations equivalent under multiplicative groups.

    Group analysis is performed on the symbolic cost expressions.  Rows are
    collapsed only when the exact derived-measure columns also agree; this
    keeps ceil based effects (for example, a changed number of tiles) intact.
    The representative maximizes the rightmost axis in each group, matching
    the stage axis order.
    """
    if frame.is_empty():
        return frame

    groups = set()
    for expression in expressions:
        groups.update(multiplicative_product_groups(expression, proportional=False))

    # A group is useful to this frame only when every member is a configurable
    # tile symbol.  In particular, do not accidentally group hardware or
    # extent symbols that happen to occur in the same polynomial.
    available = {
        column.removeprefix("cfg:")
        for column in frame.columns
        if column.startswith("cfg:")
    }
    available -= set(blocked_symbols)
    groups = {
        frozenset(name for name in group if name in available)
        for group in groups
    }
    groups = {group for group in groups if len(group) > 1}
    if not groups:
        return frame

    order = {}
    for index, axis in enumerate(axis_order):
        name = getattr(axis, "name", axis)
        order[str(name)] = index

    def rank(name):
        if name in order:
            return order[name]
        # Sweep symbols are namespaced (full_l_tile, backward_r_tile, ...),
        # while the stage order is expressed in logical axis names.
        base = name.removesuffix("_tile").rsplit("_", 1)[-1]
        return order.get(base, len(order))
    # Prefer maximal groups and use stage order for deterministic ties.  The
    # sweep operates in a single config namespace, so overlapping groups are
    # reduced to a disjoint partition.
    groups = sorted(
        groups,
        key=lambda group: (
            -len(group),
            tuple(sorted(rank(name) for name in group)),
            tuple(sorted(group)),
        ),
    )
    selected = []
    claimed = set()
    for group in groups:
        group = frozenset(group - claimed)
        if len(group) > 1:
            selected.append(group)
            claimed.update(group)
    if not selected:
        return frame

    axis_columns = {
        name: f"cfg:{name}" for name in available
    }
    # Exact derived measures are the non-config columns present at this point.
    # They include num_tiles, tiles_per_program, span_tiles, and span.
    signature_columns = tuple(
        column for column in frame.columns if not column.startswith("cfg:")
    )
    grouped_columns = {
        axis_columns[name]
        for group in selected
        for name in group
    }
    other_config_columns = tuple(
        column
        for column in frame.columns
        if column.startswith("cfg:") and column not in grouped_columns
    )
    rows = frame.to_dicts()
    keep = {}
    for index, row in enumerate(rows):
        bucket = []
        score = []
        for group in selected:
            members = sorted(group, key=rank)
            values = tuple(row[axis_columns[name]] for name in members)
            product = 1
            for value in values:
                product *= int(value)
            exact = tuple(row[column] for column in signature_columns)
            bucket.append((product, exact))
            # The rightmost axis is preferred, then the next rightmost, etc.
            score.extend(values[::-1])
        # Preserve every independent tuning dimension, such as partition
        # count, in addition to the grouped product.
        key = (tuple(row[column] for column in other_config_columns), tuple(bucket))
        candidate = (tuple(score), -index)
        previous = keep.get(key)
        if previous is None or candidate > previous[0]:
            keep[key] = (candidate, index)

    indices = sorted(value[1] for value in keep.values())
    return frame[indices]


def chunked_evaluate_stage(
    stage: KernelStage,
    *,
    attributes: Iterable[str],
    configs: pl.DataFrame | Iterable[Mapping[str, Any]],
    symbols: Mapping[Any, Any] | None = None,
    functions: Mapping[Any, Any] | None = None,
    chunk_size: int = 1024,
) -> Iterator[pl.DataFrame]:
    frame = _as_frame(configs)
    instantiation = StageInstantiation.make(
        stage,
        attributes=attributes,
        symbols=symbols,
        functions=functions,
    )
    frame = _add_derived_measures(stage, frame, instantiation.substitutions)
    metrics = instantiation.metrics
    for chunk in _iter_frame_chunks(frame, chunk_size):
        metric_frame = pl.DataFrame(
            [
                _metric_series(metric.name, metric.evaluate(chunk), chunk.height)
                for metric in metrics
            ]
        )
        yield pl.concat((chunk, metric_frame), how="horizontal_extend")


def evaluate_stage(
    stage: KernelStage,
    *,
    attributes: Iterable[str],
    configs: pl.DataFrame | Iterable[Mapping[str, Any]],
    symbols: Mapping[Any, Any] | None = None,
    functions: Mapping[Any, Any] | None = None,
    chunk_size: int = 1024,
) -> pl.DataFrame:
    chunks = tuple(
        chunked_evaluate_stage(
            stage,
            attributes=attributes,
            configs=configs,
            symbols=symbols,
            functions=functions,
            chunk_size=chunk_size,
        )
    )
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks)


@dataclass(frozen=True)
class Sweep:
    default: ClassVar[Sweep]

    attributes: tuple[str, ...]
    filters: tuple[pl.Expr, ...]
    key: str | pl.Expr
    top_k: int = 20
    threshold: float = 10
    frontier_keys: tuple[str, ...] = ()
    grouped_variables: bool = True

    def add_attributes(self, *attributes: str):
        return replace(self, attributes=(*self.attributes, *attributes))

    def add_filters(self, *filters: pl.Expr):
        return replace(self, filters=(*self.filters, *filters))

    def with_frontier_keys(self, *keys: str):
        return replace(self, frontier_keys=tuple(keys))

    def with_grouped_variables(self, enabled: bool = True):
        return replace(self, grouped_variables=enabled)

    @property
    def key_expr(self):
        if isinstance(self.key, pl.Expr):
            return self.key
        return pl.col(self.key)

    def filter(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.filters:
            return frame
        return frame.filter(*self.filters)

    def sort_combine(
        self, a: pl.DataFrame, b: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        frame = a if b is None else pl.concat((a, b))
        if frame.is_empty():
            return frame
        if not self.frontier_keys:
            frame = frame.sort(self.key_expr)
            threshold = frame.select(self.key_expr.min()).item() * self.threshold
            return frame.filter(self.key_expr <= threshold).head(self.top_k)

        columns = frame.columns
        frame = frame.sort([*self.frontier_keys, self.key_expr])
        frontier_min = self.key_expr.min().over(self.frontier_keys)
        frame = frame.filter(self.key_expr <= frontier_min * self.threshold)
        return (
            frame.group_by(self.frontier_keys, maintain_order=True)
            .head(self.top_k)
            .select(columns)
        )

    def apply(self, stage: KernelStage, configs, **eval_kwargs) -> pl.DataFrame:
        frame = _as_frame(configs)
        symbols = DEFAULT_SYMBOLS | dict(eval_kwargs.get("symbols") or {})
        frame = _add_derived_measures(stage, frame, symbols)
        if self.grouped_variables:
            filter_roots = _filter_root_names(self.filters)
            names = set(self.attributes) | filter_roots
            expressions = []
            for name in names:
                if hasattr(stage.cost, name):
                    expression = getattr(stage.cost, name)
                    expressions.append(_substitute_symbols(expression, symbols))
            frame = canonicalize_grouped_configs(
                stage,
                frame,
                expressions,
                axis_order=(axis.name for axis in stage.domain.compute_axes),
                blocked_symbols=(
                    root.removeprefix("cfg:")
                    for root in filter_roots
                    if root.startswith("cfg:")
                ),
            )
        eval_kwargs = dict(eval_kwargs)
        eval_kwargs["symbols"] = symbols
        frontier = None
        for chunk in chunked_evaluate_stage(
            stage,
            attributes=self.attributes,
            configs=frame,
            **eval_kwargs,
        ):
            chunk = self.filter(chunk)
            frontier = self.sort_combine(chunk, frontier)
        return pl.DataFrame() if frontier is None else frontier


Sweep.default = Sweep(
    attributes=(
        "estimated_time",
        "compute_time",
        "traffic_time",
        "model_flops_per_second",
        "model_flops_utilization",
        "work_efficiency",
        "mma_efficiency",
        "partial_storage_ratio",
        "residency_bytes",
        "resident_programs_per_sm",
        "sm_utilization",
        "average_pipeline_stages",
        "pipeline_factor",
        "write_traffic",
        "streamed_traffic",
        "nonhiding_traffic",
    ),
    filters=(pl.col("resident_programs_per_sm") >= 1,),
    key="estimated_time",
)


__all__ = [
    "DEFAULT_FUNCTIONS",
    "DEFAULT_SYMBOLS",
    "StageEstimate",
    "StageInstantiation",
    "Sweep",
    "canonicalize_grouped_configs",
    "chunked_evaluate_stage",
    "evaluate_stage",
]
