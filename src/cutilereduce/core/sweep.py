from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
import sympy
from numbers import Number
from sympy import Function, Max, Min, Symbol, log

from .kernel_stage import KernelStage
from .variables import ATOMIC_ADD, BWD_CONTENTION, READ, WRITE


def contention_penalty(alpha, cap, contention):
    return Min(cap, alpha * log(Max(1, contention), 2))


def atomic_add_penalty(contention):
    return contention_penalty(alpha=4, cap=64, contention=contention)


DEFAULT_SYMBOLS = {
    READ: 1,
    WRITE: 1,
    ATOMIC_ADD: 2,
}

DEFAULT_FUNCTIONS = {
    BWD_CONTENTION: atomic_add_penalty,
}


@dataclass(frozen=True)
class StageEstimate:
    stage: KernelStage

    def value(self, name: str, substitutions: Mapping[object, object] | None = None):
        value = getattr(self.stage.cost, name)
        if substitutions and isinstance(value, sympy.Expr):
            return value.subs(dict(substitutions))
        return value


def _as_frame(configs: pl.DataFrame | Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    if isinstance(configs, pl.DataFrame):
        return configs
    return pl.DataFrame([{f"cfg:{k}": v for k, v in config.items()} for config in configs])


def _replace_functions(expr, functions: Mapping[Function, Any]):
    for symbol, fn in functions.items():
        expr = expr.replace(symbol, fn)
    return expr


def _substitute_symbols(expr, symbols: Mapping[Symbol | str, Any]):
    normalized = {
        Symbol(k) if isinstance(k, str) else k: v
        for k, v in symbols.items()
    }
    return expr.subs(normalized)


def _column_for_symbol(frame: pl.DataFrame, symbol: Symbol) -> str:
    candidates = (str(symbol), f"cfg:{symbol}")
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"missing config column for symbol {symbol}; expected one of {candidates}")


@dataclass(frozen=True)
class _PreparedMetric:
    name: str
    expr: Any
    symbols: tuple[Symbol, ...]
    fn: Any | None

    @classmethod
    def make(
            cls,
            *,
            name: str,
            expr: Any,
            symbols: Mapping[Symbol | str, Any],
            functions: Mapping[Function, Any],
            ):
        if isinstance(expr, sympy.Expr):
            prepared = _substitute_symbols(_replace_functions(expr, functions), symbols)
            args = tuple(sorted(prepared.free_symbols, key=str))
            return cls(
                name=name,
                expr=prepared,
                symbols=args,
                fn=sympy.lambdify(args, prepared, "numpy", cse=True),
            )
        return cls(name=name, expr=expr, symbols=(), fn=None)

    def evaluate(self, frame: pl.DataFrame):
        if self.fn is None:
            return self.expr
        args = [
            frame[_column_for_symbol(frame, symbol)].to_numpy()
            for symbol in self.symbols
        ]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self.fn(*args)


def _iter_frame_chunks(frame: pl.DataFrame, chunk_size: int) -> Iterator[pl.DataFrame]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive: {chunk_size}")
    for offset in range(0, frame.height, chunk_size):
        yield frame.slice(offset, chunk_size)


def _metric_series(name: str, value, height: int) -> pl.Series:
    if isinstance(value, np.ndarray):
        return pl.Series(name, value.astype(np.float64))
    if isinstance(value, np.generic):
        return pl.Series(name, [value.item()] * height)
    if isinstance(value, Number):
        return pl.Series(name, [float(value)] * height)
    return pl.Series(name, [value] * height)


def chunked_evaluate_stage(
        stage: KernelStage,
        *,
        attributes: Iterable[str],
        configs: pl.DataFrame | Iterable[Mapping[str, Any]],
        symbols: Mapping[Symbol | str, Any] | None = None,
        functions: Mapping[Function, Any] | None = None,
        chunk_size: int = 1024,
        ) -> Iterator[pl.DataFrame]:
    frame = _as_frame(configs)
    symbols = DEFAULT_SYMBOLS | dict(symbols or {})
    functions = DEFAULT_FUNCTIONS | dict(functions or {})
    metrics = tuple(
        _PreparedMetric.make(
            name=name,
            expr=getattr(stage.cost, name),
            symbols=symbols,
            functions=functions,
        )
        for name in attributes
    )
    for chunk in _iter_frame_chunks(frame, chunk_size):
        metric_frame = pl.DataFrame([
            _metric_series(metric.name, metric.evaluate(chunk), chunk.height)
            for metric in metrics
        ])
        yield pl.concat((chunk, metric_frame), how="horizontal")


def evaluate_stage(
        stage: KernelStage,
        *,
        attributes: Iterable[str],
        configs: pl.DataFrame | Iterable[Mapping[str, Any]],
        symbols: Mapping[Symbol | str, Any] | None = None,
        functions: Mapping[Function, Any] | None = None,
        chunk_size: int = 1024,
        ) -> pl.DataFrame:
    chunks = tuple(chunked_evaluate_stage(
        stage,
        attributes=attributes,
        configs=configs,
        symbols=symbols,
        functions=functions,
        chunk_size=chunk_size,
    ))
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks)


@dataclass(frozen=True)
class Sweep:
    attributes: tuple[str, ...]
    filters: tuple[pl.Expr, ...]
    key: str | pl.Expr
    top_k: int = 20
    threshold: float = 10

    def add_attributes(self, *attributes: str):
        return replace(self, attributes=(*self.attributes, *attributes))

    def add_filters(self, *filters: pl.Expr):
        return replace(self, filters=(*self.filters, *filters))

    @property
    def key_expr(self):
        if isinstance(self.key, pl.Expr):
            return self.key
        return pl.col(self.key)

    def filter(self, frame: pl.DataFrame) -> pl.DataFrame:
        if not self.filters:
            return frame
        return frame.filter(*self.filters)

    def sort_combine(self, a: pl.DataFrame, b: pl.DataFrame | None = None) -> pl.DataFrame:
        frame = a if b is None else pl.concat((a, b))
        if frame.is_empty():
            return frame
        frame = frame.sort(self.key_expr)
        threshold = frame.select(self.key_expr.min()).item() * self.threshold
        return frame.filter(self.key_expr <= threshold).head(self.top_k)

    def apply(self, stage: KernelStage, configs, **eval_kwargs) -> pl.DataFrame:
        frontier = None
        for chunk in chunked_evaluate_stage(
            stage,
            attributes=self.attributes,
            configs=configs,
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
        "work_efficiency",
        "mma_efficiency",
        "partial_storage_ratio",
        "residency_bytes",
        "resident_programs_per_sm",
        "sm_utilization",
        "write_traffic",
        "streamed_traffic",
        "nonhiding_traffic",
    ),
    filters=(
        pl.col("resident_programs_per_sm") >= 1,
        pl.col("work_efficiency") >= 0.5,
        pl.col("partial_storage_ratio") <= 1,
    ),
    key="estimated_time",
)


__all__ = [
    "DEFAULT_FUNCTIONS",
    "DEFAULT_SYMBOLS",
    "StageEstimate",
    "Sweep",
    "atomic_add_penalty",
    "chunked_evaluate_stage",
    "contention_penalty",
    "evaluate_stage",
]
