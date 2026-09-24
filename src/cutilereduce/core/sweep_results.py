"""Typed containers for analytical sweep results.

The sweep implementation still uses Polars for columnar evaluation, but
different plan families are kept in separately typed frames.  Flattening is
an explicit presentation operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import json
from typing import Any

import polars as pl


@dataclass(frozen=True)
class SweepSchema:
    """Small runtime schema used to validate a sweep frame."""

    name: str
    required: tuple[str, ...] = ()

    def validate(self, frame: pl.DataFrame) -> None:
        missing = tuple(name for name in self.required if name not in frame.columns)
        if missing:
            raise ValueError(f"{self.name} frame is missing columns: {missing}")


@dataclass(frozen=True)
class SweepFrame:
    """A Polars frame with an explicit sweep-family schema."""

    frame: pl.DataFrame
    schema: SweepSchema = SweepSchema("sweep")

    def __post_init__(self):
        self.schema.validate(self.frame)

    @classmethod
    def empty(cls, schema: SweepSchema | None = None) -> "SweepFrame":
        return cls(pl.DataFrame(), schema or SweepSchema("sweep"))

    def to_polars(self) -> pl.DataFrame:
        return self.frame

    def to_json(self, path=None) -> str:
        """Serialize this frame as a versioned, JSON-native result document."""
        document = {
            "schema": "cutilereduce.sweep",
            "version": 1,
            "frame": self.schema.name,
            "columns": list(self.frame.columns),
            "records": self.frame.to_dicts(),
        }
        text = json.dumps(document, allow_nan=False, separators=(",", ":"))
        if path is not None:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
        return text

    @classmethod
    def from_json(cls, source, schema: SweepSchema | None = None) -> "SweepFrame":
        if hasattr(source, "read"):
            document = json.load(source)
        else:
            try:
                with open(source, encoding="utf-8") as stream:
                    document = json.load(stream)
            except (TypeError, OSError):
                document = json.loads(source)
        if document.get("schema") != "cutilereduce.sweep":
            raise ValueError("unsupported sweep JSON schema")
        if document.get("version") != 1:
            raise ValueError("unsupported sweep JSON version")
        columns = document.get("columns")
        records = document.get("records")
        if not isinstance(columns, list) or not isinstance(records, list):
            raise ValueError("invalid sweep JSON document")
        frame = pl.DataFrame(records, schema=columns, strict=False)
        return cls(frame, schema or SweepSchema(document.get("frame", "sweep")))

    def best(self, key: str = "estimated_time") -> dict[str, Any] | None:
        if self.frame.is_empty():
            return None
        return self.frame.sort(key).head(1).to_dicts()[0]

    def cross_join(self, other: "SweepFrame") -> "SweepFrame":
        return SweepFrame(
            self.frame.join(other.frame, how="cross"),
            SweepSchema(f"{self.schema.name}×{other.schema.name}"),
        )

    def join(
        self,
        other: "SweepFrame",
        *,
        on: str | list[str] | tuple[str, ...],
        suffix: str = "_right",
    ) -> "SweepFrame":
        keys = (on,) if isinstance(on, str) else tuple(on)
        missing = tuple(
            key for key in keys
            if key not in self.frame.columns or key not in other.frame.columns
        )
        if missing:
            raise ValueError(f"join keys missing from frame: {missing}")
        return SweepFrame(
            self.frame.join(other.frame, on=list(keys), suffix=suffix),
            SweepSchema(f"{self.schema.name}+{other.schema.name}", keys),
        )

    def concat(self, other: "SweepFrame", *, diagonal: bool = False) -> "SweepFrame":
        if not diagonal and self.frame.columns != other.frame.columns:
            raise ValueError("vertical concatenation requires identical schemas")
        how = "diagonal" if diagonal else "vertical"
        return SweepFrame(
            pl.concat((self.frame, other.frame), how=how),
            SweepSchema(f"{self.schema.name}+{other.schema.name}"),
        )

    def __getitem__(self, item):
        return self.frame[item]

    def __getattr__(self, name: str) -> Any:
        # Preserve convenient read-only Polars inspection (iter_rows, is_empty,
        # height, columns, ...), while mutations stay behind this wrapper.
        return getattr(self.frame, name)


@dataclass(frozen=True)
class ForwardSweepResult:
    """Full and partial forward candidates with separate schemas."""

    full: SweepFrame
    partial: SweepFrame

    @classmethod
    def empty(cls) -> "ForwardSweepResult":
        return cls(SweepFrame.empty(SweepSchema("forward.full")),
                   SweepFrame.empty(SweepSchema("forward.partial")))

    def is_empty(self) -> bool:
        return self.full.frame.is_empty() and self.partial.frame.is_empty()

    def all(self) -> SweepFrame:
        frames = [frame for frame in (self.full, self.partial)
                  if not frame.frame.is_empty()]
        if not frames:
            return SweepFrame.empty(SweepSchema("forward.all"))
        result = frames[0]
        for frame in frames[1:]:
            result = result.concat(frame, diagonal=True)
        if "forward_estimated_time" in result.frame.columns:
            result = SweepFrame(
                result.frame.sort("forward_estimated_time"), result.schema
            )
        return result

    def to_polars(self) -> pl.DataFrame:
        return self.all().frame

    def best(self, key: str = "forward_estimated_time") -> dict[str, Any] | None:
        return self.all().best(key)

    def __getitem__(self, item):
        return self.all()[item]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.all().frame, name)


@dataclass(frozen=True)
class BackwardSweepResult:
    """Backward candidates kept separate for each loop-axis choice."""

    by_loop_axis: Mapping[str, SweepFrame]

    @classmethod
    def empty(cls) -> "BackwardSweepResult":
        return cls({})

    def is_empty(self) -> bool:
        return not self.by_loop_axis or all(
            frame.frame.is_empty() for frame in self.by_loop_axis.values()
        )

    def all(self) -> SweepFrame:
        frames = [frame for frame in self.by_loop_axis.values()
                  if not frame.frame.is_empty()]
        if not frames:
            return SweepFrame.empty(SweepSchema("backward.all"))
        result = frames[0]
        for frame in frames[1:]:
            result = result.concat(frame, diagonal=True)
        if "estimated_time" in result.frame.columns:
            result = SweepFrame(result.frame.sort("estimated_time"), result.schema)
        return result

    def to_polars(self) -> pl.DataFrame:
        return self.all().frame

    def best(self, key: str = "estimated_time") -> dict[str, Any] | None:
        return self.all().best(key)

    def __getitem__(self, item):
        return self.all()[item]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.all().frame, name)


__all__ = ["SweepSchema", "SweepFrame", "ForwardSweepResult", "BackwardSweepResult"]
