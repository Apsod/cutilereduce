"""Bound recursive schedule IR shared by analysis and native emission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from .templates import Iteration, MapFoldRegion, RegionCall, Reference, FoldTopology

class BoundLoopKind(Enum):
    SERIAL = "serial"
    PARALLEL = "parallel"
    VECTORIZED = "vectorized"


class BoundStorage(Enum):
    PRIVATE = "private"
    THREAD_LOCAL = "thread_local"
    MATERIALIZED = "materialized"

@dataclass(frozen=True)
class BoundIteration:
    logical: Iteration
    extent: int
    grain: int
    count: int
    loop: BoundLoopKind


class BoundExpr:
    pass


@dataclass(frozen=True)
class BoundLeaf(BoundExpr):
    expression: MapFoldRegion | RegionCall | Reference


@dataclass(frozen=True)
class BoundLoop(BoundExpr):
    iteration: BoundIteration
    body: BoundExpr
    operation: str
    topology: FoldTopology | None = None
    reorder: bool = False


@dataclass(frozen=True)
class BoundBinding:
    name: str
    expression: BoundExpr
    storage: BoundStorage


@dataclass(frozen=True)
class BoundStage:
    name: str
    bindings: tuple[BoundBinding, ...]
    tile_sizes: tuple[tuple[str, int], ...] = ()
    scratch_bytes: int = 0



@dataclass(frozen=True)
class BoundPlan:
    source: Any
    sizes: tuple[tuple[str, int], ...]
    target: Any
    forward: tuple[BoundStage, ...]
    backward: tuple[BoundStage, ...]
