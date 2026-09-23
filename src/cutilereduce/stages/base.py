from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from cutilereduce.core import symbolic as expr

from cutilereduce.core.axis import Axis, AxisId, axis_id
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import (
    BufferStorage,
    KernelBuffers,
    READ,
    RESIDENT,
    WRITE,
    StageAccess,
)
from cutilereduce.core.stage_domain import StageDomain


def resolve_axis_id(space, key: str | Axis | AxisId) -> AxisId:
    match key:
        case AxisId():
            return key
        case str():
            matches = tuple(axis.id for axis in space.axes if axis.name == key)
            match matches:
                case (id,):
                    return id
                case ():
                    raise KeyError(f"unknown axis name {key!r}")
                case _:
                    raise KeyError(f"ambiguous axis name {key!r}: {matches}")
        case _ if hasattr(key, "id"):
            return axis_id(key)
        case _:
            raise KeyError(f"unsupported axis key: {key!r}")


def normalize_axis_mapping(
    space, mapping: Mapping[str | Axis | AxisId, Any]
) -> dict[AxisId, Any]:
    return {resolve_axis_id(space, k): v for k, v in mapping.items()}


@dataclass(frozen=True)
class BufferUse:
    bundle: BufferBundle
    access: StageAccess
    storage: BufferStorage
    axis_map: Mapping[AxisId, AxisId] = MappingProxyType({})
    storage_axis_map: Mapping[AxisId, AxisId] = MappingProxyType({})

    @classmethod
    def read(
        cls,
        bundle: BufferBundle,
        storage: BufferStorage = BufferStorage.Ordinary,
        **kwargs,
    ) -> BufferUse:
        return cls(bundle=bundle, access=READ, storage=storage, **kwargs)

    @classmethod
    def write(
        cls,
        bundle: BufferBundle,
        storage: BufferStorage = BufferStorage.Ordinary,
        **kwargs,
    ) -> BufferUse:
        return cls(bundle=bundle, access=WRITE, storage=storage, **kwargs)

    @classmethod
    def read_resident(
        cls,
        bundle: BufferBundle,
        storage: BufferStorage = BufferStorage.Ordinary,
        **kwargs,
    ) -> BufferUse:
        return cls(bundle=bundle, access=READ | RESIDENT, storage=storage, **kwargs)

    @classmethod
    def resident(
        cls,
        bundle: BufferBundle,
        storage: BufferStorage = BufferStorage.Intermediate,
        **kwargs,
    ) -> BufferUse:
        return cls(bundle=bundle, access=RESIDENT, storage=storage, **kwargs)

    def bind(self, domain: StageDomain) -> KernelBuffers:
        return KernelBuffers.make(
            self.bundle,
            domain,
            self.access,
            self.storage,
            axis_map=self.axis_map,
            storage_axis_map=self.storage_axis_map,
        )


def bind_buffer_uses(domain: StageDomain, uses: tuple[BufferUse, ...]) -> KernelBuffers:
    buffers = KernelBuffers(values=())
    for use in uses:
        buffers = buffers | use.bind(domain)
    return buffers


@dataclass(frozen=True)
class StageSchedule:
    extents: Mapping[AxisId, int | expr.Expr]
    tiles: Mapping[AxisId, int | expr.Expr]
    programs: Mapping[AxisId, int | expr.Expr]
    loop: AxisId | None = None
    namespace: str = "stage"

    @classmethod
    def make(
        cls,
        space,
        *,
        extents: Mapping[str | Axis | AxisId, int | expr.Expr],
        tiles: Mapping[str | Axis | AxisId, int | expr.Expr],
        programs: Mapping[str | Axis | AxisId, int | expr.Expr] | None = None,
        loop: str | Axis | AxisId | None = None,
        namespace: str = "stage",
    ) -> StageSchedule:
        return cls(
            extents=MappingProxyType(normalize_axis_mapping(space, extents)),
            tiles=MappingProxyType(normalize_axis_mapping(space, tiles)),
            programs=MappingProxyType(normalize_axis_mapping(space, programs or {})),
            loop=None if loop is None else resolve_axis_id(space, loop),
            namespace=namespace,
        )

    def extent(self, axis: Axis) -> int | expr.Expr:
        return expr.sympify(self.extents.get(axis.id, expr.Symbol(axis.name)))

    def tile(self, axis: Axis) -> int | expr.Expr | None:
        value = self.tiles.get(axis.id)
        return None if value is None else expr.sympify(value)

    def program(self, axis: Axis, default=1) -> int | expr.Expr:
        return expr.sympify(self.programs.get(axis.id, default))


@dataclass(frozen=True)
class BuiltStage:
    stage: KernelStage
    partials: BufferBundle | None = None
    carriers: BufferBundle | None = None
    partition_axis: Axis | None = None
    checkpoints: BufferBundle | None = None
    compiler: Callable[[BuiltStage, Any], Any] | None = None

    @property
    def domain(self) -> StageDomain:
        return self.stage.domain

    def compile(self, functions=None, **options):
        if self.compiler is None:
            raise NotImplementedError(
                f"no compiler registered for stage {self.stage.name}"
            )
        return self.compiler(self, functions, **options)


__all__ = [
    "BufferUse",
    "BuiltStage",
    "StageSchedule",
    "bind_buffer_uses",
    "normalize_axis_mapping",
    "resolve_axis_id",
]
