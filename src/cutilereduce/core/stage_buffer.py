from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from enum import Flag, Enum, auto

from .axis import AxisId
from .buffer import Buffer, BufferBundle, BufferId, ShapeData
from .stage_domain import ComputeAxis, ProgramAxis, StageDomain, StageAxes
from .utilities import TupleSet, forward, prod

import torch


class StageAccess(Flag):
    NONE = 0
    READ = auto()
    WRITE = auto()
    RESIDENT = auto()

    @property
    def is_read(self):
        return bool(self & READ)

    @property
    def is_write(self):
        return bool(self & WRITE)

    @property
    def is_io(self):
        return bool(self & (READ | WRITE))

    @property
    def is_resident(self):
        return bool(self & RESIDENT)

READ = StageAccess.READ
WRITE = StageAccess.WRITE
RESIDENT = StageAccess.RESIDENT
NONE = StageAccess.NONE

class BufferStorage(Enum):
    Ordinary = "ordinary"
    Materialized = "materialized"
    Intermediate = "intermediate"

    @property
    def is_ordinary(self):
        return self == BufferStorage.Ordinary

    @property
    def is_materialized(self):
        return self == BufferStorage.Materialized

    @property
    def is_intermediate(self):
        return self == BufferStorage.Intermediate

@dataclass(frozen=True)
class StageBuffer:
    buffer: Buffer
    domain: StageDomain
    storage: BufferStorage
    access: StageAccess
    axis_map: Mapping[AxisId, AxisId] = MappingProxyType({})
    storage_axis_map: Mapping[AxisId, AxisId] = MappingProxyType({})
    
    role = forward('buffer', 'id', 'role')
    id = forward('buffer', 'id')
    physical_axes = forward('buffer', 'axes')
    dtype = forward('buffer', 'dtype')
    torch_dtype = forward('buffer', 'torch_dtype')
    req_grad = forward('buffer', 'req_grad')
    default = forward('buffer', 'default')
    bytes_per_elem = forward('buffer', 'bytes_per_elem')
    padding_mode = forward('buffer', 'padding_mode')
    
    is_resident = forward('access', 'is_resident')
    is_read = forward('access', 'is_read')
    is_write = forward('access', 'is_write')

    is_ordinary = forward('storage', 'is_ordinary')
    is_materialized = forward('storage', 'is_materialized')
    is_intermediate = forward('storage', 'is_intermediate')

    @property
    def compute_axes(self) -> StageAxes:
        return StageAxes(
            values=tuple(
                self.domain.get(
                    self.axis_map.get(axis.id, axis.id)
                ) 
                for axis in self.buffer.axes
            )
        )

    @property
    def storage_axes(self) -> tuple[ComputeAxis | ProgramAxis, ...]:
        return self.domain.resolve_storage(
            self.storage_axis_map.get(axis.id, axis.id)
            for axis in self.buffer.axes
        )

    @property
    def stage_axes(self) -> StageAxes:
        return self.compute_axes

    @property
    def stage_index(self) -> tuple[int, ...]:
        return tuple(self.domain.index(a) for a in self.compute_axes)

    @property
    def storage_index(self) -> tuple[int, ...]:
        return tuple(self.domain.storage_index(a) for a in self.storage_axes)

    @property
    def inner_axes(self) -> StageAxes:
        return self.compute_axes.inner

    @property
    def contribution_axes(self) -> StageAxes:
        return self.domain.outer_axes | self.inner_axes

    def depends_on(self, axes: StageAxes) -> bool:
        return bool(self.compute_axes & axes)

    @property
    def is_streamed(self):
        return self.depends_on(self.domain.loop_axes)

    @property
    def is_persistent(self):
        return not self.is_streamed

    @property
    def absent_axes(self) -> StageAxes:
        return self.contribution_axes - self.compute_axes

    @property
    def accessed_elems(self):
        return self.total.numel * self.residual_multiplicity

    @property
    def accessed_bytes(self):
        return self.accessed_elems * self.bytes_per_elem

    @property
    def residual_multiplicity(self):
        absent = self.absent_axes.ids
        return prod(
            a.programs
            for a in self.domain.program_axes
            if a.source in absent
        )

    @property
    def logical_tiles(self):
        return prod(self.storage_num_tiles(a) for a in self.storage_axes)

    @property
    def target_partitions(self):
        storage_ids = {a.id for a in self.storage_axes}
        return prod(
            a.programs
            for a in self.domain.program_axes
            if a.id in storage_ids or a.source in storage_ids
        )

    def storage_tile_extent(self, axis: ComputeAxis | ProgramAxis):
        if isinstance(axis, ComputeAxis):
            return axis.tile
        return 1

    def storage_num_tiles(self, axis: ComputeAxis | ProgramAxis):
        if isinstance(axis, ComputeAxis):
            return axis.num_tiles
        return axis.programs

    def storage_span_extent(self, axis: ComputeAxis | ProgramAxis):
        if isinstance(axis, ComputeAxis):
            return self.domain.max_span(axis)
        return 1

    @property
    def total(self):
        return ShapeData(
            self.bytes_per_elem, 
            tuple(a.extent for a in self.storage_axes),
        )

    @property
    def tile(self):
        return ShapeData(
            self.bytes_per_elem, 
            tuple(self.storage_tile_extent(a) for a in self.storage_axes),
        )

    @property
    def span(self):
        return ShapeData(
            self.bytes_per_elem, 
            tuple(self.storage_span_extent(a) for a in self.storage_axes),
        )

    def mk_empty(self, device=None):
        return torch.empty(self.total.shape, device=device, requires_grad=self.req_grad, dtype=self.torch_dtype)

    def mk_default(self, device=None):
        return torch.full(self.total.shape, self.default, device=device, requires_grad=self.req_grad, dtype=self.torch_dtype)

    def mk_zeros(self, device=None):
        return torch.zeros(
            self.total.shape,
            device=device,
            dtype=self.torch_dtype,
        )

@dataclass(frozen=True, kw_only=True)
class KernelBuffers(TupleSet[StageBuffer]):

    @classmethod
    def make(
            cls,
            bundle: BufferBundle,
            domain: StageDomain,
            access: StageAccess,
            storage: BufferStorage,
            axis_map: Mapping[AxisId, AxisId] = MappingProxyType({}),
            storage_axis_map: Mapping[AxisId, AxisId] = MappingProxyType({}),
            ):
        return cls(
            values=tuple(
                StageBuffer(
                    buffer=b, 
                    domain=domain, 
                    axis_map=axis_map, 
                    storage_axis_map=storage_axis_map,
                    storage=storage,
                    access=access,
                ) for b in bundle)
            )

    @staticmethod
    def key(x):
        match x:
            case BufferId():
                return x
            case Buffer():
                return x.id
            case StageBuffer():
                return x.id
            case _:
                raise KeyError(f'{type(x)} not BufferId')

    @property
    def buffers(self):
        return self.values

    @property
    def read(self):
        return self.subset(lambda x: x.is_read)

    @property
    def write(self):
        return self.subset(lambda x: x.is_write)

    @property
    def resident(self):
        return self.subset(lambda x: x.is_resident)

    @property
    def ordinary(self):
        return self.subset(lambda x: x.is_ordinary)

    @property
    def materialized(self):
        return self.subset(lambda x: x.is_materialized)

    @property
    def intermediate(self):
        return self.subset(lambda x: x.is_intermediate)

    @property
    def streamed(self):
        return self.subset(lambda x: x.is_streamed)

    @property
    def persistent(self):
        return self.subset(lambda x: x.is_persistent)

    @property
    def accessed_bytes(self):
        return sum(b.accessed_bytes for b in self)

    @property
    def total_bytes(self):
        return sum(b.total.bytes for b in self)

    @property
    def tile_bytes(self):
        return sum(b.tile.bytes for b in self)

    @property
    def span_bytes(self):
        return sum(b.span.bytes for b in self)

    def mk_empty(self, device=None):
        return tuple(b.mk_empty(device=device) for b in self)

    def mk_default(self, device=None):
        return tuple(b.mk_default(device=device) for b in self)

    def mk_zeros(self, device=None):
        return tuple(b.mk_zeros(device=device) for b in self)
