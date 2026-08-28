from __future__ import annotations
from enum import Enum
from dataclasses import dataclass

from .utilities import TupleSet

class AxisType(Enum):
    Logical = "logical"
    Partitioned = "partitioned"

@dataclass(frozen=True, order=True)
class AxisId:
    tag: AxisType
    name: str

@dataclass(frozen=True)
class LogicalAxis:
    id: AxisId

    @property
    def name(self) -> str:
        return self.id.name

    @classmethod
    def make(cls, name: str) -> LogicalAxis:
        return cls(id=AxisId(name=name, tag=AxisType.Logical))

    @property
    def partition_axis(self) -> PartitionAxis:
        return PartitionAxis.make(source=self, name=self.name)

@dataclass(frozen=True)
class PartitionAxis:
    id: AxisId
    source: AxisId

    @property
    def name(self) -> str:
        return self.id.name

    @classmethod
    def make(cls, source: LogicalAxis, name: str) -> PartitionAxis:
        return cls(id=AxisId(name=name, tag=AxisType.Partitioned), source=source.id)

Axis = LogicalAxis | PartitionAxis

def axis_id(key: Axis | AxisId) -> AxisId:
    match key:
        case AxisId() as k:
            return k
        case _ if hasattr(key, "id"):
            id = key.id
            if isinstance(id, AxisId):
                return id
            raise KeyError(f"{type(key)} has non-AxisId id: {id}")
        case _:
            raise KeyError(f'type of key ({type(key)}) is does not have id')

@dataclass(frozen=True)
class Axes(TupleSet[Axis]):
    @classmethod
    def make(cls, spec: str):
        return cls(values=tuple(LogicalAxis.make(d) for d in spec.split()))

    @staticmethod
    def key(x):
        return axis_id(x)

    @property
    def ids(self) -> tuple[AxisId, ...]:
        return self.keys

