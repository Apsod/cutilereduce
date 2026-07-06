from __future__ import annotations
from dataclasses import dataclass, fields
from enum import Enum
from typing import TypeVar, Self, Callable
from functools import reduce, wraps

T = TypeVar('T')
K = TypeVar('K')
K2 = TypeVar('K2')
V = TypeVar('V')

class Phase(Enum):
    fwd = 'forward'
    bwd = 'backward'

def field_names(x):
    return tuple((f.name for f in fields(x)))

def kmap(fun: Callable[[K], K2], d: dict[K, V]) -> dict[K2, V]:
    return {fun(k): v for k, v in d.items()}

def promote_type(func):
    @wraps(func)
    def wrapper(self, other):
        result = func(self, other)
        
        # Respect Python's fallback mechanism if NotImplemented is returned
        if result is NotImplemented:
            return NotImplemented
            
        # Determine the most specific class
        if isinstance(other, TupleSet) and issubclass(type(other), type(self)):
            target_cls = type(other)
        else:
            target_cls = type(self)
            
        # If the result isn't already the subclass, reconstruct it
        if type(result) is not target_cls:
            return target_cls(result.value)
        return result
        
    return wrapper

@dataclass(frozen=True)
class TupleSet[T]:
    value: tuple[T, ...]

    @classmethod
    def parse(cls, txt: str) -> TupleSet[str]:
        return cls.make(*txt.strip().split())

    @classmethod
    def make(cls, *values: T) -> TupleSet[T]:
        assert len(values) == len(set(values))
        return cls(tuple(values))

    @property
    def set(self) -> set[T]:
        return set(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, val: T) -> bool:
        return val in self.value

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, ix):
        return self.value[ix]

    def index(self, val: T):
        return self.value.index(val)

    def get(self, val):
        return self[self.index(val)]
    
    @promote_type
    def __or__(self, other: Self) -> Self:
        return TupleSet(self.value + tuple(x for x in other if x not in self))

    @promote_type
    def __and__(self, other: Self) -> Self:
        return TupleSet(tuple(x for x in self if x in other))

    @promote_type
    def __sub__(self, other: Self) -> Self:
        return TupleSet(tuple(x for x in self if x not in other))

    def __xor__(self, other: Self) -> Self:
        return (self - other) | (other - self)
    
    def __add__(self, other: Self) -> Self:
        return self | other

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other) -> bool:
        return sorted(self.value) == sorted(other.value)
    
    @classmethod
    def zero(cls):
        return cls.make()

    @classmethod
    def union(cls, *xs):
        return reduce(lambda a, b: a | b, xs, cls.zero())

    def is_superset(self, *sets):
        return all(not bool(s - self) for s in sets)

    def is_disjoint(self, *sets):
        return all(not bool(s & self) for s in sets)

    def tmap(self, f):
        return tuple(f(x) for x in self)

    def dmap(self, f):
        return {x: f(x) for x in self}
    
    def subset(self, keep):
        return type(self)(tuple(x for x in self if keep(x)))
