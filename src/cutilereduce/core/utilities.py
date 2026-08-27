import math
import sympy
from dataclasses import dataclass
from copy import replace
from typing import Self

def ceil_div(a, b):
  if isinstance(a, int) and isinstance(b, int):
      return (a + b - 1) // b
  return sympy.ceiling(sympy.sympify(a) / sympy.sympify(b))

def prod(xs):
    return math.prod(xs)

def forward(*chain):
    if not chain:
        raise ValueError("forward chain must be non-empty")

    def _get(self):
        curr = self
        for attr in chain:
            curr = getattr(curr, attr)
        return curr

    return property(_get)

def match_one(xs, key):
    iterator = (x for x in xs if key(x))
    
    try:
        first = next(iterator)
    except StopIteration:
        raise KeyError("No matching element found")

    try:
        next(iterator)
    except StopIteration:
        return first
    raise KeyError("Multiple matching elements found")

def index_one(xs, key):
    i, _ = match_one(enumerate(xs), lambda x: key(x[1]))
    return i

def partition_by(predicate, xs):
    satisfies = []
    fails = []
    for x in xs:
        if predicate(x):
            satisfies.append(x)
        else:
            fails.append(x)
    return tuple(satisfies), tuple(fails)

def flat_tuple(ts):
    return tuple(x for t in ts for x in t)

@dataclass(frozen=True, kw_only=True)
class TupleSet[T]:
    values: tuple[T, ...]
    
    @staticmethod
    def key(x):
        return x

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, ix):
        return self.values[ix]

    def index(self, x) -> int:
        return self.keys.index(self.key(x))

    def get(self, x) -> T:
        return self[self.index(x)]

    @property
    def key_set(self):
        return {self.key(x) for x in self.values}

    @property
    def keys(self):
        return tuple(self.key(x) for x in self.values)

    def __contains__(self, x) -> bool:
        return self.key(x) in self.key_set

    def subset(self, pred) -> Self:
        return replace(self, values=tuple(x for x in self if pred(x)))

    def map(self, fun) -> Self:
        return replace(self, values=tuple(fun(x) for x in self))

    def partition(self, pred) -> tuple[Self, Self]:
        satisfies, fails = partition_by(pred, self)
        return (
            replace(self, values=satisfies),
            replace(self, values=fails),
        )

    def __bool__(self):
        return bool(self.values)

    def __or__(self, other: Self) -> Self:
        return replace(
            self,
            values=(*self.values, *other.subset(lambda x: x not in self).values)
        )

    def __and__(self, other: Self) -> Self:
        return replace(
            self,
            values=self.subset(lambda x: x in other).values
        )

    def __sub__(self, other: Self) -> Self:
        return replace(
            self,
            values=self.subset(lambda x: x not in other).values
        )
