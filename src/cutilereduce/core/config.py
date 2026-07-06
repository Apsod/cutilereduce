from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import sympy

from .base import Phase

@dataclass(frozen=True)
class Config:
    phase: Phase
    group_dim: str
    num_groups: int
    total: dict[str, int]
    tiling: dict[str, int]
    symbols: dict[sympy.Symbol, int]
    functions: dict[sympy.Function, Any]


    def _eval(self, expr):
        for n, f in self.functions.items():
            expr = expr.replace(n, f)
        expr = expr.subs(self.symbols)
        return expr.evalf()

    def get_grouping(self, d: str):
        if d == self.group_dim:
            return Fraction(self.total[d], self.tiling[d] * self.num_groups)
        else:
            return 1
