from __future__ import annotations

from dataclasses import dataclass

from sympy import Min

from .axis import Axes
from .stage_domain import StageDomain
from .utilities import prod


def _axes(spec: str | Axes) -> Axes:
    if isinstance(spec, Axes):
        return spec
    return Axes.make(spec)


@dataclass(frozen=True)
class MatMulWork:
    b: Axes
    m: Axes
    n: Axes
    k: Axes

    @classmethod
    def make(
            cls,
            *,
            B: str | Axes = "",
            M: str | Axes,
            N: str | Axes,
            K: str | Axes,
            ) -> MatMulWork:
        return cls(b=_axes(B), m=_axes(M), n=_axes(N), k=_axes(K))

    @property
    def axes(self) -> Axes:
        return self.b | self.m | self.n | self.k

    def total_work(self, domain: StageDomain):
        axes = domain.resolve(self.axes)
        return 2 * prod(a.extent for a in axes)

    def tile_work(self, domain: StageDomain):
        axes = domain.resolve(self.axes)
        return 2 * prod(a.tile for a in axes)

    def span_work(self, domain: StageDomain):
        axes = domain.resolve(self.axes)
        return 2 * prod(domain.max_span(a) for a in axes)

    def tile_efficiency(self, domain: StageDomain):
        m = domain.resolve(self.m)
        n = domain.resolve(self.n)
        k = domain.resolve(self.k)
        m_prod = prod(a.tile for a in m)
        n_prod = prod(a.tile for a in n)
        k_prod = prod(a.tile for a in k)
        return (
            Min(1, m_prod * n_prod * k_prod / 16384) *
            Min(1, k_prod / 16) *
            Min(1, m_prod / 16) *
            Min(1, n_prod / 16)
        )


WorkItem = MatMulWork


@dataclass(frozen=True)
class WorkModel:
    items: tuple[WorkItem, ...] = ()

    @classmethod
    def make(cls, *items: WorkItem) -> WorkModel:
        return cls(items=tuple(items))

    def total_work(self, domain: StageDomain):
        return sum(item.total_work(domain) for item in self.items)

    def tile_work(self, domain: StageDomain):
        return sum(item.tile_work(domain) for item in self.items)

    def span_work(self, domain: StageDomain):
        return sum(item.span_work(domain) for item in self.items)

    def effective_total_work(self, domain: StageDomain):
        return sum(item.total_work(domain) / item.tile_efficiency(domain) for item in self.items)

    def effective_tile_work(self, domain: StageDomain):
        return sum(item.tile_work(domain) / item.tile_efficiency(domain) for item in self.items)
