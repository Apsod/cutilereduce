from __future__ import annotations

from dataclasses import dataclass

from .symbolic import Min

from .axis import Axes
from .stage_domain import StageDomain
from .utilities import prod
from .measure import Measure


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

    def work(self, domain: StageDomain, measure: Measure):
        """Return work over one semantic axis measure."""
        if measure is Measure.programs:
            raise ValueError("program count is not a work measure")
        axes = domain.resolve(self.axes)
        return 2 * prod(domain.measure(axis.id, measure) for axis in axes)

    def total_work(self, domain: StageDomain):
        return self.work(domain, Measure.extent)

    def tile_work(self, domain: StageDomain):
        return self.work(domain, Measure.tile)

    def span_work(self, domain: StageDomain):
        return self.work(domain, Measure.span)

    def tile_efficiency(self, domain: StageDomain):
        m = domain.resolve(self.m)
        n = domain.resolve(self.n)
        k = domain.resolve(self.k)
        m_prod = prod(domain.measure(a.id, Measure.tile) for a in m)
        n_prod = prod(domain.measure(a.id, Measure.tile) for a in n)
        k_prod = prod(domain.measure(a.id, Measure.tile) for a in k)
        return (
            Min(1, m_prod * n_prod * k_prod / 16384)
            * Min(1, k_prod / 16)
            * Min(1, m_prod / 16)
            * Min(1, n_prod / 16)
        )


WorkItem = MatMulWork


@dataclass(frozen=True)
class WorkModel:
    items: tuple[WorkItem, ...] = ()

    @classmethod
    def make(cls, *items: WorkItem) -> WorkModel:
        return cls(items=tuple(items))

    def work(self, domain: StageDomain, measure: Measure):
        if measure is Measure.programs:
            raise ValueError("program count is not a work measure")
        return sum(item.work(domain, measure) for item in self.items)

    def effective_work(self, domain: StageDomain, measure: Measure):
        if measure is Measure.programs:
            raise ValueError("program count is not a work measure")
        return sum(
            item.work(domain, measure) / item.tile_efficiency(domain)
            for item in self.items
        )

    # Compatibility names for callers that still use the old metric-specific API.
    def total_work(self, domain: StageDomain):
        return self.work(domain, Measure.extent)

    def tile_work(self, domain: StageDomain):
        return self.work(domain, Measure.tile)

    def span_work(self, domain: StageDomain):
        return self.work(domain, Measure.span)

    def effective_tile_work(self, domain: StageDomain):
        return self.effective_work(domain, Measure.tile)

    def effective_total_work(self, domain: StageDomain):
        return self.effective_work(domain, Measure.extent)


def matmul(
    *, M: str | Axes, N: str | Axes, K: str | Axes, B: str | Axes = ""
) -> MatMulWork:
    return MatMulWork.make(B=B, M=M, N=N, K=K)


def workmodel(*items: WorkItem) -> WorkModel:
    return WorkModel.make(*items)
