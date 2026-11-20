"""Concrete region decisions, shared by CPU emission and cost analysis."""

from dataclasses import dataclass

from cutilereduce.experimental.ir.core import TensorRegion
from .fusion import RegionFusion


@dataclass(frozen=True)
class CpuRegionImplementation:
    """Shape-independent fusion, output and buffer-reuse decisions.

    Binding extents changes loop bounds, not the selected execution structure.
    Consumers must not rerun fusion after this realization has been selected.
    """

    region: TensorRegion
    fusion: RegionFusion

    def estimate(self, extents=None):
        from .analysis import analyze_cpu_region
        return analyze_cpu_region(self.region, extents, implementation=self)

    def emit(self, name, sizes, *, nested_names=None):
        from .regions import _emit_region
        return _emit_region(name, self.region, sizes, nested_names=nested_names,
                            implementation=self)


def realize_cpu_region(region):
    return CpuRegionImplementation(region, RegionFusion(region))


class CpuRegionRealizations:
    """Compilation-owned region realizations, reused across shapes and consumers."""

    def __init__(self):
        self._regions = {}

    def realize(self, region):
        key = id(region)
        if key not in self._regions:
            self._regions[key] = realize_cpu_region(region)
        return self._regions[key]

    def analyze(self, region, extents=None):
        return self.realize(region).estimate(extents)
