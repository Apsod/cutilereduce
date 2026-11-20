"""Explicit GPU hardware description, independent of discovery and cost policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuTileTarget:
    architecture: str
    multiprocessor_count: int
    device_count: int = 1
    contraction_flops_per_second: float | None = None
    scalar_ops_per_second: float | None = None
    transcendental_ops_per_second: float | None = None
    bytes_per_second: float | None = None

    def __post_init__(self):
        if not self.architecture or self.multiprocessor_count <= 0 or self.device_count <= 0:
            raise ValueError("GPU target requires an architecture and positive resource counts")
