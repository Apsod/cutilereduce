from enum import Enum
from dataclasses import dataclass


class Measure(Enum):
    """Semantic measures exposed by a stage realization."""

    extent = "extent"
    tile = "tile"
    programs = "programs"
    span = "span"
    num_tiles = "num_tiles"
    tiles_per_program = "tiles_per_program"
    span_tiles = "span_tiles"


@dataclass(frozen=True)
class MeasureRef:
    """Typed handle for a stage measure.

    The domain owns the expression behind a reference.  Callers can retain
    this handle for dependency analysis without inventing symbol names.
    """

    axis: object
    measure: Measure

    def __str__(self) -> str:
        return f"{self.axis}:{self.measure.value}"
