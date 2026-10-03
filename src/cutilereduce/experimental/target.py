"""Small target description shared by schedule binding and backend lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


CostHook = Callable[[object, object, dict[str, int]], float]


@dataclass(frozen=True)
class TargetModel:
    """Target identity and optional legality/cost hooks for schedule selection."""

    name: str
    capabilities: frozenset[str]
    cost_hook: CostHook | None = None
    default_tile_size: int = 64
    supported_schedules: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name must be non-empty")
        if self.default_tile_size <= 0:
            raise ValueError("default tile size must be positive")

    def require(self, *capabilities: str) -> None:
        missing = set(capabilities) - self.capabilities
        if missing:
            raise NotImplementedError(
                f"target {self.name!r} lacks capabilities: {sorted(missing)}"
            )

    def supports_schedule(self, name: str) -> bool:
        return not self.supported_schedules or name in self.supported_schedules

    def estimate(self, program: object, schedule: object, shape: dict[str, int]) -> float | None:
        if self.cost_hook is None:
            return None
        return float(self.cost_hook(program, schedule, shape))

    def tile_domain(self, axis: str, shape: dict[str, int]) -> range:
        if axis not in shape or type(shape[axis]) is not int or shape[axis] <= 0:
            raise ValueError(f"shape must provide a positive extent for {axis!r}")
        return range(1, shape[axis] + 1)

    def bind_tiles(
        self,
        shape: dict[str, int],
        axes: tuple[str, ...],
        overrides: dict[str, int] | None = None,
    ) -> dict[str, int]:
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(shape)
        if unknown:
            raise ValueError(f"tile overrides name unknown axes: {sorted(unknown)}")
        unsupported = set(overrides) - set(axes)
        if unsupported:
            raise ValueError(
                f"target does not tile these axes in this schedule: {sorted(unsupported)}"
            )
        result = {}
        for axis in axes:
            domain = self.tile_domain(axis, shape)
            value = overrides.get(axis, min(shape[axis], self.default_tile_size))
            if type(value) is not int or value not in domain:
                raise ValueError(f"tile size for {axis!r} is outside its legal domain")
            result[axis] = value
        return result
