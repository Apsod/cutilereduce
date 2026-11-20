"""Backend extension contracts; concrete implementations live outside planning.

Specialization exposes legal decisions. Realization fixes cost-relevant choices
before either estimation or code emission. No contract requires matching backend
module layouts or implicit hardware discovery.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from .analysis import BackendModel, ScheduleCandidate
from .ssa import SpecializedSchedule, TiledStage
from .bound import BoundPlan


@dataclass(frozen=True)
class BackendProgram:
    """A specialized program and the analytical rules available for its target."""

    program: Any
    hardware: Any
    analytical_target: BackendModel
    schedule_space: Any
    sizes: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class BoundExecution:
    forward: BoundPlan
    backward: BoundPlan | None
    gradient_stage: TiledStage | None


@dataclass(frozen=True)
class TiledExecution:
    forward: SpecializedSchedule
    backward: SpecializedSchedule | None


@dataclass(frozen=True)
class Implementation:
    """Concrete execution IR shared by analysis and compilation.

    Estimates must analyze these schedules, rather than independently choosing
    fusion, storage or launch boundaries. Backend-private metadata may describe
    physical operations without changing the shared scheduled representation.
    """

    backend_program: BackendProgram
    candidate: ScheduleCandidate
    forward: SpecializedSchedule
    backward: SpecializedSchedule | None = None
    metadata: Any = None
    execution: BoundExecution | TiledExecution | None = None


class Backend(Protocol):
    """Specialization and code emission; search and selection remain shared."""

    def specialize(self, program: Any, hardware: Any, *, sizes: dict[str, int]) -> BackendProgram: ...

    def realize(self, program: BackendProgram, candidate: ScheduleCandidate) -> Implementation: ...

    def compile(self, implementation: Implementation) -> Any: ...

    def execution_adapter(self) -> "ExecutionAdapter": ...


class ExecutionAdapter(Protocol):
    """Backend-specific synchronization and workload measurement."""

    def measure(self, executable: Any, inputs: tuple, *, backward: bool,
                warmup: int, iterations: int) -> float: ...
