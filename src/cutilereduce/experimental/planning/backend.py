"""Backend extension contracts; concrete implementations live outside planning.

Specialization exposes legal decisions. Realization fixes cost-relevant choices
before either estimation or code emission. No contract requires matching backend
module layouts or implicit hardware discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol, TYPE_CHECKING

from .analysis import BackendModel, ScheduleCandidate
from .ssa import SpecializedSchedule, TiledStage
from .bound import BoundPlan

if TYPE_CHECKING:
    from .stages import Stage, StagedPlan


@dataclass(frozen=True)
class BackendProgram:
    """A specialized program and the analytical rules available for its target."""

    program: Any
    hardware: Any
    analytical_target: BackendModel
    schedule_space: Any
    sizes: tuple[tuple[str, int], ...] = ()

    @cached_property
    def stage_namespace(self):
        from .stages import namespace
        return namespace(self)


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


class StageRunner(Protocol):
    spec: Stage

    def run(self, context: dict) -> dict: ...


class StageValidator(Protocol):
    def context(self, stage: Stage) -> dict: ...

    def validate(self, runner: StageRunner) -> None: ...


class StagedBackend(Backend, Protocol):
    """Optional executable-unit contract for independent empirical search.

    A stage reads immutable declared artifacts and creates fresh declared outputs.
    All schedule-dependent layouts, checkpoint geometry, placement and shared
    resource choices must appear in artifact contracts or explicit couplings.
    Fused/overlapping units must remain one jointly tuned stage.
    """

    def stage_plan(self, program: BackendProgram, candidate: ScheduleCandidate) -> StagedPlan: ...

    def compile_stage(self, program: BackendProgram, stage: Stage) -> StageRunner: ...

    def stage_fixture(self, program: BackendProgram, inputs: tuple, reference: Any,
                      *, backward: bool, rtol: float, atol: float) -> StageValidator: ...

    def assemble(self, program: BackendProgram, stages: tuple[StageRunner, ...]) -> Any: ...


class StageExecutionAdapter(ExecutionAdapter, Protocol):
    def measure_stage(self, run: Any, inputs: tuple, *, warmup: int, iterations: int) -> float: ...


class StageSearchBackend(StagedBackend, Protocol):
    """Optional factorized enumeration, asserting coverage of executable units.

    Without this hook the planner retains the full analytical candidate grid and
    still deduplicates its stages before empirical validation/measurement.
    """

    def stage_candidates(self, program: BackendProgram, problem: Any,
                         parameter_domains: dict | None = None) -> Any: ...
