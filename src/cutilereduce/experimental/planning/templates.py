"""Hardware-independent algorithmic plans for experimental fold operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cutilereduce.experimental.compiler.operator import FoldProgram


class FoldTopology(Enum):
    """An algorithmic carrier-combination topology."""

    ONLINE = "online"
    TREE = "tree"


class ArtifactKind(Enum):
    GLOBAL_EXECUTION = "global_execution"
    PARTIAL_EXECUTION = "partial_execution"
    PREFIX_EXECUTION = "prefix_execution"
    ACCUMULATOR = "accumulator"
    OUTPUT = "output"
    INPUT_COTANGENT = "input_cotangent"


@dataclass(frozen=True)
class Iteration:
    """One recursive subdivision of a logical axis."""

    axis: str
    index: str
    kind: str
    parameter: str

    def __post_init__(self) -> None:
        if self.kind not in {"chunks", "partitions"}:
            raise ValueError(f"unknown iteration kind: {self.kind!r}")
        if not self.axis or not self.index or not self.parameter:
            raise ValueError("iteration names and parameters must be non-empty")


@dataclass(frozen=True)
class ScheduleParameter:
    """A schedule parameter whose upper bound comes from the problem shape."""

    name: str
    axis: str
    kind: str
    minimum: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.axis:
            raise ValueError("schedule parameter name and axis must be non-empty")
        if self.kind not in {"chunk_size", "partition_count"}:
            raise ValueError(f"unknown schedule parameter kind: {self.kind!r}")
        if self.minimum <= 0:
            raise ValueError("schedule parameter minimum must be positive")

    def domain(self, shape: dict[str, int]) -> range:
        if self.axis not in shape or shape[self.axis] <= 0:
            raise ValueError(f"shape must provide a positive extent for {self.axis!r}")
        return range(self.minimum, shape[self.axis] + 1)


class PlanExpr:
    """Base class for recursive plan expressions."""


@dataclass(frozen=True)
class Reference(PlanExpr):
    name: str


@dataclass(frozen=True)
class RegionCall(PlanExpr):
    """Invoke one symbolic region supplied by FoldProgram."""

    region: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class MapFoldRegion(PlanExpr):
    """Invoke map_fold over the current recursively selected factor region."""

    region: str = "map_fold"


@dataclass(frozen=True)
class ParallelMap(PlanExpr):
    iteration: Iteration
    body: PlanExpr


@dataclass(frozen=True)
class CarrierFold(PlanExpr):
    iteration: Iteration
    body: PlanExpr
    topology: FoldTopology
    reorder: bool = False


@dataclass(frozen=True)
class CarrierScan(PlanExpr):
    iteration: Iteration
    body: PlanExpr
    exclusive: bool = True


@dataclass(frozen=True)
class GradientFold(PlanExpr):
    """Accumulate partial input cotangents over a recursive subdivision."""

    iteration: Iteration
    body: PlanExpr
    topology: FoldTopology


@dataclass(frozen=True)
class PlanBinding:
    name: str
    expression: PlanExpr
    persist: bool = False
    artifact: ArtifactKind | None = None

    def __post_init__(self) -> None:
        if self.persist != (self.artifact is not None):
            raise ValueError(
                "persisted bindings must declare an artifact kind and "
                "non-persisted bindings must not"
            )


@dataclass(frozen=True)
class PlanStage:
    """An abstract fusion boundary containing ordered symbolic bindings."""

    name: str
    bindings: tuple[PlanBinding, ...]

    def __post_init__(self) -> None:
        names = tuple(binding.name for binding in self.bindings)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate bindings in stage {self.name!r}")


@dataclass(frozen=True)
class ForwardScheduleTemplate:
    name: str
    stages: tuple[PlanStage, ...]
    parameters: tuple[ScheduleParameter, ...] = ()


@dataclass(frozen=True)
class BackwardRequirements:
    global_execution: bool = True
    prefix_iteration: Iteration | None = None


@dataclass(frozen=True)
class BackwardScheduleTemplate:
    name: str
    stages: tuple[PlanStage, ...]
    requirements: BackwardRequirements
    parameters: tuple[ScheduleParameter, ...] = ()


@dataclass(frozen=True)
class PersistencePlan:
    """Forward artifacts selected to satisfy a backward plan."""

    artifacts: tuple[ArtifactKind, ...]


@dataclass(frozen=True)
class TrainingScheduleTemplate:
    forward: ForwardScheduleTemplate
    backward: BackwardScheduleTemplate
    persistence: PersistencePlan


@dataclass(frozen=True)
class ScheduleSpace:
    """Shape- and target-independent schedule alternatives for a FoldProgram."""

    program: FoldProgram
    forward: tuple[ForwardScheduleTemplate, ...]
    backward: tuple[BackwardScheduleTemplate, ...]

    def __post_init__(self) -> None:
        if not self.forward:
            raise ValueError("schedule space must contain a forward schedule")
        if len({schedule.name for schedule in self.forward}) != len(self.forward):
            raise ValueError("forward schedule names must be unique")
        if len({schedule.name for schedule in self.backward}) != len(self.backward):
            raise ValueError("backward schedule names must be unique")

    def compose(
        self, forward: ForwardScheduleTemplate, backward: BackwardScheduleTemplate
    ) -> TrainingScheduleTemplate:
        if forward not in self.forward or backward not in self.backward:
            raise ValueError("schedules must belong to this schedule space")
        return _compose_training_schedule(forward, backward)

    def analyze_shape(
        self,
        schedule: ForwardScheduleTemplate | BackwardScheduleTemplate,
        shape: dict[str, int],
    ) -> dict[str, range]:
        """Return shape-bounded symbolic parameter domains, without a target."""
        if schedule not in (*self.forward, *self.backward):
            raise ValueError("schedule must belong to this schedule space")
        expected = set(self.program.spec.axes)
        if set(shape) != expected or any(
            type(extent) is not int or extent <= 0 for extent in shape.values()
        ):
            raise ValueError(
                "shape must define every logical axis with a positive extent"
            )
        return {
            parameter.name: parameter.domain(shape) for parameter in schedule.parameters
        }

    def select_forward(self, shape: dict[str, int], target) -> ForwardScheduleTemplate:
        """Select a legal template using a target cost hook when supplied."""
        candidates = tuple(
            schedule
            for schedule in self.forward
            if target.supports_schedule(schedule.name)
        )
        if not candidates:
            raise NotImplementedError(
                f"target {target.name!r} supports no forward schedule"
            )
        for schedule in candidates:
            self.analyze_shape(schedule, shape)
        if target.cost_hook is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda schedule: target.estimate(self.program, schedule, shape),
        )

    def select_backward(
        self, shape: dict[str, int], target
    ) -> BackwardScheduleTemplate:
        """Select a backward template independently of the forward topology."""
        candidates = tuple(
            schedule
            for schedule in self.backward
            if target.supports_schedule(schedule.name)
        )
        if not candidates:
            raise NotImplementedError(
                f"target {target.name!r} supports no backward schedule"
            )
        for schedule in candidates:
            self.analyze_shape(schedule, shape)
        if target.cost_hook is None:
            return candidates[0]
        return min(
            candidates,
            key=lambda schedule: target.estimate(self.program, schedule, shape),
        )


def _parameters_for(stages: tuple[PlanStage, ...]) -> tuple[ScheduleParameter, ...]:
    found: dict[str, ScheduleParameter] = {}
    for stage in stages:
        for binding in stage.bindings:
            for node in _walk(binding.expression):
                if not isinstance(
                    node, (ParallelMap, CarrierFold, CarrierScan, GradientFold)
                ):
                    continue
                kind = (
                    "partition_count"
                    if node.iteration.kind == "partitions"
                    else "chunk_size"
                )
                parameter = ScheduleParameter(
                    node.iteration.parameter, node.iteration.axis, kind
                )
                previous = found.setdefault(parameter.name, parameter)
                if previous != parameter:
                    raise ValueError(
                        f"inconsistent schedule parameter {parameter.name!r}"
                    )
    return tuple(found.values())


def _walk(expression: PlanExpr):
    yield expression
    match expression:
        case (
            ParallelMap(body=body)
            | CarrierFold(body=body)
            | CarrierScan(body=body)
            | GradientFold(body=body)
        ):
            yield from _walk(body)
        case _:
            return


def _references(expression: PlanExpr) -> tuple[str, ...]:
    return tuple(node.name for node in _walk(expression) if isinstance(node, Reference))


def _validate_expr(problem: FoldProgram, expression: PlanExpr) -> None:
    fold_axes = set(problem.spec.fold_axes)
    for node in _walk(expression):
        if isinstance(node, (ParallelMap, CarrierFold, CarrierScan, GradientFold)):
            if node.iteration.axis not in fold_axes:
                raise ValueError(
                    f"plan iteration axis {node.iteration.axis!r} is not a fold axis"
                )
        if isinstance(node, CarrierFold) and node.reorder:
            if not problem.execution_commutative:
                raise ValueError("reordered carrier fold requires commutative E")
        if isinstance(node, RegionCall):
            if not hasattr(problem, node.region):
                raise ValueError(f"unknown planning region: {node.region!r}")


def validate_forward_plan(problem: FoldProgram, plan: ForwardScheduleTemplate) -> None:
    available = set(problem.spec.inputs)
    for stage in plan.stages:
        stage_names: set[str] = set()
        for binding in stage.bindings:
            _validate_expr(problem, binding.expression)
            missing = set(_references(binding.expression)) - available - stage_names
            if missing:
                raise ValueError(
                    f"stage {stage.name!r} references unavailable values "
                    f"{sorted(missing)}"
                )
            stage_names.add(binding.name)
        available.update(binding.name for binding in stage.bindings if binding.persist)
    if not any(
        binding.artifact is ArtifactKind.GLOBAL_EXECUTION
        for stage in plan.stages
        for binding in stage.bindings
    ):
        raise ValueError("forward plan does not produce global execution state")


def validate_backward_plan(
    problem: FoldProgram, plan: BackwardScheduleTemplate
) -> None:
    requires_prefix = plan.requirements.prefix_iteration is not None
    if requires_prefix != problem.requires_prefix:
        raise ValueError("backward prefix requirement disagrees with fold algebra")
    for stage in plan.stages:
        for binding in stage.bindings:
            _validate_expr(problem, binding.expression)


def _fold_iterations(problem: FoldProgram) -> tuple[Iteration, Iteration]:
    if len(problem.spec.fold_axes) != 1:
        raise NotImplementedError("initial plan generation requires one fold axis")
    axis = problem.spec.fold_axes[0]
    return (
        Iteration(axis, "partition", "partitions", "partition_count"),
        Iteration(axis, "chunk", "chunks", f"{axis}_chunk"),
    )


def _output_stage(global_name: str = "global_execution") -> PlanStage:
    return PlanStage(
        "project_output",
        (
            PlanBinding(
                "output",
                RegionCall("project_output", (global_name,)),
                persist=True,
                artifact=ArtifactKind.OUTPUT,
            ),
        ),
    )


def _generate_forward_schedules(
    problem: FoldProgram,
) -> tuple[ForwardScheduleTemplate, ...]:
    """Construct the initial recursive, hardware-independent forward plans."""
    partition, chunk = _fold_iterations(problem)
    online = ForwardScheduleTemplate(
        "online",
        (
            PlanStage(
                "map_fold_online",
                (
                    PlanBinding(
                        "global_execution",
                        CarrierFold(chunk, MapFoldRegion(), FoldTopology.ONLINE),
                        persist=True,
                        artifact=ArtifactKind.GLOBAL_EXECUTION,
                    ),
                ),
            ),
            _output_stage(),
        ),
    )
    tree = ForwardScheduleTemplate(
        "fused_tree",
        (
            PlanStage(
                "map_fold_tree",
                (
                    PlanBinding(
                        "global_execution",
                        CarrierFold(
                            chunk,
                            MapFoldRegion(),
                            FoldTopology.TREE,
                            reorder=problem.execution_commutative,
                        ),
                        persist=True,
                        artifact=ArtifactKind.GLOBAL_EXECUTION,
                    ),
                ),
            ),
            _output_stage(),
        ),
    )
    partial_expression = ParallelMap(
        partition,
        CarrierFold(chunk, MapFoldRegion(), FoldTopology.ONLINE),
    )
    partial = ForwardScheduleTemplate(
        "partial_fold",
        (
            PlanStage(
                "partial_map_fold",
                (
                    PlanBinding(
                        "partial_execution",
                        partial_expression,
                        persist=True,
                        artifact=ArtifactKind.PARTIAL_EXECUTION,
                    ),
                ),
            ),
            PlanStage(
                "combine_partials",
                (
                    PlanBinding(
                        "global_execution",
                        CarrierFold(
                            partition,
                            Reference("partial_execution"),
                            FoldTopology.TREE,
                            reorder=problem.execution_commutative,
                        ),
                        persist=True,
                        artifact=ArtifactKind.GLOBAL_EXECUTION,
                    ),
                ),
            ),
            _output_stage(),
        ),
    )
    plans: tuple[ForwardScheduleTemplate, ...] = (online, tree, partial)
    if problem.map_finalize_map_vjp_accumulate is not None:
        checkpointed = ForwardScheduleTemplate(
            "partial_scan_checkpoints",
            (
                partial.stages[0],
                PlanStage(
                    "scan_partials",
                    (
                        PlanBinding(
                            "prefix_execution",
                            CarrierScan(partition, Reference("partial_execution")),
                            persist=True,
                            artifact=ArtifactKind.PREFIX_EXECUTION,
                        ),
                        PlanBinding(
                            "global_execution",
                            RegionCall(
                                "execution_combine",
                                ("prefix_execution", "partial_execution"),
                            ),
                            persist=True,
                            artifact=ArtifactKind.GLOBAL_EXECUTION,
                        ),
                    ),
                ),
                _output_stage(),
            ),
        )
        plans = (*plans, checkpointed)
    plans = tuple(
        ForwardScheduleTemplate(plan.name, plan.stages, _parameters_for(plan.stages))
        for plan in plans
    )
    for plan in plans:
        validate_forward_plan(problem, plan)
    return plans


def _generate_backward_schedules(
    problem: FoldProgram,
) -> tuple[BackwardScheduleTemplate, ...]:
    """Construct backward topology independently of any forward topology."""
    if problem.embed_output_vjp is None:
        return ()
    partition, chunk = _fold_iterations(problem)
    prefix = partition if problem.requires_prefix else None
    region = (
        "map_finalize_map_vjp_accumulate"
        if problem.requires_prefix
        else "map_finalize_map_vjp"
    )
    arguments = ("inputs", "accumulator") + (
        ("prefix_execution",) if prefix is not None else ()
    )
    plan = BackwardScheduleTemplate(
        "prefix" if prefix is not None else "independent",
        (
            PlanStage(
                "embed_output_vjp",
                (
                    PlanBinding(
                        "accumulator",
                        RegionCall(
                            "embed_output_vjp",
                            ("global_execution", "output_cotangent"),
                        ),
                    ),
                ),
            ),
            PlanStage(
                "map_finalize_map_vjp",
                (
                    PlanBinding(
                        "input_cotangent",
                        ParallelMap(
                            partition,
                            GradientFold(
                                chunk,
                                RegionCall(region, arguments),
                                FoldTopology.ONLINE,
                            ),
                        ),
                        persist=True,
                        artifact=ArtifactKind.INPUT_COTANGENT,
                    ),
                ),
            ),
        ),
        BackwardRequirements(prefix_iteration=prefix),
    )
    plan = BackwardScheduleTemplate(
        plan.name, plan.stages, plan.requirements, _parameters_for(plan.stages)
    )
    validate_backward_plan(problem, plan)
    return (plan,)


def build_schedule_space(problem: FoldProgram) -> ScheduleSpace:
    """Describe valid forward and backward schedule alternatives for a program."""
    return ScheduleSpace(
        program=problem,
        forward=_generate_forward_schedules(problem),
        backward=_generate_backward_schedules(problem),
    )


def _compose_training_schedule(
    forward: ForwardScheduleTemplate, backward: BackwardScheduleTemplate
) -> TrainingScheduleTemplate:
    """Match backward requirements against artifacts persisted by forward."""
    artifacts = {
        binding.artifact
        for stage in forward.stages
        for binding in stage.bindings
        if binding.artifact is not None
    }
    required = {ArtifactKind.GLOBAL_EXECUTION}
    if backward.requirements.prefix_iteration is not None:
        required.add(ArtifactKind.PREFIX_EXECUTION)
    missing = required - artifacts
    if missing:
        raise ValueError(
            "forward plan does not provide backward artifacts: "
            f"{sorted(kind.value for kind in missing)}"
        )
    persistence = PersistencePlan(tuple(sorted(required, key=lambda x: x.value)))
    return TrainingScheduleTemplate(forward, backward, persistence)


def format_plan(
    plan: ForwardScheduleTemplate | BackwardScheduleTemplate | TrainingScheduleTemplate,
) -> str:
    """Render the recursive topology without introducing hardware concepts."""
    if isinstance(plan, TrainingScheduleTemplate):
        return (
            "training {\n"
            + _indent(format_plan(plan.forward))
            + "\n"
            + _indent(format_plan(plan.backward))
            + "\n}"
        )
    lines = [
        f"{'backward' if isinstance(plan, BackwardScheduleTemplate) else 'forward'} {plan.name} {{"
    ]
    for stage in plan.stages:
        lines.append(f"  stage {stage.name} {{")
        for binding in stage.bindings:
            lines.extend(_format_binding(binding, 4))
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _indent(value: str, amount: int = 2) -> str:
    prefix = " " * amount
    return "\n".join(prefix + line for line in value.splitlines())


def _format_binding(binding: PlanBinding, indent: int) -> list[str]:
    suffix = f" persist[{binding.artifact.value}]" if binding.artifact else ""
    head = " " * indent + f"{binding.name} ={suffix}"
    return [head, *_format_expr(binding.expression, indent + 2)]


def _format_expr(expression: PlanExpr, indent: int) -> list[str]:
    prefix = " " * indent
    match expression:
        case Reference(name=name):
            return [f"{prefix}ref {name}"]
        case RegionCall(region=region, arguments=arguments):
            return [f"{prefix}call {region}({', '.join(arguments)})"]
        case MapFoldRegion():
            return [f"{prefix}call map_fold(current_region)"]
        case ParallelMap(iteration=iteration, body=body):
            line = f"{prefix}parallel {iteration.index} in {iteration.kind}({iteration.axis}, {iteration.parameter})"
        case CarrierFold(
            iteration=iteration, body=body, topology=topology, reorder=reorder
        ):
            order = ", reorder" if reorder else ""
            line = f"{prefix}{topology.value}_fold {iteration.index} in {iteration.kind}({iteration.axis}, {iteration.parameter}){order}"
        case CarrierScan(iteration=iteration, body=body, exclusive=exclusive):
            mode = "exclusive" if exclusive else "inclusive"
            line = f"{prefix}{mode}_scan {iteration.index} in {iteration.kind}({iteration.axis}, {iteration.parameter})"
        case GradientFold(iteration=iteration, body=body, topology=topology):
            line = f"{prefix}{topology.value}_gradient_fold {iteration.index} in {iteration.kind}({iteration.axis}, {iteration.parameter})"
        case _:
            raise TypeError(f"unknown plan expression: {expression!r}")
    return [line, *_format_expr(body, indent + 2)]


__all__ = [
    "ArtifactKind",
    "BackwardScheduleTemplate",
    "BackwardRequirements",
    "CarrierFold",
    "CarrierScan",
    "FoldTopology",
    "ForwardScheduleTemplate",
    "GradientFold",
    "Iteration",
    "MapFoldRegion",
    "ParallelMap",
    "PersistencePlan",
    "PlanBinding",
    "PlanExpr",
    "PlanStage",
    "ScheduleParameter",
    "ScheduleSpace",
    "build_schedule_space",
    "Reference",
    "RegionCall",
    "TrainingScheduleTemplate",
    "format_plan",
    "validate_backward_plan",
    "validate_forward_plan",
]
