"""CPU implementation of the shared backend extension contract."""

from dataclasses import dataclass

from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.backend import BackendProgram, Implementation, BoundExecution
from cutilereduce.experimental.planning.scheduled import schedule_backward_program, CommitMode
from cutilereduce.experimental.planning.ssa import SpecializedSchedule, lower_backward_stage
from cutilereduce.experimental.planning.templates import build_schedule_space
from .schedule import schedule_cpu
from .tuning import candidate_cpu_configs, cpu_analytical_target


@dataclass(frozen=True)
class CpuImplementationConfig:
    forward: object
    backward: object
    scheduled_backward: object


class CpuBackend:
    def specialize(self, program, hardware, *, sizes):
        return BackendProgram(program, hardware, cpu_analytical_target(hardware),
                              build_schedule_space(program), tuple(sorted(sizes.items())))

    def realize(self, program, candidate):
        problem = PlanningProblem(program.program, program.analytical_target, dict(program.sizes), program.schedule_space)
        candidate = problem.validate_candidate(candidate)
        plan, forward, backward = candidate_cpu_configs(problem, candidate)
        scheduled = schedule_cpu(program.program, plan, dict(program.sizes),
                                 target=program.hardware, config=forward)
        bprogram = None
        if candidate.backward and not program.program.requires_prefix:
            params = dict(candidate.parameters)
            axes = (*program.program.spec.batch_axes, *program.program.spec.fold_axes)
            tiles = {a: params.get(f"backward_tile.{a}", params[f"tile.{a}"]) for a in axes}
            bprogram = schedule_backward_program(program.program, dict(program.sizes),
                loop_axis=candidate.loop_axis or program.program.spec.fold_axes[0], tiles=tiles, loop_programs=params.get("loop_programs", 1))
        if bprogram is not None:
            for gradient in bprogram.gradients:
                if gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED and gradient.dtype != "float32":
                    raise NotImplementedError("CPU atomic gradients require float32")
        backward_bound = None
        cost_gradient = bprogram
        if candidate.backward:
            backward_bound = schedule_cpu(program.program, program.schedule_space.backward[0],
                dict(program.sizes), target=program.hardware, config=backward)
            if cost_gradient is None:
                params = dict(candidate.parameters)
                axes = (*program.program.spec.batch_axes, *program.program.spec.fold_axes)
                tiles = {a: params.get(f"backward_tile.{a}", params[f"tile.{a}"]) for a in axes}
                cost_gradient = schedule_backward_program(program.program, dict(program.sizes),
                    loop_axis=program.program.spec.fold_axes[0], tiles=tiles,
                    loop_programs=backward.partition_count or 1)
        gradient_stage = lower_backward_stage(cost_gradient) if cost_gradient is not None else None
        implementation = SpecializedSchedule(program.program, plan, program.hardware,
                                             program.sizes, candidate.parameters, scheduled)
        return Implementation(program, candidate, implementation,
                              metadata=CpuImplementationConfig(forward, backward, bprogram),
                              execution=BoundExecution(scheduled, backward_bound, gradient_stage))

    def compile(self, implementation):
        from .runtime import compile_cpu
        program = implementation.backend_program
        config = implementation.metadata
        execution = implementation.execution
        if not isinstance(execution, BoundExecution) or execution.forward is not implementation.forward.implementation:
            raise ValueError("CPU implementation and execution IR disagree")
        if config.scheduled_backward is not None and execution.gradient_stage != lower_backward_stage(config.scheduled_backward):
            raise ValueError("CPU gradient lowering and analyzed SSA disagree")
        return compile_cpu(None, dict(program.sizes), plan=implementation.forward.template,
            schedule=config.forward, backward_schedule=config.backward,
            backward=implementation.candidate.backward, backward_program=config.scheduled_backward,
            target=program.hardware, _program=program.program,
            _forward_scheduled=execution.forward, _backward_scheduled=execution.backward,
            _region_realizations=getattr(program.analytical_target.region_analyzer, "__self__", None))

    def execution_adapter(self):
        from .execution import CpuExecution
        return CpuExecution()
