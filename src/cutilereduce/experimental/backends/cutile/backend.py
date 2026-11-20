"""cuTile implementation of the shared backend extension contract."""

from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.backend import BackendProgram, Implementation, TiledExecution
from cutilereduce.experimental.planning.templates import build_schedule_space
from .runtime import realize_cutile_candidate, compile_cutile_realization


class CuTileBackend:
    def __init__(self, *, analytical_target=None):
        self.analytical_target = analytical_target

    def specialize(self, program, hardware, *, sizes):
        from dataclasses import asdict, replace
        from .analysis import cutile_analytical_target
        if hardware.device_count != 1:
            raise NotImplementedError("distributed GPU execution is not implemented")
        target = (cutile_analytical_target(hardware) if self.analytical_target is None else
                  replace(self.analytical_target, hardware_identity=asdict(hardware),
                          workers=hardware.multiprocessor_count))
        return BackendProgram(program, hardware, target, build_schedule_space(program),
                              tuple(sorted(sizes.items())))

    def realize(self, program, candidate):
        problem = PlanningProblem(program.program, program.analytical_target, dict(program.sizes), program.schedule_space)
        forward, backward = realize_cutile_candidate(problem, candidate)
        return Implementation(program, problem.canonical_candidate(candidate), forward, backward, execution=TiledExecution(forward, backward))

    def compile(self, implementation):
        program = implementation.backend_program
        execution = implementation.execution
        if not isinstance(execution, TiledExecution) or execution.forward is not implementation.forward or execution.backward is not implementation.backward:
            raise ValueError("cuTile implementation and analyzed execution IR disagree")
        return compile_cutile_realization(program.program, dict(program.sizes),
                                          implementation.forward, implementation.backward)

    def execution_adapter(self):
        from .execution import CuTileExecution
        return CuTileExecution()
