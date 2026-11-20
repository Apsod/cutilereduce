"""cuTile tuning with synchronized timings and fresh backward accumulators."""

import statistics

from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.tuning import tune_candidates
from cutilereduce.experimental.backends.cpu.tuning import validate_callable
from .runtime import compile_cutile_candidate


def tune_cutile(operator, sizes, inputs, *, reference, analytical_target,
                backward=False, candidate_count=20, sweep=None, warmup=3,
                iterations=20, rtol=1e-3, atol=1e-4):
    import torch
    if analytical_target.backend != "cutile":
        raise ValueError("tune_cutile requires a cuTile analytical target")
    if warmup < 0 or iterations <= 0:
        raise ValueError("invalid benchmark iteration counts")
    problem = PlanningProblem(operator.lower(backward=backward), analytical_target, sizes)

    def measure(executable):
        def run():
            outputs = executable(*inputs)
            if backward:
                outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                differentiable = tuple(x for x in inputs if x.requires_grad)
                torch.autograd.grad(outputs, differentiable, tuple(torch.ones_like(x) for x in outputs), allow_unused=True)
        device = inputs[0].device
        with torch.cuda.device(device):
            for _ in range(warmup):
                run()
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1e-3)
        return statistics.median(samples)

    return tune_candidates(
        problem, compile_candidate=lambda c: compile_cutile_candidate(operator, problem, c),
        validate=lambda f: validate_callable(f, inputs, reference, backward=backward, rtol=rtol, atol=atol),
        measure=measure, sweep=sweep, backward=backward, candidate_count=candidate_count,
    )
