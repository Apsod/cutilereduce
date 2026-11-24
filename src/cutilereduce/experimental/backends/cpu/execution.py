"""CPU workload and isolated-stage measurement for the shared empirical sweep."""

import statistics
import time


class CpuExecution:
    def measure_stage(self, run, inputs, *, warmup, iterations):
        if warmup < 0 or iterations <= 0:
            raise ValueError("invalid benchmark iteration counts")
        for _ in range(warmup):
            run()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            run()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    def measure(self, executable, inputs, *, backward, warmup, iterations):
        import torch
        def run():
            outputs = executable(*inputs)
            if backward:
                outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                torch.autograd.grad(outputs, tuple(x for x in inputs if x.requires_grad),
                    tuple(torch.ones_like(x) for x in outputs), allow_unused=True)
        return self.measure_stage(run, inputs, warmup=warmup, iterations=iterations)
