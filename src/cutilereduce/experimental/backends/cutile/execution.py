"""Synchronized GPU workload measurement, independent of schedule selection."""

import statistics


class CuTileExecution:
    def measure(self, executable, inputs, *, backward, warmup, iterations):
        import torch
        if warmup < 0 or iterations <= 0:
            raise ValueError("invalid benchmark iteration counts")
        def run():
            outputs = executable(*inputs)
            if backward:
                outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                differentiable = tuple(x for x in inputs if x.requires_grad)
                torch.autograd.grad(outputs, differentiable,
                                    tuple(torch.ones_like(x) for x in outputs), allow_unused=True)
        return self.measure_stage(run, inputs, warmup=warmup, iterations=iterations)

    def measure_stage(self, run, inputs, *, warmup, iterations):
        import torch
        if warmup < 0 or iterations <= 0:
            raise ValueError("invalid benchmark iteration counts")
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
