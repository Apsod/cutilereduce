"""CPU workload measurement adapter for the shared empirical sweep."""


class CpuExecution:
    def measure(self, executable, inputs, *, backward, warmup, iterations):
        from .benchmark import benchmark_cpu, benchmark_cpu_training
        if backward:
            return benchmark_cpu_training(executable, inputs, warmup=warmup,
                                          iterations=iterations).full_median_seconds
        return benchmark_cpu(executable, inputs, warmup=warmup,
                             iterations=iterations).median_seconds
