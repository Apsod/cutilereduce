"""Small steady-state benchmark helper for compiled CPU fold plans."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cpu_codegen import CompiledCpuFold
from .cpu_cost import CpuPlanCost, estimate_cpu_plan
from .scheduled import CommitMode, LocalLifetime, ScheduledBackwardProgram


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class CpuBenchmarkResult:
    compile_seconds: float
    median_seconds: float
    minimum_seconds: float
    iterations: int
    plan_cost: CpuPlanCost
    reference_median_seconds: float | None = None

    @property
    def speedup(self) -> float | None:
        if self.reference_median_seconds is None:
            return None
        return self.reference_median_seconds / self.median_seconds


@dataclass(frozen=True)
class ScheduledMemoryMetrics:
    program_count: int
    active_programs: int
    loop_iterations: int
    persistent_bytes_per_program: int
    streamed_bytes_per_program: int
    maximum_local_bytes: int
    atomic_updates: int
    direct_store_bytes: int


@dataclass(frozen=True)
class CpuTrainingBenchmarkResult:
    compile_seconds: float
    forward_median_seconds: float
    backward_median_seconds: float
    full_median_seconds: float
    iterations: int
    memory: ScheduledMemoryMetrics | None = None


@dataclass(frozen=True)
class CpuRssMeasurement:
    baseline_bytes: int
    peak_bytes: int

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


def _measure(function: Callable[[], Any], iterations: int) -> tuple[float, float]:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def benchmark_cpu(
    compiled: CompiledCpuFold,
    inputs: tuple[Any, ...],
    *,
    reference: Callable[..., Any] | None = None,
    warmup: int = 3,
    iterations: int = 20,
) -> CpuBenchmarkResult:
    """Measure steady-state execution; compilation is intentionally excluded."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    for _ in range(warmup):
        compiled(*inputs)
    median, minimum = _measure(lambda: compiled(*inputs), iterations)
    reference_median = None
    if reference is not None:
        for _ in range(warmup):
            reference(*inputs)
        reference_median, _ = _measure(lambda: reference(*inputs), iterations)
    return CpuBenchmarkResult(
        compiled.compile_seconds,
        median,
        minimum,
        iterations,
        estimate_cpu_plan(compiled.problem, compiled.scheduled),
        reference_median,
    )


def _elements(axes: tuple[str, ...], sizes: dict[str, int]) -> int:
    result = 1
    for axis in axes:
        result *= sizes[axis]
    return result


def scheduled_memory_metrics(
    compiled: CompiledCpuFold,
    program: ScheduledBackwardProgram | None = None,
) -> ScheduledMemoryMetrics:
    """Compute allocation and commit counts implied by a scheduled program."""
    program = program or compiled.backward_program
    if program is None:
        raise ValueError("compiled fold has no scheduled backward program")
    sizes = dict(compiled.scheduled.sizes)
    scheduled_axes = {
        axis.axis: axis for axis in (*program.program_axes, program.loop_axis)
    }

    def tile_elements(axes: tuple[str, ...]) -> int:
        return _elements(
            axes,
            {
                axis: min(sizes[axis], scheduled_axes[axis].tile)
                if axis in scheduled_axes
                else sizes[axis]
                for axis in axes
            },
        )

    region = (
        compiled.problem.map_finalize_map_vjp_accumulate
        if compiled.problem.requires_prefix
        else compiled.problem.map_finalize_map_vjp
    )
    if region is None:
        raise ValueError("compiled fold has no backward region")
    result_nodes = dict(region.results)
    persistent = 0
    streamed = 0
    atomic_updates = 0
    direct_store_bytes = 0
    for gradient in program.gradients:
        node = region.nodes[result_nodes[gradient.result_name]]
        tile_bytes = tile_elements(gradient.axes) * _DTYPE_BYTES[node.dtype]
        if gradient.lifetime is LocalLifetime.PERSISTENT:
            persistent += tile_bytes
        else:
            streamed += tile_bytes
        output_elements = _elements(gradient.axes, sizes)
        if gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED:
            writer_multiplicity = 1
            for axis in program.program_axes:
                if axis.axis in gradient.contention_axes:
                    writer_multiplicity *= axis.count
            atomic_updates += output_elements * writer_multiplicity
        elif gradient.commit_mode is CommitMode.STORE:
            direct_store_bytes += output_elements * _DTYPE_BYTES[node.dtype]
    program_count = math.prod(axis.count for axis in program.program_axes)
    active = min(compiled.scheduled.target.worker_count, program_count)
    return ScheduledMemoryMetrics(
        program_count,
        active,
        program.loop_axis.count,
        persistent,
        streamed,
        active * (persistent + streamed),
        atomic_updates,
        direct_store_bytes,
    )


def benchmark_cpu_training(
    compiled: CompiledCpuFold,
    inputs: tuple[Any, ...],
    *,
    warmup: int = 3,
    iterations: int = 20,
) -> CpuTrainingBenchmarkResult:
    """Measure forward, prepared backward, and complete training execution."""
    import torch

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    differentiable = tuple(value for value in inputs if value.requires_grad)
    if not differentiable:
        raise ValueError("training benchmark requires differentiable inputs")

    def prepare():
        output = compiled(*inputs)
        outputs = output if isinstance(output, tuple) else (output,)
        cotangents = tuple(torch.ones_like(value) for value in outputs)
        return outputs, cotangents

    prepared_outputs, prepared_cotangents = prepare()

    def backward_only():
        torch.autograd.grad(
            prepared_outputs,
            differentiable,
            prepared_cotangents,
            retain_graph=True,
        )

    def full():
        outputs, cotangents = prepare()
        torch.autograd.grad(outputs, differentiable, cotangents)

    for _ in range(warmup):
        compiled(*inputs)
        backward_only()
        full()
    forward_median, _ = _measure(lambda: compiled(*inputs), iterations)
    backward_median, _ = _measure(backward_only, iterations)
    full_median, _ = _measure(full, iterations)
    memory = (
        scheduled_memory_metrics(compiled)
        if compiled.backward_program is not None
        else None
    )
    return CpuTrainingBenchmarkResult(
        compiled.compile_seconds,
        forward_median,
        backward_median,
        full_median,
        iterations,
        memory,
    )


def _resident_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    raise RuntimeError("CPU RSS measurement requires Linux /proc")


def measure_cpu_peak_rss(
    function: Callable[[], Any], *, poll_seconds: float = 0.001
) -> CpuRssMeasurement:
    """Sample process RSS during one call; run in an isolated process for cold RSS."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    baseline = _resident_bytes()
    peak = baseline
    running = True

    def sample() -> None:
        nonlocal peak
        while running:
            peak = max(peak, _resident_bytes())
            time.sleep(poll_seconds)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        function()
    finally:
        running = False
        thread.join()
    peak = max(peak, _resident_bytes())
    return CpuRssMeasurement(baseline, peak)


__all__ = [
    "CpuBenchmarkResult",
    "CpuRssMeasurement",
    "CpuTrainingBenchmarkResult",
    "ScheduledMemoryMetrics",
    "benchmark_cpu",
    "benchmark_cpu_training",
    "measure_cpu_peak_rss",
    "scheduled_memory_metrics",
]
