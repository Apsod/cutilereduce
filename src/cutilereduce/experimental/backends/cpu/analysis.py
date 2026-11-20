"""Cost the CPU realization, using the emitter's fusion/storage decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from cutilereduce.core import symbolic as sym
from cutilereduce.experimental.ir.core import TensorNode, TensorRegion
from cutilereduce.experimental.ir.cost import (
    KernelCost, OperationCost, RegionCost, RegionRealizationCost,
    analyze_region, elements,
)
from cutilereduce.experimental.ir.facts import analyze_region_facts
from .fusion import RegionFusion


def _node_work(node, nodes, sizes):
    # Analyze an isolated operation without traversing its original producers.
    leaves = tuple(TensorNode(i, "argument", (), nodes[n].axes, nodes[n].dtype)
                   for i, n in enumerate(node.inputs))
    operation = replace(node, id=len(leaves), inputs=tuple(range(len(leaves))))
    region = TensorRegion(tuple(sizes), (*leaves, operation),
                          tuple((str(i), i) for i in range(len(leaves))), (("result", operation.id),))
    return analyze_region(region, sizes).work


def analyze_cpu_region(region: TensorRegion, extents=None) -> RegionCost:
    """Symbolic CPU kernel work, traffic, allocations and retained storage.

    Native groups count inlined operations at their actual loop domain, including
    duplicated cheap producers. Fused row sums/dots are counted once. ATen fallback
    operations retain dense estimates; internal library packing is not inferred.
    """
    sizes = {a: sym.Symbol(a) for a in region.axes}
    sizes.update(extents or {})
    facts = analyze_region_facts(region)
    logical = facts.cost.subs(sizes)
    contraction_facts = {c.node_id: c for c in facts.contractions}
    fusion = RegionFusion(region)
    nodes = region.nodes
    storage = {i: i for _, i in region.arguments}
    retained = {}
    kernels = []
    peaks = [0]

    def size(i):
        return facts.values[i].nbytes(sizes)

    for step in fusion.steps:
        node = nodes[step.root]
        group = fusion.groups.get(step.root)
        work = OperationCost()
        count = elements(group.axes if group else node.axes, sizes)
        if group:
            for i in step.inside:
                inline = nodes[i]
                if inline.op in {"constant", "full", "zero", "domain_valid", "stop_gradient"}:
                    continue
                work += _node_work(replace(inline, axes=group.axes), nodes, sizes)
            if group.dot:
                work += OperationCost(arithmetic=count)
            if group.reduction:
                work += OperationCost(reduction_ops=elements(node.axes, sizes) *
                                      sym.SaturateSub(elements(group.reduce_axes, sizes), 1))
            if len(step.outputs) > 1:
                reduction = nodes[step.outputs[1]]
                work += OperationCost(reduction_ops=elements(reduction.axes, sizes) *
                                      sym.SaturateSub(elements(reduction.attribute("over"), sizes), 1))
        else:
            work = _node_work(node, nodes, sizes)
        reads = sum(size(i) for i in dict.fromkeys(step.inputs))
        writes = sum(size(i) for i in step.outputs) if step.kind != "alias" else 0
        allocated = 0
        touched = {storage.get(i, i): size(i) for i in step.inputs}
        for i in step.outputs:
            if step.kind == "alias":
                storage[i] = storage.get(step.inputs[0], step.inputs[0])
            elif i == step.root and step.reused_from is not None:
                storage[i] = storage[step.reused_from]
            else:
                storage[i] = i
                allocated += size(i)
                retained[i] = size(i)
            touched[storage[i]] = size(i)
        shape = ()
        if step.kind == "gemm":
            dimensions = contraction_facts[node.id].dimensions(sizes)
            shape = tuple(dimensions[role] for role in ("B", "M", "N", "K"))
        kernels.append(KernelCost(step.kind, work, reads, writes, allocated,
                                  sum(touched.values()), count, shape, step.reused_from, step.outputs))
        peaks.append(sum(retained.values()))
    realization = RegionRealizationCost(tuple(kernels), sym.Max(*peaks))
    return RegionCost(sum((k.work for k in kernels), OperationCost()),
                      logical.argument_bytes, logical.result_bytes,
                      realization.peak_owned_bytes,
                      "CPU fusion groups and last-use reuse; retained Tensor locals; dense ATen fallback estimates",
                      realization)


@dataclass(frozen=True)
class CpuCostModel:
    """Transparent throughput/latency priors, not fitted attention tile scores.

    Rates are per core except shared LLC/DRAM and atomics. Cache penalties are
    soft slowdown priors for throughput calibrated in cache. Unknown capacities
    disable their penalties. These assumptions are serialized with each sweep.
    """

    workers: int = 1
    vector_lanes: int = 8
    l2_bytes: int | None = None
    l3_bytes: int | None = None
    gemm_flops_per_second: float = 64e9
    scalar_ops_per_second: float = 32e9
    transcendental_ops_per_second: float = 5e8
    l2_bytes_per_second: float = 100e9
    l3_bytes_per_second: float = 40e9
    memory_bytes_per_second: float = 20e9
    native_call_seconds: float = 2.5e-7
    aten_call_seconds: float = 2e-6
    gemm_call_seconds: float = 3e-6
    allocation_seconds: float = 5e-7
    region_call_seconds: float = 2.5e-7
    atomic_updates_per_second: float = 50e6
    l2_slowdown: float = 0.5
    l3_slowdown: float = 0.75

    def __post_init__(self):
        import math
        if self.workers <= 0 or self.vector_lanes <= 0:
            raise ValueError("workers and vector_lanes must be positive")
        for name, value in asdict(self).items():
            if name.endswith("bytes"):
                if value is not None and value <= 0:
                    raise ValueError(f"{name} must be positive")
            elif name not in {"workers", "vector_lanes"} and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
            if name.endswith("per_second") and value <= 0:
                raise ValueError(f"{name} must be positive")

    def estimate_stage(self, stage):
        compute = memory = dispatch = penalty = calls = allocated = reads = writes = 0
        execution = 0
        for invocation in stage.invocations:
            realization = invocation.cost.realization
            if realization is None:
                raise ValueError("CPU stage costing requires realized region costs")
            external = sym.Min(self.workers, sym.Max(1, invocation.parallel_tasks))
            dispatch += invocation.count * self.region_call_seconds / external
            for kernel in realization.kernels:
                if kernel.kind == "alias":
                    continue
                count = invocation.count
                calls += count
                reads += count * kernel.read_bytes
                writes += count * kernel.write_bytes
                allocated += count * kernel.allocated_bytes
                work = kernel.work
                scalar = work.arithmetic + work.comparisons + work.reduction_ops + work.scan_ops + work.conversions
                efficiency = 1
                if kernel.gemm_shape:
                    _, m, n, k = kernel.gemm_shape
                    efficiency = (sym.sympify(m) / (m + self.vector_lanes)
                                  * sym.sympify(n) / (n + self.vector_lanes)
                                  * sym.sympify(k) / (k + 2 * self.vector_lanes))
                    intra = sym.Min(self.workers, sym.Max(1, sym.sympify(work.contraction_flops) / 1e6))
                else:
                    intra = sym.Min(self.workers, sym.Max(1, sym.sympify(kernel.elements) / 32768))
                # More than one outer program installs a parallel guard, which
                # serializes inner ATen/native loops. Singleton owners do not.
                workers = sym.Piecewise((intra, invocation.parallel_tasks <= 1), (external, True))
                base = count * (work.contraction_flops / (self.gemm_flops_per_second * efficiency)
                                + scalar / self.scalar_ops_per_second
                                + work.transcendentals / self.transcendental_ops_per_second) / workers
                ws = sym.Max(1, kernel.working_set_bytes)
                fraction3 = sym.Min(1, self.l3_bytes / ws) if self.l3_bytes else 1
                fraction2 = sym.Min(fraction3, self.l2_bytes / ws) if self.l2_bytes else fraction3
                spill2 = sym.SaturateSub(1, fraction2)
                spill3 = sym.SaturateSub(1, fraction3)
                bytes_moved = count * (kernel.read_bytes + kernel.write_bytes)
                transfer = bytes_moved * (
                    fraction2 / (self.l2_bytes_per_second * workers)
                    + sym.SaturateSub(fraction3, fraction2) / self.l3_bytes_per_second
                    + spill3 / self.memory_bytes_per_second)
                stalls = base * (self.l2_slowdown * spill2 + self.l3_slowdown * spill3)
                compute += base
                memory += transfer
                penalty += stalls
                execution += sym.Max(base + stalls, transfer)
                allocation_count = (len(kernel.outputs) - int(kernel.reused_from is not None)
                                    if kernel.kind != "alias" else 0)
                latency = (self.gemm_call_seconds if kernel.kind == "gemm" else
                           self.native_call_seconds if kernel.kind in {"pointwise", "reduction"} else self.aten_call_seconds)
                dispatch += count * (latency + allocation_count * self.allocation_seconds) / external
        # Region arguments/commits outside the kernels and local accumulation
        # are conservatively charged as cache traffic; atomics have a distinct
        # latency prior rather than pretending to be ordinary writes.
        commits = (stage.write_bytes / (self.l2_bytes_per_second * sym.Min(self.workers, sym.Max(1, stage.parallel_tasks)))
                   + stage.atomic_bytes / 4 / self.atomic_updates_per_second)
        estimated = dispatch + commits + execution
        return replace(stage, estimated_seconds=estimated, kernel_calls=calls,
                       realization_read_bytes=reads, realization_write_bytes=writes,
                       realization_allocated_bytes=allocated, compute_seconds=compute,
                       memory_seconds=memory, dispatch_seconds=dispatch,
                       cache_penalty_seconds=penalty)
