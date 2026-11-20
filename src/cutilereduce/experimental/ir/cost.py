"""Symbolic operation counts and value sizes; region boundaries imply no traffic."""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
from typing import Any

from cutilereduce.core import symbolic as sym
from .core import ContractionAxes, TensorProgram, TensorRegion
from .ops import ELEMENTWISE_OPS, multiplication_chain

DTYPE_BYTES = {
    "bool": 1, "int8": 1, "int16": 2, "float16": 2, "bfloat16": 2,
    "int32": 4, "float32": 4, "int64": 8, "float64": 8,
}


def product(values):
    result = 1
    for value in values:
        result = result * value
    return result


def bind(value, values: Mapping[str, Any]):
    return value.subs(values) if isinstance(value, sym.Expr) else value


def elements(axes, extents: Mapping[str, Any]):
    return product(extents[axis] for axis in axes)


@dataclass(frozen=True)
class OperationCost:
    """Scalar counts, exact for the stated algorithm (FMA counts as two FLOPs).

    Reductions and scans use a serial linear-work convention, independent of
    their eventual tree/span. Backend timing rates remain separate.
    """

    contraction_flops: Any = 0
    arithmetic: Any = 0
    comparisons: Any = 0
    transcendentals: Any = 0
    conversions: Any = 0
    reduction_ops: Any = 0
    scan_ops: Any = 0

    def __add__(self, other: OperationCost) -> OperationCost:
        def add(left, right):
            if not isinstance(left, sym.Expr) and left == 0:
                return right
            if not isinstance(right, sym.Expr) and right == 0:
                return left
            return left + right
        return OperationCost(**{f.name: add(getattr(self, f.name), getattr(other, f.name))
                                for f in fields(self)})

    def scale(self, count) -> OperationCost:
        return OperationCost(**{f.name: (0 if not isinstance(value := getattr(self, f.name), sym.Expr) and value == 0 else value * count)
                                for f in fields(self)})

    def subs(self, values: Mapping[str, Any]) -> OperationCost:
        return OperationCost(**{f.name: bind(getattr(self, f.name), values) for f in fields(self)})

    @property
    def total(self):
        return sum(value for f in fields(self)
                   if isinstance(value := getattr(self, f.name), sym.Expr) or value != 0)


@dataclass(frozen=True)
class RegionCost:
    work: OperationCost
    argument_bytes: tuple[tuple[str, Any], ...]
    result_bytes: tuple[tuple[str, Any], ...]
    live_bytes_upper_bound: Any
    convention: str = "linear reductions/scans; dense values; no implied boundary traffic"
    realization: RegionRealizationCost | None = None

    def subs(self, values: Mapping[str, Any]) -> RegionCost:
        return RegionCost(
            self.work.subs(values),
            tuple((name, bind(size, values)) for name, size in self.argument_bytes),
            tuple((name, bind(size, values)) for name, size in self.result_bytes),
            bind(self.live_bytes_upper_bound, values), self.convention,
            self.realization.subs(values) if self.realization is not None else None,
        )


def analyze_region(region: TensorRegion | TensorProgram, extents=None) -> RegionCost:
    """Analyze the supplied IR, without modifying it or allocating tensor values.

    Extents default to named symbols. Shared SSA producers and structured
    associative invocations are counted once. Unknown operations fail explicitly.
    The live-byte estimate is a conservative dense SSA bound, not register usage.
    """
    sizes = {axis: sym.Symbol(axis) for axis in region.axes}
    sizes.update(extents or {})
    nodes = region.nodes
    results = region.results if isinstance(region, TensorRegion) else region.outputs
    arguments = (region.arguments if isinstance(region, TensorRegion) else
                 tuple((str(dict(n.attributes).get("name", n.id)), n.id)
                       for n in nodes if n.op == "input"))
    reachable = set()
    pending = [node_id for _, node_id in results]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(nodes[node_id].inputs)
    work = OperationCost()
    groups = set()
    allocations = {}
    last_use = {}
    for node in nodes:
        if node.id not in reachable:
            continue
        for operand in node.inputs:
            last_use[operand] = node.id
    for _, node_id in results:
        last_use[node_id] = len(nodes)
    peaks = [0]
    live = {}
    free_ops = {"argument", "input", "constant", "full", "zero", "axis_index",
                "domain_valid", "stop_gradient", "materialize"}
    transcendental_ops = {"exp", "exp2", "log", "log2", "log1p", "expm1",
                          "sin", "cos", "tanh", "sigmoid", "sqrt", "rsqrt"}
    for node in nodes:
        if node.id not in reachable:
            continue
        count = elements(node.axes, sizes)
        attrs = dict(node.attributes)
        if node.op in {"contract", "contract_accumulate"}:
            partition = attrs["contraction_axes"]
            if not isinstance(partition, ContractionAxes):
                raise TypeError("invalid contraction metadata")
            work += OperationCost(contraction_flops=2 * count * elements(partition.reduce, sizes))
        elif node.op.startswith("reduce_"):
            work += OperationCost(reduction_ops=count * sym.SaturateSub(elements(attrs["over"], sizes), 1))
        elif node.op.startswith("scan_"):
            axis = attrs["over"]
            scan_extent = elements((axis,) if isinstance(axis, str) else axis, sizes)
            outer = elements(tuple(a for a in node.axes if a not in ((axis,) if isinstance(axis, str) else axis)), sizes)
            work += OperationCost(scan_ops=outer * sym.SaturateSub(scan_extent, 1))
        elif node.op.startswith("associative_"):
            nested = attrs["combine_region"]
            # Multiple result fields (and scan final/prefix) describe one invocation.
            key = (node.inputs, repr(nested), attrs["over"])
            if key not in groups:
                groups.add(key)
                work += analyze_region(nested, sizes).work.scale(elements(attrs["over"], sizes))
        elif node.op in free_ops:
            pass
        elif node.op == "cast":
            work += OperationCost(conversions=count)
        elif node.op in transcendental_ops:
            work += OperationCost(transcendentals=count)
        elif node.op == "powi":
            exponent = attrs["exponent"]
            work += OperationCost(arithmetic=count * (len(multiplication_chain(abs(exponent))) + int(exponent < 0) if exponent else 0))
        elif node.op in {"where", "mask_identity"} or node.op.startswith("compare_"):
            work += OperationCost(comparisons=count)
        elif node.op in ELEMENTWISE_OPS:
            category = ELEMENTWISE_OPS[node.op].differentiation
            work += (OperationCost(comparisons=count) if category == "none"
                     else OperationCost(arithmetic=count))
        else:
            raise NotImplementedError(f"no analytical rule for tensor op {node.op!r}")
        if node.op not in free_ops:
            allocations[node.id] = count * DTYPE_BYTES[node.dtype]
            live[node.id] = allocations[node.id]
        peaks.append(sum(live.values()))
        live = {key: size for key, size in live.items() if last_use.get(key, key) > node.id}
    def byte_sizes(items):
        return tuple((name, elements(nodes[i].axes, sizes) * DTYPE_BYTES[nodes[i].dtype]) for name, i in items)
    return RegionCost(work, byte_sizes(arguments), byte_sizes(results), sym.Max(*peaks))


def analyze_operation(node, nodes, extents):
    """Count one operation without charging its producers.

    Shares the region operation rules, including backend-independent BMNK work.
    Input signatures are retained for rules that inspect operand dimensions.
    """
    from .core import TensorNode
    leaves = tuple(TensorNode(i, "argument", (), nodes[n].axes, nodes[n].dtype)
                   for i, n in enumerate(node.inputs))
    from dataclasses import replace
    operation = replace(node, id=len(leaves), inputs=tuple(range(len(leaves))))
    region = TensorRegion(tuple(extents), (*leaves, operation),
                          tuple((str(i), i) for i in range(len(leaves))),
                          (("result", operation.id),))
    return analyze_region(region, extents).work


@dataclass(frozen=True)
class KernelCost:
    """One realized execution boundary, with symbolic scalar counts and bytes.

    Working set counts distinct storage touched by this kernel, including a
    reused output only once. Traffic counts reads/writes separately. GEMM shape
    is (batch, M, N, K); actual packing and cache hits remain target estimates.
    """

    kind: str
    work: OperationCost
    read_bytes: Any
    write_bytes: Any
    allocated_bytes: Any
    working_set_bytes: Any
    elements: Any
    gemm_shape: tuple[Any, ...] = ()
    reused_from: int | None = None
    outputs: tuple[int, ...] = ()

    def subs(self, values):
        return KernelCost(self.kind, self.work.subs(values),
                          bind(self.read_bytes, values), bind(self.write_bytes, values),
                          bind(self.allocated_bytes, values), bind(self.working_set_bytes, values),
                          bind(self.elements, values), tuple(bind(x, values) for x in self.gemm_shape),
                          self.reused_from, self.outputs)


@dataclass(frozen=True)
class RegionRealizationCost:
    kernels: tuple[KernelCost, ...]
    peak_owned_bytes: Any
    convention: str = "realization boundaries; dense contiguous reuse; no cache-hit guarantee"

    @property
    def allocated_bytes(self):
        return sum(k.allocated_bytes for k in self.kernels)

    def subs(self, values):
        return RegionRealizationCost(tuple(k.subs(values) for k in self.kernels),
                                     bind(self.peak_owned_bytes, values), self.convention)
