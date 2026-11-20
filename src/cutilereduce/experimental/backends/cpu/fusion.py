"""Native SIMD expression/reduction fusion for named-axis CPU tensor regions.

GEMMs, scans, mixed dtypes and unsupported scalar operations remain ATen
boundaries. Expressions use actual input strides; only contiguous internal
last-use buffers may be recycled. Formal arguments are always borrowed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from cutilereduce.experimental.ir.core import ContractionAxes, TensorRegion


HEADERS = """#ifndef CUTILEREDUCE_CPU_FUSION_HEADERS
#define CUTILEREDUCE_CPU_FUSION_HEADERS
// The extension is compiled for the local CPU. Standalone portable emission
// without -march=native retains the generic vector implementation.
#if defined(__AVX2__) && !defined(CPU_CAPABILITY)
#define CPU_CAPABILITY AVX2
#define CPU_CAPABILITY_AVX2
#endif
#include <ATen/cpu/vec/vec.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <limits>
#endif
"""

_UNARY = {"neg", "exp", "exp2", "log", "log2", "log1p", "expm1", "sqrt",
          "rsqrt", "abs", "sin", "cos", "tanh", "sigmoid", "stop_gradient"}
_BINARY = {"add", "subtract", "multiply", "divide", "minimum", "maximum"}
_CONSTANT = {"constant", "full", "zero", "domain_valid"}
_SELECT = {"where", "mask_identity"}


def _scalar_op(node):
    return (node.op in _UNARY | _BINARY | _CONSTANT | _SELECT
            or node.op.startswith("compare_") or node.op in {"logical_and", "logical_or", "logical_not"})


@dataclass(frozen=True)
class FusionGroup:
    root: int
    leaves: tuple[int, ...]
    inside: tuple[int, ...]
    axes: tuple[str, ...]
    reduce_axes: tuple[str, ...] = ()
    reduction: str | None = None
    dot: bool = False


@dataclass(frozen=True)
class EmissionStep:
    root: int
    kind: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    inside: tuple[int, ...] = ()
    reused_from: int | None = None


class RegionFusion:
    def __init__(self, region: TensorRegion):
        self.region = region
        self.results = {i for _, i in region.results}
        self.arguments = {i for _, i in region.arguments}
        self.users = Counter(i for n in region.nodes for i in n.inputs)
        self.groups = {}
        boundaries = {n.id for n in region.nodes
                      if not _scalar_op(n) or n.id in self.results
                      or (self.users[n.id] > 1 and n.dtype != "bool"
                          and n.op not in _CONSTANT | {"add", "subtract", "multiply", "neg"})}
        boundaries.update(self.arguments)
        for node in region.nodes:
            if node.id not in boundaries or node.id in self.arguments or node.dtype not in {"float32", "float64"}:
                continue
            reduction = None
            reduce_axes = ()
            axes = node.axes
            dot = False
            starts = (node.id,)
            if node.op in {"reduce_sum", "reduce_max"}:
                reduction = "sum" if node.op == "reduce_sum" else "max"
                source = region.nodes[node.inputs[0]]
                reduce_axes = tuple(node.attribute("over"))
                axes = source.axes
                if not reduce_axes or axes[-len(reduce_axes):] != reduce_axes:
                    continue
                starts = node.inputs
            elif node.op in {"contract", "contract_accumulate"}:
                partition = dict(node.attributes).get("contraction_axes")
                if (node.op != "contract" or not isinstance(partition, ContractionAxes)
                        or partition.lhs_free or partition.rhs_free
                        or dict(node.attributes).get("precision") is not None):
                    continue
                dot = True
                reduction = "sum"
                reduce_axes = partition.reduce
                if not reduce_axes:
                    continue
                axes = (*node.axes, *reduce_axes)
                starts = node.inputs
            elif not _scalar_op(node) or node.op in _CONSTANT:
                continue
            inside = set()
            leaves = set()
            valid = True

            def visit(i, root=False):
                nonlocal valid
                if i in inside or i in leaves:
                    return
                value = region.nodes[i]
                if not root and i in boundaries:
                    leaves.add(i)
                    if value.dtype != node.dtype:
                        valid = False
                    return
                if not _scalar_op(value) or value.dtype not in {node.dtype, "bool"}:
                    valid = False
                    return
                if not set(value.axes) <= set(axes):
                    valid = False
                    return
                inside.add(i)
                for operand in value.inputs:
                    visit(operand)

            for i in starts:
                visit(i, root=i == node.id)
            if valid and leaves:
                # A flattened reduction can SIMD-load an operand if it carries
                # all reduced axes or none. Other broadcasts use ATen fallback.
                if any(set(region.nodes[i].axes) & set(reduce_axes) not in (set(), set(reduce_axes)) for i in leaves):
                    continue
                self.groups[node.id] = FusionGroup(node.id, tuple(sorted(leaves)), tuple(sorted(inside)),
                                                    axes, reduce_axes, reduction, dot)
        # A materialized pointwise value can produce its row sum in the same
        # pass. This retains the value for GEMM without reading it again to sum.
        self.combined_reductions = {}
        for node in region.nodes:
            if node.op != "reduce_sum" or node.id not in self.groups:
                continue
            source = region.nodes[node.inputs[0]]
            group = self.groups.get(source.id)
            if (group is not None and group.reduction is None and source.axes
                    and tuple(node.attribute("over")) == source.axes[-1:]
                    and source.id not in self.combined_reductions):
                self.combined_reductions[source.id] = node.id
        self.required = set()

        def require(i):
            if i in self.required:
                return
            self.required.add(i)
            for operand in (self.groups[i].leaves if i in self.groups else region.nodes[i].inputs):
                require(operand)
        for i in self.results:
            require(i)
        self.last_use = {}
        for node in region.nodes:
            if node.id in self.required:
                for operand in (self.groups[node.id].leaves if node.id in self.groups else node.inputs):
                    self.last_use[operand] = node.id
        for i in self.results:
            self.last_use[i] = len(region.nodes)
        # Fallback casts/detaches can alias their input storage. Protect the
        # complete alias chain even if a later SSA name outlives its producer.
        self.protected = set(self.arguments) | self.results
        for node in region.nodes:
            if (node.id in self.required and node.id not in self.groups
                    and (node.op in {"cast", "stop_gradient"} or node.op.startswith("associative_"))):
                pending = [node.id]
                while pending:
                    i = pending.pop()
                    self.protected.add(i)
                    if region.nodes[i].op in {"cast", "stop_gradient"} or region.nodes[i].op.startswith("associative_"):
                        pending.extend(region.nodes[i].inputs)

        # This shape-independent storage plan is shared by code emission and
        # analytical costing. C++ Tensor locals retain their storage to region
        # return; a last-use buffer may be recycled but is not implicitly freed.
        raw = []
        emitted = set(self.arguments)
        for node in region.nodes:
            if node.id not in self.required or node.id in emitted:
                continue
            group = self.groups.get(node.id)
            outputs = (node.id,)
            if node.id in self.combined_reductions:
                outputs += (self.combined_reductions[node.id],)
            if group is not None:
                kind = "reduction" if group.reduction else "pointwise"
                inputs, emitted_inside = group.leaves, group.inside
            else:
                inputs, emitted_inside = node.inputs, ()
                kind = "gemm" if node.op in {"contract", "contract_accumulate"} else "aten"
                if node.op in _CONSTANT:
                    kind = "factory"
                if node.op == "stop_gradient" or (node.op == "cast" and region.nodes[node.inputs[0]].dtype == node.dtype):
                    kind = "alias"
            raw.append(EmissionStep(node.id, kind, inputs, outputs, emitted_inside))
            emitted.update(outputs)
        self.last_use = {i: len(region.nodes) for i in self.results}
        for step in raw:
            for i in step.inputs:
                if i not in self.results:
                    self.last_use[i] = step.root
        owned = set()
        contiguous = set()
        steps = []
        for step in raw:
            node = region.nodes[step.root]
            reused = None
            if step.kind == "pointwise":
                reused = next((i for i in step.inputs if i in owned and i in contiguous
                               and i not in self.protected and self.last_use.get(i) == step.root
                               and region.nodes[i].axes == node.axes), None)
            if reused is not None:
                owned.discard(reused)
            if step.kind != "alias" and not node.op.startswith("associative_"):
                owned.update(step.outputs)
            if step.kind in {"pointwise", "reduction", "factory"}:
                contiguous.update(step.outputs)
            elif step.kind == "gemm":
                partition = node.attribute("contraction_axes")
                if (*partition.batch, *partition.lhs_free, *partition.rhs_free) == node.axes:
                    contiguous.add(node.id)
            steps.append(EmissionStep(step.root, step.kind, step.inputs, step.outputs, step.inside, reused))
        self.steps = tuple(steps)
        self.reused_buffers = {step.root: step.reused_from for step in steps}

    def emit(self, root, values, sizes, owned):
        group = self.groups[root]
        region = self.region
        node = region.nodes[root]
        combined = self.combined_reductions.get(root)
        reducing = group.reduction is not None or combined is not None
        reduction = group.reduction or ("sum" if combined is not None else None)
        scalar = "float" if node.dtype == "float32" else "double"
        vector = f"at::vec::Vectorized<{scalar}>"
        output = f"v{root}"
        reused = self.reused_buffers[root]
        shape = "{" + ", ".join(str(sizes[a]) for a in node.axes) + "}"
        dtype = "at::kFloat" if scalar == "float" else "at::kDouble"
        allocation = f"at::empty({shape}, at::TensorOptions().dtype({dtype}))"
        if reused is not None:
            allocation = f"({values[reused]}.is_contiguous() ? {values[reused]} : {allocation})"
            owned.discard(reused)
        owned.add(root)
        lines = [f"  // fused {'reduction' if group.reduction else 'pointwise'}; "
                 + (f"reuse last-use %{reused}" if reused is not None else "fresh output"),
                 f"  auto {output} = {allocation};", f"  {{ using Vec = {vector};",
                 f"    auto* out = {output}.data_ptr<{scalar}>();"]
        if combined is not None:
            sum_node = region.nodes[combined]
            sum_shape = "{" + ", ".join(str(sizes[a]) for a in sum_node.axes) + "}"
            lines.insert(1, f"  auto v{combined} = at::empty({sum_shape}, at::TensorOptions().dtype({dtype}));")
            lines.append(f"    auto* sum_out = v{combined}.data_ptr<{scalar}>();")
        for i in group.leaves:
            lines.append(f"    const auto* p{i} = {values[i]}.const_data_ptr<{scalar}>();")
        inner_axes = group.reduce_axes if group.reduction else group.axes[-1:]
        outer_axes = tuple(a for a in group.axes if a not in inner_axes)
        inner = math.prod(sizes[a] for a in inner_axes)
        outer = math.prod(sizes[a] for a in outer_axes)

        def offset(i, axes, index):
            value = region.nodes[i]
            terms = []
            for k, axis in enumerate(axes):
                if axis in value.axes:
                    divisor = math.prod(sizes[a] for a in axes[k + 1:])
                    terms.append(f"(({index} / {divisor}) % {sizes[axis]}) * {values[i]}.stride({value.axes.index(axis)})")
            return " + ".join(terms) or "0"

        checks = []
        strides = {}
        for i in group.leaves:
            value = region.nodes[i]
            carries = [a for a in inner_axes if a in value.axes]
            strides[i] = f"{values[i]}.stride({value.axes.index(inner_axes[-1])})" if carries else "0"
            if carries:
                checks.append(f"({strides[i]} == 1)")
                for k, axis in enumerate(inner_axes[:-1]):
                    expected = math.prod(sizes[a] for a in inner_axes[k + 1:])
                    checks.append(f"({values[i]}.stride({value.axes.index(axis)}) == {expected})")
        vectorizable = " && ".join(checks) or "true"
        lines.extend([f"    const bool vectorizable = {vectorizable};",
                      f"    at::parallel_for(0, {outer}, {max(1, 32768 // max(inner, 1))}, [&](int64_t begin, int64_t end) {{",
                      "      for (int64_t row = begin; row < end; ++row) {"])
        for i in group.leaves:
            lines.append(f"        const auto* r{i} = p{i} + ({offset(i, outer_axes, 'row')});")
        if reducing:
            initial = f"-std::numeric_limits<{scalar}>::infinity()" if reduction == "max" else f"{scalar}(0)"
            lines.extend([f"        {scalar} total = {initial};", f"        Vec accumulated({initial});"])
        lines.extend(["        int64_t j = 0;", f"        if (vectorizable) for (; j + Vec::size() <= {inner}; j += Vec::size()) {{"])
        for i in group.leaves:
            lines.append(f"          Vec a{i} = " + (f"Vec(r{i}[0]);" if strides[i] == "0" else f"Vec::loadu(r{i} + j);"))
        expr_lines, expression = self.expression(group, vector=True, scalar=scalar)
        lines.extend(f"          {line}" for line in expr_lines)
        if combined is not None:
            lines.append(f"          ({expression}).store(out + row * {inner} + j);")
        if reduction == "sum":
            lines.append(f"          accumulated = accumulated + {expression};")
        elif reduction == "max":
            lines.append(f"          accumulated = at::vec::maximum(accumulated, {expression});")
        else:
            lines.append(f"          ({expression}).store(out + row * {inner} + j);")
        lines.append("        }")
        if reducing:
            lines.extend([f"        {scalar} lanes[Vec::size()];", "        accumulated.store(lanes);",
                          "        for (int lane = 0; lane < Vec::size(); ++lane) {"])
            lines.append("          total += lanes[lane];" if reduction == "sum" else
                         "          total = std::isnan(lanes[lane]) || lanes[lane] > total ? lanes[lane] : total;")
            lines.append("        }")
        lines.append(f"        for (; j < {inner}; ++j) {{")
        for i in group.leaves:
            lines.append(f"          {scalar} a{i} = r{i}[{offset(i, inner_axes, 'j')}];")
        expr_lines, expression = self.expression(group, vector=False, scalar=scalar)
        lines.extend(f"          {line}" for line in expr_lines)
        if combined is not None:
            lines.append(f"          out[row * {inner} + j] = {expression};")
        if reduction == "sum":
            lines.append(f"          total += {expression};")
        elif reduction == "max":
            lines.append(f"          auto item = {expression}; total = std::isnan(item) || item > total ? item : total;")
        else:
            lines.append(f"          out[row * {inner} + j] = {expression};")
        lines.append("        }")
        if reducing:
            lines.append("        sum_out[row] = total;" if combined is not None else "        out[row] = total;")
        lines.extend(["      }", "    });", "  }"])
        return lines

    def expression(self, group, *, vector, scalar):
        from .regions import _number
        values = {i: f"a{i}" for i in group.leaves}
        lines = []
        nodes = self.region.nodes
        def constant(value):
            return f"{'Vec' if vector else scalar}({_number(value)})"
        def get(i):
            if i in values:
                return values[i]
            n = nodes[i]
            x = [get(k) for k in n.inputs]
            op = n.op
            if op in _CONSTANT:
                if op == "domain_valid":
                    expression = "true"
                elif n.dtype == "bool":
                    expression = "true" if dict(n.attributes).get("value", 0) else "false"
                else:
                    expression = constant(0 if op == "zero" else n.attribute("value"))
            elif op in _BINARY - {"minimum", "maximum"}:
                symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
                expression = f"({x[0]} {symbol} {x[1]})"
            elif op in {"minimum", "maximum"}:
                function = "min" if op == "minimum" else "max"
                expression = (f"at::vec::{op}({x[0]}, {x[1]})" if vector else
                              f"(std::isnan({x[0]}) ? {x[0]} : (std::isnan({x[1]}) ? {x[1]} : std::{function}({x[0]}, {x[1]})))")
            elif op.startswith("compare_"):
                name = op.removeprefix("compare_")
                cmp = {"eq": "==", "equal": "==", "ne": "!=", "not_equal": "!=", "lt": "<", "less": "<", "le": "<=", "less_equal": "<=", "gt": ">", "greater": ">", "ge": ">=", "greater_equal": ">="}[name]
                expression = f"({x[0]} {cmp} {x[1]})"
            elif op in _SELECT:
                if x[0] in {"true", "false"}:
                    expression = x[1] if x[0] == "true" else x[2]
                else:
                    if vector and n.dtype == "bool":
                        # SIMD predicates are all-bits masks, not floating 0/1.
                        # A selected true value must remain a mask when this
                        # boolean expression controls another blend or bitwise op.
                        def mask(value):
                            if value == "true":
                                return "(Vec(0) == Vec(0))"
                            if value == "false":
                                return "(Vec(0) != Vec(0))"
                            return value
                        expression = f"Vec::blendv({mask(x[2])}, {mask(x[1])}, {mask(x[0])})"
                    else:
                        expression = (f"Vec::blendv({x[2]}, {x[1]}, {x[0]})" if vector else f"({x[0]} ? {x[1]} : {x[2]})")
            elif op == "stop_gradient":
                expression = x[0]
            elif op == "neg":
                expression = f"(-{x[0]})"
            elif op == "sigmoid":
                exponential = f"(-{x[0]}).exp()" if vector else f"std::exp(-{x[0]})"
                expression = f"({constant(1)} / ({constant(1)} + {exponential}))"
            elif op == "rsqrt":
                expression = f"({constant(1)} / " + (f"{x[0]}.sqrt())" if vector else f"std::sqrt({x[0]}))")
            elif op in {"logical_and", "logical_or", "logical_not"}:
                symbol = {"logical_and": "&" if vector else "&&", "logical_or": "|" if vector else "||", "logical_not": "~" if vector else "!"}[op]
                if op == "logical_not" and x[0] in {"true", "false"}:
                    expression = "false" if x[0] == "true" else "true"
                elif op == "logical_and" and any(a in {"true", "false"} for a in x):
                    expression = "false" if "false" in x else next((a for a in x if a != "true"), "true")
                elif op == "logical_or" and any(a in {"true", "false"} for a in x):
                    expression = "true" if "true" in x else next((a for a in x if a != "false"), "false")
                else:
                    expression = f"({symbol}{x[0]})" if len(x) == 1 else f"({x[0]} {symbol} {x[1]})"
            else:
                expression = f"{x[0]}.{op}()" if vector else f"std::{op}({x[0]})"
            if op in _CONSTANT:
                values[i] = expression
            else:
                if vector and n.dtype == "bool" and expression in {"true", "false"}:
                    expression = "(Vec(0) == Vec(0))" if expression == "true" else "(Vec(0) != Vec(0))"
                name = f"e{i}"
                lines.append(f"auto {name} = {expression};")
                values[i] = name
            return values[i]
        if group.dot:
            root = nodes[group.root]
            result = f"({get(root.inputs[0])} * {get(root.inputs[1])})"
        elif group.reduction:
            result = get(nodes[group.root].inputs[0])
        else:
            result = get(group.root)
        return lines, result
