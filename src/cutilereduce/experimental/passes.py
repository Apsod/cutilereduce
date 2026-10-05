"""Canonical optimization pipeline for the experimental tensor IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import ContractionAxes, TensorNode, TensorProgram, TensorRegion


@dataclass(frozen=True)
class AxisAffineIndex:
    """A logical axis coordinate plus a constant offset."""

    axis: str | None
    offset: int | float = 0


@dataclass(frozen=True)
class TileMaskComparison:
    """An identity mask suitable for tile-interval satisfiability analysis."""

    operation: str
    left: AxisAffineIndex
    right: AxisAffineIndex


def discover_tile_mask(region: TensorRegion) -> TileMaskComparison | None:
    """Recognize one shared identity mask over affine logical coordinates."""
    conditions = {node.inputs[0] for node in region.nodes if node.op == "mask_identity"}
    if len(conditions) != 1:
        return None
    condition = region.nodes[conditions.pop()]
    if not condition.op.startswith("compare_"):
        return None

    def affine(node: TensorNode) -> AxisAffineIndex | None:
        if node.op == "axis_index":
            return AxisAffineIndex(str(node.attribute("axis")))
        if node.op == "constant":
            value = node.attribute("value")
            return (
                AxisAffineIndex(None, value)
                if isinstance(value, (int, float))
                else None
            )
        if node.op not in {"add", "subtract"}:
            return None
        left = affine(region.nodes[node.inputs[0]])
        right = affine(region.nodes[node.inputs[1]])
        if left is None or right is None:
            return None
        if node.op == "add":
            if left.axis is not None and right.axis is None:
                return AxisAffineIndex(left.axis, left.offset + right.offset)
            if right.axis is not None and left.axis is None:
                return AxisAffineIndex(right.axis, right.offset + left.offset)
        elif left.axis is not None and right.axis is None:
            return AxisAffineIndex(left.axis, left.offset - right.offset)
        return None

    left = affine(region.nodes[condition.inputs[0]])
    right = affine(region.nodes[condition.inputs[1]])
    if left is None or right is None:
        return None
    return TileMaskComparison(condition.op.removeprefix("compare_"), left, right)


def _attribute(node: TensorNode, name: str, default: Any = None) -> Any:
    return dict(node.attributes).get(name, default)


def _renumber(
    program: TensorProgram,
    kept: set[int],
) -> TensorProgram:
    remap = {old: new for new, old in enumerate(sorted(kept))}
    nodes = tuple(
        TensorNode(
            id=remap[node.id],
            op=node.op,
            inputs=tuple(remap[value] for value in node.inputs),
            axes=node.axes,
            dtype=node.dtype,
            attributes=node.attributes,
        )
        for node in program.nodes
        if node.id in kept
    )
    outputs = tuple((name, remap[node_id]) for name, node_id in program.outputs)
    return TensorProgram(program.axes, nodes, outputs)


def simplify_constants(program: TensorProgram) -> TensorProgram:
    """Eliminate elementary identities and redundant casts."""
    nodes: list[TensorNode] = []
    remap: dict[int, int] = {}

    def constant_value(node_id: int) -> int | float | None:
        node = nodes[node_id]
        if node.op not in {"constant", "full"}:
            return None
        value = _attribute(node, "value")
        return value if isinstance(value, int | float) else None

    for old in program.nodes:
        inputs = tuple(remap[value] for value in old.inputs)
        alias: int | None = None
        zero_result = False
        if old.op == "cast" and nodes[inputs[0]].dtype == old.dtype:
            alias = inputs[0]
        elif old.op == "neg" and nodes[inputs[0]].op == "neg":
            alias = nodes[inputs[0]].inputs[0]
        elif old.op in {"add", "subtract", "multiply", "divide"}:
            left, right = inputs
            left_value = constant_value(left)
            right_value = constant_value(right)
            if old.op == "add":
                if nodes[right].op == "zero" and nodes[left].axes == old.axes:
                    alias = left
                elif nodes[left].op == "zero" and nodes[right].axes == old.axes:
                    alias = right
                elif right_value == 0 and nodes[left].axes == old.axes:
                    alias = left
                elif left_value == 0 and nodes[right].axes == old.axes:
                    alias = right
            elif old.op == "subtract":
                if nodes[right].op == "zero" and nodes[left].axes == old.axes:
                    alias = left
                elif right_value == 0 and nodes[left].axes == old.axes:
                    alias = left
                elif nodes[left].op == "zero" and nodes[right].axes == old.axes:
                    new_id = len(nodes)
                    remap[old.id] = new_id
                    nodes.append(
                        TensorNode(
                            new_id,
                            "neg",
                            (right,),
                            old.axes,
                            old.dtype,
                        )
                    )
                    continue
            elif old.op == "multiply":
                if nodes[left].op == "zero" or nodes[right].op == "zero":
                    zero_result = True
                elif right_value == 1 and nodes[left].axes == old.axes:
                    alias = left
                elif left_value == 1 and nodes[right].axes == old.axes:
                    alias = right
                elif right_value == -1 and nodes[left].axes == old.axes:
                    new_id = len(nodes)
                    remap[old.id] = new_id
                    nodes.append(
                        TensorNode(new_id, "neg", (left,), old.axes, old.dtype)
                    )
                    continue
                elif left_value == -1 and nodes[right].axes == old.axes:
                    new_id = len(nodes)
                    remap[old.id] = new_id
                    nodes.append(
                        TensorNode(new_id, "neg", (right,), old.axes, old.dtype)
                    )
                    continue
            elif old.op == "divide":
                if right_value == 1 and nodes[left].axes == old.axes:
                    alias = left

        if alias is not None:
            remap[old.id] = alias
            continue
        if old.op.startswith("reduce_") and nodes[inputs[0]].op == "zero":
            zero_result = True
        if old.op == "contract" and any(nodes[value].op == "zero" for value in inputs):
            zero_result = True
        new_id = len(nodes)
        remap[old.id] = new_id
        if zero_result:
            nodes.append(TensorNode(new_id, "zero", (), old.axes, old.dtype))
            continue
        nodes.append(
            TensorNode(
                id=new_id,
                op=old.op,
                inputs=inputs,
                axes=old.axes,
                dtype=old.dtype,
                attributes=old.attributes,
            )
        )

    return TensorProgram(
        program.axes,
        tuple(nodes),
        tuple((name, remap[node_id]) for name, node_id in program.outputs),
    )


def eliminate_dead_code(program: TensorProgram) -> TensorProgram:
    """Retain only nodes reachable from declared program outputs."""
    live: set[int] = set()
    pending = [node_id for _, node_id in program.outputs]
    while pending:
        node_id = pending.pop()
        if node_id in live:
            continue
        live.add(node_id)
        pending.extend(program.node(node_id).inputs)
    return _renumber(program, live)


def _contraction_axes(
    program_axes: tuple[str, ...],
    lhs: TensorNode,
    rhs: TensorNode,
    reduce: tuple[str, ...],
) -> ContractionAxes:
    order = {axis: index for index, axis in enumerate(program_axes)}

    def ordered(values: set[str]) -> tuple[str, ...]:
        return tuple(sorted(values, key=order.__getitem__))

    reduced = set(reduce)
    lhs_axes = set(lhs.axes) - reduced
    rhs_axes = set(rhs.axes) - reduced
    return ContractionAxes(
        batch=ordered(lhs_axes & rhs_axes),
        lhs_free=ordered(lhs_axes - rhs_axes),
        rhs_free=ordered(rhs_axes - lhs_axes),
        reduce=ordered(reduced),
    )


def discover_contractions(program: TensorProgram) -> TensorProgram:
    """Recognize sum-of-products and zero-masked products as contractions."""
    nodes: list[TensorNode] = []
    remap: dict[int, int] = {}

    def is_zero(node: TensorNode) -> bool:
        if node.op == "zero":
            return True
        return node.op in {"constant", "full"} and _attribute(node, "value") == 0

    for old in program.nodes:
        mapped_inputs = tuple(remap[value] for value in old.inputs)
        if old.op == "reduce_sum" and len(old.inputs) == 1:
            producer = program.nodes[old.inputs[0]]
            condition_id: int | None = None
            if producer.op == "where" and len(producer.inputs) == 3:
                condition, product_id, zero_id = producer.inputs
                product = program.nodes[product_id]
                if (
                    is_zero(program.nodes[zero_id])
                    and program.nodes[condition].dtype == "bool"
                    and product.op == "multiply"
                    and len(product.inputs) == 2
                ):
                    condition_id = condition
            else:
                product = producer

            reduce = _attribute(old, "over")
            if (
                product.op == "multiply"
                and len(product.inputs) == 2
                and isinstance(reduce, tuple)
            ):
                lhs = nodes[remap[product.inputs[0]]]
                rhs = nodes[remap[product.inputs[1]]]
                reduction = set(reduce)
                if condition_id is not None:
                    condition = nodes[remap[condition_id]]
                    # Move a zeroing mask onto one operand only when it does
                    # not require expanding that operand to a larger rank.
                    selected: int | None = None
                    for candidate, operand in enumerate((lhs, rhs)):
                        if (
                            set(condition.axes) <= set(operand.axes)
                            and operand.dtype == product.dtype
                        ):
                            selected = candidate
                            break
                    if selected is not None:
                        operand = (lhs, rhs)[selected]
                        zero = TensorNode(
                            id=len(nodes),
                            op="zero",
                            inputs=(),
                            axes=operand.axes,
                            dtype=operand.dtype,
                        )
                        nodes.append(zero)
                        masked = TensorNode(
                            id=len(nodes),
                            op="where",
                            inputs=(condition.id, operand.id, zero.id),
                            axes=operand.axes,
                            dtype=operand.dtype,
                        )
                        nodes.append(masked)
                        if selected == 0:
                            lhs = masked
                        else:
                            rhs = masked
                    else:
                        lhs = rhs = None
                if lhs is not None and rhs is not None and reduction <= (
                    set(lhs.axes) & set(rhs.axes)
                ):
                    result_axes = tuple(
                        axis
                        for axis in program.axes
                        if axis in (set(lhs.axes) | set(rhs.axes)) - reduction
                    )
                    if result_axes == old.axes:
                        partition = _contraction_axes(
                            program.axes, lhs, rhs, reduce
                        )
                        contraction = TensorNode(
                            id=len(nodes),
                            op="contract",
                            inputs=(lhs.id, rhs.id),
                            axes=old.axes,
                            dtype=old.dtype,
                            attributes=(
                                ("contraction_axes", partition),
                                ("implementation", partition.kind),
                            ),
                        )
                        nodes.append(contraction)
                        remap[old.id] = contraction.id
                        continue

        new_id = len(nodes)
        remap[old.id] = new_id
        nodes.append(
            TensorNode(
                id=new_id,
                op=old.op,
                inputs=mapped_inputs,
                axes=old.axes,
                dtype=old.dtype,
                attributes=old.attributes,
            )
        )

    rewritten = TensorProgram(
        program.axes,
        tuple(nodes),
        tuple((name, remap[node_id]) for name, node_id in program.outputs),
    )
    return eliminate_dead_code(rewritten)


def canonicalize_contractions(program: TensorProgram) -> TensorProgram:
    """Recompute normalized named-axis partitions and implementation hints."""
    nodes = list(program.nodes)
    for index, node in enumerate(nodes):
        if node.op not in {"contract", "contract_accumulate"}:
            continue
        old_partition = _attribute(node, "contraction_axes")
        if not isinstance(old_partition, ContractionAxes):
            raise TypeError("contract node has invalid axis metadata")
        lhs = nodes[node.inputs[0]]
        rhs = nodes[node.inputs[1]]
        partition = _contraction_axes(program.axes, lhs, rhs, old_partition.reduce)
        nodes[index] = TensorNode(
            id=node.id,
            op=node.op,
            inputs=node.inputs,
            axes=node.axes,
            dtype=node.dtype,
            attributes=(
                ("contraction_axes", partition),
                ("implementation", partition.kind),
            ),
        )
    return TensorProgram(program.axes, tuple(nodes), program.outputs)


def plan_materializations(program: TensorProgram) -> TensorProgram:
    """Validate the prototype's conservative output-only store policy."""
    output_ids = {node_id for _, node_id in program.outputs}
    for node in program.nodes:
        if node.op == "materialize" and node.id not in output_ids:
            raise ValueError("internal materialization requires an execution plan")
    for name, node_id in program.outputs:
        node = program.node(node_id)
        if node.op != "materialize" or _attribute(node, "name") != name:
            raise ValueError(f"output {name!r} is not its materialization boundary")
    return program


def optimize_program(program: TensorProgram) -> TensorProgram:
    """Run the canonical tensor-IR passes in their required order."""
    program = simplify_constants(program)
    program = eliminate_dead_code(program)
    program = discover_contractions(program)
    program = canonicalize_contractions(program)
    return plan_materializations(program)


def program_to_region(program: TensorProgram) -> TensorRegion:
    """Remove a standalone program's storage boundary and expose SSA results."""
    result_ids: list[tuple[str, int]] = []
    removable: set[int] = set()
    for name, node_id in program.outputs:
        node = program.node(node_id)
        if node.op == "materialize":
            result_ids.append((name, node.inputs[0]))
            removable.add(node_id)
        else:
            result_ids.append((name, node_id))
    kept = set(range(len(program.nodes))) - removable
    symbolic = _renumber(
        TensorProgram(program.axes, program.nodes, tuple(result_ids)), kept
    )
    symbolic = simplify_constants(symbolic)
    symbolic = eliminate_dead_code(symbolic)
    symbolic = discover_contractions(symbolic)
    symbolic = canonicalize_contractions(symbolic)
    nodes = tuple(
        TensorNode(
            node.id,
            "argument" if node.op == "input" else node.op,
            node.inputs,
            node.axes,
            node.dtype,
            node.attributes,
        )
        for node in symbolic.nodes
    )
    symbolic = TensorProgram(symbolic.axes, nodes, symbolic.outputs)
    arguments = tuple(
        (str(_attribute(node, "name")), node.id)
        for node in symbolic.nodes
        if node.op == "argument"
    )
    return TensorRegion(symbolic.axes, symbolic.nodes, arguments, symbolic.outputs)


__all__ = [
    "AxisAffineIndex",
    "TileMaskComparison",
    "canonicalize_contractions",
    "discover_contractions",
    "discover_tile_mask",
    "eliminate_dead_code",
    "optimize_program",
    "plan_materializations",
    "program_to_region",
    "simplify_constants",
]
