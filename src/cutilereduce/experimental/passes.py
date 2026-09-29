"""Canonical optimization pipeline for the experimental tensor IR."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .ir import ContractionAxes, TensorNode, TensorProgram


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
            elif old.op == "multiply":
                if nodes[left].op == "zero" or nodes[right].op == "zero":
                    zero_result = True
                elif right_value == 1 and nodes[left].axes == old.axes:
                    alias = left
                elif left_value == 1 and nodes[right].axes == old.axes:
                    alias = right
            elif old.op == "divide":
                if right_value == 1 and nodes[left].axes == old.axes:
                    alias = left

        if alias is not None:
            remap[old.id] = alias
            continue
        if old.op.startswith("reduce_") and nodes[inputs[0]].op == "zero":
            zero_result = True
        if old.op == "contract" and any(
            nodes[value].op == "zero" for value in inputs
        ):
            zero_result = True
        new_id = len(nodes)
        remap[old.id] = new_id
        if zero_result:
            nodes.append(
                TensorNode(new_id, "zero", (), old.axes, old.dtype)
            )
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
    """Recognize sum-of-products as named-axis contractions."""
    nodes = list(program.nodes)
    uses = Counter(value for node in nodes for value in node.inputs)
    for index, node in enumerate(nodes):
        if node.op != "reduce_sum" or len(node.inputs) != 1:
            continue
        product = nodes[node.inputs[0]]
        if product.op != "multiply" or len(product.inputs) != 2:
            continue
        lhs = nodes[product.inputs[0]]
        rhs = nodes[product.inputs[1]]
        reduce = _attribute(node, "over")
        if not isinstance(reduce, tuple):
            continue
        if not set(reduce) <= set(lhs.axes) & set(rhs.axes):
            continue
        partition = _contraction_axes(program.axes, lhs, rhs, reduce)
        nodes[index] = TensorNode(
            id=node.id,
            op="contract",
            inputs=product.inputs,
            axes=node.axes,
            dtype=node.dtype,
            attributes=(
                ("contraction_axes", partition),
                ("implementation", partition.kind),
            ),
        )

    rewritten = TensorProgram(program.axes, tuple(nodes), program.outputs)
    # A recognized multiply commonly becomes orphaned. Compacting it here is
    # part of constructing the replacement, not a second global DCE policy.
    if any(uses[node.id] == 1 for node in nodes if node.op == "multiply"):
        return eliminate_dead_code(rewritten)
    return rewritten


def canonicalize_contractions(program: TensorProgram) -> TensorProgram:
    """Recompute normalized named-axis partitions and implementation hints."""
    nodes = list(program.nodes)
    for index, node in enumerate(nodes):
        if node.op != "contract":
            continue
        old_partition = _attribute(node, "contraction_axes")
        if not isinstance(old_partition, ContractionAxes):
            raise TypeError("contract node has invalid axis metadata")
        lhs = nodes[node.inputs[0]]
        rhs = nodes[node.inputs[1]]
        partition = _contraction_axes(
            program.axes, lhs, rhs, old_partition.reduce
        )
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


__all__ = [
    "canonicalize_contractions",
    "discover_contractions",
    "eliminate_dead_code",
    "optimize_program",
    "plan_materializations",
    "simplify_constants",
]
