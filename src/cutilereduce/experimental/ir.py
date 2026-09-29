"""A small, target-independent tensor IR with named logical axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractionAxes:
    """Axis partition used to select a contraction implementation."""

    batch: tuple[str, ...]
    lhs_free: tuple[str, ...]
    rhs_free: tuple[str, ...]
    reduce: tuple[str, ...]

    @property
    def kind(self) -> str:
        if not self.lhs_free and not self.rhs_free:
            return "batched_dot" if self.batch else "dot"
        if self.lhs_free and self.rhs_free:
            return "batched_gemm" if self.batch else "gemm"
        return "batched_gemv" if self.batch else "gemv"


@dataclass(frozen=True)
class TensorNode:
    id: int
    op: str
    inputs: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str
    attributes: tuple[tuple[str, Any], ...] = ()

    def attribute(self, name: str) -> Any:
        return dict(self.attributes)[name]

    @property
    def materialized(self) -> bool:
        return self.op in {"input", "materialize"}


@dataclass(frozen=True)
class TensorProgram:
    """A topologically ordered tensor program with explicit output stores."""

    axes: tuple[str, ...]
    nodes: tuple[TensorNode, ...]
    outputs: tuple[tuple[str, int], ...]

    def node(self, node_id: int) -> TensorNode:
        return self.nodes[node_id]

    def output_node(self, name: str) -> TensorNode:
        output_id = dict(self.outputs)[name]
        return self.node(output_id)

    @property
    def contractions(self) -> tuple[TensorNode, ...]:
        return tuple(node for node in self.nodes if node.op == "contract")

    @property
    def reductions(self) -> tuple[TensorNode, ...]:
        return tuple(node for node in self.nodes if node.op.startswith("reduce_"))

    @property
    def scans(self) -> tuple[TensorNode, ...]:
        return tuple(node for node in self.nodes if node.op.startswith("scan_"))

    @property
    def materializations(self) -> tuple[TensorNode, ...]:
        return tuple(node for node in self.nodes if node.materialized)

    def format(self) -> str:
        lines = []
        output_names = {node_id: name for name, node_id in self.outputs}
        for node in self.nodes:
            inputs = ", ".join(f"%{value}" for value in node.inputs)
            attrs = ", ".join(f"{key}={value!r}" for key, value in node.attributes)
            details = ", ".join(value for value in (inputs, attrs) if value)
            suffix = f" -> {output_names[node.id]}" if node.id in output_names else ""
            lines.append(
                f"%{node.id} = {node.op}({details}) : "
                f"tensor[{', '.join(node.axes)}; {node.dtype}]{suffix}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class TensorValue:
    builder: TensorBuilder
    id: int

    @property
    def node(self) -> TensorNode:
        return self.builder.nodes[self.id]

    @property
    def axes(self) -> tuple[str, ...]:
        return self.node.axes

    @property
    def dtype(self) -> str:
        return self.node.dtype

    def __add__(self, other: TensorValue | int | float) -> TensorValue:
        return self.builder.elementwise("add", self, other)

    def __sub__(self, other: TensorValue | int | float) -> TensorValue:
        return self.builder.elementwise("subtract", self, other)

    def __mul__(self, other: TensorValue | int | float) -> TensorValue:
        return self.builder.elementwise("multiply", self, other)

    def __truediv__(self, other: TensorValue | int | float) -> TensorValue:
        return self.builder.elementwise("divide", self, other)


class TensorBuilder:
    """Construct tensor expressions while retaining axis dependencies."""

    def __init__(self, axes: tuple[str, ...]):
        if len(set(axes)) != len(axes):
            raise ValueError(f"duplicate axes: {axes}")
        self.axes = axes
        self.nodes: list[TensorNode] = []
        self.outputs: list[tuple[str, int]] = []

    def _ordered(self, axes: set[str]) -> tuple[str, ...]:
        unknown = axes - set(self.axes)
        if unknown:
            raise ValueError(f"unknown axes: {sorted(unknown)}")
        return tuple(axis for axis in self.axes if axis in axes)

    def _add(
        self,
        op: str,
        inputs: tuple[TensorValue, ...],
        axes: tuple[str, ...],
        dtype: str,
        **attributes: Any,
    ) -> TensorValue:
        node = TensorNode(
            id=len(self.nodes),
            op=op,
            inputs=tuple(value.id for value in inputs),
            axes=axes,
            dtype=dtype,
            attributes=tuple(attributes.items()),
        )
        self.nodes.append(node)
        return TensorValue(self, node.id)

    def input(self, name: str, axes: tuple[str, ...], dtype: str) -> TensorValue:
        self._ordered(set(axes))
        return self._add("input", (), axes, dtype, name=name)

    def constant(self, value: int | float, dtype: str = "float32") -> TensorValue:
        return self._add("constant", (), (), dtype, value=value)

    def zero(self, axes: tuple[str, ...], dtype: str = "float32") -> TensorValue:
        """Construct a structural zero without implying tensor allocation."""
        return self._add("zero", (), self._ordered(set(axes)), dtype)

    def full(
        self, axes: tuple[str, ...], value: int | float, dtype: str = "float32"
    ) -> TensorValue:
        ordered_axes = self._ordered(set(axes))
        return self._add("full", (), ordered_axes, dtype, value=value)

    def _value(self, value: TensorValue | int | float, dtype: str) -> TensorValue:
        if isinstance(value, TensorValue):
            if value.builder is not self:
                raise ValueError("cannot mix values from different tensor programs")
            return value
        return self.constant(value, dtype)

    def elementwise(
        self,
        op: str,
        lhs: TensorValue,
        rhs: TensorValue | int | float | None = None,
    ) -> TensorValue:
        if lhs.builder is not self:
            raise ValueError("value belongs to another tensor program")
        values = (lhs,) if rhs is None else (lhs, self._value(rhs, lhs.dtype))
        axes = self._ordered(set().union(*(value.axes for value in values)))
        return self._add(op, values, axes, lhs.dtype)

    def exp(self, value: TensorValue) -> TensorValue:
        return self.elementwise("exp", value)

    def log(self, value: TensorValue) -> TensorValue:
        return self.elementwise("log", value)

    def compare(
        self, op: str, lhs: TensorValue, rhs: TensorValue | int | float
    ) -> TensorValue:
        right = self._value(rhs, lhs.dtype)
        axes = self._ordered(set(lhs.axes) | set(right.axes))
        return self._add(f"compare_{op}", (lhs, right), axes, "bool")

    def where(
        self,
        condition: TensorValue,
        when_true: TensorValue,
        when_false: TensorValue,
    ) -> TensorValue:
        if condition.dtype != "bool":
            raise ValueError("where condition must have bool dtype")
        axes = self._ordered(
            set(condition.axes) | set(when_true.axes) | set(when_false.axes)
        )
        return self._add(
            "where", (condition, when_true, when_false), axes, when_true.dtype
        )

    def cast(self, value: TensorValue, dtype: str) -> TensorValue:
        if value.dtype == dtype:
            return value
        return self._add("cast", (value,), value.axes, dtype)

    def reduce(self, op: str, value: TensorValue, over: tuple[str, ...]) -> TensorValue:
        missing = set(over) - set(value.axes)
        if missing:
            raise ValueError(f"cannot reduce absent axes: {sorted(missing)}")
        axes = tuple(axis for axis in value.axes if axis not in set(over))
        return self._add(f"reduce_{op}", (value,), axes, value.dtype, over=over)

    def scan(
        self,
        op: str,
        value: TensorValue,
        *,
        over: str,
        exclusive: bool = False,
        reverse: bool = False,
    ) -> TensorValue:
        if over not in value.axes:
            raise ValueError(f"cannot scan absent axis: {over!r}")
        return self._add(
            f"scan_{op}",
            (value,),
            value.axes,
            value.dtype,
            over=over,
            exclusive=exclusive,
            reverse=reverse,
        )

    def contract(
        self,
        lhs: TensorValue,
        rhs: TensorValue,
        *,
        over: tuple[str, ...],
        dtype: str | None = None,
    ) -> TensorValue:
        missing_lhs = set(over) - set(lhs.axes)
        missing_rhs = set(over) - set(rhs.axes)
        if missing_lhs or missing_rhs:
            raise ValueError(
                "contraction axes must occur in both operands: "
                f"lhs missing {sorted(missing_lhs)}, rhs missing {sorted(missing_rhs)}"
            )
        reduced = set(over)
        lhs_axes = set(lhs.axes) - reduced
        rhs_axes = set(rhs.axes) - reduced
        partition = ContractionAxes(
            batch=self._ordered(lhs_axes & rhs_axes),
            lhs_free=self._ordered(lhs_axes - rhs_axes),
            rhs_free=self._ordered(rhs_axes - lhs_axes),
            reduce=self._ordered(reduced),
        )
        output_axes = self._ordered(lhs_axes | rhs_axes)
        return self._add(
            "contract",
            (lhs, rhs),
            output_axes,
            dtype or lhs.dtype,
            contraction_axes=partition,
            implementation=partition.kind,
        )

    def materialize(self, name: str, value: TensorValue) -> TensorValue:
        if name in dict(self.outputs):
            raise ValueError(f"duplicate output name: {name}")
        result = self._add("materialize", (value,), value.axes, value.dtype, name=name)
        self.outputs.append((name, result.id))
        return result

    def finish(self) -> TensorProgram:
        return TensorProgram(self.axes, tuple(self.nodes), tuple(self.outputs))


__all__ = [
    "ContractionAxes",
    "TensorBuilder",
    "TensorNode",
    "TensorProgram",
    "TensorValue",
]
