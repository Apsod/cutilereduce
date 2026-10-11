"""A small, target-independent tensor IR with named logical axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from cutilereduce.experimental.ir.ops import (
    COMPARISONS,
    REDUCTIONS,
    SCANS,
    ELEMENTWISE_OPS,
    validate_elementwise,
)


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
        return tuple(
            node
            for node in self.nodes
            if node.op in {"contract", "contract_accumulate"}
        )

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
        regions: list[TensorProgram] = []
        output_names = {node_id: name for name, node_id in self.outputs}
        for node in self.nodes:
            inputs = ", ".join(f"%{value}" for value in node.inputs)
            formatted_attrs = []
            for key, value in node.attributes:
                if isinstance(value, TensorProgram):
                    if value not in regions:
                        regions.append(value)
                    formatted_attrs.append(f"{key}=@region{regions.index(value)}")
                else:
                    formatted_attrs.append(f"{key}={value!r}")
            attrs = ", ".join(formatted_attrs)
            details = ", ".join(value for value in (inputs, attrs) if value)
            suffix = f" -> {output_names[node.id]}" if node.id in output_names else ""
            lines.append(
                f"%{node.id} = {node.op}({details}) : "
                f"tensor[{', '.join(node.axes)}; {node.dtype}]{suffix}"
            )
        for index, region in enumerate(regions):
            lines.append(f"region @region{index} {{")
            lines.extend(f"  {line}" for line in region.format().splitlines())
            lines.append("}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TensorSignature:
    """The symbolic type of one tensor region argument or result."""

    axes: tuple[str, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class TensorRegion:
    """A closed symbolic computation with formal arguments and SSA yields.

    Unlike a :class:`TensorProgram`, a region does not imply loads or stores at
    its boundary.  Planning decides where its formal arguments live and where
    its yielded values are materialized.
    """

    axes: tuple[str, ...]
    nodes: tuple[TensorNode, ...]
    arguments: tuple[tuple[str, int], ...]
    results: tuple[tuple[str, int], ...]

    @property
    def outputs(self) -> tuple[tuple[str, int], ...]:
        """Compatibility alias for code inspecting named results."""
        return self.results

    @property
    def materializations(self) -> tuple[TensorNode, ...]:
        return tuple(node for node in self.nodes if node.op == "materialize")

    def as_program(self) -> TensorProgram:
        """View the region as IR without introducing materialization nodes."""
        return TensorProgram(self.axes, self.nodes, self.results)

    def format(self) -> str:
        argument_ids = {node_id for _, node_id in self.arguments}
        argument_lines = []
        for name, node_id in self.arguments:
            node = self.nodes[node_id]
            argument_lines.append(
                f"  %{node_id} {name}: tensor[{', '.join(node.axes)}; {node.dtype}]"
            )

        lines = ["region("]
        lines.extend(
            f"{line}{',' if index + 1 < len(argument_lines) else ''}"
            for index, line in enumerate(argument_lines)
        )
        lines.append(") {")

        nested_regions: list[TensorProgram] = []
        for node in self.nodes:
            if node.id in argument_ids:
                continue
            inputs = ", ".join(f"%{value}" for value in node.inputs)
            formatted_attrs = []
            for key, value in node.attributes:
                if isinstance(value, TensorProgram):
                    if value not in nested_regions:
                        nested_regions.append(value)
                    formatted_attrs.append(
                        f"{key}=@region{nested_regions.index(value)}"
                    )
                else:
                    formatted_attrs.append(f"{key}={value!r}")
            attrs = ", ".join(formatted_attrs)
            details = ", ".join(value for value in (inputs, attrs) if value)
            lines.append(
                f"  %{node.id} = {node.op}({details}) : "
                f"tensor[{', '.join(node.axes)}; {node.dtype}]"
            )

        yielded = ", ".join(f"{name}=%{node_id}" for name, node_id in self.results)
        lines.append(f"  yield {yielded}")
        for index, region in enumerate(nested_regions):
            lines.append(f"  region @region{index} {{")
            lines.extend(f"    {line}" for line in region.format().splitlines())
            lines.append("  }")
        lines.append("}")
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

    def __neg__(self) -> TensorValue:
        return self.builder.neg(self)

    def __radd__(self, other: int | float) -> TensorValue:
        return self + other

    def __rsub__(self, other: int | float) -> TensorValue:
        return self.builder.elementwise(
            "subtract", self.builder._value(other, self.dtype), self
        )

    def __rmul__(self, other: int | float) -> TensorValue:
        return self * other

    def __rtruediv__(self, other: int | float) -> TensorValue:
        return self.builder.elementwise(
            "divide", self.builder._value(other, self.dtype), self
        )

    def __pow__(self, exponent: int) -> TensorValue:
        return self.builder.powi(self, exponent)

    def __abs__(self) -> TensorValue:
        return self.builder.abs(self)

    def __and__(self, other: TensorValue | bool) -> TensorValue:
        return self.builder.logical_and(self, other)

    __rand__ = __and__

    def __or__(self, other: TensorValue | bool) -> TensorValue:
        return self.builder.logical_or(self, other)

    __ror__ = __or__

    def __invert__(self) -> TensorValue:
        return self.builder.logical_not(self)

    def __floordiv__(self, other: TensorValue | int) -> TensorValue:
        return self.builder.floor_divide(self, other)

    def __rfloordiv__(self, other: int) -> TensorValue:
        if type(other) is not int:
            raise TypeError("integer floor division requires an integer scalar")
        return self.builder.floor_divide(self.builder._value(other, self.dtype), self)

    def __mod__(self, other: TensorValue | int) -> TensorValue:
        return self.builder.remainder(self, other)

    def __rmod__(self, other: int) -> TensorValue:
        if type(other) is not int:
            raise TypeError("integer remainder requires an integer scalar")
        return self.builder.remainder(self.builder._value(other, self.dtype), self)

    def __bool__(self) -> bool:
        raise TypeError(
            "symbolic tensors have no Python truth value; use ctx.where and logical operations"
        )


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

    def axis_index(self, axis: str, dtype: str = "int64") -> TensorValue:
        """Return the logical global coordinate along one named axis."""
        if axis not in self.axes:
            raise ValueError(f"unknown axis: {axis!r}")
        if dtype not in {"int32", "int64"}:
            raise ValueError("axis indices require int32 or int64 dtype")
        return self._add("axis_index", (), (axis,), dtype, axis=axis)

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
        if op not in ELEMENTWISE_OPS:
            raise ValueError(f"unknown elementwise operation: {op!r}")
        category = ELEMENTWISE_OPS[op].category
        if rhs is not None and not isinstance(rhs, TensorValue):
            if category == "boolean" and type(rhs) is not bool:
                raise ValueError(f"{op} requires a boolean literal")
            if category == "integer" and type(rhs) is not int:
                raise ValueError(f"{op} requires an integer literal")
        values = (lhs,) if rhs is None else (lhs, self._value(rhs, lhs.dtype))
        validate_elementwise(op, tuple(value.dtype for value in values))
        if op in {"powi", "clamp"}:
            raise ValueError(f"use ctx.{op} to supply operation attributes")
        axes = self._ordered(set().union(*(value.axes for value in values)))
        return self._add(op, values, axes, lhs.dtype)

    def exp(self, value: TensorValue) -> TensorValue:
        return self.elementwise("exp", value)

    def log(self, value: TensorValue) -> TensorValue:
        return self.elementwise("log", value)

    def neg(self, value: TensorValue) -> TensorValue:
        return self.elementwise("neg", value)

    def sqrt(self, value: TensorValue) -> TensorValue:
        return self.elementwise("sqrt", value)

    def rsqrt(self, value: TensorValue) -> TensorValue:
        return self.elementwise("rsqrt", value)

    def abs(self, value: TensorValue) -> TensorValue:
        return self.elementwise("abs", value)

    def sin(self, value: TensorValue) -> TensorValue:
        return self.elementwise("sin", value)

    def cos(self, value: TensorValue) -> TensorValue:
        return self.elementwise("cos", value)

    def tanh(self, value: TensorValue) -> TensorValue:
        return self.elementwise("tanh", value)

    def sigmoid(self, value: TensorValue) -> TensorValue:
        return self.elementwise("sigmoid", value)

    def log1p(self, value: TensorValue) -> TensorValue:
        return self.elementwise("log1p", value)

    def expm1(self, value: TensorValue) -> TensorValue:
        return self.elementwise("expm1", value)

    def stop_gradient(self, value: TensorValue) -> TensorValue:
        return self.elementwise("stop_gradient", value)

    def minimum(self, lhs: TensorValue, rhs: TensorValue | int | float) -> TensorValue:
        return self.elementwise("minimum", lhs, rhs)

    def maximum(self, lhs: TensorValue, rhs: TensorValue | int | float) -> TensorValue:
        return self.elementwise("maximum", lhs, rhs)

    def powi(self, value: TensorValue, exponent: int) -> TensorValue:
        self._value(value, value.dtype)
        validate_elementwise("powi", (value.dtype,))
        if type(exponent) is not int:
            raise ValueError("powi requires a Python integer exponent")
        return self._add(
            "powi",
            (value,),
            self._ordered(set(value.axes)),
            value.dtype,
            exponent=exponent,
        )

    def clamp(
        self,
        value: TensorValue,
        *,
        min: int | float | None = None,
        max: int | float | None = None,
    ) -> TensorValue:
        self._value(value, value.dtype)
        validate_elementwise("clamp", (value.dtype,))
        if min is None and max is None:
            raise ValueError("clamp requires at least one bound")
        for bound in (min, max):
            if bound is not None:
                if type(bound) not in {int, float} or math.isnan(bound):
                    raise ValueError("clamp bounds must be non-NaN numeric constants")
                if value.dtype.startswith("int") and type(bound) is not int:
                    raise ValueError("integer clamp requires integer bounds")
        if min is not None and max is not None and min > max:
            raise ValueError("clamp minimum must not exceed maximum")
        return self._add(
            "clamp",
            (value,),
            self._ordered(set(value.axes)),
            value.dtype,
            min=min,
            max=max,
        )

    def logical_and(self, lhs: TensorValue, rhs: TensorValue | bool) -> TensorValue:
        return self.elementwise("logical_and", lhs, rhs)

    def logical_or(self, lhs: TensorValue, rhs: TensorValue | bool) -> TensorValue:
        return self.elementwise("logical_or", lhs, rhs)

    def logical_not(self, value: TensorValue) -> TensorValue:
        return self.elementwise("logical_not", value)

    def floor_divide(self, lhs: TensorValue, rhs: TensorValue | int) -> TensorValue:
        return self.elementwise("floor_divide", lhs, rhs)

    def remainder(self, lhs: TensorValue, rhs: TensorValue | int) -> TensorValue:
        return self.elementwise("remainder", lhs, rhs)

    def softplus(self, value: TensorValue) -> TensorValue:
        return self.maximum(value, 0.0) + self.log1p(self.exp(-self.abs(value)))

    def logaddexp(
        self, lhs: TensorValue, rhs: TensorValue | int | float
    ) -> TensorValue:
        right = self._value(rhs, lhs.dtype)
        equal = self.compare("equal", lhs, right)
        difference = self.where(equal, self.constant(0.0, lhs.dtype), lhs - right)
        return self.maximum(lhs, right) + self.log1p(self.exp(-self.abs(difference)))

    def compare(
        self, op: str, lhs: TensorValue, rhs: TensorValue | int | float
    ) -> TensorValue:
        if op not in COMPARISONS:
            raise ValueError(f"unknown comparison: {op!r}")
        self._value(lhs, lhs.dtype)
        right = self._value(rhs, lhs.dtype)
        if lhs.dtype != right.dtype:
            raise ValueError(
                "comparison operands require matching dtypes; use an explicit cast"
            )
        axes = self._ordered(set(lhs.axes) | set(right.axes))
        return self._add(f"compare_{op}", (lhs, right), axes, "bool")

    def where(
        self,
        condition: TensorValue,
        when_true: TensorValue,
        when_false: TensorValue,
    ) -> TensorValue:
        for value in (condition, when_true, when_false):
            self._value(value, value.dtype)
        if when_true.dtype != when_false.dtype:
            raise ValueError(
                "where branches require matching dtypes; use an explicit cast"
            )
        if condition.dtype != "bool":
            raise ValueError("where condition must have bool dtype")
        axes = self._ordered(
            set(condition.axes) | set(when_true.axes) | set(when_false.axes)
        )
        return self._add(
            "where", (condition, when_true, when_false), axes, when_true.dtype
        )

    def mask_identity(
        self,
        condition: TensorValue,
        value: TensorValue,
        identity: TensorValue,
        *,
        mask_group: int,
    ) -> TensorValue:
        """Select a mapped value or a synthetic algebraic identity."""
        if condition.dtype != "bool":
            raise ValueError("identity mask condition must have bool dtype")
        axes = self._ordered(set(condition.axes) | set(value.axes) | set(identity.axes))
        return self._add(
            "mask_identity",
            (condition, value, identity),
            axes,
            value.dtype,
            mask_group=mask_group,
        )

    def domain_valid(self, axes: tuple[str, ...]) -> TensorValue:
        """Reference the active schedule's physical tile domain, not a tensor input."""
        unknown = set(axes) - set(self.axes)
        if unknown:
            raise ValueError(f"domain axes are not in this program: {sorted(unknown)}")
        return self._add("domain_valid", (), axes, "bool")

    def cast(self, value: TensorValue, dtype: str) -> TensorValue:
        self._value(value, value.dtype)
        if value.dtype == dtype:
            return value
        return self._add("cast", (value,), value.axes, dtype)

    def reduce(self, op: str, value: TensorValue, over: tuple[str, ...]) -> TensorValue:
        if op not in REDUCTIONS:
            raise ValueError(f"unsupported reduction: {op!r}")
        self._value(value, value.dtype)
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
        if op not in SCANS:
            raise ValueError(f"unsupported scan: {op!r}")
        self._value(value, value.dtype)
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

    def associative_result(
        self,
        op: str,
        inputs: tuple[TensorValue, ...],
        *,
        axes: tuple[str, ...],
        dtype: str,
        region: TensorProgram,
        result_path: tuple[str, ...],
        over: tuple[str, ...],
    ) -> TensorValue:
        """Add one result of a structured associative fold or scan region."""
        return self._add(
            op,
            inputs,
            axes,
            dtype,
            over=over,
            result_path=result_path,
            combine_region=region,
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

    def contract_accumulate(
        self,
        accumulator: TensorValue,
        lhs: TensorValue,
        rhs: TensorValue,
        *,
        over: tuple[str, ...],
        dtype: str | None = None,
    ) -> TensorValue:
        """Contract two tensors directly into an existing output accumulator."""
        result = self.contract(lhs, rhs, over=over, dtype=dtype)
        if accumulator.axes != result.axes:
            raise ValueError(
                "contraction accumulator axes must match the contraction result: "
                f"{accumulator.axes} != {result.axes}"
            )
        if accumulator.dtype != result.dtype:
            raise ValueError(
                "contraction accumulator dtype must match the result: "
                f"{accumulator.dtype} != {result.dtype}"
            )
        node = self.nodes[result.id]
        self.nodes[result.id] = TensorNode(
            node.id,
            "contract_accumulate",
            (lhs.id, rhs.id, accumulator.id),
            node.axes,
            node.dtype,
            node.attributes,
        )
        return result

    def materialize(self, name: str, value: TensorValue) -> TensorValue:
        if name in dict(self.outputs):
            raise ValueError(f"duplicate output name: {name}")
        result = self._add("materialize", (value,), value.axes, value.dtype, name=name)
        self.outputs.append((name, result.id))
        return result

    def finish(self) -> TensorProgram:
        return TensorProgram(self.axes, tuple(self.nodes), tuple(self.outputs))


class TensorRegionBuilder(TensorBuilder):
    """Symbolically construct a region from formal tensor arguments."""

    def __init__(self, axes: tuple[str, ...]):
        super().__init__(axes)
        self.arguments: list[tuple[str, int]] = []

    def expect(
        self, name: str, signature: TensorSignature, *, label: str | None = None
    ) -> TensorValue:
        """Declare a formal argument; no concrete tensor is inspected."""
        if name in dict(self.arguments):
            raise ValueError(f"duplicate argument name: {name}")
        value = self._add(
            "argument",
            (),
            self._ordered(set(signature.axes)),
            signature.dtype,
            name=name,
            label=label or name,
        )
        self.arguments.append((name, value.id))
        return value

    def yield_value(self, name: str, value: TensorValue) -> None:
        """Declare an SSA result without adding a storage boundary."""
        if name in dict(self.outputs):
            raise ValueError(f"duplicate result name: {name}")
        if value.builder is not self:
            raise ValueError("region result belongs to another builder")
        self.outputs.append((name, value.id))

    def finish_region(self) -> TensorRegion:
        return TensorRegion(
            self.axes,
            tuple(self.nodes),
            tuple(self.arguments),
            tuple(self.outputs),
        )


__all__ = [
    "ContractionAxes",
    "TensorBuilder",
    "TensorNode",
    "TensorProgram",
    "TensorRegion",
    "TensorRegionBuilder",
    "TensorSignature",
    "TensorValue",
]
