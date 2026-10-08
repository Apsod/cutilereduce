"""Contracts for the elementary tensor DSL, independent of execution targets."""

from dataclasses import dataclass
from typing import Literal


FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
INTEGER_DTYPES = frozenset({"int32", "int64"})
NUMERIC_DTYPES = FLOAT_DTYPES | INTEGER_DTYPES


@dataclass(frozen=True)
class ElementwiseOp:
    arity: int
    category: Literal["float", "numeric", "integer", "boolean", "any"]
    differentiation: Literal["rule", "none", "unsupported"] = "rule"


ELEMENTWISE_OPS = {
    **{
        name: ElementwiseOp(2, "numeric")
        for name in ("add", "subtract", "multiply", "minimum", "maximum")
    },
    "divide": ElementwiseOp(2, "float"),
    **{name: ElementwiseOp(1, "numeric") for name in ("neg", "abs", "clamp")},
    **{
        name: ElementwiseOp(1, "float")
        for name in (
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "powi",
            "sin",
            "cos",
            "tanh",
            "sigmoid",
            "log1p",
            "expm1",
        )
    },
    "stop_gradient": ElementwiseOp(1, "any", "none"),
    "logical_and": ElementwiseOp(2, "boolean", "none"),
    "logical_or": ElementwiseOp(2, "boolean", "none"),
    "logical_not": ElementwiseOp(1, "boolean", "none"),
    "floor_divide": ElementwiseOp(2, "integer", "none"),
    "remainder": ElementwiseOp(2, "integer", "none"),
}

COMPARISONS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    }
)
REDUCTIONS = frozenset({"sum", "max"})
SCANS = frozenset({"sum"})


def validate_elementwise(op: str, dtypes: tuple[str, ...]) -> ElementwiseOp:
    if op not in ELEMENTWISE_OPS:
        raise ValueError(f"unknown elementwise operation: {op!r}")
    contract = ELEMENTWISE_OPS[op]
    if len(dtypes) != contract.arity:
        raise ValueError(f"{op} requires {contract.arity} operand(s)")
    accepted = {
        "float": FLOAT_DTYPES,
        "numeric": NUMERIC_DTYPES,
        "integer": INTEGER_DTYPES,
        "boolean": frozenset({"bool"}),
        "any": NUMERIC_DTYPES | {"bool"},
    }[contract.category]
    if any(dtype not in accepted for dtype in dtypes):
        raise ValueError(f"{op} requires {contract.category} operands, got {dtypes}")
    if len(set(dtypes)) != 1:
        raise ValueError(f"{op} requires matching dtypes; use an explicit cast")
    return contract


def multiplication_chain(exponent: int) -> tuple[tuple[int, int], ...]:
    """Addition chain for a positive integer power; index 0 is the base.

    Each pair multiplies two previous values and appends its result. The final
    value is the power, with logarithmic chain size.
    """
    if exponent < 1:
        raise ValueError("multiplication chain requires a positive exponent")
    steps: list[tuple[int, int]] = []
    current = 0
    for bit in bin(exponent)[3:]:
        steps.append((current, current))
        current = len(steps)
        if bit == "1":
            steps.append((current, 0))
            current = len(steps)
    return tuple(steps)
