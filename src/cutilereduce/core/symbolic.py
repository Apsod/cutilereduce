"""Small analytical layer over concrete nonnegative polynomials."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np

from .poly import Polynomial, mono


@dataclass(frozen=True)
class Variable:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple[Any, ...]

    def __str__(self):
        if self.op in {"min", "max"}:
            return f"{self.op}({', '.join(map(str, self.args))})"
        if self.op == "if":
            return f"if {self.args[0]} then {self.args[1]} else {self.args[2]}"
        return f"{self.op}({', '.join(map(str, self.args))})"


def _raw(value):
    if isinstance(value, Expr):
        return value.value
    if isinstance(value, str):
        return Variable(value)
    return value


def _expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float, Fraction, np.number)):
        return value
    if isinstance(value, Polynomial) and not value.variables:
        return value.monomials[0].coefficient if value.monomials else 0
    if isinstance(value, Node) and not _names(value):
        return _evaluate(value, {})
    return Expr(value)


def _poly(value):
    if isinstance(value, Polynomial):
        return value
    if isinstance(value, (int, Fraction)):
        return Polynomial.constant(value)
    if isinstance(value, Variable):
        return mono(1, **{value.name: 1})
    return None


def _combine(op, *values):
    values = tuple(_raw(v) for v in values)
    polys = tuple(_poly(v) for v in values)
    if all(p is not None for p in polys):
        if op == "add":
            result = Polynomial.zero()
            for p in polys:
                result = result + p
            return result
        if op == "mul":
            result = Polynomial.constant(1)
            for p in polys:
                result = result * p
            return result
    # Addition and multiplication are monotone over nonnegative measures.
    # Distribute across extrema so the analytical tree has a canonical form.
    extrema = next((v for v in values if isinstance(v, Node) and v.op in {"min", "max"}), None)
    if extrema is not None and all(
        _poly(v) is not None or v is extrema for v in values
    ):
        others = [v for v in values if v is not extrema]
        branches = []
        for child in extrema.args:
            branches.append(_combine(op, *(others + [child])))
        return Node(extrema.op, tuple(_raw(branch) for branch in branches))
    return Node(op, values)


@dataclass(frozen=True)
class Expr:
    value: Any

    def __add__(self, other):
        return _expr(_combine("add", self, other))

    __radd__ = __add__

    def __sub__(self, other):
        return SaturateSub(self, other)

    def __rsub__(self, other):
        return SaturateSub(other, self)

    def __mul__(self, other):
        return _expr(_combine("mul", self, other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        left, right = _raw(self), _raw(other)
        lp, rp = _poly(left), _poly(right)
        if lp is not None and rp is not None:
            return _expr(lp / rp)
        return _expr(Node("div", (left, right)))

    def __rtruediv__(self, other):
        left, right = _raw(other), _raw(self)
        lp, rp = _poly(left), _poly(right)
        if lp is not None and rp is not None:
            return _expr(lp / rp)
        return _expr(Node("div", (left, right)))

    def __floordiv__(self, other):
        return _expr(Node("floor_div", (_raw(self), _raw(other))))

    def __pow__(self, other):
        base = _poly(_raw(self))
        if base is not None and isinstance(other, int) and other >= 0:
            result = Polynomial.constant(1)
            for _ in range(other):
                result = result * base
            return _expr(result)
        return _expr(Node("pow", (_raw(self), _raw(other))))

    def __le__(self, other):
        return _expr(Node("<=", (_raw(self), _raw(other))))

    def __lt__(self, other):
        return _expr(Node("<", (_raw(self), _raw(other))))

    def __ge__(self, other):
        return _expr(Node(">=", (_raw(self), _raw(other))))

    def __gt__(self, other):
        return _expr(Node(">", (_raw(self), _raw(other))))

    @property
    def free_symbols(self):
        return frozenset(Expr(Variable(n)) for n in _names(self.value))

    @property
    def name(self):
        if isinstance(self.value, Variable):
            return self.value.name
        raise AttributeError("only variables have names")

    def subs(self, values: Mapping):
        values = {
            (_raw(k).name if isinstance(_raw(k), Variable) else str(k)): _raw(v)
            for k, v in values.items()
        }
        result = _substitute(self.value, values)
        if not _names(result):
            return _evaluate(result, {})
        return _expr(result)

    def replace(self, old, new):
        target = _raw(old)
        return _expr(_replace(self.value, target, _raw(new)))

    def has(self, value):
        return _raw(value) in _walk(self.value)

    def cse(self, *, min_uses=2, min_complexity=2, prefix="_cse"):
        return cse(
            self,
            min_uses=min_uses,
            min_complexity=min_complexity,
            prefix=prefix,
        )

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)


def _walk(value):
    value = _raw(value)
    if isinstance(value, Variable):
        return {value}
    if isinstance(value, Polynomial):
        return {Variable(n) for n in value.variables}
    if isinstance(value, Node):
        result = set()
        for arg in value.args:
            result |= _walk(arg)
        return result
    return set()


def _names(value):
    return {v.name for v in _walk(value)}


def _substitute(value, values):
    value = _raw(value)
    if isinstance(value, Variable):
        return values.get(value.name, value)
    if isinstance(value, Polynomial):
        # A logical extent can be bound to a tile expression such as min(N,T).
        # Polynomial.subs handles constants/renaming; composite replacements
        # require rebuilding the expression rather than silently retaining N.
        composite = any(
            name in values and not isinstance(values[name], (int, float, Fraction, Variable, np.number))
            for name in value.variables
        )
        if not composite:
            return value.subs(values)
        terms = []
        for monomial in value.monomials:
            factors = [monomial.coefficient]
            for name, exponent in monomial.exponents.mapping:
                replacement = values.get(name, Variable(name))
                if exponent == 1:
                    factors.append(replacement)
                else:
                    factors.append(Node("pow", (replacement, exponent)))
            terms.append(_combine("mul", *factors))
        return _combine("add", *terms) if terms else Polynomial.zero()
    if isinstance(value, Node):
        return Node(value.op, tuple(_substitute(a, values) for a in value.args))
    return value


def _replace(value, old, new):
    value = _raw(value)
    if value == old:
        return new
    if isinstance(value, Polynomial) and isinstance(old, Variable):
        if isinstance(new, Variable) and old.name in value.variables:
            from .poly import Exponents, Monomial

            monomials = []
            for monomial in value.monomials:
                powers = dict(monomial.exponents.mapping)
                exponent = powers.pop(old.name, 0)
                if exponent:
                    powers[new.name] = powers.get(new.name, 0) + exponent
                monomials.append(Monomial.make(monomial.coefficient, Exponents.from_dict(powers)))
            return Polynomial.from_monomials(monomials)
        return value
    if isinstance(value, Node):
        return Node(value.op, tuple(_replace(a, old, new) for a in value.args))
    return value


def Symbol(name: str, **_kwargs):
    return Expr(Variable(name))


def sympify(value):
    return _expr(_raw(value))


def Min(*args):
    return (
        min(args)
        if all(isinstance(a, (int, float)) for a in args)
        else _expr(Node("min", tuple(_raw(a) for a in args)))
    )


def Max(*args):
    return (
        max(args)
        if all(isinstance(a, (int, float)) for a in args)
        else _expr(Node("max", tuple(_raw(a) for a in args)))
    )


def SaturateSub(left, right):
    return _expr(Node("saturate_sub", (_raw(left), _raw(right))))


def Reciprocal(value):
    return _expr(1) / _expr(value)


def floor(value):
    return _expr(Node("floor", (_raw(value),)))


def ceiling(value):
    return _expr(Node("ceil", (_raw(value),)))


def log(value, base=math.e):
    return _expr(Node("log", (_raw(value), _raw(base))))


class _Piecewise:
    def __call__(self, *pieces):
        if isinstance(pieces[0][1], (bool, np.bool_)):
            for value, condition in pieces:
                if condition:
                    return _expr(value)
        if len(pieces) == 2:
            value, condition = pieces[0]
            other, fallback = pieces[1]
            condition = _raw(condition)
            value_raw, other_raw = _raw(value), _raw(other)
            if (
                isinstance(condition, Node)
                and condition.op in {">=", "<="}
                and fallback is True
                and value_raw in condition.args
                and other_raw in condition.args
            ):
                if condition.op == ">=":
                    return Min(other_raw, value_raw)
                return Max(other_raw, value_raw)
        result = _raw(pieces[-1][0])
        for value, condition in reversed(pieces[:-1]):
            result = Node("if", (_raw(condition), _raw(value), result))
        return _expr(result)


Piecewise = _Piecewise()


def _evaluate(value, values):
    value = _raw(value)
    if isinstance(value, Polynomial):
        result = 0
        for m in value.monomials:
            term = m.coefficient
            for name, exponent in m.exponents.mapping:
                term *= np.asarray(values[name]) ** float(exponent)
            result += term
        return result
    if isinstance(value, Variable):
        return values[value.name]
    if isinstance(value, (int, float, Fraction, np.number)):
        return value
    if isinstance(value, Node):
        args = [_evaluate(a, values) for a in value.args]
        if value.op == "add":
            return sum(args)
        if value.op == "mul":
            return math.prod(args)
        if value.op == "sub":
            return args[0] - args[1]
        if value.op == "saturate_sub":
            return np.maximum(0, args[0] - args[1])
        if value.op == "div":
            return args[0] / args[1]
        if value.op == "floor_div":
            return np.floor(args[0] / args[1]).astype(int)
        if value.op == "pow":
            return np.power(args[0], args[1])
        if value.op == "min":
            result = args[0]
            for arg in args[1:]:
                result = np.minimum(result, arg)
            return result
        if value.op == "max":
            result = args[0]
            for arg in args[1:]:
                result = np.maximum(result, arg)
            return result
        if value.op == "if":
            return np.where(args[0], args[1], args[2])
        if value.op in {"<", "<=", ">", ">=", "=="}:
            return {
                "<": args[0] < args[1],
                "<=": args[0] <= args[1],
                ">": args[0] > args[1],
                ">=": args[0] >= args[1],
                "==": args[0] == args[1],
            }[value.op]
        if value.op in {"floor", "ceil", "log"}:
            return {
                "floor": np.floor,
                "ceil": np.ceil,
                "log": lambda x, b: np.log(x) / np.log(b),
            }[value.op](*args)
        raise ValueError(value.op)
    return value


def evaluate(expression, values):
    return _evaluate(expression, values)


def evaluate_batch(expression, values):
    """Evaluate an expression on a row-wise batch using NumPy broadcasting."""
    normalized = {
        (_raw(key).name if isinstance(_raw(key), Variable) else str(key).removeprefix("cfg:")): value
        for key, value in values.items()
    }
    arrays = {
        name: np.asarray(value)
        for name, value in normalized.items()
    }
    # Broadcast once up front so mismatched row batches fail clearly.
    if arrays:
        np.broadcast_arrays(*arrays.values())
    return _evaluate(expression, arrays)


@dataclass(frozen=True)
class CSEProgram:
    """A named, dependency-ordered DAG extracted from an expression."""

    bindings: tuple[tuple[Expr, Any], ...]
    result: Any

    @property
    def free_symbols(self):
        generated = {symbol.name for symbol, _ in self.bindings}
        symbols = set(
            self.result.free_symbols if isinstance(self.result, Expr) else ()
        )
        for _, expression in self.bindings:
            if isinstance(expression, Expr):
                symbols.update(expression.free_symbols)
        return frozenset(symbol for symbol in symbols if symbol.name not in generated)

    def evaluate(self, values):
        context = {
            (_raw(key).name if isinstance(_raw(key), Variable) else str(key)): value
            for key, value in values.items()
        }
        for symbol, expression in self.bindings:
            context[symbol.name] = _evaluate(_raw(expression), context)
        return _evaluate(_raw(self.result), context)

    def substitute(self, values):
        normalized = {
            (_raw(key).name if isinstance(_raw(key), Variable) else str(key)): _raw(value)
            for key, value in values.items()
        }
        bindings = tuple(
            (symbol, _expr(_substitute(_raw(expression), normalized)))
            for symbol, expression in self.bindings
        )
        result = _expr(_substitute(_raw(self.result), normalized))
        return CSEProgram(bindings, result)

    def __str__(self):
        lines = [f"let {symbol} = {expression}" for symbol, expression in self.bindings]
        lines.append(f"in {self.result}")
        return ";\n".join(lines)


def _expression_children(value):
    value = _raw(value)
    if isinstance(value, Node):
        return value.args
    return ()


def _expression_complexity(value):
    value = _raw(value)
    if isinstance(value, Node):
        return 1 + sum(_expression_complexity(child) for child in value.args)
    if isinstance(value, Polynomial):
        return 1 + len(value.monomials) + sum(
            len(monomial.exponents.mapping) for monomial in value.monomials
        )
    return 1


def cse(expression, *, min_uses=2, min_complexity=2, prefix="_cse"):
    """Extract repeated nontrivial expression subtrees into named bindings."""
    root = _raw(expression)
    counts = Counter()

    def count(value):
        value = _raw(value)
        if isinstance(value, (Node, Polynomial)):
            counts[value] += 1
        for child in _expression_children(value):
            count(child)

    count(root)
    candidates = {
        value
        for value, uses in counts.items()
        if uses >= min_uses
        and _expression_complexity(value) >= min_complexity
        and (
            isinstance(value, Node)
            or len(value.monomials) > 1
            or any(
                len(monomial.exponents.mapping) > 1
                for monomial in value.monomials
            )
        )
    }
    candidates.discard(root)

    used_names = set(_names(root))
    next_index = 0
    bindings = []
    assigned = {}

    def fresh_symbol():
        nonlocal next_index
        while f"{prefix}{next_index}" in used_names:
            next_index += 1
        name = f"{prefix}{next_index}"
        next_index += 1
        used_names.add(name)
        return Symbol(name)

    def rewrite(value, *, is_root=False):
        value = _raw(value)
        if isinstance(value, Node):
            rewritten = Node(
                value.op,
                tuple(rewrite(child) for child in value.args),
            )
        else:
            rewritten = value
        if not is_root and value in candidates:
            symbol = assigned.get(value)
            if symbol is None:
                symbol = fresh_symbol()
                assigned[value] = symbol
                bindings.append((symbol, _expr(rewritten)))
            return symbol.value
        return rewritten

    result = _expr(rewrite(root, is_root=True))
    return CSEProgram(tuple(bindings), result)


def dependencies(expression):
    return expression.free_symbols if isinstance(expression, Expr) else frozenset()


def canonicalize_polynomial(expression, **_kwargs):
    value = _raw(expression)
    if isinstance(value, Polynomial):
        return {tuple(m.exponents.mapping): m.coefficient for m in value.monomials}
    return None


def monomial_supports(expression, **_kwargs):
    value = _raw(expression)
    polynomial = canonicalize_polynomial(value)
    if polynomial is not None:
        return frozenset(frozenset(name for name, _ in powers) for powers in polynomial)
    if isinstance(value, Variable):
        return frozenset((frozenset((value.name,)),))
    if isinstance(value, Node):
        return frozenset().union(*(monomial_supports(arg) for arg in value.args))
    return frozenset()


def monomial_exponent_vectors(expression, **kwargs):
    vectors = set()

    def visit(value):
        value = _raw(value)
        polynomial = canonicalize_polynomial(value, **kwargs)
        if polynomial is not None:
            vectors.update(polynomial)
            return
        if isinstance(value, Node):
            for child in value.args:
                visit(child)

    visit(expression)
    return tuple(sorted(vectors))


def monomial_signature(expression, **kwargs):
    polynomial = canonicalize_polynomial(expression, **kwargs)
    return None if polynomial is None else tuple(polynomial)


def coupled_variable_groups(expression, **kwargs):
    supports = tuple(s for s in monomial_supports(expression, **kwargs) if s)
    result = []
    for support in supports:
        if len(support) > 1:
            result.append(frozenset(support))
    return tuple(result)


def multiplicative_product_groups(expression, *, proportional=True, **kwargs):
    """Find candidate product bundles from exponent-matrix rank.

    A group is a candidate when its exponent columns are all proportional.
    The restricted exponent matrix therefore has rank one, meaning that the
    corresponding polynomial region depends on one multiplicative coordinate
    (up to powers).  Regions are analyzed independently so unrelated
    measures do not contaminate one another.
    """
    regions = []

    def visit(value):
        value = _raw(value)
        polynomial = canonicalize_polynomial(value, **kwargs)
        if polynomial is not None:
            regions.append(tuple(polynomial))
            return
        if isinstance(value, Node):
            for child in value.args:
                visit(child)

    visit(expression)
    names = sorted({name for region in regions for vector in region for name, _ in vector})
    if len(names) < 2:
        return ()

    def columns_proportional(left, right):
        """Whether two columns are proportional, including zero entries."""
        scale = None
        for a, b in zip(left, right):
            if a == 0 and b == 0:
                continue
            if a == 0 or b == 0:
                return False
            ratio = a / b
            if scale is None:
                scale = ratio
            elif ratio != scale:
                return False
        return True

    compatible = {name: set() for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            valid = True
            for vectors in regions:
                rows = [dict(vector) for vector in vectors]
                left_column = tuple(Fraction(row.get(left, 0)) for row in rows)
                right_column = tuple(Fraction(row.get(right, 0)) for row in rows)
                if proportional:
                    compatible_columns = columns_proportional(left_column, right_column)
                else:
                    compatible_columns = left_column == right_column
                if not compatible_columns:
                    valid = False
                    break
            if valid:
                compatible[left].add(right)
                compatible[right].add(left)

    # Maximal cliques of the compatibility graph are exactly the maximal
    # groups whose restricted exponent matrix has rank at most one in every
    # polynomial leaf.
    groups = set()

    def bron_kerbosch(current, possible, excluded):
        if not possible and not excluded:
            if len(current) > 1:
                groups.add(frozenset(current))
            return
        for vertex in tuple(possible):
            bron_kerbosch(
                current | {vertex},
                possible & compatible[vertex],
                excluded & compatible[vertex],
            )
            possible.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(set(), set(names), set())
    return tuple(sorted(groups, key=lambda group: tuple(sorted(group))))


def exponent_rank(expression, **kwargs):
    vectors = monomial_exponent_vectors(expression, **kwargs)
    return len(vectors)


def is_expr(value):
    return isinstance(value, Expr)


def is_symbol(value):
    return isinstance(value, Expr) and isinstance(value.value, Variable)


def lambdify(args, expression, _backend="numpy", *, cse=False, **_kwargs):
    names = [a.name if isinstance(a, Expr) else str(a) for a in args]
    program = globals()["cse"](expression) if cse else None
    if program is None:
        return lambda *values: evaluate(expression, dict(zip(names, values, strict=True)))
    return lambda *values: program.evaluate(dict(zip(names, values, strict=True)))


__all__ = [
    "Expr",
    "Symbol",
    "Min",
    "Max",
    "SaturateSub",
    "Reciprocal",
    "Piecewise",
    "floor",
    "ceiling",
    "log",
    "evaluate",
    "evaluate_batch",
    "CSEProgram",
    "cse",
    "dependencies",
    "is_expr",
    "is_symbol",
    "lambdify",
]
