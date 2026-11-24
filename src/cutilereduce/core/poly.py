"""Canonical polynomial algebra used by the analytical expression layer."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import math
from collections.abc import Mapping


@dataclass(frozen=True, order=True)
class Exponents:
    mapping: tuple[tuple[str, Fraction], ...]

    @classmethod
    def make(cls, **exponents):
        return cls.from_dict(exponents)

    @classmethod
    def from_dict(cls, values: Mapping[str, int | Fraction]):
        return cls(tuple(sorted(
            (name, Fraction(value))
            for name, value in values.items()
            if value != 0
        )))

    def __mul__(self, other):
        values = dict(self.mapping)
        for name, exponent in other.mapping:
            values[name] = values.get(name, 0) + exponent
        return type(self).from_dict(values)

    def __invert__(self):
        return type(self).from_dict({name: -value for name, value in self.mapping})


@dataclass(frozen=True, order=True)
class Monomial:
    coefficient: Fraction
    exponents: Exponents

    @classmethod
    def make(cls, coefficient, exponents):
        coefficient = Fraction(coefficient)
        if coefficient < 0:
            raise ValueError("monomials are nonnegative")
        return cls(coefficient, exponents)

    @property
    def is_zero(self):
        return self.coefficient == 0

    def __mul__(self, other):
        return type(self).make(
            self.coefficient * other.coefficient,
            self.exponents * other.exponents,
        )

    def __invert__(self):
        if self.is_zero:
            raise ZeroDivisionError("zero has no inverse")
        return type(self).make(1 / self.coefficient, ~self.exponents)


@dataclass(frozen=True, order=True)
class Polynomial:
    monomials: tuple[Monomial, ...] = ()

    @cached_property
    def _hash(self):
        return hash(self.monomials)

    def __hash__(self):
        return self._hash

    def __reduce__(self):
        return type(self), (self.monomials,)

    @classmethod
    def from_monomials(cls, monomials):
        coefficients = {}
        for monomial in monomials:
            if monomial.is_zero:
                continue
            coefficients[monomial.exponents] = (
                coefficients.get(monomial.exponents, Fraction(0))
                + monomial.coefficient
            )
        return cls(tuple(sorted(
            Monomial.make(coefficient, exponents)
            for exponents, coefficient in coefficients.items()
            if coefficient != 0
        )))

    @classmethod
    def constant(cls, value):
        value = Fraction(value)
        return cls.from_monomials(()) if value == 0 else cls.from_monomials(
            (Monomial.make(value, Exponents(())),)
        )

    @classmethod
    def zero(cls):
        return cls(())

    @property
    def is_zero(self):
        return not self.monomials

    @property
    def is_monomial(self):
        return len(self.monomials) <= 1

    @property
    def variables(self):
        return frozenset(
            name for monomial in self.monomials
            for name, _ in monomial.exponents.mapping
        )

    def __str__(self):
        if self.is_zero:
            return "0"
        terms = []
        for monomial in self.monomials:
            factors = [name if exponent == 1 else f"{name}^{exponent}"
                       for name, exponent in monomial.exponents.mapping]
            coefficient = monomial.coefficient
            if not factors:
                term = str(coefficient)
            elif coefficient == 1:
                term = "*".join(factors)
            else:
                term = f"{coefficient}*{'*'.join(factors)}"
            terms.append(term)
        return " + ".join(terms)

    def subs(self, values: Mapping):
        result = type(self).zero()
        for monomial in self.monomials:
            coefficient = monomial.coefficient
            exponents = {}
            for name, exponent in monomial.exponents.mapping:
                value = values.get(name)
                if value is None:
                    exponents[name] = exponent
                elif hasattr(value, "name") and exponent.denominator == 1:
                    # Symbolic renaming (used when hardware/config symbols are
                    # mapped to sweep-column symbols).
                    exponents.pop(name, None)
                    exponents[value.name] = exponents.get(value.name, 0) + exponent
                elif isinstance(value, (int, float, Fraction)) and exponent.denominator == 1:
                    coefficient *= Fraction(value) ** exponent.numerator
                else:
                    exponents[name] = exponent
            result = result + type(self).from_monomials((Monomial.make(coefficient, Exponents.from_dict(exponents)),))
        return result

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            other = type(self).constant(other)
        return type(self).from_monomials((*self.monomials, *other.monomials))

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            other = type(self).constant(other)
        return type(self).from_monomials(
            left * right
            for left in self.monomials
            for right in other.monomials
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Polynomial) or not other.is_monomial or other.is_zero:
            raise ValueError("polynomial division requires a monomial denominator")
        if other.monomials[0].coefficient <= 0:
            raise ValueError("polynomial division requires a positive denominator")
        return self * type(self).from_monomials((~other.monomials[0],))

    def reciprocal(self):
        if not self.is_monomial or self.is_zero:
            raise ValueError("only nonzero monomials have polynomial reciprocals")
        return type(self).from_monomials((~self.monomials[0],))

    def nonnegative(self):
        """Conservative sign proof for nonnegative variables."""
        return all(monomial.coefficient >= 0 for monomial in self.monomials)


def mono(coefficient, **exponents):
    return Polynomial.from_monomials((Monomial.make(
        coefficient, Exponents.make(**exponents)
    ),))


def polynomial_upper_bound(
    polynomial: Polynomial,
    variable: str,
    limit,
    substitutions: Mapping[str, int | float | Fraction] | None = None,
):
    """Return a conservative integer bound for ``polynomial <= limit``.

    All variables other than ``variable`` must be substituted.  Every
    nonnegative monomial contributes independently to the bound, so the
    result remains safe for coupled products.
    """
    substitutions = dict(substitutions or {})
    remaining = float(limit)
    bounds = []
    found = False
    for monomial in polynomial.monomials:
        powers = dict(monomial.exponents.mapping)
        exponent = powers.pop(variable, 0)
        coefficient = float(monomial.coefficient)
        for name, power in powers.items():
            if name not in substitutions:
                return None
            coefficient *= float(substitutions[name]) ** float(power)
        if exponent == 0:
            remaining -= coefficient
            continue
        if exponent < 0 or coefficient <= 0:
            return None
        found = True
        if remaining < 0:
            return 0
        bounds.append(math.floor((remaining / coefficient) ** (1 / float(exponent))))
    if not found:
        return None
    return max(0, min(bounds, default=0))


__all__ = [
    "Exponents",
    "Monomial",
    "Polynomial",
    "mono",
    "polynomial_upper_bound",
]
