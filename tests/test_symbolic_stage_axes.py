import unittest

import numpy as np

from cutilereduce.core import (
    AxisRole,
    Axes,
    ComputeAxis,
    ComputeAxes,
    ProgramAxis,
    StageDomain,
)
from cutilereduce.core import expr


class SymbolicStageAxisTests(unittest.TestCase):
    def test_compute_axis_owns_namespaced_tile(self):
        (axis,) = Axes.make("g")
        compute = ComputeAxis.make(axis, AxisRole.Batch, namespace="full")

        self.assertEqual(str(compute.extent), "g")
        self.assertEqual(str(compute.tile), "full_g_tile")
        self.assertFalse(compute.resolved)

    def test_inner_axis_defaults_to_full_extent(self):
        (axis,) = Axes.make("dqk")
        compute = ComputeAxis.make(axis, AxisRole.Inner, namespace="full")

        self.assertEqual(compute.extent, compute.tile)
        self.assertEqual(str(compute.extent), "dqk")

    def test_domain_exposes_owned_parameters(self):
        (axis,) = Axes.make("g")
        compute = ComputeAxis.make(axis, AxisRole.Batch, namespace="full")
        program = ProgramAxis.make(axis, axis.id, expr.Symbol("full_g_programs"))
        domain = StageDomain.make("test", ComputeAxes(values=(compute,)))
        domain = StageDomain(
            name=domain.name,
            compute_axes=domain.compute_axes,
            program_axes=type(domain.program_axes)(values=(program,)),
        )

        self.assertEqual(
            set(domain.parameters), {"g", "full_g_tile", "full_g_programs"}
        )

    def test_expression_normalization_flattens_sums_and_removes_identities(self):
        x = expr.Symbol("x")
        y = expr.Symbol("y")
        z = expr.Symbol("z")

        self.assertEqual(str(0 + x), "x")
        self.assertEqual(str((x + y) + z), "x + y + z")
        self.assertEqual(str(1 * x), "x")
        self.assertEqual(str((x * y) * z), "x*y*z")
        self.assertEqual(str(x / 1), "x")

    def test_piecewise_min_shape_becomes_min(self):
        x = expr.Symbol("x")
        bounded = expr.Piecewise((32, 32 <= x), (x, True))

        self.assertEqual(str(bounded), "min(x, 32)")

    def test_polynomial_mapper_collects_monomials(self):
        a, b, x, y = (expr.Symbol(name) for name in ("a", "b", "x", "y"))
        polynomial = expr.canonicalize_polynomial((a + b) * (x + y) + 2 * a * x)

        self.assertEqual(
            polynomial,
            {
                (("a", 1), ("x", 1)): 3,
                (("a", 1), ("y", 1)): 1,
                (("b", 1), ("x", 1)): 1,
                (("b", 1), ("y", 1)): 1,
            },
        )
        self.assertIsNone(expr.canonicalize_polynomial(expr.Min(a, b)))

    def test_support_analysis_works_inside_nonpolynomial_expression(self):
        h, lsize, g, r = (expr.Symbol(name) for name in ("h", "l", "g", "r"))
        cost = expr.Min(32, (h * lsize * g) / expr.Max(1, r)) + expr.log(lsize * g)

        supports = expr.monomial_supports(cost)
        self.assertIn(frozenset(("h", "l", "g")), supports)
        self.assertIn(frozenset(("l", "g")), supports)
        self.assertIn(frozenset(("r",)), supports)
        self.assertIn(frozenset(("l", "g")), expr.coupled_variable_groups(cost))

    def test_cse_extracts_repeated_nontrivial_nodes(self):
        x, y = (expr.Symbol(name) for name in ("x", "y"))
        repeated = expr.floor(x / y)
        expression = repeated * repeated + repeated

        program = expr.cse(expression)

        self.assertEqual(len(program.bindings), 2)
        self.assertEqual(program.free_symbols, frozenset((x, y)))
        self.assertEqual(
            program.evaluate({"x": 7, "y": 3}),
            expr.evaluate(expression, {"x": 7, "y": 3}),
        )

    def test_cse_vector_evaluation_and_lambdify_agree(self):
        x, y = (expr.Symbol(name) for name in ("x", "y"))
        repeated = expr.floor(x / y)
        expression = repeated * repeated + repeated
        xs = np.array([7, 10])
        ys = np.array([3, 4])

        program = expr.cse(expression)
        expected = expr.evaluate(expression, {"x": xs, "y": ys})
        actual = program.evaluate({"x": xs, "y": ys})
        compiled = expr.lambdify((x, y), expression, cse=True)(xs, ys)

        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(compiled.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
