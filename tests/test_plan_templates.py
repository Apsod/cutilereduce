import unittest

from cutilereduce.core.sweep import DEFAULT_FUNCTIONS, DEFAULT_SYMBOLS, evaluate_stage
from cutilereduce.fold import (
    backward_plan_templates,
    compose_plans,
    forward_plan_templates,
    joint_plan_templates,
)
from cutilereduce.fold.templates import FoldPlanTemplate
from cutilereduce.fold.paths import backward_plans, forward_plans
from cutilereduce.fold.persistence import fold_plan_from_record, plan_record
from cutilereduce.util.spec import SPECMAP
from examples.affine_attention import affine_attention_spec
from examples.xentropy import xentropy_spec


class PlanTemplateTests(unittest.TestCase):
    def test_enumeration_and_checkpoint_requirement(self):
        commutative = xentropy_spec()
        self.assertEqual(
            [template.name for template in forward_plan_templates(commutative)],
            ["full", "partial"],
        )
        self.assertEqual(
            {template.name for template in backward_plan_templates(commutative)},
            {"commutative"},
        )
        self.assertEqual(
            [stage.stage.name for stage in forward_plan_templates(commutative)[1].stages],
            ["map_fold_partial", "fold"],
        )
        general = affine_attention_spec()
        self.assertEqual(
            [template.name for template in backward_plan_templates(general)],
            ["full_recompute", "checkpointed"],
        )
        self.assertEqual(
            {template.name for template in joint_plan_templates(general)},
            {"full+full_recompute", "partial+full_recompute", "partial+checkpointed"},
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            compose_plans(forward_plans(general)[0], backward_plans(general)[1])

    def test_cost_remains_symbolic_until_values_are_supplied(self):
        spec = xentropy_spec()
        backward = backward_plans(spec)[0].with_loop("v")
        template = FoldPlanTemplate(spec, forward_plans(spec)[0], backward)
        self.assertEqual(
            [name for name, _ in template.stage_costs],
            ["map_fold", "embed", "recompute_map_finalize_grad_write"],
        )
        self.assertEqual(
            template.estimated_time,
            template.forward_estimated_time + template.backward_estimated_time,
        )
        self.assertIn("V", template.parameters)
        self.assertIn("full_v_tile", template.parameters)
        self.assertNotIn("BANDWIDTH", template.parameters)
        partially_bound = template.evaluate({"v": 128}, hardware=SPECMAP["rtx5080"])
        self.assertNotIn("V", {str(symbol) for symbol in partially_bound.free_symbols})
        self.assertIn("full_v_tile", {str(symbol) for symbol in partially_bound.free_symbols})

        values = {
            "b": 32, "v": 128, "d": 64,
            "full_b_tile": 16, "full_v_tile": 32,
            "backward_b_tile": 16, "backward_v_tile": 32,
            "backward_group_count": 1, "embed_b_tile": 16,
        }
        plan = template.bind(values)
        hardware = SPECMAP["rtx5080"]
        direct = sum(stage.stage.cost.estimated_time for stage in (*plan.forward, *plan.backward))
        for function, replacement in DEFAULT_FUNCTIONS.items():
            direct = direct.replace(function, replacement)
        direct = direct.subs(DEFAULT_SYMBOLS | hardware)
        self.assertAlmostEqual(float(template.evaluate(values, hardware=hardware)), float(direct))
        with self.assertRaisesRegex(ValueError, "conflicting values"):
            template.evaluate({"v": 128, "V": 64})

    def test_partial_plan_constraints(self):
        spec = xentropy_spec()
        template = FoldPlanTemplate(spec, forward_plans(spec)[1])
        values = {
            "B": 32, "V": 128, "D": 64,
            "partial_b_tile": 16, "partial_v_tile": 32,
            "combine_b_tile": 16, "partition_count": 2,
        }
        self.assertEqual(
            [stage.stage.name for stage in template.bind(values).forward],
            ["map_fold_partial", "fold"],
        )
        with self.assertRaisesRegex(ValueError, "exceeds fold tile count"):
            template.bind(values | {"partition_count": 5})

    def test_residency_stays_symbolic_after_hardware_substitution(self):
        cost = forward_plans(affine_attention_spec())[0].stage_costs[0][1]
        occupancy = cost.resident_programs_per_sm.subs(SPECMAP["a100"])
        self.assertIn("full_r_tile", {str(symbol) for symbol in occupancy.free_symbols})
        self.assertEqual(occupancy.subs({symbol: 1 for symbol in occupancy.free_symbols}), 32)
        self.assertEqual(occupancy.subs({symbol: 32 for symbol in occupancy.free_symbols}), 0)

    def test_sweep_cost_matches_bound_plan(self):
        path = forward_plans(affine_attention_spec())[0]
        values = {
            "db": 16, "dqk": 128, "dv": 128, "g": 4, "h": 8,
            "l": 4096, "r": 4096, "full_g_tile": 1, "full_h_tile": 1,
            "full_l_tile": 32, "full_r_tile": 64,
        }
        hardware = SPECMAP["a100"]
        config = {
            key.upper() if key in {axis.name for axis in path.spec.axes} else key: value
            for key, value in values.items()
        }
        sweep = evaluate_stage(
            path.stages[0].stage,
            attributes=("resident_programs_per_sm", "estimated_time"),
            configs=[config], symbols=hardware,
        )
        bound = path.bind(values).forward[0].stage.cost
        substitutions = DEFAULT_SYMBOLS | hardware
        self.assertEqual(sweep["resident_programs_per_sm"][0], 2)
        self.assertEqual(sweep["resident_programs_per_sm"][0], float(bound.resident_programs_per_sm.subs(substitutions)))
        self.assertAlmostEqual(sweep["estimated_time"][0], float(bound.estimated_time.subs(substitutions)))

    def test_checkpointed_path_owns_carrier_and_persistence(self):
        spec = affine_attention_spec()
        template = compose_plans(forward_plans(spec)[1], backward_plans(spec)[1])
        sizes = {"H": 1, "L": 33, "G": 2, "R": 65, "DQK": 32, "DV": 32, "DB": 16}
        values = {
            key: sizes[key] if key in sizes else (
                2 if key == "partition_count" else 32 if key == "partial_r_tile" else 1
            )
            for key in template.parameters
        }
        plan = template.bind(values)
        self.assertEqual(plan.backward[1].checkpoints, plan.forward[1].carriers)
        record = plan_record(plan)
        self.assertEqual(plan_record(fold_plan_from_record(spec, record)), record)
