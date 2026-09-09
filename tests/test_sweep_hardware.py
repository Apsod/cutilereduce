import importlib
import unittest


class SweepHardwareTests(unittest.TestCase):
    def test_commutative_fold_requires_hardware(self):
        sweep = importlib.import_module("cutilereduce.fold.commutative.sweep")
        with self.assertRaisesRegex(TypeError, "hardware"):
            getattr(sweep, "sweep_commutative_fold")(object(), sizes={})

    def test_commutative_backward_requires_hardware(self):
        sweep = importlib.import_module("cutilereduce.fold.commutative.sweep")
        with self.assertRaisesRegex(TypeError, "hardware"):
            getattr(sweep, "sweep_commutative_backward")(object(), sizes={})

    def test_general_fold_requires_hardware(self):
        sweep = importlib.import_module("cutilereduce.fold.general.sweep")
        with self.assertRaisesRegex(TypeError, "hardware"):
            getattr(sweep, "sweep_general_fold")(object(), sizes={})

    def test_general_backward_requires_hardware(self):
        sweep = importlib.import_module("cutilereduce.fold.general.sweep")
        with self.assertRaisesRegex(TypeError, "hardware"):
            getattr(sweep, "sweep_general_backward")(
                object(), sizes={}, forward_plan=object(),
            )


if __name__ == "__main__":
    unittest.main()
