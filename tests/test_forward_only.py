import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl

from examples.cli import ExampleParser, run_example
from examples.xentropy import xentropy_spec
from cutilereduce.fold.commutative.tune import tune_commutative_fold_plan


class ForwardOnlyTests(unittest.TestCase):
    def test_tuning_skips_backward(self):
        module = 'cutilereduce.fold.commutative.tune'
        stage = SimpleNamespace(stage=SimpleNamespace(
            name='map_fold', domain=SimpleNamespace(compute_axes=()),
        ))
        forward = SimpleNamespace(forward=(stage,))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch(f'{module}.sweep_commutative_fold',
                                      return_value=pl.DataFrame({'path': ['single']})))
            backward = stack.enter_context(patch(f'{module}.sweep_commutative_backward'))
            stack.enter_context(patch(f'{module}.commutative_fold_plan_from_config',
                                      return_value=forward))
            stack.enter_context(patch(f'{module}.stage_key', return_value='fwd'))
            tune = stack.enter_context(patch(f'{module}.tune_fold_stages',
                                             return_value={'fwd': 1.0}))
            make = stack.enter_context(patch(f'{module}.FoldPlan.make', return_value=forward))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            spec = xentropy_spec()
            tune_commutative_fold_plan(spec, {}, 1, 0, functions=None,
                                       hardware={}, backward=False)
        backward.assert_not_called()
        tune.assert_called_once()
        self.assertEqual(make.call_args.args[2], ())

    def test_shared_flag_controls_execution(self):
        spec = xentropy_spec()
        sizes = dict(b=2, v=4, d=2)
        parser = ExampleParser(spec, sizes)
        args = parser.parse_args([
            '--forward-only', '--validate', '--hardware', 'rtx5080',
        ])
        with patch('examples.cli.make_inputs', return_value=()) as inputs, patch(
            'examples.cli.validate'
        ) as validate, patch('examples.cli.benchmark_full') as benchmark:
            run_example('test', spec, sizes, object(), args=args,
                        reference=object(), references={})
        inputs.assert_called_once()
        self.assertFalse(validate.call_args.kwargs['backward'])
        self.assertFalse(benchmark.call_args.kwargs['backward'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([
                '--forward-only', '--torch-compile', '--hardware', 'rtx5080',
            ])
