import unittest
from unittest.mock import patch

from examples.cli import main
from examples.xentropy import xentropy_spec


class ExampleMainTests(unittest.TestCase):
    def test_tune_and_load_share_build_and_execution(self):
        spec = xentropy_spec()
        sizes = dict(b=2, v=4, d=2)
        for load in (False, True):
            with self.subTest(load=load), patch('cutilereduce.fold.FoldOperator') as cls, patch(
                'cutilereduce.util.runner.print_plan'
            ), patch('examples.cli.run_example') as run:
                operator = cls.return_value
                argv = [
                    '--forward-only',
                    '--hardware', 'rtx5080',
                    '--max-storage-ratio', '2.5',
                    '--save-plan', 'saved.json',
                ]
                if load:
                    argv += ['--load-plan', 'existing.json']
                main('test', spec, None, sizes, reference=None, references={}, argv=argv)
                if load:
                    operator.load_plan.assert_called_once_with('existing.json', sizes)
                    operator.tune.assert_not_called()
                    plan = operator.load_plan.return_value
                else:
                    operator.load_plan.assert_not_called()
                    self.assertFalse(operator.tune.call_args.kwargs['backward'])
                    self.assertEqual(
                        operator.tune.call_args.kwargs['max_storage_ratio'],
                        2.5,
                    )
                    plan = operator.tune.return_value
                operator.save_plan.assert_called_once_with(
                    plan, 'saved.json', metadata={'hardware': 'rtx5080'},
                )
                operator.build.assert_called_once_with(
                    plan,
                    backward=False,
                    torch_compile=False,
                    programmatic_dependent_launch=False,
                )
                self.assertIs(run.call_args.args[3], operator.build.return_value)

    def test_programmatic_dependent_launch_reaches_build(self):
        spec = xentropy_spec()
        sizes = dict(b=2, v=4, d=2)
        with patch('cutilereduce.fold.FoldOperator') as cls, patch(
            'cutilereduce.util.runner.print_plan'
        ), patch('examples.cli.run_example'):
            operator = cls.return_value
            main(
                'test', spec, None, sizes, reference=None, references={},
                argv=['--hardware', 'rtx5080', '--programmatic-dependent-launch'],
            )
        self.assertTrue(operator.build.call_args.kwargs['programmatic_dependent_launch'])
