import unittest
from unittest.mock import patch

from examples.cli import main
from examples.xentropy import xentropy_spec


class ExampleMainTests(unittest.TestCase):
    def test_tune_and_load_share_build_and_execution(self):
        spec = xentropy_spec()
        sizes = dict(b=2, v=4, d=2)
        for load in (False, True):
            with self.subTest(load=load), patch('cutilereduce.fold.tune_fold_plan') as tune, patch(
                'cutilereduce.fold.persistence.load_fold_plan'
            ) as load_plan, patch(
                'cutilereduce.fold.persistence.save_fold_plan'
            ) as save_plan, patch(
                'cutilereduce.fold.build_fold'
            ) as build, patch(
                'cutilereduce.util.runner.print_plan'
            ), patch('examples.cli.run_example') as run:
                argv = [
                    '--forward-only', '--hardware', 'rtx5080',
                    '--max-storage-ratio', '2.5', '--save-plan', 'saved.json',
                ]
                if load:
                    argv += ['--load-plan', 'existing.json']
                main('test', spec, None, sizes, reference=None, references={}, argv=argv)
                if load:
                    load_plan.assert_called_once_with(spec, 'existing.json', sizes)
                    tune.assert_not_called()
                    plan = load_plan.return_value
                else:
                    load_plan.assert_not_called()
                    self.assertFalse(tune.call_args.kwargs['backward'])
                    self.assertEqual(tune.call_args.kwargs['max_storage_ratio'], 2.5)
                    plan = tune.return_value
                save_plan.assert_called_once_with(
                    plan, 'saved.json', metadata={'hardware': 'rtx5080'},
                )
                build.assert_called_once_with(
                    plan, None, backward=False, torch_compile=False,
                    programmatic_dependent_launch=False,
                )
                self.assertIs(run.call_args.args[3], build.return_value)

    def test_programmatic_dependent_launch_reaches_build(self):
        spec = xentropy_spec()
        sizes = dict(b=2, v=4, d=2)
        with patch('cutilereduce.fold.tune_fold_plan'), patch(
            'cutilereduce.fold.build_fold'
        ) as build, patch('cutilereduce.util.runner.print_plan'), patch(
            'examples.cli.run_example'
        ):
            main(
                'test', spec, None, sizes, reference=None, references={},
                argv=['--hardware', 'rtx5080', '--programmatic-dependent-launch'],
            )
        self.assertTrue(build.call_args.kwargs['programmatic_dependent_launch'])
