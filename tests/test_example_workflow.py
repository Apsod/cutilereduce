import unittest
from unittest.mock import patch

import torch

from examples.cli import ExampleParser, run_example, validate
from examples.xentropy import INITIALIZERS, reference, xentropy_spec


class WorkflowTests(unittest.TestCase):
    def test_shared_inputs_and_built_callables(self):
        spec = xentropy_spec()
        sizes = dict(b=3, v=7, d=4)
        args = ExampleParser(spec, sizes).parse_args([
            "--validate", "--benchmark-memory", "--torch-compile",
        ])
        def function(*inputs):
            return reference(*inputs)
        with patch('examples.cli.validate') as check, patch(
            'examples.cli.benchmark_full'
        ) as benchmark:
            run_example(
                'test', spec, sizes, function, args=args,
                reference=reference, references={'PyTorch': reference},
                initializers=INITIALIZERS, device='cpu',
            )
        self.assertIs(check.call_args.args[0], function)
        self.assertIs(check.call_args.args[1], benchmark.call_args.args[1])
        implementations = benchmark.call_args.args[2]
        self.assertNotIn('CuTile eager', implementations)
        self.assertIs(implementations['CuTile torch.compile'], function)
        self.assertIs(implementations['PyTorch'], reference)
        self.assertTrue(benchmark.call_args.kwargs['measure_memory'])

    def test_forward_only_validation_and_spec_gradient_names(self):
        spec = xentropy_spec()
        inputs = spec.mk_inputs(dict(b=3, v=7, d=4), device='cpu', **INITIALIZERS)
        with patch('cutilereduce.util.runner.validate_precision_matrix') as check:
            validate(reference, inputs, reference, spec=spec,
                     accuracy_matrix=True, backward=False)
        self.assertEqual(check.call_args.kwargs['input_names'], ('ctx', 'trg'))
        self.assertFalse(check.call_args.kwargs['backward'])
        self.assertEqual(check.call_args.kwargs['reference_dtypes']['PyTorch FP64'],
                         torch.float64)

    def test_disabled_workflow_does_not_allocate(self):
        spec = xentropy_spec()
        sizes = dict(b=3, v=7, d=4)
        args = ExampleParser(spec, sizes).parse_args(['--benchmark-seconds', '0'])
        with patch('examples.cli.make_inputs') as make:
            run_example('test', spec, sizes, reference, args=args,
                        reference=reference, references={})
        make.assert_not_called()
