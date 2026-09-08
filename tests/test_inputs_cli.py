import contextlib
import io
import unittest

import torch

from examples.xentropy import xentropy_spec
from examples.attention import attention_spec
from examples.cli import ExampleParser


class InputTests(unittest.TestCase):
    def test_inputs_follow_spec_and_callbacks_see_all_sizes(self):
        spec = xentropy_spec()
        sizes = dict(b=3, v=7, d=4)
        ctx, trg, targets = spec.mk_inputs(
            sizes, device="cpu",
            ctx=lambda t, s: t.fill_(s["v"]),
            targets=lambda t, s: t.random_(s["v"]),
        )
        self.assertEqual(ctx.shape, (3, 4))
        self.assertEqual(trg.shape, (7, 4))
        self.assertEqual(targets.shape, (3,))
        self.assertTrue(torch.all(ctx == 7))
        self.assertTrue(torch.all((targets >= 0) & (targets < 7)))
        for tensor, buffer in zip((ctx, trg, targets), spec.input, strict=True):
            self.assertEqual(tensor.dtype, buffer.torch_dtype)
            self.assertEqual(tensor.requires_grad, buffer.req_grad)
            self.assertTrue(tensor.is_leaf)
        (ctx.float().sum() + trg.float().sum()).backward()
        self.assertIsNotNone(ctx.grad)

    def test_invalid_initialization(self):
        spec = xentropy_spec()
        sizes = dict(b=3, v=7, d=4)
        for kwargs in ({}, {"typo": lambda t, s: t},
                       {"ctx": lambda t, s: t.double()}):
            with self.assertRaises(ValueError):
                spec.mk_inputs(sizes, device="cpu", **kwargs)
        with self.assertRaises(ValueError):
            spec.mk_inputs(dict(b=0, v=7, d=4), device="cpu")
        with self.assertRaises(ValueError):
            spec.mk_inputs(dict(b=3, v=7), device="cpu")

    def test_replacements_become_leaves(self):
        spec = xentropy_spec()
        source = torch.ones((3, 4), dtype=torch.bfloat16, requires_grad=True)
        ctx, _, _ = spec.mk_inputs(
            dict(b=3, v=7, d=4), device="cpu",
            ctx=lambda t, s: source * 2,
            targets=lambda t, s: t.zero_(),
        )
        self.assertTrue(ctx.is_leaf)
        self.assertTrue(ctx.requires_grad)

    def test_generated_cli_and_aliases(self):
        defaults = dict(h=2, l=8, g=2, r=8, dqk=4, dv=4)
        parser = ExampleParser(attention_spec(), defaults, aliases={"h": "heads"})
        args = parser.parse_args(["--heads", "3", "--r", "12"])
        self.assertEqual(parser.sizes(args), defaults | {"h": 3, "r": 12})
        for argv in (["--r", "0"], ["--candidates", "-1"],
                     ["--benchmark-seconds", "nan"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
