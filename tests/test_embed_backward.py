from dataclasses import replace
import unittest

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import BufferRole
from cutilereduce.fold import FoldOperator, FoldPlan
from cutilereduce.fold.embed import embed_candidates
from cutilereduce.fold.persistence import fold_plan_from_record, plan_record
from cutilereduce.stages.base import StageSchedule
from cutilereduce.stages.embed import Embed
from cutilereduce.util.spec import SPECMAP
from examples import affine_attention, attention, xentropy


def example_cases():
    return (
        (xentropy, xentropy.xentropy_spec(), dict(b=33, v=65, d=32), dict(b=16, v=32)),
        (attention, attention.attention_spec(), dict(h=1, l=33, g=2, r=65, dqk=32, dv=32), dict(h=1, l=16, g=2, r=32)),
        (affine_attention, affine_attention.affine_attention_spec(), dict(h=1, l=33, g=2, r=65, dqk=32, dv=32, db=16), dict(h=1, l=16, g=2, r=32)),
    )


class EmbedPlanTests(unittest.TestCase):
    def test_stage_contracts_and_persistence(self):
        for module, spec, sizes, tiles in example_cases():
            for path in ("full", "partial"):
                with self.subTest(example=module.__name__, path=path):
                    op = FoldOperator(spec, module.FUNCTIONS)
                    plan = op.plan(sizes, path=path, tiles=tiles, partitions=2 if path == "partial" else 1)
                    embed, finalize = plan.backward
                    self.assertEqual(embed.stage.name, "embed")
                    self.assertEqual(tuple(b.id for b in embed.stage.write_buffers), spec.embedded.keys)
                    self.assertTrue(all(b.role in (BufferRole.Output, BufferRole.OutputGrad) for b in embed.stage.read_buffers))
                    self.assertTrue(all(b.role not in (BufferRole.Output, BufferRole.OutputGrad) for b in finalize.stage.read_buffers))
                    self.assertTrue(all(b.residual_multiplicity == 1 for b in embed.stage.write_buffers))
                    # Store a different embed tiling from the default and preserve it.
                    schedule = StageSchedule(
                        extents={a.id: a.extent for a in embed.domain.compute_axes},
                        tiles={a.id: 1 if a.axis in spec.batch else a.tile for a in embed.domain.compute_axes},
                        programs={},
                    )
                    plan = FoldPlan.make(spec, plan.forward, (Embed(spec, schedule).build(), finalize))
                    self.assertEqual(plan_record(fold_plan_from_record(spec, plan_record(plan))), plan_record(plan))

    def test_missing_embedded_spec_is_only_an_error_for_backward(self):
        module, spec, sizes, tiles = example_cases()[0]
        spec = replace(spec, embedded=type(spec.embedded)(values=()))
        op = FoldOperator(spec, module.FUNCTIONS)
        self.assertFalse(op.plan(sizes, tiles=tiles, backward=False).backward)
        with self.assertRaisesRegex(ValueError, "embedded buffers"):
            op.plan(sizes, tiles=tiles)

    def test_analytical_embed_candidates_have_unique_writers(self):
        for _, spec, sizes, _ in example_cases():
            candidates = embed_candidates(spec, sizes, hardware=SPECMAP["rtx5080"], candidate_count=4)
            self.assertTrue(candidates)
            for stage in candidates:
                self.assertIsNone(stage.domain.loop)
                self.assertTrue(all(b.residual_multiplicity == 1 for b in stage.stage.write_buffers))
                self.assertTrue(all(a.tile >= a.extent for a in stage.domain.inner_axes))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class EmbedCudaTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
        "programmatic dependent launch requires compute capability 9.0 or later",
    )
    def test_programmatic_dependent_launch_matches_sequential_backward(self):
        torch.manual_seed(23)
        module, spec, sizes, tiles = example_cases()[0]
        plan = FoldOperator(spec, module.FUNCTIONS).plan(
            sizes, path="partial", tiles=tiles, partitions=2,
        )
        inputs = spec.mk_inputs(sizes, device="cuda", **module.INITIALIZERS)
        cotangent = torch.randn((sizes["b"],), device="cuda")
        results = []
        for programmatic_dependent_launch in (False, True):
            operator_inputs = tuple(
                tensor.detach().clone().requires_grad_(tensor.requires_grad)
                for tensor in inputs
            )
            output, = FoldOperator(spec, module.FUNCTIONS).build(
                plan,
                programmatic_dependent_launch=programmatic_dependent_launch,
            )(*operator_inputs)
            gradients = torch.autograd.grad(
                output,
                tuple(tensor for tensor in operator_inputs if tensor.requires_grad),
                cotangent,
            )
            results.append((output, gradients))
        torch.testing.assert_close(results[1][0], results[0][0])
        for actual, expected in zip(results[1][1], results[0][1], strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_example_gradients(self):
        torch.manual_seed(19)
        for module, spec, sizes, tiles in example_cases():
            inputs = spec.mk_inputs(sizes, device="cuda", **module.INITIALIZERS)
            reference = attention.sdpa if module is attention else module.reference
            reference_inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
            expected = reference(*reference_inputs, dtype=torch.float32) if module is affine_attention else reference(*reference_inputs)
            cotangent = torch.randn_like(expected)
            expected_grads = torch.autograd.grad(expected, tuple(x for x in reference_inputs if x.requires_grad), cotangent)
            variants = [("full", False, None), ("partial", False, None)]
            if module is affine_attention:
                variants.append(("partial", True, None))
            else:
                variants.append(("partial", False, "b" if module is xentropy else "l"))
            for path, checkpointed, loop in variants:
                with self.subTest(example=module.__name__, path=path, checkpoints=checkpointed, loop=loop):
                    op = FoldOperator(spec, module.FUNCTIONS)
                    plan = op.plan(
                        sizes, path=path, tiles=tiles, partitions=2 if path == "partial" else 1,
                        checkpointed_backward=checkpointed, backward_loop=loop,
                    )
                    actual, = op.build(plan)(*inputs)
                    actual_grads = torch.autograd.grad(actual, tuple(x for x in inputs if x.requires_grad), cotangent)
                    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025, check_dtype=False)
                    for grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                        self.assertTrue(torch.isfinite(grad).all())
                        if module is affine_attention:
                            # BF16 cotangent rounding accumulates through the affine
                            # prefixes; the existing fused kernel has the same error.
                            relative_error = (grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
                            self.assertLess(relative_error.item(), 0.02)
                        else:
                            torch.testing.assert_close(grad, expected_grad, atol=0.04, rtol=0.04)

    def test_embed_matches_callback_and_preserves_dtype(self):
        # Comparing against native Torch operations checks the new storage boundary.
        spec = attention.attention_spec()
        sizes = dict(h=1, l=33, g=2, r=65, dqk=32, dv=32)
        schedule = StageSchedule.make(spec, extents=sizes, tiles=dict(h=1, l=16, g=2, dv=32))
        stage = Embed(spec, schedule).build()
        kernel = stage.compile(attention.FUNCTIONS)
        shapes = [(1, 33, 2), (1, 33, 2, 32)]
        z, mu = [torch.randn(shape, device="cuda") for shape in shapes]
        gz, gm = [torch.randn(shape, device="cuda") for shape in shapes]
        out = tuple(b.mk_empty(device="cuda") for b in stage.stage.write_buffers)
        ct.launch(torch.cuda.current_stream(), (int(stage.stage.program_count), 1, 1), kernel, ((z, mu, gz, gm), out))
        torch.testing.assert_close(out[0], z)
        torch.testing.assert_close(out[1], gz - (mu * gm).sum(-1))
        torch.testing.assert_close(out[2], gm.to(torch.bfloat16))
        self.assertEqual(out[2].dtype, torch.bfloat16)
