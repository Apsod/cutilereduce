from dataclasses import replace
import unittest

import cuda.tile as ct
import polars as pl
import sympy
import torch

from cutilereduce.core import (
    ATOMIC_ADD,
    Axes,
    atomic_add_write,
    chunked_evaluate_stage,
    evaluate_stage,
    matmul,
    workmodel,
)
from cutilereduce.core.axis import LogicalAxis
from cutilereduce.core.buffer import BufferSpec, Input, Output, buffer_spec, bundle_spec
from cutilereduce.core.stage_buffer import BufferStorage, KernelBuffers, WRITE
from cutilereduce.fold import (
    AlgebraKind,
    StageSchedule,
    fold_functions,
    make_fold_spec,
    mk_fold_autograd,
    mk_fold_forward,
)
from cutilereduce.fold.commutative import (
    commutative_backward_stage,
    full_fold_plan,
    partial_fold_plan,
    sweep_commutative_fold,
)
from cutilereduce.stages import (
    BufferUse,
    Fold,
    Map,
    MapFold,
    MapFoldPartial,
    RecomputeFoldMapFinalizeGradWrite,
    RecomputeMapFinalizeGradWrite,
    RecomputePrefixFoldMapFinalizeGradWrite,
    Scan,
    bind_buffer_uses,
)
from cutilereduce.util.spec import SPECMAP


@ct.function
def add_map_fold(tid, x):
    return (x,)


@ct.function
def add_map_fold_sum(tid, x):
    return (ct.sum(x, axis=1),)


@ct.function
def add_map_fold_sum_with_named_tid(tid, x):
    v = tid.indices("v")
    mask = tid.mask("v")
    return (ct.sum(ct.where(mask[None, :], x + v[None, :] * 0, 0), axis=1),)


@ct.function
def add_combine(a, b):
    return (a + b,)


@ct.function
def scaled_add_map_fold_sum(tid, x, scale):
    return (ct.sum(x * scale[:, None], axis=1),)


@ct.function
def add_embed(y, g_y):
    return (g_y,)


@ct.function
def add_map_finalize_commutative(tid, x, g_x, g_y):
    return (g_x + g_y[:, None],)


@ct.function
def add_map_finalize(tid, x, g_x, g_y, prefix):
    return (
        (g_x + g_y[:, None],),
        (prefix + ct.sum(x, axis=1),),
    )


@ct.function
def scaled_add_map_finalize(tid, x, scale, g_x, g_scale, g_y):
    return (
        g_x + scale[:, None] * g_y[:, None],
        g_scale + ct.sum(x * g_y[:, None], axis=1),
    )


def double_output(y):
    return y * 2


def xentropy_spec(*, algebra=AlgebraKind.commutative):
    return make_fold_spec(
        input={
            "ctx": buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
            "trg": buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
            "targets": buffer_spec("b", ct.int32, default=-100),
        },
        execution={
            "m": buffer_spec("b", ct.float32, default=float("-inf")),
            "e": buffer_spec("b", ct.float32, default=0),
            "u": buffer_spec("b", ct.float32, default=0),
        },
        semantic={
            "z": buffer_spec("b", ct.float32, default=float("-inf")),
            "l": buffer_spec("b", ct.float32, default=0),
        },
        map_fold_work=workmodel(matmul(M="b", N="v", K="d")),
        batch="b",
        fold="v",
        algebra=algebra,
    )


def add_spec():
    return make_fold_spec(
        input={
            "x": buffer_spec("b v", ct.float32, req_grad=True, default=0),
        },
        execution={
            "acc": buffer_spec("b", ct.float32, default=0),
        },
        semantic={
            "y": buffer_spec("b", ct.float32, default=0),
        },
        batch="b",
        fold="v",
    )


def scaled_add_spec():
    return make_fold_spec(
        input={
            "x": buffer_spec("b v", ct.float32, req_grad=True, default=0),
            "scale": buffer_spec("b", ct.float32, req_grad=True, default=0),
        },
        execution={
            "acc": buffer_spec("b", ct.float32, default=0),
        },
        semantic={
            "y": buffer_spec("b", ct.float32, default=0),
        },
        batch="b",
        fold="v",
    )


def schedule_for(spec, *, tile_v=4):
    return StageSchedule.make(
        spec,
        extents={"b": 4, "v": 8},
        tiles={"b": 2, "v": tile_v},
        loop=spec.fold,
    )


def partial_schedules_for(spec):
    partition_axis = spec.fold.partition_axis
    partial_schedule = StageSchedule.make(
        spec,
        extents={"b": 4, "v": 8},
        tiles={"b": 2, "v": 2},
        programs={partition_axis: 2},
        loop=spec.fold,
    )
    combine_schedule = StageSchedule.make(
        spec,
        extents={"b": 4, "v": 8, partition_axis: 2},
        tiles={"b": 2, partition_axis: 2},
        programs={partition_axis: 1},
        loop=partition_axis,
    )
    return partial_schedule, combine_schedule


class FoldStageConstructionTests(unittest.TestCase):
    def test_fold_spec_rejects_partition_axis_as_semantic_fold(self):
        fold = LogicalAxis.make("v")
        partition_axis = getattr(fold, "partition_axis")
        with self.assertRaisesRegex(TypeError, "fold axis must be logical"):
            make_fold_spec(
                input={"x": buffer_spec("b v", ct.float32, default=0)},
                execution={"acc": buffer_spec("b", ct.float32, default=0)},
                semantic={"y": buffer_spec("b", ct.float32, default=0)},
                batch="b",
                fold=partition_axis,
            )

    def test_fold_spec_rejects_partition_axis_in_user_buffers(self):
        fold = LogicalAxis.make("v")
        partition_axis = fold.partition_axis
        with self.assertRaisesRegex(TypeError, "buffer axes must be logical"):
            make_fold_spec(
                input={
                    "x": buffer_spec("b v", ct.float32, default=0),
                },
                execution={
                    "acc": buffer_spec("b", ct.float32, default=0),
                    "partitioned": BufferSpec(
                        axes=Axes(values=(partition_axis,)),
                        dtype=ct.float32,
                        default=0,
                    ),
                },
                semantic={"y": buffer_spec("b", ct.float32, default=0)},
                batch="b",
                fold=fold,
            )

    def test_map_fold_intermediate_and_workmodel_interface(self):
        spec = make_fold_spec(
            input={"x": buffer_spec("b v", ct.float32, req_grad=True, default=0)},
            execution={"acc": buffer_spec("b", ct.float32, default=0)},
            semantic={"y": buffer_spec("b", ct.float32, default=0)},
            map_fold_intermediate={"tmp": buffer_spec("b v", ct.float32)},
            batch="b",
            fold="v",
            map_fold_work=workmodel(
                matmul(M="b", N="v", K="d"),
                matmul(M="b", N="d", K="v"),
            ),
            map_finalize_work=workmodel(matmul(M="v", N="d", K="b")),
        )
        self.assertEqual(tuple(buffer.id.name for buffer in spec.map_fold_intermediate), ("tmp",))
        self.assertEqual(len(spec.map_fold_work.items), 2)
        assert spec.map_finalize_work is not None
        self.assertEqual(len(spec.map_finalize_work.items), 1)

    def test_partition_axis_is_valid_in_execution_schedules(self):
        spec = add_spec()
        partial_schedule, combine_schedule = partial_schedules_for(spec)
        partition_axis = spec.fold.partition_axis
        self.assertEqual(partial_schedule.program(partition_axis), 2)
        self.assertEqual(combine_schedule.loop, partition_axis.id)

        partial = MapFoldPartial.make(spec, partial_schedule)
        fold = Fold(spec, combine_schedule, partition_axis, partial.partials).build()
        scan = Scan.make(
            spec,
            combine_schedule,
            scan_axis=partition_axis,
            inputs=partial.partials,
        ).build()
        self.assertEqual(fold.partition_axis, partition_axis)
        self.assertEqual(scan.partition_axis, partition_axis)
        self.assertIsNotNone(scan.carriers)

    def test_buffer_and_basic_stage_construction(self):
        input_buffers = bundle_spec(
            Input,
            ctx=buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
            trg=buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
            targets=buffer_spec("b", ct.int32, default=-100),
        )
        output_buffers = bundle_spec(
            Output,
            z=buffer_spec("b", ct.float32, default=float("-inf")),
            l=buffer_spec("b", ct.float32, default=0),
        )
        grad_storage = input_buffers.as_grad()
        self.assertEqual(
            tuple(axis.name for axis in grad_storage.contention_axes(Axes.make("b v"))),
            ("b", "v"),
        )
        self.assertTrue(output_buffers.as_output_grad())

        spec = xentropy_spec()
        full_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16},
            tiles={"b": 4, "v": 8, "d": 16},
            loop=spec.fold,
        )
        full = MapFold(spec, full_schedule).build()
        self.assertEqual(full.stage.name, "map_fold")
        self.assertEqual(full.stage.cost.materialized_storage_bytes, 0)

        map_stage = Map(
            name="output_project",
            schedule=full_schedule,
            axes=spec.batch,
            buffer_uses=(
                BufferUse.read_resident(spec.semantic),
                BufferUse.write(spec.semantic),
            ),
        ).build()
        self.assertEqual(map_stage.stage.name, "output_project")
        self.assertIsNone(map_stage.domain.loop_axis)
        self.assertEqual(map_stage.domain.task_grid, (2,))

    def test_forward_backward_and_scan_stage_construction(self):
        spec = xentropy_spec()
        general_spec = xentropy_spec(algebra=AlgebraKind.general)
        partition_axis = spec.fold.partition_axis
        full_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16},
            tiles={"b": 4, "v": 8, "d": 16},
            loop=spec.fold,
        )
        partial_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16},
            tiles={"b": 4, "v": 8, "d": 16},
            programs={partition_axis: 2},
            loop=spec.fold,
        )
        combine_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16, partition_axis: 2},
            tiles={"b": 4, "d": 16, partition_axis: 1},
            programs={partition_axis: 1},
            loop=partition_axis,
        )
        partial = MapFoldPartial.make(spec, partial_schedule)
        fwd_fold = partial.build()
        fwd_combine = Fold(
            spec, combine_schedule, partition_axis, partial.partials,
        ).build()

        buffer_use_stage_buffers = bind_buffer_uses(
            fwd_fold.stage.domain,
            (
                BufferUse.read_resident(spec.input),
                BufferUse.write(
                    partial.partials,
                    BufferStorage.Materialized,
                    axis_map={partition_axis.id: spec.fold.id},
                ),
            ),
        )
        self.assertTrue(buffer_use_stage_buffers.read)
        self.assertTrue(buffer_use_stage_buffers.write.materialized)

        scan = Scan.make(
            general_spec,
            combine_schedule,
            scan_axis=partition_axis,
            inputs=partial.partials,
        ).build()
        self.assertEqual(scan.stage.name, "scan")
        self.assertIsNotNone(scan.carriers)
        self.assertGreater(scan.stage.cost.materialized_storage_bytes, 0)

        backward_full = RecomputeMapFinalizeGradWrite(
            spec,
            full_schedule,
            global_buffers=spec.semantic,
            output_grad=spec.semantic.as_output_grad(),
        ).build()
        self.assertEqual(backward_full.stage.name, "recompute_map_finalize_grad_write")
        self.assertTrue(backward_full.stage.cost.write_traffic.has(ATOMIC_ADD))

        backward_partitioned = RecomputeMapFinalizeGradWrite(
            spec,
            partial_schedule,
            global_buffers=spec.semantic,
            output_grad=spec.semantic.as_output_grad(),
            partition_axis=partition_axis,
        ).build()
        self.assertEqual(backward_partitioned.partition_axis, partition_axis)

        general_full = RecomputeFoldMapFinalizeGradWrite(
            general_spec,
            full_schedule,
            global_buffers=general_spec.semantic,
            output_grad=general_spec.semantic.as_output_grad(),
        ).build()
        self.assertEqual(general_full.stage.name, "recompute_fold_map_finalize_grad_write")
        self.assertIsNone(general_full.checkpoints)

        carriers = scan.carriers
        self.assertIsNotNone(carriers)
        assert carriers is not None
        general_prefix = RecomputePrefixFoldMapFinalizeGradWrite(
            general_spec,
            partial_schedule,
            global_buffers=general_spec.semantic,
            output_grad=general_spec.semantic.as_output_grad(),
            prefix=carriers,
            prefix_axis=partition_axis,
        ).build()
        self.assertEqual(
            general_prefix.stage.name,
            "recompute_prefix_fold_map_finalize_grad_write",
        )
        self.assertEqual(general_prefix.checkpoints, scan.carriers)
        self.assertTrue(general_prefix.stage.read_buffers.materialized)
        self.assertEqual(fwd_combine.stage.domain.task_grid, (2, 1))

    def test_compile_stage_builders_without_launching_cuda(self):
        spec = add_spec()
        schedule = schedule_for(spec)
        full = MapFold(spec, schedule).build()
        self.assertIsNotNone(full.compile(fold_functions(add_map_fold, add_combine)))
        self.assertIsNotNone(
            full.compile(fold_functions(add_map_fold_sum_with_named_tid, add_combine))
        )

        partial_schedule, combine_schedule = partial_schedules_for(spec)
        partial = MapFoldPartial.make(spec, partial_schedule)
        partial_stage = partial.build()
        self.assertIsNotNone(
            partial_stage.compile(fold_functions(add_map_fold, add_combine))
        )
        fold_stage = Fold(
            spec, combine_schedule, spec.fold.partition_axis, partial.partials,
        ).build()
        self.assertIsNotNone(fold_stage.compile(fold_functions(combine=add_combine)))

        backward = commutative_backward_stage(spec, schedule)
        self.assertIsNotNone(
            backward.compile(
                fold_functions(
                    embed=add_embed,
                    map_finalize=add_map_finalize_commutative,
                )
            )
        )
        general_backward = RecomputeFoldMapFinalizeGradWrite(
            spec,
            schedule,
            global_buffers=spec.semantic,
            output_grad=spec.semantic.as_output_grad(),
        ).build()
        self.assertIsNotNone(
            general_backward.compile(
                fold_functions(embed=add_embed, map_finalize=add_map_finalize)
            )
        )

        scan = Scan.make(
            spec,
            combine_schedule,
            scan_axis=spec.fold.partition_axis,
            inputs=partial.partials,
        ).build()
        carriers = scan.carriers
        self.assertIsNotNone(carriers)
        assert carriers is not None
        prefix_backward = RecomputePrefixFoldMapFinalizeGradWrite(
            spec,
            partial_schedule,
            global_buffers=spec.semantic,
            output_grad=spec.semantic.as_output_grad(),
            prefix=carriers,
            prefix_axis=spec.fold.partition_axis,
        ).build()
        self.assertTrue(prefix_backward.stage.read_buffers.materialized)
        self.assertIsNotNone(
            prefix_backward.compile(
                fold_functions(embed=add_embed, map_finalize=add_map_finalize)
            )
        )

    def test_persistent_and_streamed_buffer_classification(self):
        spec = scaled_add_spec()
        schedule = schedule_for(spec)
        stage = MapFold(spec, schedule).build()
        self.assertEqual(tuple(b.id.name for b in stage.stage.read_buffers.streamed), ("x",))
        self.assertEqual(
            tuple(b.id.name for b in stage.stage.read_buffers.persistent),
            ("scale",),
        )
        self.assertIsNotNone(
            stage.compile(fold_functions(scaled_add_map_fold_sum, add_combine))
        )

        backward = commutative_backward_stage(spec, schedule)
        self.assertEqual(
            tuple(b.id.name for b in backward.stage.write_buffers.streamed),
            ("x",),
        )
        self.assertEqual(
            tuple(b.id.name for b in backward.stage.write_buffers.persistent),
            ("scale",),
        )
        self.assertIsNotNone(
            backward.compile(
                fold_functions(embed=add_embed, map_finalize=scaled_add_map_finalize)
            )
        )

        batch_loop_schedule = StageSchedule.make(
            spec,
            extents={"b": 4, "v": 8},
            tiles={"b": 2, "v": 4},
            loop="b",
        )
        batch_loop_stage = commutative_backward_stage(spec, batch_loop_schedule)
        self.assertEqual(tuple(batch_loop_stage.stage.read_buffers.persistent), ())
        self.assertEqual(tuple(batch_loop_stage.stage.write_buffers.persistent), ())
        self.assertIsNotNone(
            batch_loop_stage.compile(
                fold_functions(embed=add_embed, map_finalize=scaled_add_map_finalize)
            )
        )

    def test_costs_atomic_evaluation_and_sweep(self):
        spec = xentropy_spec()
        partition_axis = spec.fold.partition_axis
        partial_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16},
            tiles={"b": 4, "v": 8, "d": 16},
            programs={partition_axis: 2},
            loop=spec.fold,
        )
        partial = MapFoldPartial.make(spec, partial_schedule)
        fold_stage = partial.build().stage

        self.assertEqual(fold_stage.cost.write_traffic, fold_stage.write_buffers.accessed_bytes)
        self.assertEqual(fold_stage.cost.effective_traffic, fold_stage.cost.traffic)
        self.assertEqual(
            fold_stage.cost.partial_storage_ratio,
            fold_stage.cost.materialized_storage_bytes
            / fold_stage.cost.ordinary_storage_bytes,
        )
        self.assertEqual(fold_stage.cost.work_efficiency, fold_stage.cost.mma_efficiency)

        atomic_stage = replace(fold_stage, write_model=atomic_add_write)
        self.assertEqual(atomic_stage.cost.write_traffic, fold_stage.write_buffers.accessed_bytes)

        contended_grad_buffers = KernelBuffers.make(
            spec.semantic,
            fold_stage.domain,
            WRITE,
            BufferStorage.Ordinary,
        )
        contended_stage = replace(
            fold_stage,
            buffers=contended_grad_buffers,
            write_model=atomic_add_write,
        )
        self.assertTrue(contended_stage.cost.write_traffic.has(ATOMIC_ADD))
        atomic_eval = evaluate_stage(
            contended_stage,
            attributes=("write_traffic",),
            configs=pl.DataFrame({
                "cfg:SM_COUNT": [1],
                "cfg:SMEM_PER_SM": [1024 * 1024],
                "cfg:MAX_PROGRAMS_PER_SM": [8],
            }),
        )
        self.assertEqual(atomic_eval.height, 1)

        partition_count = sympy.Symbol("partition_count")
        symbolic_schedule = StageSchedule.make(
            spec,
            extents={"b": 8, "v": 17, "d": 16},
            tiles={"b": 4, "v": 8, "d": 16},
            programs={partition_axis: partition_count},
            loop=spec.fold,
        )
        symbolic_stage = MapFoldPartial.make(spec, symbolic_schedule).build().stage
        configs = pl.DataFrame({"cfg:partition_count": [1, 2, 4]})
        chunks = tuple(
            chunked_evaluate_stage(
                symbolic_stage,
                attributes=("partial_storage_ratio", "write_traffic"),
                configs=configs,
                chunk_size=2,
            )
        )
        self.assertEqual(tuple(chunk.height for chunk in chunks), (2, 1))
        evaluated = evaluate_stage(
            symbolic_stage,
            attributes=("partial_storage_ratio", "write_traffic"),
            configs=configs,
            chunk_size=2,
        )
        self.assertEqual(evaluated.height, 3)
        scalar = symbolic_stage.cost.partial_storage_ratio.subs({partition_count: 2})
        vector = evaluated.filter(
            pl.col("cfg:partition_count") == 2
        )["partial_storage_ratio"][0]
        self.assertEqual(vector, float(scalar))

        fold_sweep = sweep_commutative_fold(
            spec,
            sizes={"b": 16, "v": 17, "d": 64},
            hardware=SPECMAP["l4"],
            max_tile=16,
            max_partition_count=2,
        )
        self.assertFalse(fold_sweep.is_empty())
        self.assertIn("single", set(fold_sweep["path"]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class FoldCudaRuntimeTests(unittest.TestCase):
    def test_add_forward_variants(self):
        spec = add_spec()
        schedule = schedule_for(spec)
        partial_schedule, combine_schedule = partial_schedules_for(spec)
        x = torch.randn(4, 8, device="cuda")
        functions = fold_functions(add_map_fold_sum, add_combine)

        full_forward = mk_fold_forward(full_fold_plan(spec, schedule), functions)
        y_full, = full_forward(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_full, x.sum(dim=1))

        double_forward = mk_fold_forward(
            full_fold_plan(spec, schedule),
            fold_functions(add_map_fold_sum, add_combine, to_output=double_output),
        )
        y_double, = double_forward(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_double, 2 * x.sum(dim=1))

        named_tid_forward = mk_fold_forward(
            full_fold_plan(spec, schedule),
            fold_functions(add_map_fold_sum_with_named_tid, add_combine),
        )
        y_named_tid, = named_tid_forward(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_named_tid, x.sum(dim=1))

        partial_forward = mk_fold_forward(
            partial_fold_plan(spec, partial_schedule, combine_schedule),
            functions,
        )
        y_partial, = partial_forward(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_partial, x.sum(dim=1))

    def test_scaled_add_forward_variants(self):
        spec = scaled_add_spec()
        schedule = schedule_for(spec)
        partial_schedule, combine_schedule = partial_schedules_for(spec)
        x = torch.randn(4, 8, device="cuda")
        scale = torch.randn(4, device="cuda")
        functions = fold_functions(scaled_add_map_fold_sum, add_combine)

        full_forward = mk_fold_forward(full_fold_plan(spec, schedule), functions)
        y_full, = full_forward(x, scale)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_full, (x * scale[:, None]).sum(dim=1))

        partial_forward = mk_fold_forward(
            partial_fold_plan(spec, partial_schedule, combine_schedule),
            functions,
        )
        y_partial, = partial_forward(x, scale)
        torch.cuda.synchronize()
        torch.testing.assert_close(y_partial, (x * scale[:, None]).sum(dim=1))

    def test_add_autograd_variants(self):
        spec = add_spec()
        schedule = schedule_for(spec)
        x = torch.randn(4, 8, device="cuda")

        x_for_grad = x.detach().clone().requires_grad_()
        add_autograd = mk_fold_autograd(
            full_fold_plan(spec, schedule, backward_schedule=schedule),
            fold_functions(
                add_map_fold_sum,
                add_combine,
                embed=add_embed,
                map_finalize=add_map_finalize_commutative,
            ),
        )
        y_add, = add_autograd(x_for_grad)
        y_add.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(x_for_grad.grad, torch.ones_like(x_for_grad))

        x_for_double_grad = x.detach().clone().requires_grad_()
        double_autograd = mk_fold_autograd(
            full_fold_plan(spec, schedule, backward_schedule=schedule),
            fold_functions(
                add_map_fold_sum,
                add_combine,
                to_output=double_output,
                embed=add_embed,
                map_finalize=add_map_finalize_commutative,
            ),
        )
        y_double, = double_autograd(x_for_double_grad)
        y_double.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x_for_double_grad.grad,
            2 * torch.ones_like(x_for_double_grad),
        )

    def test_scaled_add_autograd(self):
        spec = scaled_add_spec()
        schedule = schedule_for(spec)
        partial_schedule, combine_schedule = partial_schedules_for(spec)
        x = torch.randn(4, 8, device="cuda")
        scale = torch.randn(4, device="cuda")
        x_for_grad = x.detach().clone().requires_grad_()
        scale_for_grad = scale.detach().clone().requires_grad_()

        autograd = mk_fold_autograd(
            partial_fold_plan(
                spec,
                partial_schedule,
                combine_schedule,
                backward_schedule=schedule,
            ),
            fold_functions(
                scaled_add_map_fold_sum,
                add_combine,
                embed=add_embed,
                map_finalize=scaled_add_map_finalize,
            ),
        )
        y_scaled, = autograd(x_for_grad, scale_for_grad)
        y_scaled.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            x_for_grad.grad,
            scale_for_grad.detach()[:, None].expand_as(x_for_grad),
        )
        torch.testing.assert_close(scale_for_grad.grad, x_for_grad.detach().sum(dim=1))


if __name__ == "__main__":
    unittest.main()
