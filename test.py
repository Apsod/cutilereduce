from dataclasses import replace

import polars as pl
import sympy
import torch

from cutilereduce.core import (
    ATOMIC_ADD,
    MatMulWork,
    WorkModel,
    atomic_add_write,
    chunked_evaluate_stage,
    evaluate_stage,
)
from cutilereduce.core.stage_buffer import KernelBuffers, WRITE, BufferStorage
from cutilereduce.core.buffer import buffer_spec, bundle_spec, Internal, Input, Output
from cutilereduce.fold import AlgebraKind, fold_functions, mk_fold_autograd, mk_fold_forward, StageSchedule, make_fold_spec
from cutilereduce.fold.commutative import commutative_backward_stage, full_fold_plan, partial_fold_plan, sweep_commutative_fold
from cutilereduce.stages import (
    BufferUse,
    Fold,
    Map,
    MapFold,
    MapFoldPartial,
    RecomputeFinalizeGradWrite,
    RecomputeFoldFinalizeGradWrite,
    RecomputePrefixFoldFinalizeGradWrite,
    Scan,
    bind_buffer_uses,
)

import cuda.tile as ct


@ct.function
def add_map_reduce(tid, x):
    return (x,)


@ct.function
def add_map_reduce_sum(tid, x):
    return (ct.sum(x, axis=1),)


@ct.function
def add_map_reduce_sum_with_named_tid(tid, x):
    v = tid.indices("v")
    mask = tid.mask("v")
    return (ct.sum(ct.where(mask[None, :], x + v[None, :] * 0, 0), axis=1),)


@ct.function
def add_combine(a, b):
    return (a + b,)


@ct.function
def scaled_add_map_reduce_sum(tid, x, scale):
    return (ct.sum(x * scale[:, None], axis=1),)


@ct.function
def add_embed(y, g_y):
    return (g_y,)


@ct.function
def add_finalize(tid, x, g_x, g_y):
    return (g_x + g_y[:, None],)


@ct.function
def add_map_reduce_backward(tid, x, g_x, g_y, prefix):
    return (
        (g_x + g_y[:, None],),
        (prefix + ct.sum(x, axis=1),),
    )


@ct.function
def scaled_add_finalize(tid, x, scale, g_x, g_scale, g_y):
    return (
        g_x + scale[:, None] * g_y[:, None],
        g_scale + ct.sum(x * g_y[:, None], axis=1),
    )


def double_output(y):
    return y * 2

input = bundle_spec(
    Input,
    ctx=buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
    trg=buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
    targets=buffer_spec("b", ct.int32, req_grad=False, default=-100),
)

output = bundle_spec(
    Output,
    z = buffer_spec('b', ct.float32, default=float('-inf')),
    l = buffer_spec('b', ct.float32, default=0),
)

output_grad = output.as_output_grad()

grad_storage = input.as_grad()

grad_accumulator = bundle_spec(
    Internal('grad_acc'),
    z = buffer_spec('b', ct.float32, default=float('-inf')),
    g_z = buffer_spec('b', ct.float32, default=0),
    g_l = buffer_spec('b', ct.float32, default=0),
)

execution = bundle_spec(
    Internal('execution'),
    m = buffer_spec('b', ct.float32, default=float('-inf')),
    e = buffer_spec('b', ct.float32, default=0),
    u = buffer_spec('b', ct.float32, default=0),
)

intermediate = bundle_spec(
    Internal('intermediate'),
    logits = buffer_spec('b v', ct.float32)
)

for x in input, output, grad_accumulator, execution, intermediate, grad_storage:
    print(x)

fold_spec = make_fold_spec(
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
    output={
        "z": buffer_spec("b", ct.float32, default=float("-inf")),
        "l": buffer_spec("b", ct.float32, default=0),
    },
    batch="b",
    fold="v",
    map_fold_work=WorkModel.make(MatMulWork.make(M="b", N="v", K="d")),
)

full_schedule = StageSchedule.make(
    fold_spec,
    extents={"b": 8, "v": 17, "d": 16},
    tiles={"b": 4, "v": 8, "d": 16},
    loop=fold_spec.fold,
)
fwd_full = MapFold(fold_spec, full_schedule).build()
assert fwd_full.stage.name == "map_fold"
assert fwd_full.stage.cost.materialized_storage_bytes == 0

partition_axis = fold_spec.fold.partition_axis
partial_schedule = StageSchedule.make(
    fold_spec,
    extents={"b": 8, "v": 17, "d": 16},
    tiles={"b": 4, "v": 8, "d": 16},
    programs={partition_axis: 2},
    loop=fold_spec.fold,
)
partial = MapFoldPartial.make(fold_spec, partial_schedule)
fwd_fold = partial.build()
combine_schedule = StageSchedule.make(
    fold_spec,
    extents={"b": 8, "v": 17, "d": 16, partition_axis: 2},
    tiles={"b": 4, "d": 16, partition_axis: 1},
    programs={partition_axis: 1},
    loop=partition_axis,
)
fwd_combine = Fold(fold_spec, combine_schedule, partition_axis, partial.partials).build()

buffer_use_stage_buffers = bind_buffer_uses(fwd_fold.stage.domain, (
    BufferUse.read_resident(fold_spec.input),
    BufferUse.write(partial.partials, BufferStorage.Materialized, axis_map={partition_axis.id: fold_spec.fold.id}),
))
assert buffer_use_stage_buffers.read
assert buffer_use_stage_buffers.write.materialized

map_stage = Map(
    name="output_project",
    schedule=full_schedule,
    axes=fold_spec.batch,
    buffer_uses=(
        BufferUse.read_resident(fold_spec.output),
        BufferUse.write(fold_spec.output),
    ),
).build()
assert map_stage.stage.name == "output_project"
assert map_stage.domain.loop_axis is None
assert map_stage.domain.task_grid == (2,)

general_spec = make_fold_spec(
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
    output={
        "z": buffer_spec("b", ct.float32, default=float("-inf")),
        "l": buffer_spec("b", ct.float32, default=0),
    },
    batch="b",
    fold="v",
    map_fold_work=WorkModel.make(MatMulWork.make(M="b", N="v", K="d")),
    algebra=AlgebraKind.general,
)
scan_stage = Scan.make(
    general_spec,
    combine_schedule,
    scan_axis=partition_axis,
    inputs=partial.partials,
).build()
assert scan_stage.stage.name == "scan"
assert scan_stage.carriers is not None
assert scan_stage.stage.cost.materialized_storage_bytes > 0

bwd_full = RecomputeFinalizeGradWrite(
    fold_spec,
    full_schedule,
    global_buffers=fold_spec.output,
    output_grad=output_grad,
).build()
assert bwd_full.stage.name == "recompute_finalize_grad_write"
assert bwd_full.stage.cost.write_traffic.has(ATOMIC_ADD)

bwd_partitioned = RecomputeFinalizeGradWrite(
    fold_spec,
    partial_schedule,
    global_buffers=fold_spec.output,
    output_grad=output_grad,
    partition_axis=partition_axis,
).build()
assert bwd_partitioned.partition_axis == partition_axis

bwd_general_full = RecomputeFoldFinalizeGradWrite(
    general_spec,
    full_schedule,
    global_buffers=general_spec.output,
    output_grad=general_spec.output.as_output_grad(),
).build()
assert bwd_general_full.stage.name == "recompute_fold_finalize_grad_write"
assert bwd_general_full.checkpoints is None

bwd_general = RecomputePrefixFoldFinalizeGradWrite(
    general_spec,
    partial_schedule,
    global_buffers=general_spec.output,
    output_grad=general_spec.output.as_output_grad(),
    prefix=scan_stage.carriers,
    prefix_axis=partition_axis,
).build()
assert bwd_general.stage.name == "recompute_prefix_fold_finalize_grad_write"
assert bwd_general.checkpoints == scan_stage.carriers
assert bwd_general.stage.read_buffers.materialized

add_spec = make_fold_spec(
    input={
        "x": buffer_spec("b v", ct.float32, req_grad=True, default=0),
    },
    execution={
        "acc": buffer_spec("b", ct.float32, default=0),
    },
    output={
        "y": buffer_spec("b", ct.float32, default=0),
    },
    batch="b",
    fold="v",
)
add_schedule = StageSchedule.make(
    add_spec,
    extents={"b": 4, "v": 8},
    tiles={"b": 2, "v": 4},
    loop=add_spec.fold,
)
add_fwd = MapFold(add_spec, add_schedule).build()
add_kernel = add_fwd.compile(fold_functions(add_map_reduce, add_combine))
assert add_kernel is not None
add_named_tid_kernel = add_fwd.compile(fold_functions(add_map_reduce_sum_with_named_tid, add_combine))
assert add_named_tid_kernel is not None

add_partition_axis = add_spec.fold.partition_axis
add_partial_schedule = StageSchedule.make(
    add_spec,
    extents={"b": 4, "v": 8},
    tiles={"b": 2, "v": 2},
    programs={add_partition_axis: 2},
    loop=add_spec.fold,
)
add_partial = MapFoldPartial.make(add_spec, add_partial_schedule)
add_partial_stage = add_partial.build()
add_partial_kernel = add_partial_stage.compile(fold_functions(add_map_reduce, add_combine))
assert add_partial_kernel is not None

add_combine_schedule = StageSchedule.make(
    add_spec,
    extents={"b": 4, "v": 8, add_partition_axis: 2},
    tiles={"b": 2, add_partition_axis: 2},
    programs={add_partition_axis: 1},
    loop=add_partition_axis,
)
add_fold_stage = Fold(add_spec, add_combine_schedule, add_partition_axis, add_partial.partials).build()
add_fold_kernel = add_fold_stage.compile(fold_functions(combine=add_combine))
assert add_fold_kernel is not None
add_bwd_stage = commutative_backward_stage(add_spec, add_schedule)
add_bwd_kernel = add_bwd_stage.compile(fold_functions(embed=add_embed, finalize=add_finalize))
assert add_bwd_kernel is not None
add_general_bwd_stage = RecomputeFoldFinalizeGradWrite(
    add_spec,
    add_schedule,
    global_buffers=add_spec.output,
    output_grad=add_spec.output.as_output_grad(),
).build()
add_general_bwd_kernel = add_general_bwd_stage.compile(
    fold_functions(embed=add_embed, map_reduce_backward=add_map_reduce_backward)
)
assert add_general_bwd_kernel is not None
add_scan_stage = Scan.make(
    add_spec,
    add_combine_schedule,
    scan_axis=add_partition_axis,
    inputs=add_partial.partials,
).build()
add_general_prefix_bwd_stage = RecomputePrefixFoldFinalizeGradWrite(
    add_spec,
    add_partial_schedule,
    global_buffers=add_spec.output,
    output_grad=add_spec.output.as_output_grad(),
    prefix=add_scan_stage.carriers,
    prefix_axis=add_partition_axis,
).build()
assert add_general_prefix_bwd_stage.stage.read_buffers.materialized
add_general_prefix_bwd_kernel = add_general_prefix_bwd_stage.compile(
    fold_functions(embed=add_embed, map_reduce_backward=add_map_reduce_backward)
)
assert add_general_prefix_bwd_kernel is not None

scaled_add_spec = make_fold_spec(
    input={
        "x": buffer_spec("b v", ct.float32, req_grad=True, default=0),
        "scale": buffer_spec("b", ct.float32, req_grad=True, default=0),
    },
    execution={
        "acc": buffer_spec("b", ct.float32, default=0),
    },
    output={
        "y": buffer_spec("b", ct.float32, default=0),
    },
    batch="b",
    fold="v",
)
scaled_add_schedule = StageSchedule.make(
    scaled_add_spec,
    extents={"b": 4, "v": 8},
    tiles={"b": 2, "v": 4},
    loop=scaled_add_spec.fold,
)
scaled_add_stage = MapFold(scaled_add_spec, scaled_add_schedule).build()
assert tuple(b.id.name for b in scaled_add_stage.stage.read_buffers.streamed) == ("x",)
assert tuple(b.id.name for b in scaled_add_stage.stage.read_buffers.persistent) == ("scale",)
scaled_add_kernel = scaled_add_stage.compile(fold_functions(scaled_add_map_reduce_sum, add_combine))
assert scaled_add_kernel is not None
scaled_add_bwd_stage = commutative_backward_stage(scaled_add_spec, scaled_add_schedule)
assert tuple(b.id.name for b in scaled_add_bwd_stage.stage.write_buffers.streamed) == ("x",)
assert tuple(b.id.name for b in scaled_add_bwd_stage.stage.write_buffers.persistent) == ("scale",)
scaled_add_bwd_kernel = scaled_add_bwd_stage.compile(fold_functions(embed=add_embed, finalize=scaled_add_finalize))
assert scaled_add_bwd_kernel is not None
scaled_add_bwd_batch_loop_schedule = StageSchedule.make(
    scaled_add_spec,
    extents={"b": 4, "v": 8},
    tiles={"b": 2, "v": 4},
    loop="b",
)
scaled_add_bwd_batch_loop_stage = commutative_backward_stage(scaled_add_spec, scaled_add_bwd_batch_loop_schedule)
assert tuple(b.id.name for b in scaled_add_bwd_batch_loop_stage.stage.read_buffers.persistent) == ()
assert tuple(b.id.name for b in scaled_add_bwd_batch_loop_stage.stage.write_buffers.persistent) == ()
scaled_add_bwd_batch_loop_kernel = scaled_add_bwd_batch_loop_stage.compile(
    fold_functions(embed=add_embed, finalize=scaled_add_finalize)
)
assert scaled_add_bwd_batch_loop_kernel is not None
scaled_add_partition_axis = scaled_add_spec.fold.partition_axis
scaled_add_partial_schedule = StageSchedule.make(
    scaled_add_spec,
    extents={"b": 4, "v": 8},
    tiles={"b": 2, "v": 2},
    programs={scaled_add_partition_axis: 2},
    loop=scaled_add_spec.fold,
)
scaled_add_combine_schedule = StageSchedule.make(
    scaled_add_spec,
    extents={"b": 4, "v": 8, scaled_add_partition_axis: 2},
    tiles={"b": 2, scaled_add_partition_axis: 2},
    programs={scaled_add_partition_axis: 1},
    loop=scaled_add_partition_axis,
)

if torch.cuda.is_available():
    print('cuda is available')
    x = torch.randn(4, 8, device="cuda")
    add_functions = fold_functions(add_map_reduce_sum, add_combine)

    add_full_plan = full_fold_plan(add_spec, add_schedule)
    add_full_forward = mk_fold_forward(add_full_plan, add_functions)
    y_full, = add_full_forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_full, x.sum(dim=1))

    add_double_forward = mk_fold_forward(add_full_plan, fold_functions(add_map_reduce_sum, add_combine, to_output=double_output))
    y_double, = add_double_forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_double, 2 * x.sum(dim=1))

    add_named_tid_forward = mk_fold_forward(add_full_plan, fold_functions(add_map_reduce_sum_with_named_tid, add_combine))
    y_named_tid, = add_named_tid_forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_named_tid, x.sum(dim=1))

    add_partial_plan = partial_fold_plan(add_spec, add_partial_schedule, add_combine_schedule)
    add_partial_forward = mk_fold_forward(add_partial_plan, add_functions)
    y_partial, = add_partial_forward(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_partial, x.sum(dim=1))

    scale = torch.randn(4, device="cuda")
    scaled_add_forward = mk_fold_forward(full_fold_plan(scaled_add_spec, scaled_add_schedule), fold_functions(scaled_add_map_reduce_sum, add_combine))
    y_scaled, = scaled_add_forward(x, scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_scaled, (x * scale[:, None]).sum(dim=1))

    scaled_add_partial_forward = mk_fold_forward(
        partial_fold_plan(scaled_add_spec, scaled_add_partial_schedule, scaled_add_combine_schedule),
        fold_functions(scaled_add_map_reduce_sum, add_combine),
    )
    y_scaled_partial, = scaled_add_partial_forward(x, scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(y_scaled_partial, (x * scale[:, None]).sum(dim=1))

    x_for_grad = x.detach().clone().requires_grad_()
    add_autograd = mk_fold_autograd(
        full_fold_plan(add_spec, add_schedule, backward_schedule=add_schedule),
        fold_functions(add_map_reduce_sum, add_combine, embed=add_embed, finalize=add_finalize),
    )
    y_add, = add_autograd(x_for_grad)
    y_add.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(x_for_grad.grad, torch.ones_like(x_for_grad))

    x_for_double_grad = x.detach().clone().requires_grad_()
    add_double_autograd = mk_fold_autograd(
        full_fold_plan(add_spec, add_schedule, backward_schedule=add_schedule),
        fold_functions(add_map_reduce_sum, add_combine, to_output=double_output, embed=add_embed, finalize=add_finalize),
    )
    y_add_double, = add_double_autograd(x_for_double_grad)
    y_add_double.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(x_for_double_grad.grad, 2 * torch.ones_like(x_for_double_grad))

    x_for_scaled_grad = x.detach().clone().requires_grad_()
    scale_for_grad = scale.detach().clone().requires_grad_()
    scaled_add_autograd = mk_fold_autograd(
        partial_fold_plan(
            scaled_add_spec,
            scaled_add_partial_schedule,
            scaled_add_combine_schedule,
            backward_schedule=scaled_add_schedule,
        ),
        fold_functions(scaled_add_map_reduce_sum, add_combine, embed=add_embed, finalize=scaled_add_finalize),
    )
    y_scaled_auto, = scaled_add_autograd(x_for_scaled_grad, scale_for_grad)
    y_scaled_auto.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(x_for_scaled_grad.grad, scale_for_grad.detach()[:, None].expand_as(x_for_scaled_grad))
    torch.testing.assert_close(scale_for_grad.grad, x_for_scaled_grad.detach().sum(dim=1))
else:
    print('cuda is not available')

print(fwd_fold.stage.domain.task_grid)
print(fwd_fold.stage.cost.traffic)
print(fwd_combine.stage.domain.task_grid)
print(fwd_combine.stage.cost.traffic)

assert fwd_fold.stage.cost.write_traffic == fwd_fold.stage.write_buffers.accessed_bytes
assert fwd_fold.stage.cost.effective_traffic == fwd_fold.stage.cost.traffic
assert fwd_fold.stage.cost.partial_storage_ratio == fwd_fold.stage.cost.materialized_storage_bytes / fwd_fold.stage.cost.ordinary_storage_bytes
assert fwd_fold.stage.cost.work_efficiency == fwd_fold.stage.cost.mma_efficiency

atomic_stage = replace(fwd_fold.stage, write_model=atomic_add_write)
assert atomic_stage.cost.write_traffic == fwd_fold.stage.write_buffers.accessed_bytes

contended_grad_buffers = KernelBuffers.make(
    fold_spec.output,
    fwd_fold.stage.domain,
    WRITE,
    BufferStorage.Ordinary,
)
contended_stage = replace(fwd_fold.stage, buffers=contended_grad_buffers, write_model=atomic_add_write)
assert contended_stage.cost.write_traffic.has(ATOMIC_ADD)
atomic_eval = evaluate_stage(
    contended_stage,
    attributes=("write_traffic",),
    configs=pl.DataFrame({
        "cfg:SM_COUNT": [1],
        "cfg:SMEM_PER_SM": [1024 * 1024],
        "cfg:MAX_PROGRAMS_PER_SM": [8],
    }),
)
assert atomic_eval.height == 1

partition_count = sympy.Symbol("partition_count")
symbolic_schedule = StageSchedule.make(
    fold_spec,
    extents={"b": 8, "v": 17, "d": 16},
    tiles={"b": 4, "v": 8, "d": 16},
    programs={partition_axis: partition_count},
    loop=fold_spec.fold,
)
symbolic_stage = MapFoldPartial.make(fold_spec, symbolic_schedule).build().stage
configs = pl.DataFrame({"cfg:partition_count": [1, 2, 4]})
chunks = tuple(chunked_evaluate_stage(
    symbolic_stage,
    attributes=("partial_storage_ratio", "write_traffic"),
    configs=configs,
    chunk_size=2,
))
assert tuple(chunk.height for chunk in chunks) == (2, 1)
evaluated = evaluate_stage(
    symbolic_stage,
    attributes=("partial_storage_ratio", "write_traffic"),
    configs=configs,
    chunk_size=2,
)
assert evaluated.height == 3
scalar = symbolic_stage.cost.partial_storage_ratio.subs({partition_count: 2})
vector = evaluated.filter(pl.col("cfg:partition_count") == 2)["partial_storage_ratio"][0]
assert vector == float(scalar)

fold_sweep = sweep_commutative_fold(
    fold_spec,
    sizes={"b": 16, "v": 17, "d": 64},
    max_tile=16,
    max_partition_count=2,
)
assert not fold_sweep.is_empty()
assert "single" in set(fold_sweep["path"])
