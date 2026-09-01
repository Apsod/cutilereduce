from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage, StageAccess
from cutilereduce.core.stage_domain import AxisRole, ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses
from cutilereduce.stages.codegen import (
    StageFunctions,
    identity,
    make_buffer_helper,
    make_buffer_split,
    stage_grid_info,
)
from cutilereduce.stages.map_fold import batch_program_axes, fold_compute_axes


@dataclass(frozen=True)
class Scan:
    spec: object
    schedule: StageSchedule
    scan_axis: Axis
    inputs: BufferBundle
    carriers: BufferBundle
    outputs: BufferBundle | None = None
    combine: object | None = None
    initial: object | None = None
    to_semantic: object | None = None
    exclusive: bool = False

    @classmethod
    def make(
            cls,
            spec,
            schedule: StageSchedule,
            *,
            scan_axis: Axis,
            inputs: BufferBundle,
            outputs: BufferBundle | None = None,
            combine=None,
            initial=None,
            to_semantic=None,
            exclusive: bool = False,
            ) -> Scan:
        return cls(
            spec=spec,
            schedule=schedule,
            scan_axis=scan_axis,
            inputs=inputs,
            carriers=inputs,
            outputs=outputs,
            combine=combine,
            initial=initial,
            to_semantic=to_semantic,
            exclusive=exclusive,
        )

    def build(self) -> BuiltStage:
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.scan_axis,
            fold_extent=self.schedule.extent(self.scan_axis),
            fold_tile=self.schedule.tile(self.scan_axis),
            fold_role=AxisRole.Scan,
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        program_axes.append(ProgramAxis(
            axis=self.scan_axis,
            source=self.scan_axis.id,
            programs=self.schedule.program(self.scan_axis, 1),
        ))
        domain = StageDomain(
            name="scan",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.scan_axis.id,
        )
        uses = [BufferUse(
            bundle=self.carriers,
            access=StageAccess.READ | StageAccess.WRITE | StageAccess.RESIDENT,
            storage=BufferStorage.Materialized,
        )]
        if self.outputs is not None:
            uses.append(BufferUse.write(self.outputs))
        buffers = bind_buffer_uses(domain, tuple(uses))
        return BuiltStage(
            stage=KernelStage("scan", domain, buffers, self.spec.combine_work),
            partials=self.inputs,
            carriers=self.carriers,
            partition_axis=self.scan_axis,
            compiler=lambda stage, functions: compile_scan_stage(
                stage,
                functions,
                combine=self.combine,
                initial=self.initial,
                to_semantic=self.to_semantic,
                exclusive=self.exclusive,
            ),
        )


def make_scan_program(stage: BuiltStage, combine, carriers: BufferBundle, initial=None, to_semantic=None, *, exclusive=False):
    grid = stage_grid_info(stage)
    read = make_buffer_helper(stage.stage.read_buffers)
    all_writes = tuple(stage.stage.write_buffers)
    carrier_writes = tuple(buffer for buffer in all_writes if buffer.id in carriers)
    output_writes = tuple(buffer for buffer in all_writes if buffer.id not in carriers)
    has_outputs = bool(output_writes)
    write_split = make_buffer_split(all_writes, carrier_writes, output_writes)
    split = write_split.functions
    write_carrier = make_buffer_helper(carrier_writes)
    write_output = make_buffer_helper(output_writes)
    to_semantic = to_semantic or identity

    if stage.partition_axis is None:
        raise ValueError("scan stage has no partition axis")
    scan_axis = stage.partition_axis.id
    carrier_shapes = tuple(
        buffer.tile.shape[:position] + buffer.tile.shape[position + 1:]
        for buffer in carrier_writes
        for position in (
            tuple(axis.id for axis in buffer.storage_axes).index(scan_axis),
        )
    )

    def execution_carrier(materialized_carrier):
        """Remove the materialized scan coordinate from user-facing values."""
        ret = ()
        for i, shape in ct.static_iter(enumerate(carrier_shapes)):
            ret += (ct.reshape(materialized_carrier[i], shape),)
        return ret

    def load_carrier(stage_tid, read_buffers):
        return execution_carrier(read.load(stage_tid, read_buffers))

    def init_carrier():
        return execution_carrier(write_carrier.init())

    if exclusive:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, acc, read_buffers, carrier_buffers):
            value = load_carrier(loop_stage_tid, read_buffers)
            write_carrier.store(loop_stage_tid, carrier_buffers, acc)
            return combine(*acc, *value)
    else:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, acc, read_buffers, carrier_buffers):
            value = load_carrier(loop_stage_tid, read_buffers)
            acc = combine(*acc, *value)
            write_carrier.store(loop_stage_tid, carrier_buffers, acc)
            return acc

    if exclusive:
        loop = grid.loop_with_tail(loop_body, start=0)
        initial_fn = init_carrier if initial is None else initial

        @ct.function
        def scan_program(tid, program_tid, read_buffers, write_buffers):
            carrier_buffers = split.left(write_buffers)
            output_buffers = split.right(write_buffers)
            acc = initial_fn()
            acc = loop(tid, program_tid, acc, read_buffers, carrier_buffers)
            if has_outputs:
                stage_tid = tid + program_tid
                write_output.store(stage_tid, output_buffers, to_semantic(*acc))
            return acc
    elif initial is None:
        loop = grid.loop_with_tail(loop_body, start=1)

        @ct.function
        def scan_program(tid, program_tid, read_buffers, write_buffers):
            carrier_buffers = split.left(write_buffers)
            output_buffers = split.right(write_buffers)
            loop_offset, _, _ = grid.loop_offset_and_size(program_tid)
            loop_tid = grid.set_loop_index(tid, loop_offset)
            acc = load_carrier(loop_tid + program_tid, read_buffers)
            write_carrier.store(loop_tid + program_tid, carrier_buffers, acc)
            acc = loop(tid, program_tid, acc, read_buffers, carrier_buffers)
            if has_outputs:
                stage_tid = tid + program_tid
                write_output.store(stage_tid, output_buffers, to_semantic(*acc))
            return acc
    else:
        loop = grid.loop_with_tail(loop_body, start=0)

        @ct.function
        def scan_program(tid, program_tid, read_buffers, write_buffers):
            carrier_buffers = split.left(write_buffers)
            output_buffers = split.right(write_buffers)
            acc = loop(tid, program_tid, initial(), read_buffers, carrier_buffers)
            if has_outputs:
                stage_tid = tid + program_tid
                write_output.store(stage_tid, output_buffers, to_semantic(*acc))
            return acc

    return scan_program


def compile_scan_stage(
        stage: BuiltStage,
        functions: StageFunctions | None = None,
        *,
        combine=None,
        initial=None,
        to_semantic=None,
        exclusive=False,
        ):
    grid = stage_grid_info(stage)
    if functions is not None:
        combine = combine or functions.combine
        to_semantic = to_semantic or functions.to_semantic
    scan_program = make_scan_program(
        stage,
        combine,
        stage.carriers,
        initial=initial,
        to_semantic=to_semantic,
        exclusive=exclusive,
    )

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        scan_program(tid, program_tid, read_buffers, write_buffers)

    return kernel


__all__ = [
    "Scan",
    "compile_scan_stage",
    "make_scan_program",
]
