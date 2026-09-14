from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cuda.tile as ct

from cutilereduce.core.axis import Axes
from cutilereduce.core.stage_domain import AxisRole
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule
from cutilereduce.stages.codegen import full_tuple, make_buffer_helper, require_resolved, scatter_tuple
from cutilereduce.stages.map import Map

if TYPE_CHECKING:
    from cutilereduce.fold.plan import FoldSpec


def embed_axes(spec):
    axes = Axes(values=())
    for buffer in (*spec.semantic, *spec.embedded):
        axes = axes | buffer.axes
    return axes


@dataclass(frozen=True)
class Embed:
    spec: FoldSpec
    schedule: StageSchedule

    def build(self) -> BuiltStage:
        if not self.spec.embedded:
            raise ValueError("backward requires declared embedded buffers")
        axes = embed_axes(self.spec)
        batch = tuple(axis for axis in axes if axis in self.spec.batch)
        for buffer in self.spec.embedded:
            if any(axis not in buffer.axes for axis in batch):
                raise ValueError("embedded outputs must include every embed batch axis")
        stage = Map(
            name="embed",
            schedule=self.schedule,
            axes=axes,
            roles={axis.id: AxisRole.Batch if axis in batch else AxisRole.Inner for axis in axes},
            buffer_uses=(
                BufferUse.read_resident(self.spec.semantic),
                BufferUse.read_resident(self.spec.semantic.as_output_grad()),
                BufferUse.write(self.spec.embedded),
                BufferUse.resident(self.spec.embedded),
            ),
        ).build()
        return replace(stage, compiler=compile_embed_stage)


def default_embed_stage(spec, backward_stage):
    """A standalone batch tiling, also used for explicitly constructed plans."""
    axes = embed_axes(spec)
    extents = {axis.id: backward_stage.domain.get(axis).extent for axis in axes}
    tiles = {axis.id: 1 << (int(extents[axis.id]) - 1).bit_length() for axis in axes}
    budget = 128
    for axis in reversed(tuple(axes)):
        if axis in spec.batch:
            tile = min(tiles[axis.id], budget)
            tiles[axis.id] = tile
            budget = max(1, budget // tile)
    return Embed(spec, StageSchedule(extents=extents, tiles=tiles, programs={})).build()


def compile_embed_stage(stage, functions, *, programmatic_dependent_launch=False):
    require_resolved(stage)
    embed = functions.embed
    if embed is None:
        raise ValueError("embed stage requires an embed callback")
    load = make_buffer_helper(stage.stage.read_buffers).load
    store = make_buffer_helper(stage.stage.write_buffers).store
    task_grid = tuple(int(n) for n in stage.domain.task_grid)
    rank = len(stage.domain.compute_axes)
    program_to_compute = tuple(
        stage.domain.index(axis.source) for axis in stage.domain.program_axes
    )
    output_count = len(stage.stage.write_buffers)

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        if programmatic_dependent_launch:
            ct.grid_dependency_control_launch_dependents()
        pid = ct.bid(0)
        program_tid = ()
        for size in ct.static_iter(task_grid):
            program_tid += (pid % size,)
            pid = pid // size
        tid = scatter_tuple(program_tid, full_tuple(0, rank), program_to_compute)
        values = embed(*load(tid, read_buffers))
        ct.static_assert(len(values) == output_count, "embed result count must match embedded buffers")
        store(tid, write_buffers, values)

    return kernel
