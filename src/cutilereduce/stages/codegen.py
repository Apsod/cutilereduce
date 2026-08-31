from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cuda.tile as ct

from cutilereduce.core.buffer import BufferId
from cutilereduce.core.stage_buffer import StageBuffer
from cutilereduce.stages.base import BuiltStage


@dataclass(frozen=True)
class StageFunctions:
    map_reduce: Any = None
    combine: Any = None
    to_semantic: Any = None
    to_output: Any = None
    embed: Any = None
    finalize: Any = None
    map_backward: Any = None
    map_reduce_backward: Any = None


class Bundle:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return tuple(getattr(self, k) for k in key)
        return getattr(self, key)


@ct.function
def retile(original, index: ct.Constant[tuple[int, ...]]):
    ret = ()
    for i in ct.static_iter(index):
        ret += (original[i],)
    return ret

@ct.function
def scatter_tuple(source, target, source_to_target: ct.Constant[tuple[int, ...]]):
    ret = target
    for source_i, target_i in ct.static_iter(enumerate(source_to_target)):
        ret = set_at(ret, target_i, source[source_i])
    return ret

@ct.function
def full_tuple(val, n: ct.Constant[int]):
    ret = ()
    for _ in ct.static_iter(range(n)):
        ret += (val,)
    return ret

@ct.function
def merge_tuples(
        left,
        right,
        origin: ct.Constant[tuple[int, ...]],
        index: ct.Constant[tuple[int, ...]],
        ):
    ret = ()
    for side, i in ct.static_iter(zip(origin, index, strict=True)):
        if side == 0:
            ret += (left[i],)
        else:
            ret += (right[i],)
    return ret


@ct.function
def set_at(original, index: ct.Constant[int], value):
    ret = ()
    for i in ct.static_iter(range(len(original))):
        if i == index:
            ret += (value,)
        else:
            ret += (original[i],)
    return ret


def inverse_p(p):
    ret = [None] * len(p)
    for i, j in enumerate(p):
        ret[j] = i
    return tuple(ret)


def ctmap(fun, xs):
    def _ctmap(*args):
        ret = ()
        for x in ct.static_iter(xs):
            ret += (fun(*args, x),)
        return ret

    return _ctmap


def ctzipmap(fun, xs, *, nzips=1):
    def _ctzipmap(*args):
        ret = ()
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i, x in ct.static_iter(enumerate(xs)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            ret += (fun(*static, *current, x),)
        return ret

    return _ctzipmap


def ctzipdo(fun, xs, *, nzips=1):
    def _ctzipdo(*args):
        static = args[:-nzips]
        zipped = args[-nzips:]
        for i, x in ct.static_iter(enumerate(xs)):
            current = ()
            for j in ct.static_iter(range(nzips)):
                current += (zipped[j][i],)
            fun(*static, *current, x)

    return _ctzipdo


def require_resolved(stage: BuiltStage) -> None:
    if not stage.domain.resolved:
        raise ValueError(f"cannot compile unresolved stage domain: {stage.domain}")


def as_int_tuple(values) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


@dataclass(frozen=True)
class StageBufferInfo:
    index: ct.Constant[tuple[int, ...]]
    tile_shape: ct.Constant[tuple[int, ...]]
    padding_mode: ct.Constant[ct.PaddingMode]
    default: ct.Constant[Any]
    dtype: ct.Constant[ct.DType]
    multiplicity: ct.Constant[int]

    @classmethod
    def make(cls, buffer: StageBuffer) -> StageBufferInfo:
        return cls(
            index=as_int_tuple(buffer.storage_index),
            tile_shape=as_int_tuple(buffer.tile.shape),
            padding_mode=buffer.padding_mode,
            default=buffer.default,
            dtype=buffer.dtype,
            multiplicity=math.ceil(buffer.residual_multiplicity),
        )


def make_buffer_helper(buffers):
    infos = tuple(StageBufferInfo.make(b) for b in buffers)
    num = len(infos)

    def _view_load(tid, view, info):
        return view.load(retile(tid, info.index))

    def _view_store_add(tid, view, tile, info):
        if info.multiplicity == 1:
            view.store(retile(tid, info.index), tile)
        else:
            view.atomic_store_add(retile(tid, info.index), tile)

    @dataclass(frozen=True)
    class Views:
        views: tuple[ct.TiledView, ...]

        def load(self, tid):
            return ctzipmap(_view_load, infos)(tid, self.views)

        def store_add(self, tid, tiles):
            ctzipdo(_view_store_add, infos, nzips=2)(tid, self.views, tiles)

    def _mk_view(buffer, info):
        return buffer.tiled_view(info.tile_shape, padding_mode=info.padding_mode)

    def _mk_views(raw_buffers):
        return Views(ctzipmap(_mk_view, infos)(raw_buffers))

    def _load(tid, buffer, info):
        return ct.load(buffer, retile(tid, info.index), info.tile_shape, padding_mode=info.padding_mode)

    def _store(tid, buffer, tile, info):
        ct.store(buffer, retile(tid, info.index), tile)

    def _init(info):
        return ct.full(info.tile_shape, info.default, info.dtype)

    return Bundle(
        view=_mk_views,
        load=ctzipmap(_load, infos),
        store=ctzipdo(_store, infos, nzips=2),
        init=ctmap(_init, infos),
        num=num,
    )


def _buffer_index(buffers: tuple[StageBuffer, ...]) -> dict[BufferId, int]:
    return {buffer.id: i for i, buffer in enumerate(buffers)}


def _buffer_indices(all_buffers: tuple[StageBuffer, ...], selected_buffers: tuple[StageBuffer, ...]) -> tuple[int, ...]:
    index = _buffer_index(all_buffers)
    return tuple(index[buffer.id] for buffer in selected_buffers)


def make_buffer_split(
        all_buffers,
        left_buffers,
        right_buffers,
        ):
    all_buffers = tuple(all_buffers)
    left_buffers = tuple(left_buffers)
    right_buffers = tuple(right_buffers)
    all_ids = tuple(buffer.id for buffer in all_buffers)
    left_index = _buffer_index(left_buffers)
    right_index = _buffer_index(right_buffers)
    left_arg_index = _buffer_indices(all_buffers, left_buffers)
    right_arg_index = _buffer_indices(all_buffers, right_buffers)
    origin = []
    index = []
    for buffer_id in all_ids:
        if buffer_id in left_index:
            origin.append(0)
            index.append(left_index[buffer_id])
        elif buffer_id in right_index:
            origin.append(1)
            index.append(right_index[buffer_id])
        else:
            raise KeyError(f"buffer {buffer_id} is not covered by split")

    def _left(raw_buffers):
        return retile(raw_buffers, left_arg_index)

    def _right(raw_buffers):
        return retile(raw_buffers, right_arg_index)

    def _merge(left_tiles, right_tiles):
        return merge_tuples(left_tiles, right_tiles, tuple(origin), tuple(index))

    return Bundle(
        left=_left,
        right=_right,
        merge=_merge,
        left_buffers=left_buffers,
        right_buffers=right_buffers,
    )


def make_buffer_project(all_buffers, selected_buffers):
    index = _buffer_indices(tuple(all_buffers), tuple(selected_buffers))

    def _project(values):
        return retile(values, index)

    return _project


@dataclass(frozen=True)
class StageGridInfo:
    task_grid: ct.Constant[tuple[int, ...]]
    axis_names: ct.Constant[tuple[str, ...]]
    program_to_compute: ct.Constant[tuple[int, ...]]
    loop_compute_index: ct.Constant[int]
    loop_program_index: ct.Constant[int]
    loop_span_base: ct.Constant[int]
    loop_span_remainder: ct.Constant[int]
    compute_rank: ct.Constant[int]

    @classmethod
    def make(cls, stage: BuiltStage) -> StageGridInfo:
        require_resolved(stage)
        domain = stage.domain
        loop_axis = domain.loop_axis
        if loop_axis is None:
            raise ValueError(f"map/fold compile requires one loop axis: {domain}")
        loop_program_axis = domain.program_axis_for(loop_axis)
        loop_programs = 1 if loop_program_axis is None else int(loop_program_axis.programs)
        loop_tiles = int(loop_axis.num_tiles)
        loop_program_index = (
            -1
            if loop_program_axis is None
            else domain.program_axes.index(loop_program_axis)
        )
        return cls(
            task_grid=as_int_tuple(domain.task_grid),
            axis_names=tuple(axis.name for axis in domain.compute_axes),
            program_to_compute=tuple(
                domain.index(axis.source)
                for axis in domain.program_axes
                if axis.source in domain.compute_axes
            ),
            loop_compute_index=domain.index(loop_axis),
            loop_program_index=loop_program_index,
            loop_span_base=loop_tiles // loop_programs,
            loop_span_remainder=loop_tiles % loop_programs,
            compute_rank=len(domain.compute_axes),
        )

    @ct.function
    def init(self):
        pid = ct.bid(0)
        program_tid = ()
        for programs in ct.static_iter(self.task_grid):
            coord = pid % programs
            program_tid += (coord,)
            pid = pid // programs
        compute_tid = scatter_tuple(
            program_tid,
            full_tuple(0, self.compute_rank),
            self.program_to_compute
        )
        return compute_tid, program_tid

    @ct.function
    def loop_offset_and_size(self, program_tid):
        gid = 0 if self.loop_program_index < 0 else program_tid[self.loop_program_index]
        offset = self.loop_span_base * gid + ct.minimum(gid, self.loop_span_remainder)
        size = self.loop_span_base
        extra = (gid < self.loop_span_remainder)
        return offset, size, extra

    @ct.function
    def set_loop_index(self, tid, value):
        return set_at(tid, self.loop_compute_index, value)

    def loop_with_tail(self, body, *, static: bool = True, start: int = 0):
        @ct.function
        def _loop(tid, program_tid, carry, *args):
            loop_offset, loop_size, extra = self.loop_offset_and_size(program_tid)
            if static:
                for i in ct.static_iter(range(start, loop_size)):
                    loop_tid = self.set_loop_index(tid, loop_offset + i)
                    loop_stage_tid = loop_tid + program_tid
                    carry = body(loop_tid, loop_stage_tid, carry, *args)
            else:
                for i in range(start, loop_size):
                    loop_tid = self.set_loop_index(tid, loop_offset + i)
                    loop_stage_tid = loop_tid + program_tid
                    carry = body(loop_tid, loop_stage_tid, carry, *args)
            if extra:
                loop_tid = self.set_loop_index(tid, loop_offset + loop_size)
                loop_stage_tid = loop_tid + program_tid
                carry = body(loop_tid, loop_stage_tid, carry, *args)
            return carry
        return _loop

    @dataclass(frozen=True)
    class TidInfo:
        tid: tuple[int, ...]
        axis_names: ct.Constant[tuple[str, ...]]
        tile_shapes: ct.Constant[tuple[int, ...]]
        totals: ct.Constant[tuple[int, ...]]

        def axis_index(self, axis: ct.Constant[str | int]):
            if isinstance(axis, str):
                return self.axis_names.index(axis)
            return axis

        def shape(self, axis: ct.Constant[str | int], *more):
            ret = (self.tile_shapes[self.axis_index(axis)],)
            for current in ct.static_iter(more):
                ret += (self.tile_shapes[self.axis_index(current)],)
            if more:
                return ret
            return ret[0]

        def offset(self, axis: ct.Constant[str | int]):
            axis = self.axis_index(axis)
            return self.tid[axis] * self.tile_shapes[axis]

        def indices(self, axis: ct.Constant[str | int]):
            axis = self.axis_index(axis)
            tile = self.tile_shapes[axis]
            return self.offset(axis) + ct.arange(tile, dtype=ct.int32)

        def mask(self, axis: ct.Constant[str | int], *more):
            ix = self.axis_index(axis)
            ret = self.indices(ix) < self.totals[ix]
            for current in ct.static_iter(more):
                ix = self.axis_index(current)
                ret = ct.expand_dims(ret, -1)
                ret &= self.indices(ix) < self.totals[ix]
            return ret

    def tid_info(self, tid):
        compute_axes = self._compute_axes
        return StageGridInfo.TidInfo(
            tid=tid,
            axis_names=self.axis_names,
            tile_shapes=tuple(int(a.tile) for a in compute_axes),
            totals=tuple(int(a.extent) for a in compute_axes),
        )


def stage_grid_info(stage: BuiltStage):
    info = StageGridInfo.make(stage)
    object.__setattr__(info, "_compute_axes", stage.domain.compute_axes)
    return info


def identity(*xs):
    return xs


__all__ = [
    "Bundle",
    "StageBufferInfo",
    "StageFunctions",
    "StageGridInfo",
    "as_int_tuple",
    "ctmap",
    "ctzipdo",
    "ctzipmap",
    "identity",
    "inverse_p",
    "full_tuple",
    "make_buffer_project",
    "make_buffer_split",
    "make_buffer_helper",
    "merge_tuples",
    "require_resolved",
    "retile",
    "scatter_tuple",
    "set_at",
    "stage_grid_info",
]
