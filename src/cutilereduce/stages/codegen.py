from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cuda.tile as ct

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


@dataclass(frozen=True)
class StageGridInfo:
    task_grid: ct.Constant[tuple[int, ...]]
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
        compute_tid = ()
        program_tid = ()
        for _ in ct.static_iter(range(self.compute_rank)):
            compute_tid += (0,)
        for programs in ct.static_iter(self.task_grid):
            coord = pid % programs
            program_tid += (coord,)
            pid = pid // programs
        for i, compute_index in ct.static_iter(enumerate(self.program_to_compute)):
            compute_tid = set_at(compute_tid, compute_index, program_tid[i])
        return compute_tid, program_tid

    @ct.function
    def loop_offset_and_size(self, program_tid):
        gid = 0 if self.loop_program_index < 0 else program_tid[self.loop_program_index]
        offset = self.loop_span_base * gid + ct.minimum(gid, self.loop_span_remainder)
        size = self.loop_span_base + (gid < self.loop_span_remainder)
        return offset, size

    @ct.function
    def set_loop_index(self, tid, value):
        return set_at(tid, self.loop_compute_index, value)

    @dataclass(frozen=True)
    class TidInfo:
        tid: tuple[int, ...]
        tile_shapes: ct.Constant[tuple[int, ...]]
        totals: ct.Constant[tuple[int, ...]]

        def offset(self, axis: ct.Constant[int]):
            return self.tid[axis] * self.tile_shapes[axis]

        def indices(self, axis: ct.Constant[int]):
            tile = self.tile_shapes[axis]
            return self.offset(axis) + ct.arange(tile, dtype=ct.int32)

        def mask(self, axis: ct.Constant[int]):
            return self.indices(axis) < self.totals[axis]

    def tid_info(self, tid):
        compute_axes = self._compute_axes
        return StageGridInfo.TidInfo(
            tid=tid,
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
    "make_buffer_helper",
    "require_resolved",
    "retile",
    "set_at",
    "stage_grid_info",
]
