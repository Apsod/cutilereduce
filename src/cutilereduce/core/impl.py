from __future__ import annotations
from dataclasses import dataclass
import linecache
import math
import uuid

import cuda.tile as ct

from .base import Phase
from .buffer import BufferDep, BufferRole, ConcreteBuffer
from .grid import ConcreteGrid
from .spec import ConcreteSpec

@dataclass(frozen=True)
class TileCtx:
    tile: ct.Tile
    index: tuple[int,...]

    def indices_along(self, i):
        s = self.tile.shape[i]
        return ct.arange(s, dtype=ct.int32) + self.index[i] * s

    def start_along(self, i):
        return self.tile.shape[i] * self.index[i]

    def stop_along(self, i):
        return self.tile.shape[i] * (self.index[i] + 1)

@ct.function
def retile(original, index):
    ret = ()
    for i in ct.static_iter(index):
        ret += (original[i],)
    return ret

def mk_fwd(spec, map_reduce, combine):

    grid = spec.grid
    group_dim = grid.group_dim

    group_tiles = group_dim.num_tiles
    group_size_base = group_tiles // spec.groups
    group_remainder = group_tiles % spec.groups

    batch_buffers = tuple(b for b in spec.read_buffers if not b.is_grouped)
    group_buffers = tuple(b for b in spec.read_buffers if b.is_grouped)
    load_order = tuple(b.program_index for b in batch_buffers + group_buffers)

    @ct.function
    def batch_loads(views, tid):
        tiles = ()
        for i, index in ct.static_iter(
                (b.program_index, b.dims.grid_index)
                for b
                in batch_buffers
                ):
            tile_index = retile(tid, index)
            tile = views[i].load(tile_index)
            tiles += (TileCtx(tile,tile_index),)
        return tiles

    @ct.function
    def group_loads(views, tid):
        tiles = ()
        for pix, gixs in ct.static_iter(
                (b.program_index, b.dims.grid_index)
                for b
                in group_buffers
                ):
            tixs = retile(tid, gixs)
            tile = views[pix].load(tixs)
            tiles += (TileCtx(tile,tixs),)
        return tiles
    
    shape_prefix = tuple(d.num_programs for d in grid.dims[:group_dim.grid_index])
    shape_suffix = tuple(d.num_programs for d in grid.dims[group_dim.grid_index+1:])

    @ct.function
    def mk_tid():
        pid = ct.bid(0)
        tid = ()

        for s in ct.static_iter(shape_prefix):
            lid = pid % s
            tid += (lid,)
            pid = pid // s
        
        s = ct.static_eval(spec.groups)
        lid = pid % s
        pid = pid // s
        tid += (lid * group_size_base + ct.minimum(lid, group_remainder),)
        size = group_size_base + (lid < group_remainder)

        for s in ct.static_iter(shape_suffix):
            lid = pid % s
            tid += (lid,)
            pid = pid // s

        return tid, size

    @ct.function
    def increment_group(original):
        ret = ()
        for i, d in ct.static_iter(
                (d.grid_index, 1 if d.grouped else 0)
                for d
                in grid.dims
                ):
            ret += (original[i]+d,)
        return ret


    @ct.function
    def store(views, tid, tiles):
        for pix, gixs in ct.static_iter(
                (b.program_index, b.dims.grid_index)
                for b
                in spec.write_buffers
                ):
            views[pix].store(retile(tid, gixs), tiles[pix])

    @ct.function
    def mk_views(buffers):
        i = ct.static_eval(len(spec.input))
        inbuffs = buffers[:i]
        inviews = ()
        for pix, shape in ct.static_iter(
            (b.program_index, b.dims.tile_shape)
            for b 
            in spec.input_buffers
            ):
            view = inbuffs[pix].tiled_view(shape)
            inviews += (view,)

        outbuffs = buffers[i:]
        outviews = ()
        for pix, shape in ct.static_iter(
            (b.program_index, b.dims.tile_shape)
            for b 
            in spec.output_buffers
            ):
            view = outbuffs[pix].tiled_view(shape)
            outviews += (view,)

        return inviews, outviews

    @ct.function
    def fwd(buffers):
        input, output = mk_views(buffers)
        tid0, size = mk_tid()

        tid = tid0

        btiles = batch_loads(input, tid)
        gtiles = group_loads(input, tid)
        tiles = retile(btiles + gtiles, load_order)

        acc = map_reduce(*tiles)

        for _ in range(1, size):
            tid = increment_group(tid)
            gtiles = group_loads(input, tid)
            tiles = retile(btiles + gtiles, load_order)
            acc = combine(acc, map_reduce(*tiles))
        store(output, tid0, acc)

    
    nargs = len(spec.input) + len(spec.output)
    csv_args = ','.join(f'arg_{i}' for i in range(nargs))
    ns = {'ct': ct, 'fwd': fwd}

    source = (
            "@ct.kernel\n"
            f"def kernel({csv_args}):\n"
            f"  fwd(({csv_args},))\n"
            )

    filename = f"<cutilereduce_kernel_{uuid.uuid4().hex}>"
    linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
    )

    code = compile(source, filename, "exec")
    exec(code, ns)
    return ns['kernel']
