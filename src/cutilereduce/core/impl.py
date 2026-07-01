from __future__ import annotations
from dataclasses import dataclass

import cuda.tile as ct
import math

from .base import Phase
from .buffer import BufferDep, BufferRole, ConcreteBuffer
from .grid import ConcreteGrid


@dataclass(frozen=True)
class BufferView:
    buffer_dims: tuple[int, ...]           # Internal index space -> grid index space mapping (injective)
    req_grad: bool
    role: BufferRole

    @staticmethod
    def from_buffer(buffer: ConcreteBuffer) -> BufferView:
        return BufferView(
                buffer_dims = tuple(d.grid_index for d in buffer.spec),
                req_grad = buffer.req_grad,
                role = buffer.role
        )

    ###### STATIC ######

    @property
    def buffer2outer(self, grid):
        return tuple(grid.axis_to_outer(d) for d in self.buffer_dims)
    
    @property
    def is_output(self):
        return self.role == BufferRole.Output

    @property
    def is_input(self):
        return self.role == BufferRole.Input
    
    def is_write(self, phase):
        match phase:
            case Phase.fwd: return self.is_output
            case Phase.bwd: return self.req_grad

    def is_read(self, phase):
        match phase:
            case Phase.fwd: return self.is_input
            case Phase.bwd: return self.is_input | self.is_output

    def is_grouped(self, grid):
        return grid.group_dim in self.buffer_dims

    @property
    def rank(self):
        return len(self.buffer_dims)

    def tile_shape(self, grid):
        return tuple(grid.dim_tiling[d] for d in self.buffer_dims)

def tuple_replace_at(original, replace_val, replace_ix):
    replaced = ()
    for i, v in enumerate(original):
        if i == replace_ix:
            replaced += (replace_val,)
        else:
            replaced += (v,)
    return replaced

@dataclass(frozen=True)
class GridView:
    dim_tiling: tuple[int, ...] # Tiling of all dims
    dim_total: tuple[int, ...]  # Total sizes of all dims 

    outer_dims: tuple[int, ...] # Index into dim_* corresponding to outer dims

    group_dim: int              # Index into dim_* corresponding to grouped dim
    num_groups: int             # Number of programs along group_dim

    @staticmethod
    def from_grid(grid : ConcreteGrid) -> GridView:
        return GridView(
                dim_tiling = tuple(d.tile_exp for d in grid.dims),
                dim_total = tuple(d.total_var for d in grid.dims),
                outer_dims = tuple(d.grid_index for d in grid.outer),
                group_dim = grid.group_dim.grid_index,
                num_groups = grid.config.num_groups
        )

    
    ###### STATIC ######

    def axis_to_outer(self, i):
        if i in self.outer_dims:
            return self.outer_dims.index(i)
        else:
            return None

    @property
    def group_outer_dim(self):
        return self.axis_to_outer(self.group_dim)

    def programs_along(self, i):
        if i == self.group_dim:
            return self.num_groups
        else:
            return ct.cdiv(self.dim_total[i], self.dim_tiling[i])

    def tiles_along(self, i):
        return ct.cdiv(self.dim_total[i], self.dim_tiling[i])
    
    @property
    def outer_shape(self):
        return tuple(self.programs_along(i) for i in self.outer_dims)

    @property
    def shape(self):
        return tuple(self.programs_along(i) for i in range(len(self.dim_tiling)))
    
    @property
    def tasks(self):
        return math.prod(self.shape)

    @property
    def grouping_info(self):
        tiles = self.tiles_along(self.group_dim)
        quot = tiles // self.num_groups
        rem = tiles % self.num_groups
        return quot, rem


def permutation_inverse(perm):
    inverse = [None] * len(perm)
    for i, val in enumerate(perm):
        inverse[val] = i
    return tuple(inverse)

@dataclass(frozen=True)
class ProgramView:
    buffers: tuple[BufferView]
    grid: GridView
    phase: Phase

    @staticmethod
    def from_spec(spec : ConcreteSpec) -> ProgramView:
        return ProgramView(
                buffers = tuple(BufferView.from_buffer(b) for b in (*spec.input.values(), *spec.output.values())),
                grid = GridView.from_grid(spec.grid),
                phase = spec.phase,

        )

    ###### STATIC ######

    @property
    def load_order(self):
        load2original = tuple(
                i
                for i, b 
                in enumerate(self.reads)
                if not b.is_grouped(self.grid)
                ) 
        load2original += tuple(
                i
                for i, b 
                in enumerate(self.reads)
                if b.is_read(self.phase) and b.is_grouped(self.grid)
                )
        return load2original

    @property
    def reads_index(self):
        return tuple(i for i, b in enumerate(self.buffers) if b.is_read(self.phase))

    @property
    def writes_index(self):
        return tuple(i for i, b in enumerate(self.buffers) if b.is_write(self.phase))

    @property
    def tasks(self):
        return self.grid.tasks

    @property
    def shape(self):
        return self.grid.program_shape

    @property
    def inputs(self):
        return tuple(b for b in self.buffers if b.is_input)

    @property
    def output(self):
        return tuple(b for b in self.buffers if b.is_output)

    @property
    def reads(self):
        return tuple(b for b in self.buffers if b.is_read(self.phase))

    @property
    def writes(self):
        return tuple(b for b in self.buffers if b.is_write(self.phase))
    
    ###### DYNAMIC ######

#def make(spec, map_reduce, combine):
#    grid = GridView.from_spec(spec)
#    inputs, outputs = BufferView.from_spec(spec)
#    meta = Meta.from_spec(spec)
#    
#    @ct.function()
#    def fwd(input_arrays, output_arrays):
#        pid = grid.pid
#        gsize, gstart = grid.group_info
#        
#        tid = ct.static_eval(grid.mk_tid(pid, gstart))
#        batch_loads = io.batch_load(tid, inputs, input_arrays)
#        fold_loads = io.fold_loads(tid, inputs, input_arrays)
#
#        acc = map_reduce(*io.pack(batch_loads, fold_loads))
#
#        for off in ct.range(1, gsize):
#            tid = ct.static_eval(grid.mk_tid(pid, gstart+off))
#            fold_loads = io.fold_loads(tid, inputs, input_arrays)
#            tmp = map_reduce(*io.pack(batch_loads, fold_loads))
#            acc = combine(acc, tmp)
#        
#        # This assumes num_groups=1 ... 
#        io.store_outputs(tid, outputs, output_arrays, acc)
#    
#    src = f"""
#ct.kernel(...)
#def kernel({argstuff}):
#    fwd({input_tuple}, {output_tuple})
#"""
#    compile blablabla
