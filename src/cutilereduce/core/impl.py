from dataclasses import dataclass

import cuda.tile as ct

from .buffer import BufferDep, BufferRole


@dataclass(frozen=True)
class BufferView:
    buffer_dims: tuple[int, ...]           # Internal index space -> grid index space mapping (injective)
    req_grad: bool
    role: BufferRole

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

    ###### DYNAMIC ######

    def grid_tid2buffer_tid(self, grid_tid):
        buffer_tid = ()
        for d in ct.static_iter(self.buffer2outer):
            buffer_tid += (grid_tid[d] if d is not None else 0,)
        return bid

    def tiled(self, grid, array):
        return array.tiled_view(self.tile_shape(grid))

    def load(self, view, tid):
        return view.load(self.tid2bid(tid))

    def store(self, view, arr, tid):
        return view.store(self.tid2bid(tid), arr)

    def atomic_add(self, view, arr, tid):
        return view.atomic_add(self.tid2bid(tid), arr)

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
    
    ###### STATIC ######

    def axis_to_outer(self, i):
        if i in self.outer_dims:
            return self.outer_dims.index(i)
        else:
            return None

    @property
    def group_outer_dim(self):
        return self.axis2outer(self.group_dim)

    def programs_along(self, i):
        if i == self.group_dim:
            return self.num_groups
        else:
            return ct.cdiv(self.dim_total[i], self.dim_tiling[i])

    def tiles_along(self, i):
        return ct.cdiv(self.dim_total[i], self.dim_tiling[i])
    
    @property
    def program_shape(self):
        return tuple(self.programs_along(i) for i in self.outer_dims)
    
    @property
    def tasks(self):
        return ct.prod(self.program_shape)

    ###### DYNAMIC ######

    @property
    def pid(self):
        i = ct.bid(0)
        gids = ()
        for t in ct.static_iter(self.program_shape):
            gids += (i % t,)
            i = i // t
        return gids

    @property
    def tid_info(self):
        pid = self.pid
        gid = pid[self.group_outer_dim]
        tiles = self.tiles_along(self.group_dim)
        quot = tiles // self.num_groups
        rem = tiles % self.num_groups
        size = quot + (gid < rem)
        start = gid * quot + ct.minimum(gid, rem)
        tid = ()



    @property
    def gid(self):
        return self.pid[self.group_outer_dim]

    @property
    def group_info(self):
        tiles = ct.cdiv(self.dim_total[self.group_dim], self.dim_tiling[self.group_dim])
        quot = tiles // self.num_groups
        rem = tiles % self.num_groups
        gid = self.gid
        size = quot + (gid < rem)
        start = gid * quot + ct.minimum(gid, rem)
        return size, start

    def mk_tid(self, pid, gix):
        out = ()
        for i, tid in enumerate(pid):
            out += (gix if i == self.group_outer_dim else tid,)
        return out

@dataclass(frozen=True)
class ProgramView:
    buffers: list[BufferView]
    grid: GridView
    phase: Phase

    ###### STATIC ######

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
