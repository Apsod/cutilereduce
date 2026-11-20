"""Symbolic analytical planning shared by CPU and cuTile search.

All counts are dense padded-tile upper bounds. Traffic models explicit region
arguments/commits with loop-invariant inputs resident; it is not cache traffic.
Time and occupancy are heuristics. No device discovery or compilation occurs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from itertools import product as cartesian_product
from functools import cached_property
import json
import math
from pathlib import Path
from typing import Any

from cutilereduce.core import symbolic as sym
from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.ir.cost import (
    DTYPE_BYTES, OperationCost, RegionCost, analyze_region, bind, elements, product,
)
from cutilereduce.experimental.planning.templates import build_schedule_space
from cutilereduce.experimental.planning.accounting import ExecutionScope, ScheduleDomain, ceildiv



@dataclass(frozen=True)
class AnalyticalTarget:
    """Explicit backend limits and rates; rates default to named symbols.

    Resident bytes are a conservative SSA capacity proxy, not a prediction of
    physical register/shared-memory allocation. Limits are opt-in for that reason.
    """

    backend: str
    workers: int = 1
    contraction_flops_per_second: Any = None
    scalar_ops_per_second: Any = None
    transcendental_ops_per_second: Any = None
    bytes_per_second: Any = None
    launch_seconds: Any = 0
    max_resident_bytes: int | None = None
    max_auxiliary_bytes: int | None = None
    atomic_weight: float = 2.0
    tile_choices: tuple[int, ...] = (16, 32, 64, 128)
    region_analyzer: Callable | None = field(default=None, repr=False, compare=False)
    stage_estimator: Callable | None = field(default=None, repr=False, compare=False)
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in {"cpu", "cutile"}:
            raise ValueError("analytical backend must be 'cpu' or 'cutile'")
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        if not self.tile_choices or any(type(t) is not int or t <= 0 for t in self.tile_choices):
            raise ValueError("tile choices must be positive integers")
        if self.atomic_weight < 1:
            raise ValueError("atomic weight must be at least one")
        for name in ("max_resident_bytes", "max_auxiliary_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("contraction_flops_per_second", "scalar_ops_per_second",
                     "transcendental_ops_per_second", "bytes_per_second"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, sym.Symbol(name))
            elif isinstance(value, (int, float)) and (value <= 0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.launch_seconds, (int, float)) and self.launch_seconds < 0:
            raise ValueError("launch_seconds must be nonnegative")

    @property
    def forward_families(self):
        return (("online",) if self.backend == "cutile" else
                ("online", "fused_tree", "partial_fold", "partial_scan_checkpoints"))


@dataclass(frozen=True)
class RegionInvocation:
    name: str
    cost: RegionCost
    count: Any
    parallel_tasks: Any = 1

    def subs(self, values):
        return RegionInvocation(self.name, self.cost.subs(values), bind(self.count, values),
                                bind(self.parallel_tasks, values))


@dataclass(frozen=True)
class StageCost:
    name: str
    direction: str
    work: OperationCost
    read_bytes: Any = 0
    write_bytes: Any = 0
    atomic_bytes: Any = 0
    resident_bytes_upper_bound: Any = 0
    allocated_bytes: Any = 0
    parallel_tasks: Any = 1
    serial_steps: Any = 1
    estimated_seconds: Any = 0
    kernel_calls: Any = 0
    realization_read_bytes: Any = 0
    realization_write_bytes: Any = 0
    realization_allocated_bytes: Any = 0
    compute_seconds: Any = 0
    memory_seconds: Any = 0
    dispatch_seconds: Any = 0
    cache_penalty_seconds: Any = 0
    invocations: tuple[RegionInvocation, ...] = ()
    accounting_domain: ScheduleDomain | None = None

    def subs(self, values):
        return replace(self, work=self.work.subs(values),
                       invocations=tuple(i.subs(values) for i in self.invocations),
                       accounting_domain=self.accounting_domain.subs(values) if self.accounting_domain else None, **{
            f.name: bind(getattr(self, f.name), values) for f in fields(self)
            if f.name not in {"name", "direction", "work", "invocations", "accounting_domain"}
        })


@dataclass(frozen=True)
class ScheduleCost:
    stages: tuple[StageCost, ...]
    auxiliary_peak_bytes: Any
    saved_bytes: Any
    convention: str = "padded dense upper bounds; heuristic roofline timing; no cache model"

    @property
    def work(self):
        return sum((stage.work for stage in self.stages), OperationCost())

    @property
    def estimated_seconds(self):
        return sum(stage.estimated_seconds for stage in self.stages)

    @property
    def forward_seconds(self):
        return sum(s.estimated_seconds for s in self.stages if s.direction == "forward")

    @property
    def backward_seconds(self):
        return sum(s.estimated_seconds for s in self.stages if s.direction == "backward")

    @property
    def resident_bytes_upper_bound(self):
        return sym.Max(*(s.resident_bytes_upper_bound for s in self.stages))

    def subs(self, values):
        return replace(self, stages=tuple(s.subs(values) for s in self.stages),
                       auxiliary_peak_bytes=bind(self.auxiliary_peak_bytes, values),
                       saved_bytes=bind(self.saved_bytes, values))


@dataclass(frozen=True)
class ScheduleCandidate:
    family: str
    parameters: tuple[tuple[str, int], ...]
    backward: bool = False
    loop_axis: str | None = None

    def __post_init__(self):
        if len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("candidate parameter names must be unique")
        if any(type(v) is not int or v <= 0 for _, v in self.parameters):
            raise ValueError("candidate parameters must be positive integers")

    @property
    def key(self):
        return (self.family, tuple(sorted(self.parameters)), self.backward, self.loop_axis)

    def to_dict(self):
        return {"family": self.family, "parameters": dict(self.parameters),
                "backward": self.backward, "loop_axis": self.loop_axis}

    @classmethod
    def from_dict(cls, value):
        return cls(value["family"], tuple(sorted(value["parameters"].items())),
                   value["backward"], value["loop_axis"])


@dataclass(frozen=True)
class CandidateEstimate:
    candidate: ScheduleCandidate
    cost: ScheduleCost | None
    rejection_reasons: tuple[str, ...] = ()

    @property
    def accepted(self):
        return not self.rejection_reasons


@dataclass(frozen=True)
class AnalyticalSweep:
    estimates: tuple[CandidateEstimate, ...]
    truncated: bool = False
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    def shortlist(self, count: int = 20, *, per_family: int = 1):
        if count <= 0 or per_family < 0:
            raise ValueError("count must be positive and per_family nonnegative")
        def score(estimate):
            value = estimate.cost.estimated_seconds
            if isinstance(value, sym.Expr) and value.free_symbols:
                raise ValueError("ranking requires bound sizes, schedule parameters, and hardware rates")
            result = float(value)
            if not math.isfinite(result) or result < 0:
                raise ValueError("analytical score must be finite and nonnegative")
            return result
        ranked = sorted((e for e in self.estimates if e.accepted), key=score)
        selected = []
        families = dict.fromkeys(e.candidate.family for e in ranked)
        for i in range(per_family):
            for family in families:
                matching = [e for e in ranked if e.candidate.family == family]
                if i < len(matching) and len(selected) < count:
                    selected.append(matching[i])
        keys = {e.candidate.key for e in selected}
        selected.extend(e for e in ranked if e.candidate.key not in keys)
        return tuple(selected[:count])

    def to_dict(self):
        def scalar(value):
            if isinstance(value, sym.Expr):
                return str(value)
            return float(value)
        return {"version": 1, "truncated": self.truncated, "model": dict(self.model_metadata), "estimates": [
            {"candidate": e.candidate.to_dict(), "rejection_reasons": list(e.rejection_reasons),
             "metrics": None if e.cost is None else {
                 "estimated_seconds": scalar(e.cost.estimated_seconds),
                 "forward_seconds": scalar(e.cost.forward_seconds),
                 "backward_seconds": scalar(e.cost.backward_seconds),
                 "auxiliary_peak_bytes": scalar(e.cost.auxiliary_peak_bytes),
                 "resident_bytes_upper_bound": scalar(e.cost.resident_bytes_upper_bound),
                 "work": {f.name: scalar(getattr(e.cost.work, f.name)) for f in fields(OperationCost)},
                 "stages": [{f.name: (s.work.total if f.name == "work" else getattr(s, f.name))
                             if f.name in {"name", "direction"} else
                             scalar(s.work.total if f.name == "work" else getattr(s, f.name))
                             for f in fields(StageCost) if f.name not in {"invocations", "accounting_domain"}} | {
                     "domain": s.accounting_domain.to_dict() if s.accounting_domain else None,
                     "regions": [{"name": inv.name, "count": scalar(inv.count),
                                  "kernel_count": len(inv.cost.realization.kernels) if inv.cost.realization else None,
                                  "convention": inv.cost.convention} for inv in s.invocations]
                 } for s in e.cost.stages],
                 "convention": e.cost.convention,
             }} for e in self.estimates]}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(frozen=True)
class PlanningProblem:
    program: FoldProgram
    target: AnalyticalTarget
    sizes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.program.spec.fold_axes) != 1:
            raise NotImplementedError("analytical schedule planning currently requires one fold axis")
        supplied = dict(self.sizes or {})
        if set(supplied) - set(self.program.spec.axes):
            raise ValueError("unknown logical sizes")
        if any(type(v) is not int or v <= 0 for v in supplied.values()):
            raise ValueError("sizes must be positive integers")
        object.__setattr__(self, "sizes", supplied)

    @property
    def facts(self):
        return self.program.region_facts

    @cached_property
    def regions(self) -> dict[str, RegionCost]:
        from cutilereduce.experimental.ir.core import TensorRegion
        from cutilereduce.experimental.ir.passes import realize_contraction_precision
        if self.target.region_analyzer is None and self.target.backend != "cutile":
            return {name: facts.cost for name, facts in self.facts.items()}
        analyzer = self.target.region_analyzer or analyze_region
        return {f.name: analyzer(realize_contraction_precision(value)
                                if self.target.backend == "cutile" else value)
                for f in fields(self.program)
                if isinstance(value := getattr(self.program, f.name), TensorRegion)}

    @cached_property
    def _region_bindings(self):
        return {}

    def bound_region(self, name, dimensions):
        key = (name, tuple((a, dimensions[a]) for a in self.program.spec.axes))
        if all(isinstance(value, int) for _, value in key[1]):
            if key not in self._region_bindings:
                self._region_bindings[key] = self.regions[name].subs(dimensions)
            return self._region_bindings[key]
        return self.regions[name].subs(dimensions)

    def estimate(self, family="online", *, parameters=None, backward=False, loop_axis=None):
        """Return symbolic costs, accepting partial shape and parameter bindings."""
        space = build_schedule_space(self.program)
        available = {p.name: p for p in space.forward}
        if family not in available or family not in self.target.forward_families:
            raise NotImplementedError(f"{self.target.backend} does not support {family!r}")
        if backward:
            if not space.backward:
                raise ValueError("program has no backward regions")
            space.compose(available[family], space.backward[0])
            if self.target.backend == "cutile" and self.program.requires_prefix:
                raise NotImplementedError("cuTile backward does not support ordered prefixes")
        axis = self.program.spec.fold_axes[0]
        loop_axis = loop_axis or axis
        if loop_axis not in (*self.program.spec.batch_axes, axis):
            raise ValueError("backward loop axis must be a batch or fold axis")
        if self.program.requires_prefix and loop_axis != axis:
            raise ValueError("prefix backward must loop over the ordered fold axis")
        p = dict(parameters or {})
        allowed = ({"partition_count", "loop_programs"}
                   | {f"{prefix}.{a}" for prefix in ("tile", "backward_tile")
                      for a in (*self.program.spec.batch_axes, axis)}
                   | {f"embed_tile.{a}" for a in self.program.spec.batch_axes})
        if set(p) - allowed:
            raise ValueError(f"unknown schedule parameters: {sorted(set(p) - allowed)}")
        if any(isinstance(v, (int, float)) and (v <= 0 or not math.isfinite(v)) for v in p.values()):
            raise ValueError("schedule parameters must be positive")
        if self.target.backend == "cutile":
            for name, value in p.items():
                if name.startswith(("tile.", "backward_tile.", "embed_tile.")) and isinstance(value, int):
                    if value & (value - 1):
                        raise ValueError("cuTile physical tiles must be powers of two")
                    logical = name.split(".", 1)[1]
                    if logical in self.sizes and value > self.sizes[logical]:
                        raise ValueError("cuTile tile exceeds logical extent")
        if self.target.backend == "cutile" and p.get("partition_count", 1) != 1:
            raise ValueError("online cuTile forward requires one fold partition")
        if self.program.requires_prefix and "loop_programs" in p:
            raise ValueError("prefix backward loop programs are determined by checkpoint partitions")
        for name in ("partition_count", "loop_programs"):
            logical = axis if name == "partition_count" else loop_axis
            if logical in self.sizes and isinstance(p.get(name), int) and p[name] > self.sizes[logical]:
                raise ValueError(f"{name} exceeds the logical extent")
        n = {a: sym.Symbol(a) for a in self.program.spec.axes}
        n.update(self.sizes)
        tiles = {a: sym.Min(n[a], p.get(f"tile.{a}", sym.Symbol(f"tile.{a}")))
                 for a in (*self.program.spec.batch_axes, axis)}
        extent = n | tiles
        owners = product(ceildiv(n[a], tiles[a]) for a in self.program.spec.batch_axes)
        partitions = (p.get("partition_count", sym.Symbol("partition_count"))
                      if family.startswith("partial") else 1)
        chunks = ceildiv(ceildiv(n[axis], partitions), tiles[axis])
        stages = []
        target = self.target

        def timed(name, direction, work, reads=0, writes=0, atomics=0,
                  resident=0, allocated=0, tasks=1, steps=1, invocations=(), domain=None):
            scalar = work.arithmetic + work.comparisons + work.conversions + work.reduction_ops + work.scan_ops
            utilization = sym.Min(1, sym.sympify(tasks) / target.workers)
            compute = (work.contraction_flops / target.contraction_flops_per_second
                       + scalar / target.scalar_ops_per_second
                       + work.transcendentals / target.transcendental_ops_per_second)
            memory = (reads + writes + atomics * target.atomic_weight) / target.bytes_per_second
            time = sym.Max(compute, memory) / utilization + target.launch_seconds
            stage = StageCost(name, direction, work, reads, writes, atomics,
                              resident, allocated, tasks, steps, time, invocations=invocations, accounting_domain=domain)
            if target.stage_estimator is not None:
                stage = target.stage_estimator(stage)
            stages.append(stage)
            return stage

        def region(name, dims):
            return self.bound_region(name, dims)

        def bytes_of(name, dims, result=True):
            cost = region(name, dims)
            return sum(v for _, v in (cost.result_bytes if result else cost.argument_bytes))

        online = family != "fused_tree"
        local = region("map_fold_combine" if online else "map_fold", extent)
        calls = owners * partitions * chunks
        carrier_local = bytes_of("map_fold", extent)
        forward_domain = ScheduleDomain.from_maps(
            n, tiles=tiles,
            owners={**{a: ceildiv(n[a], tiles[a]) for a in self.program.spec.batch_axes}, axis: partitions},
            iteration_axes=(*self.program.spec.batch_axes, axis), loop_axes=(axis,),
            scope=ExecutionScope("forward_owner"),
        )
        input_uses = {name: forward_domain.buffer(value) for name, value in self.facts["map_fold"].arguments}
        inputs = {name: use.tile_bytes for name, use in input_uses.items()}
        reads = sum(use.padded_accessed_bytes for use in input_uses.values())
        global_carrier = bytes_of("map_fold", n)
        partials = global_carrier * partitions if family.startswith("partial") else 0
        tree_scratch = global_carrier * chunks if family == "fused_tree" else 0
        work = local.work.scale(calls)
        forward_invocations = [RegionInvocation("map_fold_combine" if online else "map_fold", local, calls,
                                                owners * partitions * (chunks if not online else 1))]
        if not online:
            combine_count = owners * sym.SaturateSub(chunks, 1)
            combine_cost = region("execution_combine", extent)
            work += combine_cost.work.scale(combine_count)
            forward_invocations.append(RegionInvocation("execution_combine", combine_cost, combine_count, owners))
        tasks = owners * partitions * (chunks if not online else 1)
        timed("map_fold", "forward", work, reads, partials or global_carrier,
              resident=local.live_bytes_upper_bound + sum(inputs.values()) + 2 * carrier_local,
              allocated=partials or global_carrier, tasks=tasks, steps=chunks if online else 1,
              invocations=tuple(forward_invocations), domain=forward_domain)
        checkpoints = 0
        if family.startswith("partial"):
            checkpoints = global_carrier * partitions if self.program.requires_prefix and backward else 0
            timed("scan_partials" if checkpoints else "combine_partials", "forward",
                  region("execution_combine", n).work.scale(sym.SaturateSub(partitions, 1)),
                  partials, global_carrier + checkpoints,
                  resident=region("execution_combine", extent).live_bytes_upper_bound + 2 * carrier_local,
                  allocated=global_carrier + checkpoints, tasks=owners, steps=partitions,
                  invocations=(RegionInvocation("execution_combine", region("execution_combine", n),
                                                 sym.SaturateSub(partitions, 1), owners),))
        output_bytes = bytes_of("project_output", n)
        project = region("project_output", extent)
        timed("project_output", "forward", project.work.scale(owners), global_carrier,
              output_bytes, resident=project.live_bytes_upper_bound + carrier_local,
              allocated=output_bytes, tasks=owners,
              invocations=(RegionInvocation("project_output", project, owners, owners),),
              domain=ScheduleDomain.from_maps(n, tiles=tiles,
                  owners={a: ceildiv(n[a], tiles[a]) for a in self.program.spec.batch_axes},
                  iteration_axes=self.program.spec.batch_axes, scope=ExecutionScope("project_owner")))
        saved = global_carrier + checkpoints if backward else 0
        auxiliary = sym.Max(partials + global_carrier + checkpoints,
                            tree_scratch + global_carrier)
        if target.backend == "cutile":
            # The online emitter fuses projection and keeps E in registers.
            # Only training stores a global carrier for the later embed launch.
            first, projection = stages
            stages.clear()
            timed("map_fold_online", "forward", first.work + projection.work,
                  first.read_bytes, output_bytes + saved,
                  resident=sym.Max(first.resident_bytes_upper_bound,
                                   projection.resident_bytes_upper_bound),
                  allocated=output_bytes + saved, tasks=first.parallel_tasks,
                  steps=first.serial_steps, domain=forward_domain)
            auxiliary = saved
        if backward:
            bextent = n | {a: sym.Min(n[a], p.get(f"backward_tile.{a}", tiles[a])) for a in tiles}
            eextent = (n if target.backend == "cpu" else
                       n | {a: sym.Min(n[a], p.get(f"embed_tile.{a}", tiles[a])) for a in self.program.spec.batch_axes})
            embed_tasks = 1 if target.backend == "cpu" else product(ceildiv(n[a], eextent[a]) for a in self.program.spec.batch_axes)
            embed = region("embed_output_vjp", eextent)
            accumulator = bytes_of("embed_output_vjp", n)
            timed("embed_output_vjp", "backward", embed.work.scale(embed_tasks),
                  sum(v for _, v in embed.argument_bytes) * embed_tasks, accumulator,
                  resident=embed.live_bytes_upper_bound + sum(v for _, v in embed.argument_bytes),
                  allocated=accumulator, tasks=embed_tasks,
                  invocations=(RegionInvocation("embed_output_vjp", embed, embed_tasks, embed_tasks),),
                  domain=ScheduleDomain.from_maps(n, tiles=eextent,
                      owners={a: ceildiv(n[a], eextent[a]) for a in self.program.spec.batch_axes},
                      iteration_axes=self.program.spec.batch_axes, scope=ExecutionScope("embed_owner")))
            if self.program.requires_prefix:
                loop_programs = partitions
            else:
                loop_programs = p.get("loop_programs", 1)
            grid_axes = tuple(a for a in tiles if a != loop_axis)
            grid = {a: ceildiv(n[a], bextent[a]) for a in grid_axes}
            programs = product(grid.values()) * loop_programs
            steps = ceildiv(ceildiv(n[loop_axis], bextent[loop_axis]), loop_programs)
            name = ("map_finalize_map_vjp_accumulate" if self.program.requires_prefix else "map_finalize_map_vjp")
            final = region(name, bextent)
            backward_domain = ScheduleDomain.from_maps(
                n, tiles={a: bextent[a] for a in tiles},
                owners=grid | {loop_axis: loop_programs},
                iteration_axes=(*self.program.spec.batch_axes, axis), loop_axes=(loop_axis,),
                scope=ExecutionScope("backward_owner"),
            )
            backward_facts = self.facts[name]
            read = sum(backward_domain.buffer(value).padded_accessed_bytes
                       for _, value in backward_facts.arguments)
            writes = atomics = persistent = 0
            for arg, value in backward_facts.results:
                if not arg.startswith("partial_grad_input."):
                    continue
                use = backward_domain.buffer(value)
                size = use.tile_bytes
                multiplicity = programs * (steps if use.streamed else 1)
                ownership = use.ownership
                if use.persistent:
                    persistent += size
                # The accounting requires a reduction. This initial executor
                # realizes unresolved inter-owner reductions as atomic adds.
                if isinstance(ownership.owners_per_target, (int, float)) and ownership.owners_per_target == 1:
                    writes += size * multiplicity
                else:
                    atomics += size * multiplicity
            work = final.work.scale(programs * steps)
            work += OperationCost(arithmetic=(persistent / 4) * programs * steps)
            gradients = sum(elements(spec.axes, n) * DTYPE_BYTES[spec.dtype]
                            for spec in self.program.spec.inputs.values() if spec.differentiable)
            timed("map_finalize_map_vjp", "backward", work, read, writes, atomics,
                  resident=final.live_bytes_upper_bound + persistent + sum(v for _, v in final.argument_bytes),
                  allocated=gradients, tasks=programs, steps=steps,
                  invocations=(RegionInvocation(name, final, programs * steps, programs),),
                  domain=backward_domain)
            auxiliary = sym.Max(auxiliary, saved + accumulator)
        return ScheduleCost(tuple(stages), auxiliary, saved,
                            "CPU realization-aware heuristic" if target.stage_estimator else
                            "padded dense upper bounds; heuristic roofline timing; no cache model")

    def canonical_candidate(self, candidate: ScheduleCandidate) -> ScheduleCandidate:
        """Normalize equivalent defaults before costing, compiling or comparing."""
        p = dict(candidate.parameters)
        if p.get("partition_count") == 1 and not candidate.family.startswith("partial"):
            p.pop("partition_count")
        if p.get("loop_programs") == 1 and not self.program.requires_prefix:
            p.pop("loop_programs")
        for axis in (*self.program.spec.batch_axes, *self.program.spec.fold_axes):
            for prefix in ("backward_tile", "embed_tile"):
                if p.get(f"{prefix}.{axis}") == p.get(f"tile.{axis}"):
                    p.pop(f"{prefix}.{axis}", None)
        loop = (candidate.loop_axis or self.program.spec.fold_axes[0]) if candidate.backward else None
        return ScheduleCandidate(candidate.family, tuple(sorted(p.items())), candidate.backward, loop)

    def evaluate(self, candidate: ScheduleCandidate) -> CandidateEstimate:
        candidate = self.canonical_candidate(candidate)
        reasons = []
        try:
            cost = self.estimate(candidate.family, parameters=dict(candidate.parameters),
                                 backward=candidate.backward, loop_axis=candidate.loop_axis)
        except (ValueError, NotImplementedError) as error:
            return CandidateEstimate(candidate, None, (str(error),))
        for metric, limit in ((cost.resident_bytes_upper_bound, self.target.max_resident_bytes),
                              (cost.auxiliary_peak_bytes, self.target.max_auxiliary_bytes)):
            if limit is not None:
                if isinstance(metric, sym.Expr) and metric.free_symbols:
                    raise ValueError("resource pruning requires bound sizes and schedule parameters")
                if float(metric) > limit:
                    reasons.append(f"resource estimate {float(metric):g} exceeds limit {limit:g}")
        return CandidateEstimate(candidate, cost, tuple(reasons))

    def tile_domain_bounds(self):
        """Necessary polynomial input-residency bounds for initial domain pruning.

        Set other tiles to one to obtain safe independent upper bounds. Full SSA
        and carrier residency still require candidate-level resource checks.
        """
        from cutilereduce.core.poly import Polynomial, mono, polynomial_upper_bound
        if set(self.sizes) != set(self.program.spec.axes):
            raise ValueError("tile bounds require every logical size")
        axes = (*self.program.spec.batch_axes, *self.program.spec.fold_axes)
        limit = self.target.max_resident_bytes
        if limit is None:
            return {axis: self.sizes[axis] for axis in axes}
        polynomial = Polynomial.zero()
        for spec in self.program.spec.inputs.values():
            coefficient = DTYPE_BYTES[spec.dtype] * product(self.sizes[a] for a in spec.axes if a not in axes)
            polynomial += mono(coefficient, **{f"tile.{a}": 1 for a in spec.axes if a in axes})
        result = {}
        for axis in axes:
            variable = f"tile.{axis}"
            bound = polynomial_upper_bound(polynomial, variable, limit,
                                            {f"tile.{a}": 1 for a in axes if a != axis})
            result[axis] = self.sizes[axis] if bound is None else min(self.sizes[axis], bound)
        return result

    def candidates(self, *, backward=False, max_candidates=1_000_000, parameter_domains=None):
        """Search independent forward/backward tiles with a deterministic budget.

        Prefix-free training explores the Cartesian product of stage tile grids.
        Cyclic diagonals visit every forward and backward shape before moving
        to the next pairing, retaining stage and topology diversity under a cap.
        Prefix checkpoint schedules retain their shared chunk layout.
        Explicit backward domains replace the corresponding default domains.
        """
        if set(self.sizes) != set(self.program.spec.axes):
            raise ValueError("candidate enumeration requires every problem size")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        parameter_domains = dict(parameter_domains or {})
        if any(not values or any(type(v) is not int or v <= 0 for v in values)
               for values in parameter_domains.values()):
            raise ValueError("parameter domains must contain positive integers")
        axes = (*self.program.spec.batch_axes, *self.program.spec.fold_axes)
        domains = []
        bounds = self.tile_domain_bounds()
        for axis in axes:
            size = self.sizes[axis]
            if self.target.backend == "cutile":
                values = tuple(t for t in self.target.tile_choices if t & (t - 1) == 0)
                values = tuple(sorted({1} | {t for t in values if t <= size}))
            else:
                values = tuple(sorted({min(size, t) for t in self.target.tile_choices} | {size}))
            values = parameter_domains.pop(f"tile.{axis}", values)
            domains.append(tuple(v for v in values if v <= bounds[axis]))
        families = [f for f in build_schedule_space(self.program).forward
                    if f.name in self.target.forward_families
                    and (not backward or not self.program.requires_prefix or f.name == "partial_scan_checkpoints")]
        allowed_domains = ({"partition_count", "loop_programs"}
                           | {f"backward_tile.{a}" for a in axes}
                           | {f"embed_tile.{a}" for a in self.program.spec.batch_axes})
        if set(parameter_domains) - allowed_domains:
            raise ValueError(f"unknown parameter domains: {sorted(set(parameter_domains) - allowed_domains)}")
        backward_domains = []
        if backward and not self.program.requires_prefix:
            for axis, domain in zip(axes, domains, strict=True):
                backward_domains.append(tuple(parameter_domains.pop(f"backward_tile.{axis}", domain)))
        forward_count = math.prod(map(len, domains))
        backward_count = math.prod(map(len, backward_domains))

        def grid_value(grid, index):
            # Mixed-radix indexing avoids materializing an unbounded tile grid.
            values = []
            for domain in reversed(grid):
                index, digit = divmod(index, len(domain))
                values.append(domain[digit])
            return tuple(reversed(values))

        count = 0
        seen = set()
        # Alternating offsets cover both directions of decoupling early. With
        # equal grids the first diagonal covers every shape in both stages.
        offsets = (((j + 1) // 2 if j % 2 else -(j // 2)) for j in range(backward_count))
        pairings = ((grid_value(domains, i), grid_value(backward_domains, (i + offset) % backward_count))
                    for offset in offsets
                    for i in range(forward_count))
        for values, backward_values in pairings:
            params = {f"tile.{axis}": value for axis, value in zip(axes, values, strict=True)}
            params.update({f"backward_tile.{axis}": value
                           for axis, value in zip(axes, backward_values, strict=False)})
            for plan in families:
                partitions = (tuple(sorted({1, min(self.sizes[axes[-1]], self.target.workers)}))
                              if plan.name.startswith("partial") else (1,))
                loops = (axes[-1],) if self.program.requires_prefix or not backward else axes
                for partition in partitions:
                    for loop in loops:
                        extra_domains = parameter_domains | ({"loop_programs": (1, min(self.target.workers, self.sizes[loop]))}
                                                              if backward and not self.program.requires_prefix and "loop_programs" not in parameter_domains else {})
                        keys = tuple(extra_domains)
                        for extra_values in cartesian_product(*extra_domains.values()):
                            overrides = dict(zip(keys, extra_values, strict=True))
                            candidate = ScheduleCandidate(plan.name, tuple(sorted((params | {"partition_count": partition} | overrides).items())), backward, loop)
                            if candidate.key in seen:
                                continue
                            seen.add(candidate.key)
                            if count >= max_candidates:
                                return
                            count += 1
                            yield candidate

    def sweep(self, candidates=None, *, backward=False, max_candidates=1_000_000, parameter_domains=None):
        source = self.candidates(backward=backward, max_candidates=max_candidates + 1,
                                 parameter_domains=parameter_domains) if candidates is None else candidates
        seen = set()
        estimates = []
        truncated = False
        for candidate in source:
            candidate = self.canonical_candidate(candidate)
            if candidates is None and len(estimates) >= max_candidates:
                truncated = True
                break
            if candidate.key not in seen:
                seen.add(candidate.key)
                estimates.append(self.evaluate(candidate))
        return AnalyticalSweep(tuple(estimates), truncated, self.target.model_metadata)
