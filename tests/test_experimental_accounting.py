"""General schedule accounting on asymmetric named-axis problems."""

from dataclasses import replace

import pytest

from cutilereduce.core import symbolic as sym
from cutilereduce.core.measure import Measure
from cutilereduce.experimental.planning.accounting import AxisDomain, ExecutionScope, ScheduleDomain, contribution_ownership
from cutilereduce.experimental.planning.scheduled import CommitMode, build_backward_program, resolve_writes, schedule_backward_program
from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental import FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, StructuredValue, TensorSpec
from cutilereduce.experimental.planning.analysis import PlanningProblem, ScheduleCandidate
from cutilereduce.experimental.ir.facts import ValueFacts
from cutilereduce.experimental.backends.cpu.tuning import cpu_analytical_target


SIZES = {"h": 4, "l": 12, "g": 3, "r": 10, "dqk": 8, "dv": 6}


def attention():
    spec = FoldSpec({"query": TensorSpec(("h", "l", "g", "dqk")),
                     "key": TensorSpec(("h", "r", "dqk")),
                     "value": TensorSpec(("h", "r", "dv"))},
                    ("h", "l", "g"), ("r",))
    def factor(ctx, x):
        return StructuredValue({"log_weight": ctx.contract(x["query"], x["key"], over=("dqk",)),
                                "value": x["value"]})
    return FoldOperator(spec, LogWeightedSum(("dv",)),
                        FoldMaps(factor, lambda ctx, x: StructuredValue({"value": x["value"]}))).lower()


def domain(tiles, *, loop="r", partitions=1, scope=None):
    effective = SIZES | tiles
    owners = {a: (partitions if a == loop else (SIZES[a] + effective[a] - 1) // effective[a])
              for a in ("h", "l", "g", "r")}
    return ScheduleDomain.from_maps(SIZES, tiles=tiles, owners=owners,
                                   iteration_axes=("h", "l", "g", "r"), loop_axes=(loop,), scope=scope)


def test_dsl_infers_contraction_roles_dependencies_and_polynomial_work():
    p = attention()
    facts = p.region_facts["map_fold"]
    qk, pv = facts.contractions
    assert qk.axes.batch == ("h",)
    assert qk.axes.lhs_free == ("l", "g")
    assert qk.axes.rhs_free == ("r",)
    assert qk.axes.reduce == ("dqk",)
    assert qk.dimensions(SIZES) == {"B": 4, "M": 36, "N": 10, "K": 8}
    assert pv.axes.reduce == ("r",)
    assert pv.axes.rhs_free == ("dv",)
    assert set(facts.values[qk.node_id].sources) == {"query", "key"}
    assert facts.argument("key").axes == ("h", "r", "dqk")
    assert float(facts.argument("key").nbytes(SIZES)) == 4 * 10 * 8 * 4
    assert float(qk.flops().subs(SIZES)) == float(qk.flops(SIZES))
    with pytest.raises(TypeError):
        p.region_facts["injected"] = facts


def test_equal_batch_tile_volume_has_different_contraction_and_residency_effects():
    facts = attention().region_facts["map_fold"]
    heads = domain({"h": 2, "l": 2, "g": 1, "r": 5})
    queries = domain({"h": 1, "l": 4, "g": 1, "r": 5})
    qk = facts.contractions[0]
    a, b = heads.measures(Measure.tile), queries.measures(Measure.tile)
    assert qk.flops(a) == qk.flops(b)
    assert qk.dimensions(a)["B"] == 2 and qk.dimensions(b)["B"] == 1
    assert qk.dimensions(a)["M"] == 2 and qk.dimensions(b)["M"] == 4
    assert heads.buffer(facts.argument("query")).tile_bytes == queries.buffer(facts.argument("query")).tile_bytes
    assert heads.buffer(facts.argument("key")).tile_bytes == 2 * queries.buffer(facts.argument("key")).tile_bytes
    assert heads.buffer(facts.argument("value")).tile_bytes == 2 * queries.buffer(facts.argument("value")).tile_bytes


def test_l_and_g_retiling_preserves_gemm_rows_but_changes_replication_with_tails():
    facts = attention().region_facts["map_fold"]
    a = domain({"h": 1, "l": 4, "g": 1, "r": 5})
    b = domain({"h": 1, "l": 2, "g": 2, "r": 5})
    assert facts.contractions[0].dimensions(a.measures(Measure.tile)) == facts.contractions[0].dimensions(b.measures(Measure.tile))
    key_a, key_b = (d.buffer(facts.argument("key")) for d in (a, b))
    assert key_a.ownership.owners_per_target == 9
    assert key_b.ownership.owners_per_target == 12
    assert key_a.logical_accessed_bytes < key_b.logical_accessed_bytes
    assert key_a.streamed and not key_a.persistent
    assert a.buffer(facts.argument("query")).persistent


def test_symbolic_discrete_measures_binding_and_idle_owners():
    axis = AxisDomain("r", sym.Symbol("R"), sym.Symbol("T"), sym.Symbol("P"))
    bound = axis.subs({"R": 17, "T": 4, "P": 3})
    assert float(bound.num_tiles) == 5
    assert float(bound.span_tiles) == 2
    assert float(bound.span) == 8
    assert float(axis.span.subs({"R": 17, "T": 4, "P": 3})) == 8
    idle = AxisDomain("r", 3, 4, 8)
    assert idle.active_owners == 1
    assert idle.span == 4


def test_owner_and_local_reductions_do_not_prescribe_atomics():
    scope = ExecutionScope("device", ExecutionScope("node", ExecutionScope("cluster")))
    ownership = contribution_ownership(("h", "r", "dqk"), {"h": 2, "l": 3, "g": 2, "r": 1},
                                      loop_axes=("l",), scope=scope)
    assert ownership.owner_axes == ("l", "g")
    assert ownership.local_axes == ("l",)
    assert ownership.owners_per_target == 6 and ownership.requires_reduction
    assert ownership.scope.path == ("cluster", "node", "device")
    assert not hasattr(ownership, "commit_mode")
    p = attention()
    unresolved = build_backward_program(p, SIZES, tiles={"h": 2, "l": 4, "g": 1, "r": 5}, loop_axis="r", loop_programs=2)
    deferred = resolve_writes(unresolved, shared_commit=None)
    assert any(g.commit_mode is CommitMode.UNRESOLVED for g in deferred.gradients)
    with pytest.raises(ValueError, match="lose contributions"):
        resolve_writes(unresolved, shared_commit=CommitMode.STORE)
    actual = resolve_writes(unresolved)
    assert {g.input_name: g.contention_axes for g in actual.gradients} == {
        "query": ("r",), "key": ("l", "g"), "value": ("l", "g"),
    }


def test_analytical_domains_and_executor_ownership_agree_on_asymmetric_axes():
    p = attention()
    tiles = {"h": 2, "l": 4, "g": 1, "r": 5}
    problem = PlanningProblem(p, cpu_analytical_target(CpuTarget("explicit", 1, 32, 64)), SIZES)
    cost = problem.estimate("fused_tree", parameters={f"tile.{a}": t for a, t in tiles.items()}, backward=True, loop_axis="r")
    d = cost.stages[-1].accounting_domain
    scheduled = schedule_backward_program(p, SIZES, tiles=tiles, loop_axis="r")
    assert d.owner_count == 2 * 3 * 3
    facts = p.region_facts["map_finalize_map_vjp"]
    for gradient in scheduled.gradients:
        use = d.buffer(facts.result(gradient.result_name))
        assert use.ownership.owner_axes == gradient.contention_axes
        assert use.persistent == (gradient.commit_point.value == "program_end")
    uses = [d.buffer(value) for _, value in facts.arguments]
    assert float(cost.stages[-1].read_bytes) == float(sum(u.padded_accessed_bytes for u in uses))
    report = problem.sweep([ScheduleCandidate(
        "fused_tree", tuple((f"tile.{a}", t) for a, t in tiles.items()), True, "r")]).to_dict()
    assert report["estimates"][0]["metrics"]["stages"][-1]["domain"]["scope"] == ["backward_owner"]


def test_scope_labels_do_not_change_counts_and_invalid_domains_fail():
    value = ValueFacts(0, ("h", "r", "dqk"), "float32")
    d = domain({"h": 2, "l": 4, "g": 1, "r": 5})
    other = replace(d, scope=ExecutionScope("cta", ExecutionScope("device")))
    assert d.buffer(value).padded_accessed_bytes == other.buffer(value).padded_accessed_bytes
    with pytest.raises(ValueError):
        ScheduleDomain.from_maps({"x": 3}, tiles={"y": 2})
    with pytest.raises(ValueError):
        ScheduleDomain.from_maps({"x": 3}, owners={"x": 2}, iteration_axes=())
    with pytest.raises(ValueError):
        d.buffer(ValueFacts(0, ("missing",), "float32"))
    for invalid in (0, -1, 1.5, float('inf'), True):
        with pytest.raises(ValueError):
            AxisDomain("x", invalid, 1)


def test_domain_and_buffer_counts_match_legacy_accounting():
    import cuda.tile as ct
    from cutilereduce.core.axis import Axes
    from cutilereduce.core.buffer import Buffer, BufferId, Input
    from cutilereduce.core.stage_buffer import BufferStorage, StageAccess, StageBuffer
    from cutilereduce.core.stage_domain import (
        AxisRole, ComputeAxes, ComputeAxis, ProgramAxes, ProgramAxis, StageDomain,
    )
    new = domain({"h": 2, "l": 4, "g": 1, "r": 3}, partitions=2)
    axes = Axes.make("h l g r dqk dv")
    old = StageDomain("parity", ComputeAxes(values=tuple(
        ComputeAxis.make(a, AxisRole.Fold if a.name == "r" else
                         AxisRole.Batch if a.name in new.iteration_axes else AxisRole.Inner,
                         extent=SIZES[a.name], tile=new.axis(a.name).tile) for a in axes)),
        ProgramAxes(values=tuple(ProgramAxis.make(a, a.id, new.axis(a.name).owners)
                                 for a in axes if a.name in new.iteration_axes)),
        loop=next(a.id for a in axes if a.name == "r"))
    for axis in axes:
        for measure in Measure:
            assert float(new.axis(axis.name).measure(measure)) == float(old.resolve_measure(axis.id, measure))
    for names in ("h l g dqk", "h r dqk", "h r dv", "h l g r"):
        storage = Buffer(BufferId(Input, "value"), Axes.make(names), ct.float32)
        previous = StageBuffer(storage, old, BufferStorage.Ordinary, StageAccess.READ | StageAccess.RESIDENT)
        current = new.buffer(ValueFacts(0, tuple(names.split()), "float32"))
        assert float(current.tile_bytes) == float(previous.tile.bytes)
        assert float(current.span_bytes) == float(previous.span.bytes)
        assert float(current.logical_accessed_bytes) == float(previous.accessed_bytes)
        assert float(current.ownership.owners_per_target) == float(previous.residual_multiplicity)
        assert current.streamed == previous.is_streamed
