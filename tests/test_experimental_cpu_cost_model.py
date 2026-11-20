"""CPU physical costs follow realization decisions rather than logical tensors."""

from dataclasses import replace

from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental import FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, StructuredValue, TensorSignature, TensorSpec
from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.ir.core import TensorRegionBuilder
from cutilereduce.experimental.backends.cpu.analysis import analyze_cpu_region
from cutilereduce.experimental.ir.cost import analyze_region
from cutilereduce.experimental.backends.cpu.tuning import cpu_analytical_target
from cutilereduce.experimental.backends.cpu.fusion import RegionFusion


def test_costing_shares_emitter_reuse_and_elides_intermediate_allocations():
    ctx = TensorRegionBuilder(("m", "k", "n"))
    a = ctx.expect("a", TensorSignature(("m", "k")))
    b = ctx.expect("b", TensorSignature(("k", "n")))
    product = ctx.contract(a, b, over=("k",))
    ctx.yield_value("out", ctx.exp(product * 2.0 + 1.0))
    region = ctx.finish_region()
    cost = analyze_cpu_region(region, {"m": 8, "k": 16, "n": 32})
    steps = RegionFusion(region).steps
    assert [k.reused_from for k in cost.realization.kernels] == [s.reused_from for s in steps]
    assert len(cost.realization.kernels) == 2
    assert cost.realization.kernels[-1].reused_from == product.id
    assert float(cost.realization.allocated_bytes) == 8 * 32 * 4
    assert float(cost.work.contraction_flops) == 2 * 8 * 16 * 32
    assert float(cost.realization.peak_owned_bytes) < float(analyze_region(region, {"m": 8, "k": 16, "n": 32}).live_bytes_upper_bound)


def test_cpu_costs_remain_symbolic_and_bind_like_direct_analysis():
    ctx = TensorRegionBuilder(("b", "r"))
    x = ctx.expect("x", TensorSignature(("b", "r")))
    weights = ctx.exp(x)
    ctx.yield_value("weights", weights)
    ctx.yield_value("sum", ctx.reduce("sum", weights, ("r",)))
    region = ctx.finish_region()
    symbolic = analyze_cpu_region(region)
    bound = symbolic.subs({"b": 3, "r": 17})
    direct = analyze_cpu_region(region, {"b": 3, "r": 17})
    assert len(bound.realization.kernels) == 1
    assert float(bound.realization.allocated_bytes) == float(direct.realization.allocated_bytes)
    assert float(bound.work.total) == float(direct.work.total)


def _attention():
    spec = FoldSpec({"q": TensorSpec(("l", "d")), "k": TensorSpec(("r", "d")),
                     "v": TensorSpec(("r", "v"))}, ("l",), ("r",))
    def factor(ctx, x):
        return StructuredValue({"log_weight": ctx.contract(x["q"], x["k"], over=("d",)), "value": x["v"]})
    return FoldOperator(spec, LogWeightedSum(("v",)),
                        FoldMaps(factor, lambda ctx, x: StructuredValue({"out": x["value"]}))).lower()


def test_cache_is_a_soft_ranking_signal_and_is_reported():
    program = _attention()
    hardware = CpuTarget("explicit", 1, 64, 64, l2_bytes=1310720, l3_bytes=12582912)
    sizes = {"l": 2048, "r": 2048, "d": 32, "v": 32}
    problem = PlanningProblem(program, cpu_analytical_target(hardware), sizes)
    def cost(query):
        return problem.estimate("fused_tree", parameters={"tile.l": query, "tile.r": 2048}, backward=True, loop_axis="l")
    small, full = cost(64), cost(2048)
    assert float(full.stages[0].cache_penalty_seconds) > float(small.stages[0].cache_penalty_seconds)
    assert float(full.stages[0].kernel_calls) < float(small.stages[0].kernel_calls)
    assert full.stages[0].invocations[0].cost.realization is not None
    assert problem.target.max_resident_bytes is None  # Cache is not a legality limit.
    assert problem.target.model_metadata["assumptions"]["l2_bytes"] == hardware.l2_bytes
    # Removing cache capacity removes the cache slowdown, without changing work.
    unknown = PlanningProblem(program, cpu_analytical_target(replace(hardware, l2_bytes=None, l3_bytes=None)), sizes)
    no_cache = unknown.estimate("fused_tree", parameters={"tile.l": 2048, "tile.r": 2048})
    assert float(no_cache.stages[0].cache_penalty_seconds) == 0


def test_explicit_realization_is_consumed_without_replanning(monkeypatch):
    from cutilereduce.experimental.backends.cpu.realization import realize_cpu_region
    from cutilereduce.experimental.backends.cpu import realization
    ctx = TensorRegionBuilder(("m", "k", "n"))
    a = ctx.expect("a", TensorSignature(("m", "k")))
    b = ctx.expect("b", TensorSignature(("k", "n")))
    product = ctx.contract(a, b, over=("k",))
    ctx.yield_value("out", ctx.exp(product * 2.0 + 1.0))
    region = ctx.finish_region()
    sizes = {"m": 8, "k": 16, "n": 32}
    implementation = realize_cpu_region(region)
    expected = implementation.estimate(sizes)
    source = implementation.emit("example", sizes)

    def forbid(*args, **kwargs):
        raise AssertionError("consumer attempted to redo realization")
    monkeypatch.setattr(realization, "RegionFusion", forbid)
    cost = implementation.estimate(sizes)
    assert implementation.emit("example", sizes) == source
    assert cost == expected
    assert float(cost.work.contraction_flops) == 8192
    assert float(cost.realization.allocated_bytes) == 1024
    assert [k.kind for k in cost.realization.kernels] == [s.kind for s in implementation.fusion.steps]
    # Estimation never mutates fusion/reuse choices, including symbolic binding.
    assert implementation.estimate().subs(sizes) == cost


def test_candidate_compilation_keeps_analyzed_program_and_realizations(monkeypatch):
    from cutilereduce.experimental.planning.analysis import ScheduleCandidate
    from cutilereduce.experimental.backends.cpu.tuning import compile_cpu_candidate
    from cutilereduce.experimental.backends.cpu import runtime
    from cutilereduce.experimental.backends.cpu.codegen import emit_cpp_source
    program = _attention()
    hardware = CpuTarget("explicit", 1, 32, 64)
    problem = PlanningProblem(program, cpu_analytical_target(hardware),
                              {"l": 8, "r": 8, "d": 4, "v": 4})
    candidate = ScheduleCandidate("fused_tree", (("tile.l", 8), ("tile.r", 8)))
    problem.evaluate(candidate)
    registry = problem.target.region_analyzer.__self__
    analyzed = registry.realize(program.map_fold)
    received = {}

    def compile_stub(operator, sizes, **kwargs):
        received.update(kwargs)
        return emit_cpp_source(kwargs["_program"], sizes, backward=False,
                               region_realizations=kwargs["_region_realizations"])
    monkeypatch.setattr(runtime, "compile_cpu", compile_stub)
    source = compile_cpu_candidate(None, problem, candidate, target=hardware)
    assert "PYBIND11_MODULE" in source
    assert received["_program"] is program
    assert received["_region_realizations"] is registry
    assert registry.realize(program.map_fold) is analyzed
