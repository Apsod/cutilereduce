"""CPU physical costs follow realization decisions rather than logical tensors."""

from dataclasses import replace

from cutilereduce.experimental import (
    CpuTarget, FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, PlanningProblem,
    StructuredValue, TensorRegionBuilder, TensorSignature, TensorSpec,
    analyze_cpu_region, analyze_region, cpu_analytical_target,
)
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
