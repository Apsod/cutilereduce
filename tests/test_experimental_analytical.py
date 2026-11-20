"""Analytical planning contracts, independent of device discovery and compilation."""

import json
from pathlib import Path
from dataclasses import replace

import pytest
import torch

from cutilereduce.core.poly import Polynomial
from cutilereduce.experimental import Add, Affine, FoldMaps, FoldOperator, FoldSpec, StructuredValue, TensorSpec
from cutilereduce.experimental.planning.analysis import BackendModel, PlanningProblem, ScheduleCandidate
from cutilereduce.experimental.ir.cost import analyze_region
from cutilereduce.experimental.backends.cpu.tuning import compile_cpu_candidate
from cutilereduce.experimental.backends.cutile.runtime import compile_cutile_candidate
from cutilereduce.experimental.planning.tuning import load_candidate, tune_candidates
from cutilereduce.experimental.backends.cpu.schedule import CpuTarget


def add_operator():
    return FoldOperator(
        FoldSpec({"x": TensorSpec(("b", "r"))}, ("b",), ("r",)),
        Add(), FoldMaps(lambda ctx, inputs: StructuredValue({"value": inputs["x"]})),
    )


def target(backend="cpu", **kwargs):
    from cutilereduce.experimental.backends.cutile.analysis import cutile_analytical_target
    constructor = cutile_analytical_target if backend == "cutile" else lambda **kw: BackendModel(backend, **kw)
    return constructor(
        workers=4, contraction_flops_per_second=1e12,
        scalar_ops_per_second=1e10, transcendental_ops_per_second=1e9,
        bytes_per_second=1e10, **kwargs,
    )


def test_symbolic_contractions_are_polynomial_and_count_shared_producers_once():
    from cutilereduce.experimental.ir.core import TensorRegionBuilder
    from cutilereduce.experimental import TensorSignature
    ctx = TensorRegionBuilder(("b", "m", "n", "k"))
    lhs = ctx.expect("lhs", TensorSignature(("b", "m", "k")))
    rhs = ctx.expect("rhs", TensorSignature(("b", "k", "n")))
    dot = ctx.contract(lhs, rhs, over=("k",))
    ctx.yield_value("a", dot)
    ctx.yield_value("b", dot + dot)
    cost = analyze_region(ctx.finish_region())
    assert isinstance(cost.work.contraction_flops.value, Polynomial)
    sizes = {"b": 2, "m": 3, "n": 5, "k": 7}
    bound = cost.subs(sizes)
    assert float(bound.work.contraction_flops) == 2 * 2 * 3 * 5 * 7
    assert float(bound.work.arithmetic) == 2 * 3 * 5
    assert float(dict(bound.argument_bytes)["lhs"]) == 2 * 3 * 7 * 4


def test_partial_binding_and_ragged_tile_counts():
    problem = PlanningProblem(add_operator().lower(), target(), {"b": 3})
    cost = problem.estimate(parameters={"tile.b": 2, "tile.r": 4})
    assert {str(s) for s in cost.work.total.free_symbols} == {"r"}
    bound = cost.subs({"r": 9})
    direct = PlanningProblem(problem.program, problem.target, {"b": 3, "r": 9}).estimate(
        parameters={"tile.b": 2, "tile.r": 4})
    assert float(bound.work.total) == float(direct.work.total)
    # 2 owner tiles * 3 chunks, each loads 2*4 FP32 inputs.
    assert float(bound.stages[0].read_bytes) == 2 * 3 * 2 * 4 * 4
    assert float(bound.work.total) >= 3 * (9 - 1)
    assert float(bound.estimated_seconds) > 0


def test_analysis_does_not_discover_hardware(monkeypatch):
    monkeypatch.setattr(CpuTarget, "detect", lambda: pytest.fail("hardware detection"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU query"))
    problem = PlanningProblem(add_operator().lower(), target(), {"b": 3, "r": 9})
    sweep = problem.sweep(max_candidates=20)
    assert sweep.shortlist(3)
    assert {e.candidate.family for e in sweep.estimates} == {"online", "fused_tree", "partial_fold"}
    assert all(float(e.cost.estimated_seconds) > 0 for e in sweep.estimates)


def test_storage_limits_rejection_records_and_json(tmp_path):
    problem = PlanningProblem(add_operator().lower(), target(max_auxiliary_bytes=1), {"b": 3, "r": 9})
    sweep = problem.sweep(max_candidates=20)
    assert not sweep.shortlist()
    assert all(e.rejection_reasons for e in sweep.estimates)
    path = tmp_path / "sweep.json"
    sweep.save(path)
    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert data["estimates"][0]["metrics"]["work"]["reduction_ops"] > 0


def test_prefix_compatibility_and_saved_storage():
    algebra = Affine(("d",))
    spec = FoldSpec({"scale": TensorSpec(("b", "r")), "value": TensorSpec(("b", "r", "d"))}, ("b",), ("r",))
    op = FoldOperator(spec, algebra, FoldMaps(lambda ctx, x: algebra.value(
        acting=StructuredValue({"value": x["scale"]}), payload=StructuredValue({"value": x["value"]}))))
    p = PlanningProblem(op.lower(), target(), {"b": 2, "r": 9, "d": 3})
    params = {"tile.b": 1, "tile.r": 2, "partition_count": 3}
    with pytest.raises(ValueError, match="artifacts"):
        p.estimate("online", parameters=params, backward=True)
    cost = p.estimate("partial_scan_checkpoints", parameters=params, backward=True)
    assert float(cost.saved_bytes) > 0
    assert float(cost.auxiliary_peak_bytes) >= float(cost.saved_bytes)
    assert [s.direction for s in cost.stages][-2:] == ["backward", "backward"]
    assert {e.candidate.family for e in p.sweep(backward=True, max_candidates=4).estimates} == {"partial_scan_checkpoints"}


def test_backward_loop_ownership_and_atomic_counts():
    p = PlanningProblem(add_operator().lower(), target(), {"b": 4, "r": 8})
    c = p.estimate(parameters={"tile.b": 2, "tile.r": 4}, backward=True)
    final = c.stages[-1]
    assert float(final.atomic_bytes) == 0
    assert float(final.write_bytes) == 4 * 8 * 4
    assert float(c.backward_seconds) > 0
    assert float(c.estimated_seconds) == pytest.approx(float(c.forward_seconds + c.backward_seconds))


def test_empirical_pipeline_validation_failures_dedup_and_plan_reload(tmp_path):
    p = PlanningProblem(add_operator().lower(), target(), {"b": 4, "r": 8})
    a = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)))
    b = replace(a, parameters=(("tile.b", 4), ("tile.r", 4)))
    sweep = p.sweep([a, a, b])
    assert len(sweep.estimates) == 2
    built = []
    def build(c):
        built.append(c)
        return c
    def validate(c):
        if c == a:
            raise AssertionError("wrong answer")
    result = tune_candidates(p, compile_candidate=build, validate=validate,
                             measure=lambda c: 0.01, sweep=sweep)
    assert result.selected == b
    assert len(built) == 2
    assert any(m.failure == "AssertionError: wrong answer" for m in result.measurements)
    path = tmp_path / "plan.json"
    result.save(path)
    assert load_candidate(p, path) == b
    with pytest.raises(ValueError, match="sizes"):
        load_candidate(replace(p, sizes={"b": 4, "r": 9}), path)


def test_cutile_source_emission_saves_carriers_for_training():
    op = add_operator()
    p = PlanningProblem(op.lower(), target("cutile"), {"b": 4, "r": 8})
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "r")
    compiled = compile_cutile_candidate(op, p, candidate)
    source = next(Path(compiled.source_directory.name).glob('*_forward.py')).read_text()
    assert "saved_execution_value" in source
    assert "ct.store(saved_execution_value" in source
    assert compiled.backward_schedule is not None


def test_cpu_candidate_training_matches_torch():
    op = add_operator()
    p = PlanningProblem(op.lower(), target(), {"b": 3, "r": 9})
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4),
        ("backward_tile.b", 1), ("backward_tile.r", 3)), True, "b")
    compiled = compile_cpu_candidate(op, p, candidate, target=CpuTarget("test", 2, 16, 64))
    x = torch.randn(3, 9, requires_grad=True)
    output = compiled(x)
    torch.testing.assert_close(output, x.sum(-1))
    gradient, = torch.autograd.grad(output, (x,), torch.ones_like(output))
    torch.testing.assert_close(gradient, torch.ones_like(x))


def test_nonfinite_rates_and_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        BackendModel("cpu", bytes_per_second=float('nan'))
    with pytest.raises(ValueError):
        PlanningProblem(add_operator().lower(), target(), {"b": 0})
    with pytest.raises(ValueError):
        ScheduleCandidate("online", (("tile.b", -1),))


def test_tunable_parameter_domains_and_budget_reporting():
    p = PlanningProblem(add_operator().lower(), target(), {"b": 4, "r": 8})
    sweep = p.sweep(backward=True, max_candidates=3,
                    parameter_domains={"embed_tile.b": (1, 2)})
    assert sweep.truncated
    assert len(sweep.estimates) == 3
    assert {dict(e.candidate.parameters)["embed_tile.b"] for e in sweep.estimates} == {1, 2}
    assert not p.sweep(max_candidates=1000).truncated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cutile_runtime_training_matches_torch():
    op = add_operator()
    problem = PlanningProblem(op.lower(), target("cutile"), {"b": 3, "r": 9})
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "b")
    compiled = compile_cutile_candidate(op, problem, candidate)
    x = torch.randn(3, 9, device="cuda", requires_grad=True)
    y = compiled(x)
    torch.testing.assert_close(y, x.sum(-1))
    gradient, = torch.autograd.grad(y, x, torch.ones_like(y))
    torch.testing.assert_close(gradient, torch.ones_like(x))


def test_cutile_online_fuses_projection_and_avoids_forward_carrier_storage():
    p = PlanningProblem(add_operator().lower(), target("cutile"), {"b": 4, "r": 8})
    params = {"tile.b": 2, "tile.r": 4}
    forward = p.estimate(parameters=params)
    assert len(forward.stages) == 1
    assert float(forward.auxiliary_peak_bytes) == 0
    assert float(forward.stages[0].write_bytes) == 4 * 4
    training = p.estimate(parameters=params, backward=True)
    assert len(training.stages) == 3
    assert float(training.stages[0].write_bytes) == 2 * 4 * 4


def test_polynomial_residency_bounds_narrow_the_tile_domain():
    p = PlanningProblem(add_operator().lower(), target(max_resident_bytes=32), {"b": 16, "r": 32})
    assert p.tile_domain_bounds() == {"b": 8, "r": 8}
    assert not tuple(p.candidates())  # Default tile domain begins at 16.
    c = ScheduleCandidate("online", (("tile.b", 16), ("tile.r", 16)))
    assert p.evaluate(c).rejection_reasons


def test_cpu_empirical_adapter_roundtrip(tmp_path):
    from cutilereduce.experimental.backends.cpu.tuning import tune_cpu
    op = add_operator()
    p = PlanningProblem(op.lower(), target(), {"b": 3, "r": 9})
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "b")
    inputs = (torch.randn(3, 9, requires_grad=True),)
    result = tune_cpu(op, dict(p.sizes), inputs, reference=lambda x: x.sum(-1),
                      target=CpuTarget("test", 2, 16, 64), analytical_target=p.target,
                      backward=True, sweep=p.sweep([candidate]), warmup=0, iterations=1)
    path = tmp_path / 'cpu-plan.json'
    result.save(path)
    assert load_candidate(result.problem, path) == candidate
    assert result.measurements[0].seconds > 0


def test_default_training_search_is_full_independent_stage_product():
    p = PlanningProblem(add_operator().lower(), target(tile_choices=(1, 2)), {"b": 2, "r": 2})
    candidates = tuple(p.candidates(backward=True, max_candidates=10000))
    pairs = set()
    for c in candidates:
        params = dict(c.parameters)
        pairs.add(tuple(params[f"{prefix}.{axis}"]
                        for prefix in ("tile", "backward_tile") for axis in ("b", "r")))
    from itertools import product
    assert pairs == set(product((1, 2), repeat=4))
    # The first diagonal covers all stage shapes before off-diagonal pairs.
    first = tuple(p.candidates(backward=True, max_candidates=len(candidates) // 4))
    for prefix in ("tile", "backward_tile"):
        assert {tuple(dict(c.parameters)[f"{prefix}.{a}"] for a in ("b", "r"))
                for c in first} == set(product((1, 2), repeat=2))
    overridden = tuple(p.candidates(backward=True, max_candidates=10000,
                                   parameter_domains={"backward_tile.r": (1,)}))
    assert {dict(c.parameters)["backward_tile.r"] for c in overridden} == {1}
    assert {dict(c.parameters)["backward_tile.b"] for c in overridden} == {1, 2}
    assert all(not any(k.startswith("backward_tile.") for k, _ in c.parameters)
               for c in p.candidates(backward=False))


def test_independent_tiles_change_only_backward_cost_and_lowering():
    from cutilereduce.experimental.backends.cpu.tuning import candidate_cpu_configs
    p = PlanningProblem(add_operator().lower(), target(), {"b": 4, "r": 9})
    params = {"tile.b": 4, "tile.r": 9}
    coupled = p.estimate("fused_tree", parameters=params, backward=True, loop_axis="b")
    independent = p.estimate("fused_tree", parameters=params | {
        "backward_tile.b": 1, "backward_tile.r": 3}, backward=True, loop_axis="b")
    assert coupled.forward_seconds == independent.forward_seconds
    assert coupled.stages[-1].resident_bytes_upper_bound != independent.stages[-1].resident_bytes_upper_bound
    c = ScheduleCandidate("fused_tree", tuple(sorted((params | {
        "backward_tile.b": 1, "backward_tile.r": 3}).items())), True, "b")
    _, forward, backward = candidate_cpu_configs(p, c)
    assert dict(forward.chunk_sizes) == {"r": 9}
    assert dict(backward.chunk_sizes) == {"r": 3}


def test_cutile_realization_does_not_emit_or_compile(monkeypatch):
    from cutilereduce.experimental.backends.cutile import runtime
    op = add_operator()
    problem = PlanningProblem(op.lower(), target("cutile"), {"b": 4, "r": 8})
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "b")
    def forbid(*args, **kwargs):
        raise AssertionError("realization must not emit or load source")
    monkeypatch.setattr(runtime, "emit_cutile_source", forbid)
    monkeypatch.setattr(runtime, "emit_cutile_backward_source", forbid)
    monkeypatch.setattr(runtime, "_load_source", forbid)
    forward, backward = runtime.realize_cutile_candidate(problem, candidate)
    assert forward.program is problem.program
    assert backward.program is problem.program
    assert dict(forward.shape) == dict(problem.sizes)
