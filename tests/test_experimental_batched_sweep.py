"""Batched costs match scalar realized-IR costs without realizing every row."""

from dataclasses import fields, replace

import numpy as np
import pytest

from cutilereduce.experimental import CpuTarget
from cutilereduce.experimental.backends.cpu.backend import CpuBackend
from cutilereduce.experimental.backends.cutile.backend import CuTileBackend
from cutilereduce.experimental.backends.cutile.analysis import cutile_analytical_target
from cutilereduce.experimental.backends.cutile.target import CuTileTarget
from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.batched import DeferredImplementation, resolve_implementation
from test_experimental_cost_parity import cases


@pytest.mark.parametrize("case", list(cases()), ids=lambda c: c[0])
def test_batch_matches_scalar_all_stage_metrics(case):
    _, op, sizes, family, parameters, loop = case
    from cutilereduce.experimental.planning.analysis import ScheduleCandidate
    backend = CpuBackend()
    prepared = backend.specialize(op.lower(), CpuTarget("test", 2, 32, 64), sizes=sizes)
    p = PlanningProblem(prepared.program, prepared.analytical_target, sizes)
    candidates = [ScheduleCandidate(family, tuple(sorted(parameters.items())), True, loop)]
    candidates.extend(list(p.candidates(backward=True, max_candidates=100,
        parameter_domains={f"tile.{p.program.spec.fold_axes[0]}": (2, sizes[p.program.spec.fold_axes[0]])})))
    batched = p.sweep(candidates, batch_size=17)
    scalar = p.sweep(candidates, batched=False)
    assert len(batched.estimates) == len(scalar.estimates)
    for a, b in zip(batched.estimates, scalar.estimates, strict=True):
        assert a.candidate == b.candidate
        assert a.accepted == b.accepted
        if not a.accepted:
            continue
        for metric in ("estimated_seconds", "resident_bytes_upper_bound", "saved_bytes", "auxiliary_peak_bytes"):
            np.testing.assert_allclose(getattr(a.cost, metric), float(getattr(b.cost, metric)), rtol=1e-12)
        for sa, sb in zip(a.cost.stages, b.cost.stages, strict=True):
            for f in fields(sa):
                if f.name in {"name", "direction", "work", "invocations", "accounting_domain"}:
                    continue
                np.testing.assert_allclose(float(getattr(sa, f.name)), float(getattr(sb, f.name)), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(a.cost.work.total, float(b.cost.work.total), rtol=1e-12)


def test_only_structural_representatives_and_finalists_are_realized():
    _, op, sizes, _, _, _ = next(cases())
    prepared = CpuBackend().specialize(op.lower(), CpuTarget("test", 2, 32, 64), sizes=sizes)
    calls = []
    original = prepared.analytical_target.realization_factory
    def counted(problem, candidate):
        calls.append(candidate)
        return original(problem, candidate)
    target = replace(prepared.analytical_target, realization_factory=counted, tile_choices=(1, 2, 4))
    p = PlanningProblem(prepared.program, target, sizes)
    sweep = p.sweep(backward=True, max_candidates=1000, batch_size=23)
    assert len(calls) < len(sweep.estimates) // 5
    finalists = sweep.shortlist(3, per_family=0)
    before = len(calls)
    assert all(isinstance(e.implementation, DeferredImplementation) for e in finalists)
    for e in finalists:
        assert resolve_implementation(e.implementation) is resolve_implementation(e.implementation)
    assert len(calls) == before + len(finalists)


def test_gpu_batch_and_capacity_rejections():
    _, op, _, _, _, _ = next(cases())
    target = cutile_analytical_target(workers=2, contraction_flops_per_second=1e12,
        scalar_ops_per_second=1e10, transcendental_ops_per_second=1e9, bytes_per_second=1e10,
        tile_choices=(1, 2, 4), max_auxiliary_bytes=1)
    backend = CuTileBackend(analytical_target=target)
    prepared = backend.specialize(op.lower(), CuTileTarget("test", 2), sizes={"b": 4, "r": 8})
    p = PlanningProblem(prepared.program, prepared.analytical_target, dict(prepared.sizes))
    a, b = p.sweep(backward=True), p.sweep(backward=True, batched=False)
    assert [e.accepted for e in a.estimates] == [e.accepted for e in b.estimates]
    for x, y in zip(a.estimates, b.estimates, strict=True):
        np.testing.assert_allclose(x.cost.estimated_seconds, float(y.cost.estimated_seconds), rtol=1e-12)


def test_chunking_dedup_truncation_and_scalar_fallback_are_explicit(tmp_path):
    _, op, sizes, _, _, _ = next(cases())
    prepared = CpuBackend().specialize(op.lower(), CpuTarget("test", 1, 32, 64), sizes=sizes)
    p = PlanningProblem(prepared.program, prepared.analytical_target, sizes)
    assert p.sweep(backward=True, max_candidates=3, batch_size=2).truncated
    candidates = list(p.candidates(backward=True, max_candidates=10))
    a = p.sweep(candidates + candidates, batch_size=1)
    b = p.sweep(candidates, batch_size=100)
    assert len(a.estimates) == len(b.estimates) == len({p.canonical_candidate(c).key for c in candidates})
    assert a.evaluation_metadata['scalar_fallback_candidates'] == 0
    np.testing.assert_allclose([e.cost.estimated_seconds for e in a.estimates],
                               [e.cost.estimated_seconds for e in b.estimates], rtol=1e-12)
    fallback = replace(p, target=replace(p.target, batch_key=None)).sweep(candidates)
    assert fallback.evaluation_metadata['mode'] == 'scalar'
    a.save(tmp_path / "batched.json")
    import json
    report = json.loads((tmp_path / "batched.json").read_text())
    assert len(report["estimates"]) == len(a.estimates)
    assert report["evaluation"]["mode"] == "batched"


def test_multi_metric_numeric_cse_avoids_object_arrays():
    from cutilereduce.core import symbolic as sym
    x = sym.Symbol('x')
    program = sym.cse_many((x / (x + 8), x / 3, 2 * x))
    values = np.asarray([1, 4, 16])
    results = program.evaluate({'x': values})
    for result in results:
        assert result.dtype != object
    np.testing.assert_allclose(results[0], values / (values + 8))
    np.testing.assert_allclose(results[1], values / 3)
    np.testing.assert_allclose(results[2], 2 * values)


def test_lazy_forward_selection_preserves_explicit_backward(monkeypatch):
    from types import SimpleNamespace
    import torch.utils.cpp_extension
    from cutilereduce.experimental.backends.cpu.runtime import compile_cpu
    from cutilereduce.experimental.planning.scheduled import schedule_backward_program
    _, op, sizes, _, _, _ = next(cases())
    backward = schedule_backward_program(op.lower(), sizes, loop_axis="b", tiles={"b": 2, "r": 3})
    monkeypatch.setattr(torch.utils.cpp_extension, 'load_inline', lambda *a, **kw: SimpleNamespace())
    compiled = compile_cpu(op, sizes, backward=True, backward_program=backward,
                           target=CpuTarget('test', 1, 32, 64))
    assert compiled.backward_program is backward
    assert compiled.supports_backward
