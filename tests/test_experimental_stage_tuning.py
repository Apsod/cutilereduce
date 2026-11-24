"""Stage contracts expose tuning independence without erasing data dependencies."""

from dataclasses import replace

import pytest

from cutilereduce.experimental.planning.analysis import ScheduleCandidate
from cutilereduce.experimental.planning.stages import (
    Artifact,
    Stage,
    StagedPlan,
    best_compatible,
)


def stage(slot, value, requires, produces, **kwargs):
    candidate = ScheduleCandidate("online", (("tile.b", value), ("tile.r", 2)))
    return Stage(
        slot,
        "forward",
        (slot, value),
        tuple(requires),
        tuple(produces),
        (),
        candidate,
        **kwargs,
    )


def test_independent_stage_winners_compose_across_original_plans():
    x, carrier, output = (
        Artifact("x", ()),
        Artifact("carrier", ()),
        Artifact("output", ()),
    )
    a, b = stage("map", 1, [x], [carrier]), stage("fold", 1, [carrier], [output])
    aa, bb = stage("map", 2, [x], [carrier]), stage("fold", 2, [carrier], [output])
    plan = StagedPlan((a, b), (x,)).check()
    times = {a.key: 1, b.key: 10, aa.key: 10, bb.key: 1}
    pools = {"map": [a, aa], "fold": [b, bb]}
    assert best_compatible([plan], pools, times) == (a, bb)
    assert best_compatible([plan], pools, times, {(a.key, bb.key)}) in (
        (a, b),
        (aa, bb),
    )


def test_checkpoint_layout_prevents_incompatible_winners():
    x, output = Artifact("x", ()), Artifact("output", ())
    p1, p2 = Artifact("execution", ("prefix", 1)), Artifact("execution", ("prefix", 2))
    a, b = stage("map", 1, [x], [p1]), stage("fold", 1, [p1], [output])
    aa, bb = stage("map", 2, [x], [p2]), stage("fold", 2, [p2], [output])
    times = {a.key: 1, b.key: 10, aa.key: 10, bb.key: 1}
    result = best_compatible(
        [StagedPlan((a, b), (x,))], {"map": [a, aa], "fold": [b, bb]}, times
    )
    assert result in ((a, b), (aa, bb))
    with pytest.raises(ValueError, match="artifact"):
        StagedPlan((a, bb), (x,)).check()


def test_mutable_aliases_and_resource_couplings_are_not_independent():
    x, y, z = Artifact("x", ()), Artifact("y", ()), Artifact("z", ())
    a = stage("map", 1, [x], [y], couplings=(("placement", "node0"),))
    b = stage("fold", 1, [y], [z], couplings=(("placement", "node1"),))
    with pytest.raises(ValueError, match="coupling"):
        StagedPlan((a, b), (x,)).check()
    with pytest.raises(ValueError, match="versioned"):
        StagedPlan((replace(a, produces=(x,)),), (x,)).check()
    with pytest.raises(NotImplementedError, match="overlapping"):
        StagedPlan((a,), (x,), "overlap").check()


def test_cpu_stage_keys_ignore_unrelated_backward_decisions():
    from test_experimental_backend_contract import operator
    from cutilereduce.experimental.backends.cpu.backend import CpuBackend
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget

    backend = CpuBackend()
    prepared = backend.specialize(
        operator().lower(backward=True),
        CpuTarget("test", 2, 16, 64),
        sizes={"b": 4, "r": 8},
    )
    c = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "b")
    other = replace(
        c, parameters=(*c.parameters, ("backward_tile.b", 1)), loop_axis="r"
    )
    a, b = backend.stage_plan(prepared, c), backend.stage_plan(prepared, other)
    assert a.stages[0].key == b.stages[0].key
    assert a.stages[1].key == b.stages[1].key
    assert a.stages[2].key != b.stages[2].key
    assert tuple(s.slot for s in a.stages) == ("forward", "embed", "finalize")


def test_factorized_candidates_cover_independent_tiles_without_product():
    from test_experimental_backend_contract import operator
    from cutilereduce.experimental.backends.cpu.backend import CpuBackend
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
    from cutilereduce.experimental.planning.driver import analyze

    domains = {
        "tile.b": (1, 2),
        "tile.r": (2, 4),
        "backward_tile.b": (1, 2),
        "backward_tile.r": (2, 4),
        "loop_programs": (1,),
    }
    kwargs = dict(
        backend=CpuBackend(),
        hardware=CpuTarget("test", 1, 16, 64),
        backward=True,
        parameter_domains=domains,
    )
    full = analyze(operator(), {"b": 2, "r": 4}, **kwargs)
    factored = analyze(operator(), {"b": 2, "r": 4}, stagewise=True, **kwargs)

    def keys(report):
        return {
            s.key
            for e in report.sweep.estimates
            if e.accepted
            for s in report.backend.stage_plan(report.program, e.candidate).stages
        }

    # Baseline forward geometry may add stages, but must preserve all requested ones.
    assert keys(full) <= keys(factored)
    assert len(factored.sweep.estimates) < len(full.sweep.estimates)


@pytest.mark.parametrize("reject_first_assembly", [False, True])
def test_tuner_measures_unique_stages_and_validates_only_composed_optima(
    reject_first_assembly,
):
    from collections import Counter
    from types import SimpleNamespace
    import torch
    from test_experimental_backend_contract import operator
    from cutilereduce.experimental.backends.cpu.backend import CpuBackend
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
    from cutilereduce.experimental.backends.staging import RunnableStage
    from cutilereduce.experimental.planning.driver import analyze, tune_report

    backend = CpuBackend()
    candidates = [
        ScheduleCandidate(
            "online",
            (
                ("tile.b", n),
                ("tile.r", 2),
                ("backward_tile.b", n),
                ("backward_tile.r", 2),
            ),
            True,
            "r",
        )
        for n in (1, 2)
    ]
    report = analyze(
        operator(),
        {"b": 2, "r": 4},
        backend=backend,
        hardware=CpuTarget("test", 1, 16, 64),
        backward=True,
        candidates=candidates,
    )
    compiled, validated, measured, assembled = Counter(), Counter(), Counter(), []
    plans = [backend.stage_plan(report.program, c) for c in candidates]
    latency = {
        s.key: (1 if i == (0 if s.slot == "forward" else 1) else 9)
        for i, p in enumerate(plans)
        for s in p.stages
    }

    class Fixture:
        def validate(self, runner):
            validated[runner.spec.key] += 1

        def context(self, stage):
            return {}

    class FakeBackend:
        stage_plan = backend.stage_plan

        def stage_fixture(self, *args, **kwargs):
            return Fixture()

        def compile_stage(self, prepared, spec):
            compiled[spec.key] += 1

            def run(context):
                measured[spec.key] += 1
                return latency[spec.key]

            return RunnableStage(spec, run)

        def assemble(self, prepared, runners):
            assembled.append(tuple(r.spec for r in runners))
            fail = reject_first_assembly and len(assembled) == 1
            return lambda x: x.sum(-1) + (1 if fail else 0)

    execution = SimpleNamespace(
        measure_stage=lambda run, *args, **kwargs: run(),
        measure=lambda *args, **kwargs: 0.1,
    )
    result = tune_report(
        replace(report, backend=FakeBackend()),
        (torch.randn(2, 4, requires_grad=True),),
        reference=lambda x: x.sum(-1),
        execution=execution,
        backward=True,
        candidate_count=20,
    )
    unique = {s.key for p in plans for s in p.stages}
    assert set(compiled) == unique
    assert all(compiled[k] == validated[k] == measured[k] == 1 for k in unique)
    assert tuple(s.key for s in assembled[0]) == (
        plans[0].stages[0].key,
        plans[1].stages[1].key,
        plans[1].stages[2].key,
    )
    assert len(assembled) == (2 if reject_first_assembly else 1)
    assert result.mode == "staged"
    assert len(result.stage_measurements) == len(unique)
    assert len(result.measurements) == len(assembled)


@pytest.mark.parametrize("kind", ["attention", "prefix"])
def test_native_stages_validate_and_compose_training(kind, tmp_path):
    import torch
    from cutilereduce.experimental import (
        Affine,
        LogWeightedSum,
        FoldOperator,
        FoldSpec,
        FoldMaps,
        StructuredValue,
        TensorSpec,
        tune,
    )
    from cutilereduce.experimental.backends.cpu.backend import CpuBackend
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
    from cutilereduce.experimental.planning.tuning import load_candidate

    torch.manual_seed(41)
    if kind == "attention":
        sizes = {"h": 1, "l": 2, "g": 2, "r": 3, "dqk": 2, "dv": 3}
        spec = FoldSpec(
            {
                "q": TensorSpec(("h", "l", "g", "dqk")),
                "k": TensorSpec(("h", "r", "dqk")),
                "v": TensorSpec(("h", "r", "dv")),
            },
            ("h", "l", "g"),
            ("r",),
        )
        op = FoldOperator(
            spec,
            LogWeightedSum(("dv",)),
            FoldMaps(
                lambda ctx, x: StructuredValue(
                    {
                        "log_weight": ctx.contract(x["q"], x["k"], over=("dqk",)),
                        "value": x["v"],
                    }
                ),
                lambda ctx, x: StructuredValue({"value": x["value"]}),
            ),
        )

        def reference(q, k, v):
            scores = torch.einsum("hlgd,hrd->hlgr", q, k)
            return torch.einsum("hlgr,hrv->hlgv", scores.softmax(-1), v)

        candidate = ScheduleCandidate(
            "online",
            (
                ("tile.h", 1),
                ("tile.l", 1),
                ("tile.g", 2),
                ("tile.r", 2),
                ("backward_tile.h", 1),
                ("backward_tile.l", 2),
                ("backward_tile.g", 1),
                ("backward_tile.r", 1),
            ),
            True,
            "l",
        )
    else:
        sizes = {"b": 2, "f": 4, "d": 3}
        spec = FoldSpec(
            {"scale": TensorSpec(("b", "f")), "payload": TensorSpec(("b", "f", "d"))},
            ("b",),
            ("f",),
        )
        algebra = Affine(("d",))
        op = FoldOperator(
            spec,
            algebra,
            FoldMaps(
                lambda ctx, x: algebra.value(
                    acting=StructuredValue({"value": x["scale"]}),
                    payload=StructuredValue({"value": x["payload"]}),
                ),
                lambda ctx, x: StructuredValue({"value": x["payload"]["value"]}),
            ),
        )

        def reference(scale, payload):
            result = torch.zeros_like(payload[:, 0])
            for i in range(scale.shape[1]):
                result = scale[:, i, None].exp() * result + payload[:, i]
            return result

        candidate = ScheduleCandidate(
            "partial_scan_checkpoints",
            (
                ("tile.b", 1),
                ("tile.f", 1),
                ("backward_tile.b", 2),
                ("backward_tile.f", 1),
                ("partition_count", 2),
            ),
            True,
            "f",
        )
    inputs = tuple(
        torch.randn(*(sizes[a] for a in s.axes), requires_grad=True)
        for s in spec.inputs.values()
    )
    result = tune(
        op,
        sizes,
        inputs,
        backend=CpuBackend(),
        hardware=CpuTarget("test", 2, 16, 64),
        reference=reference,
        backward=True,
        candidates=[candidate],
        warmup=0,
        iterations=2,
    )
    assert result.mode == "staged"
    assert len(result.stage_measurements) == 3
    assert all(m.failure is None for m in result.stage_measurements)
    path = tmp_path / "plan.json"
    result.save(path)
    assert load_candidate(result.problem, path) == result.selected


def test_cutile_factorization_covers_embed_on_ragged_shapes_without_device(monkeypatch):
    from test_experimental_backend_contract import operator
    from cutilereduce.experimental.backends.cutile.backend import CuTileBackend
    from cutilereduce.experimental.backends.cutile.target import CuTileTarget
    from cutilereduce.experimental.backends.cutile import runtime
    from cutilereduce.experimental.planning.driver import analyze

    backend = CuTileBackend()
    domains = {
        "tile.b": (1, 2),
        "tile.r": (1, 2),
        "backward_tile.b": (1, 2),
        "backward_tile.r": (1, 2),
        "embed_tile.b": (1, 2),
        "loop_programs": (1,),
    }
    kwargs = dict(
        backend=backend,
        hardware=CuTileTarget("test", 2),
        backward=True,
        parameter_domains=domains,
    )
    full = analyze(operator(), {"b": 3, "r": 5}, **kwargs)
    factored = analyze(operator(), {"b": 3, "r": 5}, stagewise=True, **kwargs)

    def keys(report):
        return {
            s.key
            for e in report.sweep.estimates
            if e.accepted
            for s in backend.stage_plan(report.program, e.candidate).stages
        }

    assert keys(full) == keys(factored)
    assert len(factored.sweep.estimates) < len(full.sweep.estimates)
    assert all(e.accepted for e in factored.sweep.estimates)
    source = []
    monkeypatch.setattr(
        runtime, "_load_source", lambda text, *args: source.append(text)
    )
    plan = backend.stage_plan(factored.program, factored.sweep.estimates[0].candidate)
    runners = tuple(backend.compile_stage(factored.program, s) for s in plan.stages)
    assert tuple(r.spec.slot for r in runners) == ("forward", "embed", "finalize")
    assert len(source) == 3
    assert "saved_execution" in source[0]
