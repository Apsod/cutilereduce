"""Backend adapters share preparation/realization without loading executable code."""

import torch

from cutilereduce.experimental import Add, FoldMaps, FoldOperator, FoldSpec, StructuredValue, TensorSpec
from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
from cutilereduce.experimental.planning.analysis import ScheduleCandidate
from cutilereduce.experimental.backends.cpu.backend import CpuBackend
from cutilereduce.experimental.planning.driver import analyze, compile_report, tune_report


def operator():
    return FoldOperator(FoldSpec({"x": TensorSpec(("b", "r"))}, ("b",), ("r",)),
                        Add(), FoldMaps(lambda ctx, x: StructuredValue({"value": x["x"]})))


def test_shared_driver_realizes_and_compiles_selected_candidate(monkeypatch):
    backend = CpuBackend()
    report = analyze(operator(), {"b": 4, "r": 8}, backend=backend,
                     hardware=CpuTarget("test", 1, 32, 64), backward=True)
    candidate = report.shortlist(1)[0].candidate
    implementation = report.realize(candidate)
    assert implementation.forward.program is report.program.program
    assert implementation.metadata.scheduled_backward is not None
    compiled = []
    def compile_stub(impl):
        compiled.append(impl)
        return lambda x: x.sum(-1)
    monkeypatch.setattr(backend, "compile", compile_stub)
    x = torch.randn(4, 8, requires_grad=True)
    torch.testing.assert_close(compile_report(report, candidate=candidate)(x), x.sum(-1))
    assert compiled[0].candidate == candidate
    class Execution:
        def measure(self, executable, inputs, **kwargs):
            assert kwargs["backward"]
            return 0.001
    result = tune_report(report, (x,), reference=lambda x: x.sum(-1), execution=Execution(),
                         backward=True, candidate_count=2)
    assert result.selected in {e.candidate for e in report.shortlist(2)}
    assert len(result.measurements) == 2


def test_cutile_adapter_realizes_without_device():
    from cutilereduce.experimental.backends.cutile.target import CuTileTarget
    from cutilereduce.experimental.backends.cutile.backend import CuTileBackend
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4)), True, "b")
    report = analyze(operator(), {"b": 4, "r": 8}, backend=CuTileBackend(),
                     hardware=CuTileTarget("test", 1), backward=True, candidates=[candidate])
    implementation = report.realize(candidate)
    assert implementation.forward.program is report.program.program
    assert implementation.backward.program is report.program.program


def test_new_backend_reuses_bound_ir_costing_without_planner_changes():
    from types import SimpleNamespace
    from dataclasses import replace
    from cutilereduce.experimental.api import compile
    from cutilereduce.experimental.planning.analysis import BackendModel
    from cutilereduce.experimental.planning.backend import BackendProgram, Implementation, BoundExecution
    from cutilereduce.experimental.planning.bound import (
        BoundPlan, BoundStage, BoundBinding, BoundLeaf, BoundLoop,
        BoundIteration, BoundLoopKind, BoundStorage,
    )
    from cutilereduce.experimental.planning.templates import (
        build_schedule_space, Iteration, MapFoldRegion, RegionCall, FoldTopology,
    )
    from cutilereduce.experimental.planning.ssa import SpecializedSchedule

    class PythonSumBackend:
        def specialize(self, program, hardware, *, sizes):
            target = BackendModel("python-sum-test", workers=1,
                supported_forward_families=("python_serial",), tile_choices=(2,),
                extra_parameter_domains={"items_per_chunk": (2, 4)},
                contraction_flops_per_second=1e9, scalar_ops_per_second=1e9,
                transcendental_ops_per_second=1e8, bytes_per_second=1e9)
            space = build_schedule_space(program)
            space = replace(space, forward=(replace(space.forward[0], name="python_serial"),))
            return BackendProgram(program, hardware, target, space, tuple(sizes.items()))

        def realize(self, prepared, candidate):
            params = dict(candidate.parameters)
            sizes = dict(prepared.sizes)
            grain = params["items_per_chunk"]
            loop = BoundLoop(BoundIteration(Iteration("r", "chunk", "chunks", "r_chunk"),
                sizes["r"], grain, (sizes["r"] + grain - 1) // grain, BoundLoopKind.SERIAL),
                BoundLeaf(MapFoldRegion()), "carrier_fold", FoldTopology.ONLINE)
            tiles = tuple((a, params[f"tile.{a}"]) for a in sizes)
            stages = (BoundStage("map", (BoundBinding("global_execution", loop, BoundStorage.MATERIALIZED),), tiles),
                      BoundStage("project", (BoundBinding("output", BoundLeaf(RegionCall("project_output", ("global_execution",))),
                                                          BoundStorage.MATERIALIZED),), tiles))
            template = prepared.schedule_space.forward[0]
            bound = BoundPlan(template, prepared.sizes, SimpleNamespace(name="python-sum"), stages, ())
            specialized = SpecializedSchedule(prepared.program, template, prepared.hardware,
                                             prepared.sizes, candidate.parameters, bound)
            return Implementation(prepared, candidate, specialized, execution=BoundExecution(bound, None, None))

        def compile(self, implementation):
            stage = implementation.execution.forward.forward[0]
            grain = stage.bindings[0].expression.iteration.grain
            batch_tile = dict(stage.tile_sizes)["b"]
            def run(x):
                pieces = []
                for b in range(0, x.shape[0], batch_tile):
                    tile = x[b:b + batch_tile]
                    state = torch.zeros(tile.shape[0], dtype=x.dtype)
                    for r in range(0, x.shape[1], grain):
                        state = state + tile[:, r:r + grain].sum(-1)
                    pieces.append(state)
                return torch.cat(pieces)
            return run

    backend = PythonSumBackend()
    hardware = SimpleNamespace(name="python")
    candidate = ScheduleCandidate("python_serial", (("tile.b", 2), ("tile.r", 2), ("items_per_chunk", 4)))
    report = analyze(operator(), {"b": 4, "r": 8}, backend=backend, hardware=hardware, candidates=[candidate])
    assert report.sweep.estimates[0].accepted
    assert report.sweep.estimates[0].cost.work.total > 0
    generated = analyze(operator(), {"b": 4, "r": 8}, backend=backend, hardware=hardware)
    assert {dict(e.candidate.parameters)["items_per_chunk"] for e in generated.sweep.estimates} == {2, 4}
    result = compile(operator(), {"b": 4, "r": 8}, backend=backend, hardware=hardware, candidate=candidate)
    x = torch.randn(4, 8)
    torch.testing.assert_close(result(x), x.sum(-1))


def test_report_compiles_retained_implementation_without_realizing_again(monkeypatch):
    backend = CpuBackend()
    report = analyze(operator(), {"b": 4, "r": 8}, backend=backend,
                     hardware=CpuTarget("test", 1, 32, 64))
    expected = report.shortlist(1, per_family=0)[0].implementation
    def forbid(*args, **kwargs):
        raise AssertionError("selection must not rerun realization")
    monkeypatch.setattr(backend, "realize", forbid)
    monkeypatch.setattr(backend, "compile", lambda implementation: implementation)
    assert compile_report(report) is expected


def test_public_compile_and_tune_native_training(tmp_path):
    from dataclasses import replace
    from cutilereduce.experimental import compile, tune
    from cutilereduce.experimental.planning.analysis import PlanningProblem
    from cutilereduce.experimental.planning.tuning import load_candidate
    import pytest
    op = operator()
    hardware = CpuTarget("test", 2, 16, 64)
    backend = CpuBackend()
    sizes = {"b": 3, "r": 9}
    candidate = ScheduleCandidate("online", (("tile.b", 2), ("tile.r", 4),
        ("backward_tile.b", 1), ("backward_tile.r", 3)), True, "b")
    inputs = (torch.randn(3, 9, requires_grad=True),)
    f = compile(op, sizes, backend=backend, hardware=hardware, backward=True, candidate=candidate)
    output = f(*inputs)
    cotangent = torch.randn_like(output)
    gradient, = torch.autograd.grad(output, inputs, cotangent)
    torch.testing.assert_close(output, inputs[0].sum(-1))
    torch.testing.assert_close(gradient, cotangent[:, None].expand_as(inputs[0]))
    result = tune(op, sizes, inputs, backend=backend, hardware=hardware, backward=True,
                  candidates=[candidate], reference=lambda x: x.sum(-1), warmup=0, iterations=1)
    path = tmp_path / "plan.json"
    result.save(path)
    assert load_candidate(result.problem, path) == result.selected
    altered = replace(hardware, worker_count=1)
    prepared = backend.specialize(op.lower(), altered, sizes=sizes)
    with pytest.raises(ValueError, match="hardware or cost model"):
        load_candidate(PlanningProblem(prepared.program, prepared.analytical_target, sizes), path)


def test_forward_only_compilation_does_not_advertise_traced_backward():
    from cutilereduce.experimental.backends.cpu.runtime import CompiledCpuFold
    prepared = CpuBackend().specialize(operator().lower(), CpuTarget("test", 1, 32, 64), sizes={"b": 4, "r": 8})
    candidate = ScheduleCandidate("online", (("tile.b", 4), ("tile.r", 8)))
    implementation = CpuBackend().realize(prepared, candidate)
    compiled = CompiledCpuFold(None, prepared.program, implementation.execution.forward, {}, 0)
    assert prepared.program.embed_output_vjp is not None
    assert not compiled.supports_backward
