"""The SDPA benchmark uses the same scaling, shapes and gradients as eager."""

import pytest
import torch

from examples.experimental_cpu_single_tile import reference
from examples.experimental_cpu_tiles import sdpa_reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value_dimension", [4, 3])
def test_sdpa_benchmark_reference_matches_eager_vjp(dtype, value_dimension):
    generator = torch.Generator().manual_seed(41)
    inputs = (
        (torch.randn(2, 5, 1, 4, dtype=dtype, generator=generator) / 2).requires_grad_(),
        torch.randn(2, 7, 4, dtype=dtype, generator=generator, requires_grad=True),
        torch.randn(2, 7, value_dimension, dtype=dtype, generator=generator, requires_grad=True),
    )
    actual, expected = sdpa_reference(*inputs), reference(*inputs)
    cotangent = torch.randn(expected.shape, dtype=dtype, generator=generator)
    torch.testing.assert_close(actual, expected)
    ag = torch.autograd.grad(actual, inputs, cotangent)
    eg = torch.autograd.grad(expected, inputs, cotangent)
    for a, e in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, e)


def test_benchmark_tunes_each_thread_count_and_compares_only_its_winner(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import json
    from dataclasses import asdict
    from examples import experimental_cpu_tiles as bench
    from cutilereduce.experimental.backends.cpu.schedule import CpuTarget
    from cutilereduce.experimental.planning.stage_tuning import StageMeasurement
    from cutilereduce.experimental.planning.tuning import TuningResult, TuningMeasurement

    hardware = CpuTarget('test', 2, 16, 64)
    monkeypatch.setattr(bench, 'hardware_report', lambda: (hardware, asdict(hardware)))
    class Fixed:
        backward_program = SimpleNamespace(gradients=())
        def __call__(self, *inputs):
            return reference(*inputs)
    monkeypatch.setattr(bench, 'compile_cpu_candidate', lambda *args, **kwargs: Fixed())
    tuning_calls, winner_calls = [], []
    def tune(report, inputs, **kwargs):
        threads = torch.get_num_threads()
        assert report.program.hardware.worker_count == threads
        assert report.sweep.evaluation_metadata['search_space'] == 'stage_union'
        assert kwargs['backward'] and kwargs['candidate_count'] == 2
        assert kwargs['iterations'] == 1 and kwargs['warmup'] == 0
        tuning_calls.append(threads)
        estimate = next(e for e in report.sweep.estimates if e.accepted)
        stages = report.backend.stage_plan(report.program, estimate.candidate).stages
        def winner(*values):
            winner_calls.append((threads, torch.get_num_threads()))
            return reference(*values)
        return TuningResult(winner, estimate.candidate,
            (TuningMeasurement(estimate.candidate, 0.003),), report.problem,
            stage_measurements=tuple(StageMeasurement(s.key, s.slot, s.representative,
                0.001, 0.001) for s in stages), selected_stages=stages, mode='staged')
    monkeypatch.setattr(bench, 'tune_report', tune)
    path = tmp_path / 'benchmark.json'
    original_threads = torch.get_num_threads()
    try:
        bench.main(['--length', '3', '--dimension', '2', '--threads', '1', '2',
                    '--only', 'full', '--iterations', '1', '--warmup', '0',
                    '--tune-shortlist', '2', '--tune-iterations', '1', '--tune-warmup', '0',
                    '--max-candidates', '16', '--json', str(path)])
    finally:
        torch.set_num_threads(original_threads)
    assert tuning_calls == [1, 2]
    assert winner_calls and all(selected == measured for selected, measured in winner_calls)
    report = json.loads(path.read_text())
    for thread in ('1', '2'):
        tuning = report['tuning_searches'][thread]
        assert tuning['mode'] == 'staged'
        assert {s['slot'] for s in tuning['selected_stages']} == {'forward', 'embed', 'finalize'}
        assert len(tuning['stage_measurements']) == 3
        for timings in report['results'][thread].values():
            assert set(timings) == {'full', 'torch_eager', 'torch_sdpa', f'tuned_t{thread}'}
