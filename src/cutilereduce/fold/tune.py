from __future__ import annotations

from cutilereduce.core import symbolic as expr
import torch
from cuda.tile.tune import exhaustive_search

from cutilereduce.core.variables import PEAK_TENSOR_FLOPS
from cutilereduce.fold.impl import CompiledStage


def stage_key(stage):
    return (
        stage.stage.name,
        stage.stage.domain.loop,
        stage.stage.domain.task_grid,
        tuple(
            (axis.name, axis.source, axis.programs)
            for axis in stage.stage.domain.program_axes
        ),
        tuple(
            (axis.name, axis.extent, axis.tile)
            for axis in stage.stage.domain.compute_axes
        ),
    )


def _evaluate_scalar(value, symbols):
    if isinstance(value, expr.Expr):
        value = value.subs(symbols)
    return float(value)


def empirical_model_flops_utilization(stage, *, hardware, time_us):
    if time_us <= 0:
        return 0.0
    flops = _evaluate_scalar(stage.stage.cost.total_work, hardware)
    peak = _evaluate_scalar(PEAK_TENSOR_FLOPS, hardware)
    if peak <= 0:
        return 0.0
    return flops / (time_us * 1e-6 * peak)


def format_stage_timing(prefix, stage, time_us, *, hardware, suffix=""):
    tiles = ", ".join(
        f"{axis.name}={axis.tile}" for axis in stage.stage.domain.compute_axes
    )
    mfu = empirical_model_flops_utilization(
        stage,
        hardware=hardware,
        time_us=time_us,
    )
    return (
        f"  {prefix} {stage.stage.name} ({tiles}{suffix}): "
        f"{time_us:.1f} us, {100 * mfu:.1f}% MFU"
    )


def tune_fold_stages(stages, functions, *, timeout=0, quiet=False):
    unique = {stage_key(stage): stage for stage in stages}
    compiled = {
        key: CompiledStage(stage=stage, kernel=stage.compile(functions))
        for key, stage in unique.items()
    }
    tensor_pool = {}

    def tensor_for(buffer, *, zero):
        key = (buffer.id, buffer.total.shape, buffer.torch_dtype, zero)
        if key not in tensor_pool:
            if zero:
                tensor_pool[key] = buffer.mk_zeros(device="cuda")
            elif buffer.default is None:
                tensor_pool[key] = buffer.mk_empty(device="cuda")
            else:
                tensor_pool[key] = buffer.mk_default(device="cuda")
        return tensor_pool[key]

    arguments = {
        key: (
            tuple(tensor_for(buffer, zero=False) for buffer in current.read_buffers),
            tuple(tensor_for(buffer, zero=True) for buffer in current.write_buffers),
        )
        for key, current in compiled.items()
    }
    result = exhaustive_search(
        tuple(compiled),
        torch.cuda.current_stream(),
        grid_fn=lambda key: compiled[key].launch_grid,
        kernel=lambda key: compiled[key].kernel,
        args_fn=arguments.__getitem__,
        # Keep reporting here; the native tuner also prints every result summary.
        quiet=True,
        single_run_timeout_sec=None if timeout <= 0 else timeout,
    )
    for key, error_type, message in result.failures:
        detail = " | ".join(
            line.strip() for line in message.splitlines()[:4] if line.strip()
        )
        print(
            f"  skipped {key}: {error_type.__name__}: {detail}",
            flush=True,
        )
    return {measurement.config: measurement.mean_us for measurement in result.successes}


def tune_fold_plan(
    spec,
    functions,
    sizes,
    candidate_count=20,
    timeout=0,
    *,
    hardware,
    backward=True,
    max_storage_ratio=4.0,
    quiet=False,
):
    """Search and benchmark concrete instances of the plan paths."""
    from cutilereduce.fold.plan import AlgebraKind

    if spec.algebra == AlgebraKind.commutative:
        from cutilereduce.fold.commutative.tune import tune_commutative_fold_plan

        return tune_commutative_fold_plan(
            spec,
            sizes,
            candidate_count,
            timeout,
            functions=functions,
            hardware=hardware,
            max_storage_ratio=max_storage_ratio,
            quiet_tuning=quiet,
            backward=backward,
        )
    if spec.algebra == AlgebraKind.general:
        from cutilereduce.fold.general.tune import tune_general_fold_plan

        return tune_general_fold_plan(
            spec,
            sizes,
            candidate_count,
            timeout,
            functions=functions,
            hardware=hardware,
            max_storage_ratio=max_storage_ratio,
            quiet_tuning=quiet,
            backward=backward,
        )
    raise ValueError(f"unsupported fold algebra: {spec.algebra}")


__all__ = [
    "empirical_model_flops_utilization",
    "format_stage_timing",
    "stage_key",
    "tune_fold_stages",
    "tune_fold_plan",
]
