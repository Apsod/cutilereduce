from __future__ import annotations

import torch

from cutilereduce.fold.impl import CompiledStage
from cutilereduce.util.tune import exhaustive_search


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
        kernel_fn=lambda key: compiled[key].kernel,
        args_fn=arguments.__getitem__,
        quiet=quiet,
        single_run_timeout_sec=None if timeout <= 0 else timeout,
    )
    for key, error_type, message in result.failures:
        detail = " | ".join(
            line.strip()
            for line in message.splitlines()[:4]
            if line.strip()
        )
        print(
            f"  skipped {key}: {error_type.__name__}: {detail}",
            flush=True,
        )
    return {measurement.config: measurement.mean_us for measurement in result.successes}


__all__ = ["stage_key", "tune_fold_stages"]
