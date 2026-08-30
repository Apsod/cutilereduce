from __future__ import annotations

from cutilereduce.stages.codegen import (
    Bundle,
    StageBufferInfo,
    StageFunctions,
    StageGridInfo,
    ctmap,
    ctzipdo,
    ctzipmap,
    inverse_p,
    retile,
)
from cutilereduce.stages.fold import compile_fold_stage
from cutilereduce.stages.map import compile_map_stage
from cutilereduce.stages.map_fold import compile_map_fold_stage
from cutilereduce.stages.map_fold_bwd import (
    compile_recompute_finalize_grad_write_stage,
    compile_recompute_fold_finalize_grad_write_stage,
)
from cutilereduce.stages.scan import compile_scan_stage


def compile_stage(stage, functions):
    return stage.compile(functions)


__all__ = [
    "Bundle",
    "StageBufferInfo",
    "StageFunctions",
    "StageGridInfo",
    "compile_fold_stage",
    "compile_map_stage",
    "compile_map_fold_stage",
    "compile_recompute_finalize_grad_write_stage",
    "compile_recompute_fold_finalize_grad_write_stage",
    "compile_scan_stage",
    "compile_stage",
    "ctmap",
    "ctzipdo",
    "ctzipmap",
    "inverse_p",
    "retile",
]
