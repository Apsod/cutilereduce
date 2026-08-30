from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import BufferId
from cutilereduce.core.stage_buffer import StageBuffer
from cutilereduce.fold.plan import FoldPlan, FoldStage
from cutilereduce.stages import StageFunctions


@dataclass(frozen=True)
class FoldFunctions(StageFunctions):
    map_reduce: Any = None
    combine: Any = None
    to_semantic: Any = None
    to_output: Any = None
    embed: Any = None
    finalize: Any = None
    map_backward: Any = None


def fold_functions(
        map_reduce=None,
        combine=None,
        to_semantic=None,
        to_output=None,
        *,
        embed=None,
        finalize=None,
        map_backward=None,
        ) -> FoldFunctions:
    return FoldFunctions(
        map_reduce=map_reduce,
        combine=combine,
        to_semantic=to_semantic,
        to_output=to_output,
        embed=embed,
        finalize=finalize,
        map_backward=map_backward,
    )


@dataclass(frozen=True)
class CompiledStage:
    stage: FoldStage
    kernel: Any

    @property
    def launch_grid(self) -> tuple[int, int, int]:
        return (int(self.stage.stage.program_count), 1, 1)

    @property
    def read_buffers(self) -> tuple[StageBuffer, ...]:
        return tuple(self.stage.stage.read_buffers)

    @property
    def write_buffers(self) -> tuple[StageBuffer, ...]:
        return tuple(self.stage.stage.write_buffers)

    def allocate_writes(self, device=None) -> tuple[torch.Tensor, ...]:
        return tuple(buffer.mk_empty(device=device) for buffer in self.write_buffers)


def _lookup(buffer: StageBuffer, tensors: Mapping[BufferId, torch.Tensor]) -> torch.Tensor:
    try:
        return tensors[buffer.id]
    except KeyError as err:
        raise KeyError(f"no tensor routed for buffer {buffer.id}") from err


def compile_fold_forward(plan: FoldPlan, functions: FoldFunctions) -> tuple[CompiledStage, ...]:
    return tuple(
        CompiledStage(stage=stage, kernel=stage.compile(functions))
        for stage in plan.forward
    )


def mk_map_fold_stage_kernel(stage: FoldStage, map_reduce, combine):
    return stage.compile(FoldFunctions(map_reduce=map_reduce, combine=combine))


def mk_carrier_fold_kernel(stage: FoldStage, combine, to_semantic):
    return stage.compile(FoldFunctions(combine=combine, to_semantic=to_semantic))


def mk_fold_forward(
        plan: FoldPlan,
        functions: FoldFunctions,
        *,
        device="cuda",
        ):
    compiled = compile_fold_forward(plan, functions)
    input_ids = tuple(buffer.id for buffer in plan.spec.input)
    output_ids = tuple(buffer.id for buffer in plan.spec.output)

    def forward(*inputs):
        if len(inputs) != len(input_ids):
            raise ValueError(f"expected {len(input_ids)} inputs, got {len(inputs)}")
        tensors = dict(zip(input_ids, inputs, strict=True))
        for compiled_stage in compiled:
            writes = compiled_stage.allocate_writes(device=device)
            read_args = tuple(_lookup(buffer, tensors) for buffer in compiled_stage.read_buffers)
            write_args = tuple(writes)
            ct.launch(
                torch.cuda.current_stream(),
                compiled_stage.launch_grid,
                compiled_stage.kernel,
                (read_args, write_args),
            )
            for buffer, tensor in zip(compiled_stage.write_buffers, writes, strict=True):
                tensors[buffer.id] = tensor
        return tuple(tensors[output_id] for output_id in output_ids)

    return forward


def mk_fold_autograd(plan: FoldPlan, map_reduce, combine, to_semantic, to_output, map_finalize, embed):
    raise NotImplementedError("new-core fold autograd codegen is not implemented yet")


__all__ = [
    "CompiledStage",
    "FoldFunctions",
    "compile_fold_forward",
    "fold_functions",
    "mk_carrier_fold_kernel",
    "mk_fold_autograd",
    "mk_fold_forward",
    "mk_map_fold_stage_kernel",
]
