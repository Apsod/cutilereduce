from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import BufferId, BufferRole
from cutilereduce.core.stage_buffer import StageBuffer
from cutilereduce.fold.plan import FoldPlan, FoldStage
from cutilereduce.stages import StageFunctions


@dataclass(frozen=True)
class FoldFunctions(StageFunctions):
    pass


def fold_functions(
        map_reduce=None,
        combine=None,
        to_semantic=None,
        to_output=None,
        *,
        embed=None,
        finalize=None,
        map_backward=None,
        map_reduce_backward=None,
        ) -> FoldFunctions:
    return FoldFunctions(
        map_reduce=map_reduce,
        combine=combine,
        to_semantic=to_semantic,
        to_output=to_output,
        embed=embed,
        finalize=finalize,
        map_backward=map_backward,
        map_reduce_backward=map_reduce_backward,
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

    def allocate_zero_writes(self, device=None) -> tuple[torch.Tensor, ...]:
        return tuple(buffer.mk_zeros(device=device) for buffer in self.write_buffers)


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


def compile_fold_backward(plan: FoldPlan, functions: FoldFunctions) -> tuple[CompiledStage, ...]:
    return tuple(
        CompiledStage(stage=stage, kernel=stage.compile(functions))
        for stage in plan.backward
    )


def _as_tuple(outputs):
    return outputs if isinstance(outputs, tuple) else (outputs,)


def _apply_to_output(functions: FoldFunctions, semantic_outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if functions.to_output is None:
        return semantic_outputs
    return _as_tuple(functions.to_output(*semantic_outputs))


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
        semantic_outputs = tuple(tensors[output_id] for output_id in output_ids)
        return _apply_to_output(functions, semantic_outputs)

    return forward


def _is_output_grad(buffer: StageBuffer) -> bool:
    return buffer.role == BufferRole.OutputGrad


def mk_fold_autograd(
        plan: FoldPlan,
        functions: FoldFunctions,
        *,
        device="cuda",
        ):
    if not plan.backward:
        raise ValueError("fold autograd requires a plan with backward stages")
    forward_stages = compile_fold_forward(plan, functions)
    backward_stages = compile_fold_backward(plan, functions)
    input_ids = tuple(buffer.id for buffer in plan.spec.input)
    output_ids = tuple(buffer.id for buffer in plan.spec.output)
    output_grad_ids = tuple(
        buffer.id
        for buffer in backward_stages[0].read_buffers
        if _is_output_grad(buffer)
    )
    grad_storage_ids = {
        buffer.id.as_input_grad
        for buffer in plan.spec.input
        if buffer.req_grad
    }

    def _run_forward(inputs):
        tensors = dict(zip(input_ids, inputs, strict=True))
        for compiled_stage in forward_stages:
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
        semantic_outputs = tuple(tensors[output_id] for output_id in output_ids)
        return tensors, semantic_outputs

    class CutileFoldFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            if len(inputs) != len(input_ids):
                raise ValueError(f"expected {len(input_ids)} inputs, got {len(inputs)}")
            tensors, semantic_outputs = _run_forward(inputs)
            ctx.save_for_backward(*inputs, *semantic_outputs)
            return semantic_outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            saved = ctx.saved_tensors
            inputs = saved[:len(input_ids)]
            outputs = saved[len(input_ids):]
            tensors = dict(zip(input_ids, inputs, strict=True))
            tensors.update(zip(output_ids, outputs, strict=True))
            tensors.update(zip(output_grad_ids, grad_outputs, strict=True))
            for compiled_stage in backward_stages:
                writes = compiled_stage.allocate_zero_writes(device=device)
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
            return tuple(
                tensors[input_id.as_input_grad] if input_id.as_input_grad in grad_storage_ids else None
                for input_id in input_ids
            )

    def apply(*inputs):
        semantic_outputs = _as_tuple(CutileFoldFunction.apply(*inputs))
        return _apply_to_output(functions, semantic_outputs)

    return apply


__all__ = [
    "CompiledStage",
    "FoldFunctions",
    "compile_fold_backward",
    "compile_fold_forward",
    "fold_functions",
    "mk_fold_autograd",
    "mk_fold_forward",
]
