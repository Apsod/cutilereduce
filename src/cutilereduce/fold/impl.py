from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
from itertools import count
from typing import Any

import cuda.tile as ct
import torch

from cutilereduce.core.buffer import BufferId, BufferRole
from cutilereduce.core.stage_buffer import StageBuffer
from cutilereduce.fold.plan import FoldPlan, FoldStage
from cutilereduce.stages import StageFunctions


_CUSTOM_OP_IDS = count()


@dataclass(frozen=True)
class FoldFunctions(StageFunctions):
    pass


def fold_functions(
        map_reduce=None,
        combine=None,
        to_semantic=None,
        to_output=None,
        *,
        map_reduce_combine=None,
        embed=None,
        finalize=None,
        map_backward=None,
        map_reduce_backward=None,
        ) -> FoldFunctions:
    if finalize is not None and map_reduce_backward is not None:
        raise ValueError("pass finalize, not both finalize and map_reduce_backward")
    return FoldFunctions(
        map_reduce=map_reduce,
        map_reduce_combine=map_reduce_combine,
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


def _run_compiled_stages(stages, tensors, *, device, zero_writes):
    for compiled_stage in stages:
        writes = (
            compiled_stage.allocate_zero_writes(device=device)
            if zero_writes else compiled_stage.allocate_writes(device=device)
        )
        read_args = tuple(
            _lookup(buffer, tensors) for buffer in compiled_stage.read_buffers
        )
        ct.launch(
            torch.cuda.current_stream(),
            compiled_stage.launch_grid,
            compiled_stage.kernel,
            (read_args, writes),
        )
        tensors.update(
            (buffer.id, tensor)
            for buffer, tensor in zip(
                compiled_stage.write_buffers, writes, strict=True,
            )
        )
    return tensors


def _stage_buffer_map(stages):
    return {
        buffer.id: buffer
        for stage in stages
        for buffer in (*stage.read_buffers, *stage.write_buffers)
    }


def _fake_tensor_like(reference, buffer):
    return reference.new_empty(
        tuple(int(size) for size in buffer.total.shape),
        dtype=buffer.torch_dtype,
    )


def _custom_op_digest(plan, functions):
    function_sources = []
    for name in (
            "map_reduce", "map_reduce_combine", "combine", "to_semantic", "embed", "finalize",
            "map_backward", "map_reduce_backward",
            ):
        function = getattr(functions, name)
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = repr(function)
        function_sources.append((name, source))
    stage_contracts = tuple(
        (
            stage.stage.name,
            stage.stage.domain,
            tuple((buffer.id, buffer.total.shape, buffer.dtype) for buffer in stage.stage.buffers),
        )
        for stage in (*plan.forward, *plan.backward)
    )
    contract = repr((plan.spec, stage_contracts, function_sources)).encode()
    return hashlib.sha256(contract).hexdigest()[:16]


def _make_fold_custom_ops(plan, functions):
    """Register functional whole-plan ops understood by torch.compile."""
    if not plan.backward:
        raise ValueError("fold custom autograd requires backward stages")
    forward_stages = compile_fold_forward(plan, functions)
    backward_stages = compile_fold_backward(plan, functions)
    input_ids = tuple(buffer.id for buffer in plan.spec.input)
    output_ids = tuple(buffer.id for buffer in plan.spec.output)
    output_grad_ids = tuple(
        buffer.id
        for buffer in backward_stages[0].read_buffers
        if _is_output_grad(buffer)
    )
    required_grad_inputs = tuple(
        buffer.id for buffer in plan.spec.input if buffer.req_grad
    )
    saved_forward_ids = tuple(dict.fromkeys(
        buffer.id
        for stage in backward_stages
        for buffer in stage.read_buffers
        if buffer.id not in input_ids
        and buffer.id not in output_ids
        and not _is_output_grad(buffer)
    ))
    forward_buffers = _stage_buffer_map(forward_stages)
    backward_buffers = _stage_buffer_map(backward_stages)
    op_id = f"{_custom_op_digest(plan, functions)}_{next(_CUSTOM_OP_IDS)}"

    @torch.library.custom_op(
        f"cutilereduce::fold_forward_{op_id}",
        mutates_args=(),
        device_types="cuda",
        schema="(Tensor[] inputs) -> Tensor[]",
    )
    def forward_op(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs) != len(input_ids):
            raise ValueError(f"expected {len(input_ids)} inputs, got {len(inputs)}")
        tensors = dict(zip(input_ids, inputs, strict=True))
        _run_compiled_stages(
            forward_stages,
            tensors,
            device=inputs[0].device,
            zero_writes=False,
        )
        return [
            tensors[buffer_id]
            for buffer_id in (*output_ids, *saved_forward_ids)
        ]

    @forward_op.register_fake
    def _(inputs):
        reference = inputs[0]
        return [
            _fake_tensor_like(reference, forward_buffers[buffer_id])
            for buffer_id in (*output_ids, *saved_forward_ids)
        ]

    @torch.library.custom_op(
        f"cutilereduce::fold_backward_{op_id}",
        mutates_args=(),
        device_types="cuda",
        schema="(Tensor[] saved, Tensor[] grad_outputs) -> Tensor[]",
    )
    def backward_op(
            saved: list[torch.Tensor],
            grad_outputs: list[torch.Tensor],
            ) -> list[torch.Tensor]:
        saved_ids = (*input_ids, *output_ids, *saved_forward_ids)
        tensors = dict(zip(saved_ids, saved, strict=True))
        tensors.update(zip(output_grad_ids, grad_outputs, strict=True))
        _run_compiled_stages(
            backward_stages,
            tensors,
            device=saved[0].device,
            zero_writes=True,
        )
        return [tensors[buffer_id.as_input_grad] for buffer_id in required_grad_inputs]

    @backward_op.register_fake
    def _(saved, grad_outputs):
        del grad_outputs
        reference = saved[0]
        return [
            _fake_tensor_like(reference, backward_buffers[buffer_id.as_input_grad])
            for buffer_id in required_grad_inputs
        ]

    def setup_context(ctx, inputs, output):
        original_inputs, = inputs
        ctx.save_for_backward(*original_inputs, *output)
        ctx.mark_non_differentiable(*output[len(output_ids):])

    def custom_backward(ctx, grad_outputs):
        saved = list(ctx.saved_tensors)
        semantic_outputs = saved[len(input_ids):len(input_ids) + len(output_ids)]
        materialized_grads = [
            grad if grad is not None else torch.zeros_like(output)
            for grad, output in zip(
                grad_outputs[:len(output_ids)], semantic_outputs, strict=True,
            )
        ]
        gradients = iter(backward_op(saved, materialized_grads))
        return [next(gradients) if buffer.req_grad else None for buffer in plan.spec.input]

    torch.library.register_autograd(
        forward_op,
        custom_backward,
        setup_context=setup_context,
    )
    return forward_op, backward_op


def mk_fold_autograd(
        plan: FoldPlan,
        functions: FoldFunctions,
        *,
        device="cuda",
        ):
    del device
    input_ids = tuple(buffer.id for buffer in plan.spec.input)
    output_ids = tuple(buffer.id for buffer in plan.spec.output)
    forward_op, backward_op = _make_fold_custom_ops(plan, functions)

    def apply(*inputs):
        if len(inputs) != len(input_ids):
            raise ValueError(f"expected {len(input_ids)} inputs, got {len(inputs)}")
        semantic_outputs = tuple(forward_op(list(inputs))[:len(output_ids)])
        return _apply_to_output(functions, semantic_outputs)

    apply.forward_op = forward_op
    apply.backward_op = backward_op
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
