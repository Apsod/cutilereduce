"""Reusable stage fixtures and autograd composition, independent of CPU/GPU lowering."""

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable

from .reference import _execute_region


@dataclass
class RunnableStage:
    spec: Any
    run: Callable
    owner: Any = None


class StageFixture:
    def __init__(self, prepared, inputs, reference, *, backward, rtol, atol):
        import torch

        self.prepared, self.inputs = prepared, inputs
        self.rtol, self.atol = rtol, atol
        self.contexts = {}
        values = tuple(
            x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
        )
        expected = reference(*values)
        self.outputs = expected if isinstance(expected, tuple) else (expected,)
        if len(self.outputs) != len(prepared.program.project_output.results):
            raise ValueError("reference output arity does not match the program")
        self.cotangents = tuple(
            torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=torch.Generator(device=x.device).manual_seed(107 + i),
            )
            for i, x in enumerate(self.outputs)
        )
        self.gradients = [None] * len(inputs)
        if backward:
            selected = [(i, x) for i, x in enumerate(values) if x.requires_grad]
            if not selected:
                raise ValueError(
                    "backward stage validation requires differentiable inputs"
                )
            gradients = torch.autograd.grad(
                self.outputs,
                tuple(x for _, x in selected),
                self.cotangents,
                allow_unused=True,
            )
            for (i, _), gradient in zip(selected, gradients, strict=True):
                self.gradients[i] = (
                    torch.zeros_like(values[i]) if gradient is None else gradient
                )
        self.outputs = tuple(x.detach() for x in self.outputs)

    def context(self, stage):
        import torch

        program = self.prepared.program
        sizes = dict(self.prepared.sizes)
        partitions = (
            dict(stage.representative.parameters).get("partition_count", 1)
            if program.requires_prefix
            else 1
        )
        if partitions in self.contexts:
            return self.contexts[partitions]
        values = dict(zip(program.spec.inputs, self.inputs, strict=True))
        with torch.no_grad():
            execution = _execute_region(program.map_fold, values, axis_sizes=sizes)
            if program.requires_prefix:
                axis = program.spec.fold_axes[0]
                grain = (sizes[axis] + partitions - 1) // partitions
                identity = _execute_region(
                    program.execution_identity,
                    {"_device": self.inputs[0]},
                    axis_sizes=sizes,
                )
                state = {
                    name: identity[name]
                    .reshape(
                        (1,) * (execution[name].ndim - identity[name].ndim)
                        + tuple(identity[name].shape)
                    )
                    .expand_as(execution[name])
                    .clone()
                    for name in execution
                }
                for i, start in enumerate(range(0, sizes[axis], grain)):
                    for name, tensor in state.items():
                        execution[
                            f"prefix_execution.{i}.{name.removeprefix('execution.')}"
                        ] = tensor
                    sliced = {
                        n: x.narrow(
                            spec.axes.index(axis),
                            start,
                            min(grain, sizes[axis] - start),
                        ).contiguous()
                        if axis in spec.axes
                        else x
                        for (n, spec), x in zip(
                            program.spec.inputs.items(), self.inputs, strict=True
                        )
                    }
                    part = _execute_region(program.map_fold, sliced)
                    combine = {
                        f"{side}.{n.removeprefix('execution.')}": t
                        for side, source in (("left", state), ("right", part))
                        for n, t in source.items()
                    }
                    state = _execute_region(program.execution_combine, combine)
            accumulator = {}
            if program.embed_output_vjp is not None:
                arguments = (
                    values
                    | execution
                    | {
                        f"grad_output.{n}": t
                        for (n, _), t in zip(
                            program.project_output.results, self.cotangents, strict=True
                        )
                    }
                )
                arguments.update(
                    {
                        n.replace("execution.", "saved_execution.", 1): t
                        for n, t in execution.items()
                        if n.startswith("execution.")
                    }
                )
                accumulator = _execute_region(
                    program.embed_output_vjp, arguments, axis_sizes=sizes
                )
        result = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "execution": execution,
            "carrier": {
                n: t for n, t in execution.items() if n.startswith("execution.")
            },
            "cotangents": self.cotangents,
            "accumulator": accumulator,
            "gradients": tuple(self.gradients),
        }
        self.contexts[partitions] = result
        return result

    def validate(self, executable):
        import torch

        context = self.context(executable.spec)
        # Stages must not mutate caller tensors or reusable upstream artifacts.
        protected = [
            *context["inputs"],
            *context["execution"].values(),
            *context["accumulator"].values(),
        ]
        before = [t.clone() for t in protected]
        with torch.no_grad():
            actual = executable.run(context)

        def compare(a, b):
            if isinstance(b, dict):
                if set(a) != set(b):
                    raise AssertionError("stage artifact fields differ")
                for name in b:
                    compare(a[name], b[name])
            elif isinstance(b, (tuple, list)):
                if len(a) != len(b):
                    raise AssertionError("stage result arity differs")
                for x, y in zip(a, b, strict=True):
                    compare(x, y)
            elif b is not None:
                torch.testing.assert_close(a, b, rtol=self.rtol, atol=self.atol)

        for artifact in executable.spec.produces:
            compare(actual[artifact.name], context[artifact.name])
        for tensor, old in zip(protected, before, strict=True):
            torch.testing.assert_close(tensor, old, rtol=0, atol=0, equal_nan=True)


class ComposedFold:
    def __init__(self, prepared, stages):
        self.prepared, self.stages = prepared, tuple(stages)
        self.problem = prepared.program
        self.supports_backward = any(s.spec.direction == "backward" for s in stages)

    def _run(self, direction, context):
        for stage in self.stages:
            if stage.spec.direction == direction:
                context.update(stage.run(context))
        return context

    def __call__(self, *inputs):
        if self.supports_backward and any(x.requires_grad for x in inputs):
            return _autograd().apply(self, *inputs)
        outputs = self._run("forward", {"inputs": inputs})["outputs"]
        return outputs[0] if len(outputs) == 1 else outputs


@cache
def _autograd():
    import torch

    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, compiled, *inputs):
            values = compiled._run("forward", {"inputs": inputs})
            execution = values.get("execution", {})
            ctx.compiled, ctx.names, ctx.input_count = (
                compiled,
                tuple(execution),
                len(inputs),
            )
            ctx.save_for_backward(*inputs, *execution.values())
            outputs = values["outputs"]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        @staticmethod
        def backward(ctx, *cotangents):
            values = ctx.saved_tensors
            execution = dict(zip(ctx.names, values[ctx.input_count :], strict=True))
            context = {
                "inputs": values[: ctx.input_count],
                "execution": execution,
                "carrier": {
                    n: t for n, t in execution.items() if n.startswith("execution.")
                },
                "cotangents": cotangents,
            }
            result = ctx.compiled._run("backward", context)
            return (None, *result["gradients"])

    return Function
