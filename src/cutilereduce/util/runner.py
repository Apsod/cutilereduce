
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import gc
import torch
import torch.utils.benchmark as torch_benchmark


def as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def differentiable_inputs(inputs):
    return tuple(tensor for tensor in inputs if tensor.requires_grad)


def make_output_grads(outputs, *, dtype=None):
    return tuple(
        torch.randn(
            output.shape,
            device=output.device,
            dtype=dtype or output.dtype,
        )
        if output.requires_grad else None
        for output in outputs
    )


def output_vjp(outputs, inputs, output_grads):
    """VJP of each differentiable returned tensor with its cotangent."""
    pairs = tuple(
        (output, grad)
        for output, grad in zip(outputs, output_grads, strict=True)
        if output.requires_grad
    )
    if not pairs:
        return tuple(torch.zeros_like(tensor) for tensor in inputs)
    return torch.autograd.grad(
        tuple(output for output, _ in pairs),
        inputs,
        tuple(grad.to(output.dtype) for output, grad in pairs),
        materialize_grads=True,
    )


def evaluate(function, inputs, output_grads=None):
    outputs = as_tuple(function(*inputs))
    gradients = None
    if output_grads is not None:
        gradients = output_vjp(
            outputs, differentiable_inputs(inputs), output_grads,
        )
    return (
        tuple(output.detach() for output in outputs),
        None if gradients is None else tuple(grad.detach() for grad in gradients),
    )


def run_grad(*args, **functions):
    """Evaluate implementations with identical randomly generated cotangents."""
    first = next(iter(functions.values()))
    sample = as_tuple(first(*args))
    output_grads = make_output_grads(sample)
    del sample
    return {
        name: dict(zip(("fwd", "bwd"), evaluate(function, args, output_grads)))
        for name, function in functions.items()
    }


def print_mean_absolute_errors(
        results,
        *,
        reference,
        input_names: Sequence[str],
        ):
    expected_outputs, expected_grads = results[reference]
    for implementation, (outputs, gradients) in results.items():
        if implementation == reference:
            continue
        for index, (actual, expected) in enumerate(
                zip(outputs, expected_outputs, strict=True)
                ):
            suffix = "" if len(outputs) == 1 else f" {index}"
            error = (actual.float() - expected.float()).abs().mean().item()
            print(f"{implementation} forward{suffix} mean absolute error: {error:.6g}")
        if gradients is None:
            continue
        for name, actual, expected in zip(
                input_names, gradients, expected_grads, strict=True,
                ):
            error = (actual.float() - expected.float()).abs().mean().item()
            print(f"{implementation} {name} gradient mean absolute error: {error:.6g}")


def validate_implementations(
        inputs,
        implementations: Mapping[str, Callable[..., Any]],
        *,
        reference,
        input_names,
    output_grad_dtype=None,
        ):
    first = next(iter(implementations.values()))
    sample = as_tuple(first(*inputs))
    output_grads = make_output_grads(sample, dtype=output_grad_dtype)
    del sample
    results = {
        name: evaluate(function, inputs, output_grads)
        for name, function in implementations.items()
    }
    print_mean_absolute_errors(
        results,
        reference=reference,
        input_names=input_names,
    )
    return results, output_grads


def benchmark_implementations(
        name,
        inputs,
        implementations: Mapping[str, Callable[..., Any]],
        *,
        min_run_time=0.5,
        backward=True,
        output_grad_dtype=None,
        ):
    sample = as_tuple(next(iter(implementations.values()))(*inputs))
    output_grads = make_output_grads(sample, dtype=output_grad_dtype)
    del sample
    grad_inputs = differentiable_inputs(inputs)

    def forward(function):
        with torch.no_grad():
            return function(*inputs)

    def forward_backward(function):
        outputs = as_tuple(function(*inputs))
        return output_vjp(outputs, grad_inputs, output_grads)

    for function in implementations.values():
        forward(function)
        if backward:
            forward_backward(function)
    if any(tensor.is_cuda for tensor in inputs):
        torch.cuda.synchronize()

    cases = []
    for implementation, function in implementations.items():
        cases.append(("forward", implementation, lambda f=function: forward(f)))
        if backward:
            cases.append((
                "forward + backward",
                implementation,
                lambda f=function: forward_backward(f),
            ))
    measurements = [
        torch_benchmark.Timer(
            stmt="call()",
            globals={"call": call},
            label=f"{name} {label}",
            sub_label=implementation,
            description="selected configuration",
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        for label, implementation, call in cases
    ]
    torch_benchmark.Compare(measurements).print()
    return measurements


def benchmark_memory(
        name,
        inputs,
        implementations: Mapping[str, object],
        *,
        backward=True,
        output_grad_dtype=None,
        ):
    """Measure the first-execution CUDA footprint above caller-owned tensors.

    This deliberately runs before the timing benchmark.  In particular, compiled
    functions may establish CUDA-graph pools on their first forward/backward
    execution; profiling a later replay attributes none of that persistent
    residency to the operator.
    """
    grad_inputs = differentiable_inputs(inputs)

    def backward(outputs, output_grads):
        return output_vjp(outputs, grad_inputs, output_grads)

    def measure(function):
        # Inputs exist at the call boundary, so exclude them from the result.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # This is also the first call.  Keep grad mode enabled so that it takes
        # exactly the same output path that supplies the subsequent backward.
        # (Some compiled custom-autograd functions omit saved values under
        # no-grad.)
        outputs = as_tuple(function(*inputs))
        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated() - baseline
        output_grads = make_output_grads(outputs, dtype=output_grad_dtype)
        cotangent_bytes = sum(
            grad.untyped_storage().nbytes()
            for grad in output_grads if grad is not None
        )
        if not backward:
            del outputs, output_grads
            return forward_peak, None, None

        # Keep the original baseline and peak counter: this includes pools
        # established by the first forward and by backward graph capture.  Only
        # the cotangent supplied by the caller is excluded.
        result = backward(outputs, output_grads)
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated() - baseline - cotangent_bytes
        del result, output_grads, outputs
        torch.cuda.synchronize()
        return forward_peak, full_peak, None

    rows = [(implementation, *measure(function))
            for implementation, function in implementations.items()]

    mib = 1024 ** 2
    label_width = max(len("Implementation"), *(len(row[0]) for row in rows))
    print(f"{name} first-execution CUDA memory above inputs/cotangents")
    if backward:
        print(
            f"{'Implementation'.ljust(label_width)}  "
            f"{'VJP forward'.rjust(12)}  {'VJP forward + backward'.rjust(24)}"
        )
        for implementation, forward_peak, backward_peak, _ in rows:
            print(
                f"{implementation.ljust(label_width)}  "
                f"{forward_peak / mib:10.3f} MiB  "
                f"{backward_peak / mib:18.3f} MiB"
            )
    else:
        print(f"{'Implementation'.ljust(label_width)}  {'Forward'.rjust(12)}")
        for implementation, forward_peak, _, _ in rows:
            print(
                f"{implementation.ljust(label_width)}  "
                f"{forward_peak / mib:10.3f} MiB"
            )
    return rows


def print_mae_matrix(title, values):
    labels = tuple(values)
    width = max(12, max(map(len, labels)) + 1)
    print(title)
    print("".ljust(width) + "".join(label.rjust(width) for label in labels))
    for row_label in labels:
        row = [row_label.ljust(width)]
        for column_label in labels:
            error = (
                values[row_label].double() - values[column_label].double()
            ).abs().mean().item()
            row.append(f"{error:.6g}".rjust(width))
        print("".join(row))


def validate_precision_matrix(
        function,
        reference,
        inputs,
        *,
        input_names,
        reference_dtypes=None,
        output_grad_dtype=torch.bfloat16,
        backward=True,
        pairwise=True,
        ):
    """Compare an implementation with references evaluated on typed leaves."""
    reference_dtypes = reference_dtypes or {
        "PyTorch BF16": torch.bfloat16,
        "PyTorch FP32": torch.float32,
        "PyTorch FP64": torch.float64,
    }
    outputs = as_tuple(function(*inputs))
    output_grads = (
        make_output_grads(outputs, dtype=output_grad_dtype)
        if backward else None
    )
    del outputs
    implementation_outputs, implementation_grads = evaluate(
        function,
        inputs,
        output_grads,
    )
    results = {"CuTile": (implementation_outputs, implementation_grads)}
    for label, dtype in reference_dtypes.items():
        reference_inputs = tuple(
            tensor.detach().to(dtype).requires_grad_(tensor.requires_grad)
            if tensor.is_floating_point() else tensor.detach()
            for tensor in inputs
        )
        results[label] = evaluate(reference, reference_inputs, output_grads)

    if pairwise:
        for index in range(len(implementation_outputs)):
            suffix = "" if len(implementation_outputs) == 1 else f" {index}"
            print_mae_matrix(
                f"forward{suffix} pairwise MAE",
                {label: result[0][index] for label, result in results.items()},
            )
        if implementation_grads is not None:
            for index, name in enumerate(input_names):
                print_mae_matrix(
                    f"{name} gradient pairwise MAE",
                    {label: result[1][index] for label, result in results.items()},
                )
    else:
        for label, (reference_outputs, reference_grads) in results.items():
            if label == "CuTile":
                continue
            for index, (actual, expected) in enumerate(zip(
                    implementation_outputs, reference_outputs, strict=True,
                    )):
                suffix = "" if len(implementation_outputs) == 1 else f" {index}"
                error = (actual.float() - expected.float()).abs().mean().item()
                print(f"forward{suffix} MAE vs {label}: {error:.6g}")
            if implementation_grads is not None:
                for name, actual, expected in zip(
                        input_names,
                        implementation_grads,
                        reference_grads,
                        strict=True,
                        ):
                    error = (actual.float() - expected.float()).abs().mean().item()
                    print(f"{name} gradient MAE vs {label}: {error:.6g}")
    return results, output_grads


def print_plan(plan):
    print("stages:", ", ".join(stage.stage.name for stage in plan.forward))
    if plan.backward:
        print("backward:", ", ".join(stage.stage.name for stage in plan.backward))


__all__ = [
    "as_tuple",
    "benchmark_implementations",
    "benchmark_memory",
    "evaluate",
    "make_output_grads",
    "print_mae_matrix",
    "print_mean_absolute_errors",
    "print_plan",
    "run_grad",
    "validate_implementations",
    "validate_precision_matrix",
]
