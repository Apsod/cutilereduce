"""Trace a logical fold specification into the experimental tensor IR."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .algebra import (
    AlgebraValue,
    CompositeValue,
    FoldAlgebra,
    StructuredValue,
    value_leaves,
)
from .autodiff import vjp
from .ir import TensorBuilder, TensorProgram, TensorValue
from .passes import optimize_program


@dataclass(frozen=True)
class TensorSpec:
    axes: tuple[str, ...]
    dtype: str = "float32"
    requires_grad: bool | None = None

    @property
    def differentiable(self) -> bool:
        if self.requires_grad is not None:
            return self.requires_grad
        return self.dtype not in {"bool", "int8", "int16", "int32", "int64"}


type MapFunction[ValueT: AlgebraValue] = Callable[
    [TensorBuilder, Mapping[str, TensorValue]],
    ValueT,
]

type OutputFunction[SemanticT: AlgebraValue, OutputT: AlgebraValue] = Callable[
    [TensorBuilder, SemanticT],
    OutputT,
]


@dataclass(frozen=True)
class FoldSpec:
    inputs: Mapping[str, TensorSpec]
    batch_axes: tuple[str, ...]
    fold_axes: tuple[str, ...]

    @property
    def axes(self) -> tuple[str, ...]:
        ordered = []
        for axis in (*self.batch_axes, *self.fold_axes):
            if axis not in ordered:
                ordered.append(axis)
        for spec in self.inputs.values():
            for axis in spec.axes:
                if axis not in ordered:
                    ordered.append(axis)
        return tuple(ordered)

    def check(self) -> None:
        if not self.fold_axes:
            raise ValueError("a fold requires at least one fold axis")
        overlap = set(self.batch_axes) & set(self.fold_axes)
        if overlap:
            raise ValueError(f"axes cannot be both batch and fold: {sorted(overlap)}")
        duplicates = {
            axis for axis in self.axes
            if sum(axis in spec.axes for spec in self.inputs.values()) == 0
        }
        if duplicates:
            raise ValueError(f"unused batch or fold axes: {sorted(duplicates)}")
        for name, spec in self.inputs.items():
            if len(set(spec.axes)) != len(spec.axes):
                raise ValueError(f"input {name!r} has duplicate axes: {spec.axes}")


def _inputs(
    ctx: TensorBuilder, spec: FoldSpec
) -> dict[str, TensorValue]:
    return {
        name: ctx.input(name, input_spec.axes, input_spec.dtype)
        for name, input_spec in spec.inputs.items()
    }


def _value_inputs(
    ctx: TensorBuilder,
    template: AlgebraValue,
    *,
    prefix: str,
) -> AlgebraValue:
    if isinstance(template, StructuredValue):
        return StructuredValue(
            {
                name: ctx.input(
                    f"{prefix}.{name}", value.axes, value.dtype
                )
                for name, value in template.fields.items()
            }
        )
    return CompositeValue(
        {
            name: _value_inputs(
                ctx, component, prefix=f"{prefix}.{name}"
            )
            for name, component in template.components.items()
        }
    )


def _apply_output[
    SemanticT: AlgebraValue, OutputT: AlgebraValue
](
    ctx: TensorBuilder,
    semantic: SemanticT,
    output_function: OutputFunction[SemanticT, OutputT] | None,
) -> AlgebraValue:
    return semantic if output_function is None else output_function(ctx, semantic)


def _cotangent_value(
    ctx: TensorBuilder,
    primal: AlgebraValue,
    cotangents: Mapping[tuple[str, ...], TensorValue],
    prefix: tuple[str, ...] = (),
) -> AlgebraValue:
    if isinstance(primal, StructuredValue):
        return StructuredValue(
            {
                name: cotangents[(*prefix, name)]
                if (*prefix, name) in cotangents
                else ctx.zero(value.axes, value.dtype)
                for name, value in primal.fields.items()
            }
        )
    return CompositeValue(
        {
            name: _cotangent_value(
                ctx,
                component,
                cotangents,
                (*prefix, name),
            )
            for name, component in primal.components.items()
        }
    )


def _lower_output_vjp[
    SemanticT: AlgebraValue,
    OutputT: AlgebraValue,
](
    ctx: TensorBuilder,
    semantic: SemanticT,
    output: AlgebraValue,
    output_cotangent: OutputT,
) -> SemanticT:
    primal_outputs = dict(value_leaves(output))
    cotangent_outputs = dict(value_leaves(output_cotangent))
    if primal_outputs.keys() != cotangent_outputs.keys():
        raise ValueError("output cotangent does not match output structure")
    semantic_leaves = dict(value_leaves(semantic))
    gradients = vjp(
        ctx,
        (
            (primal_outputs[path], cotangent_outputs[path])
            for path in primal_outputs
        ),
        wrt={".".join(path): value for path, value in semantic_leaves.items()},
    )
    by_path = {
        tuple(name.split(".")): value for name, value in gradients.items()
    }
    return cast(SemanticT, _cotangent_value(ctx, semantic, by_path))


def _trace_forward[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
](
    ctx: TensorBuilder,
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AdjointT],
    map_function: MapFunction[SemanticT],
    inputs: Mapping[str, TensorValue],
) -> tuple[ExecutionT, SemanticT]:
    factors = map_function(ctx, inputs)
    algebra.semantic.validate(factors, label="map result")
    lifted = algebra.lower_phi(ctx, factors)
    execution = algebra.execution.lower_fold(
        ctx, lifted, fold_axes=spec.fold_axes
    )
    for path, value in value_leaves(execution):
        remaining = set(value.axes) & set(spec.fold_axes)
        if remaining:
            raise ValueError(
                f"execution field {'.'.join(path)!r} retains fold axes "
                f"{sorted(remaining)}"
            )
    return execution, algebra.lower_psi(ctx, execution)


def compile_fold_forward[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AdjointT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
    save_for_backward: bool = True,
) -> TensorProgram:
    """Lower the forward fold and its explicit saved-state boundary."""
    spec.check()
    ctx = TensorBuilder(spec.axes)
    execution, semantic = _trace_forward(
        ctx, spec, algebra, map_function, _inputs(ctx, spec)
    )
    output = _apply_output(ctx, semantic, output_function)
    for path, value in value_leaves(output):
        ctx.materialize(".".join(path), value)
    if save_for_backward:
        for path, value in value_leaves(execution):
            ctx.materialize(f"saved_execution.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


def compile_fold[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AdjointT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
) -> TensorProgram:
    """Compatibility spelling for inference-only forward lowering."""
    return compile_fold_forward(
        spec,
        algebra,
        map_function,
        output_function=output_function,
        save_for_backward=False,
    )


def compile_fold_backward[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AdjointT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
) -> TensorProgram:
    """Lower a standalone backward program for a commutative fold.

    Saved execution fields and semantic output cotangents are explicit inputs.
    The logical map is recomputed and is the only region differentiated by
    tensor-IR reverse mode.
    """
    spec.check()
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, semantic_template = _trace_forward(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    output_template = _apply_output(
        shape_ctx, semantic_template, output_function
    )

    ctx = TensorBuilder(spec.axes)
    inputs = _inputs(ctx, spec)
    execution = cast(
        ExecutionT,
        _value_inputs(ctx, execution_template, prefix="saved_execution"),
    )
    output_cotangent = cast(
        OutputT,
        _value_inputs(ctx, output_template, prefix="grad_output"),
    )
    semantic = algebra.lower_psi(ctx, execution)
    output = _apply_output(ctx, semantic, output_function)
    semantic_cotangent = _lower_output_vjp(
        ctx, semantic, output, output_cotangent
    )
    adjoint = algebra.lower_embed(ctx, execution, semantic_cotangent)

    recomputed_factors = map_function(ctx, inputs)
    local_cotangent = algebra.lower_finalize(
        ctx, recomputed_factors, adjoint
    )
    primal_by_path = dict(value_leaves(recomputed_factors))
    cotangent_by_path = dict(value_leaves(local_cotangent))
    if primal_by_path.keys() != cotangent_by_path.keys():
        raise ValueError("finalize result does not match semantic factor structure")

    differentiable_inputs = {
        name: inputs[name]
        for name, input_spec in spec.inputs.items()
        if input_spec.differentiable
    }
    input_cotangents = vjp(
        ctx,
        (
            (primal_by_path[path], cotangent_by_path[path])
            for path in primal_by_path
        ),
        wrt=differentiable_inputs,
    )

    for name, value in input_cotangents.items():
        ctx.materialize(f"grad_input.{name}", value)
    return optimize_program(ctx.finish())


def compile_map_finalize_vjp_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AdjointT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
    with_prefix: bool = False,
) -> TensorProgram:
    """Lower ``map -> finalize -> map*`` over an arbitrary input region.

    The adjoint accumulator is an explicit input.  Any fold axes in ``spec``
    describe the region's local factor extent; this function does not assert
    that the region is a tile, a complete fold, or a kernel.  Cotangents for
    inputs that do not depend on a local fold axis are reduced over that axis
    and are therefore partial input cotangents for this region.

    Set ``with_prefix`` for a general non-commutative algebra. The execution
    prefix then becomes an explicit ``prefix.*`` input to the region.
    """
    spec.check()

    # Infer the concrete lifted axes of A from the ordinary forward/embed
    # composition.  The resulting tensor values are used only as input
    # signatures for this independent region program.
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, semantic_template = _trace_forward(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    output_template = _apply_output(
        shape_ctx, semantic_template, output_function
    )
    output_cotangent_template = cast(
        OutputT,
        _value_inputs(
            shape_ctx, output_template, prefix="unused_grad_output"
        ),
    )
    semantic_cotangent_template = _lower_output_vjp(
        shape_ctx,
        semantic_template,
        output_template,
        output_cotangent_template,
    )
    adjoint_template = algebra.lower_embed(
        shape_ctx, execution_template, semantic_cotangent_template
    )

    ctx = TensorBuilder(spec.axes)
    inputs = _inputs(ctx, spec)
    adjoint = cast(
        AdjointT,
        _value_inputs(ctx, adjoint_template, prefix="adjoint"),
    )
    prefix = (
        cast(
            ExecutionT,
            _value_inputs(ctx, execution_template, prefix="prefix"),
        )
        if with_prefix
        else None
    )
    factors = map_function(ctx, inputs)
    algebra.semantic.validate(factors, label="map result")
    local_cotangent = algebra.lower_finalize(
        ctx, factors, adjoint, prefix=prefix
    )

    primal_by_path = dict(value_leaves(factors))
    cotangent_by_path = dict(value_leaves(local_cotangent))
    if primal_by_path.keys() != cotangent_by_path.keys():
        raise ValueError("finalize result does not match semantic factor structure")

    differentiable_inputs = {
        name: inputs[name]
        for name, input_spec in spec.inputs.items()
        if input_spec.differentiable
    }
    input_cotangents = vjp(
        ctx,
        (
            (primal_by_path[path], cotangent_by_path[path])
            for path in primal_by_path
        ),
        wrt=differentiable_inputs,
    )
    for name, value in input_cotangents.items():
        ctx.materialize(f"partial_grad_input.{name}", value)
    return optimize_program(ctx.finish())


__all__ = [
    "FoldSpec",
    "MapFunction",
    "OutputFunction",
    "TensorSpec",
    "compile_fold",
    "compile_fold_backward",
    "compile_fold_forward",
    "compile_map_finalize_vjp_region",
]
