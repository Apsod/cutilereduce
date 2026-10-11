"""Symbolically lower logical fold specifications to tensor IR."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from cutilereduce.experimental.algebra import (
    AlgebraStructure,
    AlgebraValue,
    CompositeStructure,
    CompositeValue,
    FinalizeContext,
    FoldAlgebra,
    StructuredValue,
    TensorStructure,
    value_leaves,
)
from cutilereduce.experimental.ir.autodiff import vjp
from cutilereduce.experimental.ir.core import TensorBuilder, TensorProgram, TensorValue
from cutilereduce.experimental.algebra.lowering import (
    lower_fold,
    lower_fold_accumulate,
    lower_scan,
)
from cutilereduce.experimental.ir.passes import optimize_program


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


@dataclass(frozen=True)
class FoldFactor[ValueT: AlgebraValue]:
    """One semantic factor and an optional whole-interaction validity mask."""

    value: ValueT
    where: TensorValue | None = None


type MapFunction[ValueT: AlgebraValue] = Callable[
    [TensorBuilder, Mapping[str, TensorValue]],
    ValueT | FoldFactor[ValueT],
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
            axis
            for axis in self.axes
            if sum(axis in spec.axes for spec in self.inputs.values()) == 0
        }
        if duplicates:
            raise ValueError(f"unused batch or fold axes: {sorted(duplicates)}")
        for name, spec in self.inputs.items():
            if len(set(spec.axes)) != len(spec.axes):
                raise ValueError(f"input {name!r} has duplicate axes: {spec.axes}")


def _inputs(ctx: TensorBuilder, spec: FoldSpec) -> dict[str, TensorValue]:
    return {
        name: ctx.input(name, input_spec.axes, input_spec.dtype)
        for name, input_spec in spec.inputs.items()
    }


def _mapped_factor[
    ValueT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
](
    ctx: TensorBuilder,
    spec: FoldSpec,
    algebra: FoldAlgebra[ValueT, ExecutionT, AccumulatorT],
    mapped: ValueT | FoldFactor[ValueT],
) -> ValueT:
    factor = mapped.value if isinstance(mapped, FoldFactor) else mapped
    semantic_valid = mapped.where if isinstance(mapped, FoldFactor) else None
    algebra.semantic.validate(factor, label="map result")
    allowed_axes = set(spec.batch_axes) | set(spec.fold_axes)
    if semantic_valid is not None:
        if semantic_valid.dtype != "bool":
            raise ValueError("factor mask must have bool dtype")
        if set(semantic_valid.axes) - allowed_axes:
            raise ValueError("factor mask may only use batch and fold axes")
    if semantic_valid is None:
        return factor
    condition = semantic_valid
    identity = algebra.semantic.identity(ctx)
    group = len(ctx.nodes)

    def mask(value: AlgebraValue, neutral: AlgebraValue) -> AlgebraValue:
        if isinstance(value, StructuredValue):
            if not isinstance(neutral, StructuredValue):
                raise TypeError("semantic value and identity structures disagree")
            return StructuredValue(
                {
                    name: ctx.mask_identity(
                        condition, field, neutral[name], mask_group=group
                    )
                    for name, field in value.fields.items()
                }
            )
        if not isinstance(neutral, CompositeValue):
            raise TypeError("semantic value and identity structures disagree")
        return CompositeValue(
            {
                name: mask(component, neutral[name])
                for name, component in value.components.items()
            }
        )

    return cast(ValueT, mask(factor, identity))


def _mask_algebra_value(
    ctx: TensorBuilder,
    value: AlgebraValue,
    neutral: AlgebraValue,
    condition: TensorValue,
) -> AlgebraValue:
    """Select an algebra value or its neutral element over a validity domain."""
    if isinstance(value, StructuredValue):
        if not isinstance(neutral, StructuredValue):
            raise TypeError("algebra value and neutral structures disagree")
        group = len(ctx.nodes)
        return StructuredValue(
            {
                name: ctx.mask_identity(
                    condition, field, neutral[name], mask_group=group
                )
                for name, field in value.fields.items()
            }
        )
    if not isinstance(neutral, CompositeValue):
        raise TypeError("algebra value and neutral structures disagree")
    return CompositeValue(
        {
            name: _mask_algebra_value(ctx, item, neutral[name], condition)
            for name, item in value.components.items()
        }
    )


def _zero_like_value(ctx: TensorBuilder, value: AlgebraValue) -> AlgebraValue:
    if isinstance(value, StructuredValue):
        return StructuredValue(
            {
                name: ctx.full(item.axes, 0.0, item.dtype)
                for name, item in value.fields.items()
            }
        )
    return CompositeValue(
        {name: _zero_like_value(ctx, item) for name, item in value.components.items()}
    )


def _value_inputs(
    ctx: TensorBuilder,
    template: AlgebraValue,
    *,
    prefix: str,
) -> AlgebraValue:
    if isinstance(template, StructuredValue):
        return StructuredValue(
            {
                name: ctx.input(f"{prefix}.{name}", value.axes, value.dtype)
                for name, value in template.fields.items()
            }
        )
    return CompositeValue(
        {
            name: _value_inputs(ctx, component, prefix=f"{prefix}.{name}")
            for name, component in template.components.items()
        }
    )


def _apply_output[SemanticT: AlgebraValue, OutputT: AlgebraValue](
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


def _output_vjp[
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
        ((primal_outputs[path], cotangent_outputs[path]) for path in primal_outputs),
        wrt={".".join(path): value for path, value in semantic_leaves.items()},
    )
    by_path = {tuple(name.split(".")): value for name, value in gradients.items()}
    return cast(SemanticT, _cotangent_value(ctx, semantic, by_path))


def _lower_forward_values[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
](
    ctx: TensorBuilder,
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
    inputs: Mapping[str, TensorValue],
) -> tuple[ExecutionT, SemanticT]:
    factors = _mapped_factor(ctx, spec, algebra, map_function(ctx, inputs))
    lifted = algebra.to_execution(ctx, factors)
    execution = lower_fold(ctx, algebra.execution, lifted, fold_axes=spec.fold_axes)
    for path, value in value_leaves(execution):
        remaining = set(value.axes) & set(spec.fold_axes)
        if remaining:
            raise ValueError(
                f"execution field {'.'.join(path)!r} retains fold axes "
                f"{sorted(remaining)}"
            )
    return execution, algebra.to_semantic(ctx, execution)


def compile_fold_forward[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
    save_for_backward: bool = True,
) -> TensorProgram:
    """Lower the forward fold and its explicit saved-state boundary."""
    spec.check()
    ctx = TensorBuilder(spec.axes)
    execution, semantic = _lower_forward_values(
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
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
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


def compile_map_fold_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
) -> TensorProgram:
    """Lower ``E.fold o Phi^F o map`` for an arbitrary fold region."""
    spec.check()
    ctx = TensorBuilder(spec.axes)
    factors = _mapped_factor(ctx, spec, algebra, map_function(ctx, _inputs(ctx, spec)))
    execution = lower_fold(
        ctx,
        algebra.execution,
        algebra.to_execution(ctx, factors),
        fold_axes=spec.fold_axes,
    )
    for path, value in value_leaves(execution):
        ctx.materialize(f"execution.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


def compile_masked_map_fold_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
) -> TensorProgram:
    """Lower map-fold with validity supplied by its scheduled execution domain."""
    spec.check()
    ctx = TensorBuilder(spec.axes)
    valid = ctx.domain_valid((*spec.batch_axes, *spec.fold_axes))
    factors = _mapped_factor(ctx, spec, algebra, map_function(ctx, _inputs(ctx, spec)))
    execution = lower_fold(
        ctx,
        algebra.execution,
        algebra.to_execution(ctx, factors),
        fold_axes=spec.fold_axes,
        domain_valid=valid,
    )
    for path, value in value_leaves(execution):
        ctx.materialize(f"execution.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


def compile_map_fold_combine_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
) -> TensorProgram:
    """Lower ``combine(state, fold(Phi(map(tile))))`` as one region."""
    spec.check()
    ctx = TensorBuilder(spec.axes)
    inputs = _inputs(ctx, spec)

    def state_inputs(
        structure: AlgebraStructure, prefix: tuple[str, ...] = ()
    ) -> AlgebraValue:
        name = ".".join(prefix)
        if isinstance(structure, TensorStructure):
            return StructuredValue(
                {
                    field: ctx.input(
                        f"accumulator.{name + '.' if name else ''}{field}",
                        (*spec.batch_axes, *schema.inner_axes),
                        schema.dtype,
                    )
                    for field, schema in structure.fields.items()
                }
            )
        if isinstance(structure, CompositeStructure):
            return CompositeValue(
                {
                    component: state_inputs(value, (*prefix, component))
                    for component, value in structure.components.items()
                }
            )
        raise TypeError("unknown algebra structure")

    initial = cast(ExecutionT, state_inputs(algebra.execution.structure))
    valid = ctx.domain_valid((*spec.batch_axes, *spec.fold_axes))
    factors = _mapped_factor(ctx, spec, algebra, map_function(ctx, inputs))
    lifted = algebra.to_execution(ctx, factors)
    result = lower_fold_accumulate(
        ctx,
        algebra.execution,
        initial,
        lifted,
        fold_axes=spec.fold_axes,
        domain_valid=valid,
    )
    algebra.execution.validate(result, label="map-fold-combine result")
    for path, value in value_leaves(result):
        if set(value.axes) & set(spec.fold_axes):
            raise ValueError(
                f"map-fold-combine result {'.'.join(path)!r} retains fold axes"
            )
        ctx.materialize(f"execution.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


def compile_project_output_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
) -> TensorProgram:
    """Lower ``to_output o Psi`` from a completed execution carrier."""
    spec.check()
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, _ = _lower_forward_values(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    ctx = TensorBuilder(spec.axes)
    execution = cast(
        ExecutionT,
        _value_inputs(ctx, execution_template, prefix="execution"),
    )
    semantic = algebra.to_semantic(ctx, execution)
    output = _apply_output(ctx, semantic, output_function)
    for path, value in value_leaves(output):
        ctx.materialize(".".join(path), value)
    return optimize_program(ctx.finish())


def compile_fold_backward[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
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
    if algebra.finalize_context is FinalizeContext.PREFIX:
        raise ValueError(
            "standalone fold backward cannot supply required prefix context"
        )
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, semantic_template = _lower_forward_values(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    output_template = _apply_output(shape_ctx, semantic_template, output_function)

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
    semantic = algebra.to_semantic(ctx, execution)
    output = _apply_output(ctx, semantic, output_function)
    semantic_cotangent = _output_vjp(ctx, semantic, output, output_cotangent)
    accumulator = algebra.embed(ctx, execution, semantic_cotangent)

    recomputed_factors = _mapped_factor(ctx, spec, algebra, map_function(ctx, inputs))
    local_cotangent = algebra.finalize(ctx, recomputed_factors, accumulator)
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
        ((primal_by_path[path], cotangent_by_path[path]) for path in primal_by_path),
        wrt=differentiable_inputs,
    )

    for name, value in input_cotangents.items():
        ctx.materialize(f"grad_input.{name}", value)
    return optimize_program(ctx.finish())


def compile_embed_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
) -> TensorProgram:
    """Lower ``embed o output_function*`` as an independent region."""
    spec.check()
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, semantic_template = _lower_forward_values(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    output_template = _apply_output(shape_ctx, semantic_template, output_function)

    ctx = TensorBuilder(spec.axes)
    execution = cast(
        ExecutionT,
        _value_inputs(ctx, execution_template, prefix="saved_execution"),
    )
    output_cotangent = cast(
        OutputT,
        _value_inputs(ctx, output_template, prefix="grad_output"),
    )
    semantic = algebra.to_semantic(ctx, execution)
    output = _apply_output(ctx, semantic, output_function)
    semantic_cotangent = _output_vjp(ctx, semantic, output, output_cotangent)
    accumulator = algebra.embed(ctx, execution, semantic_cotangent)
    for path, value in value_leaves(accumulator):
        ctx.materialize(f"accumulator.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


def compile_map_finalize_vjp_region[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
](
    spec: FoldSpec,
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT],
    map_function: MapFunction[SemanticT],
    *,
    output_function: OutputFunction[SemanticT, OutputT] | None = None,
) -> TensorProgram:
    """Lower ``map -> finalize -> map*`` over an arbitrary input region.

    The gradient accumulator is an explicit input. Any fold axes in ``spec``
    describe the region's local factor extent; this function does not assert
    that the region is a tile, a complete fold, or a kernel.  Cotangents for
    inputs that do not depend on a local fold axis are reduced over that axis
    and are therefore partial input cotangents for this region.

    A prefix-dependent algebra receives an explicit ``prefix.*`` input,
    according to its declared ``finalize_context``.
    """
    spec.check()

    # Infer the concrete lifted axes of A from the ordinary forward/embed
    # composition.  The resulting tensor values are used only as input
    # signatures for this independent region program.
    shape_ctx = TensorBuilder(spec.axes)
    execution_template, semantic_template = _lower_forward_values(
        shape_ctx,
        spec,
        algebra,
        map_function,
        _inputs(shape_ctx, spec),
    )
    output_template = _apply_output(shape_ctx, semantic_template, output_function)
    output_cotangent_template = cast(
        OutputT,
        _value_inputs(shape_ctx, output_template, prefix="unused_grad_output"),
    )
    semantic_cotangent_template = _output_vjp(
        shape_ctx,
        semantic_template,
        output_template,
        output_cotangent_template,
    )
    accumulator_template = algebra.embed(
        shape_ctx, execution_template, semantic_cotangent_template
    )

    ctx = TensorBuilder(spec.axes)
    inputs = _inputs(ctx, spec)
    accumulator = cast(
        AccumulatorT,
        _value_inputs(ctx, accumulator_template, prefix="accumulator"),
    )
    boundary_prefix = (
        cast(
            ExecutionT,
            _value_inputs(ctx, execution_template, prefix="prefix"),
        )
        if algebra.finalize_context is FinalizeContext.PREFIX
        else None
    )
    mapped = map_function(ctx, inputs)
    valid = ctx.domain_valid((*spec.batch_axes, *spec.fold_axes))
    factors = _mapped_factor(ctx, spec, algebra, mapped)
    prefix = boundary_prefix
    updated_prefix = None
    if boundary_prefix is not None:
        if len(spec.fold_axes) != 1:
            raise ValueError(
                "prefix-dependent region lowering requires one ordered axis"
            )
        lifted = algebra.to_execution(ctx, factors)
        identity = algebra.execution.identity(ctx)
        lifted = cast(ExecutionT, _mask_algebra_value(ctx, lifted, identity, valid))
        scan = lower_scan(
            ctx,
            algebra.execution,
            lifted,
            scan_axis=spec.fold_axes[0],
            initial=boundary_prefix,
        )
        prefix = scan.prefixes
        updated_prefix = scan.final
    local_cotangent = algebra.finalize(ctx, factors, accumulator, prefix=prefix)

    primal_by_path = dict(value_leaves(factors))
    cotangent_by_path = dict(value_leaves(local_cotangent))
    if primal_by_path.keys() != cotangent_by_path.keys():
        raise ValueError("finalize result does not match semantic factor structure")

    # Validity belongs to factor contributions, before map* reduces axes
    # absent from an input. Masking the reduced destination loses this information.
    cotangent_by_path = {
        path: ctx.where(valid, value, ctx.full(value.axes, 0.0, value.dtype))
        for path, value in cotangent_by_path.items()
    }

    differentiable_inputs = {
        name: inputs[name]
        for name, input_spec in spec.inputs.items()
        if input_spec.differentiable
    }
    input_cotangents = vjp(
        ctx,
        ((primal_by_path[path], cotangent_by_path[path]) for path in primal_by_path),
        wrt=differentiable_inputs,
    )
    for name, value in input_cotangents.items():
        ctx.materialize(f"partial_grad_input.{name}", value)
    if updated_prefix is not None:
        for path, value in value_leaves(updated_prefix):
            ctx.materialize(f"updated_prefix.{'.'.join(path)}", value)
    return optimize_program(ctx.finish())


__all__ = [
    "FoldFactor",
    "FoldSpec",
    "MapFunction",
    "OutputFunction",
    "TensorSpec",
    "compile_fold",
    "compile_fold_backward",
    "compile_fold_forward",
    "compile_embed_region",
    "compile_map_fold_region",
    "compile_map_fold_combine_region",
    "compile_map_finalize_vjp_region",
    "compile_project_output_region",
]
