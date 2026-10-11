"""First-class fold operators and their symbolic fold programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cutilereduce.experimental.algebra import (
    AlgebraStructure,
    AlgebraValue,
    CompositeValue,
    FinalizeContext,
    FoldAlgebra,
    StructuredValue,
    TensorStructure,
    value_leaves,
)
from cutilereduce.experimental.compiler.fold import (
    FoldSpec,
    MapFunction,
    OutputFunction,
    compile_embed_region,
    compile_fold_backward,
    compile_fold_forward,
    compile_map_fold_region,
    compile_map_fold_combine_region,
    compile_map_finalize_vjp_region,
    compile_project_output_region,
)
from cutilereduce.experimental.ir.core import (
    TensorRegion,
    TensorRegionBuilder,
    TensorSignature,
)
from cutilereduce.experimental.ir.passes import program_to_region


@dataclass(frozen=True)
class FoldMaps[SemanticT: AlgebraValue, OutputT: AlgebraValue]:
    """The user-authored symbolic maps of a fold operator."""

    to_factor: MapFunction[SemanticT]
    to_output: OutputFunction[SemanticT, OutputT] | None = None


@dataclass(frozen=True)
class FoldProgram:
    """Normalized symbolic regions and laws presented to a fold planner."""

    spec: FoldSpec
    execution_structure: AlgebraStructure
    semantic_commutative: bool
    execution_commutative: bool
    execution_identity: TensorRegion
    execution_combine: TensorRegion
    map_fold: TensorRegion
    map_fold_combine: TensorRegion
    project_output: TensorRegion
    embed_output_vjp: TensorRegion | None
    map_finalize_map_vjp: TensorRegion | None
    map_finalize_map_vjp_accumulate: TensorRegion | None

    def __post_init__(self) -> None:
        if self.semantic_commutative and self.requires_prefix:
            raise ValueError(
                "a commutative semantic monoid must have prefix-free finalize"
            )

    @property
    def requires_prefix(self) -> bool:
        return self.map_finalize_map_vjp_accumulate is not None


def _expect_value(
    ctx: TensorRegionBuilder,
    structure: AlgebraStructure,
    template: AlgebraValue,
    *,
    prefix: str,
) -> AlgebraValue:
    if isinstance(structure, TensorStructure):
        if not isinstance(template, StructuredValue):
            raise TypeError("structure and lifted signature disagree")
        return StructuredValue(
            {
                name: ctx.expect(
                    f"{prefix}.{name}",
                    TensorSignature(template[name].axes, field.dtype),
                    label=f"{prefix} {name}",
                )
                for name, field in structure.fields.items()
            }
        )
    if not isinstance(template, CompositeValue):
        raise TypeError("structure and lifted signature disagree")
    return CompositeValue(
        {
            name: _expect_value(
                ctx, component, template[name], prefix=f"{prefix}.{name}"
            )
            for name, component in structure.components.items()
        }
    )


def _yield_value(ctx: TensorRegionBuilder, value: AlgebraValue, *, prefix: str) -> None:
    for path, leaf in value_leaves(value):
        ctx.yield_value(f"{prefix}.{'.'.join(path)}", leaf)


def _value_from_region_results(
    region: TensorRegion, structure: AlgebraStructure, prefix: str
) -> AlgebraValue:
    builder = TensorRegionBuilder(region.axes)
    signatures = {
        name: TensorSignature(region.nodes[node_id].axes, region.nodes[node_id].dtype)
        for name, node_id in region.results
    }

    def build(schema: AlgebraStructure, path: str) -> AlgebraValue:
        if isinstance(schema, TensorStructure):
            return StructuredValue(
                {
                    name: builder.expect(
                        f"template.{path}.{name}", signatures[f"{path}.{name}"]
                    )
                    for name in schema.fields
                }
            )
        return CompositeValue(
            {
                name: build(component, f"{path}.{name}")
                for name, component in schema.components.items()
            }
        )

    return build(structure, prefix)


def _execution_identity_region(spec: FoldSpec, algebra: FoldAlgebra) -> TensorRegion:
    ctx = TensorRegionBuilder(spec.axes)
    _yield_value(ctx, algebra.execution.identity(ctx), prefix="execution")
    return program_to_region(ctx.finish_region().as_program())


def _execution_combine_region(
    spec: FoldSpec, algebra: FoldAlgebra, template: AlgebraValue
) -> TensorRegion:
    ctx = TensorRegionBuilder(spec.axes)
    left = _expect_value(ctx, algebra.execution.structure, template, prefix="left")
    right = _expect_value(ctx, algebra.execution.structure, template, prefix="right")
    _yield_value(ctx, algebra.execution.combine(ctx, left, right), prefix="execution")
    return program_to_region(ctx.finish_region().as_program())


@dataclass(frozen=True)
class FoldOperator[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
    OutputT: AlgebraValue,
]:
    """A complete observable fold definition, before execution planning."""

    spec: FoldSpec
    algebra: FoldAlgebra[SemanticT, ExecutionT, AccumulatorT]
    maps: FoldMaps[SemanticT, OutputT]

    def __post_init__(self) -> None:
        self.spec.check()
        if (
            self.algebra.semantic.commutative
            and self.algebra.finalize_context is FinalizeContext.PREFIX
        ):
            raise ValueError(
                "a commutative semantic monoid must have prefix-free finalize"
            )

    def forward(self, *, save_for_backward: bool = True):
        return compile_fold_forward(
            self.spec,
            self.algebra,
            self.maps.to_factor,
            output_function=self.maps.to_output,
            save_for_backward=save_for_backward,
        )

    def backward(self):
        return compile_fold_backward(
            self.spec,
            self.algebra,
            self.maps.to_factor,
            output_function=self.maps.to_output,
        )

    def lower(self, *, backward: bool = True) -> FoldProgram:
        """Symbolically lower this DSL definition into a schedule-independent program."""
        map_fold = program_to_region(
            compile_map_fold_region(self.spec, self.algebra, self.maps.to_factor)
        )
        map_fold_combine = program_to_region(
            compile_map_fold_combine_region(
                self.spec, self.algebra, self.maps.to_factor
            )
        )
        template = cast(
            ExecutionT,
            _value_from_region_results(
                map_fold, self.algebra.execution.structure, "execution"
            ),
        )
        project = program_to_region(
            compile_project_output_region(
                self.spec,
                self.algebra,
                self.maps.to_factor,
                output_function=self.maps.to_output,
            )
        )
        embed = (
            program_to_region(
                compile_embed_region(
                    self.spec,
                    self.algebra,
                    self.maps.to_factor,
                    output_function=self.maps.to_output,
                )
            )
            if backward
            else None
        )
        backward_region = (
            program_to_region(
                compile_map_finalize_vjp_region(
                    self.spec,
                    self.algebra,
                    self.maps.to_factor,
                    output_function=self.maps.to_output,
                )
            )
            if backward
            else None
        )
        prefix = self.algebra.finalize_context is FinalizeContext.PREFIX
        return FoldProgram(
            spec=self.spec,
            execution_structure=self.algebra.execution.structure,
            semantic_commutative=self.algebra.semantic.commutative,
            execution_commutative=self.algebra.execution.commutative,
            execution_identity=_execution_identity_region(self.spec, self.algebra),
            execution_combine=_execution_combine_region(
                self.spec, self.algebra, template
            ),
            map_fold=map_fold,
            map_fold_combine=map_fold_combine,
            project_output=project,
            embed_output_vjp=embed,
            map_finalize_map_vjp=backward_region if not prefix else None,
            map_finalize_map_vjp_accumulate=backward_region if prefix else None,
        )


__all__ = ["FoldMaps", "FoldOperator", "FoldProgram"]
