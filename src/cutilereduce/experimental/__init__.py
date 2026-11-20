"""Public operator/algebra authoring and backend-neutral compilation API.

Inspection and scheduling overrides live in their owning subpackages.
"""

from cutilereduce.experimental.algebra import (
    Add,
    AddMonoid,
    Affine,
    AffineLWS,
    AffineLWSExecutionMonoid,
    AffineLWSMonoid,
    AffineMonoid,
    AlgebraStructure,
    AlgebraValue,
    CompositeStructure,
    CompositeValue,
    FinalizeContext,
    FoldAlgebra,
    LogScaleAffine,
    LogSumExp,
    LogWeightedSum,
    Monoid,
    PartialSemigroup,
    Product,
    ProductAlgebra,
    ProductMonoid,
    ProductSemigroup,
    ScanResult,
    SemidirectProduct,
    SemidirectProductAlgebra,
    SemidirectProductMonoid,
    StructuredValue,
    TensorField,
    TensorStructure,
)
from cutilereduce.experimental.ir.core import (
    ContractionPrecision,
    TensorBuilder,
    TensorSignature,
    TensorValue,
)
from cutilereduce.experimental.compiler.fold import (
    FoldFactor,
    FoldSpec,
    MapFunction,
    OutputFunction,
    TensorSpec,
)
from cutilereduce.experimental.compiler.operator import (
    FoldMaps,
    FoldOperator,
)
from .api import analyze, compile, tune
from .backends.cpu.backend import CpuBackend
from .backends.cpu.schedule import CpuTarget
from .backends.cutile.backend import CuTileBackend
from .backends.cutile.target import CuTileTarget

__all__ = ['Add', 'AddMonoid', 'Affine', 'AffineLWS', 'AffineLWSExecutionMonoid', 'AffineLWSMonoid', 'AffineMonoid', 'AlgebraStructure', 'AlgebraValue', 'CompositeStructure', 'CompositeValue', 'ContractionPrecision', 'CpuBackend', 'CpuTarget', 'CuTileBackend', 'CuTileTarget', 'FinalizeContext', 'FoldAlgebra', 'FoldFactor', 'FoldMaps', 'FoldOperator', 'FoldSpec', 'LogScaleAffine', 'LogSumExp', 'LogWeightedSum', 'MapFunction', 'Monoid', 'OutputFunction', 'PartialSemigroup', 'Product', 'ProductAlgebra', 'ProductMonoid', 'ProductSemigroup', 'ScanResult', 'SemidirectProduct', 'SemidirectProductAlgebra', 'SemidirectProductMonoid', 'StructuredValue', 'TensorBuilder', 'TensorField', 'TensorSignature', 'TensorSpec', 'TensorStructure', 'TensorValue', 'analyze', 'compile', 'tune']
