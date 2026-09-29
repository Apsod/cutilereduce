"""General fold-algebra contracts and concrete algebra instances."""

from .add import Add, AddMonoid
from .affine import Affine, AffineMonoid, LogScaleAffine
from .affine_lws import (
    AffineLWS,
    AffineLWSExecutionMonoid,
    AffineLWSMonoid,
)
from .base import (
    AlgebraStructure,
    AlgebraValue,
    FoldAlgebra,
    FinalizeContext,
    Monoid,
    PartialSemigroup,
    ScanResult,
    CompositeStructure,
    CompositeValue,
    StructuredValue,
    TensorField,
    TensorStructure,
    value_leaves,
)
from .lse import LogSumExp
from .lws import LogWeightedSum
from .product import Product, ProductAlgebra, ProductMonoid, ProductSemigroup
from .semidirect import (
    SemidirectProduct,
    SemidirectProductAlgebra,
    SemidirectProductMonoid,
)

__all__ = [
    "Add",
    "AddMonoid",
    "Affine",
    "AffineLWS",
    "AffineLWSExecutionMonoid",
    "AffineLWSMonoid",
    "AffineMonoid",
    "AlgebraStructure",
    "AlgebraValue",
    "FoldAlgebra",
    "FinalizeContext",
    "Monoid",
    "PartialSemigroup",
    "ScanResult",
    "LogSumExp",
    "LogWeightedSum",
    "LogScaleAffine",
    "CompositeStructure",
    "CompositeValue",
    "Product",
    "ProductAlgebra",
    "ProductMonoid",
    "ProductSemigroup",
    "SemidirectProduct",
    "SemidirectProductAlgebra",
    "SemidirectProductMonoid",
    "StructuredValue",
    "TensorField",
    "TensorStructure",
    "value_leaves",
]
