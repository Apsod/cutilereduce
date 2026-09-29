"""General fold-algebra contracts and concrete algebra instances."""

from .add import Add, AddMonoid
from .base import (
    AlgebraStructure,
    AlgebraValue,
    FoldAlgebra,
    Monoid,
    CompositeStructure,
    CompositeValue,
    StructuredValue,
    TensorField,
    TensorStructure,
    value_leaves,
)
from .lse import LogSumExp
from .lws import LogWeightedSum
from .product import Product, ProductAlgebra, ProductMonoid
from .semidirect import (
    Affine,
    LogScaleAffine,
    SemidirectProduct,
    SemidirectProductAlgebra,
    SemidirectProductMonoid,
)

__all__ = [
    "Add",
    "AddMonoid",
    "Affine",
    "AlgebraStructure",
    "AlgebraValue",
    "FoldAlgebra",
    "Monoid",
    "LogSumExp",
    "LogWeightedSum",
    "LogScaleAffine",
    "CompositeStructure",
    "CompositeValue",
    "Product",
    "ProductAlgebra",
    "ProductMonoid",
    "SemidirectProduct",
    "SemidirectProductAlgebra",
    "SemidirectProductMonoid",
    "StructuredValue",
    "TensorField",
    "TensorStructure",
    "value_leaves",
]
