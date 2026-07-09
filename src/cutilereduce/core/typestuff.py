# ruff: noqa: E701
import cuda.tile as ct
import torch

DType = ct.DType


def to_torch_dtype(dtype: DType) -> torch.dtype:
    match dtype:
        case ct.float64: return torch.float64     # noqa: E701
        case ct.float32: return torch.float32     # noqa: E701
        case ct.float16: return torch.float16     # noqa: E701
        case ct.bfloat16: return torch.bfloat16   # noqa: E701
        case ct.int64: return torch.int64         # noqa: E701
        case ct.int32: return torch.int32         # noqa: E701
        case ct.int16: return torch.int16         # noqa: E701
        case ct.int8: return torch.int8           # noqa: E701
        case ct.uint64: return torch.uint64       # noqa: E701
        case ct.uint32: return torch.uint32       # noqa: E701
        case ct.uint16: return torch.uint16       # noqa: E701
        case ct.uint8: return torch.uint8         # noqa: E701
