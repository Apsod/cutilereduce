import cuda.tile as ct
import polars.selectors as cs

from cutilereduce.core import MatMulWork, WorkModel
from cutilereduce.core.buffer import buffer_spec
from cutilereduce.fold import make_fold_spec, sweep_commutative_fold
from cutilereduce.util.spec import l4


def xentropy_spec():
    return make_fold_spec(
        input={
            "ctx": buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
            "trg": buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
            "targets": buffer_spec("b", ct.int32, default=-100),
        },
        execution={
            "m": buffer_spec("b", ct.float32, default=float("-inf")),
            "e": buffer_spec("b", ct.float32, default=0),
            "u": buffer_spec("b", ct.float32, default=0),
        },
        output={
            "z": buffer_spec("b", ct.float32, default=float("-inf")),
            "l": buffer_spec("b", ct.float32, default=0),
        },
        batch="b",
        fold="v",
        map_fold_work=WorkModel.make(MatMulWork.make(M="b", N="v", K="d")),
    )


def main():
    result = sweep_commutative_fold(
        xentropy_spec(),
        sizes={"b": 1024 * 16, "v": 1024 * 16, "d": 128},
        hardware=l4,
    )

    print(result.select(
        "path",
        cs.starts_with("cfg:"),
        "forward_estimated_time",
        "fold_estimated_time",
        "combine_estimated_time",
        "partial_storage_ratio",
        "resident_programs_per_sm",
    ))


if __name__ == "__main__":
    main()
