import polars as pl
from types import SimpleNamespace

from cutilereduce.core import (
    bounded_tile_configs,
    canonicalize_grouped_configs,
    expr,
    mono,
    partition_configs_for_tiles,
    polynomial_upper_bound,
)
from cutilereduce.core.variables import SMEM_PER_SM


def test_grouped_configs_keep_one_representative_per_product():
    frame = pl.DataFrame(
        {
            "cfg:l_tile": [1, 2, 4, 8],
            "cfg:g_tile": [8, 4, 2, 1],
        }
    )
    result = canonicalize_grouped_configs(
        None,
        frame,
        [mono(1, l_tile=1, g_tile=1)],
        axis_order=("l_tile", "g_tile"),
    )

    assert result.height == 1
    assert result.row(0) == (1, 8)


def test_grouped_configs_preserve_exact_derived_measure_differences():
    frame = pl.DataFrame(
        {
            "cfg:l_tile": [2, 4],
            "cfg:g_tile": [4, 2],
            "num_tiles": [8, 7],
        }
    )
    result = canonicalize_grouped_configs(
        None,
        frame,
        [mono(1, l_tile=1, g_tile=1)],
        axis_order=("l_tile", "g_tile"),
    )

    assert result.height == 2


def test_grouped_configs_preserve_independent_partition_count():
    frame = pl.DataFrame(
        {
            "cfg:l_tile": [1, 2, 1, 2],
            "cfg:g_tile": [2, 1, 2, 1],
            "cfg:partition_count": [1, 1, 2, 2],
        }
    )
    result = canonicalize_grouped_configs(
        None,
        frame,
        [mono(1, l_tile=1, g_tile=1)],
        axis_order=("l_tile", "g_tile"),
    )
    assert result.height == 2
    assert sorted(result["cfg:partition_count"].to_list()) == [1, 2]


def test_grouped_sweep_can_be_disabled():
    frame = pl.DataFrame(
        {
            "cfg:l_tile": [1, 2],
            "cfg:g_tile": [2, 1],
        }
    )
    result = canonicalize_grouped_configs(
        None,
        frame,
        [expr.Symbol("l_tile") * expr.Symbol("g_tile")],
        axis_order=("l_tile", "g_tile"),
    )
    assert result.height == 1


def test_grouped_sweep_defers_weighted_products():
    frame = pl.DataFrame(
        {
            "cfg:x_tile": [1, 2],
            "cfg:y_tile": [4, 2],
        }
    )
    result = canonicalize_grouped_configs(
        None,
        frame,
        [mono(1, x_tile=2, y_tile=1)],
        axis_order=("x_tile", "y_tile"),
    )
    assert result.height == 2


def _bound_stage(residency, storage=None):
    return SimpleNamespace(
        domain=SimpleNamespace(compute_axes=()),
        cost=SimpleNamespace(
            residency_bytes=residency,
            partial_storage_ratio=storage or expr.Symbol("partition_count"),
            resident_programs_per_sm_symbol=expr.Symbol("resident"),
        ),
    )


def test_bounded_tiles_prune_coupled_residency_products():
    stage = _bound_stage(expr.sympify(mono(1, l_tile=1, g_tile=1)))
    result = bounded_tile_configs(
        stage,
        (SimpleNamespace(id="l"), SimpleNamespace(id="g")),
        ("cfg:l_tile", "cfg:g_tile"),
        {"l": 8, "g": 8},
        8,
        symbols={SMEM_PER_SM: 16},
    )
    assert result.height > 0
    assert all(
        left * right <= 16
        for left, right in result.select(["cfg:l_tile", "cfg:g_tile"]).iter_rows()
    )


def test_partition_bound_is_inferred_from_storage_ratio():
    stage = _bound_stage(expr.sympify(mono(1)), expr.Symbol("partition_count") / 2)
    result = partition_configs_for_tiles(
        stage,
        pl.DataFrame({"cfg:fold_tile": [1]}),
        count_name="cfg:partition_count",
        fold_tile_name="cfg:fold_tile",
        fold_extent=8,
        max_storage_ratio=2,
        symbols={},
    )
    assert result["cfg:partition_count"].to_list() == [2, 3, 4]


def test_polynomial_upper_bound_handles_coupled_monomials():
    bound = polynomial_upper_bound(
        mono(2, x=1, y=1) + mono(3, x=2),
        "x",
        20,
        {"y": 2},
    )
    assert bound == 2
