import polars as pl
import pytest

from cutilereduce.core.sweep_results import SweepFrame, SweepSchema


def test_typed_frame_validates_required_columns_and_joins():
    schema = SweepSchema("map", ("cfg:partition_count",))
    left = SweepFrame(
        pl.DataFrame({"cfg:partition_count": [1, 2], "map": [3.0, 4.0]}),
        schema,
    )
    right = SweepFrame(
        pl.DataFrame({"cfg:partition_count": [1, 2], "scan": [5.0, 6.0]}),
        SweepSchema("scan", ("cfg:partition_count",)),
    )

    joined = left.join(right, on="cfg:partition_count")
    assert joined.frame.columns == ["cfg:partition_count", "map", "scan"]
    assert joined.frame.height == 2


def test_typed_frame_rejects_unkeyed_join_and_implicit_vertical_concat():
    left = SweepFrame(pl.DataFrame({"a": [1]}), SweepSchema("left", ("a",)))
    right = SweepFrame(pl.DataFrame({"b": [1]}), SweepSchema("right", ("b",)))

    with pytest.raises(ValueError, match="join keys"):
        left.join(right, on="missing")
    with pytest.raises(ValueError, match="identical schemas"):
        left.concat(right)


def test_diagonal_concat_is_explicit():
    left = SweepFrame(pl.DataFrame({"path": ["full"], "full_tile": [8]}))
    right = SweepFrame(pl.DataFrame({"path": ["partial"], "partition_count": [2]}))

    combined = left.concat(right, diagonal=True).frame
    assert combined.columns == ["path", "full_tile", "partition_count"]
    assert combined.to_dicts() == [
        {"path": "full", "full_tile": 8, "partition_count": None},
        {"path": "partial", "full_tile": None, "partition_count": 2},
    ]


def test_frame_json_round_trip():
    frame = SweepFrame(
        pl.DataFrame({"path": ["full"], "estimated_time": [1.25]}),
        SweepSchema("forward.full"),
    )
    restored = SweepFrame.from_json(frame.to_json())
    assert restored.schema.name == "forward.full"
    assert restored.frame.to_dicts() == frame.frame.to_dicts()
