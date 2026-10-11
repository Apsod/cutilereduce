"""Formatting for target-independent tiled stages."""

from .ssa import TiledStage


def format_tiled_fold(program: TiledStage) -> str:
    loop = program.loop
    if loop is None:
        raise ValueError("fold formatting requires a loop stage")
    owners = ", ".join(
        f"{axis.axis}[tile={axis.tile}, programs={axis.programs}]"
        for axis in program.effective_partitions
        if axis.axis != loop.axis.axis
    )
    fold = next(
        axis for axis in program.effective_partitions if axis.axis == loop.axis.axis
    )
    folds = f"{fold.axis}[tile={fold.tile}, programs={fold.programs}]"
    return "\n".join(
        (
            f"tiled_fold {program.name} {{",
            f"  parallel owners ({owners}) {{",
            "    %state = call @execution_identity",
            f"    serial fold ({folds}) iter_args(%state) {{",
            "      %next = call @map_fold_combine(tile_inputs, %state)",
            "      yield %next",
            "    }",
            "    %output = call @project_output(%state)",
            "    store %output",
            "  }",
            "}",
        )
    )
