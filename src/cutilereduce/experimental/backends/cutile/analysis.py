"""Explicit cuTile capabilities and shared analytical rules."""

from cutilereduce.experimental.planning.analysis import BackendModel
from cutilereduce.experimental.ir.passes import realize_contraction_precision


def cutile_analytical_target(target=None, **rates):
    """Describe supported execution without probing or loading GPU hardware."""
    from cutilereduce.experimental.planning.parametric import schedule_class
    from dataclasses import asdict
    from .target import CuTileTarget
    if target is not None:
        if target.device_count != 1:
            raise NotImplementedError("distributed GPU execution is not implemented")
        values = {k: getattr(target, k) for k in ("contraction_flops_per_second",
            "scalar_ops_per_second", "transcendental_ops_per_second", "bytes_per_second")}
        rates = {**values, "workers": target.multiprocessor_count, **rates}
    hardware = target or CuTileTarget("unspecified", rates.get("workers", 1))

    def realize(problem, candidate):
        from .backend import CuTileBackend
        from cutilereduce.experimental.planning.backend import BackendProgram
        from cutilereduce.experimental.planning.templates import build_schedule_space
        prepared = BackendProgram(problem.program, CuTileTarget(**dict(problem.target.hardware_identity)), problem.target,
            build_schedule_space(problem.program), tuple(sorted(problem.sizes.items())))
        return CuTileBackend().realize(prepared, candidate)

    return BackendModel("cutile", model_metadata={"name": "cutile_generic_v1"}, hardware_identity=asdict(hardware), supported_forward_families=("online",),
        power_of_two_tiles=True, supports_prefix=False, fuse_online_projection=True,
        embed_full_extent=False, max_partitions=1,
        region_transform=realize_contraction_precision, realization_factory=realize, batch_key=schedule_class, **rates)
