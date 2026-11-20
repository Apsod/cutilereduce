"""Inspect general region/domain facts on asymmetric attention; no target needed.

uv run -m examples.experimental_accounting
"""

import json

from cutilereduce.core.measure import Measure
from cutilereduce.experimental.planning.accounting import ExecutionScope, ScheduleDomain
from cutilereduce.experimental import FoldMaps, FoldOperator, FoldSpec, LogWeightedSum, StructuredValue, TensorSpec


def main():
    sizes = {"h": 4, "l": 12, "g": 3, "r": 10, "dqk": 8, "dv": 6}
    spec = FoldSpec({"q": TensorSpec(("h", "l", "g", "dqk")),
                     "k": TensorSpec(("h", "r", "dqk")),
                     "v": TensorSpec(("h", "r", "dv"))}, ("h", "l", "g"), ("r",))
    def factor(ctx, x):
        return StructuredValue({"log_weight": ctx.contract(x["q"], x["k"], over=("dqk",)), "value": x["v"]})
    program = FoldOperator(spec, LogWeightedSum(("dv",)), FoldMaps(factor)).lower()
    facts = program.region_facts["map_fold"]
    reports = []
    for label, tiles in (("head_tile", {"h": 2, "l": 2, "g": 1, "r": 5}),
                         ("query_tile", {"h": 1, "l": 4, "g": 1, "r": 5})):
        owners = {a: (1 if a == "r" else (sizes[a] + t - 1) // t) for a, t in tiles.items()}
        domain = ScheduleDomain.from_maps(sizes, tiles=tiles, owners=owners,
                                         iteration_axes=("h", "l", "g", "r"), loop_axes=("r",),
                                         scope=ExecutionScope("owner", ExecutionScope("device")))
        reports.append({"name": label, "domain": domain.to_dict(),
                        "qk_dimensions": facts.contractions[0].dimensions(domain.measures(Measure.tile)),
                        "qk_flops": facts.contractions[0].flops(domain.measures(Measure.tile)),
                        "inputs": {name: {"axes": value.axes, "tile_bytes": domain.buffer(value).tile_bytes,
                                          "logical_accessed_bytes": domain.buffer(value).logical_accessed_bytes,
                                          "persistent": domain.buffer(value).persistent,
                                          "owner_replication": domain.buffer(value).owner_replicas}
                                   for name, value in facts.arguments}})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
