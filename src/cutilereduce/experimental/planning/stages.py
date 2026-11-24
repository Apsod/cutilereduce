"""Executable stage contracts, identity and compatibility-aware sequential composition.

Value dependencies permit independent tuning when artifact contracts stay fixed.
Backends expose actual callable/fused units; tensor-region boundaries are not used
as a proxy for launch boundaries. Shared storage/placement choices must be encoded
in artifact representations or explicit coupling bindings.
"""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import math

from .analysis import ScheduleCandidate
from .parametric import parameter_values
from .tuning import program_fingerprint, target_fingerprint


@dataclass(frozen=True)
class Artifact:
    name: str
    representation: tuple


@dataclass(frozen=True)
class Stage:
    slot: str
    direction: str
    identity: tuple
    requires: tuple[Artifact, ...]
    produces: tuple[Artifact, ...]
    parameters: tuple[tuple[str, int], ...]
    representative: ScheduleCandidate
    cost_names: tuple[str, ...] = ()
    couplings: tuple[tuple[str, object], ...] = ()

    @cached_property
    def key(self):
        return hashlib.sha256(
            repr((self.identity, self.requires, self.produces, self.couplings)).encode()
        ).hexdigest()


@dataclass(frozen=True)
class StagedPlan:
    stages: tuple[Stage, ...]
    external: tuple[Artifact, ...]
    execution_order: str = "sequential"

    def check(self):
        if self.execution_order != "sequential":
            raise NotImplementedError("overlapping stages require a joint tuning unit")
        available = {a.name: a for a in self.external}
        if len(available) != len(self.external):
            raise ValueError("external artifact names must be unique")
        seen, bindings = set(), {}
        backward = False
        for stage in self.stages:
            if stage.direction not in ("forward", "backward"):
                raise ValueError("invalid stage direction")
            if stage.slot in seen:
                raise ValueError("stage slots must be unique within a staged plan")
            seen.add(stage.slot)
            if stage.direction == "backward":
                backward = True
            elif backward:
                raise ValueError("forward stages cannot follow backward stages")
            for artifact in stage.requires:
                if available.get(artifact.name) != artifact:
                    raise ValueError(f"incompatible/missing artifact {artifact.name!r}")
                if stage.direction == "forward" and artifact.name == "cotangents":
                    raise ValueError("forward execution cannot consume cotangents")
            for name, value in stage.couplings:
                if name in bindings and bindings[name] != value:
                    raise ValueError(f"incompatible coupling {name!r}")
                bindings[name] = value
            for artifact in stage.produces:
                if artifact.name in available:
                    raise ValueError(
                        "artifact writes need distinct versioned names; aliasing stages must be fused"
                    )
                available[artifact.name] = artifact
        return self


def namespace(prepared):
    # BackendProgram caches this once; neither kernel identity nor compatibility
    # depends on unrelated stage parameters.
    return (
        program_fingerprint(prepared.program),
        target_fingerprint(prepared.analytical_target),
        prepared.sizes,
    )


def standard_plan(prepared, candidate, *, namespace_key, embed_tiled, layout):
    """Describe the callable units shared by the current CPU/cuTile runtimes."""
    sizes, program = dict(prepared.sizes), prepared.program
    p = parameter_values_from_sizes(program, sizes, candidate)
    axes = (*program.spec.batch_axes, *program.spec.fold_axes)

    def signature(region, results=True):
        items = region.results if results else region.arguments
        return tuple(
            (
                n,
                region.nodes[i].axes,
                region.nodes[i].dtype,
                tuple(sizes[a] for a in region.nodes[i].axes),
            )
            for n, i in items
        )

    inputs = Artifact(
        "inputs",
        tuple(
            (n, s.axes, s.dtype, tuple(sizes[a] for a in s.axes))
            for n, s in program.spec.inputs.items()
        ),
    )
    outputs = Artifact("outputs", signature(program.project_output))
    execution = Artifact(
        "execution",
        (
            layout,
            signature(program.map_fold),
            ("checkpoints", p["partition_count"])
            if program.requires_prefix and candidate.backward
            else (),
        ),
    )
    cotangents = Artifact("cotangents", outputs.representation)
    forward_params = tuple((f"tile.{a}", p[f"tile.{a}"]) for a in axes)
    if candidate.family.startswith("partial"):
        forward_params += (("partition_count", p["partition_count"]),)
    forward = Stage(
        "forward",
        "forward",
        (namespace_key, "forward", candidate.family, forward_params),
        (inputs,),
        (outputs, execution) if candidate.backward else (outputs,),
        forward_params,
        candidate,
    )
    if not candidate.backward:
        return StagedPlan((forward,), (inputs,)).check()
    accumulator = Artifact("accumulator", (layout, signature(program.embed_output_vjp)))
    gradients = Artifact(
        "gradients",
        tuple(
            (n, s.axes, s.dtype)
            for n, s in program.spec.inputs.items()
            if s.differentiable
        ),
    )
    # Embed does not consume checkpoints; declare the carrier-only contract so
    # checkpoint layout cannot unnecessarily multiply embed implementations.
    carrier = Artifact("carrier", (layout, signature(program.map_fold)))
    execution = Artifact(
        "execution",
        (
            carrier.representation,
            ("checkpoints", p["partition_count"]) if program.requires_prefix else (),
        ),
    )
    forward = Stage(
        forward.slot,
        forward.direction,
        forward.identity,
        forward.requires,
        (outputs, carrier, execution),
        forward.parameters,
        candidate,
    )
    embed_params = (
        tuple(
            (f"embed_tile.{a}", p[f"embed_tile.{a}"]) for a in program.spec.batch_axes
        )
        if embed_tiled
        else ()
    )
    embed = Stage(
        "embed",
        "backward",
        (namespace_key, "embed", embed_params),
        (carrier, cotangents),
        (accumulator,),
        embed_params,
        candidate,
        ("embed_output_vjp",),
    )
    final_params = tuple((f"backward_tile.{a}", p[f"backward_tile.{a}"]) for a in axes)
    final_params += (
        (
            "loop_programs",
            p["partition_count"] if program.requires_prefix else p["loop_programs"],
        ),
    )
    finalize = Stage(
        "finalize",
        "backward",
        (namespace_key, "finalize", candidate.loop_axis, final_params),
        (inputs, execution, accumulator, cotangents),
        (gradients,),
        final_params,
        candidate,
        ("map_finalize_map_vjp",),
    )
    return StagedPlan((forward, embed, finalize), (inputs, cotangents)).check()


def parameter_values_from_sizes(program, sizes, candidate):
    # Reuse canonical fallback semantics without constructing another planner.
    from types import SimpleNamespace

    return parameter_values(SimpleNamespace(program=program, sizes=sizes), candidate)


def stage_candidates(problem, parameter_domains=None):
    """Union independent parameter sweeps, rather than their Cartesian product."""
    domains = dict(problem.target.extra_parameter_domains) | dict(
        parameter_domains or {}
    )
    axes = (*problem.program.spec.batch_axes, *problem.program.spec.fold_axes)
    full = {f"tile.{a}": (problem.sizes[a],) for a in axes}
    backward_domains = {
        k: v for k, v in domains.items() if not k.startswith("embed_tile.")
    }
    forward_domains = {
        k: v
        for k, v in domains.items()
        if not k.startswith(("backward_tile.", "embed_tile.")) and k != "loop_programs"
    }
    # Obtain the backend's standard tile domains from its forward generator.
    # This visits forward geometry once and does not repeat backward domains.
    tile_domains = {a: set() for a in axes}
    for c in problem.candidates(backward=False, parameter_domains=forward_domains):
        for a in axes:
            tile_domains[a].add(dict(c.parameters)[f"tile.{a}"])
        if (
            not problem.program.requires_prefix
            or c.family == "partial_scan_checkpoints"
        ):
            baseline = dict(c.parameters)
            for name, choices in domains.items():
                if (
                    name.startswith(("backward_tile.", "embed_tile."))
                    or name == "loop_programs"
                ):
                    baseline[name] = choices[0]
            loop = problem.program.spec.fold_axes[0]
            if not problem.program.requires_prefix:
                loop = next(
                    (
                        a
                        for a in reversed(axes)
                        if problem.sizes[a] >= baseline.get("loop_programs", 1)
                    ),
                    loop,
                )
            yield ScheduleCandidate(
                c.family, tuple(sorted(baseline.items())), True, loop
            )
    for a in axes:
        backward_domains[f"tile.{a}"] = (
            (max(tile_domains[a]),) if tile_domains[a] else full[f"tile.{a}"]
        )
        backward_domains.setdefault(
            f"backward_tile.{a}", tuple(sorted(tile_domains[a]))
        )
    for a in problem.program.spec.batch_axes:
        backward_domains[f"embed_tile.{a}"] = (
            (domains[f"embed_tile.{a}"][0],)
            if f"embed_tile.{a}" in domains
            else backward_domains[f"tile.{a}"]
        )
    yield from problem.candidates(backward=True, parameter_domains=backward_domains)
    if not problem.target.embed_full_extent:
        embed_domains = {
            f"{prefix}.{a}": (max(tile_domains[a]),)
            if tile_domains[a]
            else full[f"tile.{a}"]
            for prefix in ("tile", "backward_tile")
            for a in axes
        }
        for name, choices in domains.items():
            if name.startswith("embed_tile."):
                embed_domains[name] = choices
            elif name.startswith("backward_tile.") or name in (
                "loop_programs",
                "partition_count",
            ):
                embed_domains[name] = (choices[0],)
        for a in problem.program.spec.batch_axes:
            embed_domains.setdefault(f"embed_tile.{a}", tuple(sorted(tile_domains[a])))
        yield from problem.candidates(backward=True, parameter_domains=embed_domains)


def estimated_time(cost, stage):
    if not stage.cost_names:
        return float(cost.forward_seconds)
    template = getattr(getattr(cost, "batch", None), "template", cost)
    indices = [i for i, s in enumerate(template.stages) if s.name in stage.cost_names]
    if len(indices) != len(stage.cost_names):
        raise ValueError("stage cost selector does not match analytical execution")
    if hasattr(cost, "stage_seconds"):
        return cost.stage_seconds(indices)
    return sum(float(cost.stages[i].estimated_seconds) for i in indices)


def best_compatible(plans, pools, times, excluded=()):
    """Dynamic programming over artifact contracts, not validated plan products."""
    best = None
    width = len(excluded) + 1
    for plan in plans:
        states = {(tuple(sorted(plan.external, key=lambda a: a.name)), ()): [(0.0, ())]}
        for slot in plan.stages:
            updated = {}
            for (available_tuple, coupling_tuple), paths in states.items():
                for total, selected in paths:
                    available, bindings = (
                        {a.name: a for a in available_tuple},
                        dict(coupling_tuple),
                    )
                    for option in pools[slot.slot]:
                        if option.direction != slot.direction or any(
                            a.name in available for a in option.produces
                        ):
                            continue
                        if option.key not in times or any(
                            available.get(a.name) != a for a in option.requires
                        ):
                            continue
                        if any(
                            k in bindings and bindings[k] != v
                            for k, v in option.couplings
                        ):
                            continue
                        values = available | {a.name: a for a in option.produces}
                        shared = bindings | dict(option.couplings)
                        key = (
                            tuple(sorted(values.values(), key=lambda a: a.name)),
                            tuple(sorted(shared.items())),
                        )
                        value = (total + times[option.key], (*selected, option))
                        paths_for_key = updated.setdefault(key, [])
                        paths_for_key.append(value)
                        paths_for_key.sort(key=lambda item: item[0])
                        del paths_for_key[width:]
            states = updated
        for paths in states.values():
            for value in paths:
                if tuple(s.key for s in value[1]) in excluded:
                    continue
                if best is None or value[0] < best[0]:
                    best = value
    if best is None or not math.isfinite(best[0]):
        raise RuntimeError("no compatible measured stage plan")
    return best[1]
