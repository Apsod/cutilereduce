"""C++ emission for scheduled CPU stages, loops, storage, and gradient writes."""

from __future__ import annotations

import itertools
import math

from cutilereduce.experimental.backends.cpu.schedule import (
    CpuExpr,
    CpuLeaf,
    CpuLoop,
    CpuScheduledPlan,
)
from cutilereduce.experimental.ir.core import TensorProgram
from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.templates import (
    FoldTopology,
    MapFoldRegion,
)
from cutilereduce.experimental.ir.passes import program_to_region
from cutilereduce.experimental.planning.scheduled import (
    CommitMode,
    LocalLifetime,
    ScheduledBackwardProgram,
)
from cutilereduce.experimental.planning.ssa import liveness, scheduled_to_ssa


from .regions import _CPP_DTYPE, _attr, _cpp_name, _axes_literal, _align, _emit_region as _emit_region_raw


from contextvars import ContextVar
from functools import wraps
from .realization import CpuRegionRealizations

_realizations: ContextVar[CpuRegionRealizations | None] = ContextVar("cpu_region_realizations", default=None)


def _emit_region(name, region, sizes, **kwargs):
    realizations = _realizations.get()
    if realizations is not None:
        kwargs["implementation"] = realizations.realize(region)
    return _emit_region_raw(name, region, sizes, **kwargs)


def _with_realizations(function):
    @wraps(function)
    def emit(*args, region_realizations=None, **kwargs):
        token = _realizations.set(region_realizations or CpuRegionRealizations())
        try:
            return function(*args, **kwargs)
        finally:
            _realizations.reset(token)
    return emit


def _scheduled_local_sizes(
    program: ScheduledBackwardProgram,
    sizes: dict[str, int],
) -> tuple[dict[str, int], ...]:
    axes = (*program.program_axes, program.loop_axis)
    choices = []
    for axis in axes:
        remainder = axis.extent % axis.tile
        choices.append((axis.tile,) if remainder == 0 else (axis.tile, remainder))
    variants = []
    for extents in itertools.product(*choices):
        local = dict(sizes)
        for axis, extent in zip(axes, extents, strict=True):
            local[axis.axis] = extent
        if local not in variants:
            variants.append(local)
    return tuple(variants)


def _emit_scheduled_backward(
    problem: FoldProgram,
    program: ScheduledBackwardProgram,
    sizes: dict[str, int],
    *,
    nested_names: dict[int, str],
) -> tuple[list[str], str]:
    """Emit a program-local loop with tile-resident gradients and commits."""
    if not liveness(scheduled_to_ssa(program)):
        raise ValueError("scheduled backward must define SSA values")
    if problem.requires_prefix or program.threads_prefix:
        raise NotImplementedError(
            "scheduled CPU lowering initially supports prefix-free backward"
        )
    region = problem.map_finalize_map_vjp
    if region is None:
        raise ValueError("scheduled CPU backward requires a commutative region")
    if tuple(name for name, _ in region.arguments) != tuple(
        binding.name for binding in program.arguments
    ):
        raise ValueError("scheduled arguments do not match backward region")
    result_indices = {name: index for index, (name, _) in enumerate(region.results)}
    for gradient in program.gradients:
        if gradient.result_name not in result_indices:
            raise ValueError(f"missing scheduled result {gradient.result_name!r}")
        node = region.nodes[dict(region.results)[gradient.result_name]]
        if (
            gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED
            and node.dtype != "float32"
        ):
            raise ValueError(
                "scheduled CPU atomic gradients initially require float32, got "
                f"{gradient.input_name!r}: {node.dtype}"
            )

    local_sizes = _scheduled_local_sizes(program, sizes)
    variant_names = [
        f"scheduled_backward_region_{index}" for index in range(len(local_sizes))
    ]
    variants = [
        _emit_region(
            name,
            region,
            shape,
            nested_names=nested_names,
        )
        for name, shape in zip(variant_names, local_sizes, strict=True)
    ]

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name, _ in region.arguments
    )
    lines = [f"std::vector<torch::Tensor> scheduled_backward({parameters}) {{"]
    if (program.loop_programs == 1 and program.loop_axis.count == 1
            and all(axis.count == 1 for axis in program.program_axes)
            and len(local_sizes) == 1):
        # One complete region already owns every returned gradient. Forward its
        # results directly instead of allocating/copying destination buffers.
        arguments = ", ".join(_cpp_name(name) for name, _ in region.arguments)
        lines.append(f"  auto partial = {variant_names[0]}({arguments});")
        results = ", ".join(f"partial[{result_indices[g.result_name]}]" for g in program.gradients)
        lines.extend((f"  return {{{results}}};", "}"))
        return variants, "\n".join(lines)
    for gradient in program.gradients:
        node = region.nodes[dict(region.results)[gradient.result_name]]
        factory = "empty" if gradient.commit_mode is CommitMode.STORE else "zeros"
        lines.append(
            f"  auto grad_{_cpp_name(gradient.input_name)} = at::{factory}("
            f"{_axes_literal(gradient.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    program_count = math.prod(axis.count for axis in program.program_axes)
    lines.append("  auto run_programs = [&](int64_t begin, int64_t end) {")
    lines.append("    at::NoGradGuard no_grad;")
    lines.append(
        "    for (int64_t program_id = begin; program_id < end; ++program_id) {"
    )
    lines.append("      int64_t program_remainder = program_id;")
    for axis in reversed(program.program_axes):
        name = _cpp_name(axis.axis)
        lines.extend(
            (
                f"      const int64_t index_{name} = program_remainder % {axis.count};",
                f"      program_remainder /= {axis.count};",
                f"      const int64_t start_{name} = index_{name} * {axis.tile};",
                f"      const int64_t size_{name} = std::min<int64_t>({axis.tile}, {axis.extent} - start_{name});",
            )
        )
    for gradient in program.gradients:
        if gradient.lifetime is LocalLifetime.PERSISTENT:
            lines.append(f"      torch::Tensor local_{_cpp_name(gradient.input_name)};")
    loop = program.loop_axis
    loop_name = _cpp_name(loop.axis)
    lines.extend(
        (
            f"      for (int64_t start_{loop_name} = 0; start_{loop_name} < {loop.extent}; start_{loop_name} += {loop.tile}) {{",
            f"        const int64_t size_{loop_name} = std::min<int64_t>({loop.tile}, {loop.extent} - start_{loop_name});",
        )
    )
    scheduled_axes = {axis.axis: axis for axis in (*program.program_axes, loop)}
    for binding in program.arguments:
        variable = _cpp_name(binding.name)
        lines.append(f"        auto tile_{variable} = {variable};")
        for axis in binding.sliced_axes:
            dimension = binding.axes.index(axis)
            axis_name = _cpp_name(axis)
            lines.append(
                f"        tile_{variable} = tile_{variable}.narrow({dimension}, "
                f"start_{axis_name}, size_{axis_name});"
            )
    call_arguments = ", ".join(
        f"tile_{_cpp_name(binding.name)}" for binding in program.arguments
    )
    for index, (name, shape) in enumerate(zip(variant_names, local_sizes, strict=True)):
        conditions = (
            " && ".join(
                f"size_{_cpp_name(axis)} == {shape[axis]}" for axis in scheduled_axes
            )
            or "true"
        )
        keyword = "if" if index == 0 else "else if"
        lines.append(f"        {keyword} ({conditions}) {{")
        lines.append(f"          auto partial = {name}({call_arguments});")

        def destination_lines(gradient, source: str, indent: str) -> list[str]:
            destination = f"destination_{_cpp_name(gradient.input_name)}"
            result = [
                f"{indent}auto {destination} = grad_{_cpp_name(gradient.input_name)};"
            ]
            for axis in gradient.axes:
                if axis not in scheduled_axes:
                    continue
                dimension = gradient.axes.index(axis)
                axis_name = _cpp_name(axis)
                result.append(
                    f"{indent}{destination} = {destination}.narrow({dimension}, "
                    f"start_{axis_name}, size_{axis_name});"
                )
            if gradient.commit_mode is CommitMode.STORE:
                result.append(f"{indent}{destination}.copy_({source});")
            elif gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED:
                result.append(f"{indent}atomic_add_relaxed({destination}, {source});")
            else:
                raise ValueError("scheduled write policy is unresolved")
            return result

        for gradient in program.gradients:
            result_index = result_indices[gradient.result_name]
            source = f"partial[{result_index}]"
            local = f"local_{_cpp_name(gradient.input_name)}"
            if gradient.lifetime is LocalLifetime.PERSISTENT:
                if loop.count == 1:
                    # No later update: borrowing the partial until its commit is safe.
                    lines.append(f"          {local} = {source};")
                else:
                    # Region results can alias borrowed arguments; establish a
                    # private accumulator before subsequent in-place additions.
                    lines.append(f"          if (!{local}.defined()) {local} = {source}.clone();")
                    lines.append(f"          else {local}.add_({source});")
            else:
                lines.extend(destination_lines(gradient, source, "          "))
        lines.append("        }")
    lines.append(
        '        else { TORCH_CHECK(false, "missing scheduled tile variant"); }'
    )
    lines.append("      }")
    for gradient in program.gradients:
        if gradient.lifetime is LocalLifetime.PERSISTENT:
            lines.extend(
                destination_lines(
                    gradient,
                    f"local_{_cpp_name(gradient.input_name)}",
                    "      ",
                )
            )
    lines.extend(("    }", "  };"))
    if program_count == 1:
        # A serial owner can still use intra-op parallelism inside its GEMMs
        # and fused regions. parallel_for's serial fallback installs a guard.
        lines.append("  run_programs(0, 1);")
    else:
        lines.append(f"  at::parallel_for(0, {program_count}, 1, run_programs);")
    results = ", ".join(
        f"grad_{_cpp_name(gradient.input_name)}" for gradient in program.gradients
    )
    lines.extend((f"  return {{{results}}};", "}"))
    return variants, "\n".join(lines)


def _emit_scheduled_forward_tile(
    problem: FoldProgram,
    scheduled: CpuScheduledPlan,
    sizes: dict[str, int],
    extents: tuple[int, ...],
    *,
    nested_names: dict[int, str],
    function_name: str,
    name_suffix: str,
    map_fold_name: str,
    combine_name: str,
    identity_name: str,
    project_name: str,
) -> tuple[list[str], str, int]:
    """Emit the scheduled fold for one independently owned output tile."""
    fold_axis = problem.spec.fold_axes[0]
    full_extent = sizes[fold_axis]
    variants = []
    variant_names = {}
    for extent in extents:
        if extent == full_extent:
            variant_names[extent] = map_fold_name
            continue
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        name = f"scheduled_map_fold_{extent}_{name_suffix}"
        variant_names[extent] = name
        variants.append(
            _emit_region(
                name,
                problem.map_fold,
                local_sizes,
                nested_names=nested_names,
            )
        )

    accumulate_names = {}
    for extent in extents:
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        name = f"scheduled_map_fold_combine_{extent}_{name_suffix}"
        accumulate_names[extent] = name
        variants.append(
            _emit_region(
                name, problem.map_fold_combine, local_sizes, nested_names=nested_names
            )
        )

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name in problem.spec.inputs
    )
    lines = [f"std::vector<torch::Tensor> {function_name}({parameters}) {{"]
    lines.append(
        "  auto run_map_fold = [&](int64_t start, int64_t extent) "
        "-> std::vector<torch::Tensor> {"
    )
    for index, (name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(name)
        if fold_axis in spec.axes:
            dimension = spec.axes.index(fold_axis)
            lines.append(
                f"    auto sliced_{index} = {variable}.narrow({dimension}, start, extent).contiguous();"
            )
        else:
            lines.append(f"    const auto& sliced_{index} = {variable};")
    map_arguments = []
    input_indices = {name: index for index, name in enumerate(problem.spec.inputs)}
    for name, _ in problem.map_fold.arguments:
        if name not in input_indices:
            raise ValueError(f"map_fold argument {name!r} is not an operator input")
        map_arguments.append(f"sliced_{input_indices[name]}")
    for extent in extents:
        lines.append(
            f"    if (extent == {extent}) return {variant_names[extent]}("
            f"{', '.join(map_arguments)});"
        )
    lines.extend(
        (
            '    TORCH_CHECK(false, "missing scheduled map-fold tile variant");',
            "  };",
        )
    )

    carrier_indices = {
        name: index for index, (name, _) in enumerate(problem.map_fold.results)
    }
    carrier_nodes = {
        name: problem.map_fold.nodes[node_id]
        for name, node_id in problem.map_fold.results
    }
    lines.append(
        "  auto run_map_fold_combine = [&](int64_t start, int64_t extent, "
        "const std::vector<torch::Tensor>& state) -> std::vector<torch::Tensor> {"
    )
    for index, (name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(name)
        if fold_axis in spec.axes:
            lines.append(
                f"    auto sliced_{index} = {variable}.narrow("
                f"{spec.axes.index(fold_axis)}, start, extent).contiguous();"
            )
        else:
            lines.append(f"    const auto& sliced_{index} = {variable};")
    accumulate_arguments = []
    for name, node_id in problem.map_fold_combine.arguments:
        target_axes = problem.map_fold_combine.nodes[node_id].axes
        if name in input_indices:
            accumulate_arguments.append(
                _align(
                    f"sliced_{input_indices[name]}",
                    problem.spec.inputs[name].axes,
                    target_axes,
                )
            )
        elif name.startswith("accumulator."):
            result_name = "execution." + name.removeprefix("accumulator.")
            accumulate_arguments.append(
                _align(
                    f"state[{carrier_indices[result_name]}]",
                    carrier_nodes[result_name].axes,
                    target_axes,
                )
            )
        else:
            raise ValueError(f"unrecognized map_fold_combine argument {name!r}")
    accumulate_indices = {
        name: index for index, (name, _) in enumerate(problem.map_fold_combine.results)
    }
    accumulate_nodes = dict(problem.map_fold_combine.results)
    reordered = ", ".join(
        _align(
            f"result[{accumulate_indices[name]}]",
            problem.map_fold_combine.nodes[accumulate_nodes[name]].axes,
            carrier_nodes[name].axes,
        )
        for name in carrier_indices
    )
    for extent in extents:
        lines.extend(
            (
                f"    if (extent == {extent}) {{",
                f"      auto result = {accumulate_names[extent]}({', '.join(accumulate_arguments)});",
                f"      return {{{reordered}}};",
                "    }",
            )
        )
    lines.extend(
        ('    TORCH_CHECK(false, "missing scheduled accumulate tile variant");', "  };")
    )
    identity_indices = {
        name: index
        for index, (name, _) in enumerate(problem.execution_identity.results)
    }
    identity_nodes = dict(problem.execution_identity.results)
    identity_parts = []
    for name, node_id in problem.map_fold.results:
        carrier = problem.map_fold.nodes[node_id]
        identity_node = problem.execution_identity.nodes[identity_nodes[name]]
        aligned = _align(
            f"identity[{identity_indices[name]}]", identity_node.axes, carrier.axes
        )
        identity_parts.append(
            f"({aligned}).expand({_axes_literal(carrier.axes, sizes)})"
        )
    identity_values = ", ".join(identity_parts)
    lines.extend(
        (
            "  auto initial_state = [&]() -> std::vector<torch::Tensor> {",
            f"    auto identity = {identity_name}();",
            f"    return {{{identity_values}}};",
            "  };",
        )
    )
    combine_arguments = []
    for name, node_id in problem.execution_combine.arguments:
        side, suffix = name.split(".", 1)
        result_name = f"execution.{suffix}"
        if result_name not in carrier_indices or side not in {"left", "right"}:
            raise ValueError(
                "execution_combine arguments do not match map_fold results"
            )
        combine_arguments.append(
            _align(
                f"{side}[{carrier_indices[result_name]}]",
                carrier_nodes[result_name].axes,
                problem.execution_combine.nodes[node_id].axes,
            )
        )
    combine_indices = {
        name: index for index, (name, _) in enumerate(problem.execution_combine.results)
    }
    combine_nodes = dict(problem.execution_combine.results)
    combined_values = ", ".join(
        _align(
            f"result[{combine_indices[name]}]",
            problem.execution_combine.nodes[combine_nodes[name]].axes,
            carrier_nodes[name].axes,
        )
        for name in carrier_indices
    )
    lines.extend(
        (
            "  auto combine_carriers = [&](const std::vector<torch::Tensor>& left,",
            "                              const std::vector<torch::Tensor>& right) -> std::vector<torch::Tensor> {",
            f"    auto result = {combine_name}({', '.join(combine_arguments)});",
            f"    return {{{combined_values}}};",
            "  };",
            "  auto combine_many = [&](std::vector<std::vector<torch::Tensor>> values, bool tree) {",
            '    TORCH_CHECK(!values.empty(), "scheduled fold produced no carriers");',
            "    if (!tree) {",
            "      auto state = std::move(values[0]);",
            "      for (size_t i = 1; i < values.size(); ++i)",
            "        state = combine_carriers(state, values[i]);",
            "      return state;",
            "    }",
            "    while (values.size() > 1) {",
            "      std::vector<std::vector<torch::Tensor>> next;",
            "      next.reserve((values.size() + 1) / 2);",
            "      for (size_t i = 0; i < values.size(); i += 2) {",
            "        if (i + 1 == values.size()) next.push_back(std::move(values[i]));",
            "        else next.push_back(combine_carriers(values[i], values[i + 1]));",
            "      }",
            "      values = std::move(next);",
            "    }",
            "    return std::move(values[0]);",
            "  };",
        )
    )

    lambda_names: dict[int, str] = {}
    lambda_definitions: list[str] = []

    def emit_expression(expression: CpuExpr) -> str:
        key = id(expression)
        if key in lambda_names:
            return lambda_names[key]
        name = f"eval_carrier_{len(lambda_names)}"
        lambda_names[key] = name
        if isinstance(expression, CpuLeaf):
            if not isinstance(expression.expression, MapFoldRegion):
                raise TypeError("scheduled forward carrier leaf must call map_fold")
            body = ["    return run_map_fold(start, extent);"]
        elif isinstance(expression, CpuLoop) and expression.operation == "carrier_fold":
            if (
                expression.iteration.loop.value != "parallel"
                and expression.topology is FoldTopology.ONLINE
            ):
                grain = expression.iteration.grain
                if isinstance(expression.body, CpuLeaf):
                    if not isinstance(expression.body.expression, MapFoldRegion):
                        raise TypeError("online fold leaf must call map_fold")
                    update = (
                        "state = run_map_fold_combine(start + offset, local, state);"
                    )
                else:
                    child = emit_expression(expression.body)
                    update = f"state = combine_carriers(state, {child}(start + offset, local));"
                body = [
                    "    auto state = initial_state();",
                    f"    for (int64_t offset = 0; offset < extent; offset += {grain}) {{",
                    f"      const int64_t local = std::min<int64_t>({grain}, extent - offset);",
                    f"      {update}",
                    "    }",
                    "    return state;",
                ]
                definition = [
                    f"  std::function<std::vector<torch::Tensor>(int64_t, int64_t)> {name};",
                    f"  {name} = [&](int64_t start, int64_t extent) {{",
                    *body,
                    "  };",
                ]
                lambda_definitions.extend(definition)
                return name
            child = emit_expression(expression.body)
            grain = expression.iteration.grain
            tree = "true" if expression.topology is FoldTopology.TREE else "false"
            body = [
                f"    const int64_t count = (extent + {grain} - 1) / {grain};",
                f"    if (count == 1) return {child}(start, extent);",
                "    std::vector<std::vector<torch::Tensor>> values(count);",
            ]
            loop_body = [
                f"const int64_t offset = index * {grain};",
                f"const int64_t local = std::min<int64_t>({grain}, extent - offset);",
                f"values[index] = {child}(start + offset, local);",
            ]
            if expression.iteration.loop.value == "parallel":
                body.extend(
                    [
                        "    at::parallel_for(0, count, 1, [&](int64_t begin, int64_t end) {",
                        "      at::NoGradGuard no_grad;",
                        "      for (int64_t index = begin; index < end; ++index) {",
                        *(f"        {line}" for line in loop_body),
                        "      }",
                        "    });",
                    ]
                )
            else:
                body.extend(
                    [
                        "    for (int64_t index = 0; index < count; ++index) {",
                        *(f"      {line}" for line in loop_body),
                        "    }",
                    ]
                )
            body.append(f"    return combine_many(std::move(values), {tree});")
        else:
            raise TypeError("unsupported scheduled forward carrier expression")
        definition = [
            f"  std::function<std::vector<torch::Tensor>(int64_t, int64_t)> {name};",
            f"  {name} = [&](int64_t start, int64_t extent) {{",
            *body,
            "  };",
        ]
        lambda_definitions.extend(definition)
        return name

    root = scheduled.forward[0].bindings[0].expression
    prefix_count = 0
    if isinstance(root, CpuLoop) and root.operation == "map":
        child = emit_expression(root.body)
        grain = root.iteration.grain
        lines.extend(lambda_definitions)
        lines.extend(
            (
                f"  const int64_t partition_count = ({full_extent} + {grain} - 1) / {grain};",
                "  std::vector<std::vector<torch::Tensor>> partials(partition_count);",
            )
        )
        map_body = [
            f"const int64_t start = index * {grain};",
            f"const int64_t extent = std::min<int64_t>({grain}, {full_extent} - start);",
            f"partials[index] = {child}(start, extent);",
        ]
        if root.iteration.loop.value == "parallel":
            lines.extend(
                (
                    "  at::parallel_for(0, partition_count, 1, [&](int64_t begin, int64_t end) {",
                    "    at::NoGradGuard no_grad;",
                    "    for (int64_t index = begin; index < end; ++index) {",
                    *(f"      {line}" for line in map_body),
                    "    }",
                    "  });",
                )
            )
        else:
            lines.extend(
                (
                    "  for (int64_t index = 0; index < partition_count; ++index) {",
                    *(f"    {line}" for line in map_body),
                    "  }",
                )
            )
        combine_expression = scheduled.forward[1].bindings[0].expression
        if (
            isinstance(combine_expression, CpuLoop)
            and combine_expression.operation == "exclusive_scan"
        ):
            prefix_count = root.iteration.count
            if problem.execution_identity.arguments:
                raise ValueError("execution identity must not take arguments")
            lines.extend(
                (
                    f"  auto identity = {identity_name}();",
                    "  std::vector<std::vector<torch::Tensor>> prefixes;",
                    "  prefixes.reserve(partials.size());",
                    "  std::vector<torch::Tensor> prefix_state;",
                    "  prefix_state.reserve(identity.size());",
                    "  for (size_t i = 0; i < identity.size(); ++i) {",
                    "    std::vector<int64_t> shape(partials[0][i].dim() - identity[i].dim(), 1);",
                    "    shape.insert(shape.end(), identity[i].sizes().begin(), identity[i].sizes().end());",
                    "    prefix_state.push_back(identity[i].reshape(shape).expand_as(partials[0][i]).clone());",
                    "  }",
                    "  for (const auto& partial : partials) {",
                    "    prefixes.push_back(prefix_state);",
                    "    prefix_state = combine_carriers(prefix_state, partial);",
                    "  }",
                )
            )
        topology = (
            combine_expression.topology
            if isinstance(combine_expression, CpuLoop)
            and combine_expression.topology is not None
            else FoldTopology.ONLINE
        )
        tree = "true" if topology is FoldTopology.TREE else "false"
        lines.append(f"  auto execution = combine_many(partials, {tree});")
    else:
        evaluator = emit_expression(root)
        lines.extend(lambda_definitions)
        lines.append(f"  auto execution = {evaluator}(0, {full_extent});")

    project_arguments = []
    for name, node_id in problem.project_output.arguments:
        candidate = name.replace("saved_execution.", "execution.")
        if candidate not in carrier_indices:
            raise ValueError("project_output arguments do not match the carrier")
        project_arguments.append(
            _align(
                f"execution[{carrier_indices[candidate]}]",
                carrier_nodes[candidate].axes,
                problem.project_output.nodes[node_id].axes,
            )
        )
    lines.append(f"  auto outputs = {project_name}({', '.join(project_arguments)});")
    lines.extend(
        (
            "  std::vector<torch::Tensor> result;",
            "  result.insert(result.end(), outputs.begin(), outputs.end());",
            "  result.insert(result.end(), execution.begin(), execution.end());",
        )
    )
    if prefix_count:
        lines.extend(
            (
                "  for (const auto& prefix : prefixes)",
                "    result.insert(result.end(), prefix.begin(), prefix.end());",
            )
        )
    lines.extend(("  return result;", "}"))
    return variants, "\n".join(lines), prefix_count


def _emit_scheduled_forward(
    problem: FoldProgram,
    scheduled: CpuScheduledPlan,
    sizes: dict[str, int],
    extents: tuple[int, ...],
    *,
    nested_names: dict[int, str],
) -> tuple[list[str], str]:
    """Tile all forward batch axes and emit one native owner-program grid."""
    owner_axes = problem.spec.batch_axes
    stage_tiles = dict(scheduled.forward[0].tile_sizes)
    tile_sizes = {
        axis: min(sizes[axis], stage_tiles.get(axis, sizes[axis]))
        for axis in owner_axes
    }
    extent_choices = []
    for axis in owner_axes:
        tile = tile_sizes[axis]
        remainder = sizes[axis] % tile
        extent_choices.append((tile,) if remainder == 0 else (tile, remainder))
    owner_shapes = tuple(itertools.product(*extent_choices)) if owner_axes else ((),)

    functions: list[str] = []
    tile_functions: dict[tuple[int, ...], str] = {}
    prefix_count: int | None = None
    for shape_index, shape in enumerate(owner_shapes):
        local_sizes = dict(sizes)
        local_sizes.update(zip(owner_axes, shape, strict=True))
        suffix = f"owner_{shape_index}"
        names = {
            "map": f"map_fold_{suffix}",
            "combine": f"execution_combine_{suffix}",
            "identity": f"execution_identity_{suffix}",
            "project": f"project_output_{suffix}",
            "driver": f"scheduled_forward_{suffix}",
        }
        functions.extend(
            _emit_region(
                names[key],
                region,
                local_sizes,
                nested_names=nested_names,
            )
            for key, region in (
                ("map", problem.map_fold),
                ("combine", problem.execution_combine),
                ("identity", problem.execution_identity),
                ("project", problem.project_output),
            )
        )
        variants, driver, local_prefix_count = _emit_scheduled_forward_tile(
            problem,
            scheduled,
            local_sizes,
            extents,
            nested_names=nested_names,
            function_name=names["driver"],
            name_suffix=suffix,
            map_fold_name=names["map"],
            combine_name=names["combine"],
            identity_name=names["identity"],
            project_name=names["project"],
        )
        functions.extend((*variants, driver))
        tile_functions[shape] = names["driver"]
        if prefix_count is None:
            prefix_count = local_prefix_count
        elif prefix_count != local_prefix_count:
            raise ValueError("owner-tile variants disagree on prefix count")
    prefix_count = prefix_count or 0

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name in problem.spec.inputs
    )
    lines = [f"std::vector<torch::Tensor> scheduled_forward({parameters}) {{"]
    if all(tile_sizes[axis] == sizes[axis] for axis in owner_axes):
        arguments = ", ".join(_cpp_name(name) for name in problem.spec.inputs)
        lines.extend((f"  return {tile_functions[owner_shapes[0]]}({arguments});", "}"))
        return functions, "\n".join(lines)
    output_nodes = [
        problem.project_output.nodes[node_id]
        for _, node_id in problem.project_output.results
    ]
    carrier_nodes = [
        problem.map_fold.nodes[node_id] for _, node_id in problem.map_fold.results
    ]
    for index, node in enumerate(output_nodes):
        lines.append(
            f"  auto output_{index} = at::empty({_axes_literal(node.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    for index, node in enumerate(carrier_nodes):
        lines.append(
            f"  auto carrier_{index} = at::empty({_axes_literal(node.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    for prefix in range(prefix_count):
        for index, node in enumerate(carrier_nodes):
            lines.append(
                f"  auto prefix_{prefix}_{index} = at::empty("
                f"{_axes_literal(node.axes, sizes)}, "
                f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
            )
    owner_count = math.prod(
        math.ceil(sizes[axis] / tile_sizes[axis]) for axis in owner_axes
    )
    lines.extend(
        (
            f"  at::parallel_for(0, {owner_count}, 1, [&](int64_t begin, int64_t end) {{",
            "    at::NoGradGuard no_grad;",
            "    for (int64_t owner = begin; owner < end; ++owner) {",
            "      int64_t owner_remainder = owner;",
        )
    )
    for axis in reversed(owner_axes):
        axis_name = _cpp_name(axis)
        count = math.ceil(sizes[axis] / tile_sizes[axis])
        lines.extend(
            (
                f"      const int64_t tile_index_{axis_name} = owner_remainder % {count};",
                f"      owner_remainder /= {count};",
                f"      const int64_t start_{axis_name} = tile_index_{axis_name} * {tile_sizes[axis]};",
                f"      const int64_t extent_{axis_name} = std::min<int64_t>({tile_sizes[axis]}, {sizes[axis]} - start_{axis_name});",
            )
        )
    input_arguments = []
    for input_index, (input_name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(input_name)
        lines.append(f"      auto tile_input_{input_index} = {variable};")
        for axis in owner_axes:
            if axis not in spec.axes:
                continue
            dimension = spec.axes.index(axis)
            axis_name = _cpp_name(axis)
            lines.append(
                f"      tile_input_{input_index} = tile_input_{input_index}.narrow("
                f"{dimension}, start_{axis_name}, extent_{axis_name});"
            )
        lines.append(
            f"      tile_input_{input_index} = tile_input_{input_index}.contiguous();"
        )
        input_arguments.append(f"tile_input_{input_index}")
    for shape_index, shape in enumerate(owner_shapes):
        condition = (
            " && ".join(
                f"extent_{_cpp_name(axis)} == {extent}"
                for axis, extent in zip(owner_axes, shape, strict=True)
            )
            or "true"
        )
        keyword = "if" if shape_index == 0 else "else if"
        lines.append(f"      {keyword} ({condition}) {{")
        lines.append(
            f"        auto local = {tile_functions[shape]}({', '.join(input_arguments)});"
        )
        result_nodes = [
            *output_nodes,
            *carrier_nodes,
            *(node for _ in range(prefix_count) for node in carrier_nodes),
        ]
        result_variables = [
            *(f"output_{index}" for index in range(len(output_nodes))),
            *(f"carrier_{index}" for index in range(len(carrier_nodes))),
            *(
                f"prefix_{prefix}_{index}"
                for prefix in range(prefix_count)
                for index in range(len(carrier_nodes))
            ),
        ]
        for result_index, (node, destination) in enumerate(
            zip(result_nodes, result_variables, strict=True)
        ):
            lines.append(f"        auto destination_{result_index} = {destination};")
            for axis in owner_axes:
                if axis not in node.axes:
                    continue
                dimension = node.axes.index(axis)
                axis_name = _cpp_name(axis)
                lines.append(
                    f"        destination_{result_index} = destination_{result_index}.narrow("
                    f"{dimension}, start_{axis_name}, extent_{axis_name});"
                )
            lines.append(
                f"        destination_{result_index}.copy_(local[{result_index}]);"
            )
        lines.append("      }")
    lines.extend(
        (
            '      else { TORCH_CHECK(false, "missing scheduled owner-tile variant"); }',
            "    }",
            "  });",
            "  std::vector<torch::Tensor> result;",
        )
    )
    for index in range(len(output_nodes)):
        lines.append(f"  result.push_back(output_{index});")
    for index in range(len(carrier_nodes)):
        lines.append(f"  result.push_back(carrier_{index});")
    for prefix in range(prefix_count):
        for index in range(len(carrier_nodes)):
            lines.append(f"  result.push_back(prefix_{prefix}_{index});")
    lines.extend(("  return result;", "}"))
    return functions, "\n".join(lines)


@_with_realizations
def emit_cpp_source(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    backward: bool,
    scheduled_backward: ScheduledBackwardProgram | None = None,
    scheduled_forward: CpuScheduledPlan | None = None,
    forward_extents: tuple[int, ...] = (),
) -> str:
    """Emit one extension containing the independently callable plan regions."""
    regions = {
        "map_fold": problem.map_fold,
        "project_output": problem.project_output,
        "execution_identity": problem.execution_identity,
        "execution_combine": problem.execution_combine,
    }
    if backward:
        if problem.embed_output_vjp is None:
            raise ValueError("planning problem has no backward regions")
        regions["embed_output_vjp"] = problem.embed_output_vjp
        backward_region = (
            problem.map_finalize_map_vjp or problem.map_finalize_map_vjp_accumulate
        )
        if backward_region is None:
            raise ValueError("planning problem has no map-finalize region")
        regions["map_finalize_map_vjp"] = backward_region
    nested_programs = []
    for region in regions.values():
        for node in region.nodes:
            combine = _attr(node, "combine_region")
            if isinstance(combine, TensorProgram) and all(
                combine != existing for existing in nested_programs
            ):
                nested_programs.append(combine)
    nested_names = {
        id(program): f"combine_region_{index}"
        for index, program in enumerate(nested_programs)
    }
    nested_functions = [
        _emit_region(
            nested_names[id(program)],
            program_to_region(program),
            sizes,
            nested_names=nested_names,
        )
        for program in nested_programs
    ]
    region_functions = [
        *(
            _emit_region(
                name,
                region,
                sizes,
                nested_names=nested_names,
            )
            for name, region in regions.items()
        )
    ]
    scheduled_functions: list[str] = []
    if scheduled_forward is not None:
        variants, kernel = _emit_scheduled_forward(
            problem,
            scheduled_forward,
            sizes,
            forward_extents,
            nested_names=nested_names,
        )
        scheduled_functions.extend((*variants, kernel))
    if scheduled_backward is not None:
        variants, kernel = _emit_scheduled_backward(
            problem,
            scheduled_backward,
            sizes,
            nested_names=nested_names,
        )
        scheduled_functions.extend((*variants, kernel))
    functions = "\n\n".join(
        [
            *nested_functions,
            *region_functions,
            *scheduled_functions,
        ]
    )
    bindings = "\n".join(
        f'  m.def("{name}", &{_cpp_name(name)}, '
        "py::call_guard<py::gil_scoped_release>());"
        for name in regions
    )
    if scheduled_backward is not None:
        bindings += (
            '\n  m.def("scheduled_backward", &scheduled_backward, '
            "py::call_guard<py::gil_scoped_release>());"
        )
    if scheduled_forward is not None:
        bindings += (
            '\n  m.def("scheduled_forward", &scheduled_forward, '
            "py::call_guard<py::gil_scoped_release>());"
        )
    atomic_helper = ""
    if scheduled_backward is not None:
        atomic_helper = """
void atomic_add_relaxed(torch::Tensor destination, torch::Tensor source) {
  TORCH_CHECK(destination.scalar_type() == at::kFloat, "atomic destination must be float32");
  TORCH_CHECK(source.scalar_type() == at::kFloat, "atomic source must be float32");
  auto contiguous = source.contiguous();
  TORCH_CHECK(destination.sizes() == contiguous.sizes(), "atomic tile shape mismatch");
  auto* destination_ptr = destination.data_ptr<float>();
  const auto* source_ptr = contiguous.data_ptr<float>();
  for (int64_t linear = 0; linear < contiguous.numel(); ++linear) {
    int64_t remainder = linear;
    int64_t offset = 0;
    for (int64_t dimension = contiguous.dim() - 1; dimension >= 0; --dimension) {
      const int64_t coordinate = remainder % contiguous.size(dimension);
      remainder /= contiguous.size(dimension);
      offset += coordinate * destination.stride(dimension);
    }
    std::atomic_ref<float>(destination_ptr[offset]).fetch_add(
        source_ptr[linear], std::memory_order_relaxed);
  }
}
"""
    return f"""#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
{atomic_helper}

{functions}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
{bindings}
}}
"""


__all__ = ["emit_cpp_source"]
