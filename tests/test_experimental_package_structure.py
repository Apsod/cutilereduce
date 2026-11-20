"""Keep compiler layers independent of target execution and emission."""

import ast
import importlib.util
from pathlib import Path

import cutilereduce.experimental as exp


def test_public_entry_points_are_preserved():
    from cutilereduce.experimental.backends.cpu import compile_cpu, emit_cpp_source
    from cutilereduce.experimental.backends.cutile import emit_cutile_source
    from cutilereduce.experimental.compiler.operator import FoldOperator
    from cutilereduce.experimental.ir import TensorBuilder

    from cutilereduce.experimental.api import compile
    assert exp.compile is compile
    assert callable(compile_cpu) and callable(emit_cpp_source) and callable(emit_cutile_source)
    assert not hasattr(exp, "emit_cpp_source")
    assert exp.FoldOperator is FoldOperator
    assert exp.TensorBuilder is TensorBuilder
    assert len(exp.__all__) == len(set(exp.__all__))
    assert all(hasattr(exp, name) for name in exp.__all__)


def test_dependencies_point_toward_lower_layers():
    root = Path(exp.__file__).parent
    ranks = {"ir": 0, "algebra": 1, "compiler": 2, "planning": 3, "backends": 4}
    prefix = "cutilereduce.experimental."
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        layer = relative.parts[0]
        if layer not in ranks:
            continue  # The public facade intentionally reexports every layer.
        module = prefix + ".".join(relative.with_suffix("").parts)
        package = (
            module.removesuffix(".__init__")
            if path.name == "__init__.py"
            else module.rsplit(".", 1)[0]
        )
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            dependency = node.module or ""
            if node.level:
                dependency = importlib.util.resolve_name(
                    "." * node.level + dependency, package
                )
            if not dependency.startswith(prefix):
                continue
            dependency_layer = dependency.removeprefix(prefix).split(".")[0]
            if dependency_layer in ranks:
                assert ranks[dependency_layer] <= ranks[layer], (
                    f"{relative} depends on higher layer {dependency}"
                )


def test_region_emitters_do_not_depend_on_schedules_or_runtime():
    root = Path(exp.__file__).parent / "backends"
    for target in ("cpu", "cutile"):
        path = root / target / "regions.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                dependency = node.module or ""
                assert not dependency.startswith("cutilereduce.experimental.planning")
                assert not dependency.endswith(("schedule", "runtime", "codegen"))
