import ast

import pytest
import torch

from cutilereduce.experimental import TensorRegionBuilder, TensorSignature
from cutilereduce.experimental.ir.autodiff import vjp
from cutilereduce.experimental.backends.cutile.regions import _emit_region
from cutilereduce.experimental.backends.reference import _execute_region


@pytest.mark.parametrize(
    "operation",
    [
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "log1p",
        "expm1",
        "abs",
        "softplus",
    ],
)
def test_unary_vjp(operation):
    ctx = TensorRegionBuilder(("b", "d"))
    x = ctx.expect("x", TensorSignature(("b", "d"), "float64"))
    y = getattr(ctx, operation)(x)
    gradient = vjp(ctx, [(y, ctx.full(y.axes, 1.0, y.dtype))], wrt={"x": x})["x"]
    ctx.yield_value("y", y)
    ctx.yield_value("dx", gradient)
    value = torch.tensor(
        [[0.1, 0.3, 0.9], [1.1, 1.5, 2.0]], dtype=torch.float64, requires_grad=True
    )
    expected = (
        torch.nn.functional.softplus(value)
        if operation == "softplus"
        else getattr(torch, operation)(value)
    )
    expected.sum().backward()
    actual = _execute_region(ctx.finish_region(), {"x": value.detach()})
    torch.testing.assert_close(actual["y"], expected)
    torch.testing.assert_close(actual["dx"], value.grad)


@pytest.mark.parametrize("exponent", [-3, -1, 0, 1, 2, 7])
def test_integer_power(exponent):
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float64"))
    y = x**exponent
    gradients = vjp(ctx, [(y, ctx.full(y.axes, 1.0, y.dtype))], wrt={"x": x})
    ctx.yield_value("y", y)
    ctx.yield_value("dx", gradients.get("x", ctx.zero(x.axes, x.dtype)))
    value = torch.tensor(
        [-2.0, -0.3, 0.4, 2.0], dtype=torch.float64, requires_grad=True
    )
    expected = value**exponent
    expected.sum().backward()
    actual = _execute_region(ctx.finish_region(), {"x": value.detach()})
    torch.testing.assert_close(actual["y"], expected)
    torch.testing.assert_close(actual["dx"], value.grad)


def test_stop_gradient_is_branch_local_and_survives_optimization():
    from cutilereduce.experimental.ir.passes import program_to_region
    from cutilereduce.experimental import TensorBuilder

    ctx = TensorBuilder(("d",))
    x = ctx.input("x", ("d",), "float32")
    detached = ctx.stop_gradient(ctx.stop_gradient(x))
    y = x + detached
    gradient = vjp(ctx, [(y, ctx.full(y.axes, 1.0))], wrt={"x": x})["x"]
    ctx.materialize("y", y)
    ctx.materialize("dx", gradient)
    region = program_to_region(ctx.finish())
    assert sum(node.op == "stop_gradient" for node in region.nodes) == 1
    actual = _execute_region(region, {"x": torch.arange(4, dtype=torch.float32)})
    torch.testing.assert_close(actual["dx"], torch.ones(4))


def test_nonsmooth_conventions_and_broadcast_vjp():
    ctx = TensorRegionBuilder(("b", "d"))
    x = ctx.expect("x", TensorSignature(("b", "d"), "float32"))
    bound = ctx.expect("bound", TensorSignature(("d",), "float32"))
    outputs = [
        ctx.minimum(x, bound),
        ctx.maximum(x, bound),
        abs(x),
        ctx.clamp(x, min=-1, max=1),
    ]
    for i, y in enumerate(outputs):
        gradients = vjp(ctx, [(y, ctx.full(y.axes, 1.0))], wrt={"x": x, "bound": bound})
        ctx.yield_value(f"dx{i}", gradients["x"])
        if "bound" in gradients:
            ctx.yield_value(f"db{i}", gradients["bound"])
    value = torch.tensor([[-1.0, 0, 1], [-2, 0, 2]])
    actual = _execute_region(ctx.finish_region(), {"x": value, "bound": torch.zeros(3)})
    torch.testing.assert_close(
        actual["dx0"], torch.tensor([[1.0, 0.5, 0], [1, 0.5, 0]])
    )
    torch.testing.assert_close(actual["dx1"], 1 - actual["dx0"])
    torch.testing.assert_close(actual["db0"], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(actual["dx2"], value.sign())
    torch.testing.assert_close(actual["dx3"], (value.abs() <= 1).float())


def test_integer_boolean_ops_and_cutile_source():
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "int64"))
    y = ctx.expect("y", TensorSignature(("d",), "int64"))
    ctx.yield_value("quotient", x // y)
    ctx.yield_value("remainder", x % y)
    ctx.yield_value("mask", ~(ctx.compare("lt", x, 0) & ctx.compare("gt", y, 0)))
    region = ctx.finish_region()
    values = {"x": torch.tensor([-7, 7, -7, 7]), "y": torch.tensor([3, -3, -3, 3])}
    actual = _execute_region(region, values)
    torch.testing.assert_close(actual["quotient"], values["x"] // values["y"])
    torch.testing.assert_close(actual["remainder"], values["x"] % values["y"])
    ast.parse(_emit_region("integer_ops", region, {"d": 4}, {}))


def test_validation_and_stable_backend_limit():
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float32"))
    with pytest.raises(TypeError):
        bool(x)
    with pytest.raises((TypeError, ValueError)):
        x**1.5
    with pytest.raises(ValueError):
        ctx.elementwise("unknown", x)
    with pytest.raises(ValueError):
        ctx.reduce("unknown", x, ("d",))
    with pytest.raises(ValueError):
        ctx.clamp(x, min=2, max=1)
    ctx.yield_value("y", ctx.log1p(x))
    with pytest.raises(NotImplementedError, match="stable lowering"):
        _emit_region("stable", ctx.finish_region(), {"d": 4}, {})


def test_native_elementwise_region():
    import hashlib
    from torch.utils.cpp_extension import load_inline
    from cutilereduce.experimental.backends.cpu.regions import _emit_region as emit_cpp

    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float64"))
    integer = ctx.expect("integer", TensorSignature(("d",), "int64"))
    for operation in (
        "sqrt",
        "rsqrt",
        "abs",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "log1p",
        "expm1",
        "softplus",
    ):
        value = getattr(ctx, operation)(x)
        ctx.yield_value(operation, value)
        ctx.yield_value(
            operation + "_vjp",
            vjp(ctx, [(value, ctx.full(value.axes, 1.0, value.dtype))], wrt={"x": x})[
                "x"
            ],
        )
    ctx.yield_value("power", x**-3)
    ctx.yield_value("clamp", ctx.clamp(x, min=0.2, max=1.0))
    ctx.yield_value("maximum", ctx.maximum(x, 0.5))
    ctx.yield_value("detach", ctx.stop_gradient(x))
    ctx.yield_value("quotient", integer // -3)
    ctx.yield_value("remainder", integer % -3)
    ctx.yield_value("boolean", ~ctx.compare("gt", integer, 0))
    region = ctx.finish_region()
    source = emit_cpp("elementary", region, {"d": 4})
    module = load_inline(
        "cutilereduce_elementary_" + hashlib.sha256(source.encode()).hexdigest()[:16],
        cpp_sources=source,
        functions=["elementary"],
        extra_cflags=["-O1"],
        with_cuda=False,
    )
    inputs = {
        "x": torch.tensor([0.1, 0.3, 0.9, 2.0], dtype=torch.float64),
        "integer": torch.tensor([-7, -2, 2, 7]),
    }
    expected = _execute_region(region, inputs)
    actual = module.elementary(*inputs.values())
    for result, reference in zip(actual, expected.values(), strict=True):
        torch.testing.assert_close(result, reference)


def test_teacher_tower_per_loss_gradients():
    import importlib.util
    from pathlib import Path
    from cutilereduce.experimental import compile_cpu

    spec = importlib.util.spec_from_file_location(
        "teacher_tower",
        Path(__file__).resolve().parents[1]
        / "examples"
        / "experimental_teacher_tower.py",
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    operator = example.make_teacher_tower()
    compiled = compile_cpu(operator, {"b": 2, "r": 5}, backward=True)
    torch.manual_seed(31)
    inputs = [torch.randn(2, 5, requires_grad=True) for _ in range(3)]
    target = torch.tensor([1, 3])
    outputs = compiled(*inputs, target)
    references = [
        torch.logsumexp(inputs[i], -1)
        - (inputs[i + 1].detach().softmax(-1) * inputs[i]).sum(-1)
        for i in range(2)
    ]
    references.append(
        torch.logsumexp(inputs[2], -1) - inputs[2].gather(1, target[:, None]).squeeze(1)
    )
    for i in range(3):
        torch.testing.assert_close(outputs[i], references[i])
        actual = torch.autograd.grad(
            outputs[i].sum(), inputs, retain_graph=True, allow_unused=True
        )
        expected = torch.autograd.grad(
            references[i].sum(), inputs, retain_graph=True, allow_unused=True
        )
        for value, reference, primal in zip(actual, expected, inputs, strict=True):
            torch.testing.assert_close(
                value if value is not None else torch.zeros_like(primal),
                reference if reference is not None else torch.zeros_like(primal),
            )
    torch.testing.assert_close(outputs[3], sum(references))


def test_stable_recipes_and_equal_infinities():
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float64"))
    y = ctx.expect("y", TensorSignature(("d",), "float64"))
    ctx.yield_value("softplus", ctx.softplus(x))
    ctx.yield_value("logaddexp", ctx.logaddexp(x, y))
    ctx.yield_value("log1p", ctx.log1p(x))
    ctx.yield_value("expm1", ctx.expm1(x))
    inputs = {
        "x": torch.tensor(
            [1e-15, 0.0, 1000.0, float("inf"), float("-inf")], dtype=torch.float64
        )
    }
    inputs["y"] = inputs["x"].clone()
    actual = _execute_region(ctx.finish_region(), inputs)
    for name in ("log1p", "expm1"):
        torch.testing.assert_close(
            actual[name], getattr(torch, name)(inputs["x"]), equal_nan=True
        )
    torch.testing.assert_close(
        actual["softplus"], torch.nn.functional.softplus(inputs["x"])
    )
    torch.testing.assert_close(
        actual["logaddexp"], torch.logaddexp(inputs["x"], inputs["y"])
    )


def test_discrete_cast_cuts_gradient():
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float32"))
    y = ctx.cast(ctx.cast(x, "int64"), "float32")
    assert vjp(ctx, [(y, ctx.full(y.axes, 1.0))], wrt={"x": x}) == {}


def test_unary_reorders_noncanonical_axes():
    from cutilereduce.experimental import TensorBuilder
    from cutilereduce.experimental.ir.passes import program_to_region

    ctx = TensorBuilder(("b", "d"))
    x = ctx.input("x", ("d", "b"), "float32")
    ctx.materialize("y", ctx.sin(x))
    region = program_to_region(ctx.finish())
    value = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    actual = _execute_region(region, {"x": value})
    torch.testing.assert_close(actual["y"], value.sin().T)


def test_cutile_elementary_emission_is_valid_python():
    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float32"))
    for operation in (
        "sqrt",
        "rsqrt",
        "abs",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "neg",
        "stop_gradient",
    ):
        ctx.yield_value(operation, getattr(ctx, operation)(x))
    ctx.yield_value("power", x**-5)
    ctx.yield_value("clamp", ctx.clamp(x, min=-1, max=1))
    ctx.yield_value("min", ctx.minimum(x, 0.0))
    ast.parse(_emit_region("elementary", ctx.finish_region(), {"d": 4}, {}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_generated_cutile_elementary_gpu(tmp_path):
    import importlib.util
    import cuda.tile as ct

    ctx = TensorRegionBuilder(("d",))
    x = ctx.expect("x", TensorSignature(("d",), "float32"))
    integer = ctx.expect("integer", TensorSignature(("d",), "int32"))
    ctx.yield_value("y", ctx.sigmoid(x) + ctx.sin(x) + ctx.clamp(x**3, min=-1, max=1))
    ctx.yield_value("q", integer // -3)
    ctx.yield_value("r", integer % -3)
    region = ctx.finish_region()
    source = (
        "import cuda.tile as ct\n"
        + _emit_region("elementary", region, {"d": 32}, {})
        + """
@ct.kernel
def kernel(x, integer, y, q, r):
    value = ct.load(x, (0,), (32,))
    index = ct.load(integer, (0,), (32,))
    outputs = elementary(value, index)
    ct.store(y, (0,), outputs[0])
    ct.store(q, (0,), outputs[1])
    ct.store(r, (0,), outputs[2])
"""
    )
    path = tmp_path / "generated_elementary.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("generated_elementary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = torch.linspace(-100, 100, 32, device="cuda")
    index = torch.arange(-16, 16, dtype=torch.int32, device="cuda")
    results = (
        torch.empty_like(value),
        torch.empty_like(index),
        torch.empty_like(index),
    )
    ct.launch(
        torch.cuda.current_stream(), (1,), module.kernel, (value, index, *results)
    )
    expected = _execute_region(region, {"x": value, "integer": index})
    for result, reference in zip(results, expected.values(), strict=True):
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
