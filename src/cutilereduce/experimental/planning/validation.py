"""Reference validation shared by execution backends."""

def validate_callable(executable, inputs, reference, *, backward=False, rtol=1e-4, atol=1e-5):
    import torch

    actual_inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
    reference_inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
    actual = executable(*actual_inputs)
    expected = reference(*reference_inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected):
        raise AssertionError("reference output arity mismatch")
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, check_dtype=False)
    if backward:
        lhs = tuple(x for x in actual_inputs if x.requires_grad)
        rhs = tuple(x for x in reference_inputs if x.requires_grad)
        if not lhs:
            raise ValueError("backward validation needs differentiable inputs")
        cotangents = tuple(torch.randn(x.shape, device=x.device, dtype=x.dtype,
            generator=torch.Generator(device=x.device).manual_seed(17 + i))
            for i, x in enumerate(actual))
        ag = torch.autograd.grad(actual, lhs, cotangents, allow_unused=True)
        bg = torch.autograd.grad(expected, rhs, tuple(g.to(x.dtype) for g, x in zip(cotangents, expected, strict=True)), allow_unused=True)
        for x, a, b in zip(lhs, ag, bg, strict=True):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,
                                       torch.zeros_like(x) if b is None else b,
                                       rtol=rtol, atol=atol, check_dtype=False)

