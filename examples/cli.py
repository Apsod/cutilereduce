"""Shared CLI options for the runnable examples."""

import argparse
from cutilereduce.util.spec import SPECMAP


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_float(value):
    value = float(value)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


class ExampleParser(argparse.ArgumentParser):
    def __init__(self, spec, defaults, *, aliases=None, benchmark_seconds=1.0):
        super().__init__(description="Tune, validate and benchmark a fold example.")
        self.axis_names = tuple(axis.name for axis in spec.axes)
        if set(defaults) != set(self.axis_names):
            raise ValueError("provide one default size for every spec axis")
        aliases = aliases or {}
        if aliases.keys() - set(self.axis_names):
            raise ValueError("aliases must name spec axes")
        for name in self.axis_names:
            flags = [f"--{name}"]
            if name in aliases and aliases[name] != name:
                flags.append(f"--{aliases[name]}")
            self.add_argument(*flags, dest=name, type=positive_int,
                              default=positive_int(defaults[name]))
        self.add_argument("--candidates", type=positive_int, default=20,
                          help="configurations retained per admissible loop axis")
        self.add_argument("--timeout", type=nonnegative_float, default=0)
        self.add_argument("--validate", action="store_true")
        self.add_argument("--accuracy-matrix", action="store_true")
        self.add_argument("--seed", type=int, default=0)
        self.add_argument("--quiet-tuning", action="store_true")
        self.add_argument("--forward-only", action="store_true")
        self.add_argument("--torch-compile", action="store_true")
        self.add_argument("--benchmark-seconds", type=nonnegative_float,
                          default=benchmark_seconds)
        self.add_argument("--benchmark-memory", action="store_true")
        self.add_argument("--load-plan", metavar="PATH")
        self.add_argument("--save-plan", metavar="PATH")
        self.add_argument("--hardware", choices=list(spec.SPECMAP))

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.torch_compile and parsed.forward_only:
            self.error("--torch-compile currently requires the autograd plan")
        return parsed

    def sizes(self, args):
        return {name: getattr(args, name) for name in self.axis_names}




def make_inputs(spec, sizes, *, device="cuda", initializers=None):
    return spec.mk_inputs(sizes, device=device, **(initializers or {}))


def validate(function, inputs, reference, *, spec, accuracy_matrix=False,
             backward=True, reference_dtypes=None):
    """Validate a built callable using gradient names from its spec."""
    import torch
    from cutilereduce.util.runner import validate_precision_matrix

    if reference_dtypes is None:
        reference_dtypes = {
            "PyTorch BF16": torch.bfloat16,
            "PyTorch FP32": torch.float32,
        }
        if accuracy_matrix:
            reference_dtypes["PyTorch FP64"] = torch.float64
    return validate_precision_matrix(
        function, reference, inputs,
        input_names=tuple(b.id.name for b in spec.input if b.req_grad),
        reference_dtypes=reference_dtypes,
        backward=backward, pairwise=accuracy_matrix,
    )


def benchmark_full(name, inputs, implementations, *, min_run_time=1.0,
                   measure_memory=False, backward=True):
    """Benchmark named built callables, including any reference implementations."""
    import torch
    from cutilereduce.util.runner import benchmark_implementations, benchmark_memory

    timings = memory = None
    if min_run_time > 0:
        timings = benchmark_implementations(
            name, inputs, implementations, min_run_time=min_run_time,
            backward=backward, output_grad_dtype=torch.bfloat16,
        )
    if measure_memory:
        memory = benchmark_memory(
            name, inputs, implementations,
            backward=backward, output_grad_dtype=torch.bfloat16,
        )
    return timings, memory


def run_example(name, spec, sizes, function, *, args, reference, references,
                initializers=None, device="cuda"):
    """Run shared input construction, validation and benchmarks after building.

    ``function`` is the built callable selected by the CLI, eager or compiled.
    ``reference`` is the precision-validation callable; ``references`` maps
    benchmark labels to callables (which may use different reference dtypes).
    """
    import torch

    backward = not args.forward_only
    should_validate = args.validate or args.accuracy_matrix
    should_benchmark = args.benchmark_seconds > 0 or args.benchmark_memory
    if not (should_validate or should_benchmark):
        return
    torch.manual_seed(args.seed)
    inputs = make_inputs(spec, sizes, device=device, initializers=initializers)
    if should_validate:
        validate(function, inputs, reference, spec=spec,
                 accuracy_matrix=args.accuracy_matrix, backward=backward)
    if should_benchmark:
        label = "CuTile torch.compile" if args.torch_compile else "CuTile eager"
        implementations = {label: function, **references}
        benchmark_full(
            name, inputs, implementations,
            min_run_time=args.benchmark_seconds,
            measure_memory=args.benchmark_memory, backward=backward,
        )


def main(name, spec, functions, defaults, *, reference, references,
         initializers=None, aliases=None, benchmark_seconds=1.0, argv=None):
    """Run an example from its kernel definition and example-specific settings."""
    import torch
    from cutilereduce.fold import FoldOperator
    from cutilereduce.util.runner import print_plan
    from cutilereduce.util.spec import rtx5080

    parser = ExampleParser(spec, defaults, aliases=aliases,
                           benchmark_seconds=benchmark_seconds)
    args = parser.parse_args(argv)
    sizes = parser.sizes(args)
    hardware = SPECMAP[args.hardware]
    torch.manual_seed(args.seed)
    operator = FoldOperator(spec, functions)
    plan = (
        operator.load_plan(args.load_plan, sizes)
        if args.load_plan
        else operator.tune(
            sizes, args.candidates, args.timeout, hardware=hardware,
            quiet=args.quiet_tuning, backward=not args.forward_only,
        )
    )
    if args.save_plan:
        operator.save_plan(plan, args.save_plan, metadata={"hardware": args.hardware})
    print_plan(plan)
    function = operator.build(
        plan, backward=not args.forward_only, torch_compile=args.torch_compile,
    )
    run_example(
        name, spec, sizes, function, args=args, reference=reference,
        references=references, initializers=initializers,
    )
