.PHONY: lint test quality typecheck bench_cpu

lint:
	PYTHONPATH=. uv run ruff check ./src ./tests ./examples

test:
	PYTHONPATH=. uv run pytest

typecheck:
	PYTHONPATH=. uv run ty check ./src

quality: lint test typecheck

# CPU attention: eager + SDPA + independently tuned forward/embed/finalize.
BENCH_CPU_LENGTH ?= 4096
BENCH_CPU_DIMENSION ?= 32
BENCH_CPU_THREADS ?= 1 2 4
BENCH_CPU_ITERATIONS ?= 100
BENCH_CPU_SHORTLIST ?= 10
BENCH_CPU_TUNE_ITERATIONS ?= 100
BENCH_CPU_JSON ?= results/bench_cpu.json
BENCH_CPU_ARGS ?=

bench_cpu:
	PYTHONPATH=. uv run -m examples.experimental_cpu_tiles --length $(BENCH_CPU_LENGTH) --dimension $(BENCH_CPU_DIMENSION) --threads $(BENCH_CPU_THREADS) --iterations $(BENCH_CPU_ITERATIONS) --tune-shortlist $(BENCH_CPU_SHORTLIST) --tune-iterations $(BENCH_CPU_TUNE_ITERATIONS) --json "$(BENCH_CPU_JSON)" --only full $(BENCH_CPU_ARGS)
