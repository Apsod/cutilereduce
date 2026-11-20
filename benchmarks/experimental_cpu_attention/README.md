# CPU attention tile experiment

Measured on 2026-11-19 with an Intel Core i7-1165G7, PyTorch 2.10.0+cu128,
FP32, one head, `G=1`, and `Dqk=Dv=32`. The reference is eager
`bmm -> softmax -> bmm`, with autograd for training. Both implementations use
the same `torch.set_num_threads` setting, inputs, and random output cotangent.
Compilation is excluded; timing order is randomized after warmup. Every candidate
is checked for output and input-gradient agreement at each measured thread count.
Final confirmation runs had no concurrent test compilation.

## Confirmed result at L=R=2048

Query tile 128, full fold tile 2048, tree forward. Backward loops over query
tiles, accumulating key/value gradients locally with direct commits.

| Threads | Pass | Experimental ms | Eager PyTorch ms | Speedup |
|---:|---|---:|---:|---:|
| 1 | Forward | 9.60 | 14.82 | 1.54x |
| 1 | Forward + backward | 32.99 | 42.13 | 1.28x |
| 4 | Forward | 4.62 | 6.67 | 1.44x |
| 4 | Forward + backward | 16.61 | 21.94 | 1.32x |

The full single-tile training schedule took 55.53 ms / 31.87 ms with one / four
threads in the same confirmation run. Tiling is therefore necessary for the
training win in this experiment. At 512 tokens eager PyTorch remained faster:
the best tested forward schedule used a full tile, and the best training schedule
used full forward with backward query tiles of 128.

The initial broader sweep tried query tiles 32/64/128/256, split fold tiles,
online folds, independent backward tiles, and an atomic query-partitioned
backward. Query tiles of 128 were best among the tested larger-case schedules;
32/64 tiles and atomic accumulation were generally less favorable in the small
case. The four-thread training gain was more variable in the initial sweep
(near parity) than in the uncontended confirmation. This is a measured shortlist,
not an exhaustive search or a claim about all attention shapes.

The largest interaction matrix drops from 16 MiB to 1 MiB when queries are tiled
at 128 for the 2048-token problem. This reduces pressure on the measured 1.25 MiB
L2 / 12 MiB L3 caches; kernel-call overhead and GEMM shapes also matter. The sweep
does not separately attribute time or measure peak memory.

The experiment exposed and fixed selected boolean SIMD mask representation,
missing OpenMP compile/link flags, singleton parallel-region guards suppressing
inner parallelism, and worker-local grad mode. Generated C++ now explicitly
protects the custom differentiation path from ATen autograd inside worker threads.

Raw confirmation reports, including candidate parameters, quartiles, hardware
properties and commit modes: [512.json](512.json), [2048.json](2048.json).

## Reproduce

From the repository root:

```console
uv run -m examples.experimental_cpu_tiles --length 512 --only full query_128 full_forward_backward_query_128 --threads 1 4 --iterations 300 --json /tmp/cpu-tiles-512.json
uv run -m examples.experimental_cpu_tiles --length 2048 --only full query_128 --threads 1 4 --iterations 120 --json /tmp/cpu-tiles-2048.json
```

Omit `--only` to run the broader candidate sweep. `--dimension`, `--warmup`, and
`--threads` allow other workloads and execution settings. The native extensions
are cached; first-time compilation can take several minutes for a broad sweep.

## Independent analytical stage search

`--analytical-shortlist` now selects from independent forward/backward tile
grids before compiling or timing. At these shapes the complete coarse training
grid contains 7,500 candidates with one thread and 15,000 with four threads;
these runs use a 16,384-candidate cap and are not truncated. Backward loop axes
and loop-program counts are searched alongside tiles. The CPU cost model uses
explicit detected cache sizes and generic rate priors; timings do not fit it.
Ordered prefix/checkpoint schedules still share their checkpoint chunk layout.

Raw independent-search reports include the model, selected schedules, validation,
and randomized timing results: [independent-512.json](independent-512.json) and
[independent-2048.json](independent-2048.json).
At 512 tokens the model selected full forward and backward tiles for both thread
counts; eager remained faster in this run. This differs from the older 512-token
manual experiment, where query-tiled backward was best. The model remains a
heuristic, and the wider search alone does not guarantee a measured improvement.

```console
uv run -m examples.experimental_cpu_tiles --length 2048 --only query_128 --threads 1 4 --iterations 80 --analytical-shortlist 1 --max-candidates 16384 --json /tmp/cpu-independent-2048.json
```

At 2048 tokens both thread counts selected tree forward with query tiles 128
and full fold 2048, but backward with full query 2048, fold tiles 128 and a fold
loop. This swaps which interaction dimension is tiled in backward. The comparison
below uses each thread count's selected candidate and the previous coupled
query-128 schedule measured in the same randomized run (80 rounds):

| Threads | Independent training ms | Coupled training ms | Eager training ms | Speedup over eager |
|---:|---:|---:|---:|---:|
| 1 | 40.32 | 41.19 | 51.45 | 1.28x |
| 4 | 14.61 | 15.53 | 21.19 | 1.45x |

The independent schedule was about 2% / 6% faster than coupled tiling in this run.
Both selected entries represent the same schedule, measured separately in the
randomized order; their differences give some indication of timing variability.
These are complete forward-plus-backward measurements, not isolated backward
latencies. Outputs and random-cotangent input gradients passed at both thread
counts. Compilation and analytical search were completed before timing.
