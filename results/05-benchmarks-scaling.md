# Phase 05 — Benchmarks and scaling — results

Source: `bash scripts/bench.sh` on `ssh gpu` (neobox), 2026-11-11, from an otherwise
idle machine. 10% salt & pepper, seed 42, defaults `d=0.5 m=4 n=3` (phase-06 knee),
`-r 3` inside every process, 5 measured processes per cell, **median** reported, first
process discarded as warm-up. Machine: i7-13700H (6 P + 8 E = 14 cores / 20 threads),
RTX 4050 Laptop 70 W, OpenMP pinned with `OMP_PLACES=cores OMP_PROC_BIND=close`.

Raw data: `results/bench.csv`, `results/cuda_phases.csv`, full log `results/bench.log`.
Figures: `results/figures/speedup_vs_threads.svg`, `mpix_vs_size.svg`, `cuda_phases.svg`.

## 1. Whole table (ms, median of 5)

| impl | 512² | 1024² | 2048² | 4096² |
| --- | --- | --- | --- | --- |
| seq | 10.13 | 39.67 | 162.1 | 649.2 |
| omp 1 | 10.58 | 41.36 | 169.1 | 677.0 |
| omp 2 | 5.38 | 21.37 | 84.94 | 340.9 |
| omp 4 | 2.68 | 10.98 | 42.87 | 171.9 |
| omp 8 | 2.88 | 11.86 | 46.06 | 184.6 |
| omp 14 | 2.65 | 13.60 | 38.32 | 106.5 |
| omp 20 | 2.61 | 5.65 | 18.95 | 76.13 |
| **cuda** | **2.18** | **9.08** | **30.87** | **124.3** |
| cuda kernel only | 1.95 | 8.34 | 28.08 | 114.7 |

Throughput: `seq` sits at 25.8–26.4 Mpix/s at every size (perfectly linear in pixels —
no cache effect, the stencil does not reuse a working set). CUDA 120 → 135 Mpix/s;
`omp 20` 100 → 220 Mpix/s.

Speedup vs 1-thread seq, and efficiency = speedup/threads (plan gate: efficiency at 4
threads > 0.75):

| | 512² | 1024² | 2048² | 4096² |
| --- | --- | --- | --- | --- |
| omp 2 | 1.88 (0.94) | 1.86 (0.93) | 1.91 (0.95) | 1.90 (0.95) |
| omp 4 | 3.78 (0.95) | 3.61 (0.90) | 3.78 (0.95) | 3.78 (0.94) |
| omp 8 | 3.52 (0.44) | 3.35 (0.42) | 3.52 (0.44) | 3.52 (0.44) |
| omp 14 | 3.82 (0.27) | 2.92 (0.21) | 4.23 (0.30) | 6.10 (0.44) |
| omp 20 | 3.89 (0.19) | 7.02 (0.35) | 8.55 (0.43) | 8.53 (0.43) |
| cuda | 4.65 | 4.37 | 5.25 | 5.22 |

CUDA rows use `threads=1` and are relative to one CPU thread, as the CSV header says.

## 2. The OpenMP curve is non-monotonic — and the cause is measured, not guessed

Above 4 threads the curve is not the textbook "linear then plateau": 8 threads is
*slower* than 4, and 20 threads is 2.4× faster than 14 at 1024². It reproduces across
runs (14 threads: 9.5/13.3/13.9 ms; 20 threads: 4.6/4.6/4.7 ms), so it is a property of
this machine, not measurement noise. Isolating it with explicit `OMP_PLACES` at 1024²:

| threads | placed on | ms | note |
| --- | --- | --- | --- |
| 6 | the 6 P-cores | 7.33 | 5.4× — best per-core result |
| 12 | the 6 P-cores with SMT | 7.09 | **SMT buys nothing** (memory/pipeline bound) |
| 8 | the 8 E-cores | 20.13 | 2.0× — an E-core is ~2.9× slower here |
| 14 | `places=cores` (libgomp enumerates CPUs 0–13 = 12 P-threads + 2 E) | 10.84 | 6 E-cores sit idle, SMT contends |
| 14 | explicit 6 P + 8 E | 15.19 | static chunks make the E-cores set the wall time |
| 20 | `places=cores` (all 20 logical CPUs) | 4.60 | 8.6× — oversubscription rebalances the chunks |

So the "plateau at 14 physical cores" the plan predicted is not what this part does.
Two effects, both textbook-shaped but easy to get wrong:

1. **`OMP_PLACES=cores` is not "one place per physical core" on a hybrid CPU.** It
   enumerates in Linux CPU order, so 14 places = the 12 P-core siblings plus 2 E-cores,
   and six efficiency cores are never used. The 14-thread number is therefore not "14
   cores", it is "6 P-cores with SMT plus 2 E-cores".
2. **`schedule(static)` on heterogeneous cores sets the runtime to the slowest chunk.**
   Splitting the image into 14 equal chunks hands the same work to a 2.9×-slower core;
   with 20 chunks each thread has less to do, so the E-core chunks shrink and the wall
   time drops. That is why oversubscribing wins.

Defensible claims from this table: efficiency is 0.90–0.95 at 2 and 4 threads (the
plan's gate passes), SMT adds nothing, and above 4 threads the numbers reflect core
placement on a hybrid part rather than the algorithm. A production fix would be
`schedule(dynamic)` or a chunk size proportional to core speed — noted, not done,
because the CPU is not the deliverable.

## 3. Weak scaling

`1 thread on 1024²` = 41.4 ms vs `4 threads on 2048²` = 42.9 ms → +3.6% drift for 4×
the work and 4× the threads. That is the plan's sanity check passing: per-thread work
stayed constant and the runtime did too.

## 4. CUDA phase split

| image | H2D | kernel | D2H | total | transfer share |
| --- | --- | --- | --- | --- | --- |
| 512² | 0.095 | 1.946 | 0.133 | 2.18 | 10.5% |
| 1024² | 0.320 | 8.343 | 0.389 | 9.08 | 7.8% |
| 2048² | 1.181 | 28.076 | 1.604 | 30.87 | 9.0% |
| 4096² | 4.610 | 114.654 | 5.055 | 124.3 | 7.8% |

PCIe runs ~11 GB/s each way and scales linearly with pixels, so its share is flat at
~8–10%. Consequence for the report: at these sizes quoting end-to-end ms is fair, but at
~256² the fixed transfer cost would dominate and the GPU would lose to the CPU.

## 5. Be able to explain

1. **Why median, not mean.** Clock boost and thermal state on a 70 W laptop produce
   one-sided outliers (seq at 1024² wanders between 39 and 41 ms across repeats); a mean
   would let one throttled run move the result.
2. **Why the first process is discarded.** Page cache cold for the PPM, GPU clocks at
   210 MHz idle, and a fresh CUDA process pays ~130 ms of context creation that would
   otherwise be charged to the kernel. That is also why every binary runs with `-r 3`:
   the reported `ms` is the best of three calls inside one process.
3. **Where the serial fraction shows up.** The timed region is only the denoise call:
   PPM parse/write, noise generation and process start are outside it. Inside it the
   serial parts are the noise-free I/O-free phase barrier and, for CUDA, 8% PCIe. With
   no serial fraction to speak of, speedup tracks thread count until the hardware runs
   out — which is what the 4-thread efficiency of 0.90–0.95 says.
4. **Why the CSV can be trusted as reproducible.** The sweep was run twice (before and
   after the phase-06 defaults changed). Every median agreed within 2% — 4096² seq 637.4
   vs 649.2 ms, 4096² omp 20 76.1 vs 76.1 ms — except the 8/14-thread OpenMP cells
   discussed in §2, whose spread is a placement effect rather than drift.

## 6. Reproduce

```bash
ssh gpu && cd ~/Projects/hpc-denoise
tmux new -s bench
bash scripts/bench.sh . build 2>&1 | tee /tmp/bench.log
python3 scripts/plot.py .          # results/figures/*.svg
```
