# Phase 05 — Benchmarks and scaling — results

Source: `bash scripts/bench.sh` on `ssh gpu` (neobox), 2026-11-11, from an otherwise
idle machine. 10% salt & pepper, seed 42, defaults `d=0.5 m=4 n=3` (phase-06 knee),
`-r 3` inside every process, 5 measured processes per cell, **median** reported, first
process discarded as warm-up. Machine: i7-13700H (6 P + 8 E = 14 cores / 20 threads),
RTX 4050 Laptop 70 W, OpenMP pinned with `OMP_PLACES=cores OMP_PROC_BIND=close`.

Raw data: `results/bench.csv`, `results/cuda_phases.csv`, full log `results/bench.log`.
Figures: `results/figures/speedup_vs_threads.svg`, `mpix_vs_size.svg`, `cuda_phases.svg`.

**Capture conditions matter more than anything else on this box.** The run below had the
CPU governor at `performance` and load 0.25. The same binaries, same script, under
`powersave` with a desktop session gave every CPU cell 2.1× the wall time (4096² seq
1064 ms vs 495 ms) while the GPU column barely moved. So: always name the governor next to
a CPU number, and never mix CPU ms from two sessions in one table. All ratios here are
within this one capture.

## 1. Whole table (ms, median of 5)

| impl | 512² | 1024² | 2048² | 4096² |
| --- | --- | --- | --- | --- |
| seq | 7.93 | 30.13 | 123.9 | 495.5 |
| omp 1 | 8.43 | 32.82 | 134.7 | 538.2 |
| omp 2 | 4.49 | 17.05 | 67.69 | 271.6 |
| omp 4 | 2.38 | 9.57 | 35.56 | 142.2 |
| omp 8 | 2.72 | 10.51 | 39.14 | 160.9 |
| omp 14 | 1.70 | 6.53 | 25.00 | 104.3 |
| omp 20 | 1.23 | 4.87 | 19.13 | 75.73 |
| **cuda** | **0.34** | **1.11** | **4.19** | **16.46** |
| cuda kernel only | 0.11 | 0.42 | 1.82 | 7.52 |

Throughput: `seq` sits at 33.0–34.8 Mpix/s at every size (linear in pixels — no cache
effect, the stencil does not reuse a working set). CUDA 773 → 1019 Mpix/s; `omp 20`
213 → 222 Mpix/s. The GPU is ~4.6× the best CPU configuration at 4096² and ~3.6× at 512².

Speedup vs 1-thread seq, and efficiency = speedup/threads (plan gate: efficiency at 4
threads > 0.75):

| | 512² | 1024² | 2048² | 4096² |
| --- | --- | --- | --- | --- |
| omp 2 | 1.77 (0.88) | 1.77 (0.88) | 1.83 (0.92) | 1.83 (0.91) |
| omp 4 | 3.34 (0.83) | 3.15 (0.79) | 3.49 (0.87) | 3.48 (0.87) |
| omp 8 | 2.91 (0.36) | 2.87 (0.36) | 3.17 (0.40) | 3.08 (0.39) |
| omp 14 | 4.68 (0.33) | 4.62 (0.33) | 4.96 (0.35) | 4.75 (0.34) |
| omp 20 | 6.44 (0.32) | 6.19 (0.31) | 6.48 (0.32) | 6.54 (0.33) |
| cuda | 23.4 | 27.1 | 29.6 | 30.1 |

CUDA rows use `threads=1` and are relative to one CPU thread, as the CSV header says;
`speedup/1` would be a meaningless efficiency, so the cell is empty in the CSV.

The 4-thread gate passes but is not comfortable: 0.79 at 1024². The 1.83× that omp 2
reaches here (against 1.88× when the governor was `powersave`) is the same wall — around
2× there is not much left to win on this part.

## 2. The OpenMP curve is non-monotonic — and the cause is measured, not guessed

Above 4 threads the curve is not the textbook "linear then plateau": **8 threads is
slower than 4 at every image size** (efficiency 0.36–0.40, versus 0.79–0.88 at 4), and
that dip reproduces in every capture and at every size, so it is a property of this
machine, not measurement noise. Past 8 the curve is monotonic again: each step from 8 to
14 to 20 adds 1.3–1.4×, and the efficiency settles at 0.31–0.35 — oversubscription is
buying real throughput here rather than losing to contention.

Isolating it with explicit `OMP_PLACES` at 1024² (captured on 2026-11-11 in the faster
CPU state — read the *ratios*, not the ms):

| threads | placed on | ms | note |
| --- | --- | --- | --- |
| 6 | the 6 P-cores | 7.33 | 5.4× — best per-core result |
| 12 | the 6 P-cores with SMT | 7.09 | **SMT buys nothing** (memory/pipeline bound) |
| 8 | the 8 E-cores | 20.13 | 2.0× — an E-core is ~2.9× slower here |
| 14 | `places=cores` (libgomp enumerates CPUs 0–13 = 12 P-threads + 2 E) | 10.84 | 6 E-cores sit idle, SMT contends |
| 14 | explicit 6 P + 8 E | 15.19 | static chunks make the E-cores set the wall time |
| 20 | `places=cores` (all 20 logical CPUs) | 4.60 | oversubscription rebalances the chunks |

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

Defensible claims from this table: efficiency is 0.79–0.92 at 2 and 4 threads (the
plan's gate passes), SMT adds nothing, and above 4 threads the numbers reflect core
placement on a hybrid part rather than the algorithm. A production fix would be
`schedule(dynamic)` or a chunk size proportional to core speed — noted, not done,
because the CPU is not the deliverable.

## 3. Weak scaling

`1 thread on 1024²` = 33.2 ms vs `4 threads on 2048²` = 36.3 ms → +9.1% drift for 4×
the work and 4× the threads. That is the plan's sanity check passing: per-thread work
stayed constant and the runtime did too.

## 4. CUDA phase split

| image | H2D | kernel | D2H | total | transfer share |
| --- | --- | --- | --- | --- | --- |
| 512² | 0.091 | 0.106 | 0.134 | 0.34 | 66% |
| 1024² | 0.315 | 0.422 | 0.362 | 1.11 | 61% |
| 2048² | 1.159 | 1.815 | 1.193 | 4.19 | 56% |
| 4096² | 4.431 | 7.516 | 4.485 | 16.46 | 54% |

This is the consequence of the float metric: the kernel got 15–20× faster, the PCIe copies
did not, so transfer went from ~8% of the call to ~54–66%. PCIe still holds ~11 GB/s each
way and still scales linearly with pixels, so the share is near-flat — but it is now the
dominant term, and it is where the next CUDA optimization belongs (pinned host memory,
fewer round trips, or keeping the image resident across repeats).

## 5. Be able to explain

1. **Why median, not mean.** Clock boost and thermal state on a 70 W laptop produce
   one-sided outliers (seq at 1024² wanders between 29 and 33 ms across repeats in this
   capture, 63–68 ms in the `powersave` one); a mean would let one throttled run move the
   result.
2. **Why the first process is discarded.** Page cache cold for the PPM, GPU clocks at
   210 MHz idle, and a fresh CUDA process pays ~130 ms of context creation that would
   otherwise be charged to the kernel. That is also why every binary runs with `-r 3`:
   the reported `ms` is the best of three calls inside one process.
3. **Where the serial fraction shows up.** The timed region is only the denoise call:
   PPM parse/write, noise generation and process start are outside it. Inside it the
   serial parts are the phase barrier and, for CUDA, **the PCIe copies — now 54% of the
   call** (it was 8% while the metric was `double`). Amdahl applies to the CUDA column
   directly: halving the kernel time again would buy at most 1.2× end to end, because
   the copies are the rest.
4. **What is reproducible and what is not.** Inside one capture, every cell repeats to
   within a few percent. **Across captures, the CPU column tracks the governor:** three
   captures of these same binaries gave 4096² seq 649 / 1064 / 495 ms — governor
   unrecorded / `powersave` / `performance`. The GPU is far steadier (4096² cuda 16.5 ms
   in both captures taken under `performance`). Lesson for the report: quote ratios, state
   the governor next to any CPU number, and never mix CPU ms from two sessions.

## 6. Reproduce

```bash
ssh gpu && cd ~/Projects/hpc-denoise
tmux new -s bench
bash scripts/bench.sh . build 2>&1 | tee /tmp/bench.log
python3 scripts/plot.py .          # results/figures/*.svg
```
