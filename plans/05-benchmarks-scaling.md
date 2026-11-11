# Phase 05 — Benchmark harness and scaling

**Prereq:** phases 01–04 all pass correctness. **Time:** ~2 h.

## Goal

One script that produces the CSV every plot and table in the report comes from. Run it inside
`tmux`, never in a bare SSH shell.

## Steps

- [x] 1. `scripts/bench.sh` — configurable via env vars, defaults:
      `IMPLS="seq omp1 omp2 omp4 omp8 omp16 cuda-v1 cuda-v2 cuda-v3"`,
      `IMAGES="photo_1024 photo_4096"`, `DENSITY=0.10`, `REPS=5`.
      For each (impl, threads, image, density): run `REPS` times, take the **median** `ms`,
      keep min/max spread, discard the first run as warm-up, append one CSV row.
- [x] 2. Pin threads: `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
      Without pinning, 4-thread runs vary by >20% and the scaling curve is fiction.
- [x] 3. CSV header exactly:
      `impl,threads,width,height,density,ms,mpix_per_s,speedup,efficiency`
      `speedup = ms_seq(1 thread) / ms`, `efficiency = speedup / threads`
      (`threads=1` and `efficiency=1` for the CUDA rows; note in the file header that CUDA
      speedup is relative to 1 CPU thread).
- [x] 4. Noise-once discipline: generate the noisy image **once per (image, density)** and
      reuse it for every implementation, so all implementations see identical input.
- [x] 5. `scripts/plot.py` (stdlib only, or gnuplot): speedup vs threads, Mpixels/s vs image
      size, kernel-time-vs-transfer-time for CUDA, GPU variants bar chart → `results/*.svg`.
- [x] 6. Re-run the whole thing twice and diff the CSVs. If the curves differ in shape, the
      machine is noisy (thermal / other load) — note it and increase `REPS`.

## Verify

```bash
tmux new -s bench
bash scripts/bench.sh 2>&1 | tee results/bench.log
# pass: results/bench.csv has one row per config, medians present, efficiency at 4 threads > 0.75
```

Also produce the weak-scaling sanity check: 1 thread on 1024² vs 4 threads on 2048² should
stay roughly constant in ms.

## Be able to explain

1. Why median not mean (outliers from clock boost and cache misses).
2. Why the first run is discarded (page cache cold, GPU clocks ramping).
3. Where Amdahl's serial fraction shows up in your own numbers (I/O + noise gen + PCIe).

## Done when

`results/bench.csv` exists, is reproducible, and the shape of the curves is explainable.
