# Phase 02 — OpenMP

**Prereq:** phase 01 verified. **Time:** ~1.5 h. Any multicore laptop, no GPU.

## Goal

Add CPU threads and prove the parallel output is *byte-identical* to sequential.

## Steps

- [x] 1. `src/omp/denoise_omp.c` — copy of the phase-01 loop body with
      `#pragma omp parallel for collapse(2) schedule(static)` on both the detection loop and
      the filtering loop. No `reduction`, no atomics, no critical sections — if you need one,
      the decomposition is wrong.
- [x] 2. Keep the two loops in separate `parallel for` regions (or one region with an explicit
      `#pragma omp barrier`), so the barrier between phases is visible in the code, not implied.
- [x] 3. `main_denoise.c` also prints `threads=%d` from `omp_get_max_threads()`.
- [x] 4. Wire target `denoise_omp` in CMake with `OpenMP::OpenMP_C`.

## Verify

```bash
OMP_NUM_THREADS=8 ./build/denoise_omp /tmp/noisy.ppm /tmp/omp.ppm -d 0.6 -m 2
cmp /tmp/seq.ppm /tmp/omp.ppm && echo BYTE_IDENTICAL
```

Then the sweep (pinning matters, otherwise numbers are noise):

```bash
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build/denoise_omp /tmp/noisy_4096.ppm /tmp/o.ppm   # gennoise output, not in data/
done
```

Expect: speedup near-linear to physical cores, then a plateau. Note where it flattens and
relate it to ~9 loads per pixel (bandwidth-bound).

## Be able to explain

1. Why `cmp` passing is a stronger claim than "PSNR is the same".
2. Where the implicit barrier is, and why removing it would corrupt the image.
3. Why `schedule(static)` is right here (uniform work per row, no load imbalance) — and when
   `dynamic` would be better (it would not; say why not).

## Done when

`cmp` says identical at 1, 2, 4, 8, 16 threads, and you have a speedup table.
