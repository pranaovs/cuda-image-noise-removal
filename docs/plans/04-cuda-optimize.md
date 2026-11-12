# Phase 04 — CUDA optimization variants

**Prereq:** phase 03 gives `maxdiff=0`. **Time:** ~3 h. GPU laptop.

## Status: done, with one deviation

**Steps 1-3 and 6 below were executed on a throwaway branch and then deleted** — no
`enum KernelVariant`, no `-v` flag and no `results/cuda_variants.csv` exist in the shipped
tree. Only the naive pair ships; the numbers are in `../results/04-cuda-variants.md`.

Measured: the shared-memory tile (`v3`) changed the kernel by +0.2% at 4096², because the
kernel is bound by the double-precision fuzzy distance, not by global-memory traffic.
`__ldg`/`__restrict__` are rejected by `../cuda-course-scope.md` and were never shipped, and
the tile was deleted after the measurement — the shipped CUDA code is the simple
one-thread-per-pixel pair. Evidence: `../results/04-cuda-variants.md`.

## Goal

Three kernels that compute the same pixels, differing only in how they read memory. That
comparison is the GPU analysis section of the report.

- **v1 naive** — plain `uchar3` global loads (done in phase 03).
- **v2 read-only cache** — `__restrict__` pointers + `__ldg()` for every neighbour read.
- **v3 shared-memory tile** — each block stages an 18×18 tile (16×16 plus a 1-pixel halo) in
  shared memory, `__syncthreads()`, then every neighbour read is a shared-memory hit.

## Steps

- [x] 1. Parametrize: `enum KernelVariant {V1_NAIVE, V2_LDG, V3_SHARED}` and a `-v 1|2|3` flag;
      `kernel_variant` also printed in the output line.
- [x] 2. v2: `const uchar3 * __restrict__ in` + `__ldg(&in[idx])` in the neighbour loop.
- [x] 3. v3 tile load:
      ```cuda
      __shared__ uchar3 tile[18][18];           // 972 bytes — tiny
      // each thread loads its own pixel plus the right/bottom halo edge
      // clamp coordinates when loading (edge replicate), then __syncthreads()
      ```
      Both detect and filter kernels get their own tile load; the tile load code is the same.
- [x] 4. Block-size experiment: build once each at 8×8, 16×16, 32×32 (compile-time or a
      template parameter) and record occupancy notes. Do not guess — measure.
- [x] 5. Correctness: all three variants must still give `maxdiff=0` vs seq.
- [x] 6. Store the results table — recorded as a markdown table in `../results/04-cuda-variants.md`
      §1–§4 (the CSV was never committed; that file is the evidence).

## Verify

Nothing to verify here: the variant code was deleted after measurement (see Status). The
correctness gate it used lives on in `scripts/check.sh`, and the "expected shape" below was
wrong — v3 measured **+0.2%**, not a win, because the kernel is FP64-bound.

## Be able to explain

1. Why a `uchar3` tile can bank-conflict, and what you changed.
2. What occupancy costs the shared-memory variant (bytes per block, registers per thread).
3. Why warp divergence from the corrupted/clean branch is unavoidable but cheap in phase 2
   (clean pixels return immediately).

## Done when

All variants passed `maxdiff=0` and a timing table with block-size sensitivity was recorded
(16×16 shipped; see `results/04` §1 and §4).
