# Phase 06 — Quality sweep

**Prereq:** phase 01 (enough to run), best after 05. **Time:** ~1.5 h. CPU-only machine is fine.

## Goal

The reconstruction-quality half of the project: how good is the filter, and does parallelizing
it change the result? (Answer to the second: no, and you prove it.)

## Steps

- [x] 1. `scripts/quality.sh` — for `d ∈ {0.2,0.4,0.5,0.6,0.7,0.8,0.9}` and
      `m ∈ {1,2,3,4}` on 10% noise, print PSNR/MAE of `denoise_seq` vs clean →
      `results/quality_params.csv`. Find the knee; set those values as the defaults in
      `params.h` and note it.
- [x] 2. Noise-density sweep: `density ∈ {0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30}` with the
      chosen `d`,`m` → `results/quality_density.csv`, recording PSNR of the *noisy* input and
      of the *denoised* output (the gap is the filter's contribution).
- [x] 3. Implementation-independence table: same noisy input through seq / omp / cuda-v1..v3,
      one row each, PSNR to 4 decimals → must be identical. One table, one sentence: same
      pixels, so same quality; parallelism is free here.
- [x] 4. Visual figure: noisy / seq / omp / cuda crops side by side (nearest-neighbour upscale,
      no smoothing) → `results/figures/`. Also a 128×128 crop of a sharp edge — it is the
      clearest evidence the *switch* (clean-neighbour-only mean) preserves edges.
- [x] 5. Record failure mode: at density ≥0.30 report how often the window had zero clean
      neighbours (the median fallback path). That number justifies the fallback.

## Verify

```bash
bash scripts/quality.sh 2>&1 | tee results/quality.log
# pass: denoised PSNR - noisy PSNR >= 5 dB at density 0.10
#       identical PSNR across all implementations on the same input
#       PSNR peaks somewhere inside the d range (not at an endpoint -- if it does, widen the range)
```

## Be able to explain

1. Why PSNR is not the whole story (it rewards blurring) and why MAE and a visual crop
   accompany it.
2. What `d` controls (tolerance: lower = stricter peer group = more pixels flagged corrupted)
   and what `m` controls (how many peers are needed to declare the pixel legitimate).
3. Why the quality of a parallel implementation must equal the sequential one — because the
   output is bit-identical, not merely "similar".

## Done when

The three CSVs exist, PSNR improvement ≥5 dB at 10% noise, and the cross-implementation PSNR
table is a single value repeated.
