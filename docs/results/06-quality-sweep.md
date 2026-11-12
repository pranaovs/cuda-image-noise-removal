# Phase 06 — Quality sweep — results

Source: `bash scripts/quality.sh` on `ssh gpu`, 2026-11-11. photo_1024, salt & pepper,
seed 42. Raw data: `results/quality_params.csv`, `quality_density.csv`,
`quality_impls.csv`, `quality_fallback.csv`. Figures: `results/figures/psnr_vs_d.svg`,
`psnr_vs_density.svg`, `edge_crop.png`, `impl_strip.png`.

## 1. Parameter sweep — where the knee is (10% noise, PSNR dB / MAE)

| d \ m | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 18.47 | 25.19 | 29.34 | 29.92 | 30.13 | 30.50 | 30.58 |
| 0.4 | 18.49 | 25.56 | 30.98 | 31.50 | 30.88 | 30.54 | 30.33 |
| **0.5** | 18.46 | 25.48 | 31.15 | **32.26** | 31.61 | 30.93 | 30.09 |
| 0.6 | 18.47 | 25.47 | 31.16 | 32.16 | 32.00 | 31.37 | 30.26 |
| 0.7 | 18.31 | 24.67 | 29.27 | 30.78 | 31.14 | 31.03 | 30.28 |
| 0.8 | 18.44 | 23.80 | 27.29 | 28.55 | 28.74 | 28.66 | 28.19 |
| 0.9 | 17.82 | 22.39 | 24.61 | 25.43 | 25.43 | 25.30 | 25.24 |

Peak: **d = 0.5, m = 4 → 32.26 dB, MAE 0.91**, and it is an interior maximum in both
directions (d = 0.4 → 31.50, d = 0.6 → 32.16, m = 3 → 31.15, m = 5 → 31.61). These are
now the defaults in `src/common/params.h`; the whole grid is in the CSV.

Reading the grid: `d` too small (< 0.4) does not hurt much because it only makes the
detector stricter, but `m = 1` and `m = 2` are clearly too permissive — at `d = 0.9,
m = 1` the filter *damages* the image (17.8 dB, worse than the 13.8 dB noisy input is
only 4 dB of gain). High `d` with low `m` means "almost everything counts as a peer",
so corrupted pixels pass detection and stay in the output. That is the failure mode
worth quoting.

## 2. Noise-density sweep (d = 0.5, m = 4)

| density | noisy PSNR | denoised PSNR | gain | MAE |
| --- | --- | --- | --- | --- |
| 0.01 | 23.78 | 35.66 | +11.88 | 0.28 |
| 0.02 | 20.79 | 35.27 | +14.48 | 0.34 |
| 0.05 | 16.76 | 34.00 | +17.24 | 0.54 |
| 0.10 | 13.80 | 32.26 | **+18.46** | 0.91 |
| 0.15 | 12.04 | 30.53 | +18.49 | 1.33 |
| 0.20 | 10.80 | 28.50 | +17.70 | 1.87 |
| 0.25 | 9.81 | 25.72 | +15.91 | 2.74 |
| 0.30 | 9.02 | 23.16 | +14.14 | 4.00 |

Gain peaks around 0.10–0.15 (+18.5 dB) and degrades gracefully; the plan's gate
(≥ +5 dB at 0.10) passes by a wide margin. Note the *noisy* baseline is low even at 1%
(23.8 dB): salt & pepper replaces a pixel with 0 or 255 on all three channels, so each
hit is worth ~255² of squared error. A 1% density is a 20 dB image.

## 3. Implementation independence — the table is one repeated number

Same noisy input (10% density, d = 0.5, m = 4), PSNR to 4 decimals:

| impl | PSNR | MAE | byte-identical to seq |
| --- | --- | --- | --- |
| seq | 32.2613 | 0.9100 | — |
| omp (8 threads) | 32.2613 | 0.9100 | yes |
| cuda | 32.2613 | 0.9100 | yes |

One sentence for the report: parallelism is free here, because the output is not
"similar", it is the same bytes — one thread per pixel, one writer per pixel, no
reduction and no atomics, and the only cross-pixel interaction is the phase barrier.

## 4. Median-fallback frequency (the failure mode at high density)

`fbcount` counts corrupted pixels whose whole window is corrupted, i.e. the pixels
where the AMF has nothing to average and `core.h` falls back to the window median.

| density | corrupted px | fallback px | share of corrupted |
| --- | --- | --- | --- |
| 0.10 | 71 301 | 8 | 0.01% |
| 0.20 | 141 199 | 157 | 0.11% |
| 0.30 | 216 480 | 1 482 | 0.68% |
| 0.40 | 294 974 | 7 671 | 2.60% |
| 0.50 | 367 839 | 21 525 | 5.85% |

So the fallback is a real but rare path: 1 in 10 000 corrupted pixels at the shipped
10% operating point, 1 in 17 at 50% density. That justifies keeping it (dropping it
would leave holes) and justifies it not being optimised (it costs nothing at the
densities this filter is meant for).

## 5. Visual evidence

- `results/figures/edge_crop.png` — 128×128 crop, nearest-neighbour ×4, noisy | denoised.
  The crop location is chosen by the script (highest gradient energy in the image), not
  by hand, so the figure is reproducible.
- `results/figures/impl_strip.png` — the same crop through noisy | seq | omp | cuda. All
  three outputs are identical, which is the point.
- What to look for: the edges of the branch survive; a plain box blur of the same radius
  would have dissolved them. That is the "mean of the clean neighbours only" switch
  working, not the mean itself.

## 6. Be able to explain

1. **Why PSNR is not the whole story.** PSNR is MSE-based, so it rewards blurring: a
   slightly blurred image can score better than a sharp, correct one. That is why MAE is
   reported next to it and why the edge crop is the actual evidence of quality.
2. **What `d` and `m` do.** `d` is the fuzzy tolerance that defines a peer (smaller =
   stricter). `m` is how many peers are needed before the pixel is declared legitimate
   (larger = stricter detection). They are not interchangeable: raising `m` with a low `d`
   flags nearly everything; raising `d` with a low `m` flags nothing.
3. **Why parallel quality must equal sequential quality.** Because the output is
   byte-identical (verified with `cmp` on 512²…4096², windows 3/5/15, and the awkward
   1001×333 size), not because the algorithm was written similarly. A single differing
   byte would have shown up as a different PSNR in §3.
4. **Why the clean-image case is not exactly a no-op.** On a clean photo_4096 at
   `d = 0.5, m = 4` a few extreme-edge pixels fall below `m+1` peers and get replaced:
   PSNR 96.7 dB, i.e. a handful of pixels changing by at most 10/255. `denoise_seq`, the
   reference, does exactly the same — it is the detector's edge case, and `make check`
   asserts PSNR > 50 dB rather than byte identity for that reason.

## 7. Reproduce

```bash
ssh gpu && cd ~/Projects/hpc-denoise
bash scripts/quality.sh . build      # the four CSVs
python3 scripts/plot.py .            # psnr_vs_d.svg, psnr_vs_density.svg
```
