# Phase 01 — Common core + sequential reference

**Prereq:** read `CODEBASE.md` §1–§4. Runs on any machine with `gcc` + `make`. No GPU needed.
**Time:** ~2 h. **This phase is the reference everything else is measured against.**

## Goal

A working sequential denoiser with the exact CLI and output format every later implementation
must match.

## Steps

- [x] 1. `src/common/image.h` / `image.c` — `Image{w,h,rgb}`, `image_read`, `image_write`
      (PPM P6, `#` comments skipped in the header), `image_alloc`, `image_free`,
      `static inline unsigned char *px(const Image*, int x, int y)`.
      Fail loudly on bad magic or truncated data.
- [x] 2. `src/common/noise.h` / `noise.c` — xorshift64* RNG (seeded, reproducible);
      `add_impulse_noise(img, density, seed, mode)` where mode ∈ {salt&pepper, random-valued}.
- [x] 3. `src/common/metrics.h` / `metrics.c` — `psnr(a,b)`, `mae(a,b)`, `max_diff(a,b)`,
      `identical(a,b)` on two `Image`s of equal size.
- [x] 4. `src/common/params.h` — `DenoiseParams{int window; double d; int m;}`, defaults
      `window=3, d=0.6, m=2`, plus a tiny `-d/-m/-n` flag parser reused by all binaries.
- [x] 5. `src/common/core.h` — the algorithm, no parallelism:
      `fuzzy_dist(unsigned char a[3], unsigned char b[3])` returning
      `1 - prod(1 - |Δc|/255.0)`; `is_clean(img,x,y,p)` counting peers within `d` (centre
      included) and testing `count >= m+1`; `repair_pixel(img,mask,x,y,p,dst)` averaging only
      clean neighbours, falling back to the per-channel median of the full window when zero
      clean neighbours exist. Out-of-bounds coordinates clamp to the edge.
- [x] 6. `src/seq/denoise_seq.c` — `denoise_seq_impl`: pass 1 fills `mask`, pass 2 builds the
      output image from `img + mask`. Written to a second buffer, then swapped.
- [x] 7. `src/app/main_denoise.c` — one `main()` used by both `denoise_seq` and `denoise_omp`.
      Takes `DENOISE_FN` and `IMPL_NAME` from compile definitions. Parses
      `in.ppm out.ppm [-d] [-m] [-n] [-r reps] [-q clean.ppm]`, times with
      `clock_gettime(CLOCK_MONOTONIC)`, best-of-`reps`, prints one line:
      `impl=seq ms=.. pixels=.. mpix_s=.. [psnr=.. mae=..]`.
- [x] 8. `src/app/main_gennoise.c` — `gennoise clean.ppm noisy.ppm density seed [sp|rv]`.
- [x] 9. `src/app/main_imgcmp.c` — `imgcmp a.ppm b.ppm` printing
      `identical=yes|no maxdiff=N mae=.. psnr=..`.
- [x] 10. `CMakeLists.txt` — C99, `-O2 -Wall -Wextra`, `find_package(OpenMP)` optional,
      `option(ENABLE_CUDA OFF)` guarding the `cuda/` subtree. Targets: `denoise_seq`,
      `denoise_omp` (stub until phase 02), `gennoise`, `imgcmp`, and a `make check` target.

## Verify

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/gennoise data/clean/photo_1024.ppm /tmp/noisy.ppm 0.10 42
./build/denoise_seq /tmp/noisy.ppm /tmp/seq.ppm -d 0.6 -m 2 -q data/clean/photo_1024.ppm
#  pass: prints psnr >= ~28  (noisy is ~14)
./build/gennoise data/clean/photo_1024.ppm /tmp/clean0.ppm 0.0 1
./build/denoise_seq /tmp/clean0.ppm /tmp/out0.ppm
./build/imgcmp /tmp/clean0.ppm /tmp/out0.ppm      # pass: identical=yes (PSNR > 50)
```

## Be able to explain

1. Why phase 2 needs the whole mask before it starts (dependency on neighbour mask).
2. Why the mean excludes corrupted neighbours, and what that does at an edge vs a flat region.
3. Why `fuzzy_dist` is written as `1 - prod(...)` and not `prod(...)`.

## Done when

`make check` passes and `denoise_seq` improves PSNR by ≥5 dB on a 10%-noise image.

## Results (phase 01, verified locally)

`make -C build check` → PASS. photo_1024, 10% salt & pepper, `-d 0.6 -m 2`:
noisy 13.80 dB → 25.20 dB (+11.4), 47.8 ms, 21.9 Mpix/s.
Clean photo_4096 through `denoise_seq`: `identical=yes`.
rv noise 5%: 19.04 → 31.84 dB.

Two notes where this plan's numbers did not hold with the defaults `d=0.6, m=2`:

- The "psnr >= ~28" line needs `-m 3` (30.6 dB at `d 0.6`; 31.2 dB at `d 0.5, m 3`).
  `m=2` only demands 2 peers, so textured areas keep their noise. `check.sh` therefore
  gates on the plan's "Done when" criterion (+5 dB) plus psnr >= 25.
- photo_1024 is not noise-free in its detailed region: 180 of 1 Mi pixels (0.017%) are
  flagged as corrupted at density 0, so a byte-identical pass-through cannot be asserted on
  it. `check.sh` asserts `identical=yes` on photo_4096 instead (0 pixels flagged).
