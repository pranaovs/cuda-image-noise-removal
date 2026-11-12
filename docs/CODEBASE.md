# CODEBASE — end-to-end walkthrough

Read this top to bottom once. It is written so that after reading you can explain the whole
project at a whiteboard without looking at code.

**Status:** implemented and verified (phases 01–06). `denoise_seq`, `denoise_omp`,
`denoise_cuda`, `gennoise`, `imgcmp`, `fbcount` all build; `make check` passes; the
numbers live in `results/04`–`results/06`. Two deviations from this document, both
measured rather than argued: there is **no shared-memory CUDA variant** (it changed the
kernel time by +0.2%, so it was deleted — see `results/04-cuda-variants.md` §4), and the
shipped quality defaults are **d = 0.5, m = 4**, not 0.6/2 (phase-06 sweep).

---

## 0. The one-sentence version

> We remove salt-and-pepper noise from colour images with a two-phase, pixel-local filter;
> because every pixel is independent we parallelize it with OpenMP over CPU threads and with
> CUDA over GPU threads, prove all versions produce identical pixels, then measure speedup
> and image quality.

If you can say that, plus section 1 and section 5, you can pass the explanation.

---

## 1. The problem and the algorithm

### 1.1 What "impulsive noise" means

Salt-and-pepper: each pixel is, with probability `p`, replaced by either pure black or pure
white. Real value lost, neighbour values untouched. This is why it can be *repaired* rather
than merely blurred: the neighbours still carry the information.

Two things we measure:

- **Speed** — milliseconds, and Mpixels/second.
- **Quality** — PSNR and MAE of the result versus the clean original.

### 1.2 The two phases

Notation: image is a 2D grid of pixels, each pixel is an RGB triple. `W` is the 3×3 window
centred on the pixel (9 pixels, the pixel itself included).

**Phase 1 — detection.** Ask: *is this pixel noise?*

For the centre pixel `x_i`, compute the fuzzy distance to each of the 8 neighbours:

```
d_f(x_i, x_j) = 1 − ∏_{c ∈ {R,G,B}} ( 1 − |x_i,c − x_j,c| / 255 )
```

- identical pixels → product is 1 → `d_f = 0`
- maximally different → product is 0 → `d_f = 1`

Build the *peer group*: the neighbours whose `d_f ≤ d` (tolerance `d`, e.g. 0.6). Count them
(the centre always counts itself). If the count is at least `m + 1` (cardinality threshold,
e.g. `m = 2`), the pixel is **clean**. Otherwise it is **corrupted**.

Intuition: a real pixel in a flat area has many similar neighbours. A salt-and-pepper pixel
has none — it is an outlier surrounded by plausible values.

Output of phase 1: a **mask**, one byte per pixel, `1 = corrupted`.

**Phase 2 — filtering.** Ask: *what should this pixel have been?*

Only for corrupted pixels: replace with the arithmetic mean of the **uncorrupted** neighbours
inside `W`. Clean pixels pass through untouched. If every neighbour is corrupted (possible at
high noise density) fall back to the median of all 9 window pixels.

Intuition: a mean over *all* neighbours would drag edges toward grey — that is just a blur.
A mean over *plausible* neighbours keeps edge structure, because the corrupted ones are
excluded from the average. That exclusion is the entire reason this filter beats a box blur.

### 1.3 Why there are exactly two phases, not one

Phase 2 reads the mask of its 8 neighbours. So the mask must be complete for the whole image
before any pixel is filtered.

If you fused the phases, thread A could filter pixel `(5,5)` using the mask of `(5,6)` which
thread B has not written yet → **race condition**, and the output would depend on thread
timing. The two-phase split with one global barrier is what makes the parallel version
deterministic and identical to the sequential version.

This single fact is the most important sentence in the project. Expect it as a question.

---

## 2. Why this parallelizes (and what limits it)

### 2.1 Structured grid + stencil

- The data is a regular 2D array — a **structured grid**.
- Each output element reads a fixed neighbourhood — a **stencil** (3×3 here).
- Each output element is written by exactly one thread, and never read by another during the
  same phase.

Consequence: work items are independent inside a phase, so both OpenMP and CUDA can assign
one pixel (or one row) to one thread with zero synchronization inside the phase.

### 2.2 The only synchronization: one barrier between phases

- OpenMP: the implicit barrier at the end of the `parallel for` in phase 1.
- CUDA: two separate kernel launches — a kernel launch is itself a global barrier.

### 2.3 What actually limits us: memory traffic

Per pixel, per phase, we read 9 RGB triples and write 1. So ~27 bytes read for ~3 bytes
written. Arithmetic intensity is low: a handful of flops per byte. That means:

- **CPU**: performance plateaus once thread count exceeds what the memory system can feed.
  Near-linear speedup to the physical core count, then flat. This is a *result*, not a bug —
  report it with the bandwidth argument.
- **GPU**: the same reads hit the L1/L2 cache, but a naive kernel issues 9 separate loads per
  thread to global memory. Shared-memory tiling fixes this by loading the tile once per block.
  That table was the plan for the GPU half of the report; the measurement disagreed with
  the expectation: tiling the reads away changed the kernel by +0.2%, because the kernel
  is bound by the double-precision fuzzy distance, not by traffic (27 double terms per
  pixel against 27 bytes). `results/04-cuda-variants.md` §4 has the three probes that
  establish this. The shipped CUDA code is therefore the simple one-thread-per-pixel
  version, and the F32/F64 story is the GPU finding.

### 2.4 GPU-specific issue: warp divergence

Phase 1's decision is a branch per pixel. Within a warp (32 threads), if some pixels are
corrupted and some are not, both branches may execute. That is a real, discussable cost. It
is also why the phase-2 kernel is cheap: clean pixels exit immediately.

---

## 3. Repository map

```
docs/PLAN.md                umbrella plan (already written)
docs/CODEBASE.md            this file
docs/plans/01..06-*.md      one file per phase, execute in a new session each
docs/results/*.md           the write-up per phase; the raw numbers stay in results/*.csv
docs/GPU.md                 GPU box access, hardware, gotchas
CMakeLists.txt              build; CUDA behind a flag so a GPU-less box still builds

src/common/image.h .c       PPM P6 read/write. Image{w,h,rgb} — the ONLY data structure.
src/common/noise.h .c       xorshift RNG + salt&pepper / random-valued impulse
src/common/core.h           the algorithm itself: fuzzy distance, detect, filter. Shared by
                            seq and omp so they cannot drift apart.
src/common/metrics.h .c     PSNR, MAE, max per-channel difference
src/common/params.h         DenoiseParams{window, d, m} + CLI parsing

src/app/main_denoise.c      one main() compiled three times (seq, omp, cuda) — identical CLI, identical
                            timing code, so the two numbers are comparable
src/app/main_gennoise.c     clean.ppm -> noisy.ppm
src/app/main_imgcmp.c       compare two PPMs: psnr / mae / maxdiff / byte-identical?

src/seq/denoise_seq.c       phase1 loop + phase2 loop, plain C
src/omp/denoise_omp.c       the same two loops + OpenMP pragmas

src/cuda/kernels.cuh        C-linkable declarations: denoise_cuda() + CudaTiming (no CUDA types)
src/cuda/kernels.cu         detectKernel, filterKernel — one thread per pixel, global loads
src/cuda/host.cu            alloc, transfer, launch, time. Exposes one function to C.
src/cuda/CMakeLists.txt     denoise_cuda behind -DENABLE_CUDA=ON, output in build/

data/clean/*.ppm            photo_512 / 1024 / 2048 / 4096 (already downloaded)
scripts/bench.sh            sweep matrix -> results/*.csv
results/*.csv               committed, so plots are reproducible without re-running
```

**Why one `main_denoise.c` for two binaries:** if seq and omp had separate mains, a difference
in argument handling or timing placement could fake a speedup. Sharing the main removes that
whole class of doubt. The *only* difference between `denoise_seq` and `denoise_omp` is the
function linked in.

---

## 4. Data flow (one run, end to end)

```
clean.ppm ──gennoise──> noisy.ppm           (+ we keep clean.ppm as ground truth)
                            │
                            ├──denoise_seq──>  out_seq.ppm   prints: impl=seq ms=.. mpix_s=..
                            ├──denoise_omp──>  out_omp.ppm
                            └──denoise_cuda─>  out_cuda.ppm
                                                    │
        clean.ppm ──imgcmp── out_*.ppm ─────────────┘
                            prints: psnr=.. mae=.. maxdiff=.. identical=yes/no
```

So: generate once, denoise with each implementation, compare each result to the clean original
*and* to each other.

### 4.1 The contract every denoiser binary obeys

```
denoise_X  in.ppm  out.ppm  [-d 0.5] [-m 4] [-n 3] [-r reps] [-q clean.ppm]
```
Prints exactly one machine-readable line:
```
impl=omp threads=8 ms=12.345 pixels=1048576 mpix_s=84.96 psnr=31.42
```
`bench.sh` greps that line. Identical format from all implementations = the CSV builds itself.

### 4.2 The core function signatures (this is the whole API)

```c
/* src/common/core.h — the algorithm, no parallelism in here */
double fuzzy_dist(const unsigned char *a, const unsigned char *b);
int    is_clean(const Image *img, int x, int y, const DenoiseParams *p);
/* fills dst[0..3] with the repaired value for a corrupted pixel */
void   repair_pixel(const Image *img, const unsigned char *mask,
                    int x, int y, const DenoiseParams *p, unsigned char *dst);

/* src/seq/denoise_seq.c and src/omp/denoise_omp.c — identical except for pragmas */
void denoise_seq_impl(Image *img, unsigned char *mask, const DenoiseParams *p);
void denoise_omp_impl(Image *img, unsigned char *mask, const DenoiseParams *p);
```

`core.h` holds the *policy*; the impl files hold only the *loop and the pragma*. Because the
pixel maths is shared, seq/omp/CUDA cannot disagree about semantics — only about scheduling.

---

## 5. The three things your professor will ask

**Q1 — "Where is the parallelism?"**
Inside each phase, one pixel per thread; no writes shared between threads; one barrier between
phases. Phase 2 cannot be fused into phase 1 because it reads its neighbours' mask.

**Q2 — "What limits your speedup?"**
Measured answer, and it is not the one this section used to predict. On the CPU, speedup
flattens because of core placement on the hybrid part, not bandwidth (6 P-cores alone give
5.4×, 8 E-cores alone 2.0×, SMT adds nothing — `results/05` §2). On the GPU, removing the 9×
read redundancy with a shared-memory tile changed the kernel by +0.2%, because the kernel is
bound by the double-precision fuzzy distance: 27 double terms per pixel against 27 bytes.
Cite the probes in `results/04` §4 (double 116 ms / float 4.8 ms / integer 0.8 ms).

**Q3 — "How do you know it's correct and how do you know it's fast?"**
Correctness is *not* eyeballing: `denoise_omp` must be byte-identical to `denoise_seq`
(`cmp`), CUDA must match to max |Δchannel| == 0 (integer arithmetic everywhere), a clean image
must pass through essentially untouched (PSNR > 50 dB), and the noisy→denoised PSNR must improve by ≥5 dB.
Speed is 5 repeats, median reported, warm-up run discarded, thread pinning fixed via
`OMP_PROC_BIND`, and kernel time reported separately from PCIe transfer time.

Bonus question you should have an answer for: **"why integer arithmetic?"**
So that every implementation produces bit-identical bytes. Floating-point summation order in
the mean filter is identical anyway (fixed 3×3 order) — the point is no `fast-math`, no
reassociation, no cross-thread reduction.

---

## 6. Phase map — which file each phase creates

| Phase | Creates | Proves |
|---|---|---|
| 01 sequential core | `common/*`, `app/main_denoise.c`, `app/main_gennoise.c`, `app/main_imgcmp.c`, `seq/*`, `CMakeLists.txt` | the algorithm works at all |
| 02 OpenMP | `omp/denoise_omp.c` | parallel == serial, byte for byte |
| 03 CUDA baseline | `cuda/kernels.cu`, `host.cu`, `cuda/CMakeLists.txt` (no `main_cuda.cu`: the shared `app/main_denoise.c` is reused) | GPU == CPU byte for byte, and 4.4× (1024²) / 5.2× (4096²) over one CPU thread |
| 04 CUDA optimize | a shared-memory variant, measured then **deleted** (`results/04`) | tiling does *not* pay: the kernel is FP64-bound |
| 05 benchmarks | `scripts/bench.sh`, `results/*.csv` | scaling curves, Mpixels/s |
| 06 quality | `scripts/quality.sh` | PSNR/MAE vs noise density and vs `d`,`m` |

Run them in that order, one session each. Phase 01 must be verified before 02 starts, because
01 *is* the reference that 02 and 03 are measured against.

---

## 7. Vocabulary you will be graded on using correctly

- **Structured grid** — regular 2D array, neighbours are index-arithmetic away.
- **Stencil** — an output element computed from a fixed spatial neighbourhood.
- **Embarrassingly / data parallel** — same operation on independent elements.
- **Peer group** — the subset of the window that agrees with the centre pixel within tolerance.
- **Fuzzy metric** — distance that returns a degree of similarity in [0,1], not a binary verdict.
- **Barrier** — all threads wait; the phase boundary.
- **Race condition / dependency** — two work items touching the same location in an order-
  dependent way (here: the mask, if phases were fused).
- **Arithmetic intensity** — flops per byte moved; low here, so bandwidth-bound.
- **Amdahl's law** — max speedup limited by the serial fraction (I/O, memcpy, noise gen).
- **Warp divergence** — a branch where threads in a warp take different paths; both are paid for.
- **Occupancy** — how many warps a streaming multiprocessor can keep resident; limited by
  registers and shared memory per block.
