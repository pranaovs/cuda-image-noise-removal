# HPC Project — Parallel Image Denoising with OpenMP and CUDA

Build the paper's algorithm four ways (sequential, OpenMP, CUDA, CUDA-optimized), prove they
agree, then measure speedup and image quality. Code only for this phase.

> **Read `CODEBASE.md` first** — it is the end-to-end explanation of the codebase (what every
> file does, why, and what your professor will ask). This file is the umbrella plan.
>
> **Execution is split into phases, one session each:**
>
> | Phase | Plan file | Machine |
> |---|---|---|
> | 01 common core + sequential | `plans/01-sequential-core.md` | any (gcc) |
> | 02 OpenMP | `plans/02-openmp.md` | any multicore |
> | 03 CUDA baseline | `plans/03-cuda-baseline.md` | GPU laptop + nvcc |
> | 04 CUDA optimizations | `plans/04-cuda-optimize.md` | GPU laptop + nvcc |
> | 05 benchmarks + scaling | `plans/05-benchmarks-scaling.md` | GPU laptop (both) |
> | 06 quality sweep | `plans/06-quality-sweep.md` | CPU is enough |
>
> Phase 01 is the reference; phases 02 and 03 are verified against it.

---

## 1. Context

**Base paper:** M. G. Sánchez, V. Vidal, J. Bataller, *"A Parallel Method for Impulsive Image
Noise Removal on Hybrid CPU/GPU Systems"*, Procedia Computer Science 18 (2013) 2488–2497.
DOI: 10.1016/j.procs.2013.05.429

**Goal restated:** filter salt-and-pepper (impulsive) noise out of colour images using a
pixel-local two-phase algorithm; parallelize it over CPU threads (OpenMP) and GPU threads
(CUDA); report speedup, throughput (Mpixels/s), and reconstruction quality (PSNR/MAE).

### 1.1 The algorithm — understand this before writing any code

The image is a **structured 2D grid** of pixels. For each pixel we look at its 3×3
neighbourhood `W` (9 pixels). Two phases:

**Phase 1 — Detection (is this pixel corrupted?)**

For centre pixel `x_i`, build the *peer group* — the set of neighbours that look like `x_i`:

```
P(x_i, d) = { x_j ∈ W  :  d_f(x_i, x_j) ≤ d }
```

Using the fuzzy metric on RGB vectors (from the paper's cited work):

```
d_f(x_i, x_j) = ∏ over channels c∈{R,G,B}  ( 1 − |x_i,c − x_j,c| / 255 )
```

Inverted vs. Euclidean: identical colours → `d_f = 1`; maximally different → `d_f = 0`.
So "close" means `d_f ≥ d` in this form, and the paper's `≤ d` variant uses `1 − ...`
instead. **Note this in the report** — get the direction right when you implement
(`dist = 1 − ∏(...)` gives a true distance in [0,1], which is unambiguous; prefer that).

Decision rule:

```
if |P(x_i, d)| ≥ m + 1   → x_i is CORRUPTED-free (keep it)
else                     → x_i is CORRUPTED    (must be replaced)
```

Two tunable parameters: `d` (fuzzy tolerance, ~0.4–0.9) and `m` (cardinality, ~1–3).
The paper uses a 3×3 window (`n = 3`).

**Phase 2 — Filtering (repair corrupted pixels only)**

Arithmetic mean filter, computed **only over the uncorrupted neighbours** in `W`
(this is the "switch" that preserves edges — a plain mean would blur everything):

```
if x_i corrupted:  x_i_new = mean{ x_j ∈ W : x_j not corrupted }
else:              x_i_new = x_i
```

This is why the paper splits into two passes: phase 2 reads the corruption mask of its
neighbours, so the whole mask must be known first. **One global barrier between phases.**
That barrier is the single most important structural fact in the project.

**Why it parallelizes perfectly (the HPC point):**
- Each pixel is an independent work item, reading 9 values, writing 1 → no race conditions.
- Reads overlap → 9× redundant memory traffic when naive. Fixing that is the optimization story.
- Irregular control flow (detect → branch) makes warp divergence a real GPU issue to discuss.

---

## 2. Approach

One C/C++ repository. Four implementations sharing all infrastructure:

| Build | What it is | Purpose |
|---|---|---|
| `denoise_seq` | plain serial loop | correctness reference + speedup baseline |
| `denoise_omp` | `#pragma omp parallel for collapse(2)` | strong/weak CPU scaling |
| `denoise_cuda` | naive 1-thread-per-pixel kernels | baseline GPU |
| `denoise_cuda` | shipped: the same naive kernels (tiling measured, +0.2%, deleted) | GPU speedup + the FP64-bound argument (`results/04`) |

Everything else (image I/O, noise generation, metrics, CLI, timing) is written **once** and
shared, so all four numbers are directly comparable.

**Deliberately not doing:** no image library (PPM P6 is a 20-line parser), no OpenCV, no
externally-added dependencies, no hybrid CPU+GPU split (that's the paper's own extension, and
it's a stretch goal we name in the report as future work).

---

## 3. Dev workflow — coding on your friend's GPU machine over SSH

You develop *directly on the remote box* via SSH + tmux, not "edit locally, copy over" (that
creates version hell). Your local 16-core machine is optional for OpenMP smoke tests.

### 3.1 Get access and verify the toolchain (Step 0, ~10 min)

Ask your friend for: hostname/IP, username, and whether SSH keys are set up.

```bash
ssh you@friend-host                 # get in (add ~/.ssh/config alias if you like)
tmux new -s hpc                     # ALWAYS work in tmux: survives disconnects

nvidia-smi                          # GPU model + driver + CUDA version
nvcc --version                      # if missing: module load cuda  (or ask friend)
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA'   # core count = your OpenMP sweep max
free -g                             # RAM (image sizes must fit)
gcc --version
sinfo / squeue                     # only if it's a SLURM cluster, not a workstation
```

Write those five outputs down — they go in the report's "Experimental setup" section.

```bash
# persistent SSH alias so you never retype it (~/.ssh/config on YOUR machine)
Host gpu
    HostName friend-host
    User you
    ServerAliveInterval 60
```

Then `ssh gpu` and `rsync`/`scp` become short. `ServerAliveInterval` stops the connection
dying mid-benchmark.

### 3.2 Keep the code in git, pull remotely (recommended)

```bash
# create repo locally (this dir), push to GitHub/GitLab private repo
git init && git add -A && git commit -m "skeleton"
git remote add origin git@github.com:you/hpc-denoise.git && git push -u origin main

# on the GPU box
git clone git@github.com:you/hpc-denoise.git ~/hpc-denoise
```

Loop: edit here or in a remote editor → `git push` → `ssh gpu` → `git pull && make`.

**If the cluster has no outbound internet** (common): use your local machine as a relay:

```bash
rsync -avz --exclude build --exclude .git ./ gpu:~/hpc-denoise/
```

**VS Code Remote-SSH** is the comfortable option: install the extension, connect `ssh gpu`,
edit and use its integrated terminal there. Everything runs on the GPU box.

### 3.3 Discipline rules

1. Never run long benchmarks in a bare SSH shell — `tmux` or `nohup ... > log.txt &`.
2. Never trust a single timing run. The harness repeats and takes the median (Step 5).
3. Pin thread counts explicitly: `OMP_NUM_THREADS=8`, `OMP_PLACES=cores`,
   `OMP_PROC_BIND=close` — otherwise results are noise.
4. Keep benchmark CSVs in the repo (`results/`) so plots are reproducible offline.

---

## 4. Repository layout

```
CMakeLists.txt
src/common/image.h  image.c        # PPM P6 read/write; Image{int w,h; unsigned char*rgb}
src/common/noise.h  noise.c        # salt&pepper + random-value impulse, seedable (xorshift)
src/common/metrics.h metrics.c     # PSNR, MAE, max-abs-diff
src/common/params.h                # window size, d, m (+ defaults from the phase-06 sweep)
src/tools/fallback_count.c         # fbcount: how often the median fallback fires
src/app/main_denoise.c            # ONE main(), compiled once per impl via -DDENOISE_FN
src/seq/denoise_seq.c             # phase1 loop + phase2 loop, plain C
src/omp/denoise_omp.c             # the same two loops + OpenMP pragmas
src/cuda/host.cu  kernels.cu  kernels.cuh  CMakeLists.txt   # no main_cuda.cu: the
                                                    shared app/main_denoise.c is reused
data/clean/*.ppm                   # Kodak/Lena + a couple of large synthetic images
scripts/bench.sh                   # timing sweep  -> results/bench.csv
scripts/quality.sh                 # quality sweep -> results/quality_*.csv
scripts/plot.py                    # CSV -> results/figures/*.svg (stdlib only)
scripts/crops.py                   # PPM crops -> results/figures/*.png (stdlib only)
results/                           # CSVs committed, keep small
```

CMake shape: OpenMP always on; CUDA behind `-DENABLE_CUDA=ON`, which is the only way
`src/cuda/CMakeLists.txt` (and its `enable_language(CUDA)`) is ever parsed — so the box
without `nvcc` still builds `seq`+`omp`.

---

## 5. Steps

Each step ends with something you can run and see. Do not start the next until the current
one is verified — that is what "not one-shotting" means here.

### Step 1 — Image I/O + noise + sequential reference (~2 h)

- [x] PPM P6 read/write (`image.c`), plus a raw dump for diffing.
- [x] `add_impulse_noise(img, density, seed)` — salt&pepper; store the *clean* original too.
- [x] `denoise_seq`: detection loop → write mask → filter loop. Bit-exact, single-threaded.
- [x] `metrics.c`: PSNR, MAE vs the clean image.

**Checkpoint you should be able to explain:** why phase 2 cannot be fused into phase 1, and
what "only uncorrupted neighbours" does to an edge vs. a flat region.

### Step 2 — OpenMP version, prove correctness (~1.5 h)

- [x] Same loops + `#pragma omp parallel for collapse(2) schedule(static)`.
- [x] `cmp out_seq.ppm out_omp.ppm` → must be byte-identical (integer math, no reductions).
      Verified at 1/2/4/8/16 threads in `scripts/check.sh`.
- [x] Thread sweep on the GPU box's CPU (`OMP_NUM_THREADS`, `OMP_PROC_BIND`).
      1024²: 1→64.9, 8→18.7, 14→13.8 ms. Full sweep + curves are phase 05.

Expected shape: near-linear to physical cores, then a plateau (memory-bandwidth bound:
9 reads/pixel). If it plateaus early, that *is* a result — report it with the arithmetic
intensity argument.

### Step 3 — CUDA baseline (~3 h)

- [x] `kernels.cu`: `detectKernel` and `filterKernel`, 16×16 blocks, `grid = ceil(dim/16)`.
- [x] Two kernel launches, one `cudaDeviceSynchronize` between them = the paper's barrier.
- [x] `cudaMalloc`/`cudaMemcpy` transfer timing **reported separately** from kernel timing
      (otherwise your speedup numbers are meaningless).
- [x] Correctness check vs `seq` with max-abs-diff (expect 0 — all integer).
      Evidence: `results/03-cuda-baseline.txt` (maxdiff=0 at 1024², 4096², window 5,
      1001×333 non-multiple-of-16, clean input; identical across 5 runs).

### Step 4 — CUDA optimizations (~3 h)

> **Scope override:** `cuda-course-scope.md` (CS3015) rejects `__ldg`, `__restrict__`,
> kernel templates, vector types and dynamic shared memory, so v2-as-planned is out and
> the block-size sweep is fixed at 16×16. The shared-memory tile was written, verified and
> measured, then **deleted** (+0.2%, i.e. nothing) — shipped is the single naive kernel pair.
> Evidence and analysis in `results/04-cuda-variants.md`.

- [x] Variant A: ~~`__restrict__` + `__ldg`~~ never shipped — the scope sheet rejects both.
- [x] Variant B: a shared-memory tile (1-pixel halo, clamped load, `__syncthreads()`,
      three byte planes + mask plane) was written, verified `maxdiff=0` and measured.
      It gave **+0.4% at 1024² and +0.2% at 4096²**, so it was deleted: the shipped code
      is the simple one-thread-per-pixel kernel pair.
- [x] Phase timing moved to `cudaEvent`s, transfers still reported separately.
- [x] Evidence and the "why tiling cannot help" analysis: `results/04-cuda-variants.md`.

Result: the kernel is bound by the double-precision fuzzy distance, not by memory
traffic — detection-only in `double` costs 116 ms at 4096², the same kernel in `float`
4.8 ms, integer-only 0.8 ms. A 70 W laptop part runs FP64 at 1/64 of FP32 rate, so the
GPU loses to 20 OpenMP threads at 4096² (124 vs 76 ms). Single precision is the next
lever, and it is exactly what the paper used.

Discuss: warp divergence from the corrupted/clean branch, why the mask must be global memory
between kernels, occupancy limiters (registers vs shared memory per block).

### Step 5 — Benchmark harness (~2 h) — done, see `results/05-benchmarks-scaling.md`

- [x] `bench.sh` drives: impl ∈ {seq, omp(1..20), cuda} ×
      image size ∈ {512², 1024², 2048², 4096²} × noise density ∈ {0.05, 0.1, 0.2}.
- [x] 5 repeats per config, report **median** and a min/max spread. First image is warm-up,
      discarded (page cache, JIT, clock boost).
- [x] CSV columns: `impl,threads,width,height,density,ms,mpix_per_s,speedup,efficiency`.
- [x] Compute: speedup = T_seq/T_par, efficiency = speedup/threads, throughput = pixels/ms.

### Step 6 — Quality sweep (~1.5 h) — done, see `results/06-quality-sweep.md`

- [x] PSNR/MAE vs noise density (0.01 → 0.30).
- [x] PSNR/MAE vs `d` and `m` — find the knee, report the values you ship as defaults.
- [x] Prove quality is implementation-independent: identical PSNR for seq/omp/cuda on the
      same noisy input (they produce the same pixels, so this is a one-line table).

### Step 7 — Report (later, per your call — not in this phase)

Scaffold only; you said you'll handle it. Numbers from `results/` will already be plot-ready.

---

## 6. Verification

| What | How | Pass condition |
|---|---|---|
| OpenMP == sequential | `cmp out_seq.ppm out_omp.ppm` | identical bytes |
| CUDA == sequential | `check_diff` mode in the binaries | max \|Δchannel\| = 0 (≤1 tolerated) |
| No-op on clean image | run with density 0 | output ≈ input, PSNR > 50 dB |
| Denoising works | PSNR noisy vs denoised | PSNR improves by ≥ 5 dB at density 0.1 |
| Scaling sane | efficiency at 4 threads | > 0.75 (else something is wrong) |
| Determinism | run CUDA variant 5× | identical output, timing spread < 10% |

Also add a tiny `make check` target running the two `cmp`s and the clean-image case, so
regressions surface in one command instead of a benchmark run.

---

## 7. Time budget (total ~13 h hands-on, plus rate-limit factor)

| Step | Estimate |
|---|---|
| 0. SSH + toolchain verified | 10 min |
| 1. Sequential + I/O + metrics | ~2 h |
| 2. OpenMP + correctness | ~1.5 h |
| 3. CUDA baseline | ~3 h (first CUDA build trips compiler/arch flags) |
| 4. CUDA optimizations | ~3 h |
| 5–6. Benchmarks + quality sweep | ~3.5 h |

Add 50% if this is your first CUDA project. Step 3 is where projects stall — expect
`-arch=sm_XX` mismatches and 2D index bugs. Budget one sitting for it alone.

---

## 8. Risks to know now

1. **Fuzzy-metric direction.** The paper's `d_f ≤ d` vs. `∏(1−|Δ|/255) ≥ d` phrasing is easy to
   invert; a wrong direction detects every pixel as corrupt. Sanity test: clean image must
   pass detection untouched. Use `dist = 1 − ∏(...)` and you can't get it wrong.
2. **Friend's GPU CUDA version** may cap which compute capability you can target. Check with
   `nvidia-smi` before writing CMake flags.
3. **Timing includes PCIe transfer.** Report kernel-only and end-to-end separately or the
   speedup is misleading for small images.
4. **Salt-and-pepper at high density** can corrupt all 9 pixels of a window; the paper's AMF
   then has no valid neighbours. Handle it (fall back to the median of the whole window) and
   say so in the report — graders like this.
5. **Shared-memory tuning** is empirical. Don't pre-optimize block size; measure 8×8, 16×16,
   32×32 in Step 4.
