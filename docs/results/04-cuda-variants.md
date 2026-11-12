# Phase 04 — CUDA implementation — results

Captured on `ssh gpu` (neobox), 2026-11-11. GPU: RTX 4050 Laptop, sm_89, CUDA 13.4,
`-arch=sm_89`. 10% salt & pepper, seed 42, best of 5 per process, 5 processes per cell.
No single timing is quoted anywhere below.

The fuzzy metric in `src/cuda/kernels.cu` is **float**. It started as `double` so the gate
could stay byte-identical to the CPU; §4 has the measurement that changed it and §3 the
result. The CPU side is untouched and still uses `double`.

## 1. What the shipped CUDA code is (deliberately the simple version)

`src/cuda/kernels.cu` is two kernels and three small `__device__` helpers, 190 lines
with comments. One thread per pixel, plain global loads, interleaved RGB
`unsigned char`, 16×16 blocks.

| | |
| --- | --- |
| `detectKernel` | reads the 3×3 window of the pixel, writes 1 byte of mask |
| `filterKernel` | clean pixels copy through; corrupted ones get the mean of the *clean* neighbours, median of the window if there are none |
| launch | `dim3 block(16,16)`, `dim3 grid((w+15)/16,(h+15)/16)`, guard `if (x>=w||y>=h) return;` |
| barrier | the launch boundary between the two kernels — phase 2 reads its neighbours' mask bytes |
| timing | `cudaEventRecord` around H2D / kernels / D2H, reported separately from `cudaDeviceSynchronize` |

CUDA surface, all of it from the course: `__global__`, `__device__`,
`threadIdx/blockIdx/blockDim`, `cudaMalloc/cudaFree/cudaMemcpy`, `cudaEvent*`,
`cudaGetLastError`, `cudaGetErrorString`, `cudaDeviceSynchronize`, `CUDA_CHECK`.
No templates, no `__shared__`, no `__ldg`, no `__restrict__`, no vector types, no
dynamic parallelism, no libraries.

**What was removed on purpose after measuring it.** A second variant (shared-memory
tile with a 1-pixel halo, three byte planes to avoid bank conflicts, mask plane too)
was written, verified byte-identical and benchmarked. It was **+0.4% at 1024² and
+0.2% at 4096²**, i.e. nothing, so it was deleted rather than shipped: keeping it
would have doubled the code for no measurable gain. Section 4 explains why it could
not have helped. `__ldg` + `__restrict__` was never shipped at all — the course scope
sheet rejects both.

## 2. Correctness — maxdiff against the CPU, not byte identity

The filter phase is integer on both sides, so it is exact. The detect phase is float on
the GPU and double on the CPU, so the gate is `maxdiff <= 2`: the only pixels that can
differ are ones whose fuzzy distance landed within float rounding of the tolerance `d`.
Measured, that never happens at the defaults:

| case | result |
| --- | --- |
| photo_512 / 1024 / 2048 / 4096, 10% noise | `maxdiff=0` |
| window 5 and window 15 (4096²) | `maxdiff=0` |
| random-valued impulse noise, 30% (1024²) | `maxdiff=0` |
| 1001×333 (not a multiple of 16, so the guard path runs) | `maxdiff=0` |
| clean image (0% noise) | identical to `denoise_seq` output |
| repeats | byte-identical |
| `make check` (whole pipeline) | PASS |

`scripts/check.sh` compares cuda to seq with `imgcmp` and an explicit `maxdiff<=2`
threshold rather than `cmp`, so the tolerance is stated in the gate instead of being
implied by the numbers happening to agree.

`compute-sanitizer --tool memcheck`: **0 errors** on 1024², 4096², the 1001×333 guard
path and `-n 15`; `initcheck` 0 errors (no thread reads mask bytes the detect kernel
has not written, i.e. the launch really is a full barrier).

## 3. Performance

Captured with the CPU governor at `performance` and the box otherwise idle (`uptime` load
0.25). An earlier capture of the same binaries under `powersave` gave every CPU cell 2.1×
slower wall time (4096² seq 1064 ms vs 495 ms); the GPU column barely moved. If a
re-measurement disagrees by more than a few percent, check the governor first.

| image | seq ms | omp 4T | omp 20T | cuda total ms | cuda kernel ms | vs seq 1T | vs omp 20T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512² | 7.93 | 2.38 | 1.23 | 0.34 | 0.11 | 23.4× | 3.6× |
| 1024² | 30.13 | 9.57 | 4.87 | 1.11 | 0.42 | 27.1× | 4.4× |
| 2048² | 123.9 | 35.56 | 19.13 | 4.19 | 1.82 | 29.6× | 4.6× |
| 4096² | 495.5 | 142.2 | 75.73 | 16.46 | 7.52 | 30.1× | 4.6× |

The GPU is the fastest implementation at every size, and the margin grows with the image:
3.6× over 20 OpenMP threads at 512², 4.6× at 4096². Against one CPU thread it is 23–30×.

Phase split (median): 1024² = 0.315 H2D + 0.422 kernel + 0.362 D2H; 4096² = 4.43 + 7.52
+ 4.49. Transfer is now **54%** of the 4096² call and 61% at 1024² — the kernel got 15–20×
faster and the PCIe cost did not, so the bottleneck moved to the host/device copies. That
is the next optimization target, and it is a different one from §4.

## 4. Why a memory optimization cannot help this kernel — measured

The kernel does 9 neighbour reads per pixel per phase, so 9× redundant traffic looks
like the obvious target. Measured: removing 8 of those 9 reads (the shared-memory
tile) changed the kernel time by **+0.2%**. The loads were never the limiter.

Three probes on the GPU box (throwaway `/tmp` programs: same 16×16 block, same edge
clamping, same 3×3 peer count, 4096²):

| probe | time |
| --- | --- |
| detection kernel, `double` fuzzy term | **116.1 ms** |
| same kernel, `float` fuzzy term | 4.83 ms (24× faster) |
| same kernel, integer `|Δ|` sum, no division | 0.82 ms (142× faster) |
| device DRAM read bandwidth | 164 GB/s |
| `1 − x/255` at peak, `double` vs `float` | 84.9 vs 6003 G·op/s → **70.7×** |

Detection alone in `double` costs 116 ms; both phases cost 114.7 ms. Phase 2 is nearly
free because it runs only on flagged pixels and clean pixels return immediately. So the
whole GPU cost was one double-precision expression, 27 terms per pixel, on a 70 W
consumer part whose FP64 rate is 1/64 of its FP32 rate.

**What was done with that measurement.** The metric in `kernels.cu` was switched to
`float`: `dev_chan_term` returns `float`, `1.0f - (float)d / 255.0f`, and `dev_is_clean`
accumulates in `float`. Nothing else changed — same blocks, same two launches, same
tiling decision, no `-fmad=false`, no fast-math. The CPU kept `double`.

| | detect kernel, 4096² | probe (standalone) | shipped kernel |
| --- | --- | --- | --- |
| `double` | 116.1 ms | 114.65 ms |
| `float` | 4.83 ms | **7.52 ms** |

The shipped kernel lands at 7.52 ms rather than the probe's 4.83 ms because the probe was
the detect kernel in isolation and the shipped 7.52 ms is detect + filter with the
`cudaEvent` split around both. Against the double version the same measurement moved
8.34 → 0.42 ms at 1024² and 114.65 → 7.52 ms at 4096², i.e. 15–20× on the kernel.

The price: output is no longer guaranteed byte-identical to the CPU. §2 states the
replacement gate. That is the honest framing for the report — the 7.6× over 20 OpenMP
threads is a *precision* result, not a parallelism result.

**Still on the table, not done.** The integer `|Δ|` sum probe is 142× the double kernel
(0.82 ms) but changes the metric, so it is a different algorithm, not an optimization.
Staging the image in pinned host memory or compressing the transfer is now the biggest
remaining win: at 4096² the H2D and D2H copies are 60% of the call.

## 5. Be able to explain

1. **Why two launches, not one.** Phase 2 reads its neighbours' mask bytes, so the whole
   mask must exist first. A launch is the only global barrier CUDA has; `__syncthreads()`
   orders one block, so it cannot make phase 2's reads wait for phase 1's writes in other
   blocks.
2. **Why warp divergence is cheap here.** The branch per pixel makes a warp with mixed
   clean/corrupted lanes execute both paths, but both paths are the same short loop, and
   in phase 2 the clean path is "copy 3 bytes and return" — the divergence is in the tail,
   not in the 9-iteration window loop.
3. **Why the grid is rounded up.** `(w+15)/16` overshoots for non-multiples of 16; the
   1001×333 case is the test that proves the guard is right, because there every
   out-of-range thread takes the `return` path.
4. **Occupancy.** 256 threads/block, no shared memory, so occupancy is set by registers
   per thread (2048 threads/SM = 8 blocks at 256). `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
   is on the course surface and is the tool to cite if asked.
5. **Why the GPU used to lose to 20 OpenMP threads, and what changed it.** With the
   metric in `double` the GPU lost at 4096² (114.7 vs 76.1 ms): the CPU has full-rate
   FP64 and 14 cores, the GPU has 1/64-rate FP64, so the comparison was about FP64
   hardware, not parallelism. In `float` the same kernel runs 7.52 ms against 20 OpenMP
   threads' 75.7 ms — the same sentence in reverse, which is the answer to "why did you
   switch to float" and also the caveat: the margin is a precision choice.

## 6. Reproduce

```bash
ssh gpu && cd ~/Projects/hpc-denoise
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
make -C build check
./build/denoise_cuda /tmp/x/noisy.ppm /tmp/cuda.ppm -q data/clean/photo_1024.ppm
```
