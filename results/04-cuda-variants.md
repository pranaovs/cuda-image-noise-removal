# Phase 04 — CUDA implementation — results

Captured on `ssh gpu` (neobox), 2026-11-11. GPU: RTX 4050 Laptop, sm_89, CUDA 13.4,
`-arch=sm_89`. 10% salt & pepper, seed 42, best of 5 per process, 5 processes per cell.
No single timing is quoted anywhere below.

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

## 2. Correctness — maxdiff must be 0 (all maths is integer)

`denoise_cuda` output compared byte-for-byte with `denoise_seq` via `imgcmp`.

| case | result |
| --- | --- |
| photo_512 / 1024 / 2048 / 4096, 10% noise | `maxdiff=0` |
| window 5 and window 15 | `maxdiff=0` |
| 1001×333 (not a multiple of 16, so the guard path runs) | `maxdiff=0` |
| clean image (0% noise) | identical to `denoise_seq` output |
| repeats | byte-identical |
| `make check` (whole pipeline) | PASS |

`compute-sanitizer --tool memcheck`: **0 errors** on 1024², 4096², the 1001×333 guard
path and `-n 15`; `initcheck` 0 errors (no thread reads mask bytes the detect kernel
has not written, i.e. the launch really is a full barrier).

## 3. Performance

| image | seq ms | omp 4T | omp 20T | cuda total ms | cuda kernel ms | cuda vs seq |
| --- | --- | --- | --- | --- | --- | --- |
| 512² | 10.13 | 2.68 | 2.61 | 2.18 | 1.95 | 4.6× |
| 1024² | 39.67 | 10.98 | 5.65 | 9.08 | 8.34 | 4.4× |
| 2048² | 162.1 | 42.87 | 18.95 | 30.87 | 28.08 | 5.3× |
| 4096² | 649.2 | 171.9 | 76.13 | 124.3 | 114.7 | 5.2× |

Phase split (median): 1024² = 0.32 H2D + 8.34 kernel + 0.39 D2H; 4096² = 4.61 + 114.65
+ 5.06. PCIe is 7.8% of the call at both sizes and holds ~11 GB/s each way, so at these
image sizes the GPU is not transfer-limited — but at 256² it would be.

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

Detection alone in `double` costs 116 ms; both shipped phases cost 114.7 ms. Phase 2 is
nearly free because it runs only on flagged pixels and clean pixels return immediately.
So the whole GPU cost is one double-precision expression, 27 terms per pixel, on a
70 W consumer part whose FP64 rate is 1/64 of its FP32 rate.

**This is also why the paper's GPU numbers look better than ours:** the paper states
"single-precision calculations". On this card FP32 is 71× the FP64 rate. Switching the
metric to `float` is the single change that would move the GPU number, and it is a
precision decision, not a tiling decision — named as future work, not implemented,
because bit-identical output against the CPU is the correctness gate this project set
itself.

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
5. **Why the GPU loses to 20 OpenMP threads at 4096²** (114.7 vs 76.1 ms): the CPU has
   full-rate FP64 and 14 cores; the GPU has 1/64-rate FP64. The comparison is about FP64
   hardware, not about parallelism — the GPU is still 5.2× the single-threaded CPU.

## 6. Reproduce

```bash
ssh gpu && cd ~/Projects/hpc-denoise
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build -j
make -C build check
./build/denoise_cuda /tmp/x/noisy.ppm /tmp/cuda.ppm -q data/clean/photo_1024.ppm
```
