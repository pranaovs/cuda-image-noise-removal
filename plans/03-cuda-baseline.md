# Phase 03 — CUDA baseline (naive)

**Status: done.** Evidence: `results/03-cuda-baseline.txt` (maxdiff=0 at 1024²/4096²/window 5/
1001×333, deterministic, memcheck+initcheck 0 errors, h2d/kernel/d2h split).

**Prereq:** phases 01–02 verified. Run this on the laptop with the NVIDIA GPU.
**Time:** ~3 h. This is where projects stall — budget it as its own sitting.

## Step 0 — check the box before writing code

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version          # or: module load cuda
lscpu | grep -E 'Model name|^CPU\(s\)'
```

Write those values into `results/setup.txt` — they are the report's "Experimental setup".
Compute capability caps you: `-arch=native` is valid on CUDA ≥ 11.5; otherwise use
`-gencode arch=compute_XX,code=sm_XX` from the *actual* GPU you see in `nvidia-smi`.

## Steps

- [x] 1. `src/cuda/kernels.cuh` — `extern "C"` declarations only, so C `main` can link:
      `void denoise_cuda(Image*, unsigned char *mask, const DenoiseParams*, CudaTiming *t);`
- [x] 2. `src/cuda/kernels.cu` — as built, both kernels take `DenoiseParams p` by value
      instead of `(double d, int m)`, so `-n 5` keeps working. `__global__ detectKernel(const uchar3 *in, unsigned char
      *mask, int w, int h, double d, int m)` and `__global__ filterKernel(const uchar3 *in,
      const unsigned char *mask, uchar3 *out, int w, int h)`.
      Indexing: `int x = blockIdx.x*blockDim.x + threadIdx.x; int y = blockIdx.y*blockDim.y +
      threadIdx.y; if (x>=w || y>=h) return;` — 2D block 16×16, grid `ceil` per dimension.
      **Compare against `core.h` line by line**: same fuzzy distance, same clamp-to-edge, same
      median fallback, same integer rounding. This is where CUDA vs CPU silently diverges.
- [x] 3. `src/cuda/host.cu` — as built, the three buffers are allocated once and reused by
      later repetitions (a 100 MB `cudaMalloc` costs more than the kernels and would show up
      in `ms` but in none of the phase timers); the mask is never copied back (no host
      reader). `cudaMalloc` in/out/mask, `cudaMemcpy` H2D, launch detect,
      launch filter (the launch boundary *is* the phase barrier), `cudaMemcpy` D2H, free.
      Time **three separate things** with `cudaEvent`/host clock:
      `t_h2d`, `t_kernel` (both kernels), `t_d2h`. Transfer time is not kernel time — say so.
- [x] 4. No separate `main_cuda.cu`: the shared `src/app/main_denoise.c` is compiled into
      all three targets, so flags and timing placement cannot drift (a second CLI would make
      a speedup an artefact). Under `-DDENOISE_HAS_CUDA` it adds
      `h2d_ms=.. kernel_ms=.. d2h_ms=.. variant=naive` to the same one-line output.
- [x] 5. CMake (`src/cuda/CMakeLists.txt`, reached only via `-DENABLE_CUDA=ON`):
      `enable_language(CUDA)`, `set(CMAKE_CUDA_ARCHITECTURES native)` guarded by a
      fallback, target `denoise_cuda`, plus `-DENABLE_CUDA=ON`.
- [x] 6. Run the clean-image and 10%-noise cases and `imgcmp` CUDA vs seq. Expect
      `maxdiff=0`. If not zero, the divergence is in the edge clamp or the median fallback.

## Verify

```bash
./build/denoise_cuda /tmp/noisy.ppm /tmp/cuda.ppm -d 0.6 -m 2
./build/imgcmp /tmp/cuda.ppm /tmp/seq.ppm     # pass: identical=yes, maxdiff=0
for i in 1 2 3 4 5; do ./build/denoise_cuda /tmp/noisy.ppm /tmp/c$i.ppm; done
cmp /tmp/c1.ppm /tmp/c5.ppm                   # determinism
```

Also run with `compute-sanitizer ./build/denoise_cuda ...` once — it catches out-of-bounds
indexing that happens to produce the right answer on your tile size.

## Be able to explain

1. Why two kernel launches instead of one — the launch is the global barrier.
2. Why the kernel is faster than OpenMP despite doing the same 9 reads (thousands of
   concurrent threads hide memory latency; not fewer bytes moved).
3. Why end-to-end time for a small image can be *worse* than CPU (PCIe dominates).

## Done when

`maxdiff=0` against seq, deterministic across 5 runs, and you have h2d/kernel/d2h split times.
