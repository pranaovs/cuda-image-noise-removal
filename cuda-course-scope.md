# CUDA Scope Sheet — what this course actually teaches

Purpose: constrain a coding agent to write only CUDA that appears in `gpu-merged.pdf` (CS3015, lectures 2–6). Everything below is taken from the slides; nothing is added from outside the course.

---

## 0. Paste-ready constraint block

```
Write CUDA C++ using only this surface:
- Qualifiers: __global__, __device__, __host__, __shared__ (static + extern), __constant__
- Builtins: threadIdx, blockIdx, blockDim, gridDim; __syncthreads(), __syncwarp(),
  __activemask(), __shfl_down_sync(mask,v,d), __ballot_sync, __all_sync
- Atomics: atomicAdd, atomicSub, atomicMax, atomicMin, atomicOr, atomicExch, atomicCAS
- Host API: cudaMalloc, cudaFree, cudaMemcpy[Async], cudaMemcpyToSymbol/FromSymbol,
  cudaMallocHost, cudaMallocManaged, cudaMemPrefetchAsync, cudaDeviceSynchronize,
  cudaGetLastError, cudaPeekAtLastError, cudaGetErrorString, cudaGetDeviceProperties,
  cudaOccupancyMaxActiveBlocksPerMultiprocessor, cudaStreamCreate,
  cudaEventCreate/Record/Synchronize/ElapsedTime
- Libraries: thrust::reduce / exclusive_scan / device_vector, cublasSgemm, CuPy
- Fixed grid sizes and shared-memory tiles; block size multiple of 32, <= 1024
- Every API call wrapped in CUDA_CHECK; cudaGetLastError after every launch
Do NOT use: cooperative groups, CUDA graphs, dynamic parallelism, CUB, CUTLASS,
tensor-core intrinsics, __ldg, __restrict__, float4/int4 vector loads, texture memory,
cluster/DSMEM, __launch_bounds__, maxrregcount, half2 math.
```

---

## 1. Kernel language surface (Lecture 2)

| Item | Taught form |
|---|---|
| Kernel decl | `__global__ void k(const float* a, float* c, int n)` |
| Device fn | `__device__ float f(...)` — callable from kernel, `__host__ __device__` for both |
| Launch | `k<<<blocks, threads>>>(args)` / `k<<<blocks, threads, shmemBytes, stream>>>(args)` |
| Index | `int i = blockIdx.x * blockDim.x + threadIdx.x;` then `if (i < n)` |
| Sizing | `int threads = 256; int blocks = (n + threads - 1) / threads;` |
| Limits | 1024 threads/block, `blockDim.z<=64`, `gridDim.x<=2^31-1`, `gridDim.y/z<=65535` |
| 2-D | `x = blockIdx.x*blockDim.x+threadIdx.x` (col), `y = ...y...` (row), `idx = y*width+x` |
| Divergence | warp = 32 lanes; a branch costs both paths; keep branch work per-warp uniform |
| Sync | `__syncthreads()` = block barrier only; never inside a divergent branch |
| Warp intrinsics | `__shfl_down_sync(0xffffffffu, v, d)`, `__activemask()`, `__ballot_sync`, `__all_sync` |

Divergence-safe reduction of a value across a warp (only shuffle form taught):

```cuda
__device__ float warpSum(float v) {
    for (int d = 16; d > 0; d /= 2) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;   // lane 0 holds the warp total
}
```

## 2. Host API surface

| Function | Use in course |
|---|---|
| `cudaMalloc(&d_p, bytes)`, `cudaFree(d_p)` | device alloc; sizes always `n*sizeof(T)` |
| `cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice)` / `...DeviceToHost` | the 4-step program |
| `cudaMallocHost(&h_p, bytes)` | pinned memory, required for stream overlap |
| `cudaMemcpyAsync(dst,src,bytes,kind,stream)` | per-chunk copies with streams |
| `cudaDeviceSynchronize()` | wait; stopwatch/print/exit need it |
| `cudaGetLastError()` / `cudaPeekAtLastError()` | launch-configuration errors |
| `cudaGetErrorString(e)` | message text in the macro |
| `cudaGetDeviceProperties(&prop, 0)` | name, SM count, clocks, memory, compute capability |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, block, shmem)` | occupancy probe |
| `cudaStreamCreate(&st)`, `<<<...,st>>>`, `cudaMemcpyAsync(...,st)` | 4-stream chunk pipeline |
| `cudaEventCreate/Record/Synchronize/ElapsedTime` | kernel timing (CPU clock cannot time async work) |
| `cudaMallocManaged(&p,bytes)`, `cudaMemPrefetchAsync(p,bytes,device,stream)` | unified memory |
| `cudaMemcpyToSymbol` / `FromSymbol` | filling `__constant__` arrays |

Error macro (paste at top of file, exactly this shape):

```cuda
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
```

Timing pattern: warm-up launch → `cudaEventRecord(start)` → loop N launches → `cudaEventRecord(stop)` → `cudaEventSynchronize(stop)` → `cudaEventElapsedTime`.

## 3. Memory model as taught (Lectures 3–4)

- Registers (per thread, fastest) → `__shared__` (per block, ~48 KB, needs `__syncthreads`) → global (HBM/DRAM, hundreds of cycles) → local memory (private but DRAM-backed, from register spill).
- `__constant__` = one read-only value broadcast to all threads, filled via `cudaMemcpyToSymbol`.
- **Coalescing**: global memory moves in 32-byte sectors / 128-byte lines. Consecutive threads must touch consecutive addresses. Stride > 1 multiplies sectors fetched → "effective bandwidth" drops. Misalignment cost is small; stride cost is not.
- **Banks**: shared memory has 32 banks; lane `t` reading `s[t]` is conflict-free; `s[t*stride]` with even stride is a multi-way conflict. Fix padding: declare `s[32][33]`, not `s[32][32]`.
- **2-D rule**: threads closest in `threadIdx.x` must read closest in memory along a row, never down a column — unless you stage the tile through shared memory.
- **AoS vs SoA**: prefer one array per field when the kernel reads only one field.

## 4. Patterns the course actually builds

1. **Vector add** — the four steps: `cudaMalloc` → `cudaMemcpy` H2D → kernel → `cudaMemcpy` D2H (`cudaFree`).
2. **Grid-stride loop** — fixed grid, arbitrary `n`:
   ```cuda
   for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) c[i] = a[i] + b[i];
   ```
3. **Block-only array reverse** via `__shared__` + `__syncthreads` (step 1 load, step 2 barrier, step 3 write reversed).
4. **Tiled transpose** — load 32×32 tile to shared, `__syncthreads`, write transposed; padded tile `[32][33]` removes the column-read conflict.
5. **Naive matmul** — one thread per `C[row][col]`, loop `k`; then count its global loads.
6. **Tiled matmul** — per `k`-tile phase: load `A`/`B` tiles into shared → `__syncthreads` → accumulate → `__syncthreads`. Boundary guard: out-of-range loads store `0.0f`, all threads still execute the load and both barriers.
7. **Histogram with privatization** — `__shared__ int hist[256]` per block (`atomicAdd` or non-atomic local count since bins are per block), `__syncthreads`, then one merge per bin into global.
8. **Reduction ladder** — 4 variants, in this order: interleaved `s[t]+=s[t+s]` (conflicts) → sequential `for(half=blockDim.x/2; half>0; half/=2) if(t<half) s[t]+=s[t+half]` (conflict-free, needs power-of-two block) → two loads per thread / packed active threads → `warpSum` shuffle for the last warp, then `if (t==0) atomicAdd(total, s[0])`.
9. **Scan ladder** — serial scan (O(n) steps) → Hillis–Steele `for(offset=1;offset<n;offset*=2)` double-buffer, O(n log n) work → Blelloch up-sweep then down-sweep, O(n) work, root cleared to 0, two elements per thread; arrays bigger than one block = per-block scan + block-sum scan + add offsets.
10. **Streams** — split `n` into 4 chunks, one stream per chunk, pinned host memory, `cudaMemcpyAsync` H2D → kernel on `st[k]` → `cudaMemcpyAsync` D2H → single `cudaDeviceSynchronize()`.
11. **Unified memory** — `cudaMallocManaged` + same pointer on host and device; optional `cudaMemPrefetchAsync`; use for prototypes, not hot loops.
12. **Libraries** — `thrust::reduce`, `thrust::exclusive_scan`, `thrust::device_vector`, `cublasSgemm` (remember column-major), CuPy in Python.

## 5. Correctness checklist (slide 58 of lecture 2)

1. Sizes in bytes: `n * sizeof(T)`.
2. `h_` / `d_` naming; never pass the wrong one.
3. `blocks = (n + T - 1)/T`, and every kernel guards `if (i < n)`.
4. Block size a multiple of 32, ≤ 1024 (128/256/512 typical).
5. `CUDA_CHECK` on every call, `cudaGetLastError()` after every launch, `cudaDeviceSynchronize()` too while debugging.
6. Run `compute-sanitizer` once on every new kernel.
7. Compare against a CPU reference answer.
8. Warm up before timing; time with `cudaEvent`s; report copies and kernel separately.
9. Never `printf` timing from the kernel loop; floats sum non-deterministically across thread orderings.

## 6. Not in the course — reject if an agent emits it

Cooperative groups, `cudaGraph`, dynamic parallelism, CUB (DeviceReduce/DeviceScan), CUTLASS, tensor cores / WMMA / `half2`, `__ldg`, `__restrict__`, `float4`/`int4` vector types, texture/surface memory, distributed shared memory, thread-block clusters, `__launch_bounds__`, `-maxrregcount`, `cudaGetDeviceCount`, `cudaSetDevice`, `cudaDeviceReset`, templates inside kernels, `constexpr` tricks, `#include <cuda_runtime_api.h>` extras beyond the list above.

Also outside the course: any performance claim without a measured event-timed number, and any tutorial-style "let me explain" padding in the code.

## 7. Where each thing is taught (for citing back)

| Topic | Lecture | Slide-ish |
|---|---|---|
| launch/grid/block, limits, error macro | 2 | `<<<>>>` … "A Checklist for Every CUDA Program" |
| 2-D indexing, grid-stride | 2 | "Going 2-D", "The Grid-Stride Loop" |
| warps, divergence, occupancy, registers | 3 | "Warp Divergence", "Occupancy" |
| memory levels, qualifiers, coalescing, AoS/SoA | 3 | "Where Data Lives", "Coalescing" |
| shared memory, banks, transpose, matmul tiling, GFLOPS | 4 | "Banks and Bank Conflicts", "Tiling with Shared Memory" |
| atomics, contention, histogram privatization | 5 | "The Atomic Family", "Count Locally, Report Once" |
| reduction, shuffles, Hillis–Steele, Blelloch | 5 | "Faster Reductions", "The Hillis–Steele Scan", "The Blelloch Scan" |
| streams, pinned memory, unified memory, libraries | 6 | "One Chunk per Stream", "Unified Memory", "GPU Libraries" |
