# HPC final project - exam notes

ASCII only. Build from the repo root:

    pandoc docs/NOTES-HPC-EXAM.md -o notes.pdf --pdf-engine=tectonic \
        -V geometry:margin=1.5cm -V fontsize=9pt

# 1. What the program does

Removes salt & pepper noise from a PPM image using the peer-group / fuzzy-metric
filter from a 2013 paper, three ways:

    denoise_seq    plain C, one core        -> the reference
    denoise_omp    OpenMP, all cores
    denoise_cuda   CUDA, the GPU

Rule of the whole project: **the parallel versions must output the same bytes as
the sequential one** (maxdiff = 0, checked with cmp). Not "similar" - identical.
That is why phase 2 uses integer arithmetic only.

# 2. Image, minimum you need

- Pixel = 3 bytes (R,G,B), each 0..255. 1024x1024 = 3 MB.
- Storage: one flat array, row-major. Pixel (x,y) starts at byte `(y*w + x)*3`
  (that is `px()` in image.h). Rows are contiguous, columns are strided.
- File: PPM P6 = "P6", width, height, 255, then raw bytes. No library needed.
- Noise we use: salt & pepper = a random `density` fraction of pixels is set to
  black (0,0,0) or white (255,255,255). Same seed -> same noise everywhere.
- Quality: `PSNR = 10*log10(255^2 / MSE)` in dB (30 = good, 40 = looks identical).
  Also `maxdiff` = largest single-byte difference; ours must be 0.
  PSNR alone rewards blurring, so we also report MAE and show a crop.

Numbers: 10% noise = PSNR 13.80 dB (wrecked). Denoised = 32.26 dB (+18.5 dB).

# 3. The algorithm

Window: square, side `window` (3 by default) = 9 pixels, radius r = window/2.
`d` = fuzzy tolerance (0..1), `m` = peer count threshold. Defaults d=0.5, m=4
(peak of a 49-point sweep, not from the paper).

Fuzzy distance (multiplicative, not an average, so one bad channel is enough):

    t_c  = 1 - |a_c - b_c| / 255            (1 identical, 0 opposite)
    dist = 1 - t_R * t_G * t_B              (0 identical, ~1 different)

Phase 1 - detect: a pixel is CLEAN if at least `m+1` pixels of its window
(itself included) are within distance `d`. Else CORRUPTED. Writes mask[] = 0/1.

Phase 2 - filter: clean pixel -> copy through. Corrupted pixel -> replace with
the mean of its **clean** neighbours only (integer sum / integer count). If no
neighbour is clean, use the median of the whole window (rare fallback).

Why it preserves edges: corrupted pixels never enter the average, and a neighbour
on the far side of an edge is not a peer, so detail is not averaged away.
Why the output is bit-identical: one writer per pixel, no atomics, no reduction,
and integer division truncates the same on every machine.

**The phase barrier** - the single most important structural fact:

    phase 2 reads its neighbours' mask bytes,
    so the ENTIRE mask must be finished first.

    seq   : first loop ends
    omp   : end of the first `parallel for` (implicit barrier)
    cuda  : second kernel launch (a launch is the only global barrier)

Worked example, 3x3 window, all 100 except centre = 0 (corrupted):

    100 100 100
    100   0 100
    100 100 100

Detection: t = 1 - 100/255 = 0.608 per channel, dist = 1 - 0.608^3 = 0.775 > d,
so neighbours are not peers; only the centre is (dist 0) -> count 1 < m+1=5 ->
CORRUPTED. Repair: 8 clean neighbours of 100 -> 800/8 = 100. Exact, no blur.
A clean neighbour with 8 peers around 100 -> count 8 >= 5 -> CLEAN, untouched.

# 4. Concepts, one line each

- **Serial vs parallel**: one instruction stream vs several at once. OpenMP =
  threads on one CPU sharing RAM. CUDA = separate device with its own memory.
- **Speedup** S = T(1)/T(N). **Efficiency** E = S/N (1.0 = perfect). We get
  E = 0.90-0.95 at 4 OpenMP threads; cuda is 4.4-5.2x one CPU thread.
- **Throughput**: Mpix/s. seq stays ~26 Mpix/s at every size (work per pixel is
  constant, no reuse), so time is linear in pixels.
- **Amdahl**: S = 1/(s + (1-s)/N), s = serial fraction. Not our limit: I/O is
  outside the timed region, and inside it the only serial parts are the barrier
  and (for CUDA) PCIe at 7.8%.
- **Strong scaling**: fixed problem, more workers. **Weak scaling**: fixed work
  per worker. Ours: 1 thread/1024^2 = 41.4 ms vs 4 threads/2048^2 = 42.9 ms
  (+3.6%) -> per-thread throughput held.
- **Latency vs bandwidth**: time for one access vs bytes/s in total. Latency is
  hidden by having many threads; bandwidth is not.
- **Cache hierarchy**: registers -> L1/shared -> L2 -> DRAM, each 10x bigger and
  slower. A 3x3 stencil reads each pixel ~9x; if those hit cache it is free.
- **Arithmetic intensity** = FLOPs/bytes. Low = memory-bound, high = compute-bound.
  Measured: our GPU kernel uses ~240 MB/s of a 164 GB/s bus = 0.15% -> NOT
  memory-bound. It is compute-bound on FP64 (finding 3).
- **FP32 vs FP64**: float 32-bit, double 64-bit. Server CPUs run FP64 at half
  rate; consumer GPUs at **1/64**. Our code uses double (like the reference), so
  FP64 is the bottleneck, not memory.
- **SIMT / warp**: 32 CUDA threads run in lockstep; if they branch differently the
  warp runs both paths with lanes masked (divergence). Block sizes are multiples
  of 32 -> 16x16 = 256 threads = 8 warps.
- **Occupancy**: active warps per SM / max. Set by registers, shared memory,
  threads. Ours: 256 threads, no shared memory -> limited by 2048 threads/SM.
  Arithmetic only - ncu/nsys are not installed on the box, so say that.
- **PCIe**: the GPU's extra tax. H2D before, D2H after. Measured 4.6 / 114.7 /
  5.1 ms at 4096^2 = 7.8% of the call, ~11 GB/s each way.
- **Hybrid CPU / SMT**: i7-13700H = 6 fast P-cores + 8 slow E-cores (14 physical,
  20 logical with hyperthreading). SMT adds ~nothing here; an E-core is ~2.9x
  slower for this work. This explains finding 2.
- **Reproducibility**: float addition is not associative, so a parallel float sum
  may differ. We avoid that by integer arithmetic, one writer per pixel.

# 5. CUDA essentials

- host = CPU+RAM, device = GPU+VRAM. Kernel = `__global__` function run on GPU,
  one instance per thread. `__device__` = helper callable only from GPU code.
- A thread finds its pixel itself:

      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      if (x >= w || y >= h) return;            // guard, grid is rounded up

- Launch: `detectKernel<<<grid, block>>>(args)` with `block(16,16)`,
  `grid((w+15)/16, (h+15)/16)`. The +15 ceiling keeps 1001-wide images covered;
  the extra threads hit the guard. Test image 1001x333 exists for that path.
- 4 steps (lecture 2): `cudaMalloc` -> `cudaMemcpy` H2D -> kernels -> `cudaMemcpy`
  D2H. Buffers are allocated once and reused, or cudaMalloc would pollute the time.
- A launch is **asynchronous**, so time it with `cudaEventRecord` +
  `cudaEventElapsedTime`, never `clock_gettime`.
- Two kernels, not one, because of the phase barrier: `__syncthreads()` only
  synchronises one block, and a neighbour may live in another block.
- Every call wrapped in `CUDA_CHECK`, `cudaGetLastError()` after each launch.
- Mask never copied back to host: nothing reads it, and it would inflate D2H.
- Scope: no `__ldg`, `__restrict__`, templates, vector types, dynamic shared
  memory, cooperative groups, graphs, CUB. `dim3` is not on the sheet's list but
  a 2-D grid needs it (taught "going 2-D" form).

# 6. OpenMP essentials

- Fork-join: at `#pragma omp parallel for` a thread team starts, iterations are
  split, and the region ends with an **implicit barrier** - that is our phase
  barrier. Two regions, two barriers.
- `collapse(2)`: merges the y and x loops into h*w iterations, so threads split
  pixels, not rows.
- `schedule(static)`: equal chunks up front, no runtime overhead. Right choice
  for uniform work - and measurably the wrong choice on unequal cores (finding 2).
- No race: one writer per output byte; reads are read-only input or mask bytes
  behind the barrier. No reduction, no atomics, no critical.
- Benchmarks must export `OMP_NUM_THREADS`, `OMP_PLACES=cores`,
  `OMP_PROC_BIND=close`, or the OS migrates threads and timings wobble.

# 7. Build

There is no hand-written Makefile: CMake **generates** it.

    cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
    cmake --build build -j
    make -C build check

Targets: `gennoise`, `imgcmp`, `fbcount`, `denoise_seq`, `denoise_omp`,
`denoise_cuda` (only with `-DENABLE_CUDA=ON`), and `check` (runs check.sh).

The trick that makes the comparison fair: all three denoisers compile the **same**
`src/app/main_denoise.c`; only one macro differs -

    target_compile_definitions(denoise_seq  PRIVATE DENOISE_FN=denoise_seq_impl
                                                IMPL_NAME="seq")
    target_compile_definitions(denoise_omp  PRIVATE DENOISE_FN=denoise_omp_impl
                                                DENOISE_HAS_OMP=1)
    target_compile_definitions(denoise_cuda PRIVATE DENOISE_FN=denoise_cuda_impl
                                                DENOISE_HAS_CUDA=1)

So argument parsing, the repetition loop, the timing bracket and the report line
are identical in all three binaries. A speedup cannot be a measurement artefact.
`src/cuda/` is only parsed when the flag is on, so a box without nvcc still builds
and tests everything CPU-side.

Deliberately absent: `-use_fast_math` / `-ffast-math` (would break float agreement
with the CPU) and `-march=native` for C (would make CPU-vs-GPU machine-specific).

# 8. The code

## 8.1 image.h - the container

```c
typedef struct { int w, h; unsigned char *rgb; } Image;   /* 8-bit RGB, row-major */

static inline unsigned char *px(const Image *img, int x, int y)
{ return img->rgb + ((size_t)y * (size_t)img->w + (size_t)x) * 3; }

static inline int img_clamp(int v, int hi)          /* border -> edge pixel */
{ return v < 0 ? 0 : (v >= hi ? hi - 1 : v); }
```

## 8.2 core.h - the algorithm (no threads, no GPU: this is the spec)

```c
static inline double fuzzy_dist(const unsigned char a[3], const unsigned char b[3])
{
    double prod = 1.0;
    for (int c = 0; c < 3; c++) {
        int d = (int)a[c] - (int)b[c];
        if (d < 0) d = -d;
        prod *= 1.0 - (double)d / 255.0;
    }
    return 1.0 - prod;
}

static inline int is_clean(const Image *img, int x, int y, const DenoiseParams *p)
{                                   /* phase 1: peer count in the window */
    int r = p->window / 2, count = 0;
    const unsigned char *c = px(img, x, y);
    for (int dy = -r; dy <= r; dy++) {
        int yy = img_clamp(y + dy, img->h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = img_clamp(x + dx, img->w);
            if (fuzzy_dist(c, px(img, xx, yy)) <= p->d) count++;
        }
    }
    return count >= p->m + 1;
}

static inline void repair_pixel(const Image *img, const unsigned char *mask, int x, int y,
                                const DenoiseParams *p, unsigned char *dst)
{                                   /* phase 2: mean of the CLEAN neighbours only */
    int r = p->window / 2, sum[3] = {0,0,0}, count = 0;
    for (int dy = -r; dy <= r; dy++) {
        int yy = img_clamp(y + dy, img->h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = img_clamp(x + dx, img->w);
            if (mask[(size_t)yy * img->w + xx]) continue;   /* skip corrupted */
            const unsigned char *q = px(img, xx, yy);
            sum[0] += q[0]; sum[1] += q[1]; sum[2] += q[2];
            count++;
        }
    }
    if (count > 0) {                /* integer divide -> identical everywhere */
        dst[0] = (unsigned char)(sum[0]/count);
        dst[1] = (unsigned char)(sum[1]/count);
        dst[2] = (unsigned char)(sum[2]/count);
        return;
    }
    unsigned char vals[PARAMS_MAX_WINDOW * PARAMS_MAX_WINDOW];
    for (int c = 0; c < 3; c++) {   /* fallback: per-channel median of window */
        int n = 0;
        for (int dy = -r; dy <= r; dy++) {
            int yy = img_clamp(y + dy, img->h);
            for (int dx = -r; dx <= r; dx++) {
                int xx = img_clamp(x + dx, img->w);
                vals[n++] = px(img, xx, yy)[c];
            }
        }
        qsort(vals, (size_t)n, 1, cmp_u8);
        dst[c] = vals[n / 2];
    }
}
```

## 8.3 denoise_seq.c - two loops, one buffer swap

```c
void denoise_seq_impl(Image *img, unsigned char *mask, const DenoiseParams *p)
{
    const int w = img->w, h = img->h;

    for (int y = 0; y < h; y++)                 /* phase 1 */
        for (int x = 0; x < w; x++)
            mask[(size_t)y*w + x] = is_clean(img, x, y, p) ? 0 : 1;
                                                /* loop end = phase barrier */
    Image *out = image_alloc(w, h);             /* new buffer: phase 2 reads the
                                                   originals, cannot write in place */
    for (int y = 0; y < h; y++) {               /* phase 2 */
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y*w + x;
            unsigned char *dst = px(out, x, y);
            if (mask[i]) repair_pixel(img, mask, x, y, p, dst);
            else         memcpy(dst, px(img, x, y), 3);
        }
    }
    unsigned char *tmp = img->rgb;              /* swap instead of a 3 MB copy */
    img->rgb = out->rgb; out->rgb = tmp;
    image_free(out);
}
```

## 8.4 denoise_omp.c - same body, two pragmas

```c
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; y++)                 /* phase 1 */
        for (int x = 0; x < w; x++)
            mask[(size_t)y*w + x] = is_clean(img, x, y, p) ? 0 : 1;
    /* implicit barrier here: the mask is complete before phase 2 starts */

    Image *out = image_alloc(w, h);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; y++) {               /* phase 2, identical body */
        ...
    }
```

If asked "where is the barrier": the end of the first `parallel for`. If asked
"race condition": none - one writer per byte, reads are read-only or behind the
barrier. If asked "why static": uniform work; and admit it was the wrong choice on
this hybrid CPU (15.19 ms explicit 6P+8E split vs 4.60 ms at 20 threads).

## 8.5 kernels.cu - the GPU, one thread per pixel

```cuda
__device__ int dev_clamp(int v, int hi) { return v < 0 ? 0 : (v >= hi ? hi-1 : v); }

__device__ double dev_chan_term(int a, int b)      /* one channel of the metric */
{
    int d = a - b; if (d < 0) d = -d;
    return 1.0 - (double)d / 255.0;
}

__device__ int dev_is_clean(const unsigned char *in, int w, int h, int x, int y, DenoiseParams p)
{
    int r = p.window / 2, count = 0;
    const unsigned char *c = in + 3*((size_t)y*w + x);
    for (int dy = -r; dy <= r; dy++) {
        int yy = dev_clamp(y + dy, h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = dev_clamp(x + dx, w);
            const unsigned char *q = in + 3*((size_t)yy*w + xx);
            double dist = 1.0 - dev_chan_term(c[0], q[0]) * dev_chan_term(c[1], q[1])
                                * dev_chan_term(c[2], q[2]);
            if (dist <= p.d) count++;
        }
    }
    return count >= p.m + 1;
}

__device__ void dev_sort_u8(unsigned char *v, int n)     /* no qsort on device */
{                                                        /* only the rare fallback */
    for (int i = 1; i < n; i++) {
        unsigned char key = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j+1] = v[j]; j--; }
        v[j+1] = key;
    }
}

__global__ void detectKernel(const unsigned char *in, unsigned char *mask, int w, int h,
                             DenoiseParams p)                    /* phase 1 */
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;                /* guard: grid is rounded up */
    mask[(size_t)y*w + x] = dev_is_clean(in, w, h, x, y, p) ? 0 : 1;
}

__global__ void filterKernel(const unsigned char *in, const unsigned char *mask,
                             unsigned char *out, int w, int h, DenoiseParams p)
{                                                /* phase 2 */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int r = p.window / 2;
    size_t i = (size_t)y*w + x;
    if (!mask[i]) {                              /* clean: copy 3 bytes, done */
        out[3*i+0] = in[3*i+0]; out[3*i+1] = in[3*i+1]; out[3*i+2] = in[3*i+2];
        return;
    }
    int sum[3] = {0,0,0}, count = 0;             /* mean of clean neighbours */
    for (int dy = -r; dy <= r; dy++) {
        int yy = dev_clamp(y + dy, h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = dev_clamp(x + dx, w);
            size_t j = (size_t)yy*w + xx;
            if (mask[j]) continue;               /* corrupted: not in the average */
            sum[0] += in[3*j+0]; sum[1] += in[3*j+1]; sum[2] += in[3*j+2];
            count++;
        }
    }
    if (count > 0) {
        out[3*i+0] = (unsigned char)(sum[0]/count);
        out[3*i+1] = (unsigned char)(sum[1]/count);
        out[3*i+2] = (unsigned char)(sum[2]/count);
        return;
    }
    unsigned char vals[PARAMS_MAX_WINDOW * PARAMS_MAX_WINDOW];
    for (int c = 0; c < 3; c++) {                /* median fallback, same as core.h */
        int n = 0;
        for (int dy = -r; dy <= r; dy++) {
            int yy = dev_clamp(y + dy, h);
            for (int dx = -r; dx <= r; dx++) {
                int xx = dev_clamp(x + dx, w);
                vals[n++] = in[3*((size_t)yy*w + xx) + c];
            }
        }
        dev_sort_u8(vals, n);
        out[3*i + c] = vals[n/2];
    }
}
```

## 8.6 host.cu - allocate, copy, launch, copy, time

```cuda
#define CUDA_CHECK(call) do { cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", \
        cudaGetErrorString(e_)); exit(1); } } while (0)

void denoise_cuda(Image *img, unsigned char *mask, const DenoiseParams *p, CudaTiming *t)
{
    const int w = img->w, h = img->h;
    const size_t nbytes = (size_t)w*h*3;
    (void)mask;                       /* host mask unused: no D2H copy needed */

    buffers_reserve(nbytes, (size_t)w*h);      /* cudaMalloc once, reuse across reps */

    dim3 block(16, 16);                        /* 256 threads = 8 warps */
    dim3 grid((w + 15) / 16, (h + 15) / 16);   /* ceiling: covers non-multiples */

    CUDA_CHECK(cudaEventRecord(g_ev[0]));
    CUDA_CHECK(cudaMemcpy(g_in, img->rgb, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(g_ev[1]));                  /* <- H2D ends */

    detectKernel<<<grid, block>>>(g_in, g_mask, w, h, *p);
    CUDA_CHECK(cudaGetLastError());
    filterKernel<<<grid, block>>>(g_in, g_mask, g_out, w, h, *p);  /* phase barrier */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(g_ev[2]));                  /* <- kernels end */

    CUDA_CHECK(cudaMemcpy(img->rgb, g_out, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(g_ev[3]));
    CUDA_CHECK(cudaEventSynchronize(g_ev[3]));

    float h2d, kern, d2h;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, g_ev[0], g_ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&kern, g_ev[1], g_ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&d2h,  g_ev[2], g_ev[3]));
    if (t) { t->h2d_ms = h2d; t->kernel_ms = kern; t->d2h_ms = d2h; g_last = *t; }
}
```

Four events -> three intervals -> H2D, kernel, D2H, reported separately. Charging
PCIe to "GPU compute" would make the speedup claim wrong.

## 8.7 kernels.cuh - the C/CUDA boundary

```c
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { double h2d_ms, kernel_ms, d2h_ms; } CudaTiming;
void denoise_cuda(Image *img, unsigned char *mask, const DenoiseParams *p, CudaTiming *t);
void denoise_cuda_impl(Image *img, unsigned char *mask, const DenoiseParams *p);
void denoise_cuda_last_timing(CudaTiming *t);
#ifdef __cplusplus
}
#endif
```

Plain C only (no CUDA types), so the shared C main can print the split without
including `cuda_runtime.h`.

## 8.8 main_denoise.c - the timing loop (lines 104-141, excerpt)

```c
    for (int r = 0; r < reps; r++) {
        image_free(work); work = image_clone(src);   /* same noisy input each rep */
        double t0 = now_ms();
        DENOISE_FN(work, mask, &p);                  /* seq | omp | cuda */
        double dt = now_ms() - t0;
        if (r == 0 || dt < best_ms) best_ms = dt;    /* best of N */
    }
    printf("impl=%s threads=%d ms=%.3f pixels=%.0f mpix_s=%.2f", ...);
    if (ref) printf(" psnr=%.4f mae=%.4f", metrics_psnr(work, ref), ...);
```

The bracket is around the denoiser only: PPM read, clone and write are outside it,
so I/O cannot flatten the curves.

# 9. Findings

1. **Parallelism can be bit-exact.** seq = omp = cuda, maxdiff 0 on 512^2..4096^2,
   window 5 and 15, odd size 1001x333, repeats. PSNR agrees to 4 decimals
   (32.2613 all three). compute-sanitizer: 0 errors.
2. **The OpenMP curve is non-monotonic, cause measured.** At 1024^2: 4T 11.0,
   8T 11.9, 14T 13.6, 20T 5.65 ms. `OMP_PLACES=cores` on a hybrid CPU gives 14
   places = 12 P-threads + 2 E-cores, so 6 E-cores idle; and `static` equal chunks
   make the slow E-cores set the wall time. Probe: 6 P-cores 7.33 ms, 12 with SMT
   7.09 (SMT buys nothing), 8 E-cores 20.13, 20 threads 4.60. Fix would be
   `schedule(dynamic)`; we report the measurement instead.
3. **The GPU is FP64-bound, not memory-bound - proven.** The textbook tiling fix
   (9x redundant loads -> shared memory) was implemented, verified and measured:
   +0.2%. Probes at 4096^2, same shape: detection in double 116.1 ms, in float
   4.83 ms (24x), integer 0.82 ms. Bandwidth actually used: 240 MB/s of 164 GB/s
   = 0.15%. The kernel is 27 double divides per pixel on a part with 1/64 FP64
   rate. So the GPU beats one CPU thread 4.4-5.2x but loses to 20 OpenMP threads
   at 4096^2 (124 vs 76 ms). Single precision is the lever - and the paper used it.
4. **Defaults came from a sweep.** 7x7 grid over d,m on photo_1024: peak d=0.5,
   m=4 -> 32.26 dB (interior maximum). Old defaults gave 25.20 dB. Same code,
   +7 dB from two constants.

# 10. The paper vs us

Sanchez, Vidal, Bataller, Arnal, "A Parallel Method for Impulsive Image Noise
Removal on Hybrid CPU/GPU Systems", ICCS 2013, Procedia CS 18:2504-2507 (4 pages).

It has: the peer-group + fuzzy metric detector, the detect/filter two-phase split,
OpenMP + CUDA implementations, speedups vs sequential, and **single precision**.

We took the algorithm and the two-phase structure. Differences:
- **double precision** instead of single, to get bit-identical output vs the CPU
  reference. Measured cost on this GPU: ~24x slower than float. That is the main
  engineering trade and it is priced.
- No memory optimisation, because it measurably does not pay here (finding 3) and
  the course scope bans most of it.
- d, m re-tuned by sweep rather than copied.

"Did you reproduce the paper?" -> "The algorithm and the qualitative CPU/GPU
result, yes. Not its numbers: we kept double where it uses single, so our GPU is
far slower than theirs would be on this card - and we measured that gap (24x)
instead of hiding it."

# 11. Numbers to remember

10% salt & pepper, d=0.5 m=4 n=3, median of 5 processes (ms):

    size      seq     omp4   omp20    cuda   cuda-kernel
    512^2    10.13    2.68    2.61    2.18      1.95
    1024^2   39.67   10.98    5.65    9.08      8.34
    2048^2  162.1    42.87   18.95   30.87     28.08
    4096^2  649.2   171.9    76.13  124.3     114.7

Efficiency at 4 OpenMP threads 0.90-0.95 (gate > 0.75). cuda vs one CPU thread
4.4-5.2x. PCIe 7.8%. Weak scaling +3.6%. Throughput flat at ~26 Mpix/s (seq).
Quality: 13.80 -> 32.26 dB (+18.46), MAE 0.91; gain peaks +18.5 dB near 12-15%
noise; clean image 96.7 dB; median fallback fires for 0.01% of corrupted pixels.
Hardware: RTX 4050 Laptop, sm_89, 20 SM, 70 W; i7-13700H 6P+8E = 14 cores / 20
threads; CUDA 13.4.

# 12. Likely questions

- *What does it do?* Removes impulse noise with a fuzzy peer-group detector, three
  implementations, byte-identical output.
- *Why two phases?* Phase 2 reads neighbours' mask -> whole mask first -> a global
  barrier. seq: loop end. omp: end of parallel for. cuda: second launch.
- *Why not `__syncthreads()`?* It syncs one block; neighbours may be in another.
- *Why round the grid up?* Images need not be multiples of 16 (1001x333); extra
  threads hit the guard.
- *Why 16x16?* 2-D matches the image, 256 threads = 8 warps, well under 1024.
- *Why cudaEvent?* Launches are async; a CPU clock times the launch, not the work.
- *Why separate H2D/kernel/D2H?* PCIe vs compute, different causes.
- *Why double if it costs 24x?* Bit-identical output vs the CPU. Priced, and single
  precision is named as the next lever.
- *Did you try shared memory?* Yes, written and measured, +0.2%, deleted. The
  kernel is FP64-bound.
- *Why is 14 threads slower than 20?* Hybrid cores + `OMP_PLACES=cores` placement
  + static chunks. Measured probes in finding 2.
- *Any race conditions?* No: one writer per output byte, no reductions or atomics.
- *How do you know it is correct?* maxdiff = 0 vs seq, `make check`, sanitizer.
- *What limits your speedup?* Not Amdahl - hardware: FP64 rate on the GPU, core
  heterogeneity on the CPU.
- *One more week?* Single-precision option, profiler counters, dynamic scheduling.

# 13. Glossary

- **cudaEvent** - GPU timestamp, the only correct way to time a kernel.
- **D2H / H2D** - device-to-host / host-to-device copy over PCIe.
- **divergence** - threads in a warp taking different branches; the warp runs both.
- **efficiency** - speedup / threads.
- **kernel** - here: a `__global__` function launched on the GPU.
- **mask** - one byte per pixel: 1 = corrupted, 0 = clean.
- **memory-bound / compute-bound** - limited by bandwidth / by arithmetic.
- **occupancy** - resident warps per SM relative to the maximum.
- **peer** - a window pixel within distance d of the centre.
- **PSNR** - 10 log10(255^2/MSE) dB.
- **SM** - streaming multiprocessor; 20 on this GPU.
- **stencil** - a neighbourhood operation, ours 3x3.
- **warp** - 32 CUDA threads executing in lockstep.
- **weak scaling** - fixed work per worker, growing problem.

# 14. Demo

    ssh gpu; cd ~/Projects/hpc-denoise
    bash scripts/demo.sh . build

Prints: noisy PSNR 13.80 -> three implementations with timings and the CUDA
H2D/kernel/D2H split -> PSNR 32.26 for all three -> `cmp` byte-identical ->
writes results/figures/edge_crop.png and impl_strip.png to show.

Where to point:

- algorithm itself .............. src/common/core.h
- "why two phases" .............. src/cuda/host.cu:87 and :89 (two launches)
- launch config, guard .......... src/cuda/host.cu:78, src/cuda/kernels.cu:74
- timing ........................ src/cuda/host.cu:78-107, main_denoise.c:105
- fair comparison ............... CMakeLists.txt DENOISE_FN + shared main()
- "no race" ..................... src/omp/denoise_omp.c:17 and :24
- defaults and why .............. src/common/params.h + results/06
- the negative result ........... results/04-cuda-variants.md section 4
