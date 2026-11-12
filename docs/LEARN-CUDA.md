# CUDA revision: src/cuda

Exam revision. Assumes you have read the code once. Includes the reasoning
behind the design, not just the facts.

ASCII only.

---

## 0. Contents

```
   1  The pipeline
   2  The algorithm
   3  Why two kernels
   4  Indexing and bounds
   5  The three decisions worth defending
   6  The helpers
   7  dev_is_clean, walked through
   8  filterKernel, three paths
   9  host.cu
   10 kernels.cuh and CMakeLists
   11 File map and glossary
   12 Likely questions
```

---

## 1. The pipeline

Every CUDA program is this shape. `host.cu` is exactly this, plus timing.

```
   CPU RAM                                          GPU memory
   +-------------+                                  +-------------+
   | img->rgb    | --- cudaMemcpy H2D --------------->| g_in       |
   | (normal C)  |                                  |             |
   +-------------+                                  |             |
                                                    |  detectKernel
                                                    |  filterKernel
                                                    |             |
   +-------------+                                  |             |
   | img->rgb    | <-- cudaMemcpy D2H ----------------| g_out      |
   +-------------+                                  +-------------+
```

```
   1. cudaMalloc    ask the GPU for memory
   2. cudaMemcpy    copy the image up              (H2D = host to device)
   3. kernel<<<>>>  do the work
   4. cudaMemcpy    copy the result back           (D2H = device to host)
   (+ cudaFree)
```

A GPU cannot read normal C pointers, and normal C cannot read GPU memory. That
is the entire reason the copy steps exist.

---

## 2. The algorithm

From `src/common/params.h`:

```
   window = 3     side of the square neighbourhood (odd)
   d      = 0.5   fuzzy tolerance, 0..1
   m      = 4     required similar neighbours, not counting the centre
```

A window of 3 is a 3x3 box, 9 pixels, the pixel itself in the middle:

```
      +---+---+---+
      |   |   |   |
      +---+---+---+
      |   | P |   |
      +---+---+---+
      |   |   |   |
      +---+---+---+
```

### Similarity

Each pixel is three numbers (R, G, B), 0..255.

```
   one channel:  term = 1 - |a - b| / 255
                        identical  -> 1.0
                        255 apart  -> 0.0

   distance = 1 - (term_R * term_G * term_B)
```

The product means a pixel must match on all three channels to look similar.
Distance 0 = identical, 1 = maximally different.

### Phase 1, detection

```
   count = how many of the 9 pixels are at distance <= d, P included
   clean  if count >= m + 1
```

`m + 1` and not `m`: P always counts itself (when `dx = dy = 0` it compares
itself with itself, distance 0). So with `m = 4` you need 5 of 9 to agree.

Result: one byte per pixel into `mask[]`. 0 = clean, 1 = corrupt.

### Phase 2, filtering

```
   clean pixel     -> copy through untouched
   corrupt pixel   -> mean of the CLEAN neighbours only
   no clean
   neighbours      -> median of the whole window, per channel
```

Averaging in the corrupt neighbours would spread the noise instead of removing
it. Averaging only the clean ones is what preserves edges.

The median fallback needs every one of the 9 pixels corrupted, which happens
above roughly 30 percent noise density.

---

## 3. Why two kernels

The phase boundary is not a style choice, it is a hardware constraint.

`filterKernel` reads the mask bytes of neighbouring pixels. Those neighbours may
live in a different block. Threads in different blocks cannot synchronise.

```
   ONE KERNEL (wrong)                  TWO KERNELS (this code)
   +-------------------------+         +-------------------------+
   | detect work             |         | detectKernel            |
   | __syncthreads()         |         |   writes ALL of mask[]  |
   |   only orders my block  |         | (kernel ends = barrier) |
   | filter work             |         | filterKernel            |
   |   reads a neighbour's   |         |   reads ALL of mask[]   |
   |   mask byte that block  |         +-------------------------+
   |   B may not have written|
   +-------------------------+
```

`__syncthreads()` orders only the threads inside one 16x16 block. The only
grid-wide barrier available without cooperative launch is ending one kernel and
launching the next.

No explicit synchronisation is needed between the launches: both go to the
default stream, and a stream runs its work in order. The second kernel cannot
start before the first finishes. That ordering *is* the barrier.

---

## 4. Indexing and bounds

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

Read as: `blockIdx` = how many blocks came before me, `blockDim` = threads per
block, `threadIdx` = my slot in my block. Multiply the first two, add the third,
and you have a flat coordinate.

You never write a loop over pixels. You write the code for *one* pixel and the
launch makes a thread per pixel.

### The grid

```cuda
dim3 block(16, 16);                  // 256 threads
dim3 grid((w + 15) / 16, (h + 15) / 16);
```

`(w + 15) / 16` is integer division that rounds up. That is what makes
non-multiples of 16 work.

For `1024 x 1024`: 64 x 64 blocks, `64 * 64 * 256 = 1,048,576` threads, which is
exactly the pixel count.

For `1000 x 1000`: `(1000 + 15) / 16 = 63` blocks, `63 * 16 = 1008` threads.
Total threads `63 * 63 * 256 = 1,016,064` against 1,000,000 pixels, so about
16,000 threads have no pixel.

Those threads are real. They run the kernel. Nothing tells them they are
useless. Hence the guard at the top of both kernels:

```cuda
if (x >= w || y >= h)
    return;
```

Leaving this out is the classic CUDA bug: the thread writes one pixel outside
the buffer and corrupts whatever was next to it.

### Byte offset

```cuda
size_t i = (size_t)y * (size_t)w + (size_t)x;   // pixel index
in[3 * i + 0], in[3 * i + 1], in[3 * i + 2]     // R, G, B
```

The 3 is because there are three bytes per pixel. The `size_t` casts stop a
`1024 * 1024` pixel index overflowing a 32-bit int on large images.

---

## 5. The three decisions worth defending

These are the ones a grader will poke at. Each has a measurement behind it.

### threadIdx.x is the column, not the row

`in + 3 * (y * w + x)`: neighbouring `x` are neighbouring bytes in memory.
Neighbouring threads in a block have neighbouring `threadIdx.x`.

So 32 consecutive threads (one warp) read 32 consecutive pixels, one row, which
the hardware can fetch as a few large transactions. That is a coalesced access.

Swap x and y and the output is **identical**, but each warp now reads a column:
one byte every `3 * w` bytes, scattered across memory. Same result, many more
memory transactions.

### No shared memory tile

A 3x3 window overlaps 8/9 with its neighbour, so a shared tile is the first
optimisation anyone tries in a stencil kernel. It was written, measured, and
made the kernel **0.2 percent slower**. Deleted rather than kept as dead code.

Why: the kernel is limited by double-precision arithmetic, not memory traffic.

### double, and no -use_fast_math

The metric uses `double`. The filter uses `int` and integer division.

The correctness gate for this project is byte-identical output against
`denoise_seq`, checked by `scripts/check.sh`. So:

- `double` for the metric, same as the CPU
- integer mean, not float
- no `-use_fast_math`, which would change rounding

The cost is real: this consumer GPU runs FP64 at about 1/64 of its FP32 rate, and
that single expression, evaluated 9 times per pixel, is essentially the GPU's
whole running time. Accepted deliberately, and worth saying out loud.

---

## 6. The helpers

`__device__` means callable only from GPU code. Not launched, just called, and
inlined by the compiler.

### dev_clamp: edge replication

```cuda
__device__ int dev_clamp(int v, int hi)
{
    return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}
```

A pixel at column 0, radius 1, asks for column -1. That does not exist. Clamp
turns -1 into 0, so the window repeats the border instead of reading outside:

```
       -1   0   1
      +---+---+---+
row 0 | A | A | B |     A = pixel (0,0), B = pixel (0,1)
      +---+---+---+
```

The window never shrinks. It is always 9 samples, so `m + 1` means the same
thing at the edge as in the middle. Corners work for free: `(0,0)` with
`(-1,-1)` clamps to `(0,0)` and that corner pixel votes 4 times.

Alternatives, and why not: skipping out-of-range pixels shrinks the window to 4
samples at a corner and changes the threshold's meaning; zero padding invents
black pixels that pull the average down.

This matches `img_clamp()` in `image.h` (per the file's own comment), which is
what keeps GPU and CPU byte-identical.

### dev_chan_term: one channel's similarity

```cuda
__device__ double dev_chan_term(int a, int b)
{
    int d = a - b;
    if (d < 0)
        d = -d;
    return 1.0 - (double)d / 255.0;
}
```

The `(double)` cast matters. Without it, `1 - d/255` is integer arithmetic and
the result is always 0 or 1, losing all the fuzziness the filter depends on.

### dev_sort_u8: insertion sort

Device code has no `qsort`. Only used by the median fallback, at most 225
elements, so speed is irrelevant.

---

## 7. dev_is_clean, walked through

Phase 1 for a single pixel. This is the file's centre of gravity.

```cuda
    int r = p.window / 2;                                  // window 3 -> r = 1
    int count = 0;
    const unsigned char *c = in + 3 * ((size_t)y * (size_t)w + (size_t)x);
```

`r` is the radius, so the loops run `dy, dx` from -1 to +1: 9 iterations.
`c` points at the centre pixel.

```cuda
    for (int dy = -r; dy <= r; dy++) {
        int yy = dev_clamp(y + dy, h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = dev_clamp(x + dx, w);
            const unsigned char *q = in + 3 * ((size_t)yy * (size_t)w + (size_t)xx);
            double dist = 1.0 - dev_chan_term(c[0], q[0]) * dev_chan_term(c[1], q[1]) *
                                    dev_chan_term(c[2], q[2]);
            if (dist <= p.d)
                count++;
        }
    }
    return count >= p.m + 1;
```

`c[0], c[1], c[2]` are R, G, B of the centre. `q` is the neighbour. When
`dy = dx = 0`, `q == c`, distance 0, `count++`. That single iteration is the
centre's self-vote, and the reason the return is `m + 1`.

Cost: 9 iterations x 3 double divisions. Per pixel. A million pixels. The
comment is blunt about it, and it explains why shared memory did not help.

---

## 8. filterKernel, three paths

### Path 1: clean, copy through

```cuda
    if (!mask[i]) {
        out[3 * i + 0] = in[3 * i + 0];
        out[3 * i + 1] = in[3 * i + 1];
        out[3 * i + 2] = in[3 * i + 2];
        return;
    }
```

Three byte copies and out. `mask[i]` is 0 for clean.

### Path 2: average the clean neighbours

```cuda
            if (mask[j])
                continue;          // a corrupt neighbour must not pull the average
            sum[0] += (int)in[3 * j + 0];
            ...
            count++;
```

Same loop shape as `dev_is_clean`, but the test is now `mask[j]` instead of a
distance. Sums are `int`, and the write is `sum[c] / count`, integer division,
to stay bit-identical with the CPU.

### Path 3: median fallback

```cuda
    unsigned char vals[PARAMS_MAX_WINDOW * PARAMS_MAX_WINDOW];
    for (int c = 0; c < 3; c++) {
        ... fill vals[] with one channel of the whole window ...
        dev_sort_u8(vals, n);
        out[3 * i + c] = vals[n / 2];
    }
```

Only when `count == 0`. `PARAMS_MAX_WINDOW` is 15, hence the 225-element bound.

### On warp divergence

A warp is 32 threads in lockstep. A warp mixing clean and corrupt pixels runs
both branches. The file argues this is cheap because the clean branch is three
byte copies and a return, so the cost lands in a short tail rather than in the
9-iteration loop.

---

## 9. host.cu

### The launch

```cuda
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    cudaMemcpy(g_in, img->rgb, nbytes, cudaMemcpyHostToDevice);
    detectKernel<<<grid, block>>>(g_in, g_mask, w, h, *p);
    cudaGetLastError();
    filterKernel<<<grid, block>>>(g_in, g_mask, g_out, w, h, *p);
    cudaGetLastError();
    cudaMemcpy(img->rgb, g_out, nbytes, cudaMemcpyDeviceToHost);
```

`<<<grid, block>>>` is CUDA's launch syntax. `cudaGetLastError()` after each
launch catches bad grid sizes and bad pointers; a launch can otherwise fail
silently.

### Reused buffers

```cuda
static unsigned char *g_in = NULL, *g_out = NULL, *g_mask = NULL;
```

The benchmark repeats the call many times. `cudaMalloc` on a 50 MB buffer costs
more than the kernels do, so allocating inside the timed region would show up in
the measured milliseconds. Allocate once, reuse. `buffers_reserve` reallocates
only when the image size changes.

### CUDA_CHECK

```cuda
#define CUDA_CHECK(call) ...
```

Wraps every CUDA call, prints the error string plus file and line, then
`exit(1)`. Naming the call that failed is the difference between a one-minute
and a one-hour debug session. It exits rather than returns because a failed
launch leaves no valid image to report.

### Why cudaEvent, not a clock

A launch returns immediately; the CPU does not wait. Timing with a CPU clock
would measure the launch (microseconds), not the work.

```
   ev0        ev1              ev2        ev3
    |          |                |          |
    |   H2D    |  both kernels  |   D2H    |
    |<-------->|<-------------->|<-------->|
      h2d_ms       kernel_ms      d2h_ms
```

Three numbers, never added together. Adding the PCIe copies into `kernel_ms`
would make the reported speedup meaningless at small image sizes, where the copy
dominates.

### Why the mask never comes back

Nothing on the host reads it. Copying a megabyte back would inflate `d2h_ms`
with data nobody looks at. That is what `(void)mask;` is telling you.

---

## 10. kernels.cuh and CMakeLists

`kernels.cuh` is included by `src/app/main_denoise.c`, which is **plain C**. So
nothing CUDA may appear in it: no `__global__`, no device types. Everything is
wrapped in `extern "C"` so the C file links against the C++-compiled objects.

Consequence: the `__global__` prototypes live in `host.cu`, not in the header.
That is the whole cross-file contract.

Three functions cross the boundary:

```
   denoise_cuda             with a CudaTiming out-parameter
   denoise_cuda_impl        same, no timing. This is what DENOISE_FN points at
   denoise_cuda_last_timing fills t from the last timed call
```

`CMakeLists.txt`:

- `enable_language(CUDA)` is only parsed with `-DENABLE_CUDA=ON`, so a box with
  no `nvcc` still builds everything CPU-side
- `CMAKE_CUDA_ARCHITECTURES native` on CUDA 11.5+, else let CMake pick
- reuses `src/app/main_denoise.c` verbatim, so `-d/-m/-n/-r/-q` parsing and the
  repeat loop cannot drift between seq, omp and cuda. Only `DENOISE_FN` and
  `DENOISE_HAS_CUDA` change
- no `-use_fast_math`, for the bit-exactness reason above
- output to `build/denoise_cuda`

---

## 11. File map and glossary

```
   kernels.cu     device code: 2 kernels + 3 helpers
   kernels.cuh    C interface only, no CUDA types
   host.cu        malloc / copy / launch / timing
   CMakeLists     -DENABLE_CUDA=ON gate
```

```
   kernel        a __global__ function launched on many threads
   __global__    marks a kernel, called from host code
   __device__    callable only from GPU code
   warp          32 threads that execute together (hardware unit)
   block         256 threads here; can synchronise with __syncthreads()
   grid          all blocks in one launch
   stream        ordered queue of GPU work; default stream = in order
   H2D / D2H     host-to-device / device-to-host copy over PCIe
   coalesced     neighbouring threads reading neighbouring memory
   divergence    threads in a warp taking different branches
```

What the file deliberately does not use: shared memory tiling, `-use_fast_math`,
`__ldg`, `__restrict__`, vector types, templates, dynamic parallelism, explicit
stream synchronisation.

---

## 12. Likely questions

1. Why two kernels? Grid-wide barrier. `__syncthreads()` is per block only.
2. Why the `x >= w` return? Grid is rounded up; edge threads have no pixel.
3. Why 16x16 blocks? 256 threads, multiple of 32 (warp size), good occupancy.
4. Why `double` for the metric but `int` for the filter? Byte-identical output
   vs the CPU is the correctness gate.
5. Why no shared memory? Measured 0.2 percent slower; arithmetic-bound.
6. Why `cudaEvent` and not a CPU timer? Launches are asynchronous.
7. Why no `-use_fast_math`? It would change rounding and break bit-exactness.
8. Why is the threshold `m + 1`? The centre counts itself.
9. Why is `threadIdx.x` the column? Coalesced memory access.
10. Why three timings instead of one? Transfer and compute are different
    resources; merging them hides the GPU's actual speedup.
11. Why doesn't the mask come back to the host? Nothing reads it, and it would
    inflate `d2h_ms`.
12. When does the median fallback run? When every pixel in the window is
    corrupt, roughly 30 percent noise and above.

---

## Caveat

`src/common/core.h` and `src/common/image.h` were not read. The claims that
`dev_is_clean` matches `core.h` and `dev_clamp` matches `img_clamp` come from
comments inside `src/cuda/`. If you want them proven rather than asserted, read
those two headers.
