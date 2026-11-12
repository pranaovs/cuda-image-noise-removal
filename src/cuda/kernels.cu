/* GPU kernels for the two-phase filter: one thread per pixel, plain global
 * loads.
 *
 * This is a transcription of src/common/core.h into device code - same fuzzy
 * distance, same clamped window, same "the centre counts itself" rule, same
 * integer mean and the same median fallback. The two files must stay in
 * agreement because the correctness gate is byte-identical output against
 * denoise_seq (scripts/check.sh).
 *
 * Design decisions worth defending:
 *
 *  - Two kernels, not one. The launch boundary is the algorithm's phase
 * barrier: the filter kernel reads its neighbours' mask bytes, so the whole
 * mask must be finished first. __syncthreads() cannot do this, it only orders
 * one block; a kernel boundary is the only grid-wide barrier available without
 * cooperative launch.
 *
 *  - One thread per pixel, no shared memory. The 3x3 windows overlap 8/9, so a
 * shared tile is the first optimization anyone tries - it was written, measured
 * and deleted the kernel by +0.2%, because the kernel is limited by
 * double-precision arithmetic, not by memory traffic
 * (docs/results/04-cuda-variants.md has the probes). The tiled version was
 * deleted rather than kept as dead code.
 *
 *  - float for the fuzzy metric, integer for the filter. The metric is where
 * the course's precision discussion lives: this consumer GPU runs FP64 at 1/64
 * of its FP32 rate, so a double version of this one expression cost 116 ms at
 * 4096^2 versus 4.8 ms in float - it was essentially the GPU's whole running
 * time. float puts the GPU ahead of the CPU, at the price of no longer being
 * byte-identical to it (scripts/check.sh compares against a maxdiff tolerance
 * instead of cmp).
 *
 *    The filter stays integer on both sides, so the only divergence is a
 * borderline pixel whose fuzzy distance lands within float rounding of the
 * tolerance p.d.
 *
 * CUDA surface used, all of it from the course: __global__, __device__,
 * threadIdx, blockIdx, blockDim, 16x16 blocks (256 threads, a multiple of 32),
 * and cudaGetLastError after each launch. No templates, no __ldg, no
 * __restrict__, no vector types, no dynamic parallelism, no libraries.
 */
#include <cuda_runtime.h>

#include "kernels.cuh"

/* Edge-replicating clamp, identical to img_clamp() in image.h: a window that
 * leaves the image repeats the border row/column instead of being cut short. */
__device__ int dev_clamp(int v, int hi) {
  return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}

/* One channel of the fuzzy metric: equal channels give 1.0, 255 apart gives
 * 0.0. Same formula as fuzzy_dist() in core.h, in float instead of double. */
__device__ float dev_chan_term(int a, int b) {
  int d = a - b;
  if (d < 0)
    d = -d;
  return 1.0f - (float)d / 255.0f;
}

/* Phase 1 for one pixel: how many window pixels are within d of the centre, the
 * centre included. Clean means at least m+1 of them, i.e. m peers besides
 * itself.
 *
 * Cost: window^2 (9 for the default 3x3) cube-root-free distance evaluations,
 * each one a product of three double divisions. This is the GPU's whole running
 * time. */
__device__ int dev_is_clean(const unsigned char *in, int w, int h, int x, int y,
                            DenoiseParams p) {
  int r = p.window / 2;
  int count = 0;
  const unsigned char *c = in + 3 * ((size_t)y * (size_t)w + (size_t)x);
  for (int dy = -r; dy <= r; dy++) {
    int yy = dev_clamp(y + dy, h);
    for (int dx = -r; dx <= r; dx++) {
      int xx = dev_clamp(x + dx, w);
      const unsigned char *q = in + 3 * ((size_t)yy * (size_t)w + (size_t)xx);
      float dist = 1.0f - dev_chan_term(c[0], q[0]) *
                              dev_chan_term(c[1], q[1]) *
                              dev_chan_term(c[2], q[2]);
      if (dist <= p.d)
        count++;
    }
  }
  return count >= p.m + 1;
}

/* Insertion sort, used only by the rare median fallback (at most 225 elements).
 * Device code has no qsort, and the per-pixel cost is irrelevant because this
 * branch needs a window in which every single pixel is corrupted. */
__device__ void dev_sort_u8(unsigned char *v, int n) {
  for (int i = 1; i < n; i++) {
    unsigned char key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) {
      v[j + 1] = v[j];
      j--;
    }
    v[j + 1] = key;
  }
}

/* Kernel 1 - detection. Writes one byte per pixel: 1 = corrupted, 0 = clean.
 *
 * The grid is rounded up to whole blocks, so threads past the right/bottom edge
 * exist and must return before touching memory. 2-D indexing is deliberate:
 * threadIdx.x is the column, so consecutive threads read consecutive bytes in a
 * row and the loads coalesce into full 128-byte lines. Swapping x and y here
 * would multiply the memory transactions per warp without changing a single
 * output value. */
__global__ void detectKernel(const unsigned char *in, unsigned char *mask,
                             int w, int h, DenoiseParams p) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return; /* grid is rounded up, so the last block can hang over the edge */
  mask[(size_t)y * (size_t)w + (size_t)x] =
      dev_is_clean(in, w, h, x, y, p) ? 0 : 1;
}

/* Kernel 2 - filtering. Clean pixels are copied through unchanged; corrupted
 * ones get the mean of their *clean* neighbours only, which is what preserves
 * edges instead of blurring them. When every neighbour is corrupted (possible
 * above roughly 30% noise density) it falls back to the per-channel median of
 * the whole window.
 *
 * Divergence: a warp whose 32 pixels mix clean and corrupted runs both paths.
 * The clean path is three byte copies and a return, so the cost lands in the
 * tail rather than in the 9-iteration window loop; that is why the branch is
 * cheap here.
 *
 * Reads mask[] for the whole window, which is only valid because the detect
 * kernel has finished - that ordering is the launch boundary in host.cu, not a
 * coincidence. */
__global__ void filterKernel(const unsigned char *in, const unsigned char *mask,
                             unsigned char *out, int w, int h,
                             DenoiseParams p) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  const int r = p.window / 2;
  size_t i = (size_t)y * (size_t)w + (size_t)x;
  if (!mask[i]) {
    out[3 * i + 0] = in[3 * i + 0];
    out[3 * i + 1] = in[3 * i + 1];
    out[3 * i + 2] = in[3 * i + 2];
    return;
  }

  int sum[3] = {0, 0, 0};
  int count = 0;
  for (int dy = -r; dy <= r; dy++) {
    int yy = dev_clamp(y + dy, h);
    for (int dx = -r; dx <= r; dx++) {
      int xx = dev_clamp(x + dx, w);
      size_t j = (size_t)yy * (size_t)w + (size_t)xx;
      if (mask[j])
        continue; /* a corrupted neighbour must not pull the average */
      sum[0] += (int)in[3 * j + 0];
      sum[1] += (int)in[3 * j + 1];
      sum[2] += (int)in[3 * j + 2];
      count++;
    }
  }

  if (count > 0) {
    out[3 * i + 0] = (unsigned char)(sum[0] / count);
    out[3 * i + 1] = (unsigned char)(sum[1] / count);
    out[3 * i + 2] = (unsigned char)(sum[2] / count);
    return;
  }

  unsigned char vals[PARAMS_MAX_WINDOW * PARAMS_MAX_WINDOW];
  for (int c = 0; c < 3; c++) {
    int n = 0;
    for (int dy = -r; dy <= r; dy++) {
      int yy = dev_clamp(y + dy, h);
      for (int dx = -r; dx <= r; dx++) {
        int xx = dev_clamp(x + dx, w);
        vals[n++] = in[3 * ((size_t)yy * (size_t)w + (size_t)xx) + c];
      }
    }
    dev_sort_u8(vals, n);
    out[3 * i + c] = vals[n / 2];
  }
}
