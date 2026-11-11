/* The denoiser on the GPU: two kernels, one thread per pixel, plain global loads.
 *
 * CUDA surface used - all of it from the course:
 *   __global__, __device__, threadIdx/blockIdx/blockDim, __syncthreads (not needed
 *   here), cudaGetLastError. No shared memory, no templates, no vector types, no
 *   __ldg, no __restrict__, no dynamic parallelism. Pixels are plain interleaved RGB
 *   bytes, the same layout the CPU uses, so both read the same array the same way.
 *
 * Two kernels, not one, because the launch boundary is the phase barrier: the filter
 * kernel reads the *neighbours'* mask bytes, so the whole mask must be finished first
 * and __syncthreads() could only order one block.
 *
 * Every expression is transcribed from core.h on purpose - same fuzzy distance, same
 * clamp-to-edge, same "centre counts itself" rule, same integer mean, same local
 * median fallback. If this file and core.h drift apart the symptom is maxdiff > 0
 * against denoise_seq.
 */
#include <cuda_runtime.h>

#include "kernels.cuh"

/* Keep a coordinate inside the image by replicating the edge, like core.h does. */
__device__ int dev_clamp(int v, int hi)
{
    return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}

/* One channel of the fuzzy metric: identical channels -> 1.0, 255 apart -> 0.0.
 * The distance is 1 - (product over the three channels). */
__device__ double dev_chan_term(int a, int b)
{
    int d = a - b;
    if (d < 0)
        d = -d;
    return 1.0 - (double)d / 255.0;
}

/* Phase 1 for one pixel: count how many window pixels are within d of the centre,
 * the centre included, and call it clean when there are at least m + 1 of them. */
__device__ int dev_is_clean(const unsigned char *in, int w, int h, int x, int y, DenoiseParams p)
{
    int r = p.window / 2;
    int count = 0;
    const unsigned char *c = in + 3 * ((size_t)y * (size_t)w + (size_t)x);
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
}

/* Insertion sort, used only by the rare median fallback (n <= 225). */
__device__ void dev_sort_u8(unsigned char *v, int n)
{
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

/* Kernel 1 - detection. Writes the mask: 1 = corrupted, 0 = clean. */
__global__ void detectKernel(const unsigned char *in, unsigned char *mask, int w, int h,
                             DenoiseParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return; /* grid is rounded up, so the last block can hang over the edge */
    mask[(size_t)y * (size_t)w + (size_t)x] = dev_is_clean(in, w, h, x, y, p) ? 0 : 1;
}

/* Kernel 2 - filtering. Clean pixels are copied through; corrupted ones get the mean
 * of their *clean* neighbours only, which is what keeps edges instead of blurring
 * them. If every neighbour is corrupted (high noise density) fall back to the median
 * of the whole window. */
__global__ void filterKernel(const unsigned char *in, const unsigned char *mask,
                             unsigned char *out, int w, int h, DenoiseParams p)
{
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
