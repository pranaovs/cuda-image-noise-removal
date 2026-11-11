/* Host side of the GPU implementation: the four steps from lecture 2, plus timing.
 *
 *     cudaMalloc -> cudaMemcpy H2D -> kernel(s) -> cudaMemcpy D2H    (+ cudaFree)
 *
 * Two details matter for the numbers this code reports:
 *
 *  - Device buffers and cudaEvents are allocated once and reused across repetitions.
 *    cudaMalloc of 50 MB costs more than the kernels on a 4096^2 image, and a per-call
 *    allocation would show up in the measured milliseconds.
 *
 *  - Transfer time and kernel time are measured with cudaEvents and reported
 *    separately. A launch is asynchronous, so a CPU clock cannot time the kernel at
 *    all, and adding PCIe copies to kernel time would misattribute the GPU's speedup.
 *
 * The two kernel launches go to the default stream, so the second cannot start before
 * the first has finished: that ordering is the algorithm's phase barrier (the filter
 * kernel reads its neighbours' mask bytes). No explicit synchronization is needed
 * between them, and none is used.
 */
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"

/* __global__ prototypes live here rather than in kernels.cuh, which main_denoise.c
 * includes as plain C. These two declarations are the whole cross-file contract. */
__global__ void detectKernel(const unsigned char *in, unsigned char *mask, int w, int h,
                             DenoiseParams p);
__global__ void filterKernel(const unsigned char *in, const unsigned char *mask,
                             unsigned char *out, int w, int h, DenoiseParams p);

/* Every CUDA call can fail; naming the call that failed is the difference between a one
 * minute and a one hour debugging session. Exits rather than returning, because a failed
 * launch leaves no valid image to report on. */
#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__,          \
                    __LINE__);                                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static CudaTiming g_last = {0.0, 0.0, 0.0};

/* Reused device buffers, grown on demand and kept for the next repetition. */
static unsigned char *g_in = NULL, *g_out = NULL, *g_mask = NULL;
static size_t g_nbytes = 0, g_npx = 0;

/* Timestamps: 0 = before H2D, 1 = before the kernels, 2 = after them, 3 = after D2H. */
static cudaEvent_t g_ev[4];
static int g_ev_ready = 0;

/* (Re)allocates only when the image size changes, so a repeated run reuses the memory
 * and pays no allocation cost inside the measured region. */
static void buffers_reserve(size_t nbytes, size_t npx)
{
    if (nbytes == g_nbytes && npx == g_npx)
        return;
    if (g_in) {
        CUDA_CHECK(cudaFree(g_in));
        CUDA_CHECK(cudaFree(g_out));
        CUDA_CHECK(cudaFree(g_mask));
        g_in = g_out = NULL;
        g_mask = NULL;
    }
    CUDA_CHECK(cudaMalloc((void **)&g_in, nbytes));
    CUDA_CHECK(cudaMalloc((void **)&g_out, nbytes));
    CUDA_CHECK(cudaMalloc((void **)&g_mask, npx));
    g_nbytes = nbytes;
    g_npx = npx;
}

void denoise_cuda(Image *img, unsigned char *mask, const DenoiseParams *p, CudaTiming *t)
{
    const int w = img->w, h = img->h;
    const size_t nbytes = (size_t)w * (size_t)h * 3;
    (void)mask; /* the host mask is not needed: no D2H copy, see the end of this function */

    if (!g_ev_ready) {
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaEventCreate(&g_ev[i]));
        g_ev_ready = 1;
    }
    buffers_reserve(nbytes, (size_t)w * (size_t)h);

    /* 256 threads per block (16x16, a multiple of 32) and a grid rounded up so w and h
     * need not be multiples of 16; threads that overshoot the edge exit immediately. */
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    CUDA_CHECK(cudaEventRecord(g_ev[0]));
    CUDA_CHECK(cudaMemcpy(g_in, img->rgb, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(g_ev[1]));

    /* The launch boundary is the phase barrier: filterKernel reads mask bytes that
     * detectKernel writes, and a kernel cannot start before the previous one completes
     * on the same stream. */
    detectKernel<<<grid, block>>>(g_in, g_mask, w, h, *p);
    CUDA_CHECK(cudaGetLastError());
    filterKernel<<<grid, block>>>(g_in, g_mask, g_out, w, h, *p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(g_ev[2]));

    CUDA_CHECK(cudaMemcpy(img->rgb, g_out, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(g_ev[3]));
    CUDA_CHECK(cudaEventSynchronize(g_ev[3]));

    float h2d = 0.0f, kern = 0.0f, d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d, g_ev[0], g_ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&kern, g_ev[1], g_ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, g_ev[2], g_ev[3]));
    if (t) {
        t->h2d_ms = h2d;
        t->kernel_ms = kern;
        t->d2h_ms = d2h;
        g_last = *t;
    }
    /* The mask stays on the device: no host code reads it, and copying it back would
     * inflate d2h_ms with data nobody looks at. */
}

void denoise_cuda_impl(Image *img, unsigned char *mask, const DenoiseParams *p)
{
    CudaTiming t;
    denoise_cuda(img, mask, p, &t);
}

void denoise_cuda_last_timing(CudaTiming *t)
{
    if (t)
        *t = g_last;
}
