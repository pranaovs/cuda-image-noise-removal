/* Host side: the four steps from lecture 2 -
 *   cudaMalloc -> cudaMemcpy (H2D) -> kernel(s) -> cudaMemcpy (D2H)
 * plus cudaEvents, because a launch is asynchronous and the CPU clock cannot time it.
 *
 * Transfer time and kernel time are reported separately: if they were added into one
 * number, the PCIe copy would be charged to the GPU compute and every speedup claim
 * would be wrong.
 */
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"

/* The __global__ prototypes stay out of kernels.cuh because that header is included by
 * the plain-C main. These two lines are the whole cross-file contract. */
__global__ void detectKernel(const unsigned char *in, unsigned char *mask, int w, int h,
                             DenoiseParams p);
__global__ void filterKernel(const unsigned char *in, const unsigned char *mask,
                             unsigned char *out, int w, int h, DenoiseParams p);

/* Every CUDA call can fail; the cheapest debugging there is, is to say which one. */
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

/* Device buffers, grown on demand and kept for the next repetition: cudaMalloc of
 * ~100 MB costs more than the kernels and would show up in the reported ms. */
static unsigned char *g_in = NULL, *g_out = NULL, *g_mask = NULL;
static size_t g_nbytes = 0, g_npx = 0;

/* Timestamps: 0 = before H2D, 1 = before kernels, 2 = after kernels, 3 = after D2H. */
static cudaEvent_t g_ev[4];
static int g_ev_ready = 0;

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

    /* 256 threads per block (16x16, a multiple of 32); grid rounded up so w and h need
     * not be multiples of 16. Overshooting threads bounce off the guard in the kernel. */
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    CUDA_CHECK(cudaEventRecord(g_ev[0]));
    CUDA_CHECK(cudaMemcpy(g_in, img->rgb, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(g_ev[1]));

    /* Two launches on the default stream. The second one cannot start before the first
     * has finished, and that ordering is the phase barrier the algorithm needs. */
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
