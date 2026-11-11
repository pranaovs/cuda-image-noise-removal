#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "image.h"
#include "params.h"

/* C-callable interface to the GPU implementation.
 *
 * This header is included by src/app/main_denoise.c, which is plain C, so nothing CUDA
 * may appear here - no __global__, no device types. The kernel prototypes stay private
 * to host.cu; the only contract across the boundary is these three functions. */

#ifdef __cplusplus
extern "C" {
#endif

/* Where the time went in the most recent denoise_cuda() call. The three parts are kept
 * apart because the kernel is the GPU's work while h2d/d2h are PCIe copies; adding them
 * into one number would charge the transfer to the compute and make the speedup
 * meaningless at small image sizes. */
typedef struct {
    double h2d_ms, kernel_ms, d2h_ms;
} CudaTiming;

/* Denoises img in place. mask is accepted for signature compatibility with the CPU
 * implementations and is not used: the corruption mask never leaves the device. */
void denoise_cuda(Image *img, unsigned char *mask, const DenoiseParams *p, CudaTiming *t);

/* Same, without the timing out-parameter: this is what DENOISE_FN points at. */
void denoise_cuda_impl(Image *img, unsigned char *mask, const DenoiseParams *p);

/* Fills t with the split of the last timed call, so main can report the run that
 * actually produced the best ms. */
void denoise_cuda_last_timing(CudaTiming *t);

#ifdef __cplusplus
}
#endif

#endif
