#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "image.h"
#include "params.h"

/* C-facing declarations only: this header is included by src/app/main_denoise.c,
 * which is plain C (no CUDA types may leak here). The __global__ prototypes live
 * in host.cu, the only other translation unit that needs them. */

#ifdef __cplusplus
extern "C" {
#endif

/* Phase split of the most recent denoise_cuda() call. Transfer time is not kernel
 * time: h2d_ms and d2h_ms are PCIe, kernel_ms is the two launches together. */
typedef struct {
    double h2d_ms, kernel_ms, d2h_ms;
} CudaTiming;

void denoise_cuda(Image *img, unsigned char *mask, const DenoiseParams *p, CudaTiming *t);

/* Same work without the timing out-parameter: this is what DENOISE_FN points at. */
void denoise_cuda_impl(Image *img, unsigned char *mask, const DenoiseParams *p);

/* Timings of the most recent denoise_cuda() call, for the report line. */
void denoise_cuda_last_timing(CudaTiming *t);

#ifdef __cplusplus
}
#endif

#endif
