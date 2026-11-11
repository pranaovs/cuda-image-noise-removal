/* One main() for every denoiser binary. The only build-time difference is which
 * implementation function is linked in, so seq/omp/cuda share argument handling
 * and timing placement exactly - otherwise a speedup could be an artefact.
 *
 * Compile with: -DDENOISE_FN=denoise_seq_impl -DIMPL_NAME='"seq"'
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DENOISE_HAS_OMP
#include <omp.h>
#endif

#include "image.h"
#include "metrics.h"
#include "params.h"

#ifdef DENOISE_HAS_CUDA
#include "kernels.cuh" /* C-only header: adds the h2d/kernel/d2h split to the report */
#endif

#ifndef DENOISE_FN
#error "DENOISE_FN must name the implementation function"
#endif
#ifndef IMPL_NAME
#error "IMPL_NAME must be a quoted string naming the implementation"
#endif

void DENOISE_FN(Image *img, unsigned char *mask, const DenoiseParams *p);

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s in.ppm out.ppm [-d 0.5] [-m 4] [-n 3] [-r 5] [-q clean.ppm]\n"
            "  -d  fuzzy tolerance (0..1)\n"
            "  -m  cardinality threshold\n"
            "  -n  window side (odd)\n"
            "  -r  repetitions, best time reported (default 5)\n"
            "  -q  clean reference; adds psnr/mae to the report line\n",
            prog);
    exit(2);
}

int main(int argc, char **argv)
{
    if (argc < 3)
        usage(argv[0]);

    const char *in_path = argv[1];
    const char *out_path = argv[2];
    const char *ref_path = NULL;
    int reps = 5;

    DenoiseParams p;
    params_default(&p);

    for (int i = 3; i < argc; i++) {
        if (params_flag(argc, argv, &i, &p))
            continue;
        if (strcmp(argv[i], "-r") == 0) {
            if (++i < argc)
                reps = atoi(argv[i]);
            continue;
        }
        if (strcmp(argv[i], "-q") == 0) {
            if (++i < argc)
                ref_path = argv[i];
            continue;
        }
        fprintf(stderr, "unknown argument: %s\n", argv[i]);
        usage(argv[0]);
    }
    params_require_valid(&p);
    if (reps < 1)
        reps = 1;

    Image *src = image_read(in_path);
    Image *ref = ref_path ? image_read(ref_path) : NULL;
    if (ref && (ref->w != src->w || ref->h != src->h)) {
        fprintf(stderr, "reference image has different dimensions\n");
        return 2;
    }

    unsigned char *mask = malloc((size_t)src->w * (size_t)src->h);
    if (!mask) {
        fprintf(stderr, "out of memory for mask\n");
        return 1;
    }

    Image *work = NULL;
    double best_ms = 0.0;
#ifdef DENOISE_HAS_CUDA
    CudaTiming cur = {0.0, 0.0, 0.0}, best_t = {0.0, 0.0, 0.0};
#endif
    for (int r = 0; r < reps; r++) {
        /* every repetition starts from the pristine noisy image */
        image_free(work);
        work = image_clone(src);

        double t0 = now_ms();
        DENOISE_FN(work, mask, &p);
        double dt = now_ms() - t0;
#ifdef DENOISE_HAS_CUDA
        denoise_cuda_last_timing(&cur);
#endif

        if (r == 0 || dt < best_ms) {
            best_ms = dt;
#ifdef DENOISE_HAS_CUDA
            best_t = cur; /* report the split of the same run that gave best ms */
#endif
        }
    }

    if (image_write(work, out_path) != 0)
        return 1;

    double pixels = (double)work->w * (double)work->h;
    /* threads= is in the report line of every binary, so the CSV schema is the same
     * whether or not the target was compiled with OpenMP. */
#ifdef DENOISE_HAS_OMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif

    printf("impl=%s threads=%d ms=%.3f pixels=%.0f mpix_s=%.2f", IMPL_NAME, threads, best_ms,
           pixels, pixels / (best_ms * 1e3));
#ifdef DENOISE_HAS_CUDA
    printf(" h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f", best_t.h2d_ms, best_t.kernel_ms,
           best_t.d2h_ms);
#endif
    if (ref) {
        /* 4 decimals: phase 06 proves quality is implementation-independent, which
         * needs more resolution than 2 decimals. */
        printf(" psnr=%.4f mae=%.4f", metrics_psnr(work, ref), metrics_mae(work, ref));
    }
    printf("\n");

    image_free(src);
    image_free(work);
    image_free(ref);
    free(mask);
    return 0;
}
