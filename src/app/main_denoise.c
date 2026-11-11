/* One main() shared by all three denoiser binaries.
 *
 * The build links exactly one implementation into this file - denoise_seq_impl,
 * denoise_omp_impl or denoise_cuda_impl - selected by -DDENOISE_FN, so argument
 * handling, the repetition loop and the placement of the timer are literally the same
 * code for every implementation. If each one had its own main, a difference in where
 * the clock starts or stops could fake a speedup; this removes that whole class of
 * doubt, and the only thing that differs between the binaries is the function called.
 *
 * The output line is machine-readable on purpose - scripts/bench.sh greps it into a CSV,
 * so a new measurement is a new row and the plots stay reproducible:
 *
 *   impl=omp threads=8 ms=18.717 pixels=1048576 mpix_s=56.02 psnr=32.2613 mae=0.9100
 *
 * Reporting rules: `ms` is the best of -r repetitions (each starting from a pristine
 * copy of the noisy input), `pixels` is w*h, and the CUDA binary adds the
 * h2d/kernel/d2h split so the kernel time can be quoted without the PCIe copies.
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

/* Linked in from denoise_<impl>.c/.cu - see the CMake target for each binary. */
void DENOISE_FN(Image *img, unsigned char *mask, const DenoiseParams *p);

/* CLOCK_MONOTONIC: unaffected by wall-clock adjustments, which a timed loop needs. */
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

    /* One mask buffer for all repetitions: it is overwritten from scratch every time,
     * so nothing carries over between runs. */
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
    /* threads= is reported by every binary, so the CSV schema does not depend on
     * whether this target was compiled with OpenMP. */
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
        /* 4 decimals: the quality sweep compares implementations that must agree to the
         * last digit, which 2 decimals would hide. */
        printf(" psnr=%.4f mae=%.4f", metrics_psnr(work, ref), metrics_mae(work, ref));
    }
    printf("\n");

    image_free(src);
    image_free(work);
    image_free(ref);
    free(mask);
    return 0;
}
