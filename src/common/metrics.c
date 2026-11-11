/* Error metrics against the clean reference image.
 *
 * These are the numbers the report quotes, so they are defined once here and used by
 * imgcmp, the denoisers' -q flag and the quality sweep. All of them iterate the raster
 * in linear order: no 2-D indexing, no window, O(pixels) time and O(1) memory.
 */
#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Comparing images of different sizes is a bug in the caller, and silently averaging
 * over the shorter one would hide it, so it aborts. */
static void check_size(const Image *a, const Image *b)
{
    if (a->w != b->w || a->h != b->h) {
        fprintf(stderr, "metrics: size mismatch %dx%d vs %dx%d\n", a->w, a->h, b->w, b->h);
        exit(1);
    }
}

/* Mean absolute error per channel, in 0..255. Cheaper and easier to read than PSNR;
 * it exposes large single-pixel errors better because it does not square them. */
double metrics_mae(const Image *a, const Image *b)
{
    check_size(a, b);
    size_t n = (size_t)a->w * (size_t)a->h * 3;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += abs((int)a->rgb[i] - (int)b->rgb[i]);
    return sum / (double)n;
}

/* Largest single-channel difference, 0..255. This is the equivalence test between
 * implementations: maxdiff == 0 means the outputs are byte-identical, so no tolerance
 * and no "close enough" argument is needed. */
int metrics_max_diff(const Image *a, const Image *b)
{
    check_size(a, b);
    size_t n = (size_t)a->w * (size_t)a->h * 3;
    int worst = 0;
    for (size_t i = 0; i < n; i++) {
        int d = abs((int)a->rgb[i] - (int)b->rgb[i]);
        if (d > worst)
            worst = d;
    }
    return worst;
}

int metrics_identical(const Image *a, const Image *b)
{
    check_size(a, b);
    return memcmp(a->rgb, b->rgb, (size_t)a->w * (size_t)a->h * 3) == 0;
}

/* PSNR in dB: 10*log10(255^2 / MSE) over all channels. 8-bit peak = 255. Identical
 * images give infinite dB, reported as inf rather than as a made-up cap. */
double metrics_psnr(const Image *a, const Image *b)
{
    check_size(a, b);
    size_t n = (size_t)a->w * (size_t)a->h * 3;
    double sse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a->rgb[i] - (double)b->rgb[i];
        sse += d * d;
    }
    if (sse == 0.0)
        return INFINITY;
    return 10.0 * log10((255.0 * 255.0) / (sse / (double)n));
}
