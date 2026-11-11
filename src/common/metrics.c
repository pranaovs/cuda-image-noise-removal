#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_size(const Image *a, const Image *b)
{
    if (a->w != b->w || a->h != b->h) {
        fprintf(stderr, "metrics: size mismatch %dx%d vs %dx%d\n", a->w, a->h, b->w, b->h);
        exit(1);
    }
}

double metrics_mae(const Image *a, const Image *b)
{
    check_size(a, b);
    size_t n = (size_t)a->w * (size_t)a->h * 3;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++)
        sum += abs((int)a->rgb[i] - (int)b->rgb[i]);
    return sum / (double)n;
}

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
