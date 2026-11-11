#ifndef CORE_H
#define CORE_H

#include <string.h>

#include "image.h"
#include "params.h"

/* The algorithm, no parallelism. Both denoise_*_impl and the CUDA kernels obey
 * exactly these semantics; only the scheduling differs. */

/* Bounded fuzzy distance: 1 - prod_c (1 - |a_c-b_c|/255).
 * identical -> 0.0, maximally different -> 1.0. */
static inline double fuzzy_dist(const unsigned char a[3], const unsigned char b[3])
{
    double prod = 1.0;
    for (int c = 0; c < 3; c++) {
        int d = (int)a[c] - (int)b[c];
        if (d < 0)
            d = -d;
        prod *= 1.0 - (double)d / 255.0;
    }
    return 1.0 - prod;
}

/* Phase 1: is the centre pixel clean? Counts window pixels within d of the centre
 * (the centre always counts itself) and demands count >= m+1. Out-of-bounds
 * coordinates clamp to the edge. */
static inline int is_clean(const Image *img, int x, int y, const DenoiseParams *p)
{
    int r = p->window / 2;
    int count = 0;
    const unsigned char *c = px(img, x, y);
    for (int dy = -r; dy <= r; dy++) {
        int yy = img_clamp(y + dy, img->h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = img_clamp(x + dx, img->w);
            if (fuzzy_dist(c, px(img, xx, yy)) <= p->d)
                count++;
        }
    }
    return count >= p->m + 1;
}

static inline int cmp_u8(const void *a, const void *b)
{
    return (int)*(const unsigned char *)a - (int)*(const unsigned char *)b;
}

/* Phase 2: repair one corrupted pixel. Mean of the clean neighbours only (the
 * centre is corrupted, so it is excluded too). If no neighbour is clean, fall
 * back to the per-channel median of the whole window. Integer arithmetic only,
 * so every implementation produces identical bytes. */
static inline void repair_pixel(const Image *img, const unsigned char *mask, int x, int y,
                                const DenoiseParams *p, unsigned char *dst)
{
    int r = p->window / 2;
    int sum[3] = {0, 0, 0};
    int count = 0;

    for (int dy = -r; dy <= r; dy++) {
        int yy = img_clamp(y + dy, img->h);
        for (int dx = -r; dx <= r; dx++) {
            int xx = img_clamp(x + dx, img->w);
            if (mask[(size_t)yy * (size_t)img->w + (size_t)xx])
                continue;
            const unsigned char *q = px(img, xx, yy);
            sum[0] += q[0];
            sum[1] += q[1];
            sum[2] += q[2];
            count++;
        }
    }

    if (count > 0) {
        dst[0] = (unsigned char)(sum[0] / count);
        dst[1] = (unsigned char)(sum[1] / count);
        dst[2] = (unsigned char)(sum[2] / count);
        return;
    }

    unsigned char vals[PARAMS_MAX_WINDOW * PARAMS_MAX_WINDOW];
    for (int c = 0; c < 3; c++) {
        int n = 0;
        for (int dy = -r; dy <= r; dy++) {
            int yy = img_clamp(y + dy, img->h);
            for (int dx = -r; dx <= r; dx++) {
                int xx = img_clamp(x + dx, img->w);
                vals[n++] = px(img, xx, yy)[c];
            }
        }
        qsort(vals, (size_t)n, 1, cmp_u8);
        dst[c] = vals[n / 2];
    }
}

#endif
