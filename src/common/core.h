#ifndef CORE_H
#define CORE_H

#include <string.h>

#include "image.h"
#include "params.h"

/* Reference implementation of the paper's two-phase impulse-noise filter, written
 * with no parallelism at all. It defines the semantics the parallel versions must
 * reproduce exactly, so it is also the correctness oracle for seq/omp/cuda.
 *
 *   Phase 1  is_clean()     marks each pixel as clean or corrupted.
 *   Phase 2  repair_pixel() rewrites only the pixels phase 1 marked corrupted.
 *
 * The two phases cannot be fused: phase 2 reads the corruption state of a pixel's
 * neighbours, so the entire mask must exist before any pixel is repaired. That is
 * the single barrier in the algorithm.
 *
 * All arithmetic is integer (or exactly representable), so every implementation
 * produces bit-identical output; there is no reduction order to vary, no fast-math
 * and no fused multiply-add to disagree about.
 *
 * Cost per pixel: window-area fuzzy-distance evaluations in phase 1 (9 for the
 * default 3x3), window-area neighbour reads in phase 2. Adjacent windows overlap
 * 8/9, so the same bytes are fetched repeatedly: the stencil has low arithmetic
 * intensity, which is why the parallel versions run into memory bandwidth.
 */

/* Fuzzy distance between two RGB pixels, in [0,1]:
 *
 *     d(a,b) = 1 - product over c in {R,G,B} of ( 1 - |a_c - b_c| / 255 )
 *
 * Identical pixels give 0.0, maximally different pixels give 1.0. The paper states
 * the peer group as a similarity test; it is expressed here as a distance so that a
 * smaller number always means "more alike" and the comparison cannot be inverted.
 * |a_c - b_c| / 255 is exact in binary floating point, so the whole expression is
 * reproducible across compilers and devices. */
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

/* Phase 1 - detection. A pixel is clean when at least m+1 pixels of its window lie
 * within distance d of it (the centre is one of them, so m is the number of peers
 * required). Anything below that threshold is treated as an impulse.
 *
 * Out-of-range window coordinates are clamped to the image edge: border pixels reuse
 * their nearest real neighbours rather than being judged on a truncated window. */
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

/* qsort comparator for channels: unsigned bytes, not plain char. */
static inline int cmp_u8(const void *a, const void *b)
{
    return (int)*(const unsigned char *)a - (int)*(const unsigned char *)b;
}

/* Phase 2 - repair one corrupted pixel, writing its 3 channels to dst.
 *
 * The replacement is the arithmetic mean of the *clean* neighbours only; corrupted
 * neighbours are skipped and the centre is skipped too (it is corrupted by
 * definition). Restricting the average to clean pixels is what keeps edges sharp:
 * a plain window mean would smear the noise across a flat region and lose the edge.
 *
 * Fallback: at high noise density a window can contain no clean pixel, so there is
 * nothing to average. The per-channel median of the whole window is used instead -
 * the classic median filter, and the reason the output is still defined at 30% noise.
 *
 * Integer arithmetic only, so the CUDA version can match these bytes exactly. */
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
