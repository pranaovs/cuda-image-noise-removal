#include "core.h"

/* OpenMP version: identical body to denoise_seq_impl, only the scheduling differs.
 * Two separate `parallel for` regions, so the phase boundary is a visible implicit
 * barrier: phase 2 reads its neighbours' mask, so the whole mask must be written
 * before any pixel is repaired. No reduction, no atomics, no critical section -
 * every pixel is written by exactly one thread.
 *
 * schedule(static) is right here because the work per iteration is uniform (a
 * fixed-size window per pixel), so a static block partition has no load imbalance
 * to chase and keeps each thread on the same rows (cache/walk locality). dynamic
 * would only add scheduling overhead. */
void denoise_omp_impl(Image *img, unsigned char *mask, const DenoiseParams *p)
{
    const int w = img->w, h = img->h;

#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            mask[(size_t)y * (size_t)w + (size_t)x] = is_clean(img, x, y, p) ? 0 : 1;
    /* implicit barrier here: the mask is complete before phase 2 starts */

    Image *out = image_alloc(w, h);
#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = (size_t)y * (size_t)w + (size_t)x;
            unsigned char *dst = px(out, x, y);
            if (mask[i])
                repair_pixel(img, mask, x, y, p, dst);
            else
                memcpy(dst, px(img, x, y), 3);
        }
    }

    /* swap buffers instead of copying the whole image back */
    unsigned char *tmp = img->rgb;
    img->rgb = out->rgb;
    out->rgb = tmp;
    image_free(out);
}
