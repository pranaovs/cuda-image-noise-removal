/* Sequential implementation - the correctness reference and the speedup baseline.
 *
 * Phase 1 writes a byte per pixel (1 = corrupted, 0 = clean). Phase 2 rewrites only
 * the corrupted pixels into a separate output buffer and swaps it in at the end.
 *
 * The end of the first loop is the algorithm's only barrier: phase 2 reads its
 * neighbours' mask bytes, so the whole mask has to exist before any pixel is
 * repaired. That is why the two phases are not fused into one pass, and it is the one
 * ordering constraint every parallel version has to reproduce.
 *
 * Both loops are embarrassingly parallel - each pixel reads 9 values and writes 1, and
 * no pixel is written twice - so this file differs from denoise_omp.c only in the two
 * OpenMP pragmas.
 */
#include "core.h"

void denoise_seq_impl(Image *img, unsigned char *mask, const DenoiseParams *p)
{
    const int w = img->w, h = img->h;

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            mask[(size_t)y * (size_t)w + (size_t)x] = is_clean(img, x, y, p) ? 0 : 1;

    Image *out = image_alloc(w, h);
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
