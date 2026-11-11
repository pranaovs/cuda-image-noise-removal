#include "core.h"

/* Sequential reference: two loops, one implicit barrier between them (the end of
 * the first loop). Phase 2 cannot read a partial mask, which is the only reason
 * the phases are not fused. */
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
