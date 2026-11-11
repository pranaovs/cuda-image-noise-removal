/* OpenMP implementation - identical work to denoise_seq.c, split across CPU threads.
 *
 * Two separate `parallel for` regions, one per phase. The implicit barrier at the end
 * of the first region is the algorithm's required barrier: every thread must finish
 * writing mask bytes before any thread starts reading its neighbours' mask bytes.
 * Fusing them would let phase 2 read a mask entry another thread has not written yet.
 *
 * No reduction, no atomics, no critical section: each pixel is written by exactly one
 * thread and read by its neighbours only after the barrier, so the result is
 * byte-identical to the sequential version at any thread count - which check.sh
 * verifies at 1/2/4/8/16 threads.
 *
 * collapse(2) flattens the row/column pair into a single iteration space so the work
 * is divided evenly and each thread keeps a contiguous block of rows (cache locality).
 * schedule(static) is the right choice because every pixel costs the same: a fixed
 * window of 9 distance evaluations, so there is no load imbalance to chase and dynamic
 * scheduling would only add overhead and destroy locality.
 *
 * Scaling limit: the stencil does 9 loads per pixel with 8/9 overlap between adjacent
 * windows, so the kernel is bound by memory bandwidth, not by floating point. That is
 * why the speedup curve flattens before the thread count reaches the core count.
 */
#include "core.h"

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
