#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

/* The project's only image container: 8-bit interleaved RGB, row-major, no padding
 * and no stride, so a pixel is 3 consecutive bytes and the whole buffer is
 * w*h*3 bytes. Interleaved (not planar) is what the CPU and the CUDA kernel both
 * read, so the two implementations touch memory in the same order. */
typedef struct {
    int w, h;
    unsigned char *rgb;
} Image;

Image *image_alloc(int w, int h);   /* zeroed? no: contents are undefined */
Image *image_clone(const Image *src); /* caller owns the result */
void image_free(Image *img);        /* NULL-safe */

/* Binary PPM (P6, maxval 255). image_read exits on a malformed file after printing
 * the path and the reason; image_write returns -1 on I/O error instead of exiting. */
Image *image_read(const char *path);
int image_write(const Image *img, const char *path);

/* Byte address of pixel (x,y). No bounds check: every caller clamps first. */
static inline unsigned char *px(const Image *img, int x, int y)
{
    return img->rgb + ((size_t)y * (size_t)img->w + (size_t)x) * 3;
}

/* Edge-replicating clamp used by the stencil: out-of-range coordinates repeat the
 * border row/column instead of shortening the window. */
static inline int img_clamp(int v, int hi)
{
    return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}

#endif
