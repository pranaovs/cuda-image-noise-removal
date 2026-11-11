#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

/* The only image container in the project: 8-bit RGB, row-major, no padding. */
typedef struct {
    int w, h;
    unsigned char *rgb;
} Image;

Image *image_alloc(int w, int h);
Image *image_clone(const Image *src);
void image_free(Image *img);

/* Binary PPM (P6). Returns NULL on bad magic/truncated data (message on stderr). */
Image *image_read(const char *path);
int image_write(const Image *img, const char *path);

static inline unsigned char *px(const Image *img, int x, int y)
{
    return img->rgb + ((size_t)y * (size_t)img->w + (size_t)x) * 3;
}

static inline int img_clamp(int v, int hi)
{
    return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}

#endif
