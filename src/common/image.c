#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg, const char *path)
{
    fprintf(stderr, "image: %s: %s\n", path, msg);
    exit(1);
}

Image *image_alloc(int w, int h)
{
    if (w <= 0 || h <= 0) {
        fprintf(stderr, "image_alloc: bad size %dx%d\n", w, h);
        exit(1);
    }
    Image *img = malloc(sizeof *img);
    if (!img) {
        fprintf(stderr, "image_alloc: out of memory\n");
        exit(1);
    }
    img->w = w;
    img->h = h;
    img->rgb = malloc((size_t)w * (size_t)h * 3);
    if (!img->rgb) {
        fprintf(stderr, "image_alloc: out of memory\n");
        exit(1);
    }
    return img;
}

Image *image_clone(const Image *src)
{
    Image *img = image_alloc(src->w, src->h);
    memcpy(img->rgb, src->rgb, (size_t)src->w * (size_t)src->h * 3);
    return img;
}

void image_free(Image *img)
{
    if (!img)
        return;
    free(img->rgb);
    free(img);
}

/* Next decimal header value, skipping whitespace and '#' comments.
 * Hand-rolled because mixing ungetc with fscanf loses data on glibc. */
static int next_int(FILE *f, int *out)
{
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == EOF)
            return 0;
        if (c == '#') {
            while (c != '\n' && c != EOF)
                c = fgetc(f);
            continue;
        }
        if (c >= '0' && c <= '9')
            break;
    }
    int v = 0;
    while (c >= '0' && c <= '9') {
        v = v * 10 + (c - '0');
        c = fgetc(f);
    }
    if (c != EOF)
        ungetc(c, f);
    *out = v;
    return 1;
}

Image *image_read(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        die("cannot open", path);

    int c1 = fgetc(f), c2 = fgetc(f);
    if (c1 != 'P' || c2 != '6')
        die("not a binary PPM (P6)", path);

    int hdr[3];
    for (int i = 0; i < 3; i++) {
        if (!next_int(f, &hdr[i])) {
            fclose(f);
            die("truncated header", path);
        }
    }
    int w = hdr[0], h = hdr[1], maxval = hdr[2];
    if (maxval != 255)
        die("only 8-bit PPM (maxval 255) supported", path);

    /* exactly one whitespace byte separates the header from the raster */
    if (fgetc(f) == EOF)
        die("truncated header", path);

    Image *img = image_alloc(w, h);
    size_t want = (size_t)w * (size_t)h * 3;
    if (fread(img->rgb, 1, want, f) != want) {
        image_free(img);
        die("truncated pixel data", path);
    }
    fclose(f);
    return img;
}

int image_write(const Image *img, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "image: cannot open for writing: %s\n", path);
        return -1;
    }
    fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
    size_t n = (size_t)img->w * (size_t)img->h * 3;
    if (fwrite(img->rgb, 1, n, f) != n) {
        fprintf(stderr, "image: short write: %s\n", path);
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
}
