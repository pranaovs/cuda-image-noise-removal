/* Binary PPM (P6) reading and writing.
 *
 * PPM is used instead of an image library on purpose: the format is a short ASCII
 * header followed by raw raster bytes, so loading an image costs 40 lines and the
 * project has no external dependency on the GPU machine.
 *
 *     P6\n<width> <height>\n<maxval>\n<width*height*3 raw bytes>
 *
 * Header fields may be separated by any whitespace and may be interleaved with '#'
 * comments, which is why the reader walks the file byte by byte instead of calling
 * fscanf: mixing fscanf with ungetc loses a byte on glibc.
 *
 * Only maxval 255 (8-bit) is supported - the filter works on single bytes.
 */
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Every failure here is fatal for the run: print the path and what was wrong. */
static void die(const char *msg, const char *path)
{
    fprintf(stderr, "image: %s: %s\n", path, msg);
    exit(1);
}

/* Allocates an uninitialised w*h*3 buffer. Dimensions are validated because a
 * negative or zero size would otherwise turn into a huge size_t. */
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

/* Deep copy. The benchmark loop needs a pristine input for every repetition, so each
 * run starts from a clone of the same noisy image rather than from the previous run's
 * output. */
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

/* Reads the next decimal header field, skipping whitespace and '#' comment lines.
 * Returns 0 at end of file. One pushback byte at most, which is well defined. */
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

/* Writes a canonical 3-line header (no comment, maxval 255) so files can be diffed
 * and compared byte for byte with `cmp` across implementations. */
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
