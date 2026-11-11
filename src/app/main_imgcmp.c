/* imgcmp - compares two PPM images and prints the metrics on one line.
 *
 * This is the tool that decides whether two implementations are "the same". maxdiff=0
 * (and identical=yes) is the correctness gate used by scripts/check.sh: the CUDA and
 * OpenMP results must match the sequential result byte for byte, not approximately.
 *
 *   build/imgcmp a.ppm b.ppm
 *   identical=yes maxdiff=0 mae=0.0000 psnr=inf
 */
#include <math.h>
#include <stdio.h>

#include "image.h"
#include "metrics.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s a.ppm b.ppm\n", argv[0]);
        return 2;
    }
    Image *a = image_read(argv[1]);
    Image *b = image_read(argv[2]);
    if (a->w != b->w || a->h != b->h) {
        fprintf(stderr, "imgcmp: size mismatch %dx%d vs %dx%d\n", a->w, a->h, b->w, b->h);
        image_free(a);
        image_free(b);
        return 2;
    }

    /* Identical images have infinite PSNR; print "inf" instead of a fake large number
     * so a genuine equality can never be confused with a very good match. */
    double psnr = metrics_psnr(a, b);
    printf("identical=%s maxdiff=%d mae=%.4f psnr=", metrics_identical(a, b) ? "yes" : "no",
           metrics_max_diff(a, b), metrics_mae(a, b));
    if (isinf(psnr))
        printf("inf\n");
    else
        printf("%.2f\n", psnr);

    image_free(a);
    image_free(b);
    return 0;
}
