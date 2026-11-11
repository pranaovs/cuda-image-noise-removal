/* fbcount - how often does the repair fallback actually fire?
 *
 * Core's repair_pixel() averages the clean neighbours of a corrupted pixel. At high
 * noise density a whole window can be corrupted, leaving nothing to average, and the
 * code then falls back to the per-channel median of the window. That path is the
 * honest answer to "what breaks at 30% noise", so the report needs a measured count
 * rather than a claim about it.
 *
 * Prints, over the whole image: corrupted pixels, and how many of them had no clean
 * pixel in their window. Uses is_clean() from core.h, so the count describes the code
 * that actually ships.
 *
 *   build/fbcount noisy.ppm [-n 3] [-d 0.5] [-m 4]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "image.h"
#include "params.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s noisy.ppm [-n 3] [-d 0.5] [-m 4]\n", argv[0]);
        return 2;
    }

    DenoiseParams p;
    params_default(&p);
    for (int i = 2; i < argc; i++)
        params_flag(argc, argv, &i, &p);
    params_require_valid(&p);

    Image *img = image_read(argv[1]);
    unsigned char *mask = malloc((size_t)img->w * (size_t)img->h);
    if (!mask) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    size_t corrupted = 0, fallback = 0;
    const int r = p.window / 2;
    for (int y = 0; y < img->h; y++) {
        for (int x = 0; x < img->w; x++) {
            size_t i = (size_t)y * (size_t)img->w + (size_t)x;
            mask[i] = is_clean(img, x, y, &p) ? 0 : 1;
            if (!mask[i])
                continue;
            corrupted++;
            /* A single row-major pass has the mask complete only above and to the left,
             * so the neighbours below/right are re-tested with is_clean() instead of
             * being read from the half-built mask. Slower, but it cannot undercount. */
            int clean = 0;
            for (int dy = -r; dy <= r && !clean; dy++)
                for (int dx = -r; dx <= r && !clean; dx++)
                    if (is_clean(img, img_clamp(x + dx, img->w), img_clamp(y + dy, img->h), &p))
                        clean = 1;
            if (!clean)
                fallback++;
        }
    }

    printf("pixels=%zu corrupted=%zu fallback=%zu fallback_of_corrupted=%.4f\n",
           (size_t)img->w * (size_t)img->h, corrupted, fallback,
           corrupted ? (double)fallback / (double)corrupted : 0.0);

    free(mask);
    image_free(img);
    return 0;
}
