/* Phase-06 step 5: how often does the filter's median fallback fire?
 *
 * repair_pixel() normally averages the *clean* neighbours of a corrupted pixel. At high
 * noise density a whole 3x3 window can be corrupted, there are no clean neighbours, and
 * core.h falls back to the per-channel median of all 9. That path is rare, slow (sort
 * per channel) and it is the honest answer to "what breaks at 30% noise", so it needs a
 * number rather than a claim.
 *
 * Counts, over the whole image: corrupted pixels, and corrupted pixels whose window has
 * zero clean pixels. Uses core.h directly, so the numbers describe the shipped code.
 *
 *   build/fbcount noisy.ppm [-n 3] [-d 0.6] [-m 2]
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
        fprintf(stderr, "usage: %s noisy.ppm [-n 3] [-d 0.6] [-m 2]\n", argv[0]);
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
            /* The mask is complete for the pixels above/left but not for the ones
             * below/right, so ask the detector directly instead of reading it. */
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
