#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"
#include "noise.h"

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s clean.ppm noisy.ppm density seed [sp|rv]\n", argv[0]);
        return 2;
    }
    double density = atof(argv[3]);
    if (density < 0.0 || density > 1.0) {
        fprintf(stderr, "density must be in 0..1\n");
        return 2;
    }
    unsigned long long seed = strtoull(argv[4], NULL, 0);

    NoiseMode mode = NOISE_SALT_PEPPER;
    if (argc > 5 && !noise_mode_parse(argv[5], &mode)) {
        fprintf(stderr, "unknown noise mode: %s (use sp or rv)\n", argv[5]);
        return 2;
    }

    Image *img = image_read(argv[1]);
    add_impulse_noise(img, density, (uint64_t)seed, mode);
    int rc = image_write(img, argv[2]);
    image_free(img);
    return rc == 0 ? 0 : 1;
}
