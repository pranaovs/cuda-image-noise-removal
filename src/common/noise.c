#include "noise.h"

#include <stdlib.h>
#include <string.h>

void rng_seed(Rng *r, uint64_t seed)
{
    /* 0 is a fixed point of xorshift, so never let the state be 0 */
    r->s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 8; i++)
        (void)rng_next(r);
}

uint64_t rng_next(Rng *r)
{
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double rng_double(Rng *r)
{
    return (double)(rng_next(r) >> 11) / 9007199254740992.0; /* 2^53 */
}

uint32_t rng_below(Rng *r, uint32_t n)
{
    return (uint32_t)(rng_double(r) * (double)n);
}

int noise_mode_parse(const char *s, NoiseMode *out)
{
    if (strcmp(s, "sp") == 0 || strcmp(s, "saltpepper") == 0) {
        *out = NOISE_SALT_PEPPER;
        return 1;
    }
    if (strcmp(s, "rv") == 0 || strcmp(s, "random") == 0) {
        *out = NOISE_RANDOM_VALUED;
        return 1;
    }
    return 0;
}

const char *noise_mode_name(NoiseMode mode)
{
    return mode == NOISE_RANDOM_VALUED ? "rv" : "sp";
}

void add_impulse_noise(Image *img, double density, uint64_t seed, NoiseMode mode)
{
    Rng rng;
    rng_seed(&rng, seed);
    size_t n = (size_t)img->w * (size_t)img->h;
    for (size_t i = 0; i < n; i++) {
        unsigned char *p = img->rgb + i * 3;
        if (rng_double(&rng) >= density)
            continue;
        if (mode == NOISE_RANDOM_VALUED) {
            p[0] = (unsigned char)rng_below(&rng, 256);
            p[1] = (unsigned char)rng_below(&rng, 256);
            p[2] = (unsigned char)rng_below(&rng, 256);
        } else {
            unsigned char v = rng_below(&rng, 2) ? 255 : 0;
            p[0] = p[1] = p[2] = v;
        }
    }
}
