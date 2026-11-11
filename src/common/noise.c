/* Impulse-noise generation with a deterministic seeded RNG.
 *
 * Why not rand(): the benchmark sweeps compare implementations on the *same* noisy
 * input, and a run must be reproducible from its seed alone on any machine and libc.
 * xorshift64* gives that in 10 lines and is fast enough to generate a 4096^2 image in
 * milliseconds, which keeps it out of the measured sections.
 *
 * The generator is memoryless and uniform over pixels: each pixel is hit with
 * probability `density`, independently of its neighbours. That is the noise model the
 * paper assumes, and it is what makes the density sweep in results/ meaningful.
 */
#include "noise.h"

#include <stdlib.h>
#include <string.h>

/* State must not be 0 (xorshift's fixed point), so the seed is remapped if it is,
 * then the state is stirred a few times so that seed 1 and seed 2 diverge quickly. */
void rng_seed(Rng *r, uint64_t seed)
{
    r->s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 8; i++)
        (void)rng_next(r);
}

/* xorshift64* : three shifts/xors then a 64-bit multiply for the output scramble. */
uint64_t rng_next(Rng *r)
{
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Top 53 bits / 2^53: every result is exactly representable as a double, so
 * `rng_double(r) < density` has no rounding bias and density 0.10 hits 10% of pixels. */
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
            /* salt and pepper: black or white, never grey - the extreme value is what
             * makes an impulse separable from real image content. */
            unsigned char v = rng_below(&rng, 2) ? 255 : 0;
            p[0] = p[1] = p[2] = v;
        }
    }
}
