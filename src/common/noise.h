#ifndef NOISE_H
#define NOISE_H

#include <stdint.h>

#include "image.h"

/* The two impulse-noise models the filter is tested against. Both corrupt a pixel
 * completely rather than nudging it, which is what "impulsive" means: the value is
 * lost, not degraded. */
typedef enum {
    NOISE_SALT_PEPPER = 0,   /* hit pixel becomes pure black or pure white */
    NOISE_RANDOM_VALUED = 1  /* hit pixel becomes a uniform random colour */
} NoiseMode;

/* xorshift64*. Deterministic for a given seed on every machine and every compiler,
 * so a benchmark run is reproducible from the seed alone - no platform rand(). */
typedef struct {
    uint64_t s;
} Rng;

void rng_seed(Rng *r, uint64_t seed);
uint64_t rng_next(Rng *r);
double rng_double(Rng *r);            /* uniform in [0,1), 53-bit resolution */
uint32_t rng_below(Rng *r, uint32_t n); /* uniform in [0,n) */

int noise_mode_parse(const char *s, NoiseMode *out); /* "sp" or "rv"; 0 if unknown */
const char *noise_mode_name(NoiseMode mode);

/* Corrupts img in place: each pixel independently is hit with probability density.
 * The clean original must be kept by the caller - it is the ground truth for PSNR. */
void add_impulse_noise(Image *img, double density, uint64_t seed, NoiseMode mode);

#endif
