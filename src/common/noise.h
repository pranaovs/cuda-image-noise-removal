#ifndef NOISE_H
#define NOISE_H

#include <stdint.h>

#include "image.h"

typedef enum {
    NOISE_SALT_PEPPER = 0,  /* each hit pixel -> pure black or pure white */
    NOISE_RANDOM_VALUED = 1 /* each hit pixel -> uniform random RGB */
} NoiseMode;

/* xorshift64*: small, seeded, identical on every machine. */
typedef struct {
    uint64_t s;
} Rng;

void rng_seed(Rng *r, uint64_t seed);
uint64_t rng_next(Rng *r);
/* uniform in [0,1), 53-bit */
double rng_double(Rng *r);
/* uniform in [0,n) */
uint32_t rng_below(Rng *r, uint32_t n);

int noise_mode_parse(const char *s, NoiseMode *out); /* "sp" / "rv" */
const char *noise_mode_name(NoiseMode mode);

/* Corrupts img in place: each pixel is hit with probability density. */
void add_impulse_noise(Image *img, double density, uint64_t seed, NoiseMode mode);

#endif
