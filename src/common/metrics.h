#ifndef METRICS_H
#define METRICS_H

#include "image.h"

/* All defined for equal-size images. psnr returns INFINITY for identical input. */
double metrics_psnr(const Image *a, const Image *b);
double metrics_mae(const Image *a, const Image *b);
int metrics_max_diff(const Image *a, const Image *b);
int metrics_identical(const Image *a, const Image *b);

#endif
