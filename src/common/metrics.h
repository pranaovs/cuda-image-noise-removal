#ifndef METRICS_H
#define METRICS_H

#include "image.h"

/* Image-quality metrics against a clean reference, all defined for equal-size images
 * (a mismatch is fatal: comparing different images would silently report nonsense).
 *
 *   PSNR  peak signal-to-noise ratio in dB, higher is better; the standard quality
 *         number in the denoising literature, so the results are comparable to the
 *         paper's. Returns INFINITY for identical images.
 *   MAE   mean absolute error per channel, 0..255, lower is better.
 *   max_diff  worst single-channel difference, the correctness gate: it must be 0
 *         between two implementations, which is what proves they agree bit for bit.
 *   identical  byte-for-byte comparison of the raster.
 */
double metrics_psnr(const Image *a, const Image *b);
double metrics_mae(const Image *a, const Image *b);
int metrics_max_diff(const Image *a, const Image *b);
int metrics_identical(const Image *a, const Image *b);

#endif
