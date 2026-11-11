#ifndef PARAMS_H
#define PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Largest supported odd window side. Bounds the stack buffer repair_pixel() sorts in
 * its median fallback, so this is a memory limit, not a tuning knob. */
#define PARAMS_MAX_WINDOW 15

/* Tunables of the paper's filter, plus the CLI helpers shared by every binary so the
 * three denoisers cannot drift apart in how they read their arguments. */
typedef struct {
    int window; /* n, side of the square neighbourhood; odd, >= 3 */
    double d;   /* fuzzy tolerance in [0,1]: lower = stricter "looks like the centre" */
    int m;      /* required peer count - 1; a pixel is corrupted below m+1 peers */
} DenoiseParams;

/* Shipped defaults, taken from the parameter sweep in results/quality_params.csv:
 * d=0.5, m=4 is the maximum on 10% salt-and-pepper noise and an interior optimum in
 * both directions (d=0.4/0.6 and m=3/5 all score lower), so it is a knee rather than
 * the edge of the tested grid. */
static inline void params_default(DenoiseParams *p)
{
    p->window = 3;
    p->d = 0.5;
    p->m = 4;
}

static inline int params_valid(const DenoiseParams *p)
{
    return p->window >= 3 && p->window <= PARAMS_MAX_WINDOW && (p->window % 2) == 1 &&
           p->d >= 0.0 && p->d <= 1.0 && p->m >= 1;
}

/* Handles -d/-m/-n in argv[*i], consuming the value that follows. Returns 1 if this
 * argument belonged to the parameter set. Used by main_denoise and fbcount. */
static inline int params_flag(int argc, char **argv, int *i, DenoiseParams *p)
{
    const char *a = argv[*i];
    int *dst_int = NULL;

    if (strcmp(a, "-d") == 0) {
        if (*i + 1 >= argc)
            return 1;
        p->d = atof(argv[++(*i)]);
        return 1;
    }
    if (strcmp(a, "-m") == 0)
        dst_int = &p->m;
    else if (strcmp(a, "-n") == 0)
        dst_int = &p->window;
    if (dst_int) {
        if (*i + 1 < argc)
            *dst_int = atoi(argv[++(*i)]);
        return 1;
    }
    return 0;
}

static inline void params_require_valid(const DenoiseParams *p)
{
    if (!params_valid(p)) {
        fprintf(stderr,
                "bad params: -n %d (odd, 3..%d) -d %.3f (0..1) -m %d (>=1)\n",
                p->window, PARAMS_MAX_WINDOW, p->d, p->m);
        exit(2);
    }
}

#endif
