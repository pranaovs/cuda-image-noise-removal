#ifndef PARAMS_H
#define PARAMS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Largest odd window core.h supports (bounds the local median buffer). */
#define PARAMS_MAX_WINDOW 15

typedef struct {
    int window;  /* n, side of the square window; odd, >= 3 */
    double d;    /* fuzzy tolerance, 0..1 */
    int m;       /* peer cardinality; corrupted if peer count < m + 1 */
} DenoiseParams;

/* Defaults come from the phase-06 parameter sweep (results/quality_params.csv):
 * d=0.5, m=4 tops the grid at 32.26 dB on 10% salt & pepper, and it is an interior
 * maximum in both directions (d=0.4 and d=0.6 are worse, m=3 and m=5 are worse). */
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

/* Consumes argv[*i] and, when it is one of -d/-m/-n, also the following value.
 * Returns 1 if handled. Shared by every denoiser binary so the flags never drift. */
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
