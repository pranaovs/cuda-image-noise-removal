#!/bin/sh
# Phase-05 benchmark harness: one CSV row per (impl, threads, image, density).
#
#   bash scripts/bench.sh [root] [bindir]
#
# Env knobs (defaults in brackets):
#   IMAGES  [photo_512 photo_1024 photo_2048 photo_4096]
#   IMPLS   [seq omp1 omp2 omp4 omp8 omp14 omp20 cuda]
#   DENSITY [0.10]      REPS [5]     OUT [results/bench.csv]
#
# Discipline this script enforces so the numbers mean something:
#   * each noisy input is generated ONCE per (image, density) and reused by every
#     implementation, so all of them see byte-identical input;
#   * OMP_PLACES=cores + OMP_PROC_BIND=close, otherwise 4-thread runs vary by >20%;
#   * one warm-up process per config is discarded, then REPS processes are measured
#     and the MEDIAN is reported (clock boost + thermal noise make means useless);
#   * every binary is called with -r 3, because a fresh CUDA process pays context
#     creation (~130 ms) on its first call - with -r 1 that lands in the reported ms.
set -e

ROOT=${1:-.}
BIN=${2:-$ROOT/build}
IMAGES=${IMAGES:-"photo_512 photo_1024 photo_2048 photo_4096"}
IMPLS=${IMPLS:-"seq omp1 omp2 omp4 omp8 omp14 omp20 cuda"}
DENSITY=${DENSITY:-0.10}
REPS=${REPS:-5}
OUT=${OUT:-$ROOT/results/bench.csv}
PHASES=${PHASES:-$ROOT/results/cuda_phases.csv}
TMP=${TMPDIR:-/tmp}/hpc-bench
LOG=$ROOT/results/bench.log
mkdir -p "$TMP"

export OMP_PLACES=cores OMP_PROC_BIND=close

field() { sed -n "s/.* $2=\([0-9.]*\).*/\1/p"; }  # field <line> <name>

# one implementation run -> the binary's report line (ms is best-of-3 inside it)
run_line() {
    impl=$1 img=$2 out=$3
    case $impl in
    seq) "$BIN/denoise_seq" "$img" "$out" -r 3 ;;
    omp*) OMP_NUM_THREADS=${impl#omp} "$BIN/denoise_omp" "$img" "$out" -r 3 ;;
    cuda) "$BIN/denoise_cuda" "$img" "$out" -r 3 ;;
    *) echo "unknown impl $impl" >&2; exit 2 ;;
    esac
}

threads_of() {
    case $1 in
    seq) echo 1 ;;
    omp*) echo "${1#omp}" ;;
    *) echo 1 ;;   # cuda: the grid is not a thread count, so the column is 1
    esac
}

median() { sort -n | awk '{a[NR]=$1} END { if (NR%2) printf "%.3f\n", a[(NR+1)/2];
                                      else printf "%.3f\n", (a[NR/2]+a[NR/2+1])/2 }'; }

echo "impl,threads,width,height,density,ms,mpix_per_s,speedup,efficiency" > "$OUT"
echo "impl,width,height,kernel_ms,h2d_ms,d2h_ms,ms" > "$PHASES"
: > "$LOG"

for image in $IMAGES; do
    SRC=$ROOT/data/clean/$image.ppm
    W=$(head -c 64 "$SRC" | sed -n '2p' | cut -d' ' -f1)
    H=$(head -c 64 "$SRC" | sed -n '2p' | cut -d' ' -f2)
    NOISY=$TMP/$image-$DENSITY.ppm
    [ -f "$NOISY" ] || "$BIN/gennoise" "$SRC" "$NOISY" "$DENSITY" 42 >/dev/null
    PIXELS=$(awk -v w="$W" -v h="$H" 'BEGIN { print w*h }')
    echo "== $image ${W}x${H} density=$DENSITY" | tee -a "$LOG"

    seq_ms=""
    for impl in $IMPLS; do
        threads=$(threads_of "$impl")
        run_line "$impl" "$NOISY" "$TMP/out.ppm" >/dev/null     # warm-up, discarded
        : > "$TMP/runs.txt"
        for _ in $(seq "$REPS"); do
            line=$(run_line "$impl" "$NOISY" "$TMP/out.ppm")
            echo "$line" | field x ms >> "$TMP/runs.txt"
            case $impl in cuda) echo "$line" | sed -n "s/.* h2d_ms=\([0-9.]*\) kernel_ms=\([0-9.]*\) d2h_ms=\([0-9.]*\).*/\1 \2 \3/p" >> "$TMP/cuda.txt" ;; esac
        done
        ms=$(median < "$TMP/runs.txt")
        [ "$impl" = seq ] && seq_ms=$ms
        speedup=$(awk -v b="$seq_ms" -v t="$ms" 'BEGIN { printf "%.3f", b/t }')
        # efficiency = speedup / workers, so it is only meaningful for the CPU runs. CUDA
        # gets an empty cell: its 256-thread blocks are not the unit being scaled.
        if [ "$impl" = cuda ]; then
            eff=""
        else
            eff=$(awk -v s="$speedup" -v n="$threads" 'BEGIN { printf "%.3f", s/n }')
        fi
        mpix=$(awk -v p="$PIXELS" -v ms="$ms" 'BEGIN { printf "%.2f", p/(ms*1000.0) }')
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$impl" "$threads" "$W" "$H" "$DENSITY" \
            "$ms" "$mpix" "$speedup" "$eff" >> "$OUT"
        echo "   $impl threads=$threads ms=$ms mpix_s=$mpix speedup=$speedup" | tee -a "$LOG"

        # CUDA phase split: median of the same REPS processes
        if [ "$impl" = cuda ]; then
            k=$(awk '{print $2}' "$TMP/cuda.txt" | median)
            h2d=$(awk '{print $1}' "$TMP/cuda.txt" | median)
            d2h=$(awk '{print $3}' "$TMP/cuda.txt" | median)
            printf "%s,%s,%s,%s,%s,%s,%s\n" "$impl" "$W" "$H" "$k" "$h2d" "$d2h" "$ms" >> "$PHASES"
            rm -f "$TMP/cuda.txt"
        fi
    done
done

# Weak scaling (plan 05): same work per thread at 1024^2/1 vs 2048^2/4.
if [ -f "$ROOT/data/clean/photo_1024.ppm" ] && [ -f "$ROOT/data/clean/photo_2048.ppm" ]; then
    n1=$TMP/photo_1024-0.10.ppm
    n2=$TMP/photo_2048-0.10.ppm
    if [ -f "$n1" ] && [ -f "$n2" ]; then
        t1=$(OMP_NUM_THREADS=1 "$BIN/denoise_omp" "$n1" "$TMP/w.ppm" -r 3 | field x ms)
        t2=$(OMP_NUM_THREADS=4 "$BIN/denoise_omp" "$n2" "$TMP/w.ppm" -r 3 | field x ms)
        echo "weak scaling: 1 thread/1024^2 = ${t1} ms, 4 threads/2048^2 = ${t2} ms" | tee -a "$LOG"
    fi
fi
echo "wrote $OUT and $PHASES" | tee -a "$LOG"
