#!/bin/sh
# Phase-06 quality sweep: how good is the filter, and does parallelizing it change the
# result? Uses the CPU binary for the sweeps (identical pixels to CUDA, cheaper to run)
# and every implementation for the independence table.
#
#   bash scripts/quality.sh [root] [bindir]
#
# Writes results/quality_params.csv, results/quality_density.csv,
#        results/quality_impls.csv, results/quality_fallback.csv
set -e

ROOT=${1:-.}
BIN=${2:-$ROOT/build}
IMAGE=${IMAGE:-photo_1024}
DENSITY=${DENSITY:-0.10}
REPS=${REPS:-3}
TMP=${TMPDIR:-/tmp}/hpc-quality
mkdir -p "$TMP"

SRC=$ROOT/data/clean/$IMAGE.ppm
NOISY=$TMP/$IMAGE-$DENSITY.ppm
[ -f "$NOISY" ] || "$BIN/gennoise" "$SRC" "$NOISY" "$DENSITY" 42 >/dev/null

field() { sed -n "s/.* $2=\([0-9.]*\).*/\1/p"; }        # field <line> <name>
psnr_of() { sed -n 's/.*psnr=\([0-9.]*\).*/\1/p'; }
mae_of() { sed -n 's/.*mae=\([0-9.]*\).*/\1/p'; }

# ---- 1. parameter sweep: d x m on 10% noise ----------------------------------
echo "-- parameter sweep (d x m)"
echo "d,m,psnr,mae" > "$ROOT/results/quality_params.csv"
for d in 0.2 0.4 0.5 0.6 0.7 0.8 0.9; do
    for m in 1 2 3 4 5 6 7; do
        line=$("$BIN/denoise_seq" "$NOISY" "$TMP/q.ppm" -r 1 -d "$d" -m "$m" -q "$SRC")
        echo "$d,$m,$(echo "$line" | psnr_of),$(echo "$line" | mae_of)" >> "$ROOT/results/quality_params.csv"
        echo "   d=$d m=$m $(echo "$line" | sed -n 's/.* \(psnr=.*\)/\1/p')"
    done
done

# best (d, m) from the sweep, used for the density sweep
BEST=$(tail -n +2 "$ROOT/results/quality_params.csv" | sort -t, -k3 -g -r | head -1)
BD=$(echo "$BEST" | cut -d, -f1)
BM=$(echo "$BEST" | cut -d, -f2)
echo "   knee: d=$BD m=$BM psnr=$(echo "$BEST" | cut -d, -f3)"

# ---- 2. density sweep with the chosen d, m -----------------------------------
echo "-- density sweep (d=$BD m=$BM)"
echo "density,noisy_psnr,denoised_psnr,gain_db,mae" > "$ROOT/results/quality_density.csv"
for dens in 0.01 0.02 0.05 0.10 0.15 0.20 0.25 0.30; do
    "$BIN/gennoise" "$SRC" "$TMP/d.ppm" "$dens" 42 >/dev/null
    nin=$("$BIN/imgcmp" "$SRC" "$TMP/d.ppm" | psnr_of)
    line=$("$BIN/denoise_seq" "$TMP/d.ppm" "$TMP/dout.ppm" -r 1 -d "$BD" -m "$BM" -q "$SRC")
    nout=$(echo "$line" | psnr_of)
    gain=$(awk -v a="$nin" -v b="$nout" 'BEGIN { printf "%.2f", b - a }')
    echo "$dens,$nin,$nout,$gain,$(echo "$line" | mae_of)" >> "$ROOT/results/quality_density.csv"
    echo "   density=$dens noisy=$nin denoised=$nout gain=${gain}dB"
done

# ---- 3. implementation independence ------------------------------------------
echo "-- every implementation, same noisy input"
echo "impl,psnr,mae,identical_to_seq" > "$ROOT/results/quality_impls.csv"
"$BIN/denoise_seq" "$NOISY" "$TMP/ref.ppm" -r 1 -d "$BD" -m "$BM" -q "$SRC" >/dev/null
for impl in seq omp cuda; do
    case $impl in
    seq) line=$("$BIN/denoise_seq" "$NOISY" "$TMP/i.ppm" -r 1 -d "$BD" -m "$BM" -q "$SRC") ;;
    omp) line=$(OMP_NUM_THREADS=8 "$BIN/denoise_omp" "$NOISY" "$TMP/i.ppm" -r 1 -d "$BD" -m "$BM" -q "$SRC") ;;
    cuda) line=$("$BIN/denoise_cuda" "$NOISY" "$TMP/i.ppm" -r 1 -d "$BD" -m "$BM" -q "$SRC") ;;
    esac
    ident=$(cmp -s "$TMP/i.ppm" "$TMP/ref.ppm" && echo yes || echo no)
    echo "$impl,$(echo "$line" | psnr_of),$(echo "$line" | mae_of),$ident" >> "$ROOT/results/quality_impls.csv"
    echo "   $impl psnr=$(echo "$line" | psnr_of) identical=$ident"
done

# ---- 4. median-fallback frequency at high density ----------------------------
echo "-- median fallback frequency"
echo "density,corrupted,fallback,fallback_of_corrupted" > "$ROOT/results/quality_fallback.csv"
for dens in 0.10 0.20 0.30 0.40 0.50; do
    "$BIN/gennoise" "$SRC" "$TMP/f.ppm" "$dens" 42 >/dev/null
    out=$("$BIN/fbcount" "$TMP/f.ppm" -d "$BD" -m "$BM")
    echo "   density=$dens $out"
    # fbcount prints "pixels=N corrupted=N fallback=N fallback_of_corrupted=F"
    set -- $out
    echo "$dens,${2#corrupted=},${3#fallback=},${4#fallback_of_corrupted=}" >> "$ROOT/results/quality_fallback.csv"
done

# ---- 5. visual crops ---------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    echo "-- crops"
    "$BIN/denoise_seq" "$NOISY" "$TMP/seq.ppm" -r 1 -d "$BD" -m "$BM" >/dev/null
    OMP_NUM_THREADS=8 "$BIN/denoise_omp" "$NOISY" "$TMP/omp.ppm" -r 1 -d "$BD" -m "$BM" >/dev/null || true
    "$BIN/denoise_cuda" "$NOISY" "$TMP/cuda.ppm" -r 1 -d "$BD" -m "$BM" >/dev/null || true
    python3 "$ROOT/scripts/crops.py" "$ROOT" "$TMP" "$NOISY" || true
fi

echo "wrote results/quality_*.csv"
