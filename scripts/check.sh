#!/bin/sh
# Phase-01 correctness pipeline. Usage: scripts/check.sh <repo-root> <bin-dir>
set -e

ROOT=${1:-.}
BIN=${2:-build}
SMALL=$ROOT/data/clean/photo_1024.ppm
BIG=$ROOT/data/clean/photo_4096.ppm
TMP=${TMPDIR:-/tmp}/hpc-check
mkdir -p "$TMP"

fail() { echo "FAIL: $*"; exit 1; }
psnr_of() { sed -n 's/.*psnr=\([0-9.]*\).*/\1/p' "$1"; }

echo "-- 10% salt & pepper, photo_1024"
"$BIN/gennoise" "$SMALL" "$TMP/noisy.ppm" 0.10 42
NOISY=$("$BIN/imgcmp" "$SMALL" "$TMP/noisy.ppm" | psnr_of /dev/stdin)
echo "   noisy psnr=$NOISY"

LINE=$("$BIN/denoise_seq" "$TMP/noisy.ppm" "$TMP/seq.ppm" -q "$SMALL")
echo "   $LINE"
OUT=$(echo "$LINE" | psnr_of /dev/stdin)

awk -v a="$OUT" -v b="$NOISY" 'BEGIN { exit !(a - b >= 5.0) }' ||
    fail "psnr improvement < 5 dB ($NOISY -> $OUT)"
awk -v a="$OUT" 'BEGIN { exit !(a >= 25.0) }' || fail "psnr $OUT < 25 dB"

echo "-- random-valued impulse noise (0.05)"
"$BIN/gennoise" "$SMALL" "$TMP/rv.ppm" 0.05 7 rv
RV_IN=$("$BIN/imgcmp" "$SMALL" "$TMP/rv.ppm" | psnr_of /dev/stdin)
LINE=$("$BIN/denoise_seq" "$TMP/rv.ppm" "$TMP/rv_out.ppm" -q "$SMALL")
echo "   noisy $RV_IN -> $LINE"
RV_OUT=$(echo "$LINE" | psnr_of /dev/stdin)
awk -v a="$RV_OUT" -v b="$RV_IN" 'BEGIN { exit !(a - b >= 5.0) }' ||
    fail "rv noise improvement < 5 dB ($RV_IN -> $RV_OUT)"

# The detector is not a guaranteed no-op on a clean image: a handful of extreme-edge
# pixels can fall below the m+1 peer count (measured at the phase-06 defaults). The
# plan's criterion for this case is therefore PSNR > 50 dB, not byte identity - and the
# real correctness gate is that every implementation produces the SAME bytes, below.
echo "-- clean image is essentially untouched (photo_4096)"
"$BIN/gennoise" "$BIG" "$TMP/clean0.ppm" 0 1
"$BIN/denoise_seq" "$TMP/clean0.ppm" "$TMP/out0.ppm" -r 1 >/dev/null
ID=$("$BIN/imgcmp" "$TMP/clean0.ppm" "$TMP/out0.ppm")
echo "   $ID"
CLEAN_PSNR=$(echo "$ID" | psnr_of /dev/stdin)
awk -v p="$CLEAN_PSNR" 'BEGIN { exit !(p + 0 >= 50.0) }' ||
    fail "clean image psnr $CLEAN_PSNR below 50 dB"


echo "-- determinism"
"$BIN/denoise_seq" "$TMP/noisy.ppm" "$TMP/seq2.ppm" -r 1 >/dev/null
cmp -s "$TMP/seq.ppm" "$TMP/seq2.ppm" || fail "two sequential runs differ"

echo "-- window 5 still runs"
"$BIN/denoise_seq" "$TMP/noisy.ppm" "$TMP/w5.ppm" -n 5 -r 1 -q "$SMALL" >/dev/null

if [ -x "$BIN/denoise_omp" ]; then
    echo "-- openmp must be byte-identical to sequential"
    for t in 1 2 4 8 16; do
        OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
            "$BIN/denoise_omp" "$TMP/noisy.ppm" "$TMP/omp.ppm" -r 1 >/dev/null
        cmp -s "$TMP/seq.ppm" "$TMP/omp.ppm" || fail "omp output differs at $t threads"
    done
    echo "   identical at 1/2/4/8/16 threads"
fi

if [ -x "$BIN/denoise_cuda" ]; then
    echo "-- cuda must be byte-identical to sequential"
    LINE=$("$BIN/denoise_cuda" "$TMP/noisy.ppm" "$TMP/cuda.ppm" -r 1)
    echo "   $LINE"
    cmp -s "$TMP/seq.ppm" "$TMP/cuda.ppm" || fail "cuda output differs from seq"
    "$BIN/denoise_cuda" "$TMP/noisy.ppm" "$TMP/cuda5.ppm" -r 1 -n 5 >/dev/null
    "$BIN/denoise_seq" "$TMP/noisy.ppm" "$TMP/seq5.ppm" -r 1 -n 5 >/dev/null
    cmp -s "$TMP/seq5.ppm" "$TMP/cuda5.ppm" || fail "cuda differs at window 5"
    echo "-- cuda matches sequential on the clean image too"
    "$BIN/denoise_cuda" "$TMP/clean0.ppm" "$TMP/cuda0.ppm" -r 1 >/dev/null
    cmp -s "$TMP/out0.ppm" "$TMP/cuda0.ppm" || fail "cuda differs from seq on clean input"
    echo "-- cuda determinism"
    "$BIN/denoise_cuda" "$TMP/noisy.ppm" "$TMP/cuda_again.ppm" -r 1 >/dev/null
    cmp -s "$TMP/cuda.ppm" "$TMP/cuda_again.ppm" || fail "two cuda runs differ"
    echo "   identical across runs, window 5, and clean input"
fi

echo "PASS"
