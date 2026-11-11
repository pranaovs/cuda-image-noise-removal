#!/bin/sh
# Live demo, ~40 s: noisy in, three implementations out, prove they agree, show the
# timings side by side, and leave two PNGs you can open on the projector.
#
#   bash scripts/demo.sh [root] [bindir]
#
# Everything it prints is a number the binary printed, nothing is recomputed here, so
# the demo cannot drift from the measurements.
set -e

ROOT=${1:-.}
BIN=${2:-$ROOT/build}
IMG=${IMG:-photo_1024}
TMP=${TMPDIR:-/tmp}/hpc-demo
DENSITY=${DENSITY:-0.10}
mkdir -p "$TMP"
export OMP_PLACES=cores OMP_PROC_BIND=close

SRC=$ROOT/data/clean/$IMG.ppm
if [ ! -f "$SRC" ]; then
    echo "missing $SRC - run this from the repo root (after the tar sync)" >&2
    exit 2
fi

line() { sed -n "s/.* ms=\([0-9.]*\) pixels=[0-9]* mpix_s=\([0-9.]*\).*/\1 ms  \2 Mpix\/s/p"; }
psnr() { sed -n 's/.*psnr=\([0-9.]*\).*/\1/p'; }

echo "== 1. clean image -> 10% salt & pepper noise"
"$BIN/gennoise" "$SRC" "$TMP/noisy.ppm" "$DENSITY" 42
"$BIN/imgcmp" "$SRC" "$TMP/noisy.ppm" | sed 's/^/   noisy: /'

echo "== 2. three implementations on the same input (-d 0.5 -m 4 -n 3)"
HAVE_CUDA=0
[ -x "$BIN/denoise_cuda" ] && HAVE_CUDA=1
[ $HAVE_CUDA = 0 ] && echo "   (no denoise_cuda in $BIN - CPU only; build with -DENABLE_CUDA=ON on the GPU box)"
S=$("$BIN/denoise_seq" "$TMP/noisy.ppm" "$TMP/seq.ppm" -q "$SRC")
printf "   seq         %s\n" "$(echo "$S" | line)"
printf "   openmp 4T   %s\n" "$(OMP_NUM_THREADS=4 "$BIN/denoise_omp" "$TMP/noisy.ppm" "$TMP/omp.ppm" -q "$SRC" | line)"
printf "   openmp 20T  %s\n" "$(OMP_NUM_THREADS=20 "$BIN/denoise_omp" "$TMP/noisy.ppm" "$TMP/omp20.ppm" -q "$SRC" | line)"
if [ $HAVE_CUDA = 1 ]; then
    CLINE=$("$BIN/denoise_cuda" "$TMP/noisy.ppm" "$TMP/cuda.ppm" -q "$SRC")
    printf "   cuda        %s\n" "$(echo "$CLINE" | line)"
    echo "$CLINE" |
        sed -n 's/.*h2d_ms=\([0-9.]*\) kernel_ms=\([0-9.]*\) d2h_ms=\([0-9.]*\).*/   cuda split  H2D \1 + kernel \2 + D2H \3 ms/p'
fi

echo "== 3. quality (PSNR vs the clean original)"
for f in seq omp; do
    printf "   %-5s psnr=%s dB\n" "$f" "$("$BIN/imgcmp" "$TMP/$f.ppm" "$SRC" | psnr)"
done
printf "   noisy psnr=%s dB\n" "$("$BIN/imgcmp" "$TMP/noisy.ppm" "$SRC" | psnr)"

echo "== 4. same bytes? (the correctness claim)"
cmp -s "$TMP/seq.ppm" "$TMP/omp.ppm" && echo "   seq == omp    byte-identical"
[ $HAVE_CUDA = 1 ] && cmp -s "$TMP/seq.ppm" "$TMP/cuda.ppm" && echo "   seq == cuda   byte-identical"

echo "== 5. figures -> results/figures/"
python3 "$ROOT/scripts/crops.py" "$ROOT" "$TMP" "$TMP/noisy.ppm" 2>/dev/null || true
echo
echo "open: results/figures/edge_crop.png   (noisy | denoised, x4)"
echo "open: results/figures/impl_strip.png  (noisy | seq | omp | cuda)"
