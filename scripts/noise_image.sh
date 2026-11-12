#!/bin/sh
# One-shot visual check: jpg -> ppm -> noise -> denoise -> jpg.
#
#   bash scripts/noise_image.sh [test/image.jpg]        local build (gpu row skipped)
#   bash scripts/noise_image.sh --gpu [test/image.jpg]  run on `ssh gpu`, pull jpgs back
#
# Env knobs: DENSITIES ["0.10 0.20 0.35 0.50"]  SEED [42]  NOISE [sp|rv]
#            BIN [build]  JPGQ [2]
#
# Writes next to the input image, in this order:
#   0_original.jpg                        the jpg as decoded, once
#   d10pct_s42_1_noised.jpg               salt & pepper at 10%
#   d10pct_s42_2_seq.jpg                  denoise_seq
#   d10pct_s42_3_parallel.jpg             denoise_omp
#   d10pct_s42_4_gpu.jpg                  denoise_cuda (skipped if not built)
# Zero-padded density and stage index make `ls` sort 10% > 20% > 35% > 50% and
# noised > seq > parallel > gpu.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ "$1" = --gpu ]; then shift; GPU=1; fi
BIN=${BIN:-$ROOT/build}
SRC=${1:-$ROOT/test/image.jpg}
DENSITIES=${DENSITIES:-"0.10 0.20 0.35 0.50"}
SEED=${SEED:-42}
NOISE=${NOISE:-sp}
JPGQ=${JPGQ:-2}

[ -f "$SRC" ] || { echo "missing $SRC - drop image.jpg in test/" >&2; exit 2; }
DIR=$(cd "$(dirname "$SRC")" && pwd)
STEM=$(basename "$SRC")
STEM=${STEM%.*}

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# Remote mode: copy this script + the jpg to the GPU box (its build already has
# denoise_cuda - see docs/GPU.md for the sync/build one-liners), run it there, fetch
# only the finished jpgs back next to the input image. ffmpeg exists on both sides.
if [ "${GPU:-0}" = 1 ]; then
    R=${REMOTE:-\~/Projects/hpc-denoise}
    ssh gpu "test -x $R/build/denoise_cuda" || {
        echo "no $R/build/denoise_cuda on gpu - see docs/GPU.md, build with -DENABLE_CUDA=ON" >&2
        exit 2
    }
    ssh gpu "mkdir -p $R/test $R/scripts"
    scp -q "$0" "gpu:$R/scripts/noise_image.sh"
    scp -q "$SRC" "gpu:$R/test/$(basename "$SRC")"
    rm -f "$DIR"/0_original.jpg "$DIR"/d*pct_s*_*.jpg 2>/dev/null || true
    echo "-- running on gpu"
    ssh gpu "cd $R && DENSITIES='$DENSITIES' SEED=$SEED NOISE=$NOISE JPGQ=$JPGQ \
        bash scripts/noise_image.sh test/$(basename "$SRC")" | grep -v unconfirmed
    ssh gpu "cd $R/test && tar -czf - 0_original.jpg d*pct_s${SEED}_*.jpg" |
        tar -xzf - --warning=no-timestamp -C "$DIR"
    ls "$DIR"/0_original.jpg "$DIR"/d*pct_s${SEED}_*.jpg
    exit 0
fi

# ffmpeg reads/writes P6 for .ppm; -q:v 2 is visually lossless for the eyeball check.
to_ppm() { ffmpeg -hide_banner -loglevel error -y -i "$1" -pix_fmt rgb24 "$2"; }
to_jpg() { ffmpeg -hide_banner -loglevel error -y -i "$1" -q:v "$JPGQ" "$2"; }

# drop this script's previous output so a rerun does not leave stale files behind
rm -f "$DIR"/0_original.jpg "$DIR"/d*pct_s*_*.jpg 2>/dev/null || true

to_ppm "$SRC" "$T/original.ppm"
to_jpg "$T/original.ppm" "$DIR/0_original.jpg"

for DENSITY in $DENSITIES; do
    PCT=$(awk -v d="$DENSITY" 'BEGIN { printf "%02d", d * 100 + 0.5 }')
    PREFIX=d${PCT}pct_s${SEED}
    echo "-- $STEM  density=$DENSITY seed=$SEED mode=$NOISE"

    "$BIN/gennoise" "$T/original.ppm" "$T/noised.ppm" "$DENSITY" "$SEED" "$NOISE" >/dev/null
    "$BIN/imgcmp" "$T/noised.ppm" "$T/original.ppm" | sed 's/^/   noisy: /'
    to_jpg "$T/noised.ppm" "$DIR/${PREFIX}_1_noised.jpg"

    for IMPL in seq parallel gpu; do
        case $IMPL in
        seq) BIN_IMPL=$BIN/denoise_seq; IDX=2 ;;
        parallel) BIN_IMPL=$BIN/denoise_omp; IDX=3 ;;
        gpu) BIN_IMPL=$BIN/denoise_cuda; IDX=4 ;;
        esac
        if [ ! -x "$BIN_IMPL" ]; then
            echo "   $IMPL skipped: no $BIN_IMPL" >&2
            continue
        fi
        "$BIN_IMPL" "$T/noised.ppm" "$T/$IMPL.ppm" -r 1 -q "$T/original.ppm" | sed "s/^/   $IMPL /"
        to_jpg "$T/$IMPL.ppm" "$DIR/${PREFIX}_${IDX}_${IMPL}.jpg"
    done
done

ls "$DIR"/0_original.jpg "$DIR"/d*pct_s${SEED}_*.jpg
