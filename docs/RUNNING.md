# Running the code

All commands assume you are in `/home/dev/hpc-final`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Builds the CPU binaries into `build/`: `denoise_seq`, `denoise_omp`, `gennoise`, `imgcmp`, `fbcount`.

```sh
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j
```

Same, plus `build/denoise_cuda`. Needs `nvcc`.

## Correctness

```sh
cmake --build build --target check
```

Generates noise, runs every built implementation, asserts PSNR thresholds, and asserts CUDA output is byte-identical to `denoise_seq`.

## Run one image

```sh
build/denoise_cuda in.ppm out.ppm -q clean.ppm
```

Denoise `in.ppm` into `out.ppm`; `-q` prints PSNR/MAE against the clean reference. Swap `denoise_cuda` for `denoise_seq` or `denoise_omp`.

```sh
build/denoise_cuda in.ppm out.ppm -d 0.5 -m 4 -n 3 -r 5
```

Tune the filter: `-d` tolerance, `-m` peer threshold, `-n` window side (odd), `-r` repetitions, best time reported.

## Test data

```sh
build/gennoise data/clean/photo_1024.ppm noisy.ppm 0.10 42
```

Adds salt-and-pepper noise. Last arg `sp` (default) or `rv` for random-valued impulse.

```sh
build/imgcmp data/clean/photo_1024.ppm noisy.ppm
```

Prints `psnr=` and `mae=` between two images.

## Measurements

```sh
bash scripts/bench.sh . build
```

One CSV row per implementation/image/density into `results/bench.csv`. Warm-up run discarded, median of `REPS` reported. Knobs: `IMAGES`, `IMPLS`, `DENSITY`, `REPS`, `OUT`.

```sh
bash scripts/quality.sh . build
```

Parameter sweep, density sweep and cross-implementation identity table into `results/quality_*.csv`.

```sh
bash scripts/demo.sh . build
```

~40 s live demo: noisy in, three implementations out, proves they agree, prints timings side by side.
