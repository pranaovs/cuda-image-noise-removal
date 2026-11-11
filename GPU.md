# GPU machine — access and setup

Everything below was verified on 2026-11-11. If a value disagrees with what you see, trust
the machine and update this file.

## Access

`~/.ssh/config` on the **local** box already has the alias. No password, key-based.

```bash
ssh gpu                                  # this is the whole thing
ssh gpu 'nvcc --version'                 # non-interactive works, nvcc is on PATH
```text

| Field | Value |
| --- | --- |
| SSH alias | `gpu` |
| Host / IP | `neobox` / `10.197.21.171` |
| User | `neo` |
| Shell | zsh, `ZDOTDIR=/home/neo/.config/zsh/` |
| Repo path | `~/Projects/hpc-denoise` (= `/home/neo/Projects/hpc-denoise`) |
| Disk free | ~188 GB on `/home` |

## Hardware

```text
GPU      NVIDIA GeForce RTX 4050 Laptop GPU
VRAM     6141 MiB (6 GB)
Compute  8.9  (Ada)  ->  -arch=sm_89  /  CMAKE_CUDA_ARCHITECTURES 89
Driver   615.78.08
Power    70 W cap, laptop part, shared with display -> noisy timings
CPU      13th Gen Intel Core i7-13700H, 14 cores / 20 threads (hybrid P+E)
```text

Toolchain:

```bash
nvcc     CUDA 13.4, V13.4.92   at /usr/local/cuda-13.4/bin  (symlink /usr/local/cuda)
cmake    4.3.0
gcc      16.2.1
OpenMP   5.2
```text

OpenMP sweep max is **20 threads**, but only 14 are physical — expect the plateau at 14.
Benchmarks need `OMP_PLACES=cores OMP_PROC_BIND=close`.

## Sync (no git in this repo)

There is no git repo on either side, so `git pull` is not the loop. From the **local** repo
root:

```bash
tar --exclude=build --exclude=.cache -czf - . \
  | ssh gpu 'tar -xzf - -C ~/Projects/hpc-denoise'
```text

Note: `rsync` is **not installed locally**. The tar pipe is the substitute.

## Build and run on the GPU box

```bash
ssh gpu
cd ~/Projects/hpc-denoise
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```text

CUDA target (phase 03+) is behind `-DENABLE_CUDA=ON`:

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```text

Expected outputs: `build/denoise_seq`, `denoise_omp`, `denoise_cuda`, `gennoise`, `imgcmp`
(all in `build/`, including the CUDA one).

Everything, including the CUDA block of the correctness pipeline:

```bash
make -C build check          # -> PASS (seq/omp/cuda all byte-identical)
```

Local box: it has **no CUDA toolkit**, so never pass `-DENABLE_CUDA=ON` there and expect a
CUDA build; also clangd cannot parse `.cu`/`.cuh` locally (see `.clangd`). `nvcc` on `gpu`
is the authority for those files.

Smoke test that works today:

```bash
./build/gennoise data/clean/photo_1024.ppm /tmp/noisy.ppm 0.10 42 sp
./build/denoise_seq /tmp/noisy.ppm /tmp/seq.ppm -q data/clean/photo_1024.ppm
OMP_NUM_THREADS=8 ./build/denoise_omp /tmp/noisy.ppm /tmp/omp.ppm
./build/denoise_cuda /tmp/noisy.ppm /tmp/cuda.ppm
./build/imgcmp /tmp/seq.ppm /tmp/cuda.ppm     # identical=yes maxdiff=0
```text

## Gotchas (all hit for real)

1. **`ZDOTDIR` redirect.** zsh reads `$ZDOTDIR/.zshenv` = `/home/neo/.config/zsh/.zshenv`.
   `$HOME/.zshenv` is ignored. The CUDA `PATH` export lives in the former.
2. **`.zshenv` is read once at shell startup.** Appending to it mid-session does nothing until
   you reconnect.
3. **`-arch=native`** works here (CUDA >= 11.5 and a GPU is present), but `89` is safer for
   reproducible flags in the report.
4. **Laptop GPU timing noise.** 70 W cap, shared with the display, power state varies. Use the
   median of 5 runs with a min/max spread — never a single timing.
5. **Long runs die with the SSH session.** Always `tmux new -s hpc` first, or `nohup ... &`.
6. **6 GB VRAM is not a limit** for any image size in this project: 4096² RGB is 50 MB;
   in + out + mask is ~68 MB.

7. **`compute-sanitizer` is a separate package** (`cuda-sanitizer-13-4`, not part of
   `cuda-toolkit`): `sudo dnf install -y cuda-sanitizer-13-4` → binary at
   `/usr/local/cuda-13.4/bin/compute-sanitizer`, version 2026.3.0. Slow (kernel_ms 8.4 →
   38.7 ms on 1024²), so use `-r 1`.

## Good first commands on a new session

```bash
ssh gpu 'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv'
ssh gpu 'nvcc --version | tail -2'
ssh gpu 'ls ~/Projects/hpc-denoise/build | head'
```text

Setup details as captured at phase start are also in `results/setup.txt`.
