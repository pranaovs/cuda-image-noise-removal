#!/usr/bin/env python3
"""Turn the phase-05/06 CSVs into SVGs. Standard library only: no matplotlib, no numpy,
so the figures rebuild on any box with python3.

    python3 scripts/plot.py [results-dir]

Reads   results/bench.csv            results/cuda_phases.csv
        results/quality_params.csv   results/quality_density.csv
Writes  results/figures/speedup_vs_threads.svg
        results/figures/mpix_vs_size.svg
        results/figures/cuda_phases.svg
        results/figures/psnr_vs_density.svg
        results/figures/psnr_vs_d.svg
"""
import csv
import os
import sys

W, H = 720, 460
PAD_L, PAD_R, PAD_T, PAD_B = 70, 150, 46, 56
COLORS = ["#1b6ca8", "#c0392b", "#1e8449", "#8e44ad", "#d68910", "#117a65", "#7f8c8d"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def nice_ticks(lo, hi, n=5):
    """Round axis bounds and tick step to 1/2/5 x 10^k."""
    if hi <= lo:
        hi = lo + 1.0
    span = (hi - lo) / n
    mag = 10 ** (len(str(int(abs(span)))) - 1) if span >= 1 else 1
    for step in (1, 2, 2.5, 5, 10):
        s = step * mag
        if s >= span:
            break
    lo_t = (int(lo / s)) * s
    ticks = []
    v = lo_t
    while v <= hi + s * 0.5:
        ticks.append(round(v, 6))
        v += s
    return ticks


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    def __init__(self, title, xlabel, ylabel):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.items = []
        self.pw, self.ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def frame(self, xlo, xhi, ylo, yhi, xticks=None, yticks=None):
        pw, ph = self.pw, self.ph
        xticks = xticks if xticks is not None else nice_ticks(xlo, xhi)
        yticks = yticks if yticks is not None else nice_ticks(ylo, yhi)
        xhi = max([xhi] + xticks)
        self.xlo, self.xhi, self.ylo, self.yhi = xlo, xhi, ylo, yticks[-1]

        def X(v):
            return PAD_L + (v - xlo) / (self.xhi - xlo) * pw

        def Y(v):
            return PAD_T + ph - (v - ylo) / (self.yhi - ylo) * ph

        self.X, self.Y = X, Y
        self.items.append(
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>'
            f'<text x="{W/2:.0f}" y="26" text-anchor="middle" font-family="DejaVu Sans,sans-serif"'
            f' font-size="15" font-weight="bold">{esc(self.title)}</text>'
            f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>'
            f'<text x="{PAD_L + pw/2:.0f}" y="{H-14}" text-anchor="middle"'
            f' font-family="DejaVu Sans,sans-serif" font-size="12">{esc(self.xlabel)}</text>'
            f'<text transform="translate(16,{PAD_T + ph/2:.0f}) rotate(-90)" text-anchor="middle"'
            f' font-family="DejaVu Sans,sans-serif" font-size="12">{esc(self.ylabel)}</text>'
        )
        for t in xticks:
            self.items.append(
                f'<line x1="{X(t):.1f}" y1="{PAD_T+ph}" x2="{X(t):.1f}" y2="{PAD_T+ph+5}" stroke="#333"/>'
                f'<text x="{X(t):.1f}" y="{PAD_T+ph+20}" text-anchor="middle"'
                f' font-family="DejaVu Sans,sans-serif" font-size="11">{t:g}</text>'
            )
        for t in yticks:
            self.items.append(
                f'<line x1="{PAD_L-5}" y1="{Y(t):.1f}" x2="{PAD_L}" y2="{Y(t):.1f}" stroke="#333"/>'
                f'<line x1="{PAD_L}" y1="{Y(t):.1f}" x2="{PAD_L+pw}" y2="{Y(t):.1f}" stroke="#e8e8e8"/>'
                f'<text x="{PAD_L-9}" y="{Y(t)+4:.1f}" text-anchor="end"'
                f' font-family="DejaVu Sans,sans-serif" font-size="11">{t:g}</text>'
            )

    def series(self, name, pts, color, marks=True):
        if not pts:
            return
        d = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in pts)
        self.items.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        if marks:
            for x, y in pts:
                self.items.append(
                    f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="3.2" fill="{color}"/>'
                )
        self._legend(name, color)

    def bars(self, names, groups, colors=None):
        """groups = [(label, [v0, v1, ...]), ...]; names = series names."""
        nb = len(groups)
        ns = len(names)
        span = (self.xhi - self.xlo) / nb
        for gi, (label, vals) in enumerate(groups):
            x0 = self.xlo + gi * span + span * 0.15
            bw = span * 0.7 / ns
            for si, v in enumerate(vals):
                x = self.X(x0 + si * bw)
                self.items.append(
                    f'<rect x="{x:.1f}" y="{self.Y(v):.1f}" width="{self.X(x0+bw)-self.X(x0)-1:.1f}"'
                    f' height="{self.Y(0)-self.Y(v):.1f}" fill="{colors[si % len(colors)]}"/>'
                )
            self.items.append(
                f'<text x="{self.X(x0 + span*0.35):.1f}" y="{PAD_T+self.ph+20}" text-anchor="middle"'
                f' font-family="DejaVu Sans,sans-serif" font-size="11">{esc(label)}</text>'
            )
        for si, n in enumerate(names):
            self._legend(n, colors[si % len(colors)])

    def _legend(self, name, color):
        self.legend = getattr(self, "legend", [])
        if name in [k for k, _ in self.legend]:
            return
        self.legend.append((name, color))
        y = PAD_T + 6 + 18 * (len(self.legend) - 1)
        self.items.append(
            f'<line x1="{PAD_L+self.pw+14}" y1="{y+4}" x2="{PAD_L+self.pw+34}" y2="{y+4}"'
            f' stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{PAD_L+self.pw+40}" y="{y+8}" font-family="DejaVu Sans,sans-serif"'
            f' font-size="11">{esc(name)}</text>'
        )

    def save(self, path):
        with open(path, "w") as fh:
            fh.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
                f' viewBox="0 0 {W} {H}">\n' + "\n".join(self.items) + "\n</svg>\n"
            )
        print("wrote", path)


def fnum(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    figdir = os.path.join(rdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    bench = read_csv(os.path.join(rdir, "bench.csv"))
    phases = read_csv(os.path.join(rdir, "cuda_phases.csv"))

    # 1. OpenMP speedup vs threads, one line per image size
    sizes = sorted({int(fnum(r, "width")) for r in bench if r["impl"].startswith("omp")})
    if sizes:
        best = max([1.0] + [fnum(r, "speedup") for r in bench])
        c = Chart("OpenMP speedup vs thread count (10% noise)", "threads (OMP_NUM_THREADS)",
                  "speedup vs 1-thread sequential")
        c.frame(1, 20, 0, round(best + 0.5), xticks=[1, 2, 4, 8, 14, 16, 20])
        c.items.append(
            f'<line x1="{c.X(1):.1f}" y1="{c.Y(1):.1f}" x2="{c.X(20):.1f}" y2="{c.Y(20):.1f}"'
            f' stroke="#bbb" stroke-dasharray="5,4"/>'
        )
        for i, s in enumerate(sizes):
            pts = sorted({(int(fnum(r, "threads")), fnum(r, "speedup"))
                          for r in bench if r["impl"].startswith("omp")
                          and int(fnum(r, "width")) == s})
            c.series(f"{s}x{s}", pts, COLORS[i % len(COLORS)])
        c._legend("ideal (y=x)", "#bbb")
        c.save(os.path.join(figdir, "speedup_vs_threads.svg"))

    # 2. Throughput vs image side, best config per implementation class
    classes = [("seq", lambda r: r["impl"] == "seq"),
               ("omp 14 threads", lambda r: r["impl"] == "omp14"),
               ("cuda", lambda r: r["impl"] == "cuda")]
    rows = [(n, sorted((int(fnum(r, "width")), fnum(r, "mpix_per_s")) for r in bench if f(r)))
            for n, f in classes]
    rows = [(n, p) for n, p in rows if p]
    if rows:
        ymax = max(y for _, p in rows for _, y in p)
        c = Chart("Throughput vs image size (10% noise)", "image side (pixels)",
                  "Mpixels/s")
        c.frame(min(x for _, p in rows for x, _ in p), max(x for _, p in rows for x, _ in p),
                0, ymax * 1.15)
        for i, (n, p) in enumerate(rows):
            c.series(n, p, COLORS[i % len(COLORS)])
        c.save(os.path.join(figdir, "mpix_vs_size.svg"))

    # 3. CUDA kernel time vs PCIe transfer time
    if phases:
        sizes = sorted({int(fnum(r, "width")) for r in phases})
        names = ["kernel", "h2d", "d2h"]
        groups = []
        ymax = 0
        for s in sizes:
            rs = [r for r in phases if int(fnum(r, "width")) == s]
            vals = [sum(fnum(r, k) for r in rs) / len(rs) for k in ("kernel_ms", "h2d_ms", "d2h_ms")]
            groups.append((f"{s}x{s}", vals))
            ymax = max([ymax] + vals)
        c = Chart("CUDA: kernel time vs PCIe transfer time", "", "milliseconds")
        c.frame(0, len(sizes), 0, ymax * 1.2, xticks=[], yticks=None)
        c.bars(names, groups, ["#1b6ca8", "#d68910", "#7f8c8d"])
        c.save(os.path.join(figdir, "cuda_phases.svg"))

    # 5. PSNR vs noise density
    dens = read_csv(os.path.join(rdir, "quality_density.csv"))
    if dens:
        sets = [("noisy input", "noisy_psnr", "#c0392b"),
                ("denoised", "denoised_psnr", "#1b6ca8")]
        ymax = max(max(fnum(r, k) for k in ("noisy_psnr", "denoised_psnr")) for r in dens)
        c = Chart("Reconstruction quality vs noise density", "salt & pepper density",
                  "PSNR (dB)")
        c.frame(0, max(fnum(r, "density") for r in dens), 0, ymax + 3)
        for name, key, col in sets:
            c.series(name, [(fnum(r, "density"), fnum(r, key)) for r in dens], col)
        c.save(os.path.join(figdir, "psnr_vs_density.svg"))

    # 6. PSNR vs tolerance d, one line per m
    par = read_csv(os.path.join(rdir, "quality_params.csv"))
    if par:
        ms = sorted({int(fnum(r, "m")) for r in par})
        ymax = max(fnum(r, "psnr") for r in par)
        ymin = min(fnum(r, "psnr") for r in par)
        c = Chart("Quality vs fuzzy tolerance d, per cardinality m", "d", "PSNR (dB)")
        c.frame(min(fnum(r, "d") for r in par), max(fnum(r, "d") for r in par),
                max(0, ymin - 1), ymax + 1)
        for i, m in enumerate(ms):
            pts = sorted((fnum(r, "d"), fnum(r, "psnr")) for r in par if int(fnum(r, "m")) == m)
            c.series(f"m={m}", pts, COLORS[i % len(COLORS)])
        c.save(os.path.join(figdir, "psnr_vs_d.svg"))


if __name__ == "__main__":
    main()
