#!/usr/bin/env python3
"""Phase-06 step 4: visual evidence. Standard library only (a 20-line PNG writer).

    python3 scripts/crops.py <root> <tmpdir> <noisy.ppm>

Writes results/figures/edge_crop.png   noisy | denoised, 128x128 nearest-neighbour x4
       results/figures/impl_strip.png  noisy | seq | omp | cuda, 96x96 nearest x3

Nearest-neighbour on purpose: smoothing an upscale would hide exactly the artefact
(pixel speckle) the crop exists to show. The crop location is chosen, not hand-picked:
the script scans the image for the 128x128 block with the highest gradient energy, so
the figure lands on a real edge and is reproducible.
"""
import os
import struct
import sys
import zlib

CROP = 128
ZOOM = 4


def read_ppm(path):
    data = open(path, "rb").read()
    parts = data.split(b"\n", 3)
    if parts[0].strip() != b"P6":
        raise SystemExit(f"{path}: not a binary PPM")
    w, h = (int(v) for v in parts[1].split())
    return w, h, parts[3]


def crop(px, w, x0, y0, cw, ch):
    out = bytearray()
    for y in range(y0, y0 + ch):
        out += px[(y * w + x0) * 3:(y * w + x0 + cw) * 3]
    return bytes(out)


def best_edge_block(px, w, h, size):
    """128x128 block with the largest sum of |dI/dx| + |dI/dy| over the grey channel."""
    step = 32
    best, best_xy = -1, (0, 0)
    for y0 in range(0, h - size + 1, step):
        for x0 in range(0, w - size + 1, step):
            e = 0
            for y in range(y0, y0 + size - 4, 4):
                base = (y * w + x0) * 3
                for x in range(0, size - 4, 4):
                    i = base + x * 3
                    e += abs(px[i] - px[i + 12]) + abs(px[i] - px[i + w * 12])
            if e > best:
                best, best_xy = e, (x0, y0)
    return best_xy


def write_png(path, w, h, rgb):
    raw = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        fh.write(chunk(b"IEND", b""))
    print("wrote", path)


def zoom_nn(px, cw, ch, z):
    out = bytearray()
    for y in range(ch * z):
        row = px[(y // z) * cw * 3:(y // z + 1) * cw * 3]
        for x in range(cw * z):
            o = (x // z) * 3
            out += row[o:o + 3]
    return bytes(out)


def compose(panels, gap=6, bg=(40, 40, 40)):
    """panels: list of (pw, ph, rgb) already zoomed."""
    pw, ph = panels[0][0], panels[0][1]
    W = sum(p[0] for p in panels) + gap * (len(panels) + 1)
    H = ph + 2 * gap
    img = bytearray(bytes(bg) * (W * H))
    x = gap
    for i, (_, _, rgb) in enumerate(panels):
        for y in range(ph):
            s = y * pw * 3
            d = ((y + gap) * W + x) * 3
            img[d:d + pw * 3] = rgb[s:s + pw * 3]
        x += pw + gap
    return W, H, bytes(img)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    tmp = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hpc-quality"
    noisy_path = sys.argv[3]
    figdir = os.path.join(root, "results", "figures")
    os.makedirs(figdir, exist_ok=True)

    w, h, noisy = read_ppm(noisy_path)
    outs = []
    for name in ("seq", "omp", "cuda"):
        p = os.path.join(tmp, name + ".ppm")
        outs.append((name, read_ppm(p) if os.path.exists(p) else None))

    x0, y0 = best_edge_block(noisy, w, h, CROP)
    print(f"crop at ({x0},{y0}) 128x128, highest gradient energy in the image")

    # figure 1: noisy vs denoised, same crop, x4
    a = zoom_nn(crop(noisy, w, x0, y0, CROP, CROP), CROP, CROP, ZOOM)
    b = zoom_nn(crop(outs[0][1][2], w, x0, y0, CROP, CROP), CROP, CROP, ZOOM)
    W, H, rgb = compose([(CROP * ZOOM, CROP * ZOOM, a), (CROP * ZOOM, CROP * ZOOM, b)])
    write_png(os.path.join(figdir, "edge_crop.png"), W, H, rgb)

    # figure 2: one strip through every implementation, x3
    s = 96
    z = 3
    panels = [(s * z, s * z, zoom_nn(crop(noisy, w, x0, y0, s, s), s, s, z))]
    for name, img in outs:
        if img:
            panels.append((s * z, s * z, zoom_nn(crop(img[2], w, x0, y0, s, s), s, s, z)))
    W, H, rgb = compose(panels)
    write_png(os.path.join(figdir, "impl_strip.png"), W, H, rgb)


if __name__ == "__main__":
    main()
