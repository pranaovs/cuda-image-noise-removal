# Edge clamping (dev_clamp)

## The function

```cuda
__device__ int dev_clamp(int v, int hi)
{
    return v < 0 ? 0 : (v >= hi ? hi - 1 : v);
}
```

- `v < 0`   -> return `0`      (too far left or top)
- `v >= hi` -> return `hi - 1` (too far right or bottom)
- otherwise -> return `v` unchanged

`hi` is the image width or height. `__device__` means it is callable from GPU
code only; the compiler inlines it, it is not a kernel.

## What "repeat the border" means

The pixel outside the image is a copy of the nearest edge pixel. This is also
called clamp-to-edge padding.

### The problem

Pixel `P` sits at column 0. A 3x3 window around it wants columns `-1, 0, 1`.
Column `-1` does not exist.

```
       -1   0   1
      +---+---+---+
row 0 | ? | P |   |
      +---+---+---+
```

### The fix

`dev_clamp(-1, w)` returns `0`, so the window reads column 0 twice:

```
       -1   0   1
      +---+---+---+
row 0 | A | A | B |      A = pixel (row 0, col 0)
      +---+---+---+      B = pixel (row 0, col 1)
```

The edge value stretches outward into the space that does not exist. The window
is never cut short: it is always a full 3x3 = 9 samples, with the centre still
counting itself, so `count >= m + 1` means the same thing at the edge as in the
middle of the image.

### Corners come for free

Clamping each axis independently handles corners. Pixel `(0,0)` with offsets
`dy=-1, dx=-1` gives `(-1,-1)`, which clamps to `(0,0)`. The corner pixel then
votes four times:

```
        x=-1  x=0  x=1
      +-----+-----+-----+
y=-1  |  A  |  A  |  B  |
      +-----+-----+-----+
y= 0  |  A  |  A  |  B  |
      +-----+-----+-----+
y= 1  |  C  |  C  |  D  |
      +-----+-----+-----+
```

## Where it is used

Both kernels, in the inner loops:

```cuda
int yy = dev_clamp(y + dy, h);
int xx = dev_clamp(x + dx, w);
```

With the default window of 3 this runs 9 times per pixel. Every pixel gets
clamped coordinates, but only edge pixels actually change.

## Why this choice and not another

1. Repeat the border (this code). Window always full size, edge pixels get
   duplicated votes.
2. Skip out-of-range pixels. The window shrinks at the edge (9 samples become 4),
   thresholds behave differently, output differs from the CPU version.
3. Zero padding. Invents black pixels that are not in the image; they would count
   as neighbours and pull the average toward black.

Option 1 matches `img_clamp()` in `src/common/image.h`, which is what makes the
GPU output byte-identical to the CPU output. Change the rule and the correctness
gate in `scripts/check.sh` fails.

## Caveat

The claim that `dev_clamp` matches `img_clamp()` comes from the comment at the
top of `kernels.cu` and from the naming. `src/common/image.h` has not been read
to confirm it. If `img_clamp` instead skips out-of-range pixels, edge pixels
would differ between CPU and GPU and `check.sh` would fail.
