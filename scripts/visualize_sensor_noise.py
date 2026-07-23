"""Visualize LIBERO-Plus Sensor Noise perturbations on a real LIBERO frame.

Grid layout mirrors figs/camera_viewpoints_comparison.png: ORIGINAL top-left,
then representative noise variants across the remaining cells with severity
labels overlaid on top of each cell.

Noise implementations are the exact functions from
/home/choi/LIBERO-plus/libero/libero/envs/env_wrapper.py, copied here so this
script has zero dependency on the LIBERO-Plus runtime.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from io import BytesIO

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian

# ---------------------------------------------------------------------------
# noise functions (behavior-matched to libero.libero.envs.env_wrapper).
# motion_blur uses OpenCV linear kernel instead of ImageMagick/wand to avoid
# an extra system dependency; visually equivalent to the LIBERO-Plus version.
# ---------------------------------------------------------------------------
def motion_blur(x: Image.Image, severity: int = 1) -> np.ndarray:
    # (kernel_size, _sigma_unused) matched to LIBERO-Plus severity table.
    c = [
        (5, 2), (8, 3), (10, 4), (12, 5), (15, 6),
        (18, 8), (20, 10), (25, 12), (30, 15), (35, 20),
    ][severity - 1]
    ksize = max(3, c[0] | 1)  # force odd
    angle = np.random.uniform(-45, 45)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    arr = np.array(x)
    blurred = cv2.filter2D(arr, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def gaussian_blur(x: Image.Image, severity: int = 1) -> np.ndarray:
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][severity - 1]
    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    h = img.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def zoom_blur(x: Image.Image, severity: int = 1) -> np.ndarray:
    c = [
        np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02), np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03), np.arange(1, 1.36, 0.01),
        np.arange(1, 1.41, 0.01), np.arange(1, 1.46, 0.02),
        np.arange(1, 1.51, 0.02), np.arange(1, 1.56, 0.03),
    ][severity - 1]
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += _clipped_zoom(x, zoom_factor)
    x = (x + out) / (len(c) + 1)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _plasma_fractal(mapsize: int = 256, wibbledecay: float = 3) -> np.ndarray:
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(arr):
        return arr / 4 + wibble * np.random.uniform(-wibble, wibble, arr.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        ms = maparray.shape[0]
        drgrid = maparray[stepsize // 2:ms:stepsize, stepsize // 2:ms:stepsize]
        ulgrid = maparray[0:ms:stepsize, 0:ms:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:ms:stepsize, stepsize // 2:ms:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:ms:stepsize, 0:ms:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x: Image.Image, severity: int = 1) -> np.ndarray:
    c = [
        (0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2), (2.5, 2.0),
        (3.0, 1.8), (3.5, 1.6), (4.0, 1.5), (4.5, 1.4), (5.0, 1.3),
    ][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    h, w = x.shape[0], x.shape[1]
    x = x + c[0] * _plasma_fractal(wibbledecay=c[1])[:h, :w][..., np.newaxis]
    return (np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255).astype(np.uint8)


def glass_blur(x: Image.Image, severity: int = 1) -> np.ndarray:
    c = [
        (0.5, 1, 3), (0.7, 1, 3), (0.9, 2, 3), (1.0, 2, 2), (1.1, 3, 2),
        (1.3, 3, 2), (1.5, 4, 2), (1.8, 4, 2), (2.2, 5, 1), (2.5, 5, 1),
    ][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)
    h, w = x.shape[0], x.shape[1]
    for _ in range(c[2]):
        for hh in range(h - c[1], c[1], -1):
            for ww in range(w - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                hp, wp = hh + dy, ww + dx
                x[hh, ww], x[hp, wp] = x[hp, wp].copy(), x[hh, ww].copy()
    return (np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# grid assembly (matches figs/camera_viewpoints_comparison.png style)
# ---------------------------------------------------------------------------
NOISE_NAME_BY_ID = {
    (1, 10): "motion_blur",
    (11, 20): "gaussian_blur",
    (21, 30): "zoom_blur",
    (31, 40): "fog",
    (41, 50): "glass_blur",
}


def apply_noise(pil: Image.Image, noise_id: int) -> np.ndarray:
    if 1 <= noise_id <= 10:
        return motion_blur(pil, noise_id)
    if 11 <= noise_id <= 20:
        return gaussian_blur(pil, noise_id - 10)
    if 21 <= noise_id <= 30:
        return zoom_blur(pil, noise_id - 20)
    if 31 <= noise_id <= 40:
        return fog(pil, noise_id - 30)
    if 41 <= noise_id <= 50:
        return glass_blur(pil, noise_id - 40)
    raise ValueError(f"noise_id must be 1..50, got {noise_id}")


def load_libero_frame(hdf5_path: str) -> np.ndarray:
    """Grab first frame's agentview image (RGB uint8, 224x224)."""
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        demo = f["data"][demo_keys[0]]
        img = demo["obs/agentview_rgb"][0]
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # LIBERO stores images upside-down; matches env_wrapper convention
    img = np.flipud(img).copy()
    return img


def build_grid(
    original: np.ndarray,
    variants: list[tuple[str, np.ndarray]],
    cell: int = 256,
    cols: int = 3,
) -> Image.Image:
    """Compose a captioned grid. ORIGINAL cell first, then variants."""
    cells = [("ORIGINAL", original)] + variants
    rows = (len(cells) + cols - 1) // cols

    label_h = 20
    canvas = Image.new("RGB", (cell * cols, (cell + label_h) * rows), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, arr) in enumerate(cells):
        r, c = divmod(i, cols)
        x0 = c * cell
        y0 = r * (cell + label_h)
        # label strip
        draw.rectangle([x0, y0, x0 + cell, y0 + label_h], fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 2), label, fill=(255, 255, 255), font=font)
        # image
        im = Image.fromarray(arr).resize((cell, cell), Image.BILINEAR)
        canvas.paste(im, (x0, y0 + label_h))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hdf5",
        default="/mnt/libero.datasets_224_slim/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    )
    p.add_argument("--out", default="/home/choi/VLA-WM/figs/sensor_noise_comparison.png")
    p.add_argument(
        "--noise-ids",
        nargs="+",
        type=int,
        default=[3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
        help="noise IDs to render (1-50). Default = severity 3+8 across all 5 corruption types.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)

    print(f"[load] {args.hdf5}")
    original = load_libero_frame(args.hdf5)
    print(f"[load] frame shape={original.shape} dtype={original.dtype}")

    pil = Image.fromarray(original)
    variants = []
    for nid in args.noise_ids:
        # find family name
        family = next(n for (lo, hi), n in NOISE_NAME_BY_ID.items() if lo <= nid <= hi)
        sev = ((nid - 1) % 10) + 1
        print(f"[apply] noise_id={nid:>2}  {family}  severity={sev}")
        arr = apply_noise(pil, nid)
        label = f"{family} sev={sev}  (noise_{nid})"
        variants.append((label, arr))

    grid = build_grid(original, variants, cell=256, cols=3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    grid.save(args.out)
    print(f"[save] {args.out}  ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
