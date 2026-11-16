"""Basic CUDA kernels: a per-pixel gradient and an axis-aligned MIP along z.

python basic_kernels.py [dataset or path]  -> results/pixel_colors.png, results/mip_<dataset>.png
"""

import sys
from pathlib import Path

import numpy as np
from numba import cuda

from volume import dataset_name, load_volume, make_test_sphere, value_range

THREADS = (16, 16)


def blocks_for(width, height):
    return ((width + THREADS[0] - 1) // THREADS[0],
            (height + THREADS[1] - 1) // THREADS[1])


@cuda.jit
def pixel_colors(img):
    x, y = cuda.grid(2)
    h = img.shape[0]
    w = img.shape[1]
    if x < w and y < h:
        img[y, x, 0] = x / (w - 1)
        img[y, x, 1] = y / (h - 1)
        img[y, x, 2] = 0.25


@cuda.jit
def mip_z(vol, img):
    x, y = cuda.grid(2)
    nx = vol.shape[2]
    ny = vol.shape[1]
    nz = vol.shape[0]
    if x < nx and y < ny:
        m = 0.0
        for z in range(nz):
            v = vol[z, y, x]
            if v > m:
                m = v
        img[y, x] = m


def time_kernel(launch, runs=20):
    """Mean kernel time in ms (CUDA events), after one warm-up/compile call."""
    launch()
    cuda.synchronize()
    start, end = cuda.event(), cuda.event()
    start.record()
    for _ in range(runs):
        launch()
    end.record()
    end.synchronize()
    return cuda.event_elapsed_time(start, end) / runs


def save_image(path, img, gray=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if gray:
        plt.imsave(path, img, cmap="gray", origin="lower", vmin=0, vmax=1)
    else:
        plt.imsave(path, np.clip(img, 0, 1), origin="lower")


def main():
    Path("results").mkdir(exist_ok=True)

    w, h = 512, 512
    d_colors = cuda.device_array((h, w, 3), dtype=np.float32)
    pixel_colors[blocks_for(w, h), THREADS](d_colors)
    save_image("results/pixel_colors.png", d_colors.copy_to_host(), gray=False)
    print(f"pixel_colors: {blocks_for(w, h)} blocks of {THREADS} -> results/pixel_colors.png")

    source = sys.argv[1] if len(sys.argv) > 1 else "sphere"
    if source == "sphere":
        vol, name = make_test_sphere(), "sphere"
    else:
        vol, name = load_volume(source)[0], dataset_name(source)
    nz, ny, nx = vol.shape

    d_vol = cuda.to_device(vol)
    d_img = cuda.device_array((ny, nx), dtype=np.float32)
    ms = time_kernel(lambda: mip_z[blocks_for(nx, ny), THREADS](d_vol, d_img))
    img = d_img.copy_to_host()
    ok = np.allclose(img, vol.max(axis=0))
    lo, hi = value_range(vol)
    save_image(f"results/mip_{name}.png", (img - lo) / max(hi - lo, 1e-12))
    print(f"mip_z: {name} {vol.shape} in {ms:.3f} ms, matches NumPy: {ok} "
          f"-> results/mip_{name}.png")


if __name__ == "__main__":
    main()
