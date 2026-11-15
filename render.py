"""GPU renderer wrapper.

    r = Renderer(data, lut, spacing)
    image, ms = r.render(camera)

python render.py [dataset or path]  -> results/views_<dataset>.png
"""

import sys
from pathlib import Path

import numpy as np
from numba import config, cuda

from camera import orbit_camera
from kernel import MODE_COMPOSITE, MODE_MIP, raymarch
from tf import correct_opacity, preset_lut, presets_for
from volume import value_range

THREADS = (16, 16)
GPU_DTYPES = (np.uint8, np.uint16, np.float32)
FEATURE_FLAGS = 3

# the 8x8 warm-up launch is a single block and triggers this warning
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


class Renderer:
    def __init__(self, data, lut, spacing=(1.0, 1.0, 1.0)):
        """data: (nz, ny, nx); spacing: (sx, sy, sz) voxel size."""
        if data.dtype not in GPU_DTYPES:
            data = data.astype(np.float32)
        nz, ny, nx = data.shape
        lo, hi = value_range(data)
        self.vmin = np.float32(lo)
        self.vscale = np.float32(1.0 / (hi - lo) if hi > lo else 0.0)

        spacing = np.asarray(spacing, dtype=np.float64)
        extent = np.array([nx, ny, nz]) * spacing
        self.box = (extent / extent.max() / 2).astype(np.float32)  # longest side = 1
        self.voxel_world = spacing.min() / extent.max()            # smallest voxel in world units

        self.d_vol = cuda.to_device(np.ascontiguousarray(data))
        self.d_box = cuda.to_device(self.box)
        self.d_cam = cuda.device_array((4, 3), dtype=np.float32)
        self.d_lut = cuda.device_array((256, 4), dtype=np.float32)
        self._start_event = cuda.event()
        self._end_event = cuda.event()
        self._d_img = None
        self._h_img = None
        self.set_tf(lut)
        self.render(orbit_camera(width=8, height=8))  # compile before any timing

    def set_tf(self, lut):
        """lut: (256, 4) with per-voxel opacities (see tf.py)."""
        self.lut = np.ascontiguousarray(lut, dtype=np.float32)
        self._lut_step = None

    def _upload_lut(self, step):
        if step != self._lut_step:
            self.d_lut.copy_to_device(correct_opacity(self.lut, step))
            self._lut_step = step

    def render(self, camera, step=0.5, mode=MODE_COMPOSITE, flags=0):
        """step in (smallest) voxels. Returns (image (h, w, 3) float32, kernel time in ms)."""
        h, w = camera.height, camera.width
        if self._d_img is None or self._d_img.shape[:2] != (h, w):
            self._d_img = cuda.device_array((h, w, 3), dtype=np.float32)
            self._h_img = np.empty((h, w, 3), dtype=np.float32)
        self._upload_lut(step)
        blocks = ((w + THREADS[0] - 1) // THREADS[0], (h + THREADS[1] - 1) // THREADS[1])
        self.d_cam.copy_to_device(camera.as_array())
        step_world = np.float32(step * self.voxel_world)

        self._start_event.record()
        raymarch[blocks, THREADS](self.d_vol, self.d_lut, self.d_cam, self.d_box, step_world,
                                  self.vmin, self.vscale, mode, flags, self._d_img)
        self._end_event.record()
        self._end_event.synchronize()
        self._d_img.copy_to_host(self._h_img)
        return self._h_img, cuda.event_elapsed_time(self._start_event, self._end_event)

    def memory_bytes(self):
        """Device memory held: volume, LUT, box, camera, output image."""
        arrays = [self.d_vol, self.d_lut, self.d_box, self.d_cam, self._d_img]
        return sum(a.size * a.dtype.itemsize for a in arrays if a is not None)


def save_image(img, path):
    from PIL import Image
    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(rgb[::-1]).save(path)  # row 0 is the bottom


def save_grid(images, titles, path, cols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, title in zip(np.ravel(axes), images, titles):
        ax.imshow(np.clip(img, 0, 1), origin="lower")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    from volume import dataset_name, load_volume, make_test_sphere

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        source = 'christmastree'
        FEATURE_FLAGS = int(sys.argv[1])
    else:
        source = sys.argv[1] if len(sys.argv) > 1 else 'christmastree'
        FEATURE_FLAGS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if source == "sphere":
        data, spacing, name = make_test_sphere(), (1.0, 1.0, 1.0), "sphere"
    else:
        (data, spacing), name = load_volume(source), dataset_name(source)
    Path("results").mkdir(exist_ok=True)

    presets = list(presets_for(name))
    renderer = Renderer(data, preset_lut(name, presets[0]), spacing)
    images, titles = [], []
    for preset in presets:
        renderer.set_tf(preset_lut(name, preset))
        for az in (30, 120, 210, 300):
            img, ms = renderer.render(orbit_camera(azimuth=az, elevation=20), flags = FEATURE_FLAGS)
            images.append(img)
            titles.append(f"{preset}, azimuth {az}°  ({ms:.1f} ms)")
            print(titles[-1])
    for az in (30, 120):
        img, ms = renderer.render(orbit_camera(azimuth=az, elevation=20), mode=MODE_MIP, flags = FEATURE_FLAGS)
        images.append(img)
        titles.append(f"MIP, azimuth {az}°  ({ms:.1f} ms)")
        print(titles[-1])

    out = f"results/views_{name}.png"
    save_grid(images, titles, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
