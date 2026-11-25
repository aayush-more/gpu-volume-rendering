"""
vtk_compare.py - VTK reference renderer, timed the same way as bench.py.

Renders the same volumes with VTK's GPU ray caster (vtkGPUVolumeRayCastMapper) using the
same data, transfer function (tf.py presets), camera path, sample step and image sizes as
bench.py, and writes results/bench_<label>.csv in the same format, so the table can be
merged with the CUDA versions.

Put this file in the repo as  vtk_compare/vtk_compare.py  and run from anywhere:

    pip install vtk numpy matplotlib pillow
    python vtk_compare/vtk_compare.py                       (both datasets, full benchmark)
    python vtk_compare/vtk_compare.py --datasets christmastree --sizes 512 --runs 8 --warmup 2
                                                            (quick check, about a minute)

It only imports volume.py, tf.py and camera.py (no numba / CUDA), so the VTK side can be
developed on any machine. The FINAL timings must be run on the same NVIDIA machine as the
CUDA versions.

What is matched to the CUDA renderer (render.py / kernel.py):
    data and value range    same loader; density = (value - min) / (max - min)
    voxel positions         voxel centres at world (i + 0.5) * voxel size - half extent
    transfer function       the 256-entry RGBA LUT of tf.py, per-voxel opacity
    opacity correction      scalar opacity unit distance = one (smallest) voxel
    sample step             0.5 voxel, no jittering, no automatic sample-distance changes
    shading / background    shading off, black background, trilinear interpolation
    camera                  orbit_camera(): same eye, up vector and 40 degree vertical FOV
    views / frames          8 azimuths x 13 frames after 10 warm-up frames (median reported)

What is NOT identical (say so in the report):
    * VTK has its own internal optimizations and its own shader; we use its defaults.
    * Time per frame: CUDA events time only the kernel. VTK has no GPU timer in Python, so we
      time Render() plus a 1-pixel read-back that waits for the GPU. To remove the fixed cost
      of the window system, an empty scene is timed the same way ("baseline_ms") and
      net_ms = median_ms - baseline_ms. Use net_ms when comparing with the CUDA kernel times.
    * GPU memory: nvidia-smi "memory used" after minus before the volume is first drawn
      (blank if nvidia-smi is not available). Measured once per dataset, at the first size.
      It includes VTK's own buffers, so it is not exactly the CUDA 'arrays only' number.
    * VTK samples at point positions, so its volume is 1 voxel smaller than the CUDA box.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # data/ and results/ are relative to the repo root

import vtk  # noqa: E402
from vtk.util import numpy_support  # noqa: E402

from camera import orbit_camera  # noqa: E402
from tf import preset_lut, presets_for  # noqa: E402
from volume import dataset_name, load_volume, make_test_sphere, value_range  # noqa: E402

# keep these equal to bench.py
DEFAULT_DATASETS = ["christmastree", "spathorhynchus"]
VIEWS = list(range(0, 360, 45))
ELEVATION, DISTANCE, STEP = 20.0, 2.0, 0.5
IMAGE_AZIMUTH = 30.0
BASELINE_FRAMES = 30


def gpu_used_mb():
    """GPU memory in use (MB) from nvidia-smi, or None."""
    try:
        time.sleep(0.5)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.splitlines()[0])
    except Exception:
        return None


class VtkScene:
    def __init__(self, data, spacing, size):
        if data.dtype not in (np.uint8, np.uint16, np.float32):
            data = data.astype(np.float32)
        nz, ny, nx = data.shape
        sp = np.asarray(spacing, dtype=np.float64)
        extent = np.array([nx, ny, nz]) * sp
        box = extent / extent.max() / 2                       # same as Renderer.box
        voxel = 2 * box / np.array([nx, ny, nz])              # world size of one voxel, per axis
        self.voxel_world = float(sp.min() / extent.max())     # smallest voxel, world units
        self.lo, self.hi = value_range(data)
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0

        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetSpacing(*voxel)
        image.SetOrigin(*(-box + 0.5 * voxel))                # points at voxel centres
        # data is (nz, ny, nx) C-order: x varies fastest, which is VTK's point order
        arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(data).ravel(), deep=True)
        arr.SetName("density")
        image.GetPointData().SetScalars(arr)
        self.image = image

        self.prop = vtk.vtkVolumeProperty()
        self.prop.SetInterpolationTypeToLinear()
        self.prop.ShadeOff()
        self.prop.SetScalarOpacityUnitDistance(self.voxel_world)

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetInputData(image)
        self.mapper.SetBlendModeToComposite()
        self.mapper.SetAutoAdjustSampleDistances(0)
        self.mapper.SetLockSampleDistanceToInputSpacing(0)
        self.mapper.SetUseJittering(0)
        self.mapper.SetSampleDistance(STEP * self.voxel_world)

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.prop)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetMultiSamples(0)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(size, size)
        self._pixel = vtk.vtkUnsignedCharArray()  # target of the 1-pixel read-back
        self.has_volume = False

    def set_tf(self, lut):
        """lut: (256, 4) from tf.preset_lut (per-voxel opacity, not step-corrected)."""
        ctf = vtk.vtkColorTransferFunction()
        otf = vtk.vtkPiecewiseFunction()
        for i in range(256):
            v = self.lo + (self.hi - self.lo) * i / 255.0     # LUT entry i = density i / 255
            ctf.AddRGBPoint(v, float(lut[i, 0]), float(lut[i, 1]), float(lut[i, 2]))
            otf.AddPoint(v, float(lut[i, 3]))
        self.prop.SetColor(ctf)
        self.prop.SetScalarOpacity(otf)

    def show_volume(self, on):
        if on and not self.has_volume:
            self.ren.AddVolume(self.volume)
        elif not on and self.has_volume:
            self.ren.RemoveVolume(self.volume)
        self.has_volume = on

    def set_size(self, size):
        self.win.SetSize(size, size)
        got = self.win.GetSize()
        if tuple(got) != (size, size):
            print(f"  WARNING: asked for a {size}x{size} window, got {got[0]}x{got[1]}")

    def set_view(self, azimuth, size):
        c = orbit_camera(azimuth, ELEVATION, DISTANCE, width=size, height=size)
        cam = self.ren.GetActiveCamera()
        cam.SetParallelProjection(0)
        cam.SetPosition(*c.eye)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(*c.up)
        cam.SetUseHorizontalViewAngle(0)
        cam.SetViewAngle(2 * math.degrees(math.atan(c.half_h)))
        self.ren.ResetCameraClippingRange()

    def render_sync(self):
        """Render, then read one pixel back so the GPU is finished when we return."""
        self.win.Render()
        self.win.GetPixelData(0, 0, 0, 0, 0, self._pixel, 0)

    def timed(self):
        t = time.perf_counter()
        self.render_sync()
        return (time.perf_counter() - t) * 1000.0

    def grab_frame(self):
        """(h, w, 3) float32 in [0, 1], row 0 = bottom (same layout as bench.py frames)."""
        f = vtk.vtkWindowToImageFilter()
        f.SetInput(self.win)
        f.SetInputBufferTypeToRGB()
        f.ReadFrontBufferOff()
        f.Update()
        im = f.GetOutput()
        w, h, _ = im.GetDimensions()
        a = numpy_support.vtk_to_numpy(im.GetPointData().GetScalars()).reshape(h, w, 3)
        return a.astype(np.float32) / 255.0


def save_png(img, path):
    from PIL import Image

    rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(rgb[::-1]).save(path)  # row 0 is the bottom


def row(label, ds, preset, size, view, times, baseline, mem):
    t = np.array(times)
    med = float(np.median(t))
    return {
        "label": label,
        "flags": "vtk",
        "dataset": ds,
        "preset": preset,
        "width": size,
        "height": size,
        "step": STEP,
        "view_azimuth": view,
        "frames": len(t),
        "median_ms": round(med, 3),
        "mean_ms": round(float(t.mean()), 3),
        "min_ms": round(float(t.min()), 3),
        "max_ms": round(float(t.max()), 3),
        "fps": round(1000 / med, 1),
        "gpu_mem_mb": "" if mem is None else round(mem, 1),
        "baseline_ms": round(baseline, 3),
        "net_ms": round(med - baseline, 3),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", default="vtk", help="name of this run (default vtk)")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help="dataset names or paths, or 'sphere'")
    ap.add_argument("--presets", nargs="+", help="preset names (default: all of the dataset)")
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--no-images", action="store_true", help="do not save one frame per configuration")
    args = ap.parse_args()

    out_dir = ROOT / "results"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    per_view = max(1, -(-args.runs // len(VIEWS)))

    print(f"VTK {vtk.vtkVersion.GetVTKVersion()} benchmark '{args.label}'")
    print(f"{args.warmup} warm-up + {per_view * len(VIEWS)} timed frames per configuration\n")

    rows = []
    for source in args.datasets:
        if source == "sphere":
            data, spacing, ds = make_test_sphere(), (1.0, 1.0, 1.0), "sphere"
        else:
            (data, spacing), ds = load_volume(source), dataset_name(source)
        presets = args.presets or list(presets_for(ds))
        print(f"{ds}: {data.shape[2]}x{data.shape[1]}x{data.shape[0]} {data.dtype}, spacing {spacing}")

        first = args.sizes[0]
        scene = VtkScene(data, spacing, first)
        scene.set_tf(preset_lut(ds, presets[0]))
        print(f"  window {tuple(scene.win.GetSize())}")

        # GPU memory: empty scene before, volume drawn after (once per dataset)
        scene.set_view(VIEWS[0], first)
        for _ in range(3):
            scene.render_sync()
        mem0 = gpu_used_mb()
        scene.show_volume(True)
        for _ in range(3):
            scene.render_sync()
        mem1 = gpu_used_mb()
        mem = None if mem0 is None or mem1 is None else mem1 - mem0
        scene.show_volume(False)

        for preset in presets:
            scene.set_tf(preset_lut(ds, preset))
            for size in args.sizes:
                scene.set_size(size)
                scene.set_view(VIEWS[0], size)

                # baseline: same window, no volume
                scene.show_volume(False)
                for _ in range(5):
                    scene.render_sync()
                baseline = float(np.median([scene.timed() for _ in range(BASELINE_FRAMES)]))

                scene.show_volume(True)
                for _ in range(args.warmup):
                    scene.render_sync()
                all_times = []
                for az in VIEWS:
                    scene.set_view(az, size)
                    times = [scene.timed() for _ in range(per_view)]
                    all_times += times
                    rows.append(row(args.label, ds, preset, size, str(az), times, baseline, mem))
                rows.append(row(args.label, ds, preset, size, "all", all_times, baseline, mem))
                med = float(np.median(all_times))
                mem_txt = "n/a" if mem is None else f"{mem:7.1f} MB"
                print(f"  {preset:6s} {size:5d}x{size:<5d} median {med:8.2f} ms "
                      f"(empty scene {baseline:6.2f} ms, net {med - baseline:8.2f} ms)   "
                      f"{1000 / med:7.1f} FPS   GPU memory {mem_txt}")

            if not args.no_images:
                size = max(args.sizes)
                scene.set_size(size)
                scene.set_view(IMAGE_AZIMUTH, size)
                scene.show_volume(True)
                scene.render_sync()
                img = scene.grab_frame()
                stem = img_dir / f"{args.label}_{ds}_{preset}"
                np.save(f"{stem}.npy", img)
                save_png(img, f"{stem}.png")
            scene.show_volume(False)
        del scene, data

    csv_path = out_dir / f"bench_{args.label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path} and images in {img_dir}/")


if __name__ == "__main__":
    main()