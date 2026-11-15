"""
bench.py - measure rendering speed with fixed settings, so every version is compared fairly.

Fixed settings:
    datasets     Christmas Tree (512x499x512) and Spathorhynchus (1024x1024x750)
    presets      full and dense (per dataset, see tf.py)
    image sizes  512x512 and 1024x1024
    camera       8 views every 45 degrees around the volume, elevation 20, distance 2
    step         0.5 voxels
    timing       10 warm-up frames (not counted), then at least 100 timed frames
                 (13 per view x 8 views = 104); we report the median time per frame

Outputs (in results/):
    bench_<label>.csv              one row per view + an 'all' row per configuration
    images/<label>_<dataset>_<preset>.npy/.png   one 1024x1024 frame per configuration,
                                   used to check that an optimization didn't change the picture

Run:  python bench.py                         (Version 1 baseline, label v1)
      python bench.py --label v2 --flags 3 --compare v1
"""

import argparse
import csv
from pathlib import Path
import gc

import numpy as np
from numba import cuda

from camera import orbit_camera
from render import Renderer, save_image
from tf import preset_lut, presets_for
from volume import dataset_name, load_volume, make_test_sphere

DEFAULT_DATASETS = ["christmastree", "spathorhynchus"]
VIEWS = list(range(0, 360, 45))  # azimuths: 0, 45, ..., 315
ELEVATION, DISTANCE, STEP = 20.0, 2.0, 0.5
IMAGE_AZIMUTH = 30.0  # view saved for the image comparison


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--label", default="v1", help="name for this version, e.g. v1, v2, v3"
    )
    ap.add_argument(
        "--flags", type=int, default=0, help="FLAG_* value passed to the kernel"
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="dataset names or paths, or 'sphere'",
    )
    ap.add_argument(
        "--presets", nargs="+", help="preset names (default: all presets of the dataset)"
    )
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument(
        "--compare", help="label of an earlier run to compare images against"
    )
    args = ap.parse_args()

    out_dir = Path("results")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    per_view = max(
        1, -(-args.runs // len(VIEWS))
    )  # round up: 100 runs / 8 views -> 13 each
    try:
        gpu = cuda.get_current_device().name
        gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    except Exception:
        gpu = "unknown GPU"
    print(f"Benchmark '{args.label}' (flags={args.flags}) on {gpu}")
    print(
        f"{args.warmup} warm-up + {per_view * len(VIEWS)} timed frames per configuration\n"
    )

    rows, summary = [], []
    for source in args.datasets:
        if source == "sphere":
            data, spacing, ds = make_test_sphere(), (1.0, 1.0, 1.0), "sphere"
        else:
            (data, spacing), ds = load_volume(source), dataset_name(source)
        presets = args.presets or list(presets_for(ds))
        renderer = Renderer(data, preset_lut(ds, presets[0]), spacing)
        print(
            f"{ds}: {data.shape[2]}x{data.shape[1]}x{data.shape[0]} {data.dtype}, "
            f"spacing {spacing}"
        )
        for preset in presets:
            renderer.set_tf(preset_lut(ds, preset))
            for size in args.sizes:
                cam0 = orbit_camera(
                    VIEWS[0], ELEVATION, DISTANCE, width=size, height=size
                )
                for _ in range(args.warmup):
                    renderer.render(cam0, STEP, flags=args.flags)

                all_times = []
                for az in VIEWS:
                    cam = orbit_camera(az, ELEVATION, DISTANCE, width=size, height=size)
                    times = [
                        renderer.render(cam, STEP, flags=args.flags)[1]
                        for _ in range(per_view)
                    ]
                    all_times += times
                    rows.append(row(args, ds, preset, size, str(az), times, renderer))
                rows.append(row(args, ds, preset, size, "all", all_times, renderer))
                med = float(np.median(all_times))
                summary.append((ds, preset, size, med, renderer.memory_bytes() / 2**20))
                print(
                    f"  {preset:6s} {size:5d}x{size:<5d} median {med:8.2f} ms   "
                    f"{1000 / med:7.1f} FPS   GPU memory {summary[-1][4]:7.1f} MB"
                )

            # One frame per configuration, saved for image comparison between versions
            size = max(args.sizes)
            cam = orbit_camera(
                IMAGE_AZIMUTH, ELEVATION, DISTANCE, width=size, height=size
            )
            img, _ = renderer.render(cam, STEP, flags=args.flags)
            stem = img_dir / f"{args.label}_{ds}_{preset}"
            np.save(f"{stem}.npy", img)
            save_image(img, f"{stem}.png")
            if args.compare:
                compare_images(
                    img, img_dir / f"{args.compare}_{ds}_{preset}.npy", ds, preset
                )
        del renderer, data  # free GPU and host memory before the next dataset
        gc.collect()
        cuda.current_context().deallocations.clear()

    csv_path = out_dir / f"bench_{args.label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path} and images in {img_dir}/")


def row(args, ds, preset, size, view, times, renderer):
    t = np.array(times)
    med = float(np.median(t))
    return {
        "label": args.label,
        "flags": args.flags,
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
        "gpu_mem_mb": round(renderer.memory_bytes() / 2**20, 1),
    }


def compare_images(img, other_path, ds, preset):
    """Print how different this frame is from the same frame of another version."""
    if not Path(other_path).exists():
        print(f"   (no {other_path} to compare with)")
        return
    other = np.load(other_path)
    if other.shape != img.shape:
        print(
            f"   can't compare: {other_path} is {other.shape}, this frame is {img.shape}"
        )
        return
    diff = np.abs(np.clip(img, 0, 1) - np.clip(other, 0, 1))
    print(
        f"   image vs {Path(other_path).stem}: max difference {diff.max():.4f}, "
        f"mean {diff.mean():.6f}, pixels off by > 1/255: {(diff > 1 / 255).mean() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
