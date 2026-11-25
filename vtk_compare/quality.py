"""Image quality of every version against the naive renderer (the reference).

Reads results/images/<label>_<dataset>_<preset>.npy (the frames bench.py and
vtk_compare.py save: (h, w, 3) float32 in [0, 1], same camera, same transfer function)
and writes results/quality.csv with, per (label, dataset, preset):

    ssim_full     SSIM over the whole frame. The black background is identical in both
                  images, so this number is inflated for small objects.
    ssim_object   SSIM averaged only over pixels where either image is not black.
                  This is the number to report.
    psnr_db       PSNR over the whole frame (inf = identical).
    max_abs, mean_abs, pct_off_1_255   plain pixel differences.

No CUDA needed: run it on any machine that has the .npy files.

    python3 vtk_compare/quality.py                       # every label found vs naive
    python3 vtk_compare/quality.py --ref naive --labels ert ess ert_ess vtk
    python3 vtk_compare/quality.py --self-test           # checks the metric code
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("christmastree", "spathorhynchus", "sphere")
PRESETS = ("full", "dense")
SKIP_LABELS = {"v1"}  # v1 is the same renderer as naive


def _uniform(a, size):
    """Mean over a size x size window (valid region), via a summed-area table."""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2))
    s = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    return s / (size * size)


def _ssim_map_numpy(a, b, win=7):
    """Same formula as skimage.metrics.structural_similarity defaults (uniform 7x7
    window, K1=0.01, K2=0.03, sample covariance, data_range=1)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    n = win * win
    cov_norm = n / (n - 1)
    ux, uy = _uniform(a, win), _uniform(b, win)
    uxx, uyy, uxy = _uniform(a * a, win), _uniform(b * b, win), _uniform(a * b, win)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    return s.mean(axis=-1)  # average over the colour channels -> (h-6, w-6)


def ssim_map(a, b, win=7):
    """Per-pixel SSIM (h-6, w-6) averaged over RGB; the window edge is cropped."""
    try:
        from skimage.metrics import structural_similarity

        _, m = structural_similarity(a, b, channel_axis=2, data_range=1.0, full=True, win_size=win)
        pad = win // 2
        return m.mean(axis=-1)[pad:-pad, pad:-pad]
    except ImportError:
        return _ssim_map_numpy(a, b, win)


def compare(ref, img, win=7):
    ref = np.clip(ref.astype(np.float32), 0, 1)
    img = np.clip(img.astype(np.float32), 0, 1)
    diff = np.abs(ref - img)
    mse = float(np.mean((ref - img) ** 2))
    m = ssim_map(ref, img, win)
    pad = win // 2
    fg = (np.maximum(ref.max(-1), img.max(-1)) > 1 / 255)[pad:-pad, pad:-pad]
    return {
        "ssim_full": float(m.mean()),
        "ssim_object": float(m[fg].mean()) if fg.any() else float("nan"),
        "object_pct": float(fg.mean() * 100),
        "psnr_db": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "pct_off_1_255": float((diff > 1 / 255).mean() * 100),
    }


def parse_stem(stem):
    """'ert_ess_christmastree_full' -> ('ert_ess', 'christmastree', 'full')."""
    for ds in DATASETS:
        for pr in PRESETS:
            suffix = f"_{ds}_{pr}"
            if stem.endswith(suffix):
                return stem[: -len(suffix)], ds, pr
    return None


def self_test():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    r = compare(a, a)
    assert r["ssim_full"] > 0.9999 and r["psnr_db"] == float("inf"), r
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    r = compare(a, b)
    assert 0.05 < r["ssim_full"] < 0.99 and 20 < r["psnr_db"] < 40, r
    m1, m2 = _ssim_map_numpy(a, b), None
    try:
        from skimage.metrics import structural_similarity as ss

        m2 = ss(a, b, channel_axis=2, data_range=1.0, full=True)[1].mean(-1)[3:-3, 3:-3]
        assert np.allclose(m1, m2, atol=1e-6), np.abs(m1 - m2).max()
        print("numpy SSIM matches skimage")
    except ImportError:
        print("skimage not installed - numpy SSIM used")
    print("self-test OK", {k: round(v, 4) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="naive", help="label of the reference renderer")
    ap.add_argument("--labels", nargs="*", help="labels to score (default: all found)")
    ap.add_argument("--dir", default="results", help="results folder (relative to the repo root)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    os.chdir(ROOT)
    img_dir = Path(args.dir) / "images"
    frames = {}
    for p in sorted(img_dir.glob("*.npy")):
        parsed = parse_stem(p.stem)
        if parsed:
            frames[parsed] = p
    if not frames:
        sys.exit(f"No frames found in {img_dir}/ (expected <label>_<dataset>_<preset>.npy)")

    labels = args.labels or sorted({k[0] for k in frames} - {args.ref} - SKIP_LABELS)
    rows = []
    print(f"Reference: {args.ref}")
    print(f"{'label':10s} {'dataset':15s} {'preset':6s} {'SSIM obj':>9s} {'SSIM full':>10s} {'PSNR dB':>8s} {'max|d|':>7s} {'>1/255 %':>9s}")
    for lab in labels:
        for (l, ds, pr), path in sorted(frames.items()):
            if l != lab:
                continue
            ref_path = frames.get((args.ref, ds, pr))
            if ref_path is None:
                print(f"{lab:10s} {ds:15s} {pr:6s}  (no {args.ref} frame to compare with)")
                continue
            ref, img = np.load(ref_path), np.load(path)
            if ref.shape != img.shape:
                print(f"{lab:10s} {ds:15s} {pr:6s}  shape {img.shape} != reference {ref.shape}, skipped")
                continue
            r = compare(ref, img)
            rows.append({"label": lab, "ref": args.ref, "dataset": ds, "preset": pr,
                         "width": img.shape[1], "height": img.shape[0],
                         **{k: round(v, 6) if np.isfinite(v) else v for k, v in r.items()}})
            print(f"{lab:10s} {ds:15s} {pr:6s} {r['ssim_object']:9.4f} {r['ssim_full']:10.4f} "
                  f"{r['psnr_db']:8.2f} {r['max_abs']:7.3f} {r['pct_off_1_255']:9.2f}")
    if not rows:
        sys.exit("Nothing was compared.")
    out = Path(args.dir) / "quality.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
