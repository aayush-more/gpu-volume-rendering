"""Merge every results/bench_*.csv (+ results/quality.csv) into one results table and charts.

For each (dataset, preset, resolution) it reports, per version: frame time (median over all
timed frames), FPS, speed-up over naive, GPU memory, and SSIM / PSNR against naive.

    python3 vtk_compare/summarize.py
    python3 vtk_compare/summarize.py --exclude v1 --size 1024
    python3 vtk_compare/summarize.py --order naive ert ess ert_ess bricking vtk

Outputs (in results/):  summary.csv   summary.md   summary_speedup.png   summary_time.png
Run it on the machine's final results; it only reads CSV files, no GPU needed.

Timing note: CUDA versions are kernel-only (CUDA events). The VTK row uses net_ms
(Render() + 1-pixel readback minus an empty-scene baseline), the closest match available.
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORDER = ["naive", "ert", "ess", "ert_ess"]
NICE = {"naive": "Naive", "ert": "ERT", "ess": "ESS", "ert_ess": "ERT+ESS",
        "bricking": "Bricking", "vtk": "VTK"}
# categorical slots 1, 2 from the reference palette (blue, orange): full, dense preset
PRESET_COLOR = {"full": "#257de0", "dense": "#f47230"}
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_bench(res_dir, exclude):
    rows = []
    for path in sorted(glob.glob(str(res_dir / "bench_*.csv"))):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["label"] in exclude or r["view_azimuth"] != "all":
                    continue
                is_vtk = r.get("net_ms") not in (None, "")
                t = num(r["net_ms"]) if is_vtk else num(r["median_ms"])
                rows.append({
                    "label": r["label"], "dataset": r["dataset"], "preset": r["preset"],
                    "size": int(r["width"]), "frames": int(r["frames"]),
                    "time_ms": t, "fps": 1000 / t if t else None,
                    "mem_mb": num(r["gpu_mem_mb"]),
                })
    return rows


def load_quality(res_dir):
    q = {}
    p = res_dir / "quality.csv"
    if p.exists():
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                q[(r["label"], r["dataset"], r["preset"], int(r["width"]))] = r
    return q


def fmt(x, nd=2, dash="-"):
    if x is None:
        return dash
    if x == float("inf"):
        return "identical"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results")
    ap.add_argument("--exclude", nargs="*", default=["v1"], help="labels to ignore (v1 = naive duplicate)")
    ap.add_argument("--order", nargs="*", help="version order; default naive ert ess ert_ess, others, vtk")
    ap.add_argument("--size", type=int, default=1024, help="resolution used in the charts")
    args = ap.parse_args()

    os.chdir(ROOT)
    res = Path(args.dir)
    rows = load_bench(res, set(args.exclude))
    if not rows:
        raise SystemExit(f"No bench_*.csv with an 'all' row in {res}/")
    quality = load_quality(res)

    labels = {r["label"] for r in rows}
    order = args.order or (
        [l for l in DEFAULT_ORDER if l in labels]
        + sorted(labels - set(DEFAULT_ORDER) - {"vtk"}) + (["vtk"] if "vtk" in labels else []))
    base = {(r["dataset"], r["preset"], r["size"]): r["time_ms"] for r in rows if r["label"] == "naive"}

    out = []
    for r in rows:
        key = (r["dataset"], r["preset"], r["size"])
        q = quality.get((r["label"], r["dataset"], r["preset"], r["size"]))
        # quality frames are saved at the largest size only -> match on any size for this label
        if q is None:
            q = next((v for k, v in quality.items() if k[:3] == (r["label"], r["dataset"], r["preset"])), None)
            if q is not None and int(q["width"]) != r["size"]:
                q = None
        r = dict(r)
        r["speedup"] = base[key] / r["time_ms"] if key in base and r["time_ms"] else None
        r["ssim_object"] = num(q["ssim_object"]) if q else None
        r["psnr_db"] = num(q["psnr_db"]) if q else None
        out.append(r)

    def sort_key(r):
        return (r["dataset"], r["preset"], -r["size"], order.index(r["label"]) if r["label"] in order else 99)
    out.sort(key=sort_key)

    cols = ["dataset", "preset", "size", "label", "frames", "time_ms", "fps", "speedup", "mem_mb", "ssim_object", "psnr_db"]
    with open(res / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) and np.isfinite(r[c]) else r[c]) for c in cols})

    # markdown: one table per dataset / preset / resolution
    md = ["# Results summary", "",
          "Time = median per-frame time over all timed frames (8 views). Speed-up = naive time / version time. "
          "SSIM and PSNR compare each frame to the naive frame (same camera, transfer function, step); "
          "SSIM is averaged over object pixels only. VTK time = render minus empty-scene baseline.", ""]
    for ds, pr, size in sorted({(r["dataset"], r["preset"], r["size"]) for r in out}, key=lambda k: (k[0], k[1], -k[2])):
        md += [f"## {ds} - {pr} preset - {size}x{size}", "",
               "| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for r in (x for x in out if (x["dataset"], x["preset"], x["size"]) == (ds, pr, size)):
            md.append(f"| {NICE.get(r['label'], r['label'])} | {fmt(r['time_ms'])} | {fmt(r['fps'], 1)} | "
                      f"{fmt(r['speedup'])}x | {fmt(r['mem_mb'], 0)} | {fmt(r['ssim_object'], 4)} | {fmt(r['psnr_db'], 1)} |"
                      if r["speedup"] is not None else
                      f"| {NICE.get(r['label'], r['label'])} | {fmt(r['time_ms'])} | {fmt(r['fps'], 1)} | - | {fmt(r['mem_mb'], 0)} | "
                      f"{fmt(r['ssim_object'], 4)} | {fmt(r['psnr_db'], 1)} |")
        md.append("")
    (res / "summary.md").write_text("\n".join(md))

    charts(out, order, args.size, res)
    print("\n".join(md))
    print(f"Saved {res}/summary.csv, summary.md, summary_speedup.png, summary_time.png")


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(rows, order, size, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows if r["size"] == size})
    if not datasets:
        return
    labels = [l for l in order if any(r["label"] == l for r in rows)]
    presets = [p for p in ("full", "dense") if any(r["preset"] == p for r in rows)]

    for metric, fname, ylabel, title, log in (
        ("speedup", "summary_speedup.png", "Speed-up over naive (x)", "Speed-up over naive", False),
        ("time_ms", "summary_time.png", "Frame time (ms, log scale)", "Frame time", True),
    ):
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.2), squeeze=False, sharey=True)
        fig.patch.set_facecolor("white")
        for ax, ds in zip(axes[0], datasets):
            style(ax)
            width = 0.8 / len(presets)
            for pi, pr in enumerate(presets):
                xs, ys = [], []
                for li, lab in enumerate(labels):
                    v = next((r[metric] for r in rows if (r["dataset"], r["preset"], r["size"], r["label"]) == (ds, pr, size, lab)), None)
                    if v is not None:
                        xs.append(li + (pi - (len(presets) - 1) / 2) * width)
                        ys.append(v)
                bars = ax.bar(xs, ys, width=width - 0.03, color=PRESET_COLOR[pr], label=f"{pr} preset", zorder=3)
                for x, y in zip(xs, ys):
                    ax.text(x, y, f"{y:.1f}" + ("x" if metric == "speedup" else ""), ha="center", va="bottom",
                            fontsize=8, color=INK2)
            if metric == "speedup":
                ax.axhline(1, color=INK2, linewidth=0.8, zorder=4)
            if log:
                ax.set_yscale("log")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([NICE.get(l, l) for l in labels], color=INK)
            ax.set_title(ds, loc="left", fontsize=11, color=INK)
        axes[0][0].set_ylabel(ylabel, color=INK2)
        axes[0][-1].legend(frameon=False, labelcolor=INK2, fontsize=9)
        fig.suptitle(f"{title} - {size}x{size}, step 0.5 voxel", x=0.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout()
        fig.savefig(res / fname, dpi=200, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
