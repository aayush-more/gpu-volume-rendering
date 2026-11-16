"""Transfer functions: 256-entry RGBA lookup tables built from control points.

Control points are (density, r, g, b, opacity), interpolated linearly.
Opacities are per voxel (sample spacing REFERENCE_STEP) and are corrected
for the actual step size with correct_opacity().
"""

import numpy as np

REFERENCE_STEP = 1.0  # voxels

# Presets per dataset: {dataset: {preset: control points}}.
# "full" shows all material above the background, "dense" only the densest material.
PRESETS = {
    "christmastree": {
        "full": [
            (0.00, 0.0, 0.0, 0.0, 0.00),
            (0.03, 0.0, 0.0, 0.0, 0.00),
            (0.06, 0.2, 0.6, 0.25, 0.03),
            (0.20, 0.5, 0.8, 0.3, 0.15),
            (0.35, 1.0, 0.85, 0.4, 0.40),
            (1.00, 1.0, 1.0, 1.0, 0.80),
        ],
        "dense": [  # ornaments and metal
            (0.00, 0.0, 0.0, 0.0, 0.00),
            (0.22, 0.0, 0.0, 0.0, 0.00),
            (0.30, 1.0, 0.8, 0.35, 0.30),
            (1.00, 1.0, 1.0, 1.0, 0.80),
        ],
    },
    "spathorhynchus": {
        "full": [  # rock matrix (~0.38-0.49) faint, bone (>0.49) translucent
            (0.00, 0.0, 0.0, 0.0, 0.000),
            (0.36, 0.0, 0.0, 0.0, 0.000),
            (0.42, 0.75, 0.45, 0.25, 0.002),
            (0.47, 0.75, 0.45, 0.25, 0.003),
            (0.51, 0.95, 0.92, 0.85, 0.015),
            (0.65, 1.0, 1.0, 1.0, 0.050),
            (1.00, 1.0, 1.0, 1.0, 0.200),
        ],
        "dense": [  # bone surface only
            (0.00, 0.0, 0.0, 0.0, 0.00),
            (0.49, 0.0, 0.0, 0.0, 0.00),
            (0.53, 0.95, 0.92, 0.85, 0.08),
            (0.65, 1.0, 1.0, 1.0, 0.25),
            (1.00, 1.0, 1.0, 1.0, 0.40),
        ],
    },
    "sphere": {
        "full": [
            (0.00, 0.0, 0.0, 0.0, 0.00),
            (0.20, 0.0, 0.0, 0.0, 0.00),
            (0.30, 0.9, 0.6, 0.5, 0.10),
            (1.00, 1.0, 1.0, 1.0, 0.60),
        ],
        "dense": [
            (0.00, 0.0, 0.0, 0.0, 0.00),
            (0.60, 0.0, 0.0, 0.0, 0.00),
            (1.00, 1.0, 1.0, 1.0, 0.60),
        ],
    },
}
DEFAULT = "christmastree"


def build_lut(points):
    points = np.asarray(sorted(points), dtype=np.float32)
    density = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.empty((256, 4), dtype=np.float32)
    for c in range(4):
        lut[:, c] = np.interp(density, points[:, 0], points[:, 1 + c])
    return lut


def presets_for(dataset):
    """Preset table for a dataset (falls back to DEFAULT for unknown data)."""
    return PRESETS.get(dataset, PRESETS[DEFAULT])


def preset_lut(dataset, preset, opacity_scale=1.0):
    lut = build_lut(presets_for(dataset)[preset])
    lut[:, 3] = np.clip(lut[:, 3] * opacity_scale, 0.0, 1.0)
    return lut


def correct_opacity(lut, step):
    """alpha' = 1 - (1 - alpha)^(step / REFERENCE_STEP)"""
    out = lut.copy()
    out[:, 3] = 1.0 - (1.0 - lut[:, 3]) ** (step / REFERENCE_STEP)
    return out


def ramp_lut(cutoff=0.15, opacity=0.12):
    """Grayscale ramp, transparent below cutoff."""
    return build_lut([(0.0, 0, 0, 0, 0), (cutoff, 0, 0, 0, 0), (1.0, 1, 1, 1, opacity)])


def plot_tf(vol, lut, path, title=""):
    """Opacity curve (colored by the TF color) over the log-scale density histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from volume import histogram

    density = np.linspace(0, 1, 256)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, edges = histogram(vol, bins=128)
    ax.bar(edges[:-1], np.maximum(counts, 1), width=edges[1] - edges[0], align="edge",
           color="lightgray", log=True, label="voxels (log scale)")
    ax.set_xlabel("density")
    ax.set_ylabel("number of voxels")
    ax2 = ax.twinx()
    ax2.plot(density, lut[:, 3], color="black", linewidth=0.8, label="opacity")
    ax2.scatter(density, lut[:, 3], c=np.clip(lut[:, :3], 0, 1), s=28,
                edgecolors="gray", linewidths=0.3, zorder=3)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("opacity per voxel")
    ax2.legend(loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    # python tf.py [dataset or path]  -> results/tf_<dataset>_<preset>.png
    import sys
    from pathlib import Path

    from volume import dataset_name, load_volume

    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    data, _ = load_volume(source)
    name = dataset_name(source)
    Path("results").mkdir(exist_ok=True)
    for preset in presets_for(name):
        out = f"results/tf_{name}_{preset}.png"
        plot_tf(data, preset_lut(name, preset), out, f"{name}: {preset}")
        print(f"Saved {out}")
