"""Volume loading.

load_volume() returns (data, spacing):
    data     ndarray (nz, ny, nx) in the file's own type (uint8 / uint16 / float32 ...)
    spacing  (sx, sy, sz) voxel size, any unit (only the ratios matter)

Supported inputs:
    dataset name   key of DATASETS, e.g. "christmastree"
    *.raw          size and type from the file name, e.g. name_256x256x256_uint8.raw
    *.dat          TU Wien format: 3 x uint16 header (nx, ny, nz), then uint16 voxels
    *.mhd, *.nrrd
    folder         a .dat file inside it, or a stack of slice images (tif/png/jpg/bmp),
                   cached as volume.npy in the folder after the first load
"""

import gzip
import re
from pathlib import Path

import numpy as np

DATASETS = {
    # TU Wien, https://www.cg.tuwien.ac.at/research/vis/datasets/
    "christmastree": {"path": "data/christmastree", "spacing": (1.0, 1.0, 1.0)},
    # DigiMorph, https://digimorph.org/ (1024 x 1024 x 750, 22 mm field, 0.047 mm slices)
    "spathorhynchus": {"path": "data/spathorhynchus", "spacing": (0.0215, 0.0215, 0.047)},
}

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

DTYPES = {
    "uint8": np.uint8, "int8": np.int8,
    "uint16": np.uint16, "int16": np.int16,
    "uint32": np.uint32, "int32": np.int32,
    "float32": np.float32, "float64": np.float64,
    # MetaImage
    "MET_UCHAR": np.uint8, "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
    "MET_UINT": np.uint32, "MET_INT": np.int32,
    "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
    # NRRD
    "unsigned char": np.uint8, "uchar": np.uint8, "signed char": np.int8,
    "unsigned short": np.uint16, "ushort": np.uint16, "short": np.int16,
    "unsigned int": np.uint32, "uint": np.uint32, "int": np.int32,
    "float": np.float32, "double": np.float64,
}


def load_raw(path, shape=None, dtype=None):
    """shape is (nx, ny, nz); if omitted, parsed from the file name."""
    path = Path(path)
    if shape is None or dtype is None:
        m = re.search(r"(\d+)x(\d+)x(\d+)_([a-z]+\d+)", path.name)
        if not m:
            raise ValueError(
                f"Can't read size from '{path.name}'. Name it like "
                "name_256x256x256_uint8.raw, or pass shape=(nx, ny, nz) and dtype."
            )
        shape = shape or tuple(int(g) for g in m.groups()[:3])
        dtype = dtype or m.group(4)
    nx, ny, nz = shape
    data = np.fromfile(path, dtype=DTYPES.get(dtype, dtype))
    if data.size != nx * ny * nz:
        raise ValueError(f"{path.name}: expected {nx*ny*nz} values, file has {data.size}")
    return data.reshape(nz, ny, nx)  # x varies fastest in the file


def load_dat(path):
    """TU Wien .dat: little-endian uint16 nx, ny, nz, then nx*ny*nz uint16 values."""
    with open(path, "rb") as f:
        nx, ny, nz = (int(v) for v in np.fromfile(f, dtype="<u2", count=3))
        data = np.fromfile(f, dtype="<u2", count=nx * ny * nz)
    if data.size != nx * ny * nz:
        raise ValueError(f"{Path(path).name}: header says {nx}x{ny}x{nz}, file is too short")
    return data.reshape(nz, ny, nx)


def load_mhd(path):
    path = Path(path)
    header = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    shape = tuple(int(v) for v in header["DimSize"].split())
    dtype = DTYPES[header["ElementType"]]
    return load_raw(path.parent / header["ElementDataFile"], shape=shape, dtype=dtype)


def load_nrrd(path):
    raw = Path(path).read_bytes()
    split = raw.find(b"\n\n")
    header_text, body = raw[:split].decode("ascii", "replace"), raw[split + 2:]
    header = {}
    for line in header_text.splitlines()[1:]:
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    shape = tuple(int(v) for v in header["sizes"].split())
    dtype = np.dtype(DTYPES[header["type"]])
    if header.get("endian") == "big":
        dtype = dtype.newbyteorder(">")
    encoding = header.get("encoding", "raw")
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"NRRD encoding '{encoding}' is not supported")
    nx, ny, nz = shape
    return np.frombuffer(body, dtype=dtype, count=nx * ny * nz).reshape(nz, ny, nx)


def _natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def load_image_stack(folder):
    """One image per z slice, ordered by the numbers in the file names."""
    from PIL import Image

    files = sorted((p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
                   key=_natural_key)
    if not files:
        raise ValueError(f"No slice images found in {folder}")

    def read(p):
        img = Image.open(p)
        if img.mode in ("RGB", "RGBA", "P", "LA"):
            img = img.convert("L")
        return np.asarray(img)

    first = read(files[0])
    data = np.empty((len(files),) + first.shape, dtype=first.dtype)
    data[0] = first
    for i, p in enumerate(files[1:], start=1):
        data[i] = read(p)
    return data


def load_stack_cached(folder):
    """Image stack, cached as folder/volume.npy after the first (slow) load."""
    cache = Path(folder) / "volume.npy"
    if cache.exists():
        return np.load(cache)
    data = load_image_stack(folder)
    np.save(cache, data)
    return data


def load_path(path, shape=None, dtype=None):
    path = Path(path)
    if path.is_dir():
        dats = sorted(path.rglob("*.dat"))
        return load_dat(dats[0]) if dats else load_stack_cached(path)
    suffix = path.suffix.lower()
    if suffix == ".dat":
        return load_dat(path)
    if suffix == ".mhd":
        return load_mhd(path)
    if suffix == ".nrrd":
        return load_nrrd(path)
    return load_raw(path, shape, dtype)


def load_volume(source, shape=None, dtype=None, spacing=None):
    """source: dataset name or path. Returns (data, spacing)."""
    if source in DATASETS:
        entry = DATASETS[source]
        return load_path(entry["path"]), tuple(spacing or entry["spacing"])
    return load_path(source, shape, dtype), tuple(spacing or (1.0, 1.0, 1.0))


def dataset_name(source):
    if source in DATASETS:
        return source
    return Path(source).stem.split("_")[0].lower()


def value_range(data):
    return float(data.min()), float(data.max())


def normalize(data, lo=None, hi=None):
    """float32 copy scaled to [0, 1]."""
    if lo is None or hi is None:
        lo, hi = value_range(data)
    out = data.astype(np.float32)
    if hi > lo:
        out -= lo
        out /= hi - lo
    else:
        out[:] = 0
    return out


def histogram(data, bins=128):
    """(counts, edges) over normalized density [0, 1], computed one slice at a time."""
    lo, hi = value_range(data)
    scale = bins / (hi - lo) if hi > lo else 0.0
    counts = np.zeros(bins, dtype=np.int64)
    for sl in data:
        idx = ((sl.astype(np.float32) - lo) * scale).astype(np.int64)
        counts += np.bincount(np.minimum(idx, bins - 1).ravel(), minlength=bins)
    return counts, np.linspace(0, 1, bins + 1)


def make_test_sphere(n=128):
    """Synthetic float32 volume: dense core (1.0) inside a softer shell (0.3)."""
    c = (n - 1) / 2
    z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (n / 2)
    vol = np.zeros((n, n, n), np.float32)
    vol[r < 0.8] = 0.3
    vol[r < 0.5] = 1.0
    return vol


def save_slices(data, out_path, title=""):
    """Save the middle slice along each axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = value_range(data)
    nz, ny, nx = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        (data[nz // 2, :, :], f"z = {nz // 2}"),
        (data[:, ny // 2, :], f"y = {ny // 2}"),
        (data[:, :, nx // 2], f"x = {nx // 2}"),
    ]
    for ax, (img, label) in zip(axes, views):
        ax.imshow(img, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    # python volume.py <dataset or path>
    import sys

    Path("results").mkdir(exist_ok=True)
    source = sys.argv[1] if len(sys.argv) > 1 else "christmastree"
    data, spacing = load_volume(source)
    name = dataset_name(source)
    lo, hi = value_range(data)
    print(f"{name}: shape (z, y, x) = {data.shape}, {data.dtype}, {data.nbytes / 2**20:.0f} MB, "
          f"values {lo:g}..{hi:g}, spacing {spacing}")
    out = f"results/{name}_slices.png"
    save_slices(data, out, name)
    print(f"Saved {out}")
