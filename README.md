# GPU Volume Rendering

Direct volume rendering of 3D scalar fields with ray marching in Python + CUDA (Numba).
Naive implementation with an editable transfer function.

## Results

RTX 5070 Laptop GPU (8 GB), step 0.5 voxel, median over 104 frames from 8 views.

| Dataset | Preset | 512×512 | 1024×1024 |
| --- | --- | --- | --- |
| Christmas Tree 512×499×512 | full | 39.13 ms (25.6 FPS) | 136.84 ms (7.3 FPS) |
| Christmas Tree 512×499×512 | dense | 39.28 ms (25.5 FPS) | 137.70 ms (7.3 FPS) |
| Spathorhynchus 1024×1024×750 | full | 60.69 ms (16.5 FPS) | 197.91 ms (5.1 FPS) |
| Spathorhynchus 1024×1024×750 | dense | 60.04 ms (16.7 FPS) | 197.53 ms (5.1 FPS) |

GPU memory at 512² / 1024²: 253 / 262 MB (Christmas Tree), 1503 / 1512 MB (Spathorhynchus).
The naive renderer marches every ray through the whole volume, so the preset has no effect on time.

Full data: `results/bench_v1.csv`.

## Setup

Windows, NVIDIA GPU, Python 3.12, driver with CUDA 13 support.

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Creates `.venv`, installs dependencies and runs `check_gpu.py`.
NumPy is pinned below 2.5 because `numba-cuda` 0.30 uses `np.row_stack`.

## Usage

| Command | Output |
| --- | --- |
| `python viewer.py christmastree` | Interactive viewer |
| `python render.py christmastree` | `results\views_christmastree.png` |
| `python tf.py christmastree` | `results\tf_christmastree_<preset>.png` |
| `python reference.py` | CPU vs GPU validation |
| `python bench.py` | `results\bench_v1.csv`, `results\images\` |

`bench.py --label <name> --flags <n> --compare v1` benchmarks a variant and reports image differences against the baseline.

## Method

One thread per pixel: ray–box intersection (slab method), fixed-step marching (0.5 voxel) with trilinear sampling, 256-entry RGBA transfer function, front-to-back compositing. Opacity is corrected for step size as `1 - (1 - α)^(Δt / 1 voxel)`. Anisotropic voxel spacing is supported. The volume is kept in its stored type (uint8 / uint16) on the GPU; all kernel arithmetic is float32.

## Files

| File | |
| --- | --- |
| `volume.py` | Loading (`.dat`, `.raw`, `.mhd`, `.nrrd`, image stacks), dataset list, test sphere |
| `camera.py` | Orbit camera |
| `kernel.py` | CUDA kernel: `raymarch`, `sample`, `intersect_box` |
| `tf.py` | Transfer functions and presets |
| `render.py` | `Renderer` class |
| `reference.py` | CPU reference renderer |
| `viewer.py` | Interactive viewer |
| `bench.py` | Benchmark |
| `check_gpu.py`, `basic_kernels.py` | GPU check, basic kernels |

## Data

| Name | Source | Size | Folder |
| --- | --- | --- | --- |
| `christmastree` | [TU Wien](https://www.cg.tuwien.ac.at/research/vis/datasets/) | 512×499×512, 12-bit | `data\christmastree\` (`.dat`) |
| `spathorhynchus` | [DigiMorph](https://digimorph.ctlab.geo.utexas.edu/specimens/Spathorhynchus_fossorium) | 1024×1024×750, spacing 0.0215×0.0215×0.047 mm | `data\spathorhynchus\` (slice images) |

Download and unzip into `data\`:

- Christmas Tree: on the TU Wien page, open "Christmas tree" and download the 512x499x512 zip into `data\christmastree\`.
- Spathorhynchus: on the DigiMorph page, download "Original HRXCT scan data" (`Spathorhynchus.zip`, 1.0 GB) into `data\spathorhynchus\`. The first load caches the slices as `volume.npy` (about 50 s once).

Scripts take a dataset name or a file/folder path.

# Early Ray Termination & Empty Space Skipping

kernel.py updated to support Early Ray Termination (ERT) and Empty Space Skipping (ESS) for optimization. 

Updated render.py, viewer.py and bench.py to accept flags as second runtime argument which controls optimization used: <br>
1. 0 -> Naive Implementation
2. 1 -> Early Ray Termination
3. 2 -> Empty Space Skipping
4. 3 -> Early Ray Termination and Empty Space Skipping

### Benchmark Results (NVIDIA GeForce RTX 4070 Laptop GPU)

Commands executed to compute benchmarks:<br>

python bench.py --label naive --flags 0 <br>
python bench.py --label ert --flags 1 --compare naive <br>
python bench.py --label ess --flags 2 --compare naive <br>
python bench.py --label ert_ess --flags 3 --compare naive <br>

| Dataset | Preset | Resolution | Naive (ms / FPS) | ERT (ms / FPS) | ESS (ms / FPS) | ERT + ESS (ms / FPS) | Speedup (Naive → ERT+ESS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Christmas Tree** <br>| `full` | 512×512 | 39.65 ms (25.2 FPS) | 39.61 ms (25.2 FPS) | 4.49 ms (222.5 FPS) | **4.16 ms (240.5 FPS)** | **9.5x** |
| | `full` | 1024×1024 | 148.05 ms (6.8 FPS) | 146.69 ms (6.8 FPS) | 11.15 ms (89.7 FPS) | **10.46 ms (95.6 FPS)** | **14.2x** |
| | `dense` | 512×512 | 39.72 ms (25.2 FPS) | 39.71 ms (25.2 FPS) | 3.01 ms (332.7 FPS) | **2.87 ms (348.3 FPS)** | **13.8x** |
| | `dense` | 1024×1024 | 148.55 ms (6.7 FPS) | 147.38 ms (6.8 FPS) | 4.27 ms (234.0 FPS) | **4.07 ms (245.7 FPS)** | **36.5x** |
| **Spathorhynchus** <br>| `full` | 512×512 | 53.27 ms (18.8 FPS) | 50.88 ms (19.7 FPS) | 18.00 ms (55.5 FPS) | **15.53 ms (64.4 FPS)** | **3.4x** |
| | `full` | 1024×1024 | 187.74 ms (5.3 FPS) | 177.64 ms (5.6 FPS) | 43.93 ms (22.8 FPS) | **37.95 ms (26.4 FPS)** | **4.9x** |
| | `dense` | 512×512 | 52.68 ms (19.0 FPS) | 44.72 ms (22.4 FPS) | 17.27 ms (57.9 FPS) | **7.79 ms (128.4 FPS)** | **6.8x** |
| | `dense` | 1024×1024 | 187.74 ms (5.3 FPS) | 148.05 ms (6.8 FPS) | 41.95 ms (23.8 FPS) | **16.75 ms (59.7 FPS)** | **11.2x** |