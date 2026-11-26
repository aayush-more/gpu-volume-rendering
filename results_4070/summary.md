# Results summary

Time = median per-frame time over all timed frames (8 views). Speed-up = naive time / version time. SSIM and PSNR compare each frame to the naive frame (same camera, transfer function, step); SSIM is averaged over object pixels only. VTK time = render minus empty-scene baseline.

## christmastree - dense preset - 1024x1024

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 148.99 | 6.7 | 1.00x | 262 | - | - |
| ERT | 148.80 | 6.7 | 1.00x | 262 | 0.9999 | 64.5 |
| ESS | 4.31 | 232.1 | 34.58x | 262 | 0.8638 | 34.7 |
| ERT+ESS | 4.04 | 247.6 | 36.89x | 262 | 0.8637 | 34.7 |
| VTK | 8.38 | 119.3 | 17.77x | 257 | 0.9396 | 50.8 |

## christmastree - dense preset - 512x512

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 40.25 | 24.8 | 1.00x | 252 | - | - |
| ERT | 40.25 | 24.8 | 1.00x | 252 | - | - |
| ESS | 3.03 | 330.0 | 13.28x | 252 | - | - |
| ERT+ESS | 2.88 | 347.0 | 13.97x | 252 | - | - |
| VTK | 5.10 | 196.2 | 7.90x | 257 | - | - |

## christmastree - full preset - 1024x1024

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 148.66 | 6.7 | 1.00x | 262 | - | - |
| ERT | 147.00 | 6.8 | 1.01x | 262 | 0.9999 | 61.7 |
| ESS | 11.21 | 89.2 | 13.26x | 262 | 0.9723 | 41.0 |
| ERT+ESS | 10.52 | 95.1 | 14.14x | 262 | 0.9722 | 41.0 |
| VTK | 8.35 | 119.7 | 17.80x | 257 | 0.9793 | 39.1 |

## christmastree - full preset - 512x512

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 40.19 | 24.9 | 1.00x | 252 | - | - |
| ERT | 39.74 | 25.2 | 1.01x | 252 | - | - |
| ESS | 4.32 | 231.5 | 9.31x | 252 | - | - |
| ERT+ESS | 4.08 | 244.9 | 9.84x | 252 | - | - |
| VTK | 5.37 | 186.1 | 7.48x | 257 | - | - |

## spathorhynchus - dense preset - 1024x1024

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 189.25 | 5.3 | 1.00x | 1512 | - | - |
| ERT | 148.31 | 6.7 | 1.28x | 1512 | 0.9996 | 50.0 |
| ESS | 41.92 | 23.9 | 4.52x | 1512 | 0.9944 | 53.9 |
| ERT+ESS | 16.70 | 59.9 | 11.33x | 1512 | 0.9940 | 49.4 |
| VTK | 22.09 | 45.3 | 8.57x | 1510 | 0.9982 | 54.9 |

## spathorhynchus - dense preset - 512x512

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 55.16 | 18.1 | 1.00x | 1503 | - | - |
| ERT | 45.05 | 22.2 | 1.22x | 1503 | - | - |
| ESS | 17.32 | 57.7 | 3.19x | 1503 | - | - |
| ERT+ESS | 8.04 | 124.3 | 6.86x | 1503 | - | - |
| VTK | 13.80 | 72.5 | 4.00x | 1510 | - | - |

## spathorhynchus - full preset - 1024x1024

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 191.24 | 5.2 | 1.00x | 1512 | - | - |
| ERT | 178.18 | 5.6 | 1.07x | 1512 | 0.9993 | 54.7 |
| ESS | 44.04 | 22.7 | 4.34x | 1512 | 1.0000 | 75.6 |
| ERT+ESS | 37.77 | 26.5 | 5.06x | 1512 | 0.9992 | 54.7 |
| VTK | 30.88 | 32.4 | 6.19x | 1510 | 0.9991 | 61.5 |

## spathorhynchus - full preset - 512x512

| Version | Time (ms) | FPS | Speed-up | GPU memory (MB) | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 53.67 | 18.6 | 1.00x | 1503 | - | - |
| ERT | 51.23 | 19.5 | 1.05x | 1503 | - | - |
| ESS | 18.22 | 54.9 | 2.94x | 1503 | - | - |
| ERT+ESS | 15.67 | 63.8 | 3.42x | 1503 | - | - |
| VTK | 17.44 | 57.3 | 3.08x | 1510 | - | - |
