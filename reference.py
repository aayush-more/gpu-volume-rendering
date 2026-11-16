"""CPU reference implementation of the ray marcher, used to validate the GPU kernel.

python reference.py  -> compares CPU and GPU output on the test sphere
(pass: max difference below one 8-bit level, 1/255; the GPU uses float32)
"""

import numpy as np


def sample(vol, p):
    """Trilinear interpolation at voxel coordinates p = (x, y, z)."""
    nz, ny, nx = vol.shape
    x, y, z = (min(max(p[0], 0.0), nx - 1.0),
               min(max(p[1], 0.0), ny - 1.0),
               min(max(p[2], 0.0), nz - 1.0))
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
    fx, fy, fz = x - x0, y - y0, z - z0
    c00 = vol[z0, y0, x0] * (1 - fx) + vol[z0, y0, x1] * fx
    c10 = vol[z0, y1, x0] * (1 - fx) + vol[z0, y1, x1] * fx
    c01 = vol[z1, y0, x0] * (1 - fx) + vol[z1, y0, x1] * fx
    c11 = vol[z1, y1, x0] * (1 - fx) + vol[z1, y1, x1] * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def intersect_box(origin, direction, half):
    """Slab test against [-half, half]. Miss if t0 >= t1."""
    t0, t1 = -np.inf, np.inf
    for o, d, h in zip(origin, direction, half):
        if abs(d) < 1e-9:
            if o < -h or o > h:
                return 1.0, 0.0
            continue
        ta, tb = sorted(((-h - o) / d, (h - o) / d))
        t0, t1 = max(t0, ta), min(t1, tb)
    return t0, t1


def render_reference(vol, lut, camera, step=0.5, mip=False, spacing=(1.0, 1.0, 1.0)):
    """vol in [0, 1]; lut must already be opacity-corrected for `step`."""
    nz, ny, nx = vol.shape
    spacing = np.asarray(spacing, dtype=np.float64)
    extent = np.array([nx, ny, nz]) * spacing
    half = extent / extent.max() / 2
    scale = np.array([nx, ny, nz]) / (2 * half)
    step_world = step * spacing.min() / extent.max()
    img = np.zeros((camera.height, camera.width, 3), np.float32)

    for py in range(camera.height):
        for px in range(camera.width):
            origin, direction = camera.ray_for_pixel(px, py)
            t0, t1 = intersect_box(origin, direction, half)
            color, alpha, brightest = np.zeros(3), 0.0, 0.0
            if t0 < t1:
                t = max(t0, 0.0) + 0.5 * step_world
                while t < t1:
                    voxel = (origin + t * direction + half) * scale - 0.5
                    d = sample(vol, voxel)
                    if mip:
                        brightest = max(brightest, d)
                    else:
                        i = min(max(int(d * 255 + 0.5), 0), 255)
                        w = (1 - alpha) * lut[i, 3]
                        color += w * lut[i, :3]
                        alpha += w
                    t += step_world
            img[py, px] = brightest if mip else color
    return img


if __name__ == "__main__":
    import time

    from camera import orbit_camera
    from kernel import MODE_COMPOSITE, MODE_MIP
    from render import Renderer
    from tf import correct_opacity, preset_lut
    from volume import make_test_sphere

    vol, lut = make_test_sphere(64), preset_lut("sphere", "full")
    cam = orbit_camera(azimuth=30, elevation=20, width=64, height=64)
    for spacing in ((1.0, 1.0, 1.0), (1.0, 1.0, 2.0)):
        renderer = Renderer(vol, lut, spacing)
        for mode, mip, label in ((MODE_COMPOSITE, False, "composite"), (MODE_MIP, True, "MIP")):
            t = time.perf_counter()
            cpu = render_reference(vol, correct_opacity(lut, 0.5), cam, 0.5, mip, spacing)
            cpu_ms = (time.perf_counter() - t) * 1000
            gpu, gpu_ms = renderer.render(cam, mode=mode)
            diff = np.abs(cpu - gpu).max()
            print(f"spacing {spacing}  {label:9s}  CPU {cpu_ms:7.0f} ms   GPU {gpu_ms:6.2f} ms   "
                  f"max diff {diff:.6f}  ->  {'MATCH' if diff < 1 / 255 else 'MISMATCH'}")
