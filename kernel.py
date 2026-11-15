"""CUDA ray-marching kernel (ERT + ESS version): one thread per pixel.

The volume stays in its stored type (uint8 / uint16 / float32) and is mapped to
[0, 1] per sample with vmin and vscale.

All arithmetic is float32. Untyped Python float literals are float64 in Numba,
and FP64 runs at 1/64 rate on the RTX 5070, so constants are wrapped in float32().
"""

import math

from numba import cuda, float32

MODE_COMPOSITE = 0
MODE_MIP = 1

FLAG_EARLY_TERMINATION = 1
FLAG_EMPTY_SKIPPING = 2


@cuda.jit(device=True)
def sample(vol, x, y, z):
    """Trilinear interpolation (raw values) at continuous voxel coordinates, clamped to the grid."""
    nz, ny, nx = vol.shape
    zero, one = float32(0.0), float32(1.0)
    x = min(max(x, zero), float32(nx - 1))
    y = min(max(y, zero), float32(ny - 1))
    z = min(max(z, zero), float32(nz - 1))
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
    fx, fy, fz = x - float32(x0), y - float32(y0), z - float32(z0)
    gx, gy, gz = one - fx, one - fy, one - fz
    c00 = vol[z0, y0, x0] * gx + vol[z0, y0, x1] * fx
    c10 = vol[z0, y1, x0] * gx + vol[z0, y1, x1] * fx
    c01 = vol[z1, y0, x0] * gx + vol[z1, y0, x1] * fx
    c11 = vol[z1, y1, x0] * gx + vol[z1, y1, x1] * fx
    c0 = c00 * gy + c10 * fy
    c1 = c01 * gy + c11 * fy
    return c0 * gz + c1 * fz


@cuda.jit(device=True)
def slab(o, d, h, t0, t1):
    """Clip [t0, t1] to the slab -h <= o + t*d <= h on one axis."""
    if abs(d) < float32(1e-9):
        if o < -h or o > h:
            return float32(1.0), float32(0.0)  # parallel and outside: miss
        return t0, t1
    ta = (-h - o) / d
    tb = (h - o) / d
    if ta > tb:
        ta, tb = tb, ta
    return max(t0, ta), min(t1, tb)


@cuda.jit(device=True)
def intersect_box(ox, oy, oz, dx, dy, dz, hx, hy, hz):
    """Ray vs box [-h, h]. Miss if t0 >= t1."""
    t0, t1 = float32(-1e30), float32(1e30)
    t0, t1 = slab(ox, dx, hx, t0, t1)
    t0, t1 = slab(oy, dy, hy, t0, t1)
    t0, t1 = slab(oz, dz, hz, t0, t1)
    return t0, t1


@cuda.jit
def raymarch(vol, lut, cam, box, step, vmin, vscale, mode, flags, img):
    """
    vol   (nz, ny, nx) uint8 / uint16 / float32
    lut   (256, 4) float32 RGBA, opacity already corrected for `step`
    cam   (4, 3) float32 from Camera.as_array()
    box   (3,) float32 half extents
    step  float32 sample spacing in world units
    vmin, vscale  float32, density = (value - vmin) * vscale in [0, 1]
    img   (height, width, 3) float32 output
    """
    px, py = cuda.grid(2)
    height, width = img.shape[0], img.shape[1]
    if px >= width or py >= height:
        return
    zero, half, one = float32(0.0), float32(1.0) * float32(0.5), float32(1.0)
    alpha_max = float32(0.99)
    early_term = (flags & FLAG_EARLY_TERMINATION) != 0
    empty_skip = (flags & FLAG_EMPTY_SKIPPING) != 0
    skip_stride = float32(8.0) * step
    backtrack = float32(7.0) * step

    # ray through the pixel center
    u = float32((px + 0.5) / width * 2.0 - 1.0)
    v = float32((py + 0.5) / height * 2.0 - 1.0)
    dx = cam[1, 0] + u * cam[2, 0] + v * cam[3, 0]
    dy = cam[1, 1] + u * cam[2, 1] + v * cam[3, 1]
    dz = cam[1, 2] + u * cam[2, 2] + v * cam[3, 2]
    length = float32(math.sqrt(dx * dx + dy * dy + dz * dz))
    dx, dy, dz = dx / length, dy / length, dz / length
    ox, oy, oz = cam[0, 0], cam[0, 1], cam[0, 2]

    hx, hy, hz = box[0], box[1], box[2]
    t0, t1 = intersect_box(ox, oy, oz, dx, dy, dz, hx, hy, hz)

    r, g, b, a = zero, zero, zero, zero
    if t0 < t1:
        t0 = max(t0, zero)
        nz, ny, nx = vol.shape
        two = float32(2.0)
        # world -> voxel: (p + h) * s - 0.5
        sx, sy, sz = float32(nx) / (two * hx), float32(ny) / (two * hy), float32(nz) / (two * hz)

        t = t0 + half * step
        in_empty = False
        fine_steps = 0
        while t < t1:
            x = (ox + t * dx + hx) * sx - half
            y = (oy + t * dy + hy) * sy - half
            z = (oz + t * dz + hz) * sz - half

            if empty_skip and fine_steps == 0:
                # 1-tap nearest-neighbor probe instead of 8-tap trilinear sample()
                ix = min(max(int(x + half), 0), nx - 1)
                iy = min(max(int(y + half), 0), ny - 1)
                iz = min(max(int(z + half), 0), nz - 1)
                d_probe = (float32(vol[iz, iy, ix]) - vmin) * vscale

                if mode == MODE_MIP:
                    is_empty = d_probe <= r
                else:
                    ip = min(max(int(d_probe * float32(255.0) + half), 0), 255)
                    is_empty = lut[ip, 3] <= float32(1e-5)

                if is_empty:
                    in_empty = True
                    t += skip_stride
                    continue
                elif in_empty:
                    # Hit non-empty sample after coarse jump: rewind and step finely for 8 steps
                    in_empty = False
                    fine_steps = 8
                    t -= backtrack
                    continue

            if fine_steps > 0:
                fine_steps -= 1

            d = (sample(vol, x, y, z) - vmin) * vscale

            if mode == MODE_MIP:
                r = max(r, d)
                if early_term and r >= one:
                    break
            else:
                # front-to-back compositing
                i = min(max(int(d * float32(255.0) + half), 0), 255)
                w = (one - a) * lut[i, 3]
                r += w * lut[i, 0]
                g += w * lut[i, 1]
                b += w * lut[i, 2]
                a += w
                if early_term and a >= alpha_max:
                    break
            t += step

    if mode == MODE_MIP:
        g = r
        b = r
    img[py, px, 0] = r  # black background
    img[py, px, 1] = g
    img[py, px, 2] = b