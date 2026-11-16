"""Orbit camera looking at the origin, +z up.

World space: the volume box is centered at the origin with its longest side = 1.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    eye: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    width: int
    height: int
    half_w: float  # half screen width at distance 1
    half_h: float

    def ray_for_pixel(self, px, py):
        """(origin, unit direction) through pixel (px, py); (0, 0) is bottom-left."""
        u = (px + 0.5) / self.width * 2 - 1
        v = (py + 0.5) / self.height * 2 - 1
        d = self.forward + u * self.half_w * self.right + v * self.half_h * self.up
        return self.eye, d / np.linalg.norm(d)

    def as_array(self):
        """(4, 3) float32 for the kernel: eye, forward, right*half_w, up*half_h."""
        return np.array([self.eye, self.forward,
                         self.right * self.half_w, self.up * self.half_h], dtype=np.float32)


def orbit_camera(azimuth=30.0, elevation=20.0, distance=2.0, fov=40.0, width=512, height=512):
    """azimuth/elevation/fov in degrees."""
    elevation = float(np.clip(elevation, -89.0, 89.0))  # avoid forward parallel to up
    az, el = np.radians(azimuth), np.radians(elevation)
    eye = distance * np.array([np.cos(el) * np.cos(az),
                               np.cos(el) * np.sin(az),
                               np.sin(el)])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_h = np.tan(np.radians(fov) / 2)
    half_w = half_h * width / height
    return Camera(eye, forward, right, up, width, height, half_w, half_h)
