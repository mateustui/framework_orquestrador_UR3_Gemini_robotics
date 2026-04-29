from __future__ import annotations

import numpy as np


def rx(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_rotation(a: float, b: float, g: float) -> np.ndarray:
    return rx(a) @ ry(b) @ rz(g)


def compute_intrinsics(w: int, h: int, fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    fov_rad = np.deg2rad(fov_deg)
    fx = (w / 2.0) / np.tan(fov_rad / 2.0)
    k = np.array([[-fx, 0, w / 2], [0, -fx, h / 2], [0, 0, 1]])
    return k, np.linalg.inv(k)


def triangulate(
    k_inv: np.ndarray,
    r1: np.ndarray,
    t1: np.ndarray,
    px1: tuple[int, int],
    r2: np.ndarray,
    t2: np.ndarray,
    px2: tuple[int, int],
) -> tuple[np.ndarray, float]:
    d1 = r1 @ (k_inv @ np.array([px1[0], px1[1], 1.0]))
    d2 = r2 @ (k_inv @ np.array([px2[0], px2[1], 1.0]))

    a = np.column_stack([d1, -d2])
    b = t2 - t1
    x = np.linalg.pinv(a) @ b

    p1 = t1 + x[0] * d1
    p2 = t2 + x[1] * d2

    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2) * 1000)
