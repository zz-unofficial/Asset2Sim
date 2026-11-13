"""Deterministic local point displacement and shared physical-frame transform."""

import numpy as np


def displace_positive_x(points: np.ndarray, magnitude: float = 0.12, quantile: float = 0.8):
    """Move the positive-X tail with a smooth compact weight; preserve other points exactly."""
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite nonempty Nx3 point cloud")
    if not np.isfinite(magnitude) or magnitude < 0 or not 0 < quantile < 1:
        raise ValueError("Invalid edit magnitude or quantile")
    threshold = float(np.quantile(xyz[:, 0], quantile))
    weight = np.clip((xyz[:, 0] - threshold) / max(float(xyz[:, 0].max()) - threshold, 1e-9), 0, 1)
    result = xyz.copy()
    result[:, 0] += magnitude * weight * weight * (3 - 2 * weight)
    return result, threshold


def shared_frame_transform(control: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """Map an independently normalized edit into the control's physical frame."""
    a, b = np.asarray(control, dtype=float), np.asarray(edited, dtype=float)
    if (
        a.shape != (4, 4)
        or b.shape != (4, 4)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Expected finite 4x4 transforms")
    return a @ np.linalg.inv(b)
