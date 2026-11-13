"""Deterministic local point displacement and shared physical-frame transform."""

import numpy as np


def shrink_ring_hole(points: np.ndarray, factor: float = 0.65):
    """Move inner radial points towards a PCA-fitted ring axis with smooth falloff."""
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 10 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite Nx3 ring points")
    if not 0 < factor < 1:
        raise ValueError("Ring hole factor must lie in (0, 1)")
    center = np.median(xyz, axis=0)
    _, axes = np.linalg.eigh(np.cov((xyz - center).T))
    axis = axes[:, 0]
    relative = xyz - center
    radial = relative - np.outer(relative @ axis, axis)
    radius = np.linalg.norm(radial, axis=1)
    inner, outer = np.quantile(radius, [0.25, 0.65])
    weight = np.clip((outer - radius) / max(outer - inner, 1e-12), 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    result = xyz - radial * ((1 - factor) * weight[:, None])
    return result, {
        "center": center.tolist(),
        "axis": axis.tolist(),
        "inner_radius": float(inner),
        "outer_radius": float(outer),
        "factor": factor,
        "changed_points": int(np.count_nonzero(weight)),
    }


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
