"""Independent analytic distances to the original procedural debug solids."""

from __future__ import annotations

import numpy as np


def _polygon_distance(point: np.ndarray, radius: float) -> tuple[bool, float]:
    angles = np.arange(64) * (2 * np.pi / 64)
    vertices = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    edge = np.roll(vertices, -1, axis=0) - vertices
    relative = point - vertices
    cross = edge[:, 0] * relative[:, 1] - edge[:, 1] * relative[:, 0]
    t = np.clip(np.sum(relative * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
    distance = np.linalg.norm(relative - t[:, None] * edge, axis=1).min()
    return bool(np.all(cross >= -1e-14)), float(distance)


def distance_to_solid(object_id: str, position: np.ndarray) -> float:
    """Distance in meters from a point to the unmodified procedural solid.

    Debug references are centered and normalized to longest extent 0.20 m by
    the dataset preparation. The annulus has a 64-sided cross-section and its
    axis is Y; the box extents are [0.15, 0.10, 0.20] m.
    """
    p = np.asarray(position, dtype=float)
    if object_id == "debug_box":
        return float(np.linalg.norm(np.maximum(np.abs(p) - [0.075, 0.05, 0.10], 0)))
    if object_id == "debug_ring":
        planar = p[[0, 2]]
        in_outer, outer_distance = _polygon_distance(planar, 0.10)
        in_inner, inner_distance = _polygon_distance(planar, 0.055)
        planar_distance = outer_distance if not in_outer else inner_distance if in_inner else 0.0
        return float(np.hypot(planar_distance, max(abs(p[1]) - 0.0175, 0)))
    raise ValueError(f"No independent reference for {object_id}")


def trajectories(object_id: str) -> dict[str, float]:
    """X offsets in meters, with Z fixed at zero and motion along Y."""
    if object_id == "debug_ring":
        return {"hole": 0.0, "solid": 0.0775}
    if object_id == "debug_box":
        return {"solid": 0.0, "clear": 0.12}
    raise ValueError(f"No trajectories for {object_id}")


def reference_category(object_id: str, point: np.ndarray, radius: float, tolerance: float) -> str:
    """Distinguish a fully embedded sphere from boundary overlap or separation."""
    point = np.asarray(point, dtype=float)
    distance = distance_to_solid(object_id, point)
    if distance >= radius:
        return "separated"
    clearance = 0.0
    if object_id == "debug_box" and np.all(np.abs(point) <= [0.075, 0.05, 0.10]):
        clearance = float(np.min(np.array([0.075, 0.05, 0.10]) - np.abs(point)))
    elif object_id == "debug_ring":
        planar = point[[0, 2]]
        outside_inner, inner_distance = _polygon_distance(planar, 0.055)
        inside_outer, outer_distance = _polygon_distance(planar, 0.10)
        if inside_outer and not outside_inner and abs(point[1]) <= 0.0175:
            clearance = min(inner_distance, outer_distance, 0.0175 - abs(point[1]))
    if clearance > radius + tolerance:
        return "fully_contained"
    return "boundary_overlap"
