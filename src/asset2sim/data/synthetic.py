"""Shared reference solids and cameras for generator and geometry development."""

from __future__ import annotations

import math

import numpy as np
import trimesh


def reference_meshes() -> dict[str, tuple[str, trimesh.Trimesh]]:
    """Closed solids in a right-handed, Z-up frame; dimensions are in meters."""
    from shapely.geometry import Polygon

    convex = trimesh.creation.box(extents=[0.12, 0.08, 0.16])
    cup = trimesh.creation.revolve(
        np.array([[0, 0], [0.08, 0], [0.08, 0.15], [0.065, 0.15], [0.065, 0.02], [0, 0.02], [0, 0]]),
        sections=64,
    )
    ring = trimesh.creation.annulus(r_min=0.055, r_max=0.10, height=0.035, sections=64)
    ring.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    profile = Polygon([
        [-0.10, -0.10], [0.10, -0.10], [0.10, -0.07],
        [-0.07, -0.07], [-0.07, 0.10], [-0.10, 0.10],
    ])
    bracket = trimesh.creation.extrude_polygon(profile, height=0.03, engine="earcut")
    bracket.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return {
        "debug_box": ("convex", convex),
        "debug_cup": ("concave", cup),
        "debug_ring": ("through_hole", ring),
        "debug_bracket": ("slender", bracket),
    }


def camera_pose(azimuth: float, elevation: float, distance: float = 0.46) -> np.ndarray:
    """OpenGL camera: +X right, +Y up, looking along -Z; world +Z is up."""
    azimuth, elevation = np.deg2rad([azimuth, elevation])
    position = distance * np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, up, -forward])
    pose[:3, 3] = position
    return pose
