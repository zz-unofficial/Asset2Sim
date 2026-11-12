"""Normalized surface distances and rotation-search/ICP alignment."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"No triangle mesh: {path}")
    if not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def normalized_mesh(mesh: trimesh.Trimesh, extent: float) -> tuple[trimesh.Trimesh, np.ndarray]:
    length = float(mesh.extents.max())
    if not np.isfinite(length) or length <= 0 or extent <= 0:
        raise ValueError("Mesh extent must be positive")
    transform = np.eye(4)
    transform[:3, :3] *= extent / length
    transform[:3, 3] = -mesh.bounds.mean(axis=0) * extent / length
    result = mesh.copy()
    result.apply_transform(transform)
    return result, transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def distances(predicted: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(predicted) == 0 or len(reference) == 0:
        raise ValueError("Surface point sets must be nonempty")
    return cKDTree(reference).query(predicted)[0], cKDTree(predicted).query(reference)[0]


def surface_metrics(
    predicted: np.ndarray, reference: np.ndarray, thresholds: list[float]
) -> dict[str, float]:
    accuracy, completeness = distances(predicted, reference)
    result = {
        "chamfer_l1": float((accuracy.mean() + completeness.mean()) / 2),
        "chamfer_squared": float((np.square(accuracy).mean() + np.square(completeness).mean()) / 2),
    }
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("F-score thresholds must be positive")
        precision = float(np.mean(accuracy < threshold))
        recall = float(np.mean(completeness < threshold))
        key = f"{threshold:g}"
        result[f"precision@{key}"] = precision
        result[f"recall@{key}"] = recall
        result[f"fscore@{key}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return result


def rigid_fit(source: np.ndarray, target: np.ndarray, scale: bool = False) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    x, y = source - source_center, target - target_center
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    factor = 1.0
    if scale:
        factor = float(np.sum(singular * np.diag(correction)) / max(np.sum(x * x), 1e-15))
    transform = np.eye(4)
    transform[:3, :3] = factor * rotation
    transform[:3, 3] = target_center - factor * rotation @ source_center
    return transform


def icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    iterations: int,
    tolerance: float = 1e-6,
    scale: bool = False,
) -> np.ndarray:
    current = transform_points(source, initial)
    transform = initial.copy()
    target_tree = cKDTree(target)
    previous = np.inf
    for _ in range(iterations):
        error, indices = target_tree.query(current)
        value = float(error.mean())
        if abs(previous - value) < tolerance:
            break
        previous = value
        step = rigid_fit(current, target[indices], scale=scale)
        current = transform_points(current, step)
        transform = step @ transform
    return transform


def align_surfaces(
    source: np.ndarray,
    target: np.ndarray,
    step_degrees: int,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    if step_degrees < 1 or 360 % step_degrees:
        raise ValueError("rotation_step_degrees must be a positive divisor of 360")
    rotations = Rotation.from_euler(
        "xyz", list(product(range(0, 360, step_degrees), repeat=3)), degrees=True
    ).as_matrix()
    rotations = np.unique(np.round(rotations, 10), axis=0)
    tree = cKDTree(target)
    best_score = np.inf
    best = np.eye(4)
    for rotation in rotations:
        candidate = source @ rotation.T
        first = tree.query(candidate)[0].mean()
        second = cKDTree(candidate).query(target)[0].mean()
        score = (first + second) / 2
        if score < best_score:
            best_score = score
            best[:3, :3] = rotation
    return icp(source, target, best, iterations, tolerance), len(rotations)
