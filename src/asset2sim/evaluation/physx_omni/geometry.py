"""Shared shape metrics and a separately measured absolute-scale track."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from asset2sim.evaluation.geometry import (
    align_surfaces,
    normalized_mesh,
    scene_mesh,
    surface_metrics,
    transform_points,
)

from .audit import raw_parts


def geometry_metrics(
    asset: Path,
    reference: Path,
    *,
    count: int = 10000,
    seed: int = 0,
    reference_unit_m: float | None = None,
    prediction_to_reference: list | None = None,
) -> dict:
    """Diagnostic shape alignment; one transform for all predicted parts, no fitting each part."""
    if count < 100:
        raise ValueError("At least 100 surface samples required")
    paths = sorted(asset.glob("objs/*/*.glb"))
    meshes = [scene_mesh(path)[1] for path in paths]
    if not meshes:
        raise ValueError("No decoded part meshes")
    predicted = trimesh.util.concatenate(meshes)
    _, target = scene_mesh(reference)
    native_prediction, native_reference = predicted.copy(), target.copy()
    predicted, pn = normalized_mesh(predicted, 1.0)
    target, rn = normalized_mesh(target, 1.0)
    p, pf = trimesh.sample.sample_surface(predicted, count, seed=seed)
    r, rf = trimesh.sample.sample_surface(target, count, seed=seed)
    alignment, _ = align_surfaces(p[: min(count, 2000)], r[: min(count, 2000)], 90, 30, 1e-6)
    p = transform_points(p, alignment)
    result = surface_metrics(p, r, [0.01, 0.02, 0.05])
    neighbors = cKDTree(r).query(p)[1]
    normals = predicted.face_normals[pf] @ alignment[:3, :3].T
    result["normal_consistency_unsigned"] = float(
        np.abs((normals * target.face_normals[rf[neighbors]]).sum(axis=1)).mean()
    )
    posed = predicted.copy()
    posed.apply_transform(alignment)
    result.update(
        track="normalized_rigid_aligned_diagnostic",
        length_unit="unit_longest_side",
        sample_count=count,
        seed=seed,
        absolute_scale_status="NA_requires_verified_scale",
        prediction_transform=(alignment @ pn).tolist(),
        reference_transform=rn.tolist(),
        bbox_extent_mae_normalized=float(np.abs(posed.extents - target.extents).mean()),
        self_intersection_status="not_computed",
        reference_faces=len(target.faces),
        predicted_faces=len(predicted.faces),
        complete_predicted_parts=set(raw_parts(asset / "basic_info.txt"))
        == {int(path.parent.name) for path in paths},
    )
    result["fixed_pose"] = {
        "status": "NA_coordinate_correspondence_unverified",
        "alignment": "none",
        "normal_rule": "unsigned_surface_normal_dot",
    }
    if prediction_to_reference is not None:
        transform = np.asarray(prediction_to_reference, dtype=float)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
        ):
            raise ValueError("Invalid declared prediction-to-reference transform")
        fixed = native_prediction.copy()
        fixed.apply_transform(transform)
        fp, _ = trimesh.sample.sample_surface(fixed, count, seed=seed)
        fr, _ = trimesh.sample.sample_surface(native_reference, count, seed=seed)
        threshold_scale = native_reference.extents.max()
        result["fixed_pose"] = {
            "status": "ok",
            "alignment": "declared_transform_only",
            "length_unit": "reference_native_unit",
            "transform": transform.tolist(),
            "sample_count": count,
            **surface_metrics(fp, fr, [threshold_scale * x for x in (0.01, 0.02, 0.05)]),
        }
    if reference_unit_m is not None:
        if not np.isfinite(reference_unit_m) or reference_unit_m <= 0:
            raise ValueError("Reference meters-per-unit must be positive")
        match = re.search(
            r"^Dimension:\s*([\d.]+)\s*[*x×]\s*([\d.]+)\s*[*x×]\s*([\d.]+)",
            (asset / "basic_info.txt").read_text(),
            re.MULTILINE,
        )
        if match:
            dimensions = np.asarray([float(value) for value in match.groups()]) / 100
            if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError("Invalid predicted physical dimensions")
            native_prediction.apply_scale(dimensions.max())
            native_reference.apply_scale(reference_unit_m)
            ap, _ = trimesh.sample.sample_surface(native_prediction, count, seed=seed)
            ar, _ = trimesh.sample.sample_surface(native_reference, count, seed=seed)
            rigid, _ = align_surfaces(ap[: min(count, 2000)], ar[: min(count, 2000)], 90, 30, 1e-6)
            ap = transform_points(ap, rigid)
            native_prediction.apply_transform(rigid)
            result.update(
                absolute_scale_status="ok",
                absolute_chamfer_l1_m=surface_metrics(ap, ar, [])["chamfer_l1"],
                absolute_bbox_error_m=np.abs(
                    native_prediction.extents - native_reference.extents
                ).tolist(),
                predicted_dimension_m=dimensions.tolist(),
                absolute_rigid_transform=rigid.tolist(),
                reference_unit_m=reference_unit_m,
            )
        else:
            result["absolute_scale_status"] = "NA_prediction_dimension_missing"
    return result
