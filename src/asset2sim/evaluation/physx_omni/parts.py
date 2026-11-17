"""Part comparisons using explicit correspondences and one shared object alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from asset2sim.evaluation.geometry import scene_mesh, surface_metrics, transform_points


def part_metrics(
    asset: Path, metadata: dict, geometry: dict, metadata_dir: Path, sample_count: int = 10000
) -> list[dict]:
    references = metadata.get("part_references", {})
    matches = metadata.get("part_matches", {})
    if not references or not matches:
        return [
            {
                "field": "part_comparison",
                "status": "NA",
                "reason": "no_verified_part_correspondence",
            }
        ]
    if len(set(matches.values())) != len(matches):
        raise ValueError("Part correspondences must be one-to-one")
    prediction_transform = np.asarray(geometry["prediction_transform"])
    reference_transform = np.asarray(geometry["reference_transform"])
    rows = []
    for part_id, reference_id in matches.items():
        row = {
            "field": "part_comparison",
            "predicted_part": str(part_id),
            "reference_part": reference_id,
        }
        try:
            spec = references[reference_id]
            _, predicted = scene_mesh(asset / "objs" / str(int(part_id)) / f"{int(part_id)}.glb")
            _, reference = scene_mesh(metadata_dir / spec["mesh"])
            frame = np.asarray(spec.get("part_to_reference", np.eye(4)), dtype=float)
            if (
                frame.shape != (4, 4)
                or not np.isfinite(frame).all()
                or not np.allclose(frame[3], [0, 0, 0, 1])
            ):
                raise ValueError("Invalid reference part frame")
            p, _ = trimesh.sample.sample_surface(predicted, sample_count, seed=0)
            r, _ = trimesh.sample.sample_surface(reference, sample_count, seed=0)
            row.update(
                surface_metrics(
                    transform_points(p, prediction_transform),
                    transform_points(r, reference_transform @ frame),
                    [0.01, 0.02, 0.05],
                )
            )
            row.update(
                status="ok", length_unit="unit_longest_side", alignment="shared_object_transform"
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            row.update(status="failed", reason=str(error))
        rows.append(row)
    valid = sum(row["status"] == "ok" for row in rows)
    rows.append(
        {
            "field": "part_completeness",
            "status": "ok",
            "matched_valid": valid,
            "reference_parts": len(references),
            "recall": valid / len(references),
            "declared_correspondences": len(matches),
        }
    )
    return rows
