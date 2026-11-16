"""Audit saved rigid poses and diagnose pose-dependent AABB normalization.

The post-pose normalization diagnostic uses the previously GT-aligned frame.
It is an offline oracle diagnostic, separate from deployment and primary scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from asset2sim.artifacts import read_jsonl, write_json
from asset2sim.data.manifest import load_manifest
from asset2sim.evaluation.benchmark import write_csv
from asset2sim.evaluation.geometry import (
    icp,
    normalized_mesh,
    scene_mesh,
    surface_metrics,
    transform_points,
)


def normalize_in_pose(mesh: trimesh.Trimesh, pose: np.ndarray):
    """Apply an externally supplied pose, then use only the mesh's own bounds."""
    posed = mesh.copy()
    posed.apply_transform(pose)
    normalized, normalization = normalized_mesh(posed, 1.0)
    return normalized, normalization @ pose


def closed_vertex_fan_errors(mesh: trimesh.Trimesh) -> int:
    """Count vertices whose incident faces do not form one closed fan.

    Use after edge-watertightness checks; this catches pinched vertices that
    two-faces-per-edge alone does not rule out.
    """
    adjacency = [[] for _ in mesh.faces]
    for a, b in mesh.face_adjacency:
        adjacency[a].append(b)
        adjacency[b].append(a)
    invalid = 0
    for incident in mesh.vertex_faces:
        fan = set(incident[incident >= 0].tolist())
        if not fan:
            invalid += 1
            continue
        seen, todo = set(), [next(iter(fan))]
        degree_valid = True
        while todo:
            face = todo.pop()
            if face in seen:
                continue
            seen.add(face)
            neighbors = [other for other in adjacency[face] if other in fan]
            degree_valid &= len(neighbors) == 2
            todo.extend(other for other in neighbors if other not in seen)
        invalid += not degree_valid or seen != fan
    return int(invalid)


def canonical_surface_sample(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Diagnostic area samples invariant to UV-induced vertex/face reindexing."""
    triangles = np.asarray(mesh.triangles)
    vertex_order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, vertex_order[:, :, None], axis=1)
    keys = triangles.reshape(len(triangles), 9)
    order = np.lexsort(tuple(keys[:, i] for i in range(8, -1, -1)))
    canonical = trimesh.Trimesh(
        vertices=triangles[order].reshape(-1, 3),
        faces=np.arange(len(triangles) * 3).reshape(-1, 3),
        process=False,
    )
    return trimesh.sample.sample_surface(canonical, count, seed=seed)[0]


def audit(evaluations: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for evaluation in evaluations:
        provenance_file = evaluation / "protocol.json"
        if not provenance_file.exists():
            provenance_file = evaluation / "sources.json"
        provenance = json.loads(provenance_file.read_text())
        samples = {r["sample_id"]: r for r in load_manifest(Path(provenance["manifest"]))}
        generated = {
            m: {r["sample_id"]: r for r in read_jsonl(Path(path) / "results.jsonl")}
            for m, path in provenance["runs"].items()
        }
        original_records = read_jsonl(evaluation / "progress.jsonl")
        deployment_records = {
            (r["model"], r["sample_id"]): r for r in original_records if r["track"] == "export_pose"
        }
        for original in original_records:
            if original.get("geometry_status") != "ok":
                continue
            key = {k: original[k] for k in ("model", "sample_id", "object_id", "track")}
            folder = evaluation / "assets" / key["model"] / key["sample_id"] / key["track"]
            transforms = json.loads((folder / "transforms.json").read_text())
            rotation = np.asarray(transforms["rotation"])[:3, :3]
            row = key | {
                "evaluation": str(evaluation.resolve()),
                "determinant": float(np.linalg.det(rotation)),
                "orthogonality_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
            }
            row["rigid_valid"] = (
                abs(row["determinant"] - 1) < 1e-6 and row["orthogonality_error"] < 1e-6
            )
            native_file = folder / "native_transforms.json"
            if native_file.is_file():
                native = np.asarray(json.loads(native_file.read_text())["alignment"])[:3, :3]
                row["native_alignment_determinant"] = float(np.linalg.det(native))
                row["native_alignment_orthogonality_error"] = float(
                    np.max(np.abs(native.T @ native - np.eye(3)))
                )
                row["rigid_valid"] &= (
                    abs(row["native_alignment_determinant"] - 1) < 1e-6
                    and row["native_alignment_orthogonality_error"] < 1e-6
                )
            # Independently reconstruct saved visual geometry from original
            # assets and the recorded matrices, rather than trusting filenames.
            errors = []
            for kind in ("prediction", "reference"):
                if kind == "prediction":
                    source = (
                        Path(provenance["runs"][key["model"]])
                        / generated[key["model"]][key["sample_id"]]["mesh"]
                    )
                    matrix = np.asarray(transforms["rotation"]) @ np.asarray(
                        transforms["prediction_normalization"]
                    )
                else:
                    source = Path(samples[key["sample_id"]]["reference_mesh"])
                    matrix = np.asarray(transforms["reference_normalization"])
                _, raw = scene_mesh(source)
                _, saved = scene_mesh(folder / f"{kind}.glb")
                expected = transform_points(raw.vertices, matrix)
                error = max(
                    cKDTree(expected).query(saved.vertices)[0].max(),
                    cKDTree(saved.vertices).query(expected)[0].max(),
                )
                row[kind + "_matrix_max_error"] = float(error)
                errors.append(error)
            row["matrix_reconstruction_valid"] = bool(max(errors) < 1e-5)
            if key["track"] == "aligned_diagnostic":
                deployment_folder = folder.with_name("export_pose")
                if (deployment_folder / "transforms.json").is_file():
                    deployment_transform = json.loads(
                        (deployment_folder / "transforms.json").read_text()
                    )
                    declared = np.asarray(deployment_transform["rotation"])[:3, :3]
                    row["declared_vs_fitted_rotation_degrees"] = float(
                        np.rad2deg(
                            np.arccos(np.clip((np.trace(rotation @ declared.T) - 1) / 2, -1, 1))
                        )
                    )
                    deployment = deployment_records.get((key["model"], key["sample_id"]), {})
                    if "texture_input_lpips" in original and "texture_input_lpips" in deployment:
                        row["input_lpips_fitted_minus_declared"] = (
                            original["texture_input_lpips"] - deployment["texture_input_lpips"]
                        )
                _, predicted = scene_mesh(folder / "prediction.glb")
                _, reference = scene_mesh(folder / "reference.glb")
                row["baseline_cd"] = original["chamfer_l1"]
                row["post_pose_longest_side"] = float(predicted.extents.max())
                row["post_pose_center_norm"] = float(np.linalg.norm(predicted.bounds.mean(axis=0)))
                corrected, normalization = normalize_in_pose(predicted, np.eye(4))
                p = trimesh.sample.sample_surface(corrected, 20000, seed=0)[0]
                q = trimesh.sample.sample_surface(reference, 20000, seed=1)[0]
                initial = surface_metrics(p, q, [0.01, 0.02, 0.05])
                refinement = icp(p[:1024], q[:1024], np.eye(4), 50)
                refined = surface_metrics(transform_points(p, refinement), q, [0.01, 0.02, 0.05])
                if refined["chamfer_l1"] < initial["chamfer_l1"]:
                    values, total = refined, refinement @ normalization
                else:
                    values, total = initial, normalization
                row.update({"oracle_post_pose_" + k: v for k, v in values.items()})
                row["oracle_relative_scale"] = float(np.cbrt(np.linalg.det(total[:3, :3])))
                row["oracle_transform_from_saved_prediction"] = total.tolist()
                row["oracle_uses_gt_aligned_pose"] = True
            records.append(row)
            with (output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    write_csv(output / "per_case.csv", records)
    summary = {
        "rows": len(records),
        "rigid_invalid": sum(not r["rigid_valid"] for r in records),
        "matrix_reconstruction_invalid": sum(not r["matrix_reconstruction_valid"] for r in records),
        "native_alignment_rows": sum("native_alignment_determinant" in r for r in records),
        "native_alignment_invalid": sum(
            abs(r["native_alignment_determinant"] - 1) > 1e-6
            or r["native_alignment_orthogonality_error"] > 1e-6
            for r in records
            if "native_alignment_determinant" in r
        ),
        "scope": "Offline oracle normalization sensitivity; original primary scores unchanged",
        "groups": [],
    }
    for evaluation in evaluations:
        for model in sorted(
            {r["model"] for r in records if r["evaluation"] == str(evaluation.resolve())}
        ):
            subset = [
                r
                for r in records
                if r["evaluation"] == str(evaluation.resolve())
                and r["model"] == model
                and r["track"] == "aligned_diagnostic"
            ]
            objects = {r["object_id"] for r in subset}
            record = {
                "evaluation": str(evaluation),
                "model": model,
                "n_inputs": len(subset),
                "n_objects": len(objects),
            }
            for metric in [
                "baseline_cd",
                "oracle_post_pose_chamfer_l1",
                "oracle_post_pose_fscore@0.02",
                "oracle_relative_scale",
            ]:
                per_object = [
                    np.mean([r[metric] for r in subset if r["object_id"] == o])
                    for o in sorted(objects)
                ]
                record[metric] = float(np.median(per_object))
            summary["groups"].append(record)
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation, args.output)


if __name__ == "__main__":
    main()
