"""Reference-aware material and joint metrics with explicit NA reasons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asset2sim.evaluation.geometry import transform_points

from .audit import _number


def attribute_metrics(sample: dict, audit: dict) -> list[dict]:
    """Only score independently sourced labels; keep explicit NA reasons otherwise."""
    metadata = {}
    if sample.get("sample_metadata"):
        metadata = json.loads(Path(sample["sample_metadata"]).read_text())
    labels = metadata.get("physical_labels", {})
    rows = []
    for index, part in enumerate(audit["properties"].get("parts", [])):
        label = int(part.get("label", index))
        for field, key, unit, factor in (
            ("material", "material", "category", 1),
            ("density", "density", "kg/m3", 1000),
            ("young", "Young's Modulus (GPa)", "Pa", 1e9),
            ("poisson", "Poisson's Ratio", "dimensionless", 1),
        ):
            value = part.get(key)
            predicted = (
                value
                if field == "material"
                else (_number(value) * factor if _number(value) is not None else None)
            )
            evidence = labels.get(field, {})
            row = {
                "part_id": label,
                "field": field,
                "prediction": predicted,
                "unit": unit,
                "reference": evidence.get("value"),
                "reference_source": evidence.get("source"),
                "reference_evidence_type": evidence.get("evidence_type"),
                "error": None,
                "status": "NA",
                "reason": "no_reliable_ground_truth",
            }
            if predicted is None:
                row["reason"] = "prediction_missing"
            elif (
                evidence.get("ground_truth")
                and evidence.get("evidence_type") not in {"model_inference", "material_prior"}
                and evidence.get("source")
            ):
                if len(audit["properties"]["parts"]) > 1 and evidence.get("part_id") != label:
                    row["reason"] = "reference_part_not_matched"
                elif evidence.get("unit") != unit:
                    row["reason"] = "reference_unit_not_matched"
                else:
                    truth = evidence.get("value")
                    if field == "material" and isinstance(truth, str):
                        row.update(
                            status="ok",
                            reason=None,
                            error=float(str(predicted).casefold() != truth.casefold()),
                        )
                    elif _number(truth) is not None and (field == "poisson" or truth > 0):
                        error = (
                            abs(np.log10(predicted) - np.log10(truth))
                            if field == "young" and predicted > 0
                            else abs(predicted - truth) / truth
                            if field == "density"
                            else abs(predicted - truth)
                        )
                        row.update(status="ok", reason=None, error=float(error))
            rows.append(row)
    return rows


def articulation_metrics(sample: dict, audit: dict) -> list[dict]:
    """Score explicit joint correspondences; never match by file ordering."""
    from asset2sim.evaluation.articulation import joint_errors

    metadata = (
        json.loads(Path(sample["sample_metadata"]).read_text())
        if sample.get("sample_metadata")
        else {}
    )
    reference = {joint["name"]: joint for joint in metadata.get("kinematics", {}).get("joints", [])}
    predicted = {
        joint["name"]: joint
        for joint in audit.get("structure", {}).get("urdf", {}).get("joints", [])
    }
    matches = metadata.get("joint_matches", {})
    if not matches:
        return [
            {
                "field": "joint_comparison",
                "status": "NA",
                "reason": "no_verified_joint_correspondence",
                "reference_joints": len(reference),
                "predicted_joints": len(predicted),
            }
        ]
    if len(set(matches.values())) != len(matches):
        raise ValueError("Joint correspondences must be one-to-one")
    matrix = metadata.get("joint_prediction_to_reference")
    transform = np.asarray(matrix, dtype=float) if matrix is not None else None
    scale = None
    if transform is not None:
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
        ):
            raise ValueError("Invalid declared joint coordinate transform")
        linear = transform[:3, :3]
        scale = np.linalg.norm(linear[:, 0])
        if (
            scale <= 0
            or np.linalg.det(linear) <= 0
            or not np.allclose(linear.T @ linear, np.eye(3) * scale**2)
        ):
            raise ValueError("Joint transform must have uniform positive scale and proper rotation")
    extent = float(np.ptp(np.asarray(metadata["reference_geometry"]["bounds"]), axis=0).max())
    rows = []
    for p_name, r_name in matches.items():
        if p_name not in predicted or r_name not in reference:
            rows.append(
                {
                    "field": "joint_comparison",
                    "predicted_joint": p_name,
                    "reference_joint": r_name,
                    "status": "failed",
                    "reason": "matched_joint_missing",
                }
            )
            continue
        p, r = dict(predicted[p_name]), reference[r_name]
        row = {
            "field": "joint_comparison",
            "predicted_joint": p_name,
            "reference_joint": r_name,
            "joint_type_correct": p["type"] == r["type"],
            "status": "ok",
        }
        if transform is None:
            row.update(axis_status="NA", reason="joint_frames_not_declared")
        else:
            p["world_axis"] = (transform[:3, :3] @ p["world_axis"]).tolist()
            p["world_origin"] = transform_points(np.asarray([p["world_origin"]]), transform)[
                0
            ].tolist()
            if p["type"] == "prismatic" and p.get("range") is not None:
                p["range"] = (np.asarray(p["range"]) * scale).tolist()
            row.update(joint_errors(p, r, extent))
        rows.append(row)
    links = metadata.get("link_matches", {})
    if links:
        edges = {(joint["parent"], joint["child"]) for joint in reference.values()}
        for joint in predicted.values():
            if joint["parent"] in links and joint["child"] in links:
                rows.append(
                    {
                        "field": "parent_child",
                        "predicted_joint": joint["name"],
                        "correct": (links[joint["parent"]], links[joint["child"]]) in edges,
                        "status": "ok",
                    }
                )
    return rows
