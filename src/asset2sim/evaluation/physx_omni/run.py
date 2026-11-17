"""Batch static evaluation with complete planned-sample denominators."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import load_manifest
from asset2sim.evaluation.geometry import scene_mesh

from .attributes import articulation_metrics, attribute_metrics
from .audit import audit_asset, raw_parts
from .geometry import geometry_metrics
from .parts import part_metrics


def write_table(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def evaluate_run(run: Path, manifest: Path, output: Path, *, sample_count: int = 10000) -> dict:
    """Evaluate every planned input, including absent or failed generation records."""
    if sample_count < 100:
        raise ValueError("At least 100 surface samples required")
    samples = load_manifest(manifest)
    execution = json.loads((run / "run.json").read_text())
    index = {item["input"]["sample_id"]: item for item in execution["samples"]}
    output.mkdir(parents=True, exist_ok=False)
    pipeline, geometry, audit_rows, failures, costs, attributes = [], [], [], [], [], []
    for sample in samples:
        sid, category = sample["sample_id"], sample.get("category", "rigid")
        item = index.get(sid, {})
        stages = {stage["stage"]: stage for stage in item.get("stages", [])}
        base = {
            "sample_id": sid,
            "category": category,
            "object_id": sample["object_id"],
            "sequence_id": sample.get("sequence_id"),
            "frame_id": sample.get("frame_id"),
        }
        row = {
            **base,
            "generation_status": item.get("status", "missing"),
            "export_audit": "blocked",
        }
        for stage in ("infer", "decode", "export"):
            row[stage] = stages.get(stage, {}).get("status", "missing")
            row[f"{stage}_execution_success"] = stages.get(stage, {}).get("execution_success")
            row[f"{stage}_output_valid"] = stages.get(stage, {}).get("output_valid")
        for stage in item.get("stages", []):
            costs.append(
                {**base, **{key: value for key, value in stage.items() if key != "reason"}}
            )
            if stage["status"] == "failed":
                failures.append({**base, **stage})
        asset = run.resolve() / sid / "ours_demo/asset"
        try:
            predicted_ids = (
                raw_parts(asset / "basic_info.txt") if (asset / "basic_info.txt").is_file() else {}
            )
        except (ValueError, OSError) as error:
            predicted_ids = {}
            failures.append({**base, "stage": "raw_part_parsing", "reason": str(error)})
        row["predicted_parts"] = len(predicted_ids)
        row["part_denominator_known"] = bool(predicted_ids)
        row["decoded_parts"] = 0
        for part_id in predicted_ids:
            try:
                for suffix in ("glb", "obj"):
                    scene_mesh(asset / "objs" / str(part_id) / f"{part_id}.{suffix}")
                row["decoded_parts"] += 1
            except (ValueError, OSError, TypeError):
                pass  # Count every failed part in the fixed predicted-part denominator.
        audit = None
        try:
            audit = audit_asset(asset, category)
            write_json(output / f"{sid}_audit.json", audit)
            row["export_audit"] = audit["status"]
            audit_rows.extend({**base, **transfer} for transfer in audit["transfers"])
            row["exported_parts"] = audit["parts"]
            for error in audit["errors"]:
                failures.append({**base, "stage": "static_audit", "reason": error})
        except Exception as error:  # noqa: BLE001 -- sample audit errors remain in the denominator.
            row["export_audit"] = "failed"
            attributes.append(
                {**base, "status": "failed", "reason": f"static_audit_unavailable:{error}"}
            )
            failures.append({**base, "stage": "static_audit", "reason": str(error)})
        if audit is not None:
            try:
                attributes.extend({**base, **metric} for metric in attribute_metrics(sample, audit))
                if category == "articulated":
                    attributes.extend(
                        {**base, **metric} for metric in articulation_metrics(sample, audit)
                    )
            except (ValueError, OSError, KeyError, TypeError) as error:
                attributes.append({**base, "status": "failed", "reason": str(error)})
                failures.append(
                    {**base, "stage": "reference_attribute_evaluation", "reason": str(error)}
                )
        metrics = None
        try:
            metadata = (
                json.loads(Path(sample["sample_metadata"]).read_text())
                if sample.get("sample_metadata")
                else {}
            )
            metrics = geometry_metrics(
                asset,
                Path(sample["reference_mesh"]),
                count=sample_count,
                prediction_to_reference=metadata.get("prediction_to_reference"),
                reference_unit_m=metadata.get("reference_unit_m")
                if metadata.get("absolute_scale_verified")
                else None,
            )
            geometry.append(
                {
                    **base,
                    "status": "ok" if metrics["complete_predicted_parts"] else "partial",
                    **metrics,
                }
            )
        except Exception as error:  # noqa: BLE001 -- evaluate remaining samples after geometry errors.
            geometry.append({**base, "status": "failed", "reason": str(error)})
            failures.append({**base, "stage": "geometry_evaluation", "reason": str(error)})
        if category == "articulated" and metrics is not None:
            try:
                attributes.extend(
                    {**base, **metric}
                    for metric in part_metrics(
                        asset,
                        metadata,
                        metrics,
                        Path(sample["sample_metadata"]).parent
                        if sample.get("sample_metadata")
                        else manifest.parent,
                        sample_count=sample_count,
                    )
                )
            except (ValueError, OSError, KeyError, TypeError) as error:
                attributes.append(
                    {**base, "field": "part_comparison", "status": "failed", "reason": str(error)}
                )
                failures.append(
                    {**base, "stage": "reference_part_evaluation", "reason": str(error)}
                )
        row["end_to_end_success"] = (
            all(row[stage] == "ok" for stage in ("infer", "decode", "export"))
            and row["export_audit"] == "ok"
        )
        pipeline.append(row)
    summary = {"validation": "static_only", "n_planned": len(samples), "categories": {}}
    for category in sorted({row["category"] for row in pipeline}):
        group = [row for row in pipeline if row["category"] == category]
        summary["categories"][category] = {
            "n_planned": len(group),
            **{
                stage + "_success": sum(row[stage] == "ok" for row in group)
                for stage in ("infer", "decode", "export", "export_audit")
            },
            "end_to_end_success": sum(row["end_to_end_success"] for row in group),
            "geometry_valid": sum(
                row["status"] == "ok" and row["category"] == category for row in geometry
            ),
            "fixed_pose_valid": sum(
                row.get("fixed_pose", {}).get("status") == "ok" and row["category"] == category
                for row in geometry
            ),
            "predicted_parts": sum(row.get("predicted_parts", 0) for row in group),
            "decoded_parts": sum(row.get("decoded_parts", 0) for row in group),
            "attribute_evaluable_pairs": {
                field: sum(
                    row["category"] == category
                    and row.get("field") == field
                    and row["status"] == "ok"
                    for row in attributes
                )
                for field in ("material", "density", "young", "poisson", "joint_comparison")
            },
        }
    for name, rows in (
        ("pipeline_summary", pipeline),
        ("geometry_metrics", geometry),
        ("export_audit", audit_rows),
        ("failure_cases", failures),
        ("runtime_cost", costs),
    ):
        write_table(output / f"{name}.csv", rows)
    for category, filename in (
        ("rigid", "rigid_attributes"),
        ("articulated", "articulated_metrics"),
        ("deformable", "deformable_metrics"),
    ):
        rows = [row for row in attributes if row["category"] == category]
        write_table(output / f"{filename}.csv", rows)
    write_json(output / "summary.json", summary)
    return summary
