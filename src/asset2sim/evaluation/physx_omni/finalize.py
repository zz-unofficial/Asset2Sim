"""Local evaluation of returned runs, export controls, and explicit review eligibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import load_manifest

from .audit import audit_asset, raw_parts
from .run import evaluate_run, write_table


def finalize_return(config_path: Path, output: Path) -> dict:
    """Evaluate existing assets; keep raw attempts and repaired variants distinguishable."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    bundle = (config_path.parent / config["bundle"]).resolve()
    manifest = bundle / "prepared/manifest_local.jsonl"
    samples = load_manifest(manifest)
    runs = bundle / "runs"
    index = runs / "local_comparison_index"
    for sample in samples:
        for field in ("image", "reference_mesh", "sample_metadata"):
            if not Path(sample[field]).is_file():
                raise FileNotFoundError(sample[field])
        if not (index / sample["sample_id"] / "ours_demo/asset").is_dir():
            raise FileNotFoundError(index / sample["sample_id"])
    output = output.resolve()
    summary = evaluate_run(
        index, manifest, output, sample_count=int(config.get("sample_count", 10000))
    )
    attempts = []
    for record in sorted(runs.glob("server_*/run.json")):
        if record.parent.name.startswith(("server_comparison_index", "server_plan_")):
            continue
        data = json.loads(record.read_text())
        for item in data.get("samples", []):
            for stage in item.get("stages", []):
                attempts.append(
                    {
                        "source_run": record.parent.name,
                        "sample_id": item["input"]["sample_id"],
                        "decoder_cpu_offload": data.get("config", {}).get(
                            "decoder_cpu_offload", False
                        ),
                        **stage,
                    }
                )
    write_table(output / "attempt_history.csv", attempts)
    controls = []
    for spec in config.get("variants", []):
        row = {key: value for key, value in spec.items() if key != "asset"}
        asset = bundle / spec["asset"]
        try:
            audit_category = "rigid" if spec.get("condition") == "D0_default" else "deformable"
            row["flex_required"] = audit_category == "deformable"
            audit = audit_asset(asset, audit_category)
            row.update(
                status=audit["status"],
                errors=audit["errors"],
                **audit["structure"]["deformable_evidence"],
            )
            write_json(output / (spec["id"] + "_audit.json"), audit)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="failed", reason=str(error))
        controls.append(row)
    write_table(output / "export_controls.csv", controls)
    with (output / "geometry_metrics.csv").open() as stream:
        geometry_status = {row["sample_id"]: row["status"] for row in csv.DictReader(stream)}
    eligibility, review = [], []
    for sample in samples:
        metadata = json.loads(Path(sample["sample_metadata"]).read_text())
        asset = index / sample["sample_id"] / "ours_demo/asset"
        row = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "sequence_id": metadata.get("sequence_id"),
            "frame_id": metadata.get("frame_id"),
            "shape_diagnostic": geometry_status.get(sample["sample_id"], "missing"),
            "fixed_pose": "available"
            if metadata.get("prediction_to_reference") is not None
            else "NA_coordinate_correspondence_unverified",
            "absolute_scale": "available"
            if metadata.get("absolute_scale_verified")
            else "NA_reference_scale_unverified",
            "joint_correspondence": bool(metadata.get("joint_matches")),
            "part_correspondence": bool(metadata.get("part_matches")),
            "physical_label_count": len(metadata.get("physical_labels", {})),
        }
        eligibility.append(row)
        review.append(
            {
                "sample_id": sample["sample_id"],
                "input_image": sample["image"],
                "reference_mesh": sample["reference_mesh"],
                "metadata": sample["sample_metadata"],
                "predicted_parts": raw_parts(asset / "basic_info.txt"),
                "reference_parts": metadata.get("part_references", {}),
                "reference_joints": metadata.get("kinematics", {}).get("joints", []),
                "human_review": {
                    "verified": False,
                    "semantic_notes": "",
                    "part_matches": {},
                    "joint_matches": {},
                    "link_matches": {},
                    "coordinate_evidence": "",
                },
            }
        )
    write_table(output / "metric_eligibility.csv", eligibility)
    write_json(output / "review_template.json", {"samples": review})
    write_json(
        output / "local_config.json",
        {
            "bundle": str(bundle),
            "manifest": str(manifest),
            "config": config,
            "validation": "static_only",
        },
    )
    return summary
