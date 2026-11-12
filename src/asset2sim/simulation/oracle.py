"""Explicit offline reference-fit diagnostic, isolated from the primary physics batch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import trimesh
import yaml

from asset2sim.data.manifest import load_manifest, select_samples
from asset2sim.evaluation.geometry import align_surfaces, surface_metrics, transform_points
from asset2sim.geometry.prepare import load_geometry
from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.evaluate import _summary, evaluate_trajectory
from asset2sim.simulation.reference import trajectories
from asset2sim.simulation.scene import make_scene


def run_oracle_diagnostic(
    batch: Path,
    manifest: Path,
    sample_id: str,
    conditions: list[str],
    output: Path,
    config_path: Path = PROJECT_ROOT / "configs/physics/debug.yaml",
) -> dict:
    """Fit a rigid transform using ground truth for an isolated, non-deployment diagnostic."""
    batch, output = Path(batch).resolve(), Path(output).resolve()
    source = json.loads((batch / "batch.json").read_text())
    if Path(source["manifest"]).resolve() != Path(manifest).resolve():
        raise ValueError("Oracle manifest must match the primary batch manifest")
    item = select_samples(load_manifest(manifest), [sample_id])[0]
    oid = item["object_id"]
    if oid not in {"debug_box", "debug_ring"}:
        raise ValueError("Oracle contact labels exist only for debug_box and debug_ring")
    if not item.get("reference_mesh") or not Path(item["reference_mesh"]).is_file():
        raise FileNotFoundError("Independent reference mesh is missing")
    matches = [row for row in source["results"] if row["sample_id"] == sample_id]
    if not matches or source["model"] == "Reference":
        raise ValueError("Oracle requires an existing generated-asset physics batch")
    if len(set(conditions)) != len(conditions) or not conditions:
        raise ValueError("Oracle requires distinct collision conditions")
    for condition in conditions:
        candidate = [row for row in matches if row["condition"] == condition]
        if len(candidate) != 1 or candidate[0]["mujoco_status"] != "ok":
            raise ValueError(f"No successfully loaded collision asset for {condition}")
    config = yaml.safe_load(Path(config_path).read_text())
    generated, _ = load_geometry(batch / "prepared" / sample_id / "normalized.glb")
    target, _ = load_geometry(Path(item["reference_mesh"]))
    source_fit, _ = trimesh.sample.sample_surface(generated, 1200, seed=0)
    target_fit, _ = trimesh.sample.sample_surface(target, 1200, seed=1)
    transform, rotations = align_surfaces(
        source_fit, target_fit, step_degrees=90, iterations=30, tolerance=1e-6
    )
    source_test, _ = trimesh.sample.sample_surface(generated, 5000, seed=2)
    target_test, _ = trimesh.sample.sample_surface(target, 5000, seed=3)
    before = surface_metrics(source_test, target_test, [])
    after = surface_metrics(transform_points(source_test, transform), target_test, [])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "type": "offline_oracle_reference_rigid_fit",
        "warning": "Uses reference geometry to fit pose; diagnostic only, not a deployable or primary metric",
        "batch": str(batch),
        "manifest": str(Path(manifest).resolve()),
        "sample_id": sample_id,
        "object_id": oid,
        "model": source["model"],
        "reference_mesh": item["reference_mesh"],
        "alignment": {
            "method": "90 degree rotation search and rigid ICP, no scale",
            "rotation_candidates": rotations,
            "fit_samples_per_side": 1200,
            "fit_seeds": [0, 1],
            "evaluation_samples_per_side": 5000,
            "evaluation_seeds": [2, 3],
            "matrix": transform.tolist(),
            "geometry_cd_before_m": before["chamfer_l1"],
            "geometry_cd_after_m": after["chamfer_l1"],
        },
        "results": [],
        "failures": [],
    }
    for condition in conditions:
        try:
            collision_dir = batch / "collisions" / sample_id / condition
            collision = json.loads((collision_dir / "colliders.json").read_text())
            samples, trials, validation = [], [], []
            for name, x in trajectories(oid).items():
                model, _, check = make_scene(
                    collision_dir,
                    batch / "prepared" / sample_id / "normalized.glb",
                    output / "visual.obj",
                    config,
                    x,
                    xml_path=output / "scenes" / f"{condition}_{name}.xml",
                    object_transform=transform,
                )
                points, paths = evaluate_trajectory(model, config, oid, condition, name, x)
                samples.extend(points)
                trials.extend(paths)
                validation.append({"trajectory": name, **check})
            (output / f"{condition}_observations.json").write_text(
                json.dumps({"samples": samples, "trials": trials}, indent=2, allow_nan=False) + "\n"
            )
            result["results"].append(
                {
                    "condition": condition,
                    **_summary(samples, trials, collision, [collision["build_seconds"]]),
                    "scene_validation": validation,
                }
            )
        except Exception as error:  # noqa: BLE001 - keep independent diagnostic failures
            result["failures"].append({"condition": condition, "reason": str(error)})
    result["status"] = "completed_with_failures" if result["failures"] else "completed"
    (output / "oracle_diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "condition",
        "tp",
        "tn",
        "fp",
        "fn",
        "valid_samples",
        "planned_samples",
        "fpr",
        "fnr",
        "behavior_consistency",
        "valid_trials",
        "planned_trials",
        "component_count",
        "forward_mean_seconds",
        "simulation_mean_seconds",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["results"])
    return result
