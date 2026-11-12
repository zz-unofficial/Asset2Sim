"""Small, auditable physical asset batches without test-time reference fitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from asset2sim.artifacts import read_jsonl
from asset2sim.collision import build_collision
from asset2sim.data.manifest import load_manifest, select_samples
from asset2sim.geometry.prepare import PreparationOptions, prepare_mesh
from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.assets import index_generated_assets
from asset2sim.simulation.evaluate import _summary, evaluate_trajectory
from asset2sim.simulation.reference import trajectories
from asset2sim.simulation.scene import make_scene


def _write_ledger(output: Path, rows: list[dict], context: dict) -> None:
    (output / "batch.json").write_text(
        json.dumps({**context, "results": rows}, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "model",
        "sample_id",
        "object_id",
        "view_id",
        "condition",
        "generation_status",
        "preparation_status",
        "topology_status",
        "repair_operations",
        "collision_status",
        "component_count",
        "collision_build_seconds",
        "surface_error_m",
        "index_status",
        "mujoco_status",
        "pose_status",
        "contact_status",
        "contact_valid_samples",
        "contact_planned_samples",
        "tp",
        "tn",
        "fp",
        "fn",
        "fpr",
        "fnr",
        "behavior_consistency",
        "dynamic_inconsistency_count",
        "solver_warning_trials",
        "valid_trials",
        "planned_trials",
        "forward_mean_seconds",
        "simulation_mean_seconds",
        "generation_quality_status",
        "failure_stage",
        "reason",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_physics_batch(
    manifest: Path,
    output: Path,
    conditions: list[str],
    sample_ids: list[str] | None = None,
    *,
    run: Path | None = None,
    reference: bool = False,
    merge_coincident: bool = False,
    remove_degenerate: bool = False,
    remove_unreferenced: bool = False,
    config_path: Path = PROJECT_ROOT / "configs/physics/debug.yaml",
) -> dict:
    """Prepare, decompose and probe selected assets; failures remain in the ledger."""
    if reference == (run is not None):
        raise ValueError("Select exactly one of reference or a generation run")
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or set(conditions) - {"C0", "C1", "C2", "C3"}
    ):
        raise ValueError("Select distinct conditions C0 through C3")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    selected = select_samples(load_manifest(manifest), sample_ids)
    run = Path(run).resolve() if run else None
    run_metadata = json.loads((run / "run.json").read_text()) if run else {}
    generated = read_jsonl(run / "results.jsonl") if run else []
    lookup = {}
    for row in generated:
        sid = row.get("sample_id")
        if not sid or sid in lookup:
            raise ValueError(f"Missing or duplicate generation sample_id: {sid!r}")
        lookup[sid] = row
    model = (
        "Reference"
        if reference
        else run_metadata.get("model") or run_metadata.get("config", {}).get("model", "unknown")
    )
    if "stable-fast-3d" in model:
        model = "sf3d"
    elif "stable-point-aware-3d" in model:
        model = "spar3d"
    options = PreparationOptions(
        kind="reference" if reference else "generated",
        merge_coincident=merge_coincident,
        remove_degenerate=remove_degenerate,
        remove_unreferenced=remove_unreferenced,
    )
    config = yaml.safe_load(Path(config_path).read_text())
    collision_config = PROJECT_ROOT / config["collision_config"]
    output.mkdir(parents=True, exist_ok=False)
    context: dict[str, Any] = {
        "status": "running",
        "manifest": str(manifest),
        "generation_run": str(run) if run else None,
        "model": model,
        "run_metadata": run_metadata,
        "conditions": conditions,
        "sample_ids": [x["sample_id"] for x in selected],
        "preparation_options": vars(options),
        "physics_config": config,
        "coordinate_protocol": "Generated pose relative to reference unverified; no ICP or ground-truth alignment",
        "generation_quality_protocol": "No pre-existing per-sample debug geometry or appearance metrics supplied",
    }
    rows: list[dict] = []
    for item in selected:
        sid, oid = item["sample_id"], item["object_id"]
        generated_row = lookup.get(sid)
        base: dict[str, Any] = {
            "model": model,
            "sample_id": sid,
            "object_id": oid,
            "view_id": item["view_id"],
            "generation_status": "not_applicable" if reference else "missing",
            "preparation_status": "not_run",
            "collision_status": "not_run",
            "mujoco_status": "not_run",
            "index_status": "not_applicable" if reference else "not_run",
            "topology_status": None,
            "pose_status": "reference_coordinates"
            if reference
            else "unverified_relative_to_reference",
            "contact_status": "not_run",
            "contact_valid_samples": 0,
            "contact_planned_samples": 0,
            "valid_trials": 0,
            "planned_trials": 0,
            "repair_operations": 0,
            "generation_quality_status": "not_applicable"
            if reference
            else "no_debug_geometry_or_appearance_metrics",
            "failure_stage": None,
            "reason": None,
        }
        prepared = output / "prepared" / sid
        try:
            if reference:
                if not item.get("reference_mesh"):
                    raise FileNotFoundError("Manifest has no reference_mesh")
                source = Path(item["reference_mesh"])
            else:
                if not generated_row:
                    raise FileNotFoundError(f"Generation record missing: {sid}")
                if generated_row.get("status") != "ok":
                    base["generation_status"] = "failed"
                    raise ValueError(
                        f"Generation failed: {generated_row.get('error', 'unknown reason')}"
                    )
                mesh_path = generated_row.get("mesh")
                if not mesh_path:
                    raise FileNotFoundError(f"Generation mesh path missing: {sid}")
                source = (run / mesh_path).resolve()
                if not source.is_relative_to(run) or not source.is_file():
                    raise FileNotFoundError(f"Generated mesh missing or outside run: {source}")
                if any(
                    generated_row.get(key, item[key]) != item[key]
                    for key in ("object_id", "view_id")
                ):
                    raise ValueError("Generation record object/view mismatch")
                base["generation_status"] = "ok"
            if not source.is_file():
                raise FileNotFoundError(f"Mesh missing: {source}")
        except (OSError, ValueError) as error:
            for condition in conditions:
                rows.append(
                    {
                        **base,
                        "condition": condition,
                        "failure_stage": "generation" if not reference else "reference_missing",
                        "reason": str(error),
                    }
                )
            _write_ledger(output, rows, context)
            continue
        try:
            metadata = prepare_mesh(source, prepared, options, run)
            base["preparation_status"] = "ok"
            base["topology_status"] = metadata["physical_adaptation"]["status"]
            base["topology_detail"] = metadata["physical_adaptation"]
            base["repair_operations"] = sum(
                op["operation"]
                in {
                    "merge_coincident_vertices",
                    "remove_degenerate_faces",
                    "remove_unreferenced_vertices",
                }
                for op in metadata["operations"]
            )
            base["preparation_metadata"] = str(prepared / "metadata.json")
        except Exception as error:  # noqa: BLE001 - preserve failures from all mesh backends
            for condition in conditions:
                rows.append(
                    {
                        **base,
                        "condition": condition,
                        "preparation_status": "failed",
                        "failure_stage": "mesh_preparation",
                        "reason": str(error),
                    }
                )
            _write_ledger(output, rows, context)
            continue
        for condition in conditions:
            row = {**base, "condition": condition}
            collider_dir = output / "collisions" / sid / condition
            try:
                collision = build_collision(
                    prepared / "normalized.glb", collider_dir, condition, collision_config
                )
                row.update(
                    collision_status="ok",
                    component_count=collision["component_count"],
                    collision_build_seconds=collision["build_seconds"],
                    surface_error_m=collision["surface_error"]["symmetric_mean_m"],
                    collision_record=str(collider_dir / "colliders.json"),
                )
            except Exception as error:  # noqa: BLE001
                row.update(
                    collision_status="failed", failure_stage="collision_build", reason=str(error)
                )
                rows.append(row)
                _write_ledger(output, rows, context)
                continue
            if not reference:
                try:
                    index = index_generated_assets(manifest, run, output, (condition,), [sid])[0]
                    row["index_status"] = index["status"]
                    if index["status"] != "ready":
                        raise ValueError(index["reason"])
                except Exception as error:  # noqa: BLE001
                    row.update(
                        index_status="failed", failure_stage="physics_index", reason=str(error)
                    )
                    rows.append(row)
                    _write_ledger(output, rows, context)
                    continue
            try:
                visual = prepared / "visual.obj"
                model_mj, _, validation = make_scene(
                    collider_dir,
                    prepared / "normalized.glb",
                    visual,
                    config,
                    xml_path=output / "scenes" / sid / f"{condition}.xml",
                )
                row.update(mujoco_status="ok", scene_validation=validation)
            except Exception as error:  # noqa: BLE001
                row.update(mujoco_status="failed", failure_stage="mujoco_load", reason=str(error))
                rows.append(row)
                _write_ledger(output, rows, context)
                continue
            if oid not in {"debug_box", "debug_ring"}:
                row.update(
                    contact_status="reference_labels_unavailable",
                    reason="No validated probe/reference protocol for this object",
                )
            else:
                all_samples, all_trials = [], []
                try:
                    for name, x in trajectories(oid).items():
                        model_mj, _, _ = make_scene(
                            collider_dir, prepared / "normalized.glb", visual, config, x
                        )
                        samples, trials = evaluate_trajectory(
                            model_mj, config, oid, condition, name, x, reference_enabled=reference
                        )
                        all_samples.extend(samples)
                        all_trials.extend(trials)
                    with (collider_dir / "probe_samples.json").open("w") as stream:
                        json.dump(
                            {"samples": all_samples, "trials": all_trials},
                            stream,
                            indent=2,
                            allow_nan=False,
                        )
                    row["solver_warning_trials"] = sum(
                        bool(trial["solver_warnings"]) for trial in all_trials
                    )
                    row["dynamic_inconsistency_count"] = sum(
                        trial["behavior_consistent"] is False for trial in all_trials
                    )
                    if reference:
                        measured = _summary(
                            all_samples, all_trials, collision, [collision["build_seconds"]]
                        )
                        row.update(
                            measured,
                            contact_status="reference_evaluated",
                            contact_valid_samples=measured["valid_samples"],
                            contact_planned_samples=measured["planned_samples"],
                        )
                    else:
                        valid_trials = [trial for trial in all_trials if trial["valid"]]
                        row.update(
                            contact_status="pose_unverified_no_accuracy",
                            contact_valid_samples=0,
                            contact_planned_samples=len(all_samples),
                            valid_trials=len(valid_trials),
                            planned_trials=len(all_trials),
                            forward_mean_seconds=float(
                                np.mean([s["forward_mean_s"] for s in all_samples])
                            ),
                            simulation_mean_seconds=float(
                                np.mean([t["simulation_seconds"] for t in valid_trials])
                            )
                            if valid_trials
                            else None,
                            observed_geometric_contact_samples=sum(
                                s["geometric_contact"] for s in all_samples
                            ),
                            observed_passed_trials=sum(t["passed"] for t in valid_trials),
                            reason="Reference pose correspondence not verified; observations only, no contact accuracy",
                        )
                except Exception as error:  # noqa: BLE001
                    row.update(
                        contact_status="failed",
                        failure_stage="contact_simulation",
                        reason=str(error),
                    )
            rows.append(row)
            _write_ledger(output, rows, context)
    context["status"] = (
        "completed_with_failures" if any(row["failure_stage"] for row in rows) else "completed"
    )
    context["denominators"] = {
        "planned_samples": len(selected),
        "generation_success_samples": sum(
            any(
                row["sample_id"] == item["sample_id"] and row["generation_status"] == "ok"
                for row in rows
            )
            for item in selected
        )
        if not reference
        else None,
        "preparation_success_samples": sum(
            any(
                row["sample_id"] == item["sample_id"] and row["preparation_status"] == "ok"
                for row in rows
            )
            for item in selected
        ),
        "planned_collision_cases": len(selected) * len(conditions),
        "collision_success_cases": sum(row["collision_status"] == "ok" for row in rows),
        "mujoco_success_cases": sum(row["mujoco_status"] == "ok" for row in rows),
        "reference_contact_evaluated_cases": sum(
            row["contact_status"] == "reference_evaluated" for row in rows
        ),
    }
    _write_ledger(output, rows, context)
    return {**context, "results": rows}
