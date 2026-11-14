"""Run and record a small Menagerie UR5e / Robotiq comparison."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.grasp.controller import scripted_trial
from asset2sim.simulation.grasp.scene import create_scene


def resolve_asset(asset: dict) -> tuple[Path, Path, dict]:
    """Validate the existing batch and return prepared geometry, colliders and provenance."""
    batch = (PROJECT_ROOT / asset["batch"]).resolve()
    sid = asset["sample_id"]
    condition = asset["condition"]
    ledger = json.loads((batch / "batch.json").read_text())
    rows = [
        row
        for row in ledger["results"]
        if row["sample_id"] == sid and row["condition"] == condition
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one asset for sample_id={sid}, condition={condition} in {batch}; found {len(rows)}"
        )
    if rows[0]["collision_status"] != "ok" or rows[0]["mujoco_status"] != "ok":
        raise ValueError(
            f"Asset {sid} ({condition}) did not pass physics batch: collision={rows[0].get('collision_status')}, mujoco={rows[0].get('mujoco_status')}"
        )
    if ledger["model"] != "Reference" and rows[0].get("index_status") != "ready":
        raise ValueError("Generated asset did not pass provenance index")
    prepared = batch / "prepared" / sid
    colliders = batch / "collisions" / sid / condition
    record = json.loads((colliders / "colliders.json").read_text())
    if Path(record["input"]).resolve() != (prepared / "normalized.glb").resolve():
        raise ValueError("Collider does not belong to prepared mesh")
    return prepared, colliders, ledger


def evaluate_grasps(config_path: Path, output: Path) -> dict:
    """Run all planned trials, retaining load failures in the denominator."""
    config = yaml.safe_load(Path(config_path).read_text())
    count = int(config["trials_per_asset"])
    if count < 1:
        raise ValueError("trials_per_asset must be positive")
    menagerie = (PROJECT_ROOT / config["menagerie"]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(menagerie), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["menagerie_revision"]:
        raise ValueError("Menagerie version does not match configured version")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "environment": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "menagerie_revision": revision,
        },
        "protocol": "No object weld; success requires force-bearing bilateral finger contact, lifted hold and release onto table",
        "assets": {},
        "trials": [],
    }
    for name, asset in config["assets"].items():
        batch = (PROJECT_ROOT / asset["batch"]).resolve()
        sid = asset["sample_id"]
        condition = asset["condition"]
        try:
            prepared, colliders, ledger = resolve_asset(asset)
            model, scene = create_scene(
                menagerie, prepared, colliders, config, output / name / "model"
            )
            result["assets"][name] = {
                "status": "loaded",
                "batch": str(batch),
                "model": ledger["model"],
                "sample_id": sid,
                "condition": condition,
                **scene,
            }
        except Exception as error:  # noqa: BLE001 - preserve planned failures
            result["assets"][name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            for trial in range(count):
                result["trials"].append(
                    {
                        "asset": name,
                        "trial": trial,
                        "status": "failed",
                        "valid": False,
                        "success": False,
                        "error": str(error),
                    }
                )
            continue
        for trial in range(count):
            record = {"asset": name, "trial": trial, "status": "running"}
            try:
                path = output / name / f"trial_{trial:02d}"
                measured = scripted_trial(model, config, path, video=trial == 0)
                record.update(
                    measured,
                    status="ok" if measured["valid"] else "failed",
                    trajectory=str(path / "trajectory.csv"),
                    video=str(path / "grasp.mp4") if trial == 0 else None,
                )
            except Exception as error:  # noqa: BLE001
                record.update(
                    status="failed",
                    valid=False,
                    success=False,
                    error=f"{type(error).__name__}: {error}",
                )
            result["trials"].append(record)
            (output / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["status"] = (
        "completed_with_failures"
        if any(x["status"] != "ok" for x in result["trials"])
        else "completed"
    )
    result["summary"] = {}
    for name in config["assets"]:
        rows = [r for r in result["trials"] if r["asset"] == name]
        result["summary"][name] = {
            "planned": count,
            "valid": sum(r.get("valid", False) for r in rows),
            "successes": sum(r.get("success", False) for r in rows),
            "success_rate_all_planned": sum(r.get("success", False) for r in rows) / count,
        }
    (output / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = [
        "asset",
        "trial",
        "status",
        "valid",
        "success",
        "error",
        "hold_valid_samples",
        "hold_success_samples",
        "hold_success_fraction",
        "bilateral_contact_samples",
        "release_success",
        "settle_xy_displacement_m",
        "max_lift_m",
        "end_height_m",
        "mj_step_seconds",
        "steps",
        "video",
        "trajectory",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["trials"])
    return result
