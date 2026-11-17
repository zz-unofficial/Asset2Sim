"""Isolated official image-to-physical-asset execution, without model imports at startup."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from asset2sim.artifacts import git_revision, write_json
from asset2sim.data.manifest import load_manifest, select_samples
from asset2sim.experiments.physx_validation import sha256, validate_rle


def load_config(path: Path) -> dict[str, Any]:
    """Resolve server model paths relative to the configuration file."""
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError("PhysX-Omni config must be a mapping")
    for key in ("upstream", "model_path"):
        config[key] = str((path.parent / config[key]).resolve())
    for key in ("postprocess", "fixed_base", "deformable"):
        if not isinstance(config.get(key, False), bool):
            raise TypeError(f"{key} must be a YAML boolean")
        config.setdefault(key, False)
    config.setdefault("validate_mujoco", False)
    config.setdefault("stage_timeout_seconds", None)
    config.setdefault("measure_resources", False)
    config.setdefault("category_overrides", {})
    for key in ("validate_mujoco", "measure_resources"):
        if not isinstance(config[key], bool):
            raise TypeError(f"{key} must be a YAML boolean")
    if not isinstance(config["category_overrides"], dict):
        raise TypeError("category_overrides must be a mapping")
    for category, overrides in config["category_overrides"].items():
        if category not in {"rigid", "articulated", "deformable"} or not isinstance(
            overrides, dict
        ):
            raise ValueError("Invalid category override")
        if any(
            key not in {"postprocess", "fixed_base", "deformable"} or not isinstance(value, bool)
            for key, value in overrides.items()
        ):
            raise ValueError("Category overrides only accept boolean export switches")
    if config["stage_timeout_seconds"] is not None and config["stage_timeout_seconds"] <= 0:
        raise ValueError("stage_timeout_seconds must be positive or null")
    return config


def stage_commands(config: dict[str, Any], workspace: Path) -> list[tuple[str, list[str]]]:
    """Build argv lists; official relative outputs stay inside one sample workspace."""
    upstream = Path(config["upstream"])
    assets = str(workspace / "ours_demo")
    return [
        (
            "infer",
            [
                sys.executable,
                str(upstream / "1vlm_demo.py"),
                "--imagepath",
                str(workspace / "inputs"),
                "--modelpath",
                config["model_path"],
            ],
        ),
        (
            "decode",
            [
                sys.executable,
                str(upstream / "decoder_each.py"),
                "--name",
                "asset",
                "--basepath",
                assets,
                "--imgpath",
                str(workspace / "inputs/asset.png"),
            ],
        ),
        (
            "export",
            [
                sys.executable,
                str(upstream / "3jsongen_update.py"),
                "--basepath",
                assets,
                "--process",
                str(int(config["postprocess"])),
                "--fixed_base",
                str(int(config["fixed_base"])),
                "--deformable",
                str(int(config["deformable"])),
            ],
        ),
    ]


def check_stage(stage: str, asset: Path) -> dict[str, Any]:
    """Check artifacts because upstream catches exceptions and can exit with code zero."""
    if stage == "infer":
        import numpy as np

        if not (asset / "basic_info.txt").read_text().strip():
            raise ValueError("Empty basic_info.txt")
        parts = sorted(asset.glob("ind_*.npy"))
        coords = np.load(asset / "allind.npy", allow_pickle=False)
        if not parts or coords.ndim != 2 or coords.shape[1] != 3 or not coords.size:
            raise ValueError("Missing parts or invalid voxel coordinates")
        for part in parts:
            value = np.load(part, allow_pickle=False)
            if value.ndim != 2 or value.shape[1] != 3 or not value.size:
                raise ValueError(f"Invalid part coordinates: {part.name}")
            if not np.isfinite(value).all() or (value < 0).any() or (value >= 64).any():
                raise ValueError(f"Part coordinates must be finite and inside the 64-cube: {part}")
            if np.any(value != np.round(value)):
                raise ValueError(f"Noninteger part coordinates: {part.name}")
        if not np.isfinite(coords).all() or (coords < 0).any() or (coords >= 64).any():
            raise ValueError("Voxel coordinates must be finite and inside the 64-cube")
        if np.any(coords != np.round(coords)):
            raise ValueError("Voxel coordinates must be integer grid positions")
        declared = {
            int(value)
            for value in re.findall(
                r"^\s*l_(\d+)\s*:", (asset / "basic_info.txt").read_text(), re.MULTILINE
            )
        }
        saved = {int(part.stem.removeprefix("ind_")) for part in parts}
        if not declared or declared != saved or declared != set(range(len(declared))):
            raise ValueError("Predicted part IDs and saved voxel parts must match contiguously")
        if not np.array_equal(
            coords,
            np.concatenate(
                [np.load(asset / f"ind_{i}.npy", allow_pickle=False) for i in range(len(parts))]
            ),
        ):
            raise ValueError("Combined voxels differ from declared parts")
        from asset2sim.evaluation.physx_omni.audit import raw_parts

        descriptions = raw_parts(asset / "basic_info.txt")
        if set(descriptions) != saved or any(
            not item.get("name") or not item.get("description") for item in descriptions.values()
        ):
            raise ValueError("Raw part descriptions are incomplete")
        return {"parts": len(parts), "voxel_entries": len(coords)}
    if stage == "decode":
        import numpy as np
        import trimesh

        parts = sorted(asset.glob("ind_*.npy"))
        if not parts:
            raise ValueError("No predicted parts")
        for part in parts:
            index = part.stem.removeprefix("ind_")
            for suffix in ("glb", "obj"):
                path = asset / "objs" / index / f"{index}.{suffix}"
                mesh = trimesh.load(path, force="scene", process=False)
                if not mesh.geometry or not any(len(g.faces) for g in mesh.geometry.values()):
                    raise ValueError(f"Empty decoded geometry: {path}")
                if any(not np.isfinite(g.vertices).all() for g in mesh.geometry.values()):
                    raise ValueError(f"Nonfinite decoded geometry: {path}")
        return {"decoded_parts": len(parts)}
    if stage == "export":
        import json

        properties = json.loads((asset / "basic_info.json").read_text())
        if not properties.get("parts"):
            raise ValueError("No exported physical parts")
        for name in ("basic.urdf", "basic.xml"):
            ET.parse(asset / name)
        from asset2sim.evaluation.physx_omni.audit import audit_asset

        audit = audit_asset(asset, "rigid")
        structural_errors = [
            e
            for e in audit["errors"]
            if not e.startswith(("property_transfer:", "raw_to_json:", "invalid_physical_value:"))
        ]
        if structural_errors:
            raise ValueError(f"Export structure invalid: {structural_errors}")
        return {"physical_parts": len(properties["parts"]), "audit": audit}
    raise ValueError(f"Unknown stage: {stage}")


def validate_mjcf(asset: Path) -> dict[str, Any]:
    """Compile the official MJCF without changing geometry, mass or joints."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(asset / "basic.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qacc).all():
        raise ValueError("Nonfinite initial MuJoCo state")
    return {
        "mujoco_version": mujoco.__version__,
        "bodies": model.nbody,
        "geoms": model.ngeom,
        "joints": model.njnt,
        "body_mass_kg": model.body_mass.tolist(),
        "body_inertia_kg_m2": model.body_inertia.tolist(),
        "validation": "compile_and_forward",
    }


def run_physx_omni(
    config_path: Path,
    manifest: Path,
    output: Path,
    sample_ids: list[str] | None = None,
    *,
    plan_only: bool = False,
    validate_mujoco: bool | None = None,
    reuse_run: Path | None = None,
    from_stage: str = "infer",
    deformable: bool | None = None,
) -> dict[str, Any]:
    """Run selected existing image records; retain each failed stage and continue the batch."""
    config = load_config(config_path)
    if validate_mujoco is not None:
        config["validate_mujoco"] = validate_mujoco
    if from_stage not in {"infer", "decode", "export"}:
        raise ValueError("from_stage must be infer, decode or export")
    if from_stage != "infer" and reuse_run is None:
        raise ValueError("decode/export restart requires --reuse-run")
    upstream = Path(config["upstream"])
    for name in (
        "1vlm_demo.py",
        "decoder_each.py",
        "3jsongen_update.py",
        "dataset",
        "trellis",
        "mjcf_source",
    ):
        if not (upstream / name).exists():
            raise FileNotFoundError(upstream / name)
    records = select_samples(load_manifest(manifest), sample_ids)
    if not plan_only and from_stage == "infer" and not Path(config["model_path"]).is_dir():
        raise FileNotFoundError(
            f"Download PhysX-Omni weights on the server: {config['model_path']}"
        )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "planned" if plan_only else "running",
        "config": config,
        "upstream_commit": git_revision(upstream),
        "manifest": str(manifest.resolve()),
        "execution_policy": "serial_stage_subprocesses",
        "n_planned": len(records),
        "n_success": 0,
        "n_failed": 0,
        "samples": [],
        "from_stage": from_stage,
        "reuse_run": str(reuse_run.resolve()) if reuse_run else None,
        "route": "official_image_to_physical_asset",
        "property_source": "model_prior_not_measured_ground_truth",
        "decoder_model": "microsoft/TRELLIS-image-large",
        "processor": "Qwen/Qwen2.5-VL-7B-Instruct",
        "decoder_seed": 1,
        "vlm_do_sample": False,
        "vlm_max_length": 32768,
        "python": sys.version,
        "host": platform.platform(),
        "packages": {},
        "environment": {
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "ATTN_BACKEND", "SPCONV_ALGO", "HF_HUB_OFFLINE")
            if key in os.environ
        },
    }
    for package in ("torch", "transformers", "mujoco", "trimesh", "flash-attn", "spconv-cu120"):
        try:
            summary["packages"][package] = version(package)
        except PackageNotFoundError:
            summary["packages"][package] = None
    write_json(output / "run.json", summary)
    for record in records:
        workspace = output / record["sample_id"]
        category = record.get("category", "rigid")
        if category not in {"rigid", "articulated", "deformable"}:
            raise ValueError(f"Unknown physical category: {category}")
        sample_config = {**config, **config["category_overrides"].get(category, {})}
        if deformable is not None:
            sample_config["deformable"] = deformable
        commands = stage_commands(sample_config, workspace)
        item: dict[str, Any] = {
            "input": record,
            "category": category,
            "export_parameters": {
                key: sample_config[key] for key in ("postprocess", "fixed_base", "deformable")
            },
            "status": "planned",
            "stages": [],
            "commands": [{"stage": key, "argv": argv} for key, argv in commands],
            "asset_directory": str(workspace / "ours_demo/asset"),
            "coordinate_note": "Official GLB and OBJ differ by +90 degrees about X; no implicit Asset2Sim alignment",
        }
        summary["samples"].append(item)
        if plan_only:
            continue
        workspace.mkdir()
        for name in ("dataset", "trellis", "mjcf_source"):
            (workspace / name).symlink_to(upstream / name, target_is_directory=True)
        (workspace / "inputs").mkdir()
        from PIL import Image

        current: dict[str, Any] = {"stage": "input", "status": "running"}
        item["stages"].append(current)
        started = time.perf_counter()
        try:
            with Image.open(record["image"]) as image:
                image.convert("RGB").save(workspace / "inputs/asset.png")
            shutil.copyfile(
                record["image"], workspace / ("original_input" + Path(record["image"]).suffix)
            )
            current.update(
                status="ok",
                execution_success=True,
                output_valid=True,
                seconds=time.perf_counter() - started,
            )
            if reuse_run is not None and from_stage != "infer":
                source = reuse_run.resolve() / record["sample_id"]
                with (
                    Image.open(source / "inputs/asset.png") as previous,
                    Image.open(workspace / "inputs/asset.png") as actual,
                ):
                    import numpy as np

                    if not np.array_equal(np.asarray(previous.convert("RGB")), np.asarray(actual)):
                        raise ValueError("Reused generation input differs from selected RGB")
                shutil.copytree(source / "ours_demo", workspace / "ours_demo")
                for name in ("basic.xml", "basic.urdf", "basic_info.json"):
                    (workspace / "ours_demo/asset" / name).unlink(missing_ok=True)
            for stage, argv in commands:
                current = {
                    "stage": stage,
                    "status": "running",
                    "execution_success": None,
                    "output_valid": False,
                }
                item["stages"].append(current)
                write_json(output / "run.json", summary)
                started = time.perf_counter()
                if ("infer", "decode", "export").index(stage) < ("infer", "decode", "export").index(
                    from_stage
                ):
                    current.update(check_stage(stage, workspace / "ours_demo/asset"))
                    if stage == "infer":
                        current.update(validate_rle(workspace / "ours_demo/asset", upstream))
                    current.update(
                        status="ok",
                        reused=True,
                        output_valid=True,
                        seconds=0.0,
                        source_run=str(reuse_run.resolve()),
                    )
                    continue
                asset = workspace / "ours_demo/asset"
                if stage == "export":
                    frozen = workspace / "frozen_generation"
                    shutil.copytree(asset, frozen)
                    current["frozen_generation_sha256"] = {
                        str(p.relative_to(frozen)): sha256(p)
                        for p in frozen.rglob("*")
                        if p.is_file()
                    }
                env = dict(os.environ)
                env["PYTHONPATH"] = str(upstream) + os.pathsep + env.get("PYTHONPATH", "")
                with (workspace / f"{stage}.log").open("w") as log:
                    if config["measure_resources"]:
                        argv = [
                            sys.executable,
                            "-m",
                            "asset2sim.experiments.physx_omni_worker",
                            "--metrics",
                            str(workspace / f"{stage}_resources.json"),
                            "--",
                            *argv[1:],
                        ]
                    result = subprocess.run(
                        argv,
                        check=False,
                        cwd=workspace,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=config["stage_timeout_seconds"],
                    )
                current["returncode"] = result.returncode
                current["execution_success"] = result.returncode == 0
                metrics = workspace / f"{stage}_resources.json"
                if metrics.is_file():
                    current["resources"] = json.loads(metrics.read_text())
                asset = workspace / "ours_demo/asset"
                if stage == "export":
                    snapshot = workspace / (
                        "export_deformable" if sample_config["deformable"] else "export_default"
                    )
                    shutil.copytree(asset, snapshot)
                    shutil.copyfile(workspace / "export.log", snapshot / "export.log")
                    current["export_snapshot"] = str(snapshot)
                    current["artifact_sha256"] = {
                        str(p.relative_to(snapshot)): sha256(p)
                        for p in snapshot.rglob("*")
                        if p.is_file()
                    }
                current.update(check_stage(stage, asset))
                if stage == "infer":
                    current.update(validate_rle(asset, upstream))
                current["output_valid"] = True
                if result.returncode:
                    raise RuntimeError(
                        f"{stage} returned {result.returncode}; see {workspace / (stage + '.log')}"
                    )
                current["artifact_bytes_cumulative"] = sum(
                    path.stat().st_size
                    for path in (workspace / "ours_demo/asset").rglob("*")
                    if path.is_file()
                )
                current.update(status="ok", seconds=time.perf_counter() - started)
            if config["validate_mujoco"]:
                current = {"stage": "mujoco_load", "status": "running"}
                item["stages"].append(current)
                started = time.perf_counter()
                current.update(validate_mjcf(workspace / "ours_demo/asset"))
                current.update(status="ok", seconds=time.perf_counter() - started)
            item["status"] = "ok"
            summary["n_success"] += 1
        except Exception as error:  # noqa: BLE001 -- preserve every sample's failure and continue.
            current.update(
                status="failed",
                reason=f"{type(error).__name__}: {error}",
                seconds=time.perf_counter() - started,
            )
            item["status"] = "failed"
            summary["n_failed"] += 1
            completed = {entry["stage"] for entry in item["stages"]}
            for stage in ("infer", "decode", "export") + (
                ("mujoco_load",) if config["validate_mujoco"] else ()
            ):
                if stage not in completed:
                    item["stages"].append(
                        {"stage": stage, "status": "blocked", "reason": "previous_stage_failed"}
                    )
        write_json(output / "run.json", summary)
    if not plan_only:
        summary["status"] = "failed" if summary["n_failed"] else "complete"
    summary["categories"] = {
        category: {
            "n_planned": sum(sample["category"] == category for sample in summary["samples"]),
            "n_success": sum(
                sample["category"] == category and sample["status"] == "ok"
                for sample in summary["samples"]
            ),
        }
        for category in sorted({sample["category"] for sample in summary["samples"]})
    }
    write_json(output / "run.json", summary)
    return summary
