"""A single-image or imported-GLB path into the existing interactive robot scene."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

from asset2sim.collision import build_collision
from asset2sim.geometry.prepare import PreparationOptions, prepare_mesh
from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.grasp.controller import inverse_kinematics
from asset2sim.simulation.grasp.scene import create_scene


def asset_pipeline(
    config_path: Path,
    output: Path,
    *,
    image=None,
    mesh=None,
    run=None,
    model=None,
    up_axis=None,
    extent=None,
    prepare_only=False,
    device=None,
):
    """Record every stage, preserve raw inputs, and launch an optional human session."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "configuration": config,
        "config_path": str(config_path),
        "stages": [],
        "inputs": {
            "image": str(image) if image else None,
            "mesh": str(mesh) if mesh else None,
            "run": str(run) if run else None,
        },
    }

    def save():
        (output / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")

    stage = "input"
    try:
        save()
        selected = model or config["model"]
        snapshot = copy.deepcopy(config)
        snapshot["model"] = selected
        snapshot["target_extent_m"] = float(
            extent if extent is not None else config["target_extent_m"]
        )
        for key in ("scene_config", "interactive_config", "collision_config"):
            snapshot[key] = str((config_path.parent / config[key]).resolve())
        snapshot["model_configs"] = {
            name: str((config_path.parent / path).resolve())
            for name, path in config["model_configs"].items()
        }
        if device:
            snapshot.setdefault("model_overrides", {}).setdefault(selected, {})["device"] = device
        (output / "configuration.yaml").write_text(yaml.safe_dump(snapshot))
        record["resolved_configuration"] = snapshot
        record["prepare_only"] = prepare_only
        raw = output / "raw"
        raw.mkdir()
        if image is not None:
            stage = "generation"
            source_image = Path(image).resolve(strict=True)
            copied = raw / ("input" + source_image.suffix)
            shutil.copyfile(source_image, copied)
            manifest = output / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "asset_v0",
                        "object_id": "asset",
                        "view_id": "v0",
                        "image": str(copied),
                        "split": "interactive",
                    }
                )
                + "\n"
            )
            experiment = output / "generation.yaml"
            experiment.write_text(
                yaml.safe_dump(
                    {
                        "name": "interactive_asset",
                        "model": selected,
                        "model_config": str(
                            (config_path.parent / config["model_configs"][selected]).resolve()
                        ),
                        "manifest": str(manifest),
                        "overrides": config.get("model_overrides", {}).get(selected, {})
                        | ({"device": device} if device else {}),
                    }
                )
            )
            run = output / "generation"
            command = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                config["model_environments"][selected],
                "asset2sim",
                "run",
                "--config",
                str(experiment),
                "--output",
                str(run),
            ]
            record["generation_command"] = command
            save()
            started = time.perf_counter()
            with (output / "generation.log").open("w") as log:
                try:
                    subprocess.run(
                        command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Generation failed with exit {exc.returncode}; see {output / 'generation.log'}"
                    ) from exc
            record["stages"].append(
                {"stage": stage, "status": "ok", "seconds": time.perf_counter() - started}
            )
            mesh = run / "asset_v0/mesh.glb"
        else:
            mesh = Path(mesh).resolve(strict=True)
            if mesh.suffix.lower() != ".glb":
                raise ValueError("Single-asset import requires a GLB file")
        original = raw / "mesh.glb"
        shutil.copyfile(mesh, original)
        # A declared import ledger allows the existing strict preparation provenance
        # checks to run; it does not claim this asset was generated by Asset2Sim.
        if run is None:
            if up_axis not in ("z", "y"):
                raise ValueError("Imported GLB requires --run or explicit --up-axis y/z")
            run = output / "import"
            run.mkdir()
            rotation = np.eye(4)
            if up_axis == "y":
                rotation[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            (run / "run.json").write_text(
                json.dumps(
                    {
                        "model": "declared_external_glb",
                        "raw_mesh_coordinates": {
                            "up_axis": "+" + up_axis.upper(),
                            "units": "unknown source units; normalized to declared target meters",
                            "raw_to_model_z_up": rotation.tolist(),
                        },
                        "imported": True,
                    }
                )
            )
            (run / "results.jsonl").write_text(
                json.dumps({"sample_id": "asset_v0", "status": "ok", "mesh": str(original)}) + "\n"
            )
            mesh = original
        else:
            run = Path(run).resolve(strict=True)
        stage = "mesh_preparation"
        target = float(extent if extent is not None else config["target_extent_m"])
        metadata = prepare_mesh(
            Path(mesh),
            output / "prepared",
            PreparationOptions(kind="generated", target_extent_m=target, **config["cleaning"]),
            run,
        )
        if metadata["status"] != "completed":
            raise ValueError(metadata.get("error", "Mesh preparation failed"))
        record["stages"].append(
            {"stage": stage, "status": "ok", "physical_adaptation": metadata["physical_adaptation"]}
        )
        stage = "collision"
        collision_settings = yaml.safe_load(
            (config_path.parent / config["collision_config"]).read_text()
        )
        collision_settings["coacd"]["preprocess_mode"] = config["collision_preprocess"]
        collision_settings_path = output / "collision_settings.yaml"
        collision_settings_path.write_text(yaml.safe_dump(collision_settings))
        collision = build_collision(
            output / "prepared/normalized.glb",
            output / "collision",
            config["condition"],
            collision_settings_path,
            allow_open=config["allow_open"],
        )
        if collision["status"] != "completed":
            raise ValueError(collision.get("error", "Collision failed"))
        record["stages"].append(
            {"stage": stage, "status": "ok", "components": collision["component_count"]}
        )
        stage = "scene"
        scene_config = yaml.safe_load((config_path.parent / config["scene_config"]).read_text())
        scene_config.update(
            object_scale=1.0,
            placement_from_collision=True,
            physical_profile=config["physical_profile"],
        )
        m, info = create_scene(
            (PROJECT_ROOT / scene_config["menagerie"]).resolve(),
            output / "prepared",
            output / "collision",
            scene_config,
            output / "scene",
            visual_source=original,
            visual_transform=metadata["scene_to_physical"],
        )
        from asset2sim.geometry.prepare import load_geometry

        geometry, _ = load_geometry(output / "prepared/normalized.glb")
        if not np.allclose(info["visual"]["bounds_m"], geometry.bounds, atol=1e-6):
            raise ValueError("Visual and prepared mesh coordinate bounds disagree")
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        home = np.r_[scene_config["object_xy_m"], data.qpos[m.jnt_qposadr[joint] + 2] + 0.20]
        controls = yaml.safe_load((config_path.parent / config["interactive_config"]).read_text())
        if np.any(home < controls["workspace_min_m"]) or np.any(home > controls["workspace_max_m"]):
            raise ValueError(f"Initial approach outside workspace: {home.tolist()}")
        inverse_kinematics(m, home, np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]))
        count = sum(m.geom_bodyid[i] == obj and bool(m.geom_contype[i]) for i in range(m.ngeom))
        if count != collision["component_count"]:
            raise ValueError("Loaded collision geom count mismatch")
        checks = {
            "initial_approach_reachable": True,
            "y_width_m": info["dimensions_m"][1],
            "nominal_gripper_opening_m": 0.085,
            "y_width_within_opening": info["dimensions_m"][1] < 0.085,
            "note": "AABB opening check is not a grasp feasibility guarantee; manual pose selection required",
        }
        checks["center_inside_workspace"] = bool(
            np.all(
                np.r_[scene_config["object_xy_m"], info["object_initial_pos_m"][2]]
                >= controls["workspace_min_m"]
            )
            and np.all(
                np.r_[scene_config["object_xy_m"], info["object_initial_pos_m"][2]]
                <= controls["workspace_max_m"]
            )
        )
        if not checks["y_width_within_opening"]:
            print(
                "Placement warning: Y bounding width exceeds nominal gripper opening; choose another manual orientation.",
                flush=True,
            )
        asset = {
            "model": selected if image is not None else metadata.get("model"),
            "raw_mesh": str(original),
            "scene_config": scene_config,
            "interactive_config": controls,
            "scene": info,
            "placement_checks": checks,
        }
        (output / "asset.json").write_text(json.dumps(asset, indent=2) + "\n")
        (output / "physical_properties.json").write_text(
            json.dumps(info["mass_properties"], indent=2) + "\n"
        )
        record["stages"].append({"stage": stage, "status": "ok", "checks": checks})
        record["status"] = "prepared"
        save()
        if not prepare_only:
            stage = "interaction"
            from asset2sim.simulation.grasp.interactive import interactive_grasp

            interactive_grasp(
                (config_path.parent / config["interactive_config"]).resolve(),
                output / "interaction",
                asset_dir=output,
            )
            record["stages"].append({"stage": stage, "status": "ok"})
            record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["stages"].append(
            {"stage": stage, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise
    finally:
        save()
    return record
