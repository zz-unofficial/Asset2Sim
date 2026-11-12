"""Reproducible headless contact comparisons for procedural debug objects."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from asset2sim.collision import build_collision
from asset2sim.geometry.prepare import PreparationOptions, prepare_mesh
from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.reference import distance_to_solid, trajectories
from asset2sim.simulation.scene import make_scene


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _probe_results(model, data):
    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    contacts = [c for c in data.contact[: data.ncon] if probe in (c.geom1, c.geom2)]
    dist = min((float(c.dist) for c in contacts), default=None)
    return {
        "min_detected_distance_m": dist,
        "geometric_contact": any(c.dist <= 0 for c in contacts),
        "solver_active": any(c.efc_address >= 0 for c in contacts),
        "buffer_only": bool(contacts) and all(c.dist > 0 for c in contacts),
        "contact_count": len(contacts),
    }


def evaluate_trajectory(
    model, config, object_id: str, condition: str, name: str, x_offset: float
) -> tuple[list[dict], list[dict]]:
    start, end = config["trajectory_start_y_m"], config["trajectory_end_y_m"]
    radius, repeats = config["probe_radius_m"], config["timing_repeats"]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_y")
    qadr, dadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data = mujoco.MjData(model)
    samples = []
    for index, y in enumerate(np.linspace(start, end, config["trajectory_samples"])):
        point = np.array([x_offset, y, 0.0])
        signed = distance_to_solid(object_id, point) - radius
        valid = abs(signed) > config["reference_tolerance_m"]
        reference = signed < 0
        timings = []
        for _ in range(repeats):
            data.qpos[qadr] = y - start
            data.qvel[dadr] = 0
            begin = time.perf_counter_ns()
            mujoco.mj_forward(model, data)
            timings.append((time.perf_counter_ns() - begin) / 1e9)
        samples.append(
            {
                "object": object_id,
                "condition": condition,
                "trajectory": name,
                "sample_index": index,
                "x_m": x_offset,
                "y_m": float(y),
                "z_m": 0.0,
                "reference_distance_to_solid_m": float(signed + radius),
                "reference_contact": reference,
                "reference_valid": valid,
                **_probe_results(model, data),
                "forward_mean_s": float(np.mean(timings)),
                "forward_repeats": repeats,
            }
        )
    reference_blocked = any(row["reference_contact"] for row in samples if row["reference_valid"])
    trials = []
    for repeat in range(repeats):
        data = mujoco.MjData(model)
        begin = time.perf_counter_ns()
        for step in range(config["simulation_steps"]):
            data.ctrl[0] = (end - start) * min((step + 1) / config["drive_steps"], 1.0)
            mujoco.mj_step(model, data)
        elapsed = (time.perf_counter_ns() - begin) / 1e9
        final_position = float(start + data.qpos[qadr])
        valid = bool(np.isfinite(final_position) and np.isfinite(data.qvel).all())
        passed = bool(final_position >= end - config["pass_tolerance_m"]) if valid else None
        final_contact = None
        if valid:
            mujoco.mj_forward(model, data)
            final_contact = _probe_results(model, data)["solver_active"]
            if not passed and not final_contact:
                valid = False
        trials.append(
            {
                "object": object_id,
                "condition": condition,
                "trajectory": name,
                "repeat": repeat,
                "reference_blocked": reference_blocked,
                "final_y_m": final_position if valid else None,
                "passed": passed,
                "final_solver_active": final_contact,
                "behavior_consistent": (passed != reference_blocked) if valid else None,
                "valid": valid,
                "invalid_reason": "stalled_without_active_contact"
                if not valid and final_contact is False
                else "nonfinite_state"
                if not valid
                else "",
                "simulation_seconds": elapsed,
                "simulation_steps": config["simulation_steps"],
            }
        )
    return samples, trials


def _summary(
    samples: list[dict], trials: list[dict], collision: dict, build_times: list[float]
) -> dict:
    valid = [row for row in samples if row["reference_valid"]]
    tp = sum(row["reference_contact"] and row["geometric_contact"] for row in valid)
    fn = sum(row["reference_contact"] and not row["geometric_contact"] for row in valid)
    fp = sum(not row["reference_contact"] and row["geometric_contact"] for row in valid)
    tn = sum(not row["reference_contact"] and not row["geometric_contact"] for row in valid)
    valid_trials = [row for row in trials if row["valid"]]
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid_samples": len(valid),
        "planned_samples": len(samples),
        "reference_positive": tp + fn,
        "reference_negative": fp + tn,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
        "buffer_only_samples": sum(row["buffer_only"] for row in valid),
        "solver_active_samples": sum(row["solver_active"] for row in valid),
        "valid_trials": len(valid_trials),
        "planned_trials": len(trials),
        "behavior_consistency": sum(row["behavior_consistent"] for row in valid_trials)
        / len(valid_trials)
        if valid_trials
        else None,
        "component_count": collision["component_count"],
        "collision_build_seconds": float(np.mean(build_times)),
        "collision_build_repeats": len(build_times),
        "collision_build_times_seconds": json.dumps(build_times),
        "forward_mean_seconds": float(np.mean([row["forward_mean_s"] for row in valid]))
        if valid
        else None,
        "simulation_mean_seconds": float(
            np.mean([row["simulation_seconds"] for row in valid_trials])
        )
        if valid_trials
        else None,
    }


def run_physics_eval(config_path: Path, output: Path) -> dict:
    """Build fresh collision assets and evaluate all configured conditions in one run."""
    output = Path(output).resolve()
    config = yaml.safe_load(Path(config_path).read_text())
    if config["body_mode"] != "static":
        raise ValueError("Controlled trajectory comparisons require a static object")
    if (
        config["timing_repeats"] < 1
        or config["build_repeats"] < 1
        or config["trajectory_samples"] < 2
        or config["simulation_steps"] < 1
        or config["drive_steps"] < 1
        or config["drive_steps"] > config["simulation_steps"]
    ):
        raise ValueError("Invalid sampling, timing, or simulation count")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "environment": {
            "cpu": _cpu_name(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "coacd": importlib.metadata.version("coacd"),
            "numpy": np.__version__,
            "timing_repeats": config["timing_repeats"],
        },
        "units": {"length": "m", "time": "s", "mass": "kg"},
        "definitions": {
            "contact": "detected sphere/mesh signed distance <= 0",
            "buffer_only": "detected contact(s), all signed distances > 0",
            "solver_active": "at least one contact efc_address >= 0",
            "behavior": "finite simulated final probe position within pass tolerance of target",
            "reference": "analytic distance to programmatic 64-sided annulus or normalized box",
        },
        "results": [],
        "failures": [],
    }
    samples_all, trials_all = [], []
    try:
        for object_id, source in config["objects"].items():
            prepared = output / "prepared" / object_id
            try:
                prepare_mesh(PROJECT_ROOT / source, prepared, PreparationOptions(kind="reference"))
            except Exception as error:  # noqa: BLE001
                result["failures"].append(
                    {"object": object_id, "stage": "prepare", "error": str(error)}
                )
                continue
            for condition in config["conditions"]:
                try:
                    collision_dir = output / "collisions" / object_id / condition
                    collision = build_collision(
                        prepared / "normalized.glb",
                        collision_dir,
                        condition,
                        PROJECT_ROOT / config["collision_config"],
                    )
                    build_times = [collision["build_seconds"]]
                    for repeat in range(1, config["build_repeats"]):
                        repeated = build_collision(
                            prepared / "normalized.glb",
                            output
                            / "collision_timing"
                            / object_id
                            / condition
                            / f"repeat_{repeat:02d}",
                            condition,
                            PROJECT_ROOT / config["collision_config"],
                        )
                        if repeated["component_count"] != collision["component_count"]:
                            raise ValueError("Repeated builds produced different component counts")
                        build_times.append(repeated["build_seconds"])
                    samples, trials, validations = [], [], []
                    if config.get("validate_dynamic_body"):
                        dynamic_model, dynamic_data, dynamic_validation = make_scene(
                            collision_dir,
                            prepared / "normalized.glb",
                            prepared / "visual.obj",
                            config,
                            0.0,
                            "dynamic",
                            output / "scenes" / object_id / f"{condition}_dynamic.xml",
                        )
                        mujoco.mj_step(dynamic_model, dynamic_data)
                        if not np.isfinite(dynamic_data.qpos).all():
                            raise ValueError(
                                "Dynamic free body step produced non-finite coordinates"
                            )
                        validations.append(
                            {"trajectory": "dynamic_load_check", **dynamic_validation}
                        )
                    for name, x in trajectories(object_id).items():
                        model, _, validation = make_scene(
                            collision_dir,
                            prepared / "normalized.glb",
                            prepared / "visual.obj",
                            config,
                            x,
                            config["body_mode"],
                            output / "scenes" / object_id / f"{condition}_{name}.xml",
                        )
                        rows, behavior = evaluate_trajectory(
                            model, config, object_id, condition, name, x
                        )
                        samples.extend(rows)
                        trials.extend(behavior)
                        validations.append({"trajectory": name, **validation})
                    samples_all.extend(samples)
                    trials_all.extend(trials)
                    result["results"].append(
                        {
                            "object": object_id,
                            "condition": condition,
                            **_summary(samples, trials, collision, build_times),
                            "scene_validation": validations,
                        }
                    )
                except Exception as error:  # noqa: BLE001
                    result["failures"].append(
                        {
                            "object": object_id,
                            "condition": condition,
                            "stage": "collision_or_simulation",
                            "error": str(error),
                        }
                    )
        result["status"] = "failed" if result["failures"] else "completed"
    finally:
        result["failed_contact_samples"] = [
            {
                "object": row["object"],
                "condition": row["condition"],
                "trajectory": row["trajectory"],
                "sample_index": row["sample_index"],
                "y_m": row["y_m"],
                "reference_contact": row["reference_contact"],
                "geometric_contact": row["geometric_contact"],
            }
            for row in samples_all
            if row["reference_valid"] and row["reference_contact"] != row["geometric_contact"]
        ]
        _write_csv(output / "contact_samples.csv", samples_all)
        _write_csv(output / "trajectory_trials.csv", trials_all)
        _write_csv(
            output / "summary.csv",
            [
                {k: v for k, v in item.items() if k != "scene_validation"}
                for item in result["results"]
            ],
        )
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
