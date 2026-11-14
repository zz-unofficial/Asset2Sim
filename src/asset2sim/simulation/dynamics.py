"""Headless controlled rigid-body dynamics with fixed collision components."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections import deque
from pathlib import Path
from xml.etree import ElementTree as ET

# MuJoCo's graphics backend must be selected before its first import.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from asset2sim.geometry.prepare import load_geometry
from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.properties import estimate_properties
from asset2sim.simulation.scene import make_scene


def _values(values) -> str:
    return " ".join(str(float(v)) for v in values)


def _cpu() -> str:
    try:
        return next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        return platform.processor()


def make_dynamic_scene(
    prepared: Path, colliders: Path, properties: dict, config: dict, task: str, output: Path
):
    """Use the existing collider scene, then add gravity, ground and a sliding pusher."""
    if task not in {"drop", "placement", "push"}:
        raise ValueError(f"Unknown dynamic task {task}")
    physics, video = config["physics"], config["video"]
    collision = json.loads((colliders / "colliders.json").read_text())
    bounds = np.array([part["aabb_m"] for part in collision["components"]])
    lower, upper = bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)
    output.mkdir(parents=True, exist_ok=False)
    visual_obj, xml_path = output / "visual.obj", output / "scene.xml"
    # Existing loader verifies every component and creates a portable asset bundle.
    base_config = yaml.safe_load((PROJECT_ROOT / "configs/physics/debug.yaml").read_text())
    _, _, validation = make_scene(
        colliders,
        prepared / "normalized.glb",
        visual_obj,
        base_config,
        body_mode="dynamic",
        xml_path=xml_path,
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()
    option = root.find("option")
    option.attrib.update(
        timestep=str(physics["timestep_s"]),
        gravity=_values(physics["gravity_m_s2"]),
        solver=physics["solver"],
        iterations=str(physics["solver_iterations"]),
    )
    world = root.find("worldbody")
    old_probe = world.find("body[@name='probe']")
    world.remove(old_probe)
    root.remove(root.find("contact"))
    root.remove(root.find("actuator"))
    body = world.find("body[@name='object']")
    clearance = (
        physics["initial_clearance_m"] if task == "drop" else physics["placement_clearance_m"]
    )
    body.attrib["pos"] = _values([0, 0, -lower[2] + clearance])
    inertial = body.find("inertial")
    inertia = np.asarray(properties["inertia_kg_m2"])
    inertial.attrib.pop("diaginertia", None)
    inertial.attrib.update(
        pos=_values(properties["center_of_mass_m"]),
        mass=str(properties["mass_kg"]),
        fullinertia=_values(
            [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ]
        ),
    )
    contact_attrs = {
        "friction": _values(physics["friction"]),
        "margin": str(physics["contact_margin_m"]),
        "gap": str(physics["contact_gap_m"]),
        "solref": _values(physics["solref"]),
        "solimp": _values(physics["solimp"]),
        "condim": "3",
    }
    for geom in body.findall("geom"):
        if geom.attrib["name"].startswith("collision_"):
            geom.attrib.update(contact_attrs)
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="1 1 0.1",
        rgba="0.55 0.58 0.62 1",
        **contact_attrs,
    )
    ET.SubElement(world, "light", pos="0 -0.4 1.2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    if task == "push":
        radius = physics["push_radius_m"]
        # Keep it outside the workspace until the object has settled. The same
        # bounding-box rule positions the pusher for every asset and profile.
        pusher = ET.SubElement(world, "body", name="pusher", pos="-1 0 0.1")
        ET.SubElement(
            pusher,
            "joint",
            name="pusher_x",
            type="slide",
            axis="1 0 0",
            damping=str(physics["push_damping_n_s_m"]),
        )
        ET.SubElement(
            pusher,
            "geom",
            name="pusher_geom",
            type="sphere",
            size=str(radius),
            mass=str(physics["push_mass_kg"]),
            rgba="0.9 0.2 0.1 1",
            **contact_attrs,
        )
        actuator = ET.SubElement(root, "actuator")
        ET.SubElement(
            actuator,
            "position",
            name="push_drive",
            joint="pusher_x",
            kp=str(physics["push_kp_n_m"]),
        )
    tree.write(xml_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if not np.isclose(model.body_mass[object_id], properties["mass_kg"], rtol=1e-8):
        raise ValueError("MuJoCo compiled body mass does not match selected profile")
    if not np.allclose(model.body_ipos[object_id], properties["center_of_mass_m"], atol=1e-9):
        raise ValueError("MuJoCo compiled COM does not match selected profile")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[object_id])
    rotation = rotation.reshape(3, 3)
    compiled_inertia = rotation @ np.diag(model.body_inertia[object_id]) @ rotation.T
    if not np.allclose(compiled_inertia, inertia, rtol=1e-6, atol=1e-10):
        raise ValueError("MuJoCo compiled inertia does not match selected profile")
    return (
        model,
        data,
        {
            "component_validation": validation,
            "collision_bounds_m": [lower.tolist(), upper.tolist()],
            "initial_clearance_m": clearance,
            "scene": str(xml_path),
            "video": video,
        },
    )


def _camera(config: dict) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.10]
    camera.distance = config["camera_distance_m"]
    camera.azimuth = config["camera_azimuth_deg"]
    camera.elevation = config["camera_elevation_deg"]
    return camera


def simulate(model, data, config: dict, task: str, video_path: Path | None = None) -> dict:
    """Simulate once, optionally recording RGB frames; all timings in wall seconds."""
    physics, video = config["physics"], config["video"]
    dt = float(physics["timestep_s"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Simulation timestep must be finite and positive")
    settle_window = float(physics.get("push_settle_window_s", 0.2))
    settle_timeout = float(physics.get("push_settle_timeout_s", 3.0))
    post_hold = float(physics.get("push_post_hold_s", 0.4))
    if task == "push" and (
        settle_window <= 0
        or settle_timeout < physics["push_delay_s"] + settle_window
        or post_hold < 0
        or physics["push_duration_s"] <= 0
    ):
        raise ValueError("Invalid push settling or duration configuration")
    duration = (
        settle_timeout + physics["push_duration_s"] + post_hold
        if task == "push"
        else physics["duration_s"]
    )
    steps = round(duration / dt)
    if steps < 1 or physics["sample_interval_steps"] < 1:
        raise ValueError("Invalid simulation duration, timestep or sampling interval")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geoms = {
        i for i in range(model.ngeom) if model.geom_bodyid[i] == object_id and model.geom_contype[i]
    }
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    pusher_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
    pusher_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
    pusher_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pusher_x")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    if task == "push":
        model.body_pos[pusher_body_id] = [-1, 0, 0.1]
    mujoco.mj_forward(model, data)
    initial_pos = data.xpos[object_id].copy()
    initial_quat = data.qpos[qadr + 3 : qadr + 7].copy()
    first_ground, pusher_contact_steps, ground_steps = None, 0, 0
    pusher_start = None
    push_start_time = None
    settle_timed_out = False
    settle_history = deque(maxlen=max(1, round(settle_window / dt)))
    push_object_position = push_object_quaternion = None
    min_contact_dist = None
    samples, step_times = [], []
    renderer = writer = None
    video_error = None
    video_seconds = 0.0
    if video_path is not None:
        try:
            renderer = mujoco.Renderer(
                model, height=int(video["height"]), width=int(video["width"])
            )
            camera = _camera(video)
            writer = imageio.get_writer(
                str(video_path), fps=int(video["fps"]), codec="libx264", macro_block_size=1
            )
        except Exception as error:  # noqa: BLE001 - retain failed video while still simulating
            video_error = f"{type(error).__name__}: {error}"
            if renderer is not None:
                renderer.close()
                renderer = None
    next_frame = 0.0
    started = time.perf_counter()
    try:
        for step in range(steps):
            t = step * dt
            if task == "push":
                if pusher_start is None and t >= settle_timeout:
                    settle_timed_out = True
                    break
                if (
                    push_start_time is not None
                    and t >= push_start_time + physics["push_duration_s"] + post_hold
                ):
                    break
                settled = (
                    len(settle_history) == settle_history.maxlen
                    and all(entry[0] for entry in settle_history)
                    and max(entry[1] for entry in settle_history)
                    < physics["stable_linear_speed_m_s"]
                    and max(entry[2] for entry in settle_history)
                    < physics["stable_angular_speed_rad_s"]
                    and max(entry[3] for entry in settle_history)
                    - min(entry[3] for entry in settle_history)
                    < physics["stable_height_range_m"]
                )
                if t >= physics["push_delay_s"] and pusher_start is None and settled:
                    push_start_time = t
                    mujoco.mj_forward(model, data)
                    push_object_position = data.xpos[object_id].copy()
                    push_object_quaternion = data.qpos[qadr + 3 : qadr + 7].copy()
                    world_vertices = []
                    for geom_id in object_geoms:
                        mesh_id = model.geom_dataid[geom_id]
                        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                        vertices = model.mesh_vert[start : start + count]
                        world_vertices.append(
                            vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
                            + data.geom_xpos[geom_id]
                        )
                    world_vertices = np.concatenate(world_vertices)
                    minimum, maximum = world_vertices.min(axis=0), world_vertices.max(axis=0)
                    pusher_start = [
                        float(minimum[0] - physics["push_radius_m"] - physics["push_start_gap_m"]),
                        float((minimum[1] + maximum[1]) / 2),
                        float(max((minimum[2] + maximum[2]) / 2, physics["push_radius_m"] + 0.001)),
                    ]
                    model.body_pos[pusher_body_id] = pusher_start
                    data.qpos[model.jnt_qposadr[pusher_joint_id]] = 0
                    data.qvel[model.jnt_dofadr[pusher_joint_id]] = 0
                    mujoco.mj_forward(model, data)
                progress = (
                    np.clip((t - push_start_time) / physics["push_duration_s"], 0, 1)
                    if push_start_time is not None
                    else 0.0
                )
                data.ctrl[0] = physics["push_distance_m"] * progress
            begin = time.perf_counter()
            mujoco.mj_step(model, data)
            step_times.append(time.perf_counter() - begin)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Nonfinite simulation state at step {step}")
            ground_contact = False
            pusher_contact = False
            for contact in data.contact[: data.ncon]:
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & object_geoms and ground_id in pair:
                    ground_contact = True
                    min_contact_dist = (
                        float(contact.dist)
                        if min_contact_dist is None
                        else min(min_contact_dist, float(contact.dist))
                    )
                if pusher_id >= 0 and pair & object_geoms and pusher_id in pair:
                    pusher_contact = True
            ground_steps += int(ground_contact)
            pusher_contact_steps += int(pusher_contact)
            if ground_contact and first_ground is None:
                first_ground = float(data.time)
            if task == "push" and push_start_time is None:
                settle_history.append(
                    (
                        ground_contact,
                        float(np.linalg.norm(data.qvel[dadr : dadr + 3])),
                        float(np.linalg.norm(data.qvel[dadr + 3 : dadr + 6])),
                        float(data.qpos[qadr + 2]),
                    )
                )
            if step % int(physics["sample_interval_steps"]) == 0 or step == steps - 1:
                position = data.qpos[qadr : qadr + 3].copy()
                quat = data.qpos[qadr + 3 : qadr + 7].copy()
                rotation_matrix = np.empty(9)
                mujoco.mju_quat2Mat(rotation_matrix, quat)
                com = position + rotation_matrix.reshape(3, 3) @ model.body_ipos[object_id]
                angle = float(2 * np.arccos(np.clip(abs(float(np.dot(initial_quat, quat))), 0, 1)))
                samples.append(
                    {
                        "time_s": float(data.time),
                        "phase": (
                            "settling"
                            if push_start_time is None
                            else "pushing"
                            if t < push_start_time + physics["push_duration_s"]
                            else "holding"
                        )
                        if task == "push"
                        else task,
                        "x_m": float(position[0]),
                        "y_m": float(position[1]),
                        "z_m": float(position[2]),
                        "com_x_m": float(com[0]),
                        "com_y_m": float(com[1]),
                        "com_z_m": float(com[2]),
                        "quat_w": float(quat[0]),
                        "quat_x": float(quat[1]),
                        "quat_y": float(quat[2]),
                        "quat_z": float(quat[3]),
                        "linear_velocity_x_m_s": float(data.qvel[dadr]),
                        "linear_velocity_y_m_s": float(data.qvel[dadr + 1]),
                        "linear_velocity_z_m_s": float(data.qvel[dadr + 2]),
                        "angular_velocity_x_rad_s": float(data.qvel[dadr + 3]),
                        "angular_velocity_y_rad_s": float(data.qvel[dadr + 4]),
                        "angular_velocity_z_rad_s": float(data.qvel[dadr + 5]),
                        "linear_speed_m_s": float(np.linalg.norm(data.qvel[dadr : dadr + 3])),
                        "angular_speed_rad_s": float(
                            np.linalg.norm(data.qvel[dadr + 3 : dadr + 6])
                        ),
                        "rotation_from_start_rad": angle,
                        "ground_contact": ground_contact,
                        "pusher_contact": pusher_contact,
                        "pusher_x_m": float(
                            model.body_pos[pusher_body_id][0]
                            + data.qpos[model.jnt_qposadr[pusher_joint_id]]
                        )
                        if task == "push"
                        else None,
                        "pusher_y_m": float(model.body_pos[pusher_body_id][1])
                        if task == "push"
                        else None,
                        "pusher_z_m": float(model.body_pos[pusher_body_id][2])
                        if task == "push"
                        else None,
                        "pusher_force_n": float(data.actuator_force[0]) if task == "push" else None,
                    }
                )
            if renderer is not None and writer is not None and data.time + dt / 2 >= next_frame:
                render_started = time.perf_counter()
                try:
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    phase = (
                        (
                            "SETTLING"
                            if push_start_time is None
                            else "PUSHING"
                            if t < push_start_time + physics["push_duration_s"]
                            else "HOLDING"
                        )
                        if task == "push"
                        else task.upper()
                    )
                    ImageDraw.Draw(frame).text(
                        (10, 10),
                        f"{phase}  {data.time:.2f}s",
                        fill="white",
                        stroke_width=2,
                        stroke_fill="black",
                    )
                    writer.append_data(np.asarray(frame))
                except Exception as error:  # noqa: BLE001
                    video_error = f"{type(error).__name__}: {error}"
                    writer.close()
                    writer = None
                video_seconds += time.perf_counter() - render_started
                next_frame += 1 / float(video["fps"])
        if settle_timed_out and renderer is not None and writer is not None:
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            ImageDraw.Draw(frame).text(
                (10, 10), "NOT SETTLED - NO PUSH", fill="white", stroke_width=2, stroke_fill="black"
            )
            writer.append_data(np.asarray(frame))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    wall_seconds = time.perf_counter() - started
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number
    }
    final = samples[-1]
    tail = [s for s in samples if s["time_s"] >= data.time - physics["stability_window_s"]]
    height_range = max(s["z_m"] for s in tail) - min(s["z_m"] for s in tail)
    stable = (
        final["ground_contact"]
        and final["linear_speed_m_s"] < physics["stable_linear_speed_m_s"]
        and final["angular_speed_rad_s"] < physics["stable_angular_speed_rad_s"]
        and height_range < physics["stable_height_range_m"]
    )
    missing_push_contact = task == "push" and pusher_contact_steps == 0
    push_dx = (
        float(final["x_m"] - push_object_position[0]) if push_object_position is not None else None
    )
    push_angle = (
        float(
            2
            * np.arccos(
                np.clip(
                    abs(float(np.dot(push_object_quaternion, data.qpos[qadr + 3 : qadr + 7]))), 0, 1
                )
            )
        )
        if push_object_quaternion is not None
        else None
    )
    return {
        "task": task,
        "valid": not warnings and not missing_push_contact and not settle_timed_out,
        "warnings": warnings,
        "invalid_reason": "settle_timeout"
        if settle_timed_out
        else "pusher_no_contact"
        if missing_push_contact
        else "solver_warning"
        if warnings
        else None,
        "pusher_start_m": pusher_start,
        "push_start_time_s": push_start_time,
        "settle_window_s": settle_window if task == "push" else None,
        "settle_timeout_s": settle_timeout if task == "push" else None,
        "push_object_start_m": push_object_position.tolist()
        if push_object_position is not None
        else None,
        "push_displacement_x_m": push_dx,
        "push_rotation_rad": push_angle,
        "steps": len(step_times),
        "simulated_seconds": float(data.time),
        "step_seconds": float(sum(step_times)),
        "step_mean_us": float(np.mean(step_times) * 1e6),
        "wall_seconds": wall_seconds,
        "render_encode_seconds": video_seconds,
        "video_status": "not_requested"
        if video_path is None
        else "failed"
        if video_error
        else "completed",
        "video_error": video_error,
        "first_ground_contact_s": first_ground,
        "ground_contact_steps": ground_steps,
        "pusher_contact_steps": pusher_contact_steps,
        "min_ground_contact_distance_m": min_contact_dist,
        "displacement_x_m": float(final["x_m"] - initial_pos[0]),
        "displacement_xy_m": float(
            np.linalg.norm(np.array([final["x_m"], final["y_m"]]) - initial_pos[:2])
        ),
        "rotation_from_start_rad": final["rotation_from_start_rad"],
        "final_height_m": final["z_m"],
        "final_linear_speed_m_s": final["linear_speed_m_s"],
        "final_angular_speed_rad_s": final["angular_speed_rad_s"],
        "tail_height_range_m": float(height_range),
        "stable": bool(stable),
        "samples": samples,
    }


def _save_samples(path: Path, samples: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def run_dynamics(config_path: Path, output: Path) -> dict:
    """Run fixed-collider dynamic trials and record all successes and failures."""
    config = yaml.safe_load(Path(config_path).read_text())
    output = Path(output).resolve()
    if config["physics"]["timing_repeats"] < 1:
        raise ValueError("timing_repeats must be positive")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "config": config,
        "config_path": str(Path(config_path).resolve()),
        "environment": {
            "cpu": _cpu(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "trimesh": __import__("trimesh").__version__,
            "render_backend": os.environ.get("MUJOCO_GL"),
            "headless": True,
        },
        "physics_interpretation": "Estimated properties and material priors; no real-world values inferred",
        "timing_interpretation": "Repeated mj_step CPU wall time excludes rendering; not controlled hardware benchmarking",
        "assets": {},
        "trials": [],
    }
    tasks = ("drop", "placement", "push")
    for asset_name, asset in config["assets"].items():
        batch = (PROJECT_ROOT / asset["batch"]).resolve()
        sample_id, condition = asset["sample_id"], asset["condition"]
        prepared, colliders = (
            batch / "prepared" / sample_id,
            batch / "collisions" / sample_id / condition,
        )
        try:
            ledger = json.loads((batch / "batch.json").read_text())
            rows = [
                r
                for r in ledger["results"]
                if r["sample_id"] == sample_id and r["condition"] == condition
            ]
            if (
                len(rows) != 1
                or rows[0]["collision_status"] != "ok"
                or rows[0]["mujoco_status"] != "ok"
            ):
                raise ValueError("Batch does not contain one successfully loaded collision asset")
            if ledger["model"] != "Reference" and rows[0].get("index_status") != "ready":
                raise ValueError("Generated asset did not pass physics-index provenance validation")
            collider_info = json.loads((colliders / "colliders.json").read_text())
            if (
                collider_info["status"] != "completed"
                or Path(collider_info["input"]).resolve() != (prepared / "normalized.glb").resolve()
            ):
                raise ValueError("Collision input provenance does not match prepared mesh")
            mesh, _ = load_geometry(prepared / "normalized.glb")
            result["assets"][asset_name] = {
                "status": "ready",
                "batch": str(batch),
                "model": ledger["model"],
                "sample_id": sample_id,
                "condition": condition,
                "mesh": str(prepared / "normalized.glb"),
                "colliders": str(colliders / "colliders.json"),
                "components": collider_info["component_count"],
                "profiles": {},
            }
        except Exception as error:  # noqa: BLE001
            result["assets"][asset_name] = {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            for profile in config["profiles"]:
                for task in tasks:
                    result["trials"].append(
                        {
                            "asset": asset_name,
                            "profile": profile,
                            "task": task,
                            "status": "failed",
                            "failure_stage": "asset_load",
                            "error": str(error),
                        }
                    )
            continue
        for profile_name, profile in config["profiles"].items():
            try:
                properties = estimate_properties(mesh, profile)
                result["assets"][asset_name]["profiles"][profile_name] = {
                    "status": "ready",
                    **properties,
                }
            except Exception as error:  # noqa: BLE001
                result["assets"][asset_name]["profiles"][profile_name] = {
                    "status": "failed",
                    "error": str(error),
                }
                for task in tasks:
                    result["trials"].append(
                        {
                            "asset": asset_name,
                            "profile": profile_name,
                            "task": task,
                            "status": "failed",
                            "failure_stage": "property_estimation",
                            "error": str(error),
                        }
                    )
                continue
            for task in tasks:
                row = {
                    "asset": asset_name,
                    "profile": profile_name,
                    "task": task,
                    "status": "running",
                }
                try:
                    directory = output / asset_name / profile_name / task
                    model, data, validation = make_dynamic_scene(
                        prepared, colliders, properties, config, task, directory
                    )
                    video_path = (
                        directory / "simulation.mp4" if config["video"]["enabled"] else None
                    )
                    primary = simulate(model, data, config, task, video_path)
                    _save_samples(directory / "trajectory.csv", primary.pop("samples"))
                    timing = []
                    for _ in range(config["physics"]["timing_repeats"]):
                        repeated = simulate(model, mujoco.MjData(model), config, task)
                        timing.append(
                            {
                                "step_seconds": repeated["step_seconds"],
                                "steps": repeated.get("steps"),
                                "valid": repeated["valid"],
                                "warnings": repeated["warnings"],
                                "invalid_reason": repeated["invalid_reason"],
                            }
                        )
                    primary["timing_repeats"] = timing
                    valid_timing = [v["step_seconds"] for v in timing if v["valid"]]
                    primary["timing_valid"] = len(valid_timing)
                    primary["timing_planned"] = config["physics"]["timing_repeats"]
                    primary["timing_mean_step_seconds"] = (
                        float(np.mean(valid_timing)) if valid_timing else None
                    )
                    per_step = [
                        v["step_seconds"] / v["steps"] * 1e6
                        for v in timing
                        if v["valid"] and v["steps"]
                    ]
                    primary["timing_mean_per_step_us"] = (
                        float(np.mean(per_step)) if per_step else None
                    )
                    row.update(
                        primary,
                        status="ok"
                        if primary["valid"]
                        and primary["video_status"] != "failed"
                        and len(valid_timing) == primary["timing_planned"]
                        else "failed",
                        scene_validation=validation,
                        video=str(video_path) if video_path else None,
                        trajectory=str(directory / "trajectory.csv"),
                    )
                    if row["status"] == "failed":
                        row["failure_stage"] = "simulation_or_video"
                        timing_error = next(
                            (entry["invalid_reason"] for entry in timing if not entry["valid"]),
                            None,
                        )
                        row["error"] = (
                            primary["video_error"]
                            or primary["invalid_reason"]
                            or (str(primary["warnings"]) if primary["warnings"] else None)
                            or timing_error
                            or "invalid timing repeat"
                        )
                except Exception as error:  # noqa: BLE001
                    row.update(
                        status="failed",
                        failure_stage="simulation",
                        error=f"{type(error).__name__}: {error}",
                    )
                result["trials"].append(row)
                (output / "run.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
    result["status"] = (
        "completed_with_failures"
        if any(r["status"] != "ok" for r in result["trials"])
        else "completed"
    )
    result["planned_trials"] = len(config["assets"]) * len(config["profiles"]) * len(tasks)
    result["successful_trials"] = sum(r["status"] == "ok" for r in result["trials"])
    (output / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = [
        "asset",
        "profile",
        "task",
        "status",
        "failure_stage",
        "error",
        "valid",
        "first_ground_contact_s",
        "stable",
        "final_linear_speed_m_s",
        "final_angular_speed_rad_s",
        "tail_height_range_m",
        "displacement_x_m",
        "displacement_xy_m",
        "rotation_from_start_rad",
        "push_displacement_x_m",
        "push_rotation_rad",
        "pusher_contact_steps",
        "min_ground_contact_distance_m",
        "step_mean_us",
        "timing_valid",
        "timing_planned",
        "timing_mean_step_seconds",
        "timing_mean_per_step_us",
        "video_status",
        "video",
        "trajectory",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["trials"])
    return result
