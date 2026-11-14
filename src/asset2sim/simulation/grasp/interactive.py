"""Real-time, actuator-driven keyboard teleoperation of the existing grasp scene."""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from asset2sim.paths import PROJECT_ROOT
from asset2sim.simulation.grasp.controller import inverse_kinematics
from asset2sim.simulation.grasp.keyboard_x11 import ViewerKeys
from asset2sim.simulation.grasp.run import resolve_asset
from asset2sim.simulation.grasp.scene import create_scene


class KeyboardController:
    """Rate-limited Cartesian IK targets and gripper commands, without editing state."""

    def __init__(self, model, data, config, home):
        self.model, self.data, self.config = model, data, config
        self.target = np.array(home, dtype=float)
        self.angles = np.zeros(3)
        self.joint_limits = model.jnt_range[:6].copy()
        seed = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.ik_target = inverse_kinematics(model, self.target, seed)
        self.commands = data.qpos[:6].copy()
        self.gripper = 0.0
        self.ik_elapsed = 0.0
        self.homing = True
        self.last_error = ""

    def step(self, keys: set[str], dt: float):
        self.ik_elapsed += dt
        if self.homing and np.max(np.abs(self.data.qpos[:6] - self.ik_target)) < 0.08:
            self.homing = False
            self.ik_elapsed = 0.0
        if not self.homing:
            direction = np.array(
                [
                    ("w" in keys) - ("s" in keys),
                    ("a" in keys) - ("d" in keys),
                    ("r" in keys) - ("f" in keys),
                ],
                dtype=float,
            )
            norm = np.linalg.norm(direction)
            if norm:
                direction /= norm
            angular_direction = np.array(
                [
                    ("u" in keys) - ("o" in keys),
                    ("i" in keys) - ("p" in keys),
                    ("h" in keys) - ("l" in keys),
                ],
                dtype=float,
            )
            angular_norm = np.linalg.norm(angular_direction)
            if angular_norm:
                angular_direction /= angular_norm
            if self.ik_elapsed >= self.config["ik_period_s"]:
                candidate = np.clip(
                    self.target + direction * self.config["cartesian_speed_m_s"] * self.ik_elapsed,
                    self.config["workspace_min_m"],
                    self.config["workspace_max_m"],
                )
                angles = np.clip(
                    self.angles
                    + angular_direction * self.config["angular_speed_rad_s"] * self.ik_elapsed,
                    -self.config["rotation_limit_rad"],
                    self.config["rotation_limit_rad"],
                )
                if not np.array_equal(candidate, self.target) or not np.array_equal(
                    angles, self.angles
                ):
                    try:
                        orientation = Rotation.from_euler("xyz", angles).as_matrix() @ np.diag(
                            [1.0, -1.0, -1.0]
                        )
                        solution = inverse_kinematics(
                            self.model, candidate, self.ik_target, orientation
                        )
                        if np.any(solution < self.joint_limits[:, 0]) or np.any(
                            solution > self.joint_limits[:, 1]
                        ):
                            raise ValueError("IK violates joint limits")
                        self.ik_target = solution
                        self.target = candidate
                        self.angles = angles
                        self.last_error = ""
                    except ValueError as exc:
                        self.last_error = str(exc)
                self.ik_elapsed = 0.0
            gripper_direction = ("j" in keys) - ("k" in keys)
            self.gripper = float(
                np.clip(
                    self.gripper + gripper_direction * self.config["gripper_speed_units_s"] * dt,
                    0,
                    255,
                )
            )
        max_change = self.config["joint_speed_rad_s"] * dt
        self.commands += np.clip(self.ik_target - self.commands, -max_change, max_change)
        self.commands[:] = np.clip(self.commands, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.data.ctrl[:6] = self.commands
        self.data.ctrl[6] = self.gripper


def _contacts(model, data):
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    pads = {
        s: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"grip_{s}_pad{i}") for i in (1, 2)}
        for s in ("left", "right")
    }
    objects = {
        i for i in range(model.ngeom) if model.geom_bodyid[i] == obj and model.geom_contype[i]
    }
    forces = {"left": 0.0, "right": 0.0}
    touch_table = False
    count = 0
    for i, contact in enumerate(data.contact[: data.ncon]):
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair & objects:
            continue
        count += 1
        touch_table |= table in pair
        force = np.zeros(6)
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, force)
        for side, indices in pads.items():
            if pair & indices:
                forces[side] += max(float(force[0]), 0.0)
    return count, touch_table, forces


def _segment(
    name, index, model, data, viewer, renderer, scene_config, config, output, keys, blocked_key=None
):
    """Run a viewer segment; return a requested reset/switch or None on close."""
    directory = output / f"segment_{index:03d}_{name}"
    directory.mkdir()
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if obj < 0 or joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Interactive object must be a free body")
    for i in range(model.neq):
        if model.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT) and obj in (
            model.eq_obj1id[i],
            model.eq_obj2id[i],
        ):
            raise ValueError("Object attachment is forbidden")
    qadr = model.jnt_qposadr[joint]
    home = [*scene_config["object_xy_m"], float(data.qpos[qadr + 2] + 0.20)]
    controller = KeyboardController(model, data, config, home)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_pinch")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.35, 0, 0.18]
    camera.distance, camera.azimuth, camera.elevation = 1.15, 125, -25
    video = config["video"]
    writer = imageio.get_writer(
        str(directory / "interaction.mp4"), fps=video["fps"], codec="libx264", macro_block_size=1
    )
    trace = (directory / "trajectory.csv").open("w", newline="")
    columns = (
        [
            "time_s",
            "keys",
            "homing",
            "ik_error",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "target_rot_x_rad",
            "target_rot_y_rad",
            "target_rot_z_rad",
            "pinch_x_m",
            "pinch_y_m",
            "pinch_z_m",
            "pinch_qw",
            "pinch_qx",
            "pinch_qy",
            "pinch_qz",
            "object_x_m",
            "object_y_m",
            "object_z_m",
            "object_qw",
            "object_qx",
            "object_qy",
            "object_qz",
            "gripper_control",
            "object_contact_count",
            "table_contact",
            "left_normal_force_n",
            "right_normal_force_n",
        ]
        + [f"joint_{i}_rad" for i in range(6)]
        + [f"command_{i}_rad" for i in range(6)]
    )
    csv_writer = csv.writer(trace)
    csv_writer.writerow(columns)
    next_sample = next_frame = 0.0
    previous = {blocked_key} if blocked_key else set()
    pressed = set()
    requested = None
    steps = 0
    last_reported_error = ""
    ready_reported = False
    try:
        with nullcontext(viewer):
            viewer.sync()
            print(
                f"Viewer: {name}; moving to home pose with actuators. Focus window for controls.",
                flush=True,
            )
            while viewer.is_running():
                started = time.perf_counter()
                edge = set()
                if steps % 10 == 0:
                    state = keys.read()
                    if state is None:
                        pressed = set()
                    else:
                        pressed = state
                        edge = pressed - previous
                        previous = pressed
                if edge & {"0", "1", "2", "3", "q"}:
                    requested = next(k for k in ("q", "0", "1", "2", "3") if k in edge)
                    break
                with viewer.lock():
                    controller.step(pressed, model.opt.timestep)
                    if not controller.homing and not ready_reported:
                        print(
                            "Ready: W/S X, A/D Y, R/F Z; U/O X, I/P Y, H/L Z rotation; J/K grip; 0 reset; 1/2/3 switch; Q quit.",
                            flush=True,
                        )
                        ready_reported = True
                    mujoco.mj_step(model, data)
                    steps += 1
                    if controller.last_error != last_reported_error:
                        if controller.last_error:
                            print(f"Target rejected: {controller.last_error}", flush=True)
                        last_reported_error = controller.last_error
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError("Simulation state became non-finite")
                    if data.time >= next_sample:
                        count, table, forces = _contacts(model, data)
                        pinch_quat = np.empty(4)
                        mujoco.mju_mat2Quat(pinch_quat, data.site_xmat[site])
                        csv_writer.writerow(
                            [
                                float(data.time),
                                "".join(sorted(pressed)),
                                controller.homing,
                                controller.last_error,
                                *controller.target,
                                *controller.angles,
                                *data.site_xpos[site],
                                *pinch_quat,
                                *data.qpos[qadr : qadr + 7],
                                controller.gripper,
                                count,
                                table,
                                forces["left"],
                                forces["right"],
                                *data.qpos[:6],
                                *controller.commands,
                            ]
                        )
                        trace.flush()
                        next_sample += 1 / config["record_rate_hz"]
                    if data.time >= next_frame:
                        renderer.update_scene(data, camera=camera)
                        frame = Image.fromarray(renderer.render())
                        ImageDraw.Draw(frame).text(
                            (8, 8),
                            f"{name}  {data.time:.1f}s  {'HOMING' if controller.homing else 'CONTROL'}",
                            fill="white",
                            stroke_width=2,
                            stroke_fill="black",
                        )
                        writer.append_data(np.asarray(frame))
                        next_frame += 1 / video["fps"]
                if steps % 10 == 0:
                    viewer.sync()
                delay = model.opt.timestep - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
    finally:
        trace.close()
        writer.close()
        (directory / "segment.json").write_text(
            json.dumps(
                {
                    "asset": name,
                    "steps": steps,
                    "simulation_seconds": float(data.time),
                    "requested": requested,
                    "warnings": {
                        mujoco.mjtWarning(i).name: int(w.number)
                        for i, w in enumerate(data.warning)
                        if w.number
                    },
                },
                indent=2,
            )
            + "\n"
        )
    return requested


def select_interactive_assets(config: dict, grasp_config: dict, sample_id=None, source=None):
    """Apply a common sample and initial source without changing shared configuration."""
    resolved = copy.deepcopy(grasp_config)
    selected = sample_id if sample_id is not None else config.get("sample_id")
    if selected is not None:
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("sample_id must be a nonempty string")
        for name in ("reference", "sf3d", "spar3d"):
            resolved["assets"][name]["sample_id"] = selected
    initial = source if source is not None else config.get("initial_source", "reference")
    if initial not in ("reference", "sf3d", "spar3d"):
        raise ValueError(f"Unknown initial source: {initial}")
    return resolved, initial


def interactive_grasp(config_path: Path, output: Path, sample_id=None, source=None):
    """Launch focused-keyboard MuJoCo viewer, writing an independent session."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    grasp_config = yaml.safe_load((config_path.parent / config["grasp_config"]).read_text())
    grasp_config, initial_source = select_interactive_assets(
        config, grasp_config, sample_id, source
    )
    config["initial_source"] = initial_source
    if sample_id is not None:
        config["sample_id"] = sample_id
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    session = {"config": config, "grasp_config": grasp_config, "segments": [], "status": "running"}
    (output / "session.json").write_text(json.dumps(session, indent=2) + "\n")
    names = ["reference", "sf3d", "spar3d"]
    models = {}
    renderers = {}
    keys = None
    try:
        keys = ViewerKeys()
        for name in names:
            prepared, colliders, ledger = resolve_asset(grasp_config["assets"][name])
            models[name], info = create_scene(
                (PROJECT_ROOT / grasp_config["menagerie"]).resolve(),
                prepared,
                colliders,
                grasp_config,
                output / "models" / name,
            )
            session.setdefault("assets", {})[name] = {
                "model": ledger["model"],
                **grasp_config["assets"][name],
                **info,
            }
        for name, model in models.items():
            renderers[name] = mujoco.Renderer(
                model, height=config["video"]["height"], width=config["video"]["width"]
            )
        current = initial_source
        request = None
        model = models[current]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.35, 0, 0.18]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.15, 125, -25
            while viewer.is_running():
                index = len(session["segments"])
                request = _segment(
                    current,
                    index,
                    model,
                    data,
                    viewer,
                    renderers[current],
                    grasp_config,
                    config,
                    output,
                    keys,
                    request,
                )
                session["segments"].append(
                    {"asset": current, "path": f"segment_{index:03d}_{current}", "request": request}
                )
                (output / "session.json").write_text(json.dumps(session, indent=2) + "\n")
                if request in (None, "q") or not viewer.is_running():
                    break
                if request != "0":
                    current = names[int(request) - 1]
                model = models[current]
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                # MuJoCo 3.3.5's passive Handle has no public model reload API.
                # Its underlying loader replaces model/data in the existing window.
                sim = viewer._get_sim()
                if sim is None:
                    break
                sim.load(model, data, "")
        session["status"] = "completed"
    except BaseException as exc:
        session["status"] = "failed"
        session["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if keys:
            keys.close()
        for renderer in renderers.values():
            renderer.close()
        (output / "session.json").write_text(json.dumps(session, indent=2) + "\n")
    return session
