"""Fixed scripted UR5e joint trajectories and contact-based grasp scoring."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def score_grasp(trace, initial_height, config, warnings):
    """Score a force-bearing bilateral hold followed by actual release."""
    hold = [row for row in trace if row["phase"] == "hold"]
    good = [
        row
        for row in hold
        if row["object_z_m"] - initial_height >= config["success_height_m"]
        and row["left_contact"]
        and row["right_contact"]
        and row["left_normal_force_n"] >= config["minimum_normal_force_n"]
        and row["right_normal_force_n"] >= config["minimum_normal_force_n"]
        and not row["table_contact"]
    ]
    last = trace[-1]
    fraction = len(good) / len(hold) if hold else None
    hold_passed = bool(hold and fraction >= config["minimum_hold_fraction"])
    release = bool(
        hold_passed
        and last["phase"] == "release"
        and last["table_contact"]
        and not last["left_contact"]
        and not last["right_contact"]
    )
    success = bool(hold_passed and release and not warnings)
    return {
        "success": success,
        "valid": not bool(warnings),
        "warnings": warnings,
        "hold_valid_samples": len(hold),
        "hold_success_samples": len(good),
        "hold_success_fraction": fraction,
        "release_success": release,
    }


def inverse_kinematics(model, target, initial, orientation=None):
    """Solve a pinch position and orientation without object geometry fitting."""
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_pinch")
    if site < 0:
        raise ValueError("Gripper pinch site missing")
    desired = np.diag([1.0, -1.0, -1.0]) if orientation is None else np.asarray(orientation)

    def residual(q):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        return np.r_[
            3 * (data.site_xpos[site] - target),
            Rotation.from_matrix(desired @ rotation.T).as_rotvec(),
        ]

    result = least_squares(
        residual,
        initial,
        bounds=([-6.2, -6.2, -3.1415, -6.2, -6.2, -6.2], [6.2, 6.2, 3.1415, 6.2, 6.2, 6.2]),
        max_nfev=150,
    )
    if np.linalg.norm(result.fun) > 1e-3:
        raise ValueError(f"Fixed grasp pose unreachable: residual={np.linalg.norm(result.fun):.6f}")
    return result.x


def scripted_trial(model, config, output: Path, video: bool = True):
    """Execute approach/close/lift/hold/release with free object and actual contacts."""
    output.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(model)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if (
        obj < 0
        or joint < 0
        or model.body_parentid[obj] != 0
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError("Object must be a free body")
    qadr = model.jnt_qposadr[joint]
    p0 = data.qpos[qadr : qadr + 3].copy()
    x, y = config["object_xy_m"]
    initial = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    pre = inverse_kinematics(model, np.array([x, y, p0[2] + 0.20]), initial)
    grasp = inverse_kinematics(model, np.array([x, y, p0[2]]), pre)
    lift = inverse_kinematics(model, np.array([x, y, p0[2] + config["lift_height_m"]]), grasp)
    data.qpos[:6] = pre
    data.ctrl[:6] = pre
    data.ctrl[6] = 0
    mujoco.mj_forward(model, data)
    # Equality constraints in the gripper mechanism are allowed; the object must not be attached.
    for k in range(model.neq):
        if model.eq_type[k] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT) and (
            model.eq_obj1id[k] == obj or model.eq_obj2id[k] == obj
        ):
            raise ValueError("Object attachment is forbidden")
    obj_geoms = {
        i for i in range(model.ngeom) if model.geom_bodyid[i] == obj and model.geom_contype[i]
    }
    pads = {
        side: {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"grip_{side}_pad{i}")
            for i in [1, 2]
        }
        for side in ["left", "right"]
    }
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.35, 0, 0.18]
    camera.distance = 1.15
    camera.azimuth = 125
    camera.elevation = -25
    renderer = writer = None
    if video:
        renderer = mujoco.Renderer(
            model, height=config["video"]["height"], width=config["video"]["width"]
        )
        writer = imageio.get_writer(
            str(output / "grasp.mp4"),
            fps=config["video"]["fps"],
            codec="libx264",
            macro_block_size=1,
        )
    dt = model.opt.timestep
    duration = 9.0
    steps = round(duration / dt)
    nextframe = 0.0
    trace = []
    steptime = 0.0

    def interp(a, b, u):
        u = np.clip(u, 0, 1)
        u = u * u * (3 - 2 * u)
        return a * (1 - u) + b * u

    try:
        for step in range(steps):
            t = step * dt
            if t < 1:
                phase = "settle"
                q = pre
                grip = 0
            elif t < 3:
                phase = "approach"
                q = interp(pre, grasp, (t - 1) / 2)
                grip = 0
            elif t < 5:
                phase = "close"
                q = grasp
                grip = 255 * min((t - 3) / 1.2, 1)
            elif t < 7:
                phase = "lift"
                q = interp(grasp, lift, (t - 5) / 2)
                grip = 255
            elif t < 8:
                phase = "hold"
                q = lift
                grip = 255
            else:
                phase = "release"
                q = lift
                grip = 0
            data.ctrl[:6] = q
            data.ctrl[6] = grip
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            steptime += time.perf_counter() - start
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite simulation")
            touching = {side: False for side in pads}
            forces = {side: 0.0 for side in pads}
            table_contact = False
            for contact_index, contact in enumerate(data.contact[: data.ncon]):
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & obj_geoms:
                    force = np.zeros(6)
                    if contact.efc_address >= 0:
                        mujoco.mj_contactForce(model, data, contact_index, force)
                    for side, side_pads in pads.items():
                        if pair & side_pads:
                            touching[side] = True
                            forces[side] += max(float(force[0]), 0.0)
                    table_contact |= table in pair
            if step % 10 == 0 or step == steps - 1:
                trace.append(
                    {
                        "time_s": float(data.time),
                        "phase": phase,
                        "object_x_m": float(data.qpos[qadr]),
                        "object_y_m": float(data.qpos[qadr + 1]),
                        "object_z_m": float(data.qpos[qadr + 2]),
                        "object_quat_w": float(data.qpos[qadr + 3]),
                        "object_quat_x": float(data.qpos[qadr + 4]),
                        "object_quat_y": float(data.qpos[qadr + 5]),
                        "object_quat_z": float(data.qpos[qadr + 6]),
                        "left_contact": touching["left"],
                        "right_contact": touching["right"],
                        "left_normal_force_n": forces["left"],
                        "right_normal_force_n": forces["right"],
                        "table_contact": table_contact,
                        "gripper_control": float(grip),
                    }
                )
            if writer and data.time + dt / 2 >= nextframe:
                renderer.update_scene(data, camera=camera)
                im = Image.fromarray(renderer.render())
                ImageDraw.Draw(im).text(
                    (10, 10),
                    f"{phase.upper()}  {data.time:.2f}s",
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
                writer.append_data(np.asarray(im))
                nextframe += 1 / config["video"]["fps"]
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    with (output / "trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0]))
        w.writeheader()
        w.writerows(trace)
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number
    }
    approach = next(row for row in trace if row["phase"] == "approach")
    return {
        **score_grasp(trace, float(p0[2]), config, warnings),
        "settle_xy_displacement_m": float(
            np.linalg.norm(np.array([approach["object_x_m"], approach["object_y_m"]]) - p0[:2])
        ),
        "max_lift_m": max(r["object_z_m"] - p0[2] for r in trace),
        "end_height_m": trace[-1]["object_z_m"] - p0[2],
        "bilateral_contact_samples": sum(r["left_contact"] and r["right_contact"] for r in trace),
        "mj_step_seconds": steptime,
        "steps": steps,
        "pre_joint_target": pre.tolist(),
        "grasp_joint_target": grasp.tolist(),
        "lift_joint_target": lift.tolist(),
    }
