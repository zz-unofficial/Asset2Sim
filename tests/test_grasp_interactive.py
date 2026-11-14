"""Interactive control uses bounded actuator targets and actual MuJoCo contact."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from asset2sim.collision import build_collision
from asset2sim.simulation.grasp.interactive import KeyboardController, _contacts
from asset2sim.simulation.grasp.scene import create_scene


@pytest.fixture
def interactive_scene(tmp_path):
    menagerie = Path(".local/mujoco_menagerie")
    if not (menagerie / "universal_robots_ur5e/ur5e.xml").exists():
        pytest.skip("Menagerie assets unavailable")
    grasp = yaml.safe_load(Path("configs/physics/grasp.yaml").read_text())
    config = yaml.safe_load(Path("configs/physics/grasp_interactive.yaml").read_text())
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "normalized.glb").write_bytes(
        Path("data/samples/sf3d/reference/debug_box.glb").read_bytes()
    )
    colliders = tmp_path / "colliders"
    build_collision(prepared / "normalized.glb", colliders, "C0", assume_prepared=True)
    model, _ = create_scene(menagerie, prepared, colliders, grasp, tmp_path / "scene")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, config


def test_no_direct_joint_edits_and_bounded_commands(interactive_scene):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    original = data.qpos.copy()
    command = controller.commands.copy()
    controller.step({"w"}, 0.002)
    assert np.array_equal(data.qpos, original)
    assert (
        np.max(np.abs(controller.commands - command)) <= config["joint_speed_rad_s"] * 0.002 + 1e-12
    )
    assert np.all(controller.commands >= model.jnt_range[:6, 0])
    assert np.all(controller.commands <= model.jnt_range[:6, 1])


def test_unreachable_target_does_not_change_command(interactive_scene, monkeypatch):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    controller.homing = False
    before = controller.target.copy()

    def fail(*args):
        raise ValueError("unreachable")

    monkeypatch.setattr("asset2sim.simulation.grasp.interactive.inverse_kinematics", fail)
    controller.step({"w"}, config["ik_period_s"])
    assert np.array_equal(controller.target, before)
    assert controller.last_error == "unreachable"


def test_reference_keyboard_sequence_grasps_and_releases(interactive_scene):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    qadr = model.jnt_qposadr[joint]
    phase, start = "home", 0.0
    peak_height = 0.0
    bilateral = False
    moved_x = 0.0
    for step in range(15000):
        t = step * model.opt.timestep
        if phase == "home" and not controller.homing:
            phase = "down"
        if phase == "down" and controller.target[2] <= 0.052:
            phase, start = "downwait", t
        if phase == "downwait" and t - start > 2:
            phase = "close"
        if phase == "close" and controller.gripper >= 255:
            phase, start = "closewait", t
        if phase == "closewait" and t - start > 2:
            phase = "lift"
        if phase == "lift" and controller.target[2] >= 0.17:
            phase, start = "liftwait", t
        if phase == "liftwait" and t - start > 2:
            phase = "move"
        if phase == "move" and controller.target[0] >= 0.48:
            phase, start = "movewait", t
        if phase == "movewait" and t - start > 2:
            moved_x = data.qpos[qadr]
            phase = "open"
        if phase == "open" and controller.gripper <= 0:
            phase, start = "openwait", t
        if phase == "openwait" and t - start > 2:
            break
        keys = {"down": {"f"}, "close": {"j"}, "lift": {"r"}, "move": {"w"}, "open": {"k"}}.get(
            phase, set()
        )
        controller.step(keys, model.opt.timestep)
        mujoco.mj_step(model, data)
        _, table, forces = _contacts(model, data)
        peak_height = max(peak_height, float(data.qpos[qadr + 2]))
        bilateral |= not table and min(forces.values()) > 0.05 and data.qpos[qadr + 2] > 0.1
    assert phase == "openwait"
    assert bilateral and peak_height > 0.1 and moved_x > 0.47
    assert _contacts(model, data)[1]
    assert not any(w.number for w in data.warning)


def test_workspace_and_released_keys_hold_target(interactive_scene):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    controller.homing = False
    controller.target[0] = config["workspace_max_m"][0]
    before = controller.target.copy()
    for _ in range(30):
        controller.step({"w"}, 0.002)
    assert np.array_equal(controller.target, before)
    for _ in range(30):
        controller.step(set(), 0.002)
    assert np.array_equal(controller.target, before)


def test_homing_does_not_accumulate_input_time(interactive_scene):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    controller.ik_elapsed = 5.0
    # Simulate an already settled actuator by using its current physical position as the target.
    controller.ik_target = data.qpos[:6].copy()
    before = controller.target.copy()
    controller.step({"w"}, 0.002)
    assert not controller.homing
    assert np.array_equal(controller.target, before)


def test_rotation_tracks_target_via_actuators(interactive_scene):
    from scipy.spatial.transform import Rotation

    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_pinch")
    dt = model.opt.timestep
    baseline = np.diag([1.0, -1.0, -1.0])
    for step in range(4500):
        t = step * dt
        keys = {"h"} if not controller.homing and t < 6.5 else set()
        controller.step(keys, dt)
        mujoco.mj_step(model, data)
    relative = data.site_xmat[site].reshape(3, 3) @ baseline.T
    observed = Rotation.from_matrix(relative).as_euler("xyz")
    assert 0.15 < controller.angles[2] <= config["rotation_limit_rad"]
    assert abs(observed[2] - controller.angles[2]) < 0.08
    assert abs(observed[0]) < 0.08 and abs(observed[1]) < 0.08
    assert not any(w.number for w in data.warning)


def test_rejected_rotation_keeps_last_pose(interactive_scene, monkeypatch):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    controller.homing = False
    original_position = controller.target.copy()
    original_angles = controller.angles.copy()
    original_qpos = data.qpos.copy()

    def fail(*args):
        raise ValueError("unreachable rotation")

    monkeypatch.setattr("asset2sim.simulation.grasp.interactive.inverse_kinematics", fail)
    controller.step({"w", "h"}, config["ik_period_s"])
    assert np.array_equal(controller.angles, original_angles)
    assert np.array_equal(controller.target, original_position)
    assert np.array_equal(data.qpos, original_qpos)
    assert controller.last_error == "unreachable rotation"


@pytest.mark.parametrize(
    ("key", "axis", "direction"),
    [
        ("u", 0, 1),
        ("o", 0, -1),
        ("i", 1, 1),
        ("p", 1, -1),
        ("h", 2, 1),
        ("l", 2, -1),
    ],
)
def test_rotation_keys_and_angle_limits(interactive_scene, monkeypatch, key, axis, direction):
    model, data, config = interactive_scene
    controller = KeyboardController(model, data, config, [0.45, 0, 0.251])
    controller.homing = False
    monkeypatch.setattr(
        "asset2sim.simulation.grasp.interactive.inverse_kinematics",
        lambda model, position, initial, orientation: initial,
    )
    for _ in range(100):
        controller.step({key}, config["ik_period_s"])
    assert controller.angles[axis] == pytest.approx(direction * config["rotation_limit_rad"])
    assert all(controller.angles[j] == 0 for j in range(3) if j != axis)
    before = controller.angles.copy()
    controller.step(set(), config["ik_period_s"])
    assert np.array_equal(controller.angles, before)
