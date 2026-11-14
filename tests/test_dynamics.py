import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
import yaml

from asset2sim.simulation.batch import run_physics_batch
from asset2sim.simulation.dynamics import make_dynamic_scene, run_dynamics, simulate
from asset2sim.simulation.properties import estimate_properties


def test_uniform_box_mass_com_and_inertia():
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    result = estimate_properties(box, {"method": "uniform_density", "density_kg_m3": 1000})
    assert result["mass_kg"] == pytest.approx(1.0)
    assert np.allclose(result["center_of_mass_m"], [0, 0, 0])
    assert np.allclose(result["inertia_kg_m2"], np.eye(3) / 600, rtol=1e-6)
    fixed = estimate_properties(box, {"method": "mesh_fixed_mass", "mass_kg": 0.1})
    assert fixed["mass_kg"] == pytest.approx(0.1)
    assert np.allclose(np.asarray(fixed["inertia_kg_m2"]), np.eye(3) / 6000, rtol=1e-6)


def test_open_mesh_rejects_volume_estimation_but_baseline_explicit():
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match="ready closed topology"):
        estimate_properties(mesh, {"method": "uniform_density", "density_kg_m3": 600})
    baseline = estimate_properties(mesh, {"method": "aabb", "mass_kg": 0.1})
    assert baseline["method"] == "aabb" and baseline["volume_m3"] is None


def test_dynamic_drop_push_and_configured_inertia(tmp_path):
    batch = tmp_path / "batch"
    run_physics_batch(
        Path("data/manifests/debug.jsonl"), batch, ["C0"], ["debug_box_v0"], reference=True
    )
    prepared, colliders = batch / "prepared/debug_box_v0", batch / "collisions/debug_box_v0/C0"
    config = yaml.safe_load(Path("configs/physics/dynamics.yaml").read_text())
    from asset2sim.geometry.prepare import load_geometry

    mesh, _ = load_geometry(prepared / "normalized.glb")
    props = estimate_properties(mesh, config["profiles"]["material_prior"])
    model, data, _ = make_dynamic_scene(
        prepared, colliders, props, config, "drop", tmp_path / "drop"
    )
    baseline_props = estimate_properties(mesh, config["profiles"]["baseline"])
    baseline_model, _, _ = make_dynamic_scene(
        prepared, colliders, baseline_props, config, "drop", tmp_path / "baseline"
    )
    assert model.nmesh == baseline_model.nmesh
    assert np.array_equal(model.mesh_vert, baseline_model.mesh_vert)
    assert np.allclose(model.geom_pos, baseline_model.geom_pos)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert model.body_mass[object_id] == pytest.approx(props["mass_kg"])
    drop = simulate(model, data, config, "drop")
    assert drop["valid"] and drop["first_ground_contact_s"] == pytest.approx(
        np.sqrt(0.30 / 9.81), abs=0.02
    )
    assert all(key in drop["samples"][0] for key in ("quat_w", "com_z_m", "linear_velocity_z_m_s"))
    push_model, push_data, _ = make_dynamic_scene(
        prepared, colliders, props, config, "push", tmp_path / "push"
    )
    push = simulate(push_model, push_data, config, "push")
    assert push["valid"] and push["pusher_contact_steps"] > 0
    assert push["push_start_time_s"] >= config["physics"]["push_delay_s"]
    assert {row["phase"] for row in push["samples"]} >= {"settling", "pushing", "holding"}
    preceding = [
        row
        for row in push["samples"]
        if push["push_start_time_s"] - 0.18 <= row["time_s"] < push["push_start_time_s"]
    ]
    assert preceding and all(row["ground_contact"] for row in preceding)
    assert all(
        row["linear_speed_m_s"] < config["physics"]["stable_linear_speed_m_s"] for row in preceding
    )
    assert all(
        row["angular_speed_rad_s"] < config["physics"]["stable_angular_speed_rad_s"]
        for row in preceding
    )
    assert push["displacement_x_m"] > 0
    no_push = {**config, "physics": {**config["physics"], "push_distance_m": 0.0}}
    model, data, _ = make_dynamic_scene(
        prepared, colliders, props, no_push, "push", tmp_path / "no_push"
    )
    trial = simulate(model, data, no_push, "push")
    assert not trial["valid"] and trial["invalid_reason"] == "pusher_no_contact"
    never_settles = {
        **config,
        "physics": {
            **config["physics"],
            "stable_linear_speed_m_s": 0.0,
            "push_settle_timeout_s": 0.9,
        },
    }
    model, data, _ = make_dynamic_scene(
        prepared, colliders, props, never_settles, "push", tmp_path / "unsettled"
    )
    trial = simulate(model, data, never_settles, "push")
    assert not trial["valid"] and trial["invalid_reason"] == "settle_timeout"
    assert trial["push_start_time_s"] is None and trial["pusher_contact_steps"] == 0


def test_dynamics_ledger_retains_missing_asset_and_denominators(tmp_path):
    config = yaml.safe_load(Path("configs/physics/dynamics.yaml").read_text())
    config["assets"] = {
        "missing": {"batch": str(tmp_path / "absent"), "sample_id": "missing", "condition": "C2"}
    }
    config["profiles"] = {"baseline": config["profiles"]["baseline"]}
    config["video"]["enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_dynamics(path, tmp_path / "output")
    assert result["status"] == "completed_with_failures"
    assert result["planned_trials"] == 3 and result["successful_trials"] == 0
    assert all(row["failure_stage"] == "asset_load" for row in result["trials"])
    assert json.loads((tmp_path / "output/run.json").read_text())["planned_trials"] == 3
    with pytest.raises(FileExistsError):
        run_dynamics(path, tmp_path / "output")


@pytest.mark.parametrize("invalid_repeat", [None, 1])
def test_dynamics_records_each_timing_repeat(tmp_path, monkeypatch, invalid_repeat):
    from asset2sim.simulation import dynamics

    batch = tmp_path / "batch"
    run_physics_batch(
        Path("data/manifests/debug.jsonl"), batch, ["C0"], ["debug_box_v0"], reference=True
    )
    config = yaml.safe_load(Path("configs/physics/dynamics.yaml").read_text())
    config["assets"] = {
        "box": {"batch": str(batch), "sample_id": "debug_box_v0", "condition": "C0"}
    }
    config["profiles"] = {"baseline": config["profiles"]["baseline"]}
    config["video"]["enabled"] = False
    config["physics"]["timing_repeats"] = 3
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls = 0

    def fake_simulate(model, data, config, task, video_path=None):
        nonlocal calls
        index = calls % 4
        calls += 1
        valid = index != invalid_repeat
        return {
            "samples": [{"time_s": 0.0}],
            "valid": valid,
            "video_status": "not_requested",
            "video_error": None,
            "warnings": {},
            "invalid_reason": None if valid else "test_failure",
            "step_seconds": float(index),
        }

    monkeypatch.setattr(dynamics, "simulate", fake_simulate)
    result = dynamics.run_dynamics(config_path, tmp_path / "output")
    for trial in result["trials"]:
        assert len(trial["timing_repeats"]) == 3
        assert [r["step_seconds"] for r in trial["timing_repeats"]] == [1, 2, 3]
        assert trial["timing_planned"] == 3
        if invalid_repeat is None:
            assert trial["timing_valid"] == 3
            assert trial["timing_mean_step_seconds"] == pytest.approx(2.0)
            assert trial["status"] == "ok"
        else:
            assert trial["timing_valid"] == 2
            assert trial["timing_mean_step_seconds"] == pytest.approx(2.5)
            assert trial["status"] == "failed" and trial["error"] == "test_failure"
