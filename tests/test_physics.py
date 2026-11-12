import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from asset2sim.collision import build_collision
from asset2sim.simulation.evaluate import _probe_results, run_physics_eval
from asset2sim.simulation.reference import distance_to_solid
from asset2sim.simulation.scene import make_scene


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/physics/debug.yaml").read_text())


def test_programmatic_reference_independently_labels_hole_and_solid():
    radius = 0.01
    for y in np.linspace(-0.16, 0.16, 81):
        assert distance_to_solid("debug_ring", np.array([0, y, 0])) > radius
    assert distance_to_solid("debug_ring", np.array([0.0775, 0, 0])) == 0
    assert distance_to_solid("debug_ring", np.array([0.0775, -0.1, 0])) > radius
    assert distance_to_solid("debug_box", np.array([0, 0, 0])) == 0
    assert distance_to_solid("debug_box", np.array([0.12, 0, 0])) > radius


@pytest.fixture
def box_collision(tmp_path):
    reference = Path("data/samples/sf3d/reference/debug_box.glb")
    collision_dir = tmp_path / "collision"
    build_collision(reference, collision_dir, "C0", assume_prepared=True)
    return reference, collision_dir


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_scene_component_coordinates_and_fixed_inertia(tmp_path, config, box_collision, mode):
    reference, collision_dir = box_collision
    model, data, validation = make_scene(
        collision_dir, reference, tmp_path / "visual.obj", config, body_mode=mode
    )
    assert validation["actual_collision_geoms"] == validation["expected_components"] == 1
    assert validation["components"][0]["vertex_error_m"] < 2e-6
    assert validation["components"][0]["aabb_error_m"] < 2e-6
    assert not validation["visual_collision_enabled"]
    assert validation["body_mass_kg"] == config["body_mass_kg"]
    assert np.allclose(validation["body_inertia_kg_m2"], config["body_diagonal_inertia_kg_m2"])
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


def test_buffer_is_not_geometric_contact(tmp_path, config, box_collision):
    reference, collision_dir = box_collision
    model, data, _ = make_scene(collision_dir, reference, tmp_path / "visual.obj", config)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_y")
    data.qpos[model.jnt_qposadr[joint]] = -0.061 - config["trajectory_start_y_m"]
    mujoco.mj_forward(model, data)
    contact = _probe_results(model, data)
    assert contact["min_detected_distance_m"] == pytest.approx(0.001, abs=1e-6)
    assert contact["buffer_only"] and not contact["geometric_contact"]


def test_one_command_experiment_writes_independent_records(tmp_path, config):
    config.update(
        conditions=["C0"],
        trajectory_samples=17,
        timing_repeats=1,
        simulation_steps=1200,
        build_repeats=1,
        drive_steps=900,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    result = run_physics_eval(path, output)
    assert result["status"] == "completed"
    assert len(result["results"]) == 2
    samples = list(csv.DictReader((output / "contact_samples.csv").open()))
    assert len(samples) == 2 * 2 * 17
    assert any(
        row["object"] == "debug_ring"
        and row["trajectory"] == "hole"
        and row["geometric_contact"] == "True"
        for row in samples
    )
    assert json.loads((output / "summary.json").read_text())["results"]
    assert (output / "scenes/debug_ring/C0_dynamic.xml").exists()
    with pytest.raises(FileExistsError):
        run_physics_eval(path, output)
