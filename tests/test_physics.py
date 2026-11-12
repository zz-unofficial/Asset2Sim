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
    assert contact["solver_active"] and not contact["inactive_only"]
    data.qpos[model.jnt_qposadr[joint]] = -0.063 - config["trajectory_start_y_m"]
    mujoco.mj_forward(model, data)
    inactive = _probe_results(model, data)
    assert inactive["buffer_only"] and inactive["inactive_only"] and not inactive["solver_active"]


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


def test_saved_scene_reloads_after_directory_move(tmp_path, config, box_collision):
    import shutil

    reference, collision_dir = box_collision
    original = tmp_path / "original"
    original.mkdir()
    xml = original / "scene.xml"
    model, _, _ = make_scene(
        collision_dir, reference, original / "visual.obj", config, xml_path=xml
    )
    moved = tmp_path / "relocated"
    shutil.move(str(original), moved)
    loaded = mujoco.MjModel.from_xml_path(str(moved / "scene.xml"))
    assert loaded.ngeom == model.ngeom
    assert np.allclose(loaded.geom_pos, model.geom_pos)
    assert np.allclose(loaded.mesh_vert, model.mesh_vert)


def test_containment_is_separately_recorded():
    from asset2sim.simulation.reference import reference_category

    assert reference_category("debug_box", np.array([0, 0, 0]), 0.01, 1e-6) == "fully_contained"
    assert (
        reference_category("debug_box", np.array([0, -0.055, 0]), 0.01, 1e-6) == "boundary_overlap"
    )
    assert reference_category("debug_box", np.array([0, -0.1, 0]), 0.01, 1e-6) == "separated"


def test_zero_margin_changes_buffer_not_reference(tmp_path, config, box_collision):
    reference, collision_dir = box_collision
    results = []
    for name, margin, gap in [("buffered", 0.004, 0.002), ("zero", 0.0, 0.0)]:
        local = dict(config, contact_margin_m=margin, contact_gap_m=gap)
        model, data, _ = make_scene(collision_dir, reference, tmp_path / f"{name}.obj", local)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_y")
        data.qpos[model.jnt_qposadr[joint]] = -0.061 - config["trajectory_start_y_m"]
        mujoco.mj_forward(model, data)
        results.append(_probe_results(model, data))
    assert results[0]["buffer_only"]
    assert results[1]["contact_count"] == 0
    assert not results[0]["geometric_contact"] and not results[1]["geometric_contact"]


def test_index_generated_assets_provenance(tmp_path):
    from asset2sim.simulation.assets import index_generated_assets

    run = tmp_path / "generation"
    mesh = run / "sample" / "mesh.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"raw")
    (run / "results.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "object_id": "obj",
                "view_id": "v0",
                "status": "ok",
                "mesh": "sample/mesh.glb",
            }
        )
        + "\n"
    )
    (tmp_path / "image.png").write_bytes(b"image")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "object_id": "obj",
                "view_id": "v0",
                "image": "image.png",
                "reference_mesh": "not_read.glb",
            }
        )
        + "\n"
    )
    root = tmp_path / "physics"
    prepared = root / "prepared/sample"
    prepared.mkdir(parents=True)
    (prepared / "normalized.glb").write_bytes(b"prepared")
    (prepared / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source": str(mesh),
                "options": {"kind": "generated"},
                "physical_coordinates": {"up_axis": "+Z", "units": "meters"},
            }
        )
    )
    collision = root / "collisions/sample/C0"
    collision.mkdir(parents=True)
    (collision / "collision_000.obj").write_text("mesh")
    (collision / "colliders.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "condition": "C0",
                "input": str(prepared / "normalized.glb"),
                "component_count": 1,
                "components": [{"path": "collision_000.obj"}],
            }
        )
    )
    rows = index_generated_assets(manifest, run, root, ("C0",))
    assert rows[0]["status"] == "ready"
    (collision / "collision_000.obj").unlink()
    assert index_generated_assets(manifest, run, root, ("C0",))[0]["status"] == "failed"


def test_ablation_profiles(tmp_path, config):
    config.update(
        objects={"debug_ring": "data/samples/sf3d/reference/debug_ring.glb"},
        conditions=["C0"],
        trajectory_samples=9,
        timing_repeats=1,
        build_repeats=1,
        validate_dynamic_body=False,
        contact_profiles=[
            {"name": "buffered", "margin_m": 0.004, "gap_m": 0.002},
            {"name": "zero", "margin_m": 0.0, "gap_m": 0.0},
        ],
        probe_cases={
            "debug_ring": [
                {"name": "hole", "x_m": 0.0, "radius_m": 0.005},
                {"name": "solid", "x_m": 0.0775, "radius_m": 0.01},
            ]
        },
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    result = run_physics_eval(path, run)
    assert result["status"] == "completed"
    assert {row["profile"] for row in result["results"]} == {"buffered", "zero"}
    assert all(row["planned_samples"] == 18 for row in result["results"])
    assert (run / "summary.json").is_file()
    assert (run / "contact_samples.csv").is_file()
