import shutil
from pathlib import Path

import mujoco
import pytest
import yaml

from asset2sim.collision import build_collision
from asset2sim.simulation.grasp.controller import score_grasp, scripted_trial
from asset2sim.simulation.grasp.scene import create_scene


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/physics/grasp.yaml").read_text())


def test_success_requires_both_force_bearing_fingers_and_release(config):
    hold = {
        "phase": "hold",
        "object_z_m": 0.12,
        "left_contact": True,
        "right_contact": True,
        "left_normal_force_n": 1.0,
        "right_normal_force_n": 1.0,
        "table_contact": False,
    }
    release = {
        **hold,
        "phase": "release",
        "left_contact": False,
        "right_contact": False,
        "table_contact": True,
    }
    assert score_grasp([hold, release], 0.0, config, {})["success"]
    assert not score_grasp([{**hold, "right_normal_force_n": 0.0}, release], 0.0, config, {})[
        "success"
    ]
    assert not score_grasp([hold, {**release, "table_contact": False}], 0.0, config, {})["success"]
    assert not score_grasp([{**hold, "right_contact": False}, release], 0.0, config, {})[
        "release_success"
    ]
    assert not score_grasp([hold, release], 0.0, config, {"solver_warning": 1})["success"]


def test_reference_grasp_with_menagerie(tmp_path, config):
    menagerie = Path(".local/mujoco_menagerie")
    if not (menagerie / "universal_robots_ur5e/ur5e.xml").exists():
        pytest.skip("Menagerie assets unavailable")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "normalized.glb").write_bytes(
        Path("data/samples/sf3d/reference/debug_box.glb").read_bytes()
    )
    colliders = tmp_path / "colliders"
    build_collision(prepared / "normalized.glb", colliders, "C0", assume_prepared=True)
    model, info = create_scene(menagerie, prepared, colliders, config, tmp_path / "scene")
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert model.body_mass[obj] == pytest.approx(0.1)
    assert info["component_count"] == 1
    assert (
        sum(model.geom_bodyid[i] == obj and bool(model.geom_contype[i]) for i in range(model.ngeom))
        == 1
    )
    shutil.copytree(tmp_path / "scene", tmp_path / "moved_scene")
    reloaded = mujoco.MjModel.from_xml_path(str(tmp_path / "moved_scene" / "scene.xml"))
    assert reloaded.ngeom == model.ngeom
    assert not any(
        model.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT)
        and obj in (model.eq_obj1id[i], model.eq_obj2id[i])
        for i in range(model.neq)
    )
    result = scripted_trial(model, config, tmp_path / "trial", video=False)
    assert result["success"] and result["release_success"]
    assert result["hold_success_samples"] == result["hold_valid_samples"] > 0
