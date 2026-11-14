"""An arbitrary textured GLB must reach a portable collision-separated robot scene."""

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from asset2sim.simulation.grasp.pipeline import asset_pipeline


def test_textured_external_glb_end_to_end(tmp_path):
    root = Path.cwd()
    if not (root / ".local/mujoco_menagerie/universal_robots_ur5e/ur5e.xml").exists():
        pytest.skip("Menagerie unavailable")
    mesh = trimesh.creation.box(extents=[1, 0.5, 0.8])
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), image=Image.new("RGB", (8, 8), "red")
    )
    scene = trimesh.Scene()
    matrix = np.eye(4)
    matrix[:3, 3] = [2, 3, 4]
    scene.add_geometry(mesh, transform=matrix)
    source = tmp_path / "user.glb"
    scene.export(source)
    original = source.read_bytes()
    config = yaml.safe_load((root / "configs/physics/asset_interactive.yaml").read_text())
    for key in ("scene_config", "interactive_config", "collision_config"):
        config[key] = str((root / "configs/physics" / config[key]).resolve())
    config["condition"] = "C0"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = asset_pipeline(path, tmp_path / "output", mesh=source, up_axis="y", prepare_only=True)
    assert result["status"] == "prepared"
    asset = json.loads((tmp_path / "output/asset.json").read_text())
    assert asset["scene"]["visual"]["parts"][0]["color_texture"]
    assert source.read_bytes() == original
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "output/scene/scene.xml"))
    assert model.ntex == 1
    assert "\nvt " in (tmp_path / "output/scene/assets/visual_000.obj").read_text()
    shutil.copytree(tmp_path / "output/scene", tmp_path / "moved_scene")
    moved = mujoco.MjModel.from_xml_path(str(tmp_path / "moved_scene/scene.xml"))
    assert moved.ntex == model.ntex and moved.ngeom == model.ngeom
    assert np.allclose(max(asset["scene"]["dimensions_m"]), 0.1)
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == obj]
    assert sum(bool(model.geom_contype[i]) for i in geoms) == 1
    assert len(geoms) > 1
    with pytest.raises(FileExistsError):
        asset_pipeline(path, tmp_path / "output", mesh=source, up_axis="y", prepare_only=True)


def test_import_requires_declared_coordinates_and_saves_failure(tmp_path):
    source = tmp_path / "input.glb"
    trimesh.creation.box().export(source)
    with pytest.raises(ValueError, match="up-axis"):
        asset_pipeline(
            Path("configs/physics/asset_interactive.yaml"),
            tmp_path / "failed",
            mesh=source,
            prepare_only=True,
        )
    result = json.loads((tmp_path / "failed/pipeline.json").read_text())
    assert result["status"] == "failed"
    assert result["stages"][-1]["reason"]


def test_generation_failure_is_recorded(tmp_path, monkeypatch):
    import subprocess

    source = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(source)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr("asset2sim.simulation.grasp.pipeline.subprocess.run", fail)
    with pytest.raises(RuntimeError, match="generation.log"):
        asset_pipeline(
            Path("configs/physics/asset_interactive.yaml"),
            tmp_path / "failed",
            image=source,
            device="cpu",
            prepare_only=True,
        )
    result = json.loads((tmp_path / "failed/pipeline.json").read_text())
    assert result["stages"][-1]["stage"] == "generation"
    assert result["status"] == "failed"
    assert (tmp_path / "failed/raw/input.png").exists()
