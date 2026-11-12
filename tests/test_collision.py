import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from asset2sim.collision import build_collision
from asset2sim.collision.build import DEFAULT_CONFIG


def write_mesh(path, mesh):
    mesh.export(path)
    return path


def test_convex_hull_output_and_reproducible_metrics(tmp_path):
    mesh = write_mesh(tmp_path / "box.obj", trimesh.creation.box(extents=[0.2, 0.1, 0.1]))
    first = build_collision(mesh, tmp_path / "one", "C0", assume_prepared=True)
    second = build_collision(mesh, tmp_path / "two", "C0", assume_prepared=True)
    exported = trimesh.load(tmp_path / "one" / "collision_000.obj", force="mesh")
    assert first["component_count"] == 1
    assert exported.is_convex and exported.is_volume
    assert np.allclose(exported.bounds, [[-0.1, -0.05, -0.05], [0.1, 0.05, 0.05]])
    assert first["surface_error"] == second["surface_error"]
    assert first["total_faces"] == len(exported.faces)
    with pytest.raises(FileExistsError):
        build_collision(mesh, tmp_path / "one", "C0", assume_prepared=True)


def test_open_mesh_failure_is_recorded(tmp_path):
    box = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
    box.update_faces(np.arange(len(box.faces) - 1))
    mesh = write_mesh(tmp_path / "open.obj", box)
    with pytest.raises(RuntimeError, match="open, non-manifold"):
        build_collision(mesh, tmp_path / "failed", "C2", assume_prepared=True)
    result = json.loads((tmp_path / "failed" / "colliders.json").read_text())
    assert result["status"] == "failed"
    assert not result["input_diagnostic"]["watertight"]
    assert not list((tmp_path / "failed").glob("collision_*.obj"))


def test_open_input_requires_explicit_coacd_preprocessing(tmp_path):
    box = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
    box.update_faces(np.arange(len(box.faces) - 1))
    mesh = write_mesh(tmp_path / "open.obj", box)
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["surface_samples"] = 1000
    config["coacd"].update(mcts_iterations=20, mcts_nodes=8, resolution=1000)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError, match="requires preprocess_mode: on"):
        build_collision(mesh, tmp_path / "auto", "C2", path, assume_prepared=True, allow_open=True)
    config["coacd"]["preprocess_mode"] = "on"
    path.write_text(yaml.safe_dump(config))
    result = build_collision(
        mesh, tmp_path / "preprocessed", "C2", path, assume_prepared=True, allow_open=True
    )
    assert result["method"]["settings"]["preprocess_mode"] == "on"
    assert result["warnings"]
    assert result["component_count"] >= 1


def test_coacd_exports_separate_convex_components(tmp_path):
    ring = Path("data/samples/sf3d/reference/debug_ring.glb")
    if not ring.exists():
        pytest.skip("local prepared reference asset unavailable")
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["surface_samples"] = 1000
    config["coacd"].update(mcts_iterations=20, mcts_nodes=8, resolution=1000)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = build_collision(ring, tmp_path / "result", "C2", path, assume_prepared=True)
    assert result["component_count"] > 1
    assert result["method"]["settings"]["threshold"] == 0.05
    assert result["method"]["settings"]["seed"] == 0
    assert result["method"]["settings"]["real_metric"] is False
    for component in result["components"]:
        loaded = trimesh.load(tmp_path / "result" / component["path"], force="mesh")
        assert loaded.is_volume and loaded.is_convex
        assert len(loaded.faces) == component["faces"]


def test_missing_metadata_requires_explicit_declaration(tmp_path):
    mesh = write_mesh(tmp_path / "box.obj", trimesh.creation.box())
    with pytest.raises(RuntimeError, match="Missing Task A metadata"):
        build_collision(mesh, tmp_path / "result", "C0")
    assert json.loads((tmp_path / "result/colliders.json").read_text())["status"] == "failed"
