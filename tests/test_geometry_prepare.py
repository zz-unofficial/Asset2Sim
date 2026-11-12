import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from asset2sim.geometry.prepare import (
    PreparationOptions,
    compare_export,
    inspect_mesh,
    load_geometry,
    physical_status,
    prepare_mesh,
)


def test_scene_expands_nested_transform_and_repeated_instances(tmp_path):
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(
        frame_from="world",
        frame_to="parent",
        matrix=trimesh.transformations.translation_matrix([1, 2, 3]),
    )
    scene.add_geometry(
        trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
        node_name="first",
        geom_name="box",
        parent_node_name="parent",
        transform=trimesh.transformations.translation_matrix([0.3, 0, 0]),
    )
    scene.graph.update(
        frame_from="parent",
        frame_to="second",
        geometry="box",
        matrix=trimesh.transformations.translation_matrix([-0.3, 0, 0]),
    )
    path = tmp_path / "nested.glb"
    scene.export(path)
    mesh, nodes = load_geometry(path)
    assert len(nodes) == 2
    assert len(mesh.faces) == 24
    np.testing.assert_allclose(mesh.bounds, [[0.65, 1.95, 2.95], [1.35, 2.05, 3.05]], atol=1e-6)
    assert inspect_mesh(mesh)["connected_components"] == 2


def test_reference_preserves_meters_coordinates_and_source(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.05])
    mesh.apply_translation([0.4, 0.2, 0.1])
    source = tmp_path / "reference.glb"
    mesh.export(source)
    original = source.read_bytes()
    out = tmp_path / "out"
    metadata = prepare_mesh(source, out, PreparationOptions("reference"))
    assert source.read_bytes() == original
    assert metadata["physical_coordinates"] == {"up_axis": "+Z", "units": "meters"}
    np.testing.assert_allclose(metadata["scene_to_physical"], np.eye(4))
    result, _ = load_geometry(out / "normalized.glb")
    np.testing.assert_allclose(result.bounds, mesh.bounds, atol=1e-6)
    report = json.loads((out / "mesh_report.json").read_text())
    assert report["after"]["valid_volume"]
    assert report["after"]["watertight"]
    assert report["after_export"]["valid_volume"]
    assert report["export_validation"]["passed"]
    assert metadata["physical_adaptation"]["status"] == "ready"
    with pytest.raises(FileExistsError):
        prepare_mesh(source, out, PreparationOptions("reference"))


@pytest.mark.parametrize("extension", ["obj", "stl", "glb"])
def test_generated_rotation_scale_and_run_provenance(tmp_path, extension):
    run = tmp_path / "run"
    run.mkdir()
    source = run / f"mesh.{extension}"
    trimesh.creation.box(extents=[1, 2, 4]).export(source)
    rotation = [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    (run / "run.json").write_text(
        json.dumps(
            {
                "model": "fixture",
                "raw_mesh_coordinates": {
                    "up_axis": "+Y",
                    "units": "normalized",
                    "raw_to_model_z_up": rotation,
                },
            }
        )
    )
    (run / "results.jsonl").write_text(
        json.dumps({"sample_id": "item", "status": "ok", "mesh": source.name}) + "\n"
    )
    output = tmp_path / "result"
    metadata = prepare_mesh(source, output, PreparationOptions("generated"), run)
    physical, _ = load_geometry(output / "normalized.glb")
    np.testing.assert_allclose(physical.extents, [0.2, 0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(physical.bounds.mean(axis=0), [0, 0, 0], atol=1e-6)
    assert metadata["sample_id"] == "item"
    assert metadata["alignment"].startswith("none")
    np.testing.assert_allclose(
        np.array(metadata["scene_to_physical"])[:3, :3], np.array(rotation)[:3, :3] * 0.05
    )


def test_invalid_run_and_reference_scale_leave_auditable_failure(tmp_path):
    source = tmp_path / "wrong.glb"
    trimesh.creation.box().export(source)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Reference longest edge"):
        prepare_mesh(source, output, PreparationOptions("reference"))
    assert json.loads((output / "metadata.json").read_text())["status"] == "failed"
    assert json.loads((output / "mesh_report.json").read_text())["before"][
        "longest_extent"
    ] == pytest.approx(1)
    output2 = tmp_path / "missing_run"
    with pytest.raises(ValueError, match="require their generator run"):
        prepare_mesh(source, output2, PreparationOptions("generated"))
    assert json.loads((output2 / "mesh_report.json").read_text())["status"] == "failed"


def test_generated_mesh_rejects_undeclared_reflection_or_wrong_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    source = run / "mesh.glb"
    trimesh.creation.box().export(source)
    (run / "results.jsonl").write_text(
        json.dumps({"sample_id": "item", "status": "ok", "mesh": "mesh.glb"}) + "\n"
    )
    coordinates = {
        "up_axis": "+Y",
        "units": "normalized",
        "raw_to_model_z_up": np.diag([-1, 1, 1, 1]).tolist(),
    }
    (run / "run.json").write_text(json.dumps({"raw_mesh_coordinates": coordinates}))
    with pytest.raises(ValueError, match="proper rotation"):
        prepare_mesh(source, tmp_path / "reflection", PreparationOptions("generated"), run)
    unrelated = tmp_path / "unrelated.glb"
    trimesh.creation.box().export(unrelated)
    with pytest.raises(ValueError, match="successful output"):
        prepare_mesh(unrelated, tmp_path / "wrong_run", PreparationOptions("generated"), run)


def test_explicit_degenerate_cleanup_is_reported(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
    mesh.faces = np.vstack([mesh.faces, [0, 0, 1]])
    source = tmp_path / "degenerate.obj"
    mesh.export(source)
    raw = tmp_path / "uncleaned"
    prepare_mesh(source, raw, PreparationOptions("reference"))
    assert json.loads((raw / "mesh_report.json").read_text())["after"]["degenerate_faces"] == 1
    cleaned = tmp_path / "cleaned"
    metadata = prepare_mesh(
        source, cleaned, PreparationOptions("reference", remove_degenerate=True)
    )
    assert json.loads((cleaned / "mesh_report.json").read_text())["after"]["degenerate_faces"] == 0
    assert metadata["operations"][-1]["removed"] == 1


def test_nonfinite_is_reported_without_using_geometry_properties():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, float("nan"), 0]], faces=[[0, 1, 2]], process=False
    )
    report = inspect_mesh(mesh)
    assert not report["finite_vertices"]
    assert "non_finite_vertices" in report["issues"]


def test_split_uv_vertices_are_diagnosed_and_only_welded_when_requested(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
    mesh.unmerge_vertices()
    path = tmp_path / "seams.obj"
    mesh.export(path)
    out = tmp_path / "default"
    prepare_mesh(path, out, PreparationOptions("reference"))
    report = json.loads((out / "mesh_report.json").read_text())["after"]
    assert not report["watertight"]
    assert report["coincident_vertices"] > 0
    assert report["position_merged_diagnostic"]["watertight"]
    assert physical_status(report)["topology"] == "coincident_seam_candidate"
    assert physical_status(report)["status"] == "needs_repair"
    merged = tmp_path / "merged"
    meta = prepare_mesh(path, merged, PreparationOptions("reference", merge_coincident=True))
    assert json.loads((merged / "mesh_report.json").read_text())["after"]["watertight"]
    assert meta["operations"][-1]["operation"] == "merge_coincident_vertices"
    assert meta["physical_adaptation"]["status"] == "ready"


def test_nonmanifold_edge_is_reported_as_geometry_issue():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.2, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1]],
        faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
        process=False,
    )
    report = inspect_mesh(mesh)
    assert report["nonmanifold_edges"] == 1
    assert "nonmanifold_edges" in report["issues"]
    assert physical_status(report)["status"] == "needs_repair"
    assert physical_status(report)["topology"] == "geometry_review"


def test_roundtrip_comparison_detects_reversed_face_winding():
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
    flipped = mesh.copy()
    flipped.invert()
    validation = compare_export(inspect_mesh(mesh), inspect_mesh(flipped), 0.2, mesh, flipped)
    assert validation["checks"]["vertex_positions_preserved"]
    assert not validation["checks"]["triangle_connectivity_preserved"]
    assert not validation["passed"]


def test_export_geometry_mismatch_fails_with_diagnostics(tmp_path, monkeypatch):
    from asset2sim.geometry import prepare

    source = tmp_path / "box.glb"
    trimesh.creation.box(extents=[0.2, 0.1, 0.1]).export(source)
    load = prepare.load_geometry

    def tampered(path):
        mesh, nodes = load(path)
        if path.name == "normalized.glb":
            mesh.vertices[0, 0] += 0.01
        return mesh, nodes

    monkeypatch.setattr(prepare, "load_geometry", tampered)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="round-trip"):
        prepare_mesh(source, output, PreparationOptions("reference"))
    report = json.loads((output / "mesh_report.json").read_text())
    assert report["after_export"] is not None
    assert not report["export_validation"]["checks"]["vertex_positions_preserved"]
    assert report["physical_adaptation"]["status"] == "invalid"


@pytest.mark.parametrize("name", ["debug_box", "debug_cup", "debug_ring", "debug_bracket"])
def test_actual_debug_reference_through_cli(tmp_path, name):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/samples/sf3d/reference" / f"{name}.glb"
    before = source.read_bytes()
    output = tmp_path / name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "asset2sim.cli",
            "prepare-mesh",
            "--kind",
            "reference",
            "--mesh",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads((output / "mesh_report.json").read_text())
    assert report["export_validation"]["passed"]
    assert report["after_export"]["longest_extent"] == pytest.approx(0.2, abs=1e-6)
    assert report["physical_adaptation"]["status"] == "ready"
    assert source.read_bytes() == before


@pytest.mark.parametrize("model", ["sf3d", "spar3d"])
def test_actual_generated_mesh_roundtrip(tmp_path, model):
    root = Path(__file__).resolve().parents[1]
    run = root / "data/generated" / model / "paper_gso/run02"
    source = run / "gso_SNAIL_MEASURING_TAPE/mesh.glb"
    if not source.exists() or not (run / "results.jsonl").exists():
        pytest.skip("Local generated asset is not available")
    output = tmp_path / model
    metadata = prepare_mesh(source, output, PreparationOptions("generated"), run)
    report = json.loads((output / "mesh_report.json").read_text())
    assert report["export_validation"]["passed"]
    assert report["after_export"]["longest_extent"] == pytest.approx(0.2, abs=1e-6)
    assert metadata["physical_adaptation"]["status"] == "needs_repair"
    assert report["physical_adaptation"]["topology"] == "uv_seam_candidate"
