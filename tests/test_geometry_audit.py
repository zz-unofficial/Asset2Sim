import numpy as np
import trimesh

from asset2sim.evaluation.geometry import normalized_mesh, transform_points
from asset2sim.evaluation.geometry_audit import normalize_in_pose


def test_post_pose_bounds_removes_rotated_cube_normalization_bias():
    original = trimesh.creation.box()
    rotation = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
    rotated = original.copy()
    rotated.apply_transform(rotation)
    before, _ = normalized_mesh(rotated, 1)
    restored = before.copy()
    restored.apply_transform(np.linalg.inv(rotation))
    assert restored.extents.max() < 0.8
    after, transform = normalize_in_pose(before, np.linalg.inv(rotation))
    np.testing.assert_allclose(after.vertices, original.vertices, atol=1e-12)
    np.testing.assert_allclose(transform_points(before.vertices, transform), after.vertices)
    # Audit operates on a copy and never changes the saved asset.
    np.testing.assert_allclose(before.extents.max(), 1)


def test_official_condition_camera_is_horizontal_and_distinct_from_render_elevation():
    from asset2sim.data.synthetic import camera_pose
    from asset2sim.experiments.config import load_parameters
    from asset2sim.paths import PROJECT_ROOT

    # Both upstream utils.default_cond_c2w use this OpenGL camera matrix.
    official = np.array([[0, 0, 1, 2.5], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    config = load_parameters(PROJECT_ROOT / "configs/evaluation/generation_benchmark.yaml")
    for model in ["sf3d", "spar3d"]:
        canonical = camera_pose(0, config["source_camera_elevation_degrees"][model], 2.5)
        np.testing.assert_allclose(canonical, official, atol=1e-12)
        reference = camera_pose(120, 75, 2.5)
        rotation = reference[:3, :3] @ canonical[:3, :3].T
        np.testing.assert_allclose(rotation @ canonical[:3, :3], reference[:3, :3], atol=1e-12)
        assert not np.allclose(camera_pose(0, 20, 2.5), canonical)


def test_closed_vertex_fan_check_rejects_pinched_closed_components():
    from asset2sim.evaluation.geometry_audit import closed_vertex_fan_errors

    box = trimesh.creation.box()
    assert closed_vertex_fan_errors(box) == 0
    other = box.copy()
    other.apply_translation([1, 1, 1])
    pinched = trimesh.util.concatenate([box, other])
    pinched.merge_vertices()
    assert pinched.is_watertight
    assert closed_vertex_fan_errors(pinched) == 1


def test_deployment_normalizes_after_declared_pose_without_reference_scale():
    from asset2sim.evaluation.geometry import normalized_mesh_in_frame

    mesh = trimesh.creation.box(extents=[2, 1, 0.5])
    frame = trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0])
    frame[:3, 3] = [0.3, -0.1, 0.2]
    result, normalization = normalized_mesh_in_frame(mesh, frame)
    result.apply_transform(frame)
    assert abs(result.extents.max() - 1) < 1e-12
    np.testing.assert_allclose(result.bounds.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        transform_points(mesh.vertices, frame @ normalization), result.vertices
    )


def test_audit_detects_visual_geometry_that_disagrees_with_recorded_matrices(tmp_path):
    import json

    from PIL import Image

    from asset2sim.evaluation.geometry_audit import audit

    mesh = trimesh.creation.box()
    mesh.export(tmp_path / "reference.glb")
    Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(tmp_path / "input.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "box", "image": "input.png", "reference_mesh": "reference.glb"})
        + "\n"
    )
    run = tmp_path / "run"
    run.mkdir()
    mesh.export(run / "mesh.glb")
    (run / "results.jsonl").write_text(json.dumps({"sample_id": "box", "mesh": "mesh.glb"}) + "\n")
    evaluation = tmp_path / "evaluation"
    folder = evaluation / "assets/test/box/aligned_diagnostic"
    folder.mkdir(parents=True)
    (evaluation / "protocol.json").write_text(
        json.dumps({"manifest": str(manifest), "runs": {"test": str(run)}})
    )
    (evaluation / "progress.jsonl").write_text(
        json.dumps(
            {
                "model": "test",
                "sample_id": "box",
                "object_id": "box",
                "track": "aligned_diagnostic",
                "geometry_status": "ok",
                "chamfer_l1": 0.0,
            }
        )
        + "\n"
    )
    (folder / "transforms.json").write_text(
        json.dumps(
            {
                k: np.eye(4).tolist()
                for k in ["rotation", "prediction_normalization", "reference_normalization"]
            }
        )
    )
    mesh.export(folder / "reference.glb")
    mesh.export(folder / "prediction.glb")
    audit([evaluation], tmp_path / "correct")
    assert (
        json.loads((tmp_path / "correct/summary.json").read_text())["matrix_reconstruction_invalid"]
        == 0
    )
    shifted = mesh.copy()
    shifted.apply_translation([0.05, 0, 0])
    shifted.export(folder / "prediction.glb")
    audit([evaluation], tmp_path / "mismatch")
    assert (
        json.loads((tmp_path / "mismatch/summary.json").read_text())[
            "matrix_reconstruction_invalid"
        ]
        == 1
    )


def test_canonical_surface_samples_ignore_vertex_and_face_reindexing():
    from asset2sim.evaluation.geometry_audit import canonical_surface_sample

    original = trimesh.creation.icosphere(subdivisions=1)
    permutation = np.random.default_rng(13).permutation(len(original.vertices))
    inverse = np.argsort(permutation)
    reordered = trimesh.Trimesh(
        vertices=original.vertices[permutation],
        faces=inverse[original.faces[::-1, ::-1]],
        process=False,
    )
    np.testing.assert_array_equal(
        canonical_surface_sample(original, 1000, 0), canonical_surface_sample(reordered, 1000, 0)
    )
