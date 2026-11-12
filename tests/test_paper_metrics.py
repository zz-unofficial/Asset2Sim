import numpy as np
import pytest
import trimesh
from PIL import Image

from asset2sim.evaluation.geometry import (
    align_surfaces,
    bounded_symmetric_icp,
    distance_summary,
    normalized_mesh,
    surface_metrics,
    transform_points,
)
from asset2sim.evaluation.paper import aggregate_views, silhouette_metrics, summarize_records


def test_chamfer_is_unsquared_bidirectional_mean_and_fscore_uses_both_sides():
    source = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    target = np.array([[0.0, 0, 0]])
    result = surface_metrics(source, target, [0.1])
    assert result["chamfer_l1"] == pytest.approx(0.5)
    assert result["chamfer_squared"] == pytest.approx(1.0)
    assert result["precision@0.1"] == pytest.approx(0.5)
    assert result["recall@0.1"] == pytest.approx(1.0)
    assert result["fscore@0.1"] == pytest.approx(2 / 3)


def test_normalization_records_an_invertible_transform_and_preserves_shape():
    mesh = trimesh.creation.box(extents=[4, 2, 1])
    mesh.apply_translation([10, -2, 3])
    normalized, transform = normalized_mesh(mesh, 1)
    np.testing.assert_allclose(normalized.bounds.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(normalized.extents, [1, 0.5, 0.25])
    np.testing.assert_allclose(
        transform_points(normalized.vertices, np.linalg.inv(transform)), mesh.vertices
    )


def test_rotation_search_and_rigid_icp_recover_orientation_without_reflection():
    points = np.random.default_rng(7).normal(size=(60, 3))
    points -= points.mean(axis=0)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = points @ rotation.T + [0.03, -0.02, 0.01]
    transform, count = align_surfaces(points, target, 90, 30, 1e-10)
    assert count == 24
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(transform_points(points, transform), target, atol=1e-8)


def test_empty_surfaces_and_invalid_rotation_grid_are_reported():
    with pytest.raises(ValueError, match="nonempty"):
        surface_metrics(np.empty((0, 3)), np.ones((2, 3)), [0.1])
    with pytest.raises(ValueError, match="divisor"):
        align_surfaces(np.ones((2, 3)), np.ones((2, 3)), 7, 10, 1e-6)


def test_gso_texture_is_loaded_from_release_materials_directory(tmp_path):
    from PIL import Image

    from asset2sim.data.paper import load_source_mesh

    meshes = tmp_path / "meshes"
    textures = tmp_path / "materials/textures"
    meshes.mkdir()
    textures.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 90, 180)).save(textures / "texture.png")
    (meshes / "model.mtl").write_text("newmtl material_0\nmap_Kd texture.png\n")
    path = meshes / "model.obj"
    path.write_text(
        "mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nusemtl material_0\nf 1/1 2/2 3/3\n"
    )
    mesh = load_source_mesh(path, "gso")
    assert mesh.visual.material.image.getpixel((0, 0)) == (10, 90, 180)


def test_similarity_refinement_can_recover_scale():
    from asset2sim.evaluation.geometry import rigid_fit

    points = np.random.default_rng(1).normal(size=(50, 3))
    target = points * 1.3 + [0.1, -0.2, 0.3]
    transform = rigid_fit(points, target, scale=True)
    np.testing.assert_allclose(transform_points(points, transform), target, atol=1e-10)


def test_y_up_source_transform_is_proper_and_matches_reference_frame():
    from asset2sim.data.paper import source_to_z_up

    transform = source_to_z_up("Y")
    np.testing.assert_array_equal(transform_points(np.array([[0, 1, 0]]), transform), [[0, 0, 1]])
    assert np.linalg.det(transform[:3, :3]) == 1
    with pytest.raises(ValueError, match="source_up_axis"):
        source_to_z_up("X")


def test_bounded_scale_cannot_accumulate_shrinkage_and_keeps_symmetric_improvement():
    source = np.random.default_rng(4).normal(size=(200, 3))
    target = source * 0.4 + [0.02, -0.04, 0.01]
    result, info = bounded_symmetric_icp(source, target, np.eye(4), (0.9, 1.1), 100)
    assert 0.9 - 1e-9 <= np.cbrt(np.linalg.det(result[:3, :3])) <= 1.1 + 1e-9
    assert info["final_cd"] <= info["initial_cd"]
    assert surface_metrics(transform_points(source, result), target, [])[
        "chamfer_l1"
    ] == pytest.approx(info["final_cd"])


def test_identity_alignment_is_not_changed_by_similarity_refinement():
    points = np.random.default_rng(1).normal(size=(80, 3))
    result, info = bounded_symmetric_icp(points, points, np.eye(4))
    np.testing.assert_allclose(result, np.eye(4))
    assert info["scale"] == 1
    assert info["accepted_iterations"] == 0
    with pytest.raises(ValueError, match="contain 1"):
        bounded_symmetric_icp(points, points, np.eye(4), (0.5, 0.8))


def test_strict_fscore_and_quantiles_have_known_distance_units():
    a = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    b = a + [0, 0.015, 0]
    scores = surface_metrics(a, b, [0.01, 0.02, 0.05])
    assert scores["fscore@0.01"] == 0
    assert scores["fscore@0.02"] == 1
    assert scores["fscore@0.05"] == 1
    assert distance_summary(a, b)["hausdorff95"] == pytest.approx(0.015)


def test_mask_iou_penalizes_a_missing_shape_even_when_both_rgb_images_are_white():
    reference = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
    reference.paste((255, 255, 255, 255), (2, 2, 6, 6))
    predicted = Image.new("RGBA", (8, 8), (255, 255, 255, 0))
    result = silhouette_metrics(predicted, reference)
    assert result["mask_iou"] == 0
    assert result["foreground_area_ratio"] == 0
    assert result["psnr_foreground"] == 120
    with pytest.raises(ValueError, match="no foreground"):
        silhouette_metrics(predicted, predicted)


def test_conditioning_view_is_excluded_from_novel_view_mean():
    result = aggregate_views([{"psnr": 30}, {"psnr": 10}, {"psnr": 20}], 0)
    assert result == {"psnr": 20, "input_psnr": 30, "novel_psnr": 15}


def test_per_metric_denominators_keep_geometry_when_image_evaluation_fails():
    result = summarize_records(
        [
            {
                "status": "ok",
                "geometry_status": "ok",
                "image_status": "ok",
                "chamfer_l1": 0.1,
                "psnr": 20.0,
            },
            {
                "status": "failed",
                "geometry_status": "ok",
                "image_status": "failed",
                "chamfer_l1": 0.3,
            },
        ]
    )
    assert result["planned"] == 2
    assert result["successful"] == 1
    assert result["geometry_successful"] == 2
    assert result["chamfer_l1"] == pytest.approx(0.2)
    assert result["metric_counts"]["chamfer_l1"] == 2
    assert result["metric_counts"]["psnr"] == 1


def test_reports_reject_different_image_settings_before_writing(tmp_path):
    import json

    from asset2sim.evaluation.report import write_report

    paths = []
    for version in (1, 2):
        folder = tmp_path / f"v{version}"
        folder.mkdir()
        (folder / "evaluation.json").write_text(
            json.dumps({"image_metrics_definition": f"setting_{version}"})
        )
        paths.append(folder)
    with pytest.raises(ValueError, match="Image settings differ"):
        write_report(paths, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_model_comparison_identifies_models_and_requires_paired_objects(tmp_path):
    import json

    from asset2sim.evaluation.report import write_report

    directories = []
    for model in ("sf3d", "spar3d"):
        directory = tmp_path / model
        directory.mkdir()
        (directory / "evaluation.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "device": "cpu",
                    "profile": {"name": "same", "seed": 0},
                    "image_metrics_definition": "same",
                    "datasets": {"gso": {"planned": 1, "successful": 0}},
                }
            )
        )
        (directory / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "dataset": "gso",
                    "sample_id": "same",
                    "status": "failed",
                }
            )
            + "\n"
        )
        directories.append(directory)
    write_report(directories, tmp_path / "report")
    report = (tmp_path / "report/comparison.md").read_text()
    assert "| sf3d | gso |" in report and "| spar3d | gso |" in report
    assert (tmp_path / "report/paired.csv").is_file()
    (directories[1] / "metrics.jsonl").write_text(
        json.dumps(
            {
                "dataset": "gso",
                "sample_id": "different",
                "status": "failed",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Sample IDs differ"):
        write_report(directories, tmp_path / "invalid")
