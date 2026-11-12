import numpy as np
import pytest
import trimesh

from asset2sim.evaluation.geometry import (
    align_surfaces,
    normalized_mesh,
    surface_metrics,
    transform_points,
)


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
