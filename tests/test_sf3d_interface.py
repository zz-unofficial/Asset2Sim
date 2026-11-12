import json

import pytest

from asset2sim.generation.sf3d import SF3DConfig, load_manifest, run_manifest


def manifest_at(tmp_path, records):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_manifest_resolves_relative_paths_and_rejects_duplicate_ids(tmp_path):
    (tmp_path / "image.png").touch()
    record = {"sample_id": "box_v0", "image": "image.png"}
    path = manifest_at(tmp_path, [record])
    assert load_manifest(path)[0]["image"] == str(tmp_path / "image.png")
    manifest_at(tmp_path, [record, record])
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)


@pytest.mark.parametrize("sample_id", ["..", "../outside", "/absolute"])
def test_manifest_rejects_output_path_escape(tmp_path, sample_id):
    path = manifest_at(tmp_path, [{"sample_id": sample_id, "image": "input.png"}])
    with pytest.raises(ValueError, match="Unsafe"):
        load_manifest(path)


def test_inference_refuses_existing_output_before_loading_model(tmp_path):
    (tmp_path / "image.png").touch()
    path = manifest_at(tmp_path, [{"sample_id": "box", "image": "image.png"}])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "previous_result"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        run_manifest(SF3DConfig(), path, output)
    assert sentinel.read_text() == "preserve"


def test_reference_solids_are_closed_and_camera_basis_is_right_handed():
    import numpy as np

    from asset2sim.data.sf3d import camera_pose, reference_meshes

    for _, mesh in reference_meshes().values():
        assert mesh.is_watertight and mesh.is_volume
    pose = camera_pose(0, 20)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def test_official_export_axes_are_mapped_back_to_model_frame():
    import numpy as np

    from asset2sim.generation.sf3d import SF3D_RAW_TO_MODEL_Z_UP

    np.testing.assert_allclose(SF3D_RAW_TO_MODEL_Z_UP @ [0, 1, 0, 0], [0, 0, 1, 0])
    np.testing.assert_allclose(SF3D_RAW_TO_MODEL_Z_UP @ [0, 0, -1, 0], [1, 0, 0, 0])


def test_spar3d_generator_exports_mesh_and_sampled_points(tmp_path, monkeypatch):
    import trimesh
    from PIL import Image

    from asset2sim.generation.spar3d import SPAR3DConfig, SPAR3DGenerator

    class Model:
        def run_image(self, image, **kwargs):
            assert kwargs["return_points"] is True
            return trimesh.creation.box(), {
                "point_clouds": [trimesh.points.PointCloud([[0, 0, 0]])]
            }

    generator = SPAR3DGenerator(SPAR3DConfig(device="cpu"))
    generator.model = Model()
    monkeypatch.setattr(generator, "prepare_image", lambda path: Image.new("RGBA", (32, 32), "red"))
    result = generator.generate(tmp_path / "input.png", tmp_path / "sample")
    assert result["faces"] == 12
    assert result["points"] == 1
    assert (tmp_path / result["point_cloud"]).is_file()
    assert (tmp_path / "sample/mesh.glb").is_file()


def test_spar3d_recovery_releases_all_staged_modules():
    from asset2sim.generation.spar3d import SPAR3DConfig, SPAR3DGenerator

    class StagedModel:
        is_low_vram = True
        decoder = object()

        def __init__(self):
            self.loaded = {"points", "main", "estimator"}

        def _unload_pdiff_modules(self):
            self.loaded.discard("points")

        def _unload_main_modules(self):
            self.loaded.discard("main")

        def _unload_estimator_modules(self):
            self.loaded.discard("estimator")

    generator = SPAR3DGenerator(SPAR3DConfig(device="cpu"))
    generator.model = StagedModel()
    generator.recover()
    assert not generator.model.loaded
    assert generator.model.decoder is None
