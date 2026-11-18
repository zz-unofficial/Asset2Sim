"""Reference pairing, input preparation and evidence provenance."""

import json
import shutil

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from asset2sim.data.manifest import load_manifest
from asset2sim.data.physx_omni import prepare_dataset, prepare_rgb


def test_letterbox_preserves_aspect_and_pixel_map(tmp_path):
    source, target = tmp_path / "source.png", tmp_path / "input.png"
    Image.new("RGB", (400, 200)).save(source)
    result = prepare_rgb(source, target)
    assert Image.open(target).size == (512, 512)
    assert result["pixel_transform"] == [[1.28, 0, 0], [0, 1.28, 128], [0, 0, 1]]


def test_rigid_prepare_move_and_no_fake_pending_samples(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    trimesh.creation.box().export(tmp_path / "ref.glb")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {
                        "sample_id": "box",
                        "category": "rigid",
                        "image": "image.png",
                        "reference_mesh": "ref.glb",
                        "source": "local",
                    }
                ],
                "pending": [{"category": "articulated", "archive": "Microwave.tar"}],
            }
        )
    )
    output = tmp_path / "prepared"
    result = prepare_dataset(config, output)
    assert result["n_prepared"] == 1 and not result["ready"]
    shutil.move(output, tmp_path / "moved")
    records = load_manifest(tmp_path / "moved/manifest.jsonl")
    assert len(records) == 1
    assert all(
        __import__("pathlib").Path(records[0][key]).is_file()
        for key in ("image", "reference_mesh", "sample_metadata")
    )
    assert (tmp_path / "moved/dataset_download_manifest.md").is_file()
    with pytest.raises(FileExistsError):
        prepare_dataset(config, tmp_path / "moved")


def cloth_config(tmp_path, **overrides):
    Image.new("RGB", (8, 8)).save(tmp_path / "rgb.png")
    np.savez(tmp_path / "depth.npz", depth=np.ones((8, 8)))
    np.savez(tmp_path / "normal.npz", normals=np.ones((8, 8, 3)))
    np.savez(tmp_path / "mesh.npz", mesh=np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]))
    np.savetxt(tmp_path / "simplices.tri", [[1, 2, 3]], fmt="%d")
    sample = {
        "sample_id": "cloth_case",
        "category": "deformable",
        "sequence_id": "sequence_one",
        "frame_id": "003",
        "image": "rgb.png",
        "depth": "depth.npz",
        "normal": "normal.npz",
        "mesh_vertices": "mesh.npz",
        "triangles": "simplices.tri",
        "triangle_index_base": 1,
        "pairing_verified": True,
        "pairing_evidence": "fixture matching frame index",
        "intrinsics": [[10, 0, 4], [0, 10, 4], [0, 0, 1]],
        **overrides,
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"samples": [sample]}))
    return config


def test_cloth_keeps_same_frame_mesh_and_intrinsics(tmp_path):
    result = prepare_dataset(cloth_config(tmp_path), tmp_path / "prepared")
    assert result["n_prepared"] == 1
    metadata = json.loads((tmp_path / "prepared/cloth_case/sample_metadata.json").read_text())
    assert metadata["frame_id"] == "003"
    assert metadata["input_intrinsics"][0] == [640, 0, 256]
    mesh = trimesh.load(tmp_path / "prepared/cloth_case/reference.ply", process=False)
    assert len(mesh.vertices) == 3 and len(mesh.faces) == 1


def test_unverified_cloth_pairing_remains_failed(tmp_path):
    result = prepare_dataset(cloth_config(tmp_path, pairing_verified=False), tmp_path / "prepared")
    assert result["n_failed"] == 1
    assert "pairing" in result["samples"][0]["reason"]
    assert (tmp_path / "prepared/manifest.jsonl").read_text() == ""


def test_predictions_cannot_be_written_as_true_physical_labels(tmp_path):
    config = cloth_config(
        tmp_path,
        physical_labels={
            "young": {
                "value": 10,
                "source": "model",
                "evidence_type": "model_inference",
                "ground_truth": True,
            }
        },
    )
    result = prepare_dataset(config, tmp_path / "prepared")
    assert result["n_failed"] == 1
    assert "cannot be ground truth" in result["samples"][0]["reason"]


def test_same_sequence_cannot_count_as_two_core_cases(tmp_path):
    config = cloth_config(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["samples"].append({**value["samples"][0], "sample_id": "second_frame", "frame_id": "004"})
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="distinct sequences"):
        prepare_dataset(config, tmp_path / "prepared")


def test_visible_depth_reference_does_not_require_or_fabricate_mesh(tmp_path):
    Image.new("L", (8, 8), 255).save(tmp_path / "mask.png")
    config = cloth_config(
        tmp_path,
        reference_kind="visible_depth",
        foreground_mask="mask.png",
        depth_unit_m=1,
        object_class="tshirt",
        object_id="epfl_tshirt",
        mesh_vertices="absent.npz",
        triangles="absent.tri",
    )
    output = tmp_path / "prepared"
    assert prepare_dataset(config, output)["n_prepared"] == 1
    record = json.loads((output / "manifest.jsonl").read_text())
    assert record["reference_kind"] == "visible_depth"
    assert record["reference_mesh"] is None
    assert record["object_class"] == "tshirt"
    folder = output / record["sample_id"]
    assert (folder / "foreground.png").is_file()
    assert not list(folder.glob("*.ply"))

    from asset2sim.evaluation.physx_omni.run import evaluate_run

    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text('{"samples": []}')
    evaluated = tmp_path / "evaluated"
    summary = evaluate_run(run, output / "manifest.jsonl", evaluated, sample_count=100)
    assert summary["categories"]["deformable"]["geometry_valid"] == 0
    assert "visible_depth_reference_only" in (evaluated / "geometry_metrics.csv").read_text()
