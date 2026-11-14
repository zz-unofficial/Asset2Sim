"""Reference freeze and model configuration boundaries without GPU or downloads."""

import json
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from asset2sim.data import generation_benchmark
from asset2sim.data.manifest import load_manifest
from asset2sim.generation.trellis2 import TRELLIS2Config


class FakeRenderer:
    def __init__(self, image_size):
        self.image_size = image_size

    def render(self, mesh, pose, yfov, mode):
        return Image.new("RGBA", (self.image_size, self.image_size), (255, 0, 0, 255))

    def close(self):
        pass


def test_freeze_reference_views_and_preserve_failed_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(generation_benchmark, "MeshRenderer", FakeRenderer)
    mesh_path = tmp_path / "box.glb"
    trimesh.creation.box().export(mesh_path)
    selection = tmp_path / "candidates.jsonl"
    candidates = [
        {
            "object_id": "box",
            "reference_mesh": "box.glb",
            "source_up_axis": "Z",
            "group": "convex",
            "second_view": True,
        },
        {
            "object_id": "missing",
            "reference_mesh": "missing.glb",
            "source_up_axis": "Z",
            "group": "ring",
            "second_view": False,
        },
    ]
    selection.write_text("\n".join(json.dumps(row) for row in candidates) + "\n")
    output = tmp_path / "prepared"
    manifest = tmp_path / "manifest.jsonl"
    result = generation_benchmark.prepare(selection, output, manifest)
    assert result == {"candidates": 2, "eligible_objects": 1, "input_views": 2}
    records = load_manifest(manifest)
    assert {row["sample_id"] for row in records} == {"box_v000", "box_v008"}
    for row in records:
        with open(row["render_metadata"]) as stream:
            metadata = json.load(stream)
        assert metadata["conditioning_view"] == int(row["view_id"])
        assert len(metadata["views"]) == 16
    audit = json.loads((output / "audit.json").read_text())
    assert audit[1]["status"] == "excluded"
    with pytest.raises(FileExistsError):
        generation_benchmark.prepare(selection, output, manifest)


def test_trellis_512_config_only():
    assert TRELLIS2Config().pipeline_type == "512"
    with pytest.raises(ValueError):
        TRELLIS2Config(pipeline_type="1024")


def test_gso_selection_is_fixed_32_and_diverse():
    rows = [
        json.loads(line)
        for line in Path("experiments/datasets/generation_gso32.jsonl").read_text().splitlines()
    ]
    from collections import Counter

    assert len(rows) == 32
    assert len({row["object_id"] for row in rows}) == 32
    assert set(Counter(row["group"] for row in rows).values()) == {4}
    assert len({row["group"] for row in rows}) == 8
    assert sum(row["second_view"] for row in rows) == 8
    assert all(row["source_version"] == "1" for row in rows)
    assert all("/1/" in row["download_url"] for row in rows)
    assert all(row["pbr_eligible"] is False for row in rows)


def test_supplied_specs_export_pbr_reference(tmp_path):
    from asset2sim.data.benchmark_synthetic import prepare

    specifications = [
        json.loads(line)
        for line in Path("experiments/datasets/controlled_topology_specs_8.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["candidate_id"] for row in specifications] == [f"P{i:02d}" for i in range(1, 9)]
    spec_file = tmp_path / "one.jsonl"
    spec_file.write_text(json.dumps(specifications[0]) + "\n")
    selection = tmp_path / "selection.jsonl"
    prepare(spec_file, tmp_path / "references", selection)
    record = json.loads(selection.read_text())
    assert record["candidate_id"] == "P01"
    assert record["pbr_eligible"] is True
    assert set(record["material_channels"]) == {"baseColor", "metallic", "roughness"}
    mesh = trimesh.load(tmp_path / "references/P01_open_sheet.glb", force="mesh", process=False)
    assert mesh.visual.material.baseColorTexture is not None
    assert mesh.visual.material.metallicRoughnessTexture is not None


def test_gso_download_extract_texture_prepare_and_cache(tmp_path, monkeypatch):
    import io
    import zipfile

    from asset2sim.data import paper

    monkeypatch.setattr(paper, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generation_benchmark, "MeshRenderer", FakeRenderer)
    downloads = []

    def download(url, path):
        if path.exists():
            return
        downloads.append(url)
        path.parent.mkdir(parents=True)
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "meshes/model.obj",
                "mtllib ../materials/material.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl default\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
            )
            archive.writestr(
                "materials/material.mtl", "newmtl default\nKd 1 1 1\nmap_Kd textures/texture.png\n"
            )
            archive.writestr("materials/textures/texture.png", buffer.getvalue())

    monkeypatch.setattr(paper, "download", download)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "object_id": "gso_fixture",
                "gso_name": "fixture",
                "download_url": "https://example.test/1/fixture.zip",
                "source_up_axis": "Z",
                "group": "compact",
                "second_view": True,
                "canonical_view": "largest_silhouette",
                "dataset": "gso",
                "license": "CC BY 4.0",
            }
        )
        + "\n"
    )
    for repeat in range(2):
        output = tmp_path / f"prepared{repeat}"
        manifest = tmp_path / f"manifest{repeat}.jsonl"
        result = generation_benchmark.prepare(selection, output, manifest)
        assert result["eligible_objects"] == 1
        assert result["input_views"] == 2
        assert (output / "reference_gallery.png").is_file()
        mesh = trimesh.load(output / "gso_fixture/reference.glb", force="mesh", process=False)
        assert mesh.visual.material.baseColorTexture is not None
    assert len(downloads) == 1


def test_abo_original_glb_y_up_pbr_and_blender_input(tmp_path, monkeypatch):
    import numpy as np

    from asset2sim.evaluation.geometry import scene_mesh

    module = generation_benchmark
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    mesh = trimesh.creation.box(extents=[0.2, 0.8, 0.3])
    texture = Image.new("RGB", (8, 8), "red")
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, metallicRoughnessTexture=texture
        ),
    )
    payload = mesh.export(file_type="glb")

    def download(_url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(module, "download", download)

    def render(command, **_kwargs):
        jobs = json.loads(Path(command[-1]).read_text())
        assert jobs[0]["mode"] == "pbr"
        for job in jobs:
            folder = Path(job["output"])
            folder.mkdir()
            for index in range(len(job["views"])):
                Image.new("RGBA", (32, 32), "red").save(folder / f"{index:03d}.png")

    monkeypatch.setattr(module.subprocess, "run", render)

    def no_egl(*_args):
        raise AssertionError("ABO PBR inputs should use Blender without an EGL renderer")

    monkeypatch.setattr(module, "MeshRenderer", no_egl)
    hdri = tmp_path / "light.exr"
    hdri.touch()
    profile = tmp_path / "render.yaml"
    profile.write_text(f"hdri: {hdri}\n")
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "object_id": "abo_fixture",
                "abo_model_id": "BTEST",
                "download_url": "https://example.test/BTEST.glb",
                "dataset": "abo",
                "source_up_axis": "Y",
                "group": "wood",
                "second_view": False,
                "canonical_view": 12,
                "pbr_eligible": True,
                "input_renderer": "blender_pbr",
            }
        )
        + "\n"
    )
    output = tmp_path / "prepared"
    manifest = tmp_path / "manifest.jsonl"
    result = module.prepare(selection, output, manifest, profile)
    assert result["eligible_objects"] == 1
    record = json.loads(manifest.read_text())
    metadata = json.loads((tmp_path / record["render_metadata"]).read_text())
    assert metadata["source_up_axis"] == "Y"
    assert metadata["reference_up_axis"] == "Z"
    assert np.asarray(metadata["source_to_reference"])[2, 1] > 0
    scene, prepared = scene_mesh(output / "abo_fixture/reference.glb")
    assert prepared.extents[2] == pytest.approx(1.0)
    for geometry in scene.geometry.values():
        assert geometry.visual.material.metallicRoughnessTexture is not None
    assert (tmp_path / "data/raw/abo/BTEST.glb").read_bytes() == payload


def test_abo_selection_has_six_verified_original_materials():
    rows = [
        json.loads(line)
        for line in Path("experiments/datasets/generation_abo6.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    assert len({row["group"] for row in rows}) == 6
    for row in rows:
        assert row["source_up_axis"] == "Y"
        assert row["download_url"].startswith(
            "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"
        )
        assert row["pbr_eligible"] is True
        for material in row["gltf_materials"]:
            assert "baseColorTexture" in material["pbrMetallicRoughness"]
            assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
