"""Reference freeze and model configuration boundaries without GPU or downloads."""

import json

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
