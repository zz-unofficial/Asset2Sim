import json
from types import SimpleNamespace

import torch
import pytest

from asset2sim.generation.trellis2_artifacts import save_native_asset


def test_native_asset_preserves_material_and_geometry(tmp_path):
    mesh = SimpleNamespace(
        vertices=torch.ones(3, 3),
        faces=torch.tensor([[0, 1, 2]]),
        attrs=torch.arange(6).reshape(1, 6).float(),
        coords=torch.tensor([[1, 2, 3]]),
        voxel_size=1 / 512,
        layout={"base_color": slice(0, 3), "roughness": slice(4, 5)},
    )
    save_native_asset(mesh, tmp_path, {"texture_size": 4096})
    data = torch.load(tmp_path / "native_asset.pt", weights_only=True)
    for name in ("vertices", "faces", "attrs", "coords"):
        assert torch.equal(data[name], getattr(mesh, name))
    assert data["voxel_size"].shape == (3,)
    metadata = json.loads((tmp_path / "native_asset.json").read_text())
    assert metadata["attr_layout"]["base_color"] == [0, 3, None]
    assert metadata["parameters"]["texture_size"] == 4096
    mesh.attrs[0, 0] = float("nan")
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(ValueError, match="Non-finite"):
        save_native_asset(mesh, invalid, {})
    assert not (invalid / "native_asset.pt").exists()


def test_export_run_keeps_identity_material_condition_and_failures(tmp_path, monkeypatch):
    from asset2sim.generation import trellis2_artifacts as module

    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text(
        json.dumps({"model": "trellis2", "status": "generated_pending_export"})
    )
    (source / "inputs.jsonl").write_text('{"sample_id":"good","object_id":"cup","view_id":"000"}\n')
    (source / "good").mkdir()
    (source / "good/input.png").write_bytes(b"input")
    rows = [
        {
            "sample_id": "good",
            "object_id": "cup",
            "view_id": "000",
            "status": "export_pending",
            "raw_asset": "good/native_asset.pt",
            "inference_seconds": 2,
        },
        {"sample_id": "bad", "status": "failed", "error": "generation failed"},
    ]
    (source / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fake_export(asset, output, overrides):
        output.mkdir()
        (output / "mesh.glb").write_bytes(b"mesh")
        return {"export_seconds": 3, "export_parameters": overrides}

    monkeypatch.setattr(module, "export_native_asset", fake_export)
    output = tmp_path / "exported"
    monkeypatch.setattr("sys.argv", ["export", "--run", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 1
    result = [json.loads(s) for s in (output / "results.jsonl").read_text().splitlines()]
    assert result[0]["status"] == "ok" and result[0]["object_id"] == "cup"
    assert result[0]["inference_seconds"] == 2 and result[0]["export_seconds"] == 3
    assert result[1]["status"] == "failed"
    assert (output / "inputs.jsonl").read_text() == (source / "inputs.jsonl").read_text()


def test_official_export_rotation_has_declared_inverse():
    import numpy as np
    import yaml
    from asset2sim.paths import PROJECT_ROOT

    mapping = yaml.safe_load(
        (PROJECT_ROOT / "configs/evaluation/generation_benchmark.yaml").read_text()
    )
    inverse = np.asarray(mapping["export_to_canonical"]["trellis2"])
    official_export = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    assert np.allclose(inverse @ official_export, np.eye(4))
