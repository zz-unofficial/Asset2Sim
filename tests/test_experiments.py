import csv
import json

import pytest

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest, load_manifest
from asset2sim.experiments.config import ExperimentConfig, load_parameters
from asset2sim.experiments.runner import run_generation
from asset2sim.experiments.summary import export_runs, run_rows
from asset2sim.generation.registry import create_generator


class FakeGenerator:
    model_name = "fake"

    def __init__(self, load_error=False):
        self.parameters = {"seed": 3}
        self.load_error = load_error
        self.loaded = False
        self.recovered = 0

    def load(self):
        if self.load_error:
            raise RuntimeError("weights unavailable")
        self.loaded = True

    def metadata(self):
        return {"device": "cpu", "weights_revision": "fixture"}

    def generate(self, image_path, output_dir):
        output_dir.mkdir()
        if image_path.stem == "bad":
            raise ValueError("invalid input")
        (output_dir / "mesh.glb").write_bytes(b"fixture")
        return {"faces": 12, "vertices": 8, "inference_seconds": 0.25}

    def recover(self):
        self.recovered += 1


def dataset(tmp_path):
    records = []
    for sample_id in ("good", "bad"):
        image = tmp_path / f"{sample_id}.png"
        image.touch()
        records.append(
            {
                "sample_id": sample_id,
                "object_id": "one_object",
                "view_id": sample_id,
                "split": "debug",
                "image": image.name,
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_runner_keeps_failed_inputs_and_identity_in_exports(tmp_path):
    manifest = dataset(tmp_path)
    model = FakeGenerator()
    output = tmp_path / "run"
    assert run_generation(model, manifest, output, experiment={"name": "pair"}) == 1
    assert model.recovered == 1
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "completed_with_failures"
    assert run["config"] == {"seed": 3}
    assert run["samples"] == ["good", "bad"]
    rows = run_rows(output)
    assert [row["status"] for row in rows] == ["ok", "failed"]
    assert {row["object_id"] for row in rows} == {"one_object"}
    assert rows[1]["error"] == "ValueError: invalid input"
    target = tmp_path / "metrics.csv"
    assert export_runs([output], target) == 2
    with target.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with pytest.raises(FileExistsError):
        export_runs([output], target)


def test_load_failure_still_exports_all_planned_inputs(tmp_path):
    manifest = dataset(tmp_path)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="weights unavailable"):
        run_generation(FakeGenerator(load_error=True), manifest, output)
    rows = run_rows(output)
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"load_failed"}
    assert all("weights unavailable" in row["error"] for row in rows)


def test_interrupted_run_retains_its_full_input_set(tmp_path):
    class InterruptedGenerator(FakeGenerator):
        def generate(self, image_path, output_dir):
            raise KeyboardInterrupt

    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_generation(InterruptedGenerator(), dataset(tmp_path), output)
    rows = run_rows(output)
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"interrupted"}


def test_selection_is_validated_before_creating_run_or_loading_model(tmp_path):
    model = FakeGenerator()
    with pytest.raises(ValueError, match="Unknown"):
        run_generation(model, dataset(tmp_path), tmp_path / "run", ["missing"])
    assert not model.loaded
    assert not (tmp_path / "run").exists()


def test_experiment_resolves_relative_paths_and_preserves_defaults(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models/base.yaml").write_text("seed: 0\ntexture_resolution: 1024\n")
    (tmp_path / "models/model.yaml").write_text("extends: base.yaml\nseed: 1\n")
    (tmp_path / "experiment.yaml").write_text(
        "name: pair\nmodel: sf3d\nmodel_config: models/model.yaml\n"
        "manifest: manifest.jsonl\noverrides:\n  seed: 3\n"
    )
    monkeypatch.chdir(tmp_path.parent)
    config = ExperimentConfig.load(tmp_path / "experiment.yaml")
    assert config.manifest == tmp_path / "manifest.jsonl"
    assert config.parameters() == {"seed": 3, "texture_resolution": 1024}
    assert load_parameters(tmp_path / "models/model.yaml")["seed"] == 1


def test_config_inheritance_reports_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_parameters(tmp_path / "a.yaml")


def test_published_manifest_keeps_input_paths_after_relocation(tmp_path):
    original = load_manifest(dataset(tmp_path))
    target = tmp_path / "shared/debug.jsonl"
    export_manifest(original, target)
    assert load_manifest(target) == original


def test_generation_manifest_can_reference_geometry_on_the_evaluation_machine(tmp_path):
    path = dataset(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["reference_mesh"] = "evaluation/reference.glb"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    resolved = load_manifest(path)
    assert resolved[0]["reference_mesh"] == str(tmp_path / "evaluation/reference.glb")


def test_summary_reads_existing_run_format_without_modifying_it(tmp_path):
    manifest = dataset(tmp_path)
    output = tmp_path / "legacy"
    output.mkdir()
    write_json(
        output / "run.json",
        {
            "run_id": "legacy",
            "status": "completed",
            "manifest": str(manifest),
            "samples": ["good"],
            "config": {"model": "official/model", "seed": 0},
        },
    )
    (output / "results.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "good",
                "status": "ok",
                "faces": 12,
                "mesh": "good/mesh.glb",
            }
        )
        + "\n"
    )
    before = (output / "run.json").read_text()
    row = run_rows(output)[0]
    assert (row["object_id"], row["view_id"], row["faces"]) == ("one_object", "good", 12)
    assert row["model"] == "official/model"
    assert (output / "run.json").read_text() == before


def test_unknown_model_has_an_actionable_error():
    with pytest.raises(ValueError, match="available models: sf3d"):
        create_generator("not_registered", {})
