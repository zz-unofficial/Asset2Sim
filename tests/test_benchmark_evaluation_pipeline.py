import csv
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from asset2sim.evaluation import benchmark
from asset2sim.paths import PROJECT_ROOT


def test_complete_evaluation_reuses_reference_and_keeps_coverage(tmp_path, monkeypatch):
    reference = tmp_path / "reference.glb"
    trimesh.creation.box().export(reference)
    image = tmp_path / "input.png"
    Image.new("RGBA", (64, 64), (120, 100, 80, 255)).save(image)
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            {
                "views": [{"camera_to_world": np.eye(4).tolist()} for _ in range(16)],
                "image_size": 64,
                "yfov_degrees": 40,
                "conditioning_view": 0,
            }
        )
    )
    sample = {
        "sample_id": "box_v000",
        "object_id": "box",
        "view_id": "000",
        "split": "test",
        "image": str(image),
        "reference_mesh": str(reference),
        "render_metadata": str(cameras),
        "dataset": "test",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample) + "\n")
    runs = {}
    for model in ("sf3d", "spar3d"):
        run = tmp_path / model
        (run / "box_v000").mkdir(parents=True)
        trimesh.creation.box().export(run / "box_v000/mesh.glb")
        (run / "run.json").write_text(json.dumps({"model": model, "config": {"seed": 0}}))
        (run / "inputs.jsonl").write_text(json.dumps(sample) + "\n")
        (run / "results.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": "box_v000",
                    "status": "ok",
                    "mesh": "box_v000/mesh.glb",
                    "inference_seconds": 2,
                }
            )
            + "\n"
        )
        runs[model] = run

    class Metrics:
        def __init__(self, *args):
            pass

        def measure(self, *args):
            return {"psnr": 120.0, "ssim": 1.0, "lpips": 0.0}

    rendered = []

    def render(command, **kwargs):
        jobs = json.loads(Path(command[-1]).read_text())
        rendered.extend(jobs)
        for job in jobs:
            folder = Path(job["output"])
            folder.mkdir(parents=True)
            for i in range(len(job["views"])):
                Image.new("RGBA", (64, 64), (120, 100, 80, 255)).save(folder / f"{i:03d}.png")

    monkeypatch.setattr(benchmark, "ImageMetrics", Metrics)
    monkeypatch.setattr(benchmark.subprocess, "run", render)
    monkeypatch.setattr(benchmark, "align_surfaces", lambda *args: (np.eye(4), {}))
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/evaluation/generation_benchmark.yaml").read_text()
    )
    config.update(
        surface_points=50, alignment_points=20, cpu_threads=1, reuse_completed_output=True
    )
    config_file = tmp_path / "evaluation.yaml"
    config_file.write_text(yaml.safe_dump(config))
    output = tmp_path / "evaluation"
    benchmark.evaluate(manifest, runs, output, config_file)
    rows = list(csv.DictReader((output / "per_sample.csv").open()))
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    assert len([j for j in rendered if j["mesh"].endswith("reference.glb")]) == 1
    assert len([j for j in rendered if j["mesh"].endswith("prediction.glb")]) == 4
    coverage = list(csv.DictReader((output / "coverage.csv").open()))
    assert all(
        r["planned_inputs"] == r["geometry_valid_inputs"] == r["image_valid_inputs"] == "1"
        for r in coverage
    )
    assert len(list(csv.DictReader((output / "generation_costs.csv").open()))) == 2
    original_summary = (output / "summary.json").read_bytes()
    job_count = len(rendered)
    benchmark.evaluate(manifest, runs, output, config_file)
    assert len(rendered) == job_count
    assert (output / "summary.json").read_bytes() == original_summary
    partial = tmp_path / "aligned_checkpoint"
    partial.mkdir()
    (partial / "protocol.json").write_bytes((output / "protocol.json").read_bytes())
    (partial / "assets").symlink_to(output / "assets", target_is_directory=True)
    for filename in ("progress.jsonl", "view_progress.jsonl"):
        records = [json.loads(line) for line in (output / filename).read_text().splitlines()]
        (partial / filename).write_text(
            "".join(json.dumps(r) + "\n" for r in records if r["track"] == "aligned_diagnostic")
        )
    benchmark.evaluate(manifest, runs, tmp_path / "camera_review", config_file, checkpoint=partial)
    assert len([j for j in rendered if j["mesh"].endswith("reference.glb")]) == 1
    assert len(rendered) == job_count + 2  # Only the two prediction tracks are rerendered.
    job_count = len(rendered)
    resumed = tmp_path / "resumed"
    benchmark.evaluate(manifest, runs, resumed, config_file, checkpoint=output)
    assert len(rendered) == job_count
    assert json.loads((resumed / "summary.json").read_text())["rows"] == 4
    from asset2sim.evaluation.benchmark_parallel import merge

    combined = tmp_path / "combined"
    merge(manifest, runs, [resumed], combined)
    assert json.loads((combined / "summary.json").read_text())["rows"] == 4
    with pytest.raises(ValueError, match="duplicate"):
        merge(manifest, runs, [resumed, resumed], tmp_path / "duplicate")
    with pytest.raises(ValueError, match="Missing"):
        merge(manifest, runs, [], tmp_path / "missing")
    config["surface_points"] = 60
    config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(FileExistsError):
        benchmark.evaluate(manifest, runs, output, config_file)
    assert (output / "paired.csv").is_file()


def test_paired_bootstrap_receives_sorted_object_differences(monkeypatch):
    observed = []
    monkeypatch.setattr(
        benchmark, "bootstrap_difference", lambda values: observed.append(values) or {}
    )
    rows = [
        {
            "model": model,
            "object_id": obj,
            "track": "aligned_diagnostic",
            "generation_chamfer_l1": value if model == "a" else 0.0,
        }
        for model in ["a", "b"]
        for obj, value in [("z", 3.0), ("a", 1.0), ("m", 2.0)]
    ]
    benchmark.aggregate(rows)
    assert observed == [[1.0, 2.0, 3.0]]
