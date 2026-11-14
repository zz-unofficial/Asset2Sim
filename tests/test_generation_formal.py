import csv
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from asset2sim.data.benchmark_synthetic import create_shape
from asset2sim.data.synthetic import camera_pose
from asset2sim.evaluation import benchmark
from asset2sim.experiments.runner import run_generation


@pytest.mark.parametrize("images", [False, True])
def test_formal_keeps_missing_models_and_two_tracks(tmp_path, monkeypatch, images):
    if images:

        class Metric:
            def __init__(self, *_args):
                pass

            def measure(self, a, b):
                assert np.array_equal(np.asarray(a), np.asarray(b))
                return {"psnr": float("inf"), "ssim": 1.0, "lpips": 0.0}

        def render(command, **_kwargs):
            jobs = json.loads(Path(command[-1]).read_text())
            for job in jobs:
                folder = Path(job["output"])
                folder.mkdir(parents=True)
                for index, _ in enumerate(job["views"]):
                    Image.new("RGBA", (32, 32), "red").save(folder / f"{index:03d}.png")

        monkeypatch.setattr(benchmark, "ImageMetrics", Metric)
        monkeypatch.setattr(benchmark.subprocess, "run", render)
    mesh = trimesh.creation.box(extents=[1, 0.6, 0.3])
    reference = tmp_path / "reference.glb"
    mesh.export(reference)
    Image.new("RGBA", (32, 32), "red").save(tmp_path / "input.png")
    camera = {
        "image_size": 32,
        "yfov_degrees": 40,
        "conditioning_view": 0,
        "views": [{"camera_to_world": camera_pose(i * 22.5, 20, 2.5).tolist()} for i in range(16)],
    }
    (tmp_path / "render.json").write_text(json.dumps(camera))
    record = {
        "sample_id": "box",
        "object_id": "box",
        "view_id": "000",
        "image": "input.png",
        "reference_mesh": "reference.glb",
        "render_metadata": "render.json",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"model": "sf3d", "config": {"seed": 0}}))
    (run / "inputs.jsonl").write_text(json.dumps(record) + "\n")
    mesh.export(run / "mesh.glb")
    (run / "results.jsonl").write_text(
        json.dumps({"sample_id": "box", "status": "ok", "mesh": "mesh.glb"}) + "\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "image_metrics": images,
                "cpu_threads": 1,
                "surface_points": 200,
                "alignment_points": 100,
                "rotation_step_degrees": 90,
                "icp_iterations": 2,
                "icp_tolerance": 1e-6,
                "seed": 0,
                "export_to_canonical": {"sf3d": np.eye(4).tolist()},
            }
        )
    )
    output = tmp_path / "evaluation"
    benchmark.evaluate(manifest, {"sf3d": run, "spar3d": tmp_path / "missing"}, output, config)
    with (output / "per_sample.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert sum(row["geometry_status"] == "ok" for row in rows) == 2
    assert sum(row.get("failed_stage") == "generation" for row in rows) == 2
    if images:
        assert sum(row["image_status"] == "ok" for row in rows) == 2
        assert all(
            float(row["texture_mask_iou"]) == 1 for row in rows if row["image_status"] == "ok"
        )
        with (output / "per_view.csv").open() as stream:
            view_rows = list(csv.DictReader(stream))
        assert len(view_rows) == 32
        assert sum(row["conditioning"] == "True" for row in view_rows) == 2
    assert json.loads((output / "summary.json").read_text())["planned_objects"] == 1
    with pytest.raises(FileExistsError):
        benchmark.evaluate(manifest, {"sf3d": run}, output, config)


def test_object_pairing_does_not_count_extra_views_as_objects():
    rows = []
    for model, score in [("a", 0.1), ("b", 0.3)]:
        for object_id, views in [("one", 2), ("two", 1)]:
            for _ in range(views):
                rows.append(
                    {
                        "model": model,
                        "object_id": object_id,
                        "track": "export_pose",
                        "geometry_status": "ok",
                        "image_status": "ok",
                        "chamfer_l1": score,
                    }
                )
    objects, paired = benchmark.aggregate(rows)
    assert len(objects) == 4
    assert paired[0]["n_paired_objects"] == 2
    assert paired[0]["median_difference"] == pytest.approx(-0.2)


def test_topology_reports_seams_and_open_surface():
    original = trimesh.creation.box()
    duplicated = trimesh.Trimesh(
        vertices=original.vertices[original.faces].reshape(-1, 3),
        faces=np.arange(36).reshape(-1, 3),
        process=False,
    )
    report = benchmark.topology(duplicated)
    assert not report["raw"]["watertight"]
    assert report["position_welded_diagnostic"]["watertight"]
    plane = create_shape({"shape": "plane"})
    assert benchmark.topology(plane)["raw"]["genus"] is None


def test_controlled_specs_generate_valid_surfaces():
    specs = Path("experiments/datasets/controlled_topology_specs_8.jsonl")
    for line in specs.read_text().splitlines():
        mesh = create_shape(json.loads(line))
        assert mesh.area > 0
        assert np.isfinite(mesh.vertices).all()


def test_warm_repeats_are_all_recorded_and_excluded_from_quality(tmp_path):
    class Generator:
        model_name = "fake"

        @property
        def parameters(self):
            return {}

        def load(self):
            pass

        def metadata(self):
            return {}

        def recover(self):
            pass

        def generate(self, image_path, output_dir):
            output_dir.mkdir(parents=True)
            (output_dir / "mesh.glb").touch()
            return {"faces": 12, "inference_seconds": 0.25}

    (tmp_path / "input.png").touch()
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(json.dumps({"sample_id": "one", "image": "input.png"}) + "\n")
    output = tmp_path / "run"
    assert run_generation(Generator(), manifest, output, warm_repeats=3) == 0
    timing = json.loads((output / "timing.json").read_text())
    assert len(timing) == 4
    assert sum(not row["warmup"] for row in timing) == 3
    assert len((output / "results.jsonl").read_text().splitlines()) == 1


def test_freeze_rejects_changed_model_defaults(tmp_path, monkeypatch):
    import sys

    from asset2sim.experiments import benchmark as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "configs/models"
    folder.mkdir(parents=True)
    for model in runner.MODELS:
        (folder / f"{model}.yaml").write_text("seed: 0\n")
    for name in ("image.png", "reference.glb", "render.json"):
        (tmp_path / name).touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "one",
                "image": "image.png",
                "reference_mesh": "reference.glb",
                "render_metadata": "render.json",
                "license": "CC0",
                "category": "convex",
            }
        )
        + "\n"
    )
    config = tmp_path / "evaluation.yaml"
    config.write_text("name: frozen\n")
    freeze = tmp_path / "freeze.json"
    command = [
        "benchmark",
        "freeze",
        "one",
        "--manifest",
        str(manifest),
        "--config",
        str(config),
        "--freeze",
        str(freeze),
    ]
    monkeypatch.setattr(sys, "argv", command)
    runner.main()
    assert freeze.is_file()
    (folder / "sf3d.yaml").write_text("seed: 4\n")
    command[1] = "formal"
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(ValueError, match="frozen protocol"):
        runner.main()
