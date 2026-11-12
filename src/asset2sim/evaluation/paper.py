"""Measure generated assets using the SF3D paper's metric families."""

from __future__ import annotations

import csv
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from skimage.metrics import structural_similarity

from asset2sim.artifacts import read_jsonl, write_json
from asset2sim.data.manifest import load_manifest
from asset2sim.data.rendering import MeshRenderer, composite_rgb
from asset2sim.evaluation.geometry import (
    align_surfaces,
    icp,
    load_mesh,
    normalized_mesh,
    surface_metrics,
    transform_points,
)
from asset2sim.experiments.config import load_parameters
from asset2sim.paths import PROJECT_ROOT


class ImageMetrics:
    def __init__(self, network: str, threads: int):
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".local/torch"))
        import lpips
        import torch

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = lpips.LPIPS(net=network).eval()

    def measure(self, predicted: Image.Image, reference: Image.Image) -> dict[str, float]:
        x, y = composite_rgb(predicted), composite_rgb(reference)
        mse = float(np.mean((x - y) ** 2))
        tx = self.torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)
        ty = self.torch.from_numpy(y.transpose(2, 0, 1).copy()).unsqueeze(0)
        with self.torch.no_grad():
            perceptual = float(self.model(tx, ty, normalize=True).item())
        return {
            "psnr": -10 * np.log10(mse) if mse > 0 else float("inf"),
            "ssim": float(
                structural_similarity(
                    x,
                    y,
                    channel_axis=-1,
                    data_range=1,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
            ),
            "lpips": perceptual,
        }


def evaluate_run(run_dir: Path, config_path: Path, output: Path) -> dict:
    config = load_parameters(config_path)
    run_dir, output = run_dir.resolve(), output.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    snapshot = run_dir / "inputs.jsonl"
    inputs = read_jsonl(snapshot) if snapshot.exists() else load_manifest(Path(run["manifest"]))
    inputs = {record["sample_id"]: record for record in inputs}
    generated = {record["sample_id"]: record for record in read_jsonl(run_dir / "results.jsonl")}
    output.mkdir(parents=True, exist_ok=False)
    (output / "transforms").mkdir()
    metadata = {
        "profile": config,
        "run": str(run_dir),
        "model": run.get("model", "sf3d"),
        "device": run.get("device"),
        "status": "running",
        "scope": "public subset with explicit implementation parameters",
        "dependencies": {
            name: version(name) for name in ("numpy", "trimesh", "scipy", "scikit-image")
        },
    }
    write_json(output / "evaluation.json", metadata)
    renderer = MeshRenderer() if config["image_metrics"] else None
    image_metric = (
        ImageMetrics(config["lpips_network"], config["cpu_threads"]) if renderer else None
    )
    records = []
    try:
        with (output / "metrics.jsonl").open("x") as stream:
            for sample_id in run["samples"]:
                sample = inputs[sample_id]
                measured = generated.get(sample_id, {})
                row = {
                    key: sample.get(key) for key in ("sample_id", "object_id", "dataset", "view_id")
                }
                row["dataset"] = row["dataset"] or "custom"
                try:
                    if measured.get("status") != "ok":
                        raise ValueError(measured.get("error", "Generation did not complete"))
                    start = time.perf_counter()
                    original = load_mesh(run_dir / measured["mesh"])
                    reference = load_mesh(Path(sample["reference_mesh"]))
                    predicted, predicted_norm = normalized_mesh(
                        original, config["normalization_extent"]
                    )
                    reference_norm, reference_transform = normalized_mesh(
                        reference, config["normalization_extent"]
                    )
                    source_points = trimesh.sample.sample_surface(
                        predicted, config["surface_points"], seed=config["seed"]
                    )[0]
                    target_points = trimesh.sample.sample_surface(
                        reference_norm, config["surface_points"], seed=config["seed"] + 1
                    )[0]
                    count = config["alignment_points"]
                    fit_source, fit_target = source_points[:count], target_points[:count]
                    alignment, candidates = align_surfaces(
                        fit_source,
                        fit_target,
                        config["rotation_step_degrees"],
                        config["icp_iterations"],
                        config["icp_tolerance"],
                    )
                    row.update(
                        surface_metrics(
                            transform_points(source_points, alignment),
                            target_points,
                            config["fscore_thresholds"],
                        )
                    )
                    row.update(
                        alignment_candidates=candidates,
                        geometry_seconds=time.perf_counter() - start,
                        image_to_mesh_seconds=sum(
                            measured.get(key, 0)
                            for key in ("preprocess_seconds", "inference_seconds", "export_seconds")
                        ),
                        peak_allocated_mib=measured.get("peak_allocated_mib"),
                        faces=measured.get("faces"),
                    )
                    transforms = {
                        "prediction_normalization": predicted_norm.tolist(),
                        "reference_normalization": reference_transform.tolist(),
                        "geometry_alignment": alignment.tolist(),
                    }
                    if renderer and sample.get("render_metadata"):
                        render_path = Path(sample["render_metadata"])
                        render_config = json.loads(render_path.read_text())
                        refined = icp(
                            fit_source,
                            fit_target,
                            alignment,
                            config["image_icp_iterations"],
                            config["icp_tolerance"],
                            scale=True,
                        )
                        image_transform = (
                            np.linalg.inv(reference_transform) @ refined @ predicted_norm
                        )
                        image_mesh = original.copy()
                        image_mesh.apply_transform(image_transform)
                        transforms["image_alignment_to_reference"] = image_transform.tolist()
                        view_metrics = []
                        rendered = output / "renders" / sample_id
                        rendered.mkdir(parents=True)
                        for index, view in enumerate(render_config["views"]):
                            image = renderer.render(
                                image_mesh,
                                np.asarray(view["camera_to_world"]),
                                render_config["yfov_degrees"],
                            )
                            image.save(rendered / f"{index:03d}.png")
                            with Image.open(render_path.parent / view["image"]) as ground_truth:
                                view_metrics.append(image_metric.measure(image, ground_truth))
                        write_json(
                            rendered / "views.json",
                            [
                                {"view_id": f"{index:03d}", **values}
                                for index, values in enumerate(view_metrics)
                            ],
                        )
                        row.update(
                            {
                                name: float(np.mean([v[name] for v in view_metrics]))
                                for name in ("psnr", "ssim", "lpips")
                            }
                        )
                        row["evaluated_views"] = len(view_metrics)
                    write_json(output / "transforms" / f"{sample_id}.json", transforms)
                    row["status"] = "ok"
                    print(f"{sample_id}: CD={row['chamfer_l1']:.5f}", flush=True)
                except Exception as error:  # noqa: BLE001 - retain the full evaluation denominator
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    print(f"{sample_id}: {row['error']}", flush=True)
                records.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    finally:
        if renderer:
            renderer.close()
    groups = {}
    metric_names = [
        "chamfer_l1",
        "chamfer_squared",
        "fscore@0.1",
        "fscore@0.2",
        "fscore@0.5",
        "psnr",
        "ssim",
        "lpips",
        "image_to_mesh_seconds",
        "peak_allocated_mib",
        "faces",
    ]
    for dataset in sorted({row["dataset"] for row in records}):
        subset = [row for row in records if row["dataset"] == dataset]
        successful = [row for row in subset if row["status"] == "ok"]
        groups[dataset] = {"planned": len(subset), "successful": len(successful)}
        for name in metric_names:
            values = [row[name] for row in successful if row.get(name) is not None]
            groups[dataset][name] = float(np.mean(values)) if values else None
    metadata.update(
        status="completed"
        if all(row["status"] == "ok" for row in records)
        else "completed_with_failures",
        datasets=groups,
    )
    write_json(output / "evaluation.json", metadata)
    columns = list(dict.fromkeys(key for record in records for key in record))
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return metadata
