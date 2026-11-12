"""Geometry and appearance evaluation for fixed public object subsets."""

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
from asset2sim.data.manifest import load_manifest, select_samples
from asset2sim.data.rendering import MeshRenderer, composite_rgb
from asset2sim.evaluation.geometry import (
    align_surfaces,
    bounded_symmetric_icp,
    distance_summary,
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


def silhouette_metrics(predicted: Image.Image, reference: Image.Image) -> dict[str, float]:
    a = np.asarray(predicted.convert("RGBA"))[:, :, 3] > 127
    b = np.asarray(reference.convert("RGBA"))[:, :, 3] > 127
    union = a | b
    if not b.any():
        raise ValueError("Reference render has no foreground")
    rgb_a, rgb_b = composite_rgb(predicted), composite_rgb(reference)
    mse = float(np.mean(np.square(rgb_a[union] - rgb_b[union])))
    return {
        "mask_iou": float(np.count_nonzero(a & b) / np.count_nonzero(union)),
        "foreground_area_ratio": float(np.count_nonzero(a) / np.count_nonzero(b)),
        "psnr_foreground": -10 * np.log10(max(mse, 1e-12)),
        "reference_foreground_fraction": float(b.mean()),
    }


def aggregate_views(views: list[dict], conditioning: int) -> dict[str, float]:
    if not 0 <= conditioning < len(views) or len(views) < 2:
        raise ValueError("Need a conditioning view and at least one novel view")
    result = {}
    for name in views[0]:
        result[name] = float(np.mean([view[name] for view in views]))
        result[f"input_{name}"] = float(views[conditioning][name])
        result[f"novel_{name}"] = float(
            np.mean([view[name] for index, view in enumerate(views) if index != conditioning])
        )
    return result


def material_summary(mesh) -> dict:
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", getattr(material, "image", None))
    factor = getattr(material, "baseColorFactor", getattr(material, "diffuse", None))
    return {
        "visual_kind": mesh.visual.kind,
        "has_texture": texture is not None,
        "has_uv": getattr(mesh.visual, "uv", None) is not None,
        "base_color_factor": np.asarray(factor).tolist() if factor is not None else None,
        "metallic": getattr(material, "metallicFactor", None),
        "roughness": getattr(material, "roughnessFactor", None),
    }


def summarize_records(records: list[dict]) -> dict:
    """Aggregate each valid metric separately and retain its actual denominator."""
    result = {
        "planned": len(records),
        "successful": sum(row["status"] == "ok" for row in records),
        "geometry_successful": sum(row.get("geometry_status") == "ok" for row in records),
        "image_successful": sum(row.get("image_status") == "ok" for row in records),
        "metric_counts": {},
    }
    names = sorted(
        {
            key
            for row in records
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for name in names:
        values = [
            row[name]
            for row in records
            if isinstance(row.get(name), (int, float)) and np.isfinite(row[name])
        ]
        result[name] = float(np.mean(values)) if values else None
        result["metric_counts"][name] = len(values)
    return result


def render_object(
    renderer,
    metric,
    original,
    reference,
    render_config,
    transform,
    output: Path,
    modes: list[str],
    prefix: str = "",
) -> dict:
    mesh = original.copy()
    mesh.apply_transform(transform)
    conditioning = render_config["conditioning_view"]
    results = {}
    per_view = []
    for mode in modes:
        values = []
        prediction_folder = output if mode == "texture" else output / mode
        reference_folder = output / "reference" / mode
        prediction_folder.mkdir(parents=True, exist_ok=True)
        reference_folder.mkdir(parents=True, exist_ok=True)
        for index, view in enumerate(render_config["views"]):
            pose = np.asarray(view["camera_to_world"])
            predicted = renderer.render(mesh, pose, render_config["yfov_degrees"], mode=mode)
            target = renderer.render(reference, pose, render_config["yfov_degrees"], mode=mode)
            predicted.save(prediction_folder / f"{index:03d}.png")
            target.save(reference_folder / f"{index:03d}.png")
            measured = metric.measure(predicted, target)
            # A perfect reconstruction has a finite, explicitly capped PSNR.
            measured["psnr"] = min(float(measured["psnr"]), 120.0)
            measured.update(silhouette_metrics(predicted, target))
            values.append(measured)
            per_view.append(
                {
                    "view_id": f"{index:03d}",
                    "conditioning": index == conditioning,
                    "render_mode": mode,
                    **measured,
                }
            )
        mode_prefix = "" if mode == "texture" else f"{mode}_"
        results.update(
            {
                prefix + mode_prefix + key: value
                for key, value in aggregate_views(values, conditioning).items()
            }
        )
    write_json(output / "views.json", per_view)
    return results


def evaluate_run(run_dir: Path, config_path: Path, output: Path) -> dict:
    config = load_parameters(config_path)

    if config.get("image_alignment") != "rigid":
        raise ValueError("The primary image alignment must be rigid; bounded scale is diagnostic")
    modes = config.get("render_modes", ["texture", "pbr"])
    if not modes or modes[0] != "texture" or set(modes) - {"texture", "pbr"}:
        raise ValueError("render_modes must start with texture and may include pbr")
    run_dir, output = run_dir.resolve(), output.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    snapshot = run_dir / "inputs.jsonl"
    records = read_jsonl(snapshot) if snapshot.exists() else load_manifest(Path(run["manifest"]))
    by_id = {row["sample_id"]: row for row in records}
    selected = select_samples(
        [by_id[sample_id] for sample_id in run["samples"]], config.get("sample_ids")
    )
    generated = {row["sample_id"]: row for row in read_jsonl(run_dir / "results.jsonl")}
    output.mkdir(parents=True, exist_ok=False)
    (output / "transforms").mkdir()
    metadata = {
        "profile": config,
        "run": str(run_dir),
        "model": run["model"],
        "device": run.get("device"),
        "status": "running",
        "image_metrics_definition": "unlit stored texture RGB, unit-scale rigid alignment",
        "pbr_metrics_definition": "source-provided materials under identical fixed lighting",
        "scope": "fixed public subset; original conditioning inputs retained",
        "samples": [row["sample_id"] for row in selected],
        "dependencies": {
            name: version(name)
            for name in ("numpy", "trimesh", "scipy", "scikit-image", "lpips", "pyrender")
        },
    }
    write_json(output / "evaluation.json", metadata)
    renderer = None
    rows = []
    try:
        if config["image_metrics"]:
            image_metric = ImageMetrics(config["lpips_network"], config["cpu_threads"])
        with (output / "metrics.jsonl").open("x") as stream:
            for sample in selected:
                sample_id = sample["sample_id"]
                dataset = sample.get("dataset", "custom")
                category = sample.get("category")
                if category is None and dataset == "omniobject3d":
                    category = sample_id.split("_")[1]
                row = {
                    "sample_id": sample_id,
                    "object_id": sample["object_id"],
                    "dataset": dataset,
                    "category": category or "unlabeled",
                    "view_id": sample["view_id"],
                    "geometry_status": "pending",
                    "image_status": "pending" if config["image_metrics"] else "disabled",
                }
                try:
                    measured = generated.get(sample_id, {})
                    if measured.get("status") != "ok":
                        raise ValueError(measured.get("error", "Generation did not complete"))
                    start = time.perf_counter()
                    original = load_mesh(run_dir / measured["mesh"])
                    reference = load_mesh(Path(sample["reference_mesh"]))
                    predicted, pn = normalized_mesh(original, config["normalization_extent"])
                    target, rn = normalized_mesh(reference, config["normalization_extent"])
                    source_points = trimesh.sample.sample_surface(
                        predicted, config["surface_points"], seed=config["seed"]
                    )[0]
                    target_points = trimesh.sample.sample_surface(
                        target, config["surface_points"], seed=config["seed"] + 1
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
                    aligned = transform_points(source_points, alignment)
                    row.update(surface_metrics(aligned, target_points, config["fscore_thresholds"]))
                    row.update(distance_summary(aligned, target_points))
                    row.update(
                        geometry_status="ok",
                        geometry_seconds=time.perf_counter() - start,
                        alignment_candidates=candidates,
                        image_scale=1.0,
                        image_to_mesh_seconds=sum(
                            measured.get(key, 0)
                            for key in ("preprocess_seconds", "inference_seconds", "export_seconds")
                        ),
                        faces=measured.get("faces"),
                        peak_allocated_mib=measured.get("peak_allocated_mib"),
                    )
                    image_transform = np.linalg.inv(rn) @ alignment @ pn
                    transforms = {
                        "prediction_normalization": pn.tolist(),
                        "reference_normalization": rn.tolist(),
                        "geometry_alignment": alignment.tolist(),
                        "image_alignment_to_reference": image_transform.tolist(),
                        "image_relative_scale": 1.0,
                    }
                    write_json(output / "transforms" / f"{sample_id}.json", transforms)
                    if config["image_metrics"]:
                        render_config = json.loads(Path(sample["render_metadata"]).read_text())
                        if len(render_config["views"]) < config["minimum_render_views"]:
                            raise ValueError(
                                "Reference has fewer views than required by the profile"
                            )
                        size = render_config["image_size"]
                        if renderer is None:
                            renderer = MeshRenderer(size)
                            renderer_size = size
                        elif renderer_size != size:
                            raise ValueError("A single evaluation must use one image size")
                        folder = output / "renders" / sample_id
                        row.update(
                            render_object(
                                renderer,
                                image_metric,
                                original,
                                reference,
                                render_config,
                                image_transform,
                                folder,
                                modes,
                            )
                        )
                        write_json(
                            folder / "materials.json",
                            {
                                "reference": material_summary(reference),
                                "prediction": material_summary(original),
                            },
                        )
                        if config.get("diagnostic_bounded_scale"):
                            bounded, info = bounded_symmetric_icp(
                                fit_source,
                                fit_target,
                                alignment,
                                tuple(config["image_scale_bounds"]),
                                config["image_icp_iterations"],
                                config["icp_tolerance"],
                            )
                            row.update({f"bounded_{key}": value for key, value in info.items()})
                            bounded_transform = np.linalg.inv(rn) @ bounded @ pn
                            transforms["bounded_image_alignment_to_reference"] = (
                                bounded_transform.tolist()
                            )
                            row.update(
                                render_object(
                                    renderer,
                                    image_metric,
                                    original,
                                    reference,
                                    render_config,
                                    bounded_transform,
                                    folder / "bounded",
                                    ["texture"],
                                    "bounded_",
                                )
                            )
                            write_json(output / "transforms" / f"{sample_id}.json", transforms)
                        row.update(
                            image_status="ok",
                            evaluated_views=len(render_config["views"]),
                            novel_views=len(render_config["views"]) - 1,
                        )
                    row["status"] = "ok"
                    print(
                        f"{sample_id}: CD={row['chamfer_l1']:.5f}, FS@0.01={row['fscore@0.01']:.3f}",
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001 - retain all planned inputs
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    if row["geometry_status"] == "pending":
                        row["geometry_status"] = "failed"
                    if row["image_status"] == "pending":
                        row["image_status"] = "failed"
                    print(f"{sample_id}: {row['error']}", flush=True)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    except BaseException as error:
        metadata.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        write_json(output / "evaluation.json", metadata)
        raise
    finally:
        if renderer:
            renderer.close()
    metadata.update(
        status="completed"
        if all(row["status"] == "ok" for row in rows)
        else "completed_with_failures",
        datasets={
            dataset: summarize_records([row for row in rows if row["dataset"] == dataset])
            for dataset in sorted({row["dataset"] for row in rows})
        },
        categories={
            category: summarize_records([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
    )
    write_json(output / "evaluation.json", metadata)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return metadata
