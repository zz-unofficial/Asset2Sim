"""Paired image-to-3D benchmark using shared geometry and image metrics."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

from asset2sim.artifacts import read_jsonl, write_json
from asset2sim.data.manifest import load_manifest
from asset2sim.data.synthetic import camera_pose
from asset2sim.evaluation.geometry import (
    align_surfaces,
    distance_summary,
    normalized_mesh,
    scene_mesh,
    surface_metrics,
    transform_points,
)
from asset2sim.evaluation.paper import ImageMetrics, silhouette_metrics
from asset2sim.experiments.config import load_parameters


def topology(mesh: trimesh.Trimesh) -> dict:
    """Raw and position-welded diagnostics in unit-longest-side coordinates."""

    def report(value):
        edges = np.bincount(value.edges_unique_inverse)
        return {
            "vertices": len(value.vertices),
            "faces": len(value.faces),
            "components": len(value.split(only_watertight=False)),
            "boundary_edges": int(np.count_nonzero(edges == 1)),
            "nonmanifold_edges": int(np.count_nonzero(edges > 2)),
            "watertight": bool(value.is_watertight),
            "winding_consistent": bool(value.is_winding_consistent),
            "euler": int(value.euler_number),
            "genus": None,
            "genus_status": "not_reported_without_vertex_manifold_proof",
        }

    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=8)
    welded.remove_unreferenced_vertices()
    return {
        "raw": report(mesh),
        "position_welded_diagnostic": report(welded),
        "weld_tolerance_normalized": 1e-8,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def opening_metrics(predicted: Image.Image, reference: Image.Image) -> dict:
    """Projected silhouette holes; areas are pixels, ratios are dimensionless."""
    masks = [np.asarray(im.convert("RGBA"))[:, :, 3] > 127 for im in (predicted, reference)]
    holes = [binary_fill_holes(mask) & ~mask for mask in masks]
    union = holes[0] | holes[1]
    return {
        "opening_reference_pixels": int(holes[1].sum()),
        "opening_iou": float((holes[0] & holes[1]).sum() / union.sum()) if union.any() else None,
        "opening_recall": float((holes[0] & holes[1]).sum() / holes[1].sum())
        if holes[1].any()
        else None,
    }


def bootstrap_difference(values: list[float], seed: int = 0, repeats: int = 2000) -> dict:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, (repeats, len(values))), axis=1)
    return {
        "median_difference": float(np.median(values)),
        "ci95_low": float(np.quantile(medians, 0.025)),
        "ci95_high": float(np.quantile(medians, 0.975)),
        "n_paired_objects": len(values),
    }


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Equal weight per input within object, then paired object-level differences."""
    groups = {}
    for row in rows:
        key = (row["model"], row["object_id"], row["track"])
        groups.setdefault(key, []).append(row)
    objects = []
    for (model, object_id, track), samples in groups.items():
        result = {
            "model": model,
            "object_id": object_id,
            "track": track,
            "dataset": samples[0].get("dataset") or "custom",
            "category": samples[0].get("category") or "unlabeled",
            "n_planned_views": len(samples),
            "n_geometry_valid": sum(row.get("geometry_status") == "ok" for row in samples),
            "n_image_valid": sum(row.get("image_status") == "ok" for row in samples),
        }
        metrics = {
            key
            for row in samples
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for metric in sorted(metrics):
            valid = [
                row[metric]
                for row in samples
                if isinstance(row.get(metric), (int, float)) and np.isfinite(row[metric])
            ]
            result[metric] = float(np.mean(valid)) if valid else None
            result[f"{metric}_n"] = len(valid)
        objects.append(result)
    paired = []
    for track in sorted({row["track"] for row in rows}):
        for a, b in combinations(sorted({row["model"] for row in rows}), 2):
            left = {
                row["object_id"]: row
                for row in objects
                if row["track"] == track and row["model"] == a
            }
            right = {
                row["object_id"]: row
                for row in objects
                if row["track"] == track and row["model"] == b
            }
            for metric in (
                "chamfer_l1",
                "hausdorff95",
                "fscore@0.01",
                "fscore@0.02",
                "fscore@0.05",
                "texture_novel_lpips",
                "texture_novel_mask_iou",
                "pbr_novel_lpips",
                "pbr_novel_psnr",
                "pbr_novel_ssim",
            ):
                values = [
                    left[obj][metric] - right[obj][metric]
                    for obj in left.keys() & right.keys()
                    if left[obj].get(metric) is not None
                    and right[obj].get(metric) is not None
                    and left[obj].get(f"{metric}_n") == left[obj]["n_planned_views"]
                    and right[obj].get(f"{metric}_n") == right[obj]["n_planned_views"]
                ]
                if values:
                    paired.append(
                        {
                            "track": track,
                            "a_minus_b": f"{a}-{b}",
                            "metric": metric,
                            **bootstrap_difference(values),
                        }
                    )
    return objects, paired


def evaluate(manifest: Path, runs: dict[str, Path], output: Path, config_path: Path) -> None:
    """Evaluate all planned inputs; normalized distances, seconds and MiB are recorded."""
    config = load_parameters(config_path)
    samples = load_manifest(manifest)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "config": config,
            "manifest": str(manifest.resolve()),
            "runs": {k: str(v.resolve()) for k, v in runs.items()},
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "python": sys.version,
            "bootstrap": "2000 object-level resamples, seed=0; differences are a-b",
        },
    )
    metric = None
    metric_error = None
    if config.get("image_metrics", True):
        try:
            metric = ImageMetrics(config.get("lpips_network", "alex"), config["cpu_threads"])
        except Exception as error:  # noqa: BLE001 - keep geometry if LPIPS setup fails
            metric_error = f"{type(error).__name__}: {error}"
    rows, views, failures = [], [], []
    audits = []
    for path in {sample["reference_audit"] for sample in samples if sample.get("reference_audit")}:
        audits.extend(json.loads(Path(path).read_text()))
    write_csv(output / "reference_audit.csv", audits)
    try:
        for model, run_dir in runs.items():
            run = (
                json.loads((run_dir / "run.json").read_text())
                if (run_dir / "run.json").exists()
                else {}
            )
            generated = (
                {row["sample_id"]: row for row in read_jsonl(run_dir / "results.jsonl")}
                if (run_dir / "results.jsonl").exists()
                else {}
            )
            inputs = (
                {row["sample_id"]: row for row in read_jsonl(run_dir / "inputs.jsonl")}
                if (run_dir / "inputs.jsonl").exists()
                else {}
            )
            for sample in samples:
                sid = sample["sample_id"]
                measured = generated.get(sid, {})
                base = {
                    "model": model,
                    "sample_id": sid,
                    "object_id": sample["object_id"],
                    "view_id": sample["view_id"],
                    "category": sample.get("category"),
                    "dataset": sample.get("dataset", "custom"),
                    "pbr_eligible": sample.get("pbr_eligible", False),
                    "generation_status": measured.get("status", run.get("status", "missing_run")),
                    "run": str(run_dir),
                    "seed": run.get("config", {}).get("seed"),
                }
                for track in ("export_pose", "aligned_diagnostic"):
                    row = base | {
                        "track": track,
                        "geometry_status": "pending",
                        "image_status": "pending"
                        if config.get("image_metrics", True)
                        else "disabled",
                    }
                    stage = "generation"
                    try:
                        if measured.get("status") != "ok":
                            raise ValueError(
                                measured.get("error", run.get("error", "Missing generation result"))
                            )
                        if run.get("model") != model:
                            raise ValueError("Run model differs from requested source")
                        snapshot = inputs.get(sid, {})
                        if (
                            snapshot.get("object_id") != sample["object_id"]
                            or snapshot.get("view_id") != sample["view_id"]
                        ):
                            raise ValueError("Run input identity differs from frozen manifest")
                        stage = "geometry"
                        scene, mesh = scene_mesh(run_dir / measured["mesh"])
                        reference_scene, reference = scene_mesh(Path(sample["reference_mesh"]))
                        predicted, pn = normalized_mesh(mesh, 1.0)
                        target, rn = normalized_mesh(reference, 1.0)
                        rng_seed = config["seed"]
                        p = trimesh.sample.sample_surface(
                            predicted, config["surface_points"], seed=rng_seed
                        )[0]
                        q = trimesh.sample.sample_surface(
                            target, config["surface_points"], seed=rng_seed + 1
                        )[0]
                        camera = json.loads(Path(sample["render_metadata"]).read_text())
                        if len(camera["views"]) < config.get("minimum_render_views", 16):
                            raise ValueError("Reference has fewer camera views than required")
                        if track == "aligned_diagnostic":
                            count = config["alignment_points"]
                            alignment, _ = align_surfaces(
                                p[:count],
                                q[:count],
                                config["rotation_step_degrees"],
                                config["icp_iterations"],
                                config["icp_tolerance"],
                            )
                        else:
                            alignment = np.asarray(
                                config["export_to_canonical"][model], dtype=float
                            )
                            if (
                                alignment.shape != (4, 4)
                                or not np.allclose(
                                    alignment[:3, :3].T @ alignment[:3, :3], np.eye(3)
                                )
                                or np.linalg.det(alignment[:3, :3]) < 0
                            ):
                                raise ValueError("Export mapping must be a proper rigid rotation")
                            # Source camera, never reference geometry, determines the input-view swap.
                            view = np.asarray(
                                camera["views"][camera["conditioning_view"]]["camera_to_world"]
                            )
                            canonical = camera_pose(0, 20, 2.5)
                            rotation = np.eye(4)
                            rotation[:3, :3] = view[:3, :3] @ canonical[:3, :3].T
                            alignment = rotation @ alignment
                        row.update(
                            surface_metrics(
                                transform_points(p, alignment),
                                q,
                                config.get("fscore_thresholds", [0.01, 0.02, 0.05]),
                            )
                        )
                        row.update(distance_summary(transform_points(p, alignment), q))
                        row["geometry_status"] = "ok"
                        folder = output / "assets" / model / sid / track
                        folder.mkdir(parents=True)
                        diagnostics = {
                            "prediction": topology(predicted),
                            "reference": topology(target),
                        }
                        native = (run_dir / measured["mesh"]).parent / "native_geometry.npz"
                        if native.exists():
                            with np.load(native) as data:
                                native_mesh = trimesh.Trimesh(
                                    vertices=data["vertices"], faces=data["faces"], process=False
                                )
                                native_mesh, _ = normalized_mesh(native_mesh, 1)
                                diagnostics["native_prediction"] = topology(native_mesh)
                        write_json(folder / "topology.json", diagnostics)
                        for representation, info in diagnostics.items():
                            for method in ("raw", "position_welded_diagnostic"):
                                row.update(
                                    {
                                        f"{representation}_{method}_{k}": v
                                        for k, v in info[method].items()
                                    }
                                )
                        row["components"] = len(predicted.split(only_watertight=False))
                        row["faces"] = len(mesh.faces)
                        write_json(
                            folder / "transforms.json",
                            {
                                "prediction_normalization": pn.tolist(),
                                "reference_normalization": rn.tolist(),
                                "rotation": alignment.tolist(),
                                "uses_reference_for_pose": track == "aligned_diagnostic",
                            },
                        )
                        scene.apply_transform(alignment @ pn)
                        reference_scene.apply_transform(rn)
                        scene.export(folder / "prediction.glb")
                        reference_scene.export(folder / "reference.glb")
                        for key in (
                            "preprocess_seconds",
                            "inference_seconds",
                            "export_seconds",
                            "peak_allocated_mib",
                            "peak_reserved_mib",
                        ):
                            row[key] = measured.get(key)
                        if metric_error:
                            stage = "image_metric_initialization"
                            raise RuntimeError(metric_error)
                        if metric is not None:
                            stage = "render"
                            jobs = []
                            modes = ["texture"]
                            if sample.get("pbr_eligible", False):
                                modes.append("pbr")
                            for mode in modes:
                                for name in ("reference", "prediction"):
                                    jobs.append(
                                        {
                                            "mesh": str(folder / f"{name}.glb"),
                                            "output": str(folder / mode / name),
                                            "mode": mode,
                                            "image_size": camera["image_size"],
                                            "yfov_degrees": camera["yfov_degrees"],
                                            "views": camera["views"],
                                            "hdri": str(Path(config["hdri"]).resolve())
                                            if config.get("hdri")
                                            else None,
                                            "samples": config.get("render_samples", 32),
                                        }
                                    )
                            write_json(folder / "render_jobs.json", jobs)
                            with (folder / "blender.log").open("x") as log:
                                try:
                                    subprocess.run(
                                        [
                                            config.get("blender", "blender"),
                                            "-b",
                                            "--python",
                                            str(Path(__file__).with_name("benchmark_render.py")),
                                            "--",
                                            "--jobs",
                                            str(folder / "render_jobs.json"),
                                        ],
                                        check=True,
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                    )
                                except (subprocess.CalledProcessError, OSError) as error:
                                    row["render_error"] = str(error)
                            stage = "render" if row.get("render_error") else "image_metrics"
                            for mode in modes:
                                mode_views = []
                                for index in range(len(camera["views"])):
                                    with (
                                        Image.open(
                                            folder / mode / "prediction" / f"{index:03d}.png"
                                        ) as a,
                                        Image.open(
                                            folder / mode / "reference" / f"{index:03d}.png"
                                        ) as b,
                                    ):
                                        values = metric.measure(a, b) | silhouette_metrics(a, b)
                                        union = (np.asarray(a)[:, :, 3] > 127) | (
                                            np.asarray(b)[:, :, 3] > 127
                                        )
                                        if union.any():
                                            yy, xx = np.where(union)
                                            crop = (
                                                int(xx.min()),
                                                int(yy.min()),
                                                int(xx.max()) + 1,
                                                int(yy.max()) + 1,
                                            )
                                            if min(crop[2] - crop[0], crop[3] - crop[1]) >= 32:
                                                masked = metric.measure(a.crop(crop), b.crop(crop))
                                                values.update(
                                                    {
                                                        f"{k}_union_crop": v
                                                        for k, v in masked.items()
                                                    }
                                                )
                                        if sample.get("dataset") == "controlled_topology":
                                            openings = opening_metrics(a, b)
                                            values.update(
                                                {k: v for k, v in openings.items() if v is not None}
                                            )
                                    for name in list(values):
                                        if name.startswith("psnr"):
                                            values[name] = min(values[name], 120.0)
                                    view_row = (
                                        base
                                        | {
                                            "track": track,
                                            "render_mode": mode,
                                            "camera_id": index,
                                            "conditioning": index == camera["conditioning_view"],
                                        }
                                        | values
                                    )
                                    views.append(view_row)
                                    mode_views.append(values)
                                for name in {key for v in mode_views for key in v}:
                                    valid = [v[name] for v in mode_views if name in v]
                                    novel = [
                                        v[name]
                                        for i, v in enumerate(mode_views)
                                        if i != camera["conditioning_view"] and name in v
                                    ]
                                    row[f"{mode}_{name}"] = float(np.mean(valid))
                                    row[f"{mode}_{name}_valid_cameras"] = len(valid)
                                    row[f"{mode}_input_{name}"] = mode_views[
                                        camera["conditioning_view"]
                                    ].get(name)
                                    row[f"{mode}_novel_{name}"] = (
                                        float(np.mean(novel)) if novel else None
                                    )
                            if row.get("render_error"):
                                raise RuntimeError(row["render_error"])
                            row["image_status"] = "ok"
                        row["status"] = "ok"
                    except Exception as error:  # noqa: BLE001 - preserve every planned result
                        row.update(
                            status="failed",
                            failed_stage=stage,
                            error=f"{type(error).__name__}: {error}",
                        )
                        if row["geometry_status"] == "pending":
                            row["geometry_status"] = "failed"
                        if row["image_status"] == "pending":
                            row["image_status"] = "failed"
                        failures.append(dict(row))
                    rows.append(row)
                    with (output / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
    finally:
        write_csv(output / "per_sample.csv", rows)
        write_csv(output / "per_view.csv", views)
        write_csv(output / "failures.csv", failures)
    objects, paired = aggregate(rows)
    robustness = []
    for model in runs:
        for object_id in sorted({s["object_id"] for s in samples}):
            for track in ("export_pose", "aligned_diagnostic"):
                selected = sorted(
                    [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["object_id"] == object_id
                        and r["track"] == track
                    ],
                    key=lambda r: r["view_id"],
                )
                if len(selected) == 2:
                    record = {
                        "model": model,
                        "object_id": object_id,
                        "track": track,
                        "view_a": selected[0]["view_id"],
                        "view_b": selected[1]["view_id"],
                        "valid_geometry_views": sum(r["geometry_status"] == "ok" for r in selected),
                    }
                    for name in (
                        "chamfer_l1",
                        "fscore@0.01",
                        "texture_novel_lpips",
                        "texture_novel_mask_iou",
                    ):
                        if all(r.get(name) is not None for r in selected):
                            record[f"{name}_b_minus_a"] = selected[1][name] - selected[0][name]
                    robustness.append(record)
    write_csv(output / "robustness.csv", robustness)
    write_csv(output / "per_object.csv", objects)
    write_csv(output / "paired.csv", paired)
    timings = []
    for model, run_dir in runs.items():
        path = run_dir / "timing.json"
        if path.exists():
            timings.extend({"model": model, **row} for row in json.loads(path.read_text()))
    write_csv(output / "timing.csv", timings)
    timing_summary = []
    for model, run_dir in runs.items():
        metadata_path = run_dir / "run.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        valid = [
            row
            for row in timings
            if row["model"] == model and not row["warmup"] and row["status"] == "ok"
        ]
        summary = {
            "model": model,
            "planned_warm_repeats": metadata.get("warm_repeats", 0),
            "valid_warm_repeats": len(valid),
            "load_seconds": metadata.get("model_load_seconds"),
            "device": metadata.get("device"),
            "cpu": metadata.get("cpu"),
            "torch": metadata.get("torch"),
            "cuda": metadata.get("cuda"),
        }
        for name in (
            "preprocess_seconds",
            "inference_seconds",
            "export_seconds",
            "native_archive_seconds",
            "peak_allocated_mib",
            "peak_reserved_mib",
        ):
            values = [row[name] for row in valid if isinstance(row.get(name), (int, float))]
            summary[name] = float(np.mean(values)) if values else None
            summary[f"{name}_n"] = len(values)
        timing_summary.append(summary)
    write_csv(output / "timing_summary.csv", timing_summary)
    main_metrics = (
        "chamfer_l1",
        "hausdorff95",
        "fscore@0.01",
        "fscore@0.02",
        "fscore@0.05",
        "texture_novel_lpips",
        "texture_novel_mask_iou",
        "pbr_novel_lpips",
        "pbr_novel_psnr",
        "pbr_novel_ssim",
    )
    medians = []
    for model in runs:
        for track in ("export_pose", "aligned_diagnostic"):
            selected = [r for r in objects if r["model"] == model and r["track"] == track]
            for name in main_metrics:
                eligible_ids = {
                    s["object_id"]
                    for s in samples
                    if not name.startswith("pbr_") or s.get("pbr_eligible", False)
                }
                eligible = [r for r in selected if r["object_id"] in eligible_ids]
                valid = [
                    r[name]
                    for r in eligible
                    if r.get(name) is not None and r.get(f"{name}_n") == r["n_planned_views"]
                ]
                medians.append(
                    {
                        "model": model,
                        "track": track,
                        "metric": name,
                        "median": float(np.median(valid)) if valid else None,
                        "n_valid_objects": len(valid),
                        "n_planned_objects": len(eligible),
                    }
                )
    write_csv(output / "medians.csv", medians)
    group_medians = []
    for model, track, dataset, category in sorted(
        {(r["model"], r["track"], r["dataset"], r["category"]) for r in objects}
    ):
        selected = [
            r
            for r in objects
            if (r["model"], r["track"], r["dataset"], r["category"])
            == (model, track, dataset, category)
        ]
        for metric_name in main_metrics:
            eligible_ids = {
                s["object_id"]
                for s in samples
                if not metric_name.startswith("pbr_") or s.get("pbr_eligible", False)
            }
            eligible = [r for r in selected if r["object_id"] in eligible_ids]
            valid = [
                r[metric_name]
                for r in eligible
                if r.get(metric_name) is not None
                and r.get(f"{metric_name}_n") == r["n_planned_views"]
            ]
            group_medians.append(
                {
                    "model": model,
                    "track": track,
                    "dataset": dataset,
                    "category": category,
                    "metric": metric_name,
                    "median": float(np.median(valid)) if valid else None,
                    "n_valid_objects": len(valid),
                    "n_planned_objects": len(eligible),
                }
            )
    write_csv(output / "groups.csv", group_medians)
    write_json(
        output / "summary.json",
        {
            "planned_inputs_per_model": len(samples),
            "planned_objects": len({s["object_id"] for s in samples}),
            "rows": len(rows),
            "failed_rows": len(failures),
            "reference_candidates": len(audits),
            "reference_eligible": sum(r.get("status") == "eligible" for r in audits),
            "reference_excluded": sum(r.get("status") == "excluded" for r in audits),
            "paired": paired,
            "medians": medians,
        },
    )
    with (output / "tables.md").open("x") as stream:
        stream.write(
            "# Image-to-3D benchmark\n\n| Model | Track | Valid geometry / planned | Valid images / planned |\n| --- | --- | --- | --- |\n"
        )
        for model in runs:
            for track in ("export_pose", "aligned_diagnostic"):
                selected = [r for r in rows if r["model"] == model and r["track"] == track]
                stream.write(
                    f"| {model} | {track} | {sum(r['geometry_status'] == 'ok' for r in selected)}/{len(selected)} | {sum(r['image_status'] == 'ok' for r in selected)}/{len(selected)} |\n"
                )
        stream.write(
            "\n| Model | Track | Metric | Object median | Valid / planned objects |\n| --- | --- | --- | --- | --- |\n"
        )
        for row in medians:
            stream.write(
                f"| {row['model']} | {row['track']} | {row['metric']} | {row['median']} | {row['n_valid_objects']}/{row['n_planned_objects']} |\n"
            )
        stream.write(
            "\n| Pair (a-b) | Track | Metric | Median difference | 95% bootstrap interval | Paired objects |\n| --- | --- | --- | --- | --- | --- |\n"
        )
        for row in paired:
            stream.write(
                f"| {row['a_minus_b']} | {row['track']} | {row['metric']} | {row['median_difference']:.6g} | [{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | {row['n_paired_objects']} |\n"
            )
    gallery = Image.new("RGB", (512 * 4, 150 * len(samples)), "white")
    draw = ImageDraw.Draw(gallery)
    for y, sample in enumerate(samples):
        draw.text((0, y * 150), sample["sample_id"], fill="black")
        for x, model in enumerate(["reference", *runs]):
            first = next(iter(runs)) if model == "reference" else model
            name = "reference" if model == "reference" else "prediction"
            image = (
                output
                / "assets"
                / first
                / sample["sample_id"]
                / "export_pose"
                / "texture"
                / name
                / "000.png"
            )
            if image.exists():
                with Image.open(image) as im:
                    thumbnail = im.convert("RGB")
                    thumbnail.thumbnail((240, 120))
                    gallery.paste(thumbnail, (x * 512, y * 150 + 20))
            draw.text((x * 512, y * 150 + 8), model, fill="black")
    gallery.save(output / "qualitative.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="model=run_directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(value.split("=", 1) for value in args.run)
    evaluate(args.manifest, {k: Path(v) for k, v in runs.items()}, args.output, args.config)


if __name__ == "__main__":
    main()
