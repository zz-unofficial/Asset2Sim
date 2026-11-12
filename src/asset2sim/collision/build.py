"""Construct collision components from a prepared, meter-scale triangle mesh."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree

from asset2sim.geometry.prepare import inspect_mesh, load_geometry, physical_status
from asset2sim.paths import PROJECT_ROOT

THRESHOLDS = {"C1": 0.10, "C2": 0.05, "C3": 0.02}
DEFAULT_CONFIG = PROJECT_ROOT / "configs/collision/default.yaml"


def _surface_metrics(
    source: trimesh.Trimesh, parts: list[trimesh.Trimesh], count: int, seed: int
) -> dict[str, Any]:
    # This concatenates triangle arrays for area-weighted sampling only: exported
    # colliders remain separate, and overlapping internal faces are included.
    surfaces = trimesh.util.concatenate(parts)
    source_points, _ = trimesh.sample.sample_surface(source, count, seed=seed)
    collision_points, _ = trimesh.sample.sample_surface(surfaces, count, seed=seed + 1)
    forward = cKDTree(collision_points).query(source_points)[0]
    reverse = cKDTree(source_points).query(collision_points)[0]

    def stats(distances: np.ndarray) -> dict[str, float]:
        return {
            "mean_m": float(np.mean(distances)),
            "p95_m": float(np.percentile(distances, 95)),
            "max_m": float(np.max(distances)),
        }

    return {
        "sampling": {
            "samples_per_side": count,
            "source_seed": seed,
            "component_surfaces_seed": seed + 1,
            "unit": "m",
        },
        "input_to_component_surfaces": stats(forward),
        "component_surfaces_to_input": stats(reverse),
        "symmetric_mean_m": float((forward.mean() + reverse.mean()) / 2),
        "interpretation": "Sample-to-sample surface distances; includes internal and overlapping "
        "component faces, not union boundary or physical contact error.",
    }


def _run_coacd(mesh: trimesh.Trimesh, settings: dict, timeout: float) -> list[trimesh.Trimesh]:
    with tempfile.TemporaryDirectory(prefix="asset2sim-coacd-") as tmp:
        directory = Path(tmp)
        np.savez_compressed(directory / "input.npz", vertices=mesh.vertices, faces=mesh.faces)
        (directory / "settings.json").write_text(json.dumps(settings))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "asset2sim.collision._worker", tmp],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"CoACD exceeded {timeout:g} seconds") from exc
        if result.returncode != 0 or not (directory / "output.npz").exists():
            details = (result.stderr or result.stdout)[-3000:]
            raise RuntimeError(f"CoACD worker failed (exit {result.returncode}): {details}")
        with np.load(directory / "output.npz") as data:
            return [
                trimesh.Trimesh(
                    vertices=data[f"vertices_{i}"], faces=data[f"faces_{i}"], process=False
                )
                for i in range(int(data["count"]))
            ]


def build_collision(
    mesh_path: Path,
    output: Path,
    condition: str,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_open: bool = False,
    assume_prepared: bool = False,
) -> dict[str, Any]:
    """Build components and record results, including failures, in a fresh directory.

    Inputs are Z-up, meter-scale meshes prepared by Task A. External prepared meshes
    require an explicit declaration via ``assume_prepared``.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "condition": condition,
        "input": str(Path(mesh_path).resolve()),
        "warnings": [],
        "components": [],
        "error": None,
        "unit": "m",
        "up_axis": "+Z",
    }
    try:
        if condition not in {"C0", *THRESHOLDS}:
            raise ValueError(f"Unknown collision condition: {condition}")
        config = yaml.safe_load(Path(config_path).read_text())
        count, seed = int(config["surface_samples"]), int(config["surface_seed"])
        timeout = float(config["timeout_seconds"])
        if count < 1 or seed < 0 or timeout <= 0:
            raise ValueError("Surface sample count and timeout must be positive; seed nonnegative")
        report["configuration"] = {
            "path": str(Path(config_path).resolve()),
            "surface_samples": count,
            "surface_seed": seed,
            "timeout_seconds": timeout,
            "allow_open": allow_open,
            "assume_prepared": assume_prepared,
            "coacd": config.get("coacd"),
        }
        report["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "trimesh": importlib.metadata.version("trimesh"),
            "scipy": importlib.metadata.version("scipy"),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        }
        metadata_path = Path(mesh_path).parent / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("status") != "completed"
                or metadata.get("physical_coordinates") != {"up_axis": "+Z", "units": "meters"}
                or Path(mesh_path).name != metadata.get("output")
            ):
                raise ValueError("Input is not the completed Z-up meter mesh declared by metadata")
            report["input_metadata"] = str(metadata_path.resolve())
        elif not assume_prepared:
            raise ValueError(
                "Missing Task A metadata; explicitly declare --assume-prepared for external meshes"
            )
        else:
            report["warnings"].append("External mesh declared as prepared Z-up meters by caller")
        mesh, _ = load_geometry(Path(mesh_path))
        diagnosis = inspect_mesh(mesh)
        report["input_diagnostic"] = diagnosis
        report["input_physical_status"] = physical_status(diagnosis)
        fatal = {
            "non_finite_vertices",
            "invalid_face_indices",
            "non_finite_triangle_area",
            "zero_extent",
            "degenerate_triangles",
            "duplicate_faces",
        }
        bad = sorted(set(diagnosis["issues"]) & fatal)
        if bad:
            raise ValueError(f"Invalid input geometry: {', '.join(bad)}; prepare explicitly first")
        unreliable = (
            not diagnosis["watertight"]
            or not diagnosis["winding_consistent"]
            or not diagnosis["valid_volume"]
            or diagnosis["nonmanifold_edges"] > 0
        )
        if unreliable and not allow_open:
            raise ValueError(
                "Input is open, non-manifold or lacks a reliable volume; "
                "repair explicitly or opt in to open-input handling"
            )
        if unreliable:
            report["warnings"].append(
                "Input has unreliable topology or volume; results require review"
            )
        if diagnosis["connected_components"] > 1:
            report["warnings"].append("Multiple input components: convex geometry may bridge gaps")
        if diagnosis["unreferenced_vertices"]:
            raise ValueError(
                "Input has unreferenced vertices; explicitly remove them in prepare-mesh"
            )
        stage = time.perf_counter()
        if condition == "C0":
            report["method"] = {
                "name": "single_convex_hull",
                "algorithm": "scipy Qhull via trimesh",
            }
            parts = [mesh.convex_hull]
        else:
            import coacd

            parameters = inspect.signature(coacd.run_coacd).parameters
            settings = {
                name: parameter.default
                for name, parameter in parameters.items()
                if name not in {"mesh", "threshold"}
            }
            overrides = dict(config["coacd"])
            unknown = set(overrides) - set(settings)
            if unknown:
                raise ValueError(f"Unknown CoACD settings: {sorted(unknown)}")
            settings.update(overrides)
            settings["threshold"] = THRESHOLDS[condition]
            if settings.get("real_metric") is not False or settings.get("apx_mode") != "ch":
                raise ValueError(
                    "Conditions C1-C3 require normalized threshold and convex hull approximation"
                )
            if settings.get("preprocess_mode") not in {"auto", "on", "off"}:
                raise ValueError("Invalid CoACD preprocess_mode")
            if unreliable and settings["preprocess_mode"] != "on":
                raise ValueError("Open-input CoACD requires preprocess_mode: on in configuration")
            report["method"] = {
                "name": "coacd",
                "version": importlib.metadata.version("coacd"),
                "settings": settings,
                "preprocessing": "CoACD internal preprocessing: on and auto may alter "
                "input geometry; actual execution is not exposed by the API",
            }
            parts = _run_coacd(mesh, settings, timeout)
        report["build_seconds"] = time.perf_counter() - stage
        if not parts:
            raise RuntimeError("Decomposition produced no collision components")
        stage = time.perf_counter()
        for i, part in enumerate(parts):
            if (
                len(part.faces) == 0
                or not np.isfinite(part.vertices).all()
                or not part.is_volume
                or not part.is_convex
            ):
                raise ValueError(f"Collision component {i} is not a valid convex volume")
            name = f"collision_{i:03d}.obj"
            part.export(output / name)
            report["components"].append(
                {
                    "path": name,
                    "vertices": len(part.vertices),
                    "faces": len(part.faces),
                    "volume_m3": float(part.volume),
                    "aabb_m": part.bounds.tolist(),
                    "convex": True,
                }
            )
        report["component_count"] = len(parts)
        report["total_vertices"] = sum(len(part.vertices) for part in parts)
        report["total_faces"] = sum(len(part.faces) for part in parts)
        report["export_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        report["surface_error"] = _surface_metrics(mesh, parts, count, seed)
        report["surface_metrics_seconds"] = time.perf_counter() - stage
        report["status"] = "completed"
    except Exception as error:  # noqa: BLE001 - retain a failure record for third-party errors
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (output / "colliders.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] == "failed":
        raise RuntimeError(
            f"Collision construction failed: {report['error']} (see {output / 'colliders.json'})"
        )
    return report
