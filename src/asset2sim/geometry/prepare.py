"""Prepare a mesh for physics without altering its source asset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from asset2sim.artifacts import read_jsonl, write_json


@dataclass(frozen=True)
class PreparationOptions:
    kind: str
    target_extent_m: float = 0.20
    remove_degenerate: bool = False
    remove_unreferenced: bool = False
    min_triangle_area_m2: float = 0.0
    merge_coincident: bool = False

    def validate(self) -> None:
        if self.kind not in {"reference", "generated"}:
            raise ValueError("kind must be reference or generated")
        if not np.isfinite(self.target_extent_m) or self.target_extent_m <= 0:
            raise ValueError("target_extent_m must be finite and positive")
        if not np.isfinite(self.min_triangle_area_m2) or self.min_triangle_area_m2 < 0:
            raise ValueError("min_triangle_area_m2 must be finite and non-negative")


def load_geometry(path: Path) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Flatten every scene instance using its complete node-to-world transform."""
    if path.suffix.lower() not in {".glb", ".obj", ".stl"}:
        raise ValueError("Supported formats are GLB, OBJ and STL")
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise TypeError("Input did not load as a scene")
    nodes = []
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry = scene.graph.get(node)
        if not isinstance(scene.geometry[geometry], trimesh.Trimesh):
            raise TypeError(f"Scene node {node} is not a triangle mesh")
        if not np.isfinite(transform).all():
            raise ValueError(f"Scene node {node} has a non-finite transform")
        nodes.append({"node": node, "geometry": geometry, "node_to_scene": transform.tolist()})
    if not nodes:
        raise ValueError("Input scene contains no mesh instances")
    # Trimesh applies hierarchy transforms, including repeated instances and reflections.
    mesh = scene.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Input scene contains no triangle faces")
    return mesh, nodes


def inspect_mesh(
    mesh: trimesh.Trimesh, min_area: float = 0.0, diagnose_coincident: bool = True
) -> dict[str, Any]:
    """Report physical suitability; volume is trusted only for a closed oriented solid."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    finite = bool(np.isfinite(vertices).all())
    report: dict[str, Any] = {
        "vertices": len(vertices),
        "faces": len(faces),
        "finite_vertices": finite,
        "aabb": None,
        "longest_extent": None,
        "degenerate_faces": None,
        "connected_components": None,
        "watertight": None,
        "winding_consistent": None,
        "signed_volume": None,
        "valid_volume": False,
        "issues": [],
        "area_threshold": min_area,
        "unreferenced_vertices": None,
    }
    if not finite:
        report["issues"].append("non_finite_vertices")
        return report
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        report["issues"].append("invalid_face_indices")
        return report
    report["unreferenced_vertices"] = int(len(vertices) - len(np.unique(faces)))
    if report["unreferenced_vertices"]:
        report["issues"].append("unreferenced_vertices")
    bounds = mesh.bounds
    report["aabb"] = bounds.tolist()
    report["longest_extent"] = float(mesh.extents.max())
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all():
        report["issues"].append("non_finite_triangle_area")
        return report
    report["degenerate_faces"] = int(np.count_nonzero(areas <= min_area))
    report["connected_components"] = len(
        trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=1
        )
    )
    report["watertight"] = bool(mesh.is_watertight)
    report["winding_consistent"] = bool(mesh.is_winding_consistent)
    volume = float(mesh.volume)
    report["signed_volume"] = volume if np.isfinite(volume) else None
    report["valid_volume"] = bool(mesh.is_volume and np.isfinite(volume) and volume > 0)
    if report["degenerate_faces"]:
        report["issues"].append("degenerate_triangles")
    if report["connected_components"] > 1:
        report["issues"].append("multiple_components")
    if not report["watertight"]:
        report["issues"].append("open_or_non_manifold_surface")
    if not report["winding_consistent"]:
        report["issues"].append("inconsistent_winding")
    if not report["valid_volume"]:
        report["issues"].append("invalid_or_unreliable_volume")
    if report["longest_extent"] <= 0:
        report["issues"].append("zero_extent")
    if diagnose_coincident:
        vertices_unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
        report["coincident_vertices"] = int(len(vertices) - len(vertices_unique))
        if report["coincident_vertices"]:
            # Diagnostic only: separate touching objects can also share coordinates.
            probe = trimesh.Trimesh(vertices=vertices_unique, faces=inverse[faces], process=False)
            report["position_merged_diagnostic"] = inspect_mesh(probe, min_area, False)
            report["issues"].append("coincident_vertices")
        else:
            report["position_merged_diagnostic"] = None
    return report


def generated_coordinates(source: Path, run_dir: Path) -> tuple[np.ndarray, dict]:
    """Only accept a generator transform from the run that actually produced this mesh."""
    run_dir = run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    matches = [
        r
        for r in read_jsonl(run_dir / "results.jsonl")
        if r.get("mesh") and (run_dir / r["mesh"]).resolve() == source.resolve()
    ]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        raise ValueError("Mesh must be a successful output recorded in the supplied run")
    coordinates = run.get("raw_mesh_coordinates") or {}
    if not coordinates.get("up_axis") or not coordinates.get("units"):
        raise ValueError("Run must declare raw up_axis and units")
    matrix = np.asarray(coordinates.get("raw_to_model_z_up"), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Run must declare a finite 4x4 raw_to_model_z_up matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, 3], 0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("raw_to_model_z_up must be a proper rotation without scale or translation")
    return matrix, {
        "run": str(run_dir),
        "sample_id": matches[0]["sample_id"],
        "model": run.get("model"),
        "raw_coordinates": coordinates,
    }


def prepare_mesh(
    source: Path, output: Path, options: PreparationOptions, run_dir: Path | None = None
) -> dict:
    """Write normalized.glb and audit records to a fresh directory, including on failure."""
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recorded_options = asdict(options)
    for key, value in recorded_options.items():
        if isinstance(value, float) and not np.isfinite(value):
            recorded_options[key] = str(value)
    metadata: dict[str, Any] = {
        "status": "running",
        "source": str(source),
        "options": recorded_options,
        "operations": [],
        "physical_coordinates": {"up_axis": "+Z", "units": "meters"},
    }
    report: dict[str, Any] = {"status": "running", "before": None, "after": None}
    try:
        options.validate()
        if options.kind == "reference" and run_dir is not None:
            raise ValueError("Reference meshes do not use a generator run")
        if options.kind == "generated" and run_dir is None:
            raise ValueError("Generated meshes require their generator run directory")
        mesh, nodes = load_geometry(source)
        metadata["scene_nodes"] = nodes
        metadata["operations"].append({"operation": "flatten_scene", "instances": len(nodes)})
        report["before"] = inspect_mesh(mesh)
        if not report["before"]["finite_vertices"]:
            raise ValueError("Mesh contains non-finite vertices")
        if "invalid_face_indices" in report["before"]["issues"]:
            raise ValueError("Mesh contains out-of-range face indices")
        if "non_finite_triangle_area" in report["before"]["issues"]:
            raise ValueError("Mesh contains non-finite triangle areas")
        if report["before"]["longest_extent"] <= 0:
            raise ValueError("Mesh has zero extent")
        if options.kind == "reference":
            metadata["raw_coordinates"] = {
                "up_axis": "+Z",
                "units": "meters",
                "declared_by": "reference input",
            }
            transform = np.eye(4)
            if not np.isclose(
                report["before"]["longest_extent"], options.target_extent_m, rtol=1e-5, atol=1e-8
            ):
                raise ValueError(
                    f"Reference longest edge does not match {options.target_extent_m:g} m target"
                )
            metadata["operations"].append(
                {"operation": "preserve_reference_coordinates", "matrix": transform.tolist()}
            )
        else:
            rotation, provenance = generated_coordinates(source, run_dir)
            metadata.update(provenance)
            mesh.apply_transform(rotation)
            metadata["operations"].append(
                {"operation": "generator_to_z_up", "matrix": rotation.tolist()}
            )
            extent = float(mesh.extents.max())
            if not np.isfinite(extent) or extent <= 0:
                raise ValueError("Generated mesh has zero or invalid extent")
            scale = options.target_extent_m / extent
            normalize = np.eye(4)
            normalize[:3, :3] *= scale
            normalize[:3, 3] = -mesh.bounds.mean(axis=0) * scale
            mesh.apply_transform(normalize)
            metadata["operations"].append(
                {"operation": "center_and_scale", "scale": scale, "matrix": normalize.tolist()}
            )
            transform = normalize @ rotation
        metadata["scene_to_physical"] = transform.tolist()
        for node in nodes:
            node["node_to_physical"] = (transform @ np.asarray(node["node_to_scene"])).tolist()
        metadata["alignment"] = "none; no ground-truth geometry used"
        report["before_cleaning"] = inspect_mesh(mesh, options.min_triangle_area_m2)
        if options.merge_coincident:
            vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
            removed = int(len(mesh.vertices) - len(vertices))
            mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
            metadata["operations"].append(
                {
                    "operation": "merge_coincident_vertices",
                    "removed": removed,
                    "tolerance": 0.0,
                    "visuals_preserved": False,
                }
            )
        if options.remove_degenerate:
            keep = np.asarray(mesh.area_faces) > options.min_triangle_area_m2
            removed = int((~keep).sum())
            mesh.update_faces(keep)
            metadata["operations"].append(
                {
                    "operation": "remove_degenerate_faces",
                    "removed": removed,
                    "area_threshold_m2": options.min_triangle_area_m2,
                }
            )
        if options.remove_unreferenced:
            count = len(mesh.vertices)
            mesh.remove_unreferenced_vertices()
            metadata["operations"].append(
                {"operation": "remove_unreferenced_vertices", "removed": count - len(mesh.vertices)}
            )
        if not len(mesh.faces) or not len(mesh.vertices):
            raise ValueError("No triangles remain after cleaning")
        report["after"] = inspect_mesh(mesh, options.min_triangle_area_m2)
        mesh.export(output / "normalized.glb")
        metadata.update(status="completed", output="normalized.glb")
        report["status"] = "completed"
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        metadata.update(status="failed", error=message)
        report.update(status="failed", error=message)
        raise
    finally:
        write_json(output / "metadata.json", metadata)
        write_json(output / "mesh_report.json", report)
    return metadata
