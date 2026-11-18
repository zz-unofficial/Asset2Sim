"""Prepare selected local physical-asset references without downloading datasets."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest, load_manifest, select_samples

CATEGORIES = ("rigid", "articulated", "deformable")


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Unsafe sample ID: {value!r}")
    return value


def prepare_rgb(source: Path, target: Path) -> dict[str, Any]:
    """Letterbox an image; preserve aspect ratio and record source-to-input pixels."""
    with Image.open(source) as original:
        rgba = original.convert("RGBA")
        width, height = rgba.size
        ratio = min(512 / width, 512 / height)
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        left, top = (512 - size[0]) // 2, (512 - size[1]) // 2
        canvas = Image.new("RGB", (512, 512), (128, 128, 128))
        resized = rgba.resize(size, Image.Resampling.LANCZOS)
        canvas.paste(resized.convert("RGB"), (left, top), resized.getchannel("A"))
        canvas.save(target)
    affine = np.array([[size[0] / width, 0, left], [0, size[1] / height, top], [0, 0, 1]])
    return {
        "source_size": [width, height],
        "input_size": [512, 512],
        "pixel_transform": affine.tolist(),
        "background_rgb": [128, 128, 128],
    }


def epfl_mesh(vertices: Path, triangles: Path, index_base: int = 0):
    """Read per-frame EPFL vertices plus the shared triangle connectivity."""
    import trimesh

    with np.load(vertices, allow_pickle=False) as data:
        points = np.asarray(data["mesh"], dtype=float).reshape(-1, 3)
    faces = np.loadtxt(triangles, dtype=int, ndmin=2) - index_base
    if index_base not in (0, 1) or faces.shape[1] != 3:
        raise ValueError("Specify the verified triangle index base (0 or 1)")
    if not np.isfinite(points).all() or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Invalid EPFL mesh vertices or triangle indices")
    return trimesh.Trimesh(points, faces, process=False)


def prepare_dataset(config_path: Path, output: Path) -> dict[str, Any]:
    """Prepare real selected files, keep failures, and leave unfrozen cases pending."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    samples = config.get("samples", [])
    ids = [safe_id(sample["sample_id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    sequences = [
        sample.get("sequence_id") for sample in samples if sample.get("category") == "deformable"
    ]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Core Cloth cases must come from distinct sequences")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, records = [], []

    def resolve(value: str) -> Path:
        return (config_path.parent / value).resolve()

    for spec in samples:
        item = {
            "sample_id": spec["sample_id"],
            "category": spec.get("category"),
            "status": "failed",
        }
        results.append(item)
        try:
            category = spec["category"]
            if category not in CATEGORIES:
                raise ValueError(f"Unknown category: {category}")
            base = {}
            if spec.get("source_manifest"):
                base = select_samples(
                    load_manifest(resolve(spec["source_manifest"])),
                    [spec.get("source_sample_id", spec["sample_id"])],
                )[0]
            image = Path(base["image"]) if base else resolve(spec["image"])
            reference = (
                Path(base["reference_mesh"])
                if base
                else (resolve(spec["reference_mesh"]) if spec.get("reference_mesh") else None)
            )
            folder = output / spec["sample_id"]
            folder.mkdir()
            metadata = {
                "sample_id": spec["sample_id"],
                "category": category,
                "source": spec.get("source", base.get("dataset")),
                "original_image": str(image),
                "original_reference": str(reference),
                "source_record": base,
                "physical_labels": spec.get("physical_labels", {}),
                "absolute_scale_verified": bool(spec.get("absolute_scale_verified", False)),
                "coordinate_convention": spec.get("coordinate_convention", "source_metadata"),
                "units": spec.get("units", "unverified"),
                "reference_unit_m": spec.get("reference_unit_m"),
                "paths": {},
            }
            for name, label in metadata["physical_labels"].items():
                if (
                    not isinstance(label, dict)
                    or not label.get("source")
                    or not label.get("evidence_type")
                ):
                    raise ValueError(f"Physical label {name} requires source and evidence_type")
                if label["evidence_type"] in {"model_inference", "material_prior"} and label.get(
                    "ground_truth"
                ):
                    raise ValueError(f"Predicted/prior {name} cannot be ground truth")
            if metadata["absolute_scale_verified"] and (
                not isinstance(metadata["reference_unit_m"], (int, float))
                or not np.isfinite(metadata["reference_unit_m"])
                or metadata["reference_unit_m"] <= 0
            ):
                raise ValueError("Verified scale requires positive reference_unit_m")
            prep = prepare_rgb(image, folder / "input.png")
            shutil.copyfile(image, folder / ("original_input" + image.suffix))
            metadata["input_preparation"] = prep
            depth_only = spec.get("reference_kind") == "visible_depth"
            if depth_only and (category != "deformable" or reference is not None):
                raise ValueError(
                    "Visible-depth reference requires a deformable sample without a mesh"
                )
            if category == "deformable":
                if not spec.get("sequence_id") or spec.get("frame_id") is None:
                    raise ValueError("Cloth requires sequence_id and frame_id")
                if not spec.get("pairing_verified") or not spec.get("pairing_evidence"):
                    raise ValueError("Cloth RGB/mesh pairing requires verified evidence")
                if "intrinsics" not in spec:
                    raise ValueError("Cloth requires original camera intrinsics")
                for key, archive_key in (("depth", "depth"), ("normal", "normals")):
                    source = resolve(spec[key])
                    with np.load(source, allow_pickle=False) as data:
                        array = data[archive_key]
                        if (key == "depth" and array.ndim != 2) or (
                            key == "normal" and (array.ndim != 3 or array.shape[-1] != 3)
                        ):
                            raise ValueError(f"Invalid {key} array shape")
                        if list(array.shape[:2]) != list(reversed(prep["source_size"])):
                            raise ValueError(f"{key} and RGB dimensions differ")
                    shutil.copyfile(source, folder / f"{key}.npz")
                    metadata["paths"][key] = f"{key}.npz"
                if depth_only:
                    scale = float(spec["depth_unit_m"])
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError("depth_unit_m must be finite and positive")
                    with Image.open(resolve(spec["foreground_mask"])) as mask_image:
                        mask = np.asarray(mask_image.convert("L")) > 0
                    with np.load(folder / "depth.npz", allow_pickle=False) as data:
                        depth = data["depth"]
                    if mask.shape != depth.shape or not np.any(
                        mask & np.isfinite(depth) & (depth > 0)
                    ):
                        raise ValueError("Foreground mask must overlap valid source depth")
                    Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "foreground.png")
                    metadata["paths"]["foreground_mask"] = "foreground.png"
                    metadata.update(reference_kind="visible_depth", depth_unit_m=scale)
                else:
                    for key in ("mesh_vertices", "triangles"):
                        source = resolve(spec[key])
                        shutil.copyfile(source, folder / source.name)
                        metadata["paths"][key] = source.name
                    epfl_mesh(
                        resolve(spec["mesh_vertices"]),
                        resolve(spec["triangles"]),
                        spec.get("triangle_index_base", 0),
                    ).export(folder / "reference.ply")
                    reference = folder / "reference.ply"
                intrinsic = np.asarray(spec["intrinsics"], dtype=float)
                if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                    raise ValueError("Invalid camera intrinsics")
                metadata.update(
                    sequence_id=spec["sequence_id"],
                    frame_id=str(spec["frame_id"]),
                    pairing_evidence=spec["pairing_evidence"],
                    intrinsics=intrinsic.tolist(),
                    input_intrinsics=(np.asarray(prep["pixel_transform"]) @ intrinsic).tolist(),
                )
            elif category == "articulated":
                from asset2sim.evaluation.articulation import read_urdf

                urdf = resolve(spec["urdf"])
                reference_tree = read_urdf(urdf, spec.get("joint_state", {}))
                if not any(
                    joint["type"] in {"revolute", "continuous", "prismatic"}
                    for joint in reference_tree["joints"]
                ):
                    raise ValueError("Articulated reference requires a movable joint")
                metadata.update(
                    kinematics=reference_tree,
                    joint_state=spec.get("joint_state", {}),
                    original_urdf=str(urdf),
                    render_evidence=spec.get("render_evidence"),
                )
                if not metadata["render_evidence"]:
                    raise ValueError("Articulated render requires pose/camera evidence")
                shutil.copyfile(urdf, folder / "reference.urdf")
                metadata["paths"]["urdf"] = "reference.urdf"
                metadata["joint_matches"] = spec.get("joint_matches", {})
                metadata["link_matches"] = spec.get("link_matches", {})
                metadata["joint_prediction_to_reference"] = spec.get(
                    "joint_prediction_to_reference"
                )
                metadata["part_matches"] = spec.get("part_matches", {})
                metadata["part_references"] = {}
                for part_id, part in spec.get("part_references", {}).items():
                    safe_id(str(part_id))
                    destination = folder / f"part_{part_id}.glb"
                    from asset2sim.evaluation.geometry import scene_mesh

                    _, part_mesh = scene_mesh(resolve(part["mesh"]))
                    part_mesh.export(destination)
                    metadata["part_references"][part_id] = {**part, "mesh": destination.name}
            metadata["paths"]["image"] = "input.png"
            if not depth_only:
                if reference is None or not reference.is_file():
                    raise FileNotFoundError(f"Missing reference geometry: {reference}")
                from asset2sim.evaluation.geometry import scene_mesh

                _, mesh = scene_mesh(reference)
                metadata["reference_geometry"] = {
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "bounds": mesh.bounds.tolist(),
                }
                if reference.parent != folder:
                    destination = folder / ("reference" + reference.suffix)
                    # GLB/PLY/STL are self-contained; external OBJ/USD dependencies remain in source.
                    if reference.suffix.lower() in {".glb", ".ply", ".stl"}:
                        shutil.copyfile(reference, destination)
                    else:
                        mesh.export(folder / "reference.ply")
                        destination = folder / "reference.ply"
                    reference = destination
                metadata["paths"]["reference_mesh"] = reference.name
            write_json(folder / "sample_metadata.json", metadata)
            record = {
                "sample_id": spec["sample_id"],
                "object_id": base.get("object_id", spec.get("object_id", spec["sample_id"])),
                "view_id": base.get(
                    "view_id", str(spec.get("view_id", spec.get("frame_id", "provided")))
                ),
                "image": str(folder / "input.png"),
                "reference_mesh": str(reference) if reference is not None else None,
                "reference_kind": "visible_depth" if depth_only else "mesh",
                "sample_metadata": str(folder / "sample_metadata.json"),
                "category": category,
                "dataset": metadata["source"],
                "source": metadata["source"],
                "split": "evaluation",
            }
            if category == "deformable":
                record.update(
                    sequence_id=spec["sequence_id"],
                    frame_id=str(spec["frame_id"]),
                    object_id=spec.get("object_id", "epfl_cloth"),
                    object_class=spec.get("object_class", "cloth"),
                    evaluation_unit="sequence_case",
                )
            records.append(record)
            item["status"] = "ok"
        except Exception as error:  # noqa: BLE001 -- retain invalid inputs in the preparation table.
            item["reason"] = f"{type(error).__name__}: {error}"
    export_manifest(records, output / "manifest.jsonl")
    for record in records:
        record["sample_metadata"] = str(Path(record["sample_metadata"]).relative_to(output))
        for key in ("image", "reference_mesh"):
            if record.get(key):
                record[key] = str(Path(record[key]).relative_to(output))
    fields = [
        "sample_id",
        "category",
        "dataset",
        "source",
        "object_id",
        "view_id",
        "sequence_id",
        "frame_id",
        "object_class",
        "evaluation_unit",
        "image",
        "reference_mesh",
        "reference_kind",
        "sample_metadata",
        "split",
    ]
    with (output / "dataset_manifest.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    downloads = [
        "# Dataset downloads",
        "",
        "| File | Status | Bytes | Source |",
        "| --- | --- | ---: | --- |",
    ]
    for item in config.get("downloads", []):
        path = resolve(item["path"])
        downloads.append(
            f"| {path} | {'present' if path.is_file() else 'pending'} | {path.stat().st_size if path.is_file() else 'NA'} | {item['source']} |"
        )
    downloads.append("\nRigid inputs reuse local files. This command performs no downloads.\n")
    (output / "dataset_download_manifest.md").write_text("\n".join(downloads))
    summary = {
        "samples": results,
        "n_selected": len(samples),
        "n_prepared": len(records),
        "n_failed": len(samples) - len(records),
        "pending": config.get("pending", []),
        "ready": len(records) == len(samples) and not config.get("pending"),
    }
    write_json(output / "preparation.json", summary)
    return summary
