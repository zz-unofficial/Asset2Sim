"""Prepare fixed GSO/ABO and procedural references for shared benchmark inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest
from asset2sim.data.paper import download, load_source_mesh, source_mesh, source_to_z_up
from asset2sim.data.rendering import MeshRenderer
from asset2sim.data.synthetic import camera_pose
from asset2sim.evaluation.geometry import normalized_mesh, scene_mesh
from asset2sim.experiments.config import load_parameters
from asset2sim.paths import PROJECT_ROOT


def reference_source(row: dict, selection: Path) -> tuple[Path, trimesh.Scene, trimesh.Trimesh]:
    """Reuse cached GSO OBJ or original ABO GLB; preserve their visual materials."""
    if row.get("abo_model_id"):
        identifier = row["abo_model_id"]
        if not identifier.isalnum():
            raise ValueError("Invalid ABO model ID")
        source = PROJECT_ROOT / "data/raw/abo" / f"{identifier}.glb"
        download(row["download_url"], source)
        scene, mesh = scene_mesh(source)
        for geometry in scene.geometry.values():
            material = getattr(geometry.visual, "material", None)
            if (
                getattr(material, "baseColorTexture", None) is None
                or getattr(material, "metallicRoughnessTexture", None) is None
            ):
                raise ValueError(
                    "ABO reference lost a required base-color or metallic/roughness texture"
                )
        return source, scene, mesh
    if row.get("gso_name"):
        source = source_mesh({"name": row["gso_name"], "url": row["download_url"]}, "gso")
        mesh = load_source_mesh(source, "gso")
        if (
            getattr(mesh.visual, "kind", None) != "texture"
            or getattr(mesh.visual.material, "image", None) is None
        ):
            raise ValueError("GSO reference is missing its diffuse texture")
        return source, trimesh.Scene(mesh.copy()), mesh
    source = (selection.parent / row["reference_mesh"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    scene, mesh = scene_mesh(source)
    return source, scene, mesh


def prepare(
    selection: Path, output: Path, manifest: Path, render_config: Path | None = None
) -> dict:
    selection = selection.resolve()
    output = output.resolve()
    manifest = manifest.resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError("Use new output and manifest paths for each frozen dataset")
    candidates = [json.loads(line) for line in selection.read_text().splitlines() if line.strip()]
    if not candidates:
        raise ValueError("Selection is empty")
    ids = [row["object_id"] for row in candidates]
    if len(ids) != len(set(ids)) or any(Path(name).name != name for name in ids):
        raise ValueError("Object IDs must be unique safe path components")
    if any(
        not row.get("group") or not isinstance(row.get("second_view"), bool) for row in candidates
    ):
        raise ValueError("Each candidate needs an audited group and boolean second_view")
    output.mkdir(parents=True)
    audit = []
    records = []
    renderer = None
    try:
        for row in candidates:
            object_id = row["object_id"]
            print(f"Preparing reference {object_id}", flush=True)
            entry = {"object_id": object_id, "group": row.get("group", "unverified")}
            try:
                if row.get("audit_status", "approved") != "approved":
                    raise ValueError("Candidate awaits reference inspection and audit approval")
                source, source_scene, mesh = reference_source(row, selection)
                axis_transform = source_to_z_up(row.get("source_up_axis", "Z"))
                source_scene.apply_transform(axis_transform)
                mesh.apply_transform(axis_transform)
                if not len(mesh.faces) or mesh.area <= 0:
                    raise ValueError("Reference lacks usable visible geometry")
                _normalized, transform = normalized_mesh(mesh, 1.0)
                folder = output / object_id
                folder.mkdir(parents=True)
                source_scene.apply_transform(transform)
                source_scene.export(folder / "reference.glb")
                views = [
                    {
                        "image": f"views/{index:03d}.png",
                        "camera_to_world": camera_pose(
                            index * 22.5, row.get("elevation_degrees", 20.0), 2.5
                        ).tolist(),
                    }
                    for index in range(16)
                ]
                foreground = []
                use_blender = row.get("input_renderer") == "blender_pbr"
                render_profile = None
                if use_blender:
                    render_profile = load_parameters(
                        render_config
                        or PROJECT_ROOT / "configs/evaluation/generation_benchmark.yaml"
                    )
                    hdri = Path(render_profile.get("hdri", ""))
                    if not hdri.is_absolute():
                        hdri = PROJECT_ROOT / hdri
                    if not hdri.is_file():
                        raise FileNotFoundError(f"Reference PBR HDRI missing: {hdri}")
                    jobs = [
                        {
                            "mesh": str(folder / "reference.glb"),
                            "output": str(folder / "views"),
                            "mode": "pbr",
                            "image_size": 512,
                            "yfov_degrees": 40,
                            "views": views,
                            "hdri": str(hdri),
                            "samples": render_profile.get("render_samples", 32),
                        }
                    ]
                    write_json(folder / "input_render_jobs.json", jobs)
                    with (folder / "input_render.log").open("x") as log:
                        subprocess.run(
                            [
                                render_profile.get("blender", "blender"),
                                "-b",
                                "--python",
                                str(PROJECT_ROOT / "src/asset2sim/evaluation/benchmark_render.py"),
                                "--",
                                "--jobs",
                                str(folder / "input_render_jobs.json"),
                            ],
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                else:
                    if renderer is None:
                        renderer = MeshRenderer(512)
                    (folder / "views").mkdir()
                for index, view in enumerate(views):
                    if use_blender:
                        with Image.open(folder / view["image"]) as opened:
                            image = opened.convert("RGBA")
                    else:
                        image = renderer.render(
                            source_scene, np.asarray(view["camera_to_world"]), 40.0, mode="pbr"
                        )
                    mask = np.asarray(image)[:, :, 3] > 127
                    if not mask.any():
                        raise ValueError(f"Empty reference silhouette at view {index}")
                    foreground.append(
                        {
                            "pixels": int(mask.sum()),
                            "touches_frame": bool(
                                mask[0].any()
                                or mask[-1].any()
                                or mask[:, 0].any()
                                or mask[:, -1].any()
                            ),
                        }
                    )
                    image.save(folder / "views" / f"{index:03d}.png")
                metadata = {
                    "image_size": 512,
                    "yfov_degrees": 40.0,
                    "views": views,
                    "source_to_reference": (transform @ axis_transform).tolist(),
                    "source_up_axis": row.get("source_up_axis", "Z"),
                    "reference_up_axis": "Z",
                    "render_mode": "pbr",
                    "source_file": str(source),
                    "foreground": foreground,
                    "input_renderer": row.get("input_renderer", "pyrender_neutral"),
                    "render_profile": render_profile,
                }
                canonical = row.get("canonical_view", 0)
                if canonical == "largest_silhouette":
                    canonical = int(np.argmax([entry["pixels"] for entry in foreground]))
                if not isinstance(canonical, int) or not 0 <= canonical < 16:
                    raise ValueError("canonical_view must be an integer in [0,15]")
                object_records = []
                for index in (
                    [canonical, (canonical + 8) % 16]
                    if row.get("second_view", False)
                    else [canonical]
                ):
                    render_file = folder / f"render_v{index:03d}.json"
                    write_json(render_file, metadata | {"conditioning_view": index})
                    object_records.append(
                        {
                            "sample_id": f"{object_id}_v{index:03d}",
                            "object_id": object_id,
                            "view_id": f"{index:03d}",
                            "split": "generation_benchmark",
                            "dataset": row.get("dataset", "custom"),
                            "category": row.get("group", "unverified"),
                            "image": str(folder / views[index]["image"]),
                            "reference_mesh": str(folder / "reference.glb"),
                            "render_metadata": str(render_file),
                            "source": str(source),
                            "download_url": row.get("download_url"),
                            "source_version": row.get("source_version"),
                            "source_name": row.get("gso_name", row.get("display_name")),
                            "abo_model_id": row.get("abo_model_id"),
                            "pilot": row.get("pilot", False),
                            "selection_reason": row.get("selection_reason"),
                            "conditioning_rule": row.get("canonical_view", 0),
                            "source_url": row.get("source_url", ""),
                            "license": row.get("license", ""),
                            "license_url": row.get("license_url"),
                            "pbr_eligible": row.get("pbr_eligible", False),
                            "material_channels": row.get("material_channels", {}),
                            "reference_audit": str(output / "audit.json"),
                            "candidate_id": row.get("candidate_id"),
                            "specification": row.get("specification"),
                            "source_file_identifier": row.get("source_file_identifier"),
                            "reference_generator": row.get("reference_generator"),
                            "target_visibility": row.get("target_visibility"),
                        }
                    )
                entry.update(
                    status="eligible",
                    vertices=len(mesh.vertices),
                    faces=len(mesh.faces),
                    watertight=bool(mesh.is_watertight),
                )
                records.extend(object_records)
            except Exception as error:  # noqa: BLE001 - audit all candidates
                entry.update(status="excluded", reason=f"{type(error).__name__}: {error}")
            audit.append(entry)
            print(
                f"{object_id}: {entry['status']}"
                + (f" ({entry['reason']})" if entry.get("reason") else ""),
                flush=True,
            )
    finally:
        if renderer is not None:
            renderer.close()
        write_json(output / "audit.json", audit)
    if not records:
        raise ValueError("No eligible references; inspect audit.json")
    export_manifest(records, manifest)
    summary = {
        "candidates": len(candidates),
        "eligible_objects": sum(row["status"] == "eligible" for row in audit),
        "input_views": len(records),
    }
    write_json(
        output / "dataset.json",
        {
            **summary,
            "selection": str(selection),
            "manifest": str(manifest),
            "rendering": {
                "image_size": 512,
                "azimuth_views": 16,
                "elevation": "per-object preregistered",
                "input_renderers": sorted(
                    {row.get("input_renderer", "pyrender_neutral") for row in candidates}
                ),
            },
        },
    )
    gallery = Image.new("RGB", (512, 180 * len(candidates)), "white")
    draw = ImageDraw.Draw(gallery)
    for index, candidate in enumerate(candidates):
        matching = [r for r in records if r["object_id"] == candidate["object_id"]]
        draw.text((4, index * 180 + 4), candidate["object_id"], fill="black")
        for column, record in enumerate(matching):
            with Image.open(record["image"]) as image:
                thumbnail = image.convert("RGBA")
                thumbnail.thumbnail((240, 150))
                gallery.paste(thumbnail, (column * 256, index * 180 + 24), thumbnail)
        if not matching:
            draw.text(
                (4, index * 180 + 30), "reference preparation failed: see audit.json", fill="red"
            )
    gallery.save(output / "reference_gallery.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=Path("experiments/datasets/generation_gso32.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--render-config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.selection, args.output, args.manifest, args.render_config), indent=2
        )
    )


if __name__ == "__main__":
    main()
