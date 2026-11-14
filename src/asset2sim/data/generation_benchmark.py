"""Audit locally supplied reference GLBs and freeze shared benchmark inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest
from asset2sim.data.rendering import MeshRenderer
from asset2sim.data.synthetic import camera_pose
from asset2sim.evaluation.geometry import normalized_mesh, scene_mesh


def prepare(selection: Path, output: Path, manifest: Path) -> dict:
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
        renderer = MeshRenderer(512)
        for row in candidates:
            object_id = row["object_id"]
            entry = {"object_id": object_id, "group": row.get("group", "unverified")}
            try:
                source = (selection.parent / row["reference_mesh"]).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                if row.get("source_up_axis") != "Z":
                    raise ValueError("Export references as Z-up GLB and declare source_up_axis=Z")
                source_scene, mesh = scene_mesh(source)
                if not len(mesh.faces) or mesh.area <= 0:
                    raise ValueError("Reference lacks usable visible geometry")
                _normalized, transform = normalized_mesh(mesh, 1.0)
                folder = output / object_id
                (folder / "views").mkdir(parents=True)
                source_scene.apply_transform(transform)
                source_scene.export(folder / "reference.glb")
                views = []
                foreground = []
                for index in range(16):
                    pose = camera_pose(index * 22.5, 20.0, 2.5)
                    image = renderer.render(source_scene, pose, 40.0, mode="pbr")
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
                    views.append(
                        {"image": f"views/{index:03d}.png", "camera_to_world": pose.tolist()}
                    )
                metadata = {
                    "image_size": 512,
                    "yfov_degrees": 40.0,
                    "views": views,
                    "source_to_reference": transform.tolist(),
                    "source_up_axis": "Z",
                    "reference_up_axis": "Z",
                    "render_mode": "pbr",
                    "source_file": str(source),
                    "foreground": foreground,
                }
                canonical = row.get("canonical_view", 0)
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
                            "dataset": row.get("dataset", "toys4k"),
                            "category": row.get("group", "unverified"),
                            "image": str(folder / views[index]["image"]),
                            "reference_mesh": str(folder / "reference.glb"),
                            "render_metadata": str(render_file),
                            "source": row["reference_mesh"],
                            "source_url": row.get("source_url", ""),
                            "license": row.get("license", ""),
                            "pbr_eligible": row.get("pbr_eligible", False),
                            "material_channels": row.get("material_channels", {}),
                            "reference_audit": str(output / "audit.json"),
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
            "rendering": "pyrender fixed 512 RGBA, 16 azimuths, 20 degree elevation",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
