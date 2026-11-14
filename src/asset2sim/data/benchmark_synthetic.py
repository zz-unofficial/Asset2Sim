"""Generate the eight declared controlled topology references on the server."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from asset2sim.artifacts import read_jsonl
from asset2sim.data.synthetic import reference_meshes


def create_shape(spec: dict) -> trimesh.Trimesh:
    shape = spec["shape"]
    if shape == "ring":
        mesh = trimesh.creation.annulus(
            r_min=spec["inner_radius"],
            r_max=spec["outer_radius"],
            height=spec["height"],
            sections=64,
        )
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        return mesh
    if shape == "cup":
        r, h, t = spec["radius"], spec["height"], spec["thickness"]
        return trimesh.creation.revolve(
            [[0, 0], [r, 0], [r, h], [r - t, h], [r - t, t], [0, t], [0, 0]], sections=64
        )
    if shape == "box":
        return trimesh.creation.box(extents=spec["extents"])
    if shape == "bracket":
        return reference_meshes()["debug_bracket"][1]
    if shape == "plane":
        return trimesh.Trimesh(
            vertices=[[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            process=False,
        )
    if shape == "open_cylinder":
        return trimesh.creation.revolve(
            [[spec["radius"], 0], [spec["radius"], spec["height"]]], sections=64
        )
    raise ValueError(f"Unknown controlled shape: {shape}")


def prepare(specs: Path, output: Path, selection: Path, debug: bool = False) -> None:
    if output.exists() or selection.exists():
        raise FileExistsError("Use new synthetic output and selection paths")
    output.mkdir(parents=True)
    selection.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    debug_meshes = reference_meshes() if debug else {}
    definitions = (
        [
            {"object_id": name, "shape": "debug", "group": group}
            for name, (group, _mesh) in debug_meshes.items()
        ]
        if debug
        else read_jsonl(specs)
    )
    for spec in definitions:
        name = spec["object_id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe synthetic object ID")
        mesh = debug_meshes[name][1] if debug else create_shape(spec)
        mesh.visual.vertex_colors = [160, 180, 210, 255]
        path = output / f"{name}.glb"
        mesh.export(path)
        rows.append(
            {
                "object_id": name,
                "reference_mesh": os.path.relpath(path.resolve(), selection.resolve().parent),
                "source_up_axis": "Z",
                "group": spec["group"],
                "dataset": "debug" if debug else "controlled_topology",
                "second_view": False,
                "pbr_eligible": False,
                "license": "CC0-1.0",
                "source_url": "procedural specification",
                "specification": spec,
                "canonical_view": 4 if spec["shape"] == "ring" else 0,
            }
        )
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specs", type=Path, default=Path("experiments/datasets/controlled_topology_specs_8.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    prepare(args.specs, args.output, args.selection, args.debug)
