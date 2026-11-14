"""Generate the eight declared controlled topology references on the server."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Point, Polygon

from asset2sim.artifacts import read_jsonl
from asset2sim.data.synthetic import reference_meshes


def create_shape(spec: dict) -> trimesh.Trimesh:
    if "shape_name" in spec:
        shape = spec["shape_name"]
        if shape in {"open_sheet", "fabric_strip"}:
            x, y = np.meshgrid(
                np.linspace(-0.5, 0.5, 41),
                np.linspace(
                    -0.35 if shape == "open_sheet" else -0.15,
                    0.35 if shape == "open_sheet" else 0.15,
                    17,
                ),
            )
            z = 0.18 * x**2 if shape == "open_sheet" else 0.07 * np.sin(4 * np.pi * x)
            vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            faces = []
            for j in range(16):
                for i in range(40):
                    a = j * 41 + i
                    faces.extend([[a, a + 1, a + 42], [a, a + 42, a + 41]])
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if shape == "perforated_plate":
            boundary = Polygon([[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25]])
            for x in [-0.3, 0, 0.3]:
                boundary = boundary.difference(Point(x, 0).buffer(0.09, quad_segs=32))
            return trimesh.creation.extrude_polygon(boundary, height=0.06, engine="earcut")
        if shape == "slotted_block":
            outer = [[-0.5, -0.3], [0.5, -0.3], [0.5, 0.3], [-0.5, 0.3]]
            hole = [[-0.36, -0.025], [0.36, -0.025], [0.36, 0.025], [-0.36, 0.025]]
            return trimesh.creation.extrude_polygon(
                Polygon(outer, holes=[hole]), height=0.3, engine="earcut"
            )
        if shape == "wire_frame":
            vertices = np.array(
                [[x, y, z] for x in [-0.4, 0.4] for y in [-0.4, 0.4] for z in [-0.4, 0.4]]
            )
            rods = [
                trimesh.creation.cylinder(radius=0.025, segment=[a, b], sections=16)
                for i, a in enumerate(vertices)
                for b in vertices[i + 1 :]
                if np.count_nonzero(a != b) == 1
            ]
            return trimesh.util.concatenate(rods)
        if shape == "hollow_cup":
            return create_shape({"shape": "cup", "radius": 0.4, "height": 0.8, "thickness": 0.05})
        if shape == "nested_shell":
            outer = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
            inner = trimesh.creation.icosphere(subdivisions=3, radius=0.25)
            inner.invert()
            return trimesh.util.concatenate([outer, inner])
        if shape == "thin_fins":
            parts = [trimesh.creation.box(extents=[0.25, 0.25, 0.7])]
            for axis in [0, 1]:
                for sign in [-1, 1]:
                    size = [0.008, 0.008, 0.5]
                    size[axis] = 0.4
                    fin = trimesh.creation.box(extents=size)
                    offset = np.zeros(3)
                    offset[axis] = sign * 0.3
                    fin.apply_translation(offset)
                    parts.append(fin)
            return trimesh.util.concatenate(parts)
        raise ValueError(f"Unknown declared topology specification: {shape}")
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
        name = spec.get("object_id") or f"{spec['candidate_id']}_{spec['shape_name']}"
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe synthetic object ID")
        mesh = debug_meshes[name][1] if debug else create_shape(spec)
        # Deterministic PBR channels make the supplied synthetic material prior explicit.
        yy, xx = np.indices((64, 64))
        checker = ((xx // 8 + yy // 8) % 2).astype(np.uint8)
        color = np.stack([80 + 80 * checker, 130 + 50 * checker, 190 - 50 * checker], axis=-1)
        metallic = 30 + 170 * checker
        roughness = 70 + 130 * (1 - checker)
        maps = output / f"{name}_materials"
        maps.mkdir()
        Image.fromarray(color).save(maps / "basecolor.png")
        Image.fromarray(metallic).save(maps / "metallic.png")
        Image.fromarray(roughness).save(maps / "roughness.png")
        packed = np.stack([np.full_like(metallic, 255), roughness, metallic], axis=-1)
        spans = np.maximum(mesh.extents, 1e-9)
        uv = (mesh.vertices[:, :2] - mesh.bounds[0, :2]) / spans[:2]
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(color),
            metallicRoughnessTexture=Image.fromarray(packed),
            metallicFactor=1,
            roughnessFactor=1,
            doubleSided=spec.get("shape_name") == "fabric_strip",
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        path = output / f"{name}.glb"
        mesh.export(path)
        rows.append(
            {
                "object_id": name,
                "reference_mesh": os.path.relpath(path.resolve(), selection.resolve().parent),
                "source_up_axis": "Z",
                "group": spec.get("group", spec.get("structural_target")),
                "dataset": "debug" if debug else "controlled_topology",
                "second_view": False,
                "pbr_eligible": not debug,
                "material_channels": {
                    "baseColor": "deterministic checker texture",
                    "metallic": "deterministic checker texture",
                    "roughness": "deterministic checker texture",
                },
                "license": "CC0-1.0",
                "source_url": "procedural specification",
                "specification": spec,
                "canonical_view": 4 if spec.get("shape") == "ring" else 0,
                "candidate_id": spec.get("candidate_id"),
                "reference_generator": "trimesh analytic surfaces and polygon extrusion",
                "target_visibility": "fully enclosed internal surface"
                if spec.get("shape_name") == "nested_shell"
                else "inspect reference renders before freezing",
                "elevation_degrees": 75.0
                if spec.get("shape_name") in {"perforated_plate", "slotted_block"}
                else 20.0,
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
