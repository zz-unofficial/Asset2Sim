"""Render original textured meshes and individual C2 hulls for visual inspection."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from asset2sim.data.rendering import MeshRenderer
from asset2sim.data.synthetic import camera_pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control-mesh", type=Path, required=True)
    p.add_argument("--edited-mesh", type=Path, required=True)
    p.add_argument("--control-batch", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    metadata = json.loads((a.control_batch / "prepared/debug_ring_v0/metadata.json").read_text())
    transform = np.array(metadata["scene_to_physical"])
    meshes = []
    for path in (a.control_mesh, a.edited_mesh):
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.apply_transform(transform)
        meshes.append(mesh)
    for directory in (a.control_batch / "collisions/debug_ring_v0/C2", a.comparison / "collision"):
        record = json.loads((directory / "colliders.json").read_text())
        parts = []
        for index, item in enumerate(record["components"]):
            part = trimesh.load(directory / item["path"], force="mesh", process=False)
            part.visual.face_colors = [
                60 + (index * 73) % 170,
                70 + (index * 47) % 150,
                100 + (index * 31) % 140,
                255,
            ]
            parts.append(part)
        meshes.append(trimesh.util.concatenate(parts))
    renderer = MeshRenderer(image_size=400)
    canvas = Image.new("RGB", (800, 860), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        for i, (mesh, label) in enumerate(
            zip(
                meshes,
                [
                    "Control: textured mesh",
                    "Edited: textured mesh",
                    "Control: C2 components",
                    "Edited: C2 components",
                ],
            )
        ):
            img = renderer.render(mesh, camera_pose(45, 20, 0.45), mode="pbr")
            x = (i % 2) * 400
            y = (i // 2) * 430
            canvas.paste(img, (x, y + 25), img)
            draw.text((x + 10, y + 5), label, fill="black")
    finally:
        renderer.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.output)


if __name__ == "__main__":
    main()
