"""Export transformed visual geometry and glTF color textures for MuJoCo."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh


def add_visual(root, body, source: Path, transform, assets: Path, scale: float):
    scene = trimesh.load(source, force="scene", process=False)
    records = []
    bounds = []
    for node in sorted(scene.graph.nodes_geometry):
        matrix, key = scene.graph.get(node)
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(np.asarray(transform) @ matrix)
        bounds.append(mesh.bounds * scale)
        material = getattr(mesh.visual, "material", None)
        materials = getattr(material, "materials", None)
        if materials:
            groups = [
                (m, np.flatnonzero(mesh.visual.face_materials == i))
                for i, m in enumerate(materials)
            ]
        else:
            groups = [(material, np.arange(len(mesh.faces)))]
        for material, faces in groups:
            if not len(faces):
                continue
            part = mesh.submesh([faces], append=True, repair=False)
            name = f"visual_{len(records):03d}"
            # OBJ carries vertex UVs; MuJoCo materials are declared explicitly below.
            obj_text = trimesh.exchange.obj.export_obj(part, include_texture=True)
            obj_text = "\n".join(
                line
                for line in obj_text.splitlines()
                if not line.startswith(("mtllib ", "usemtl "))
            )
            (assets / f"{name}.obj").write_text(obj_text + "\n")
            ET.SubElement(
                root.find("asset"),
                "mesh",
                name=name,
                file=f"{name}.obj",
                scale=f"{scale} {scale} {scale}",
            )
            image = getattr(material, "baseColorTexture", None)
            if image is None:
                image = getattr(material, "image", None)
            color = getattr(material, "baseColorFactor", None)
            if color is None:
                color = getattr(material, "diffuse", [255, 255, 255, 255])
            color = np.asarray(color, dtype=float)
            if color.max() > 1:
                color /= 255
            attrs = {"name": name + "_material", "rgba": " ".join(map(str, color))}
            textured = image is not None and getattr(part.visual, "uv", None) is not None
            if textured:
                image.convert("RGB").save(assets / f"{name}.png")
                ET.SubElement(
                    root.find("asset"),
                    "texture",
                    name=name + "_texture",
                    type="2d",
                    file=f"assets/{name}.png",
                )
                attrs["texture"] = name + "_texture"
            roughness = getattr(material, "roughnessFactor", None)
            if roughness is not None:
                attrs["shininess"] = str(1 - float(roughness))
            ET.SubElement(root.find("asset"), "material", **attrs)
            ET.SubElement(
                body,
                "geom",
                name=name,
                type="mesh",
                mesh=name,
                material=name + "_material",
                contype="0",
                conaffinity="0",
                mass="0",
            )
            records.append(
                {
                    "node": str(node),
                    "mesh": name,
                    "color_texture": bool(textured),
                    "faces": len(part.faces),
                }
            )
    if not records:
        raise ValueError("Visual asset has no triangles")
    return {
        "parts": records,
        "bounds_m": [
            np.min([b[0] for b in bounds], axis=0).tolist(),
            np.max([b[1] for b in bounds], axis=0).tolist(),
        ],
        "material_conversion": "Base color, UV texture and roughness approximation; normal/metallic/emission maps and alpha blending are not reproduced exactly",
    }
