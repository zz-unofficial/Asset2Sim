"""Compose official Menagerie models with an existing physical asset."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import trimesh

from asset2sim.geometry.prepare import load_geometry
from asset2sim.simulation.properties import estimate_properties


def _ns(root):
    names = {}
    for node in root.iter():
        if "name" in node.attrib:
            names[node.attrib["name"]] = "grip_" + node.attrib["name"]
        if node.tag == "default" and "class" in node.attrib:
            names[node.attrib["class"]] = "grip_" + node.attrib["class"]
        if node.tag == "mesh" and "file" in node.attrib and "name" not in node.attrib:
            name = Path(node.attrib["file"]).stem
            node.attrib["name"] = name
            names[name] = "grip_" + name
    for node in root.iter():
        for attr in (
            "name",
            "class",
            "childclass",
            "mesh",
            "material",
            "body1",
            "body2",
            "joint",
            "joint1",
            "joint2",
            "tendon",
        ):
            if node.get(attr) in names:
                node.set(attr, names[node.get(attr)])
    return root


def create_scene(menagerie: Path, prepared: Path, colliders: Path, config: dict, output: Path):
    """Build a portable MJCF with separately exported convex components."""
    arm_path = menagerie / "universal_robots_ur5e"
    grip_path = menagerie / "robotiq_2f85"
    arm = ET.parse(arm_path / "ur5e.xml").getroot()
    grip = _ns(ET.parse(grip_path / "2f85.xml").getroot())
    record = json.loads((colliders / "colliders.json").read_text())
    if (
        record.get("status") != "completed"
        or len(record["components"]) != record["component_count"]
        or not record["components"]
    ):
        raise ValueError("Collision record invalid or incomplete")
    mesh, _ = load_geometry(prepared / "normalized.glb")
    scale = float(config["object_scale"])
    if not 0 < scale <= 1:
        raise ValueError("Object scale must be in (0,1]")
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    arm.find("compiler").set("meshdir", "assets")
    arm.find("option").attrib.update(
        timestep=str(config["timestep_s"]), gravity="0 0 -9.81", cone="elliptic", impratio="10"
    )
    arm.remove(arm.find("keyframe"))
    for root, directory, prefix in [(arm, arm_path, "arm_"), (grip, grip_path, "gripper_")]:
        for item in root.find("asset").findall("mesh"):
            original = item.get("file")
            dest = prefix + original
            shutil.copyfile(directory / "assets" / original, assets / dest)
            item.set("file", dest)
            if item.get("name") is None:
                item.set("name", Path(original).stem)
    arm.find("default").append(copy.deepcopy(grip.find("default").find("default")))
    for child in grip.find("asset"):
        arm.find("asset").append(copy.deepcopy(child))
    wrist = arm.find(".//body[@name='wrist_3_link']")
    adapter = ET.SubElement(wrist, "body", name="gripper_adapter", pos="0 0.1 0", quat="-1 1 0 0")
    adapter.append(copy.deepcopy(grip.find("worldbody").find("body")))
    for section in ["contact", "tendon", "equality"]:
        target = arm.find(section)
        if target is None:
            target = ET.SubElement(arm, section)
        for child in grip.find(section):
            target.append(copy.deepcopy(child))
    for child in grip.find("actuator"):
        arm.find("actuator").append(copy.deepcopy(child))
    world = arm.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="plane",
        size="1.5 1.5 0.1",
        pos=f"0 0 {config['table_height_m']}",
        friction="1 0.005 0.0001",
        rgba=".5 .5 .5 1",
    )
    ET.SubElement(world, "light", pos="0 -0.5 1.5", dir="0 0 -1", diffuse=".8 .8 .8")
    x, y = config["object_xy_m"]
    z = config["table_height_m"] - mesh.bounds[0, 2] * scale + 0.001
    body = ET.SubElement(world, "body", name="object", pos=f"{x} {y} {z}")
    ET.SubElement(body, "freejoint", name="object_free")
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    props = estimate_properties(scaled, {"method": "aabb", "mass_kg": config["object_mass_kg"]})
    inertia = np.array(props["inertia_kg_m2"])
    com = props["center_of_mass_m"]
    ET.SubElement(
        body,
        "inertial",
        pos=" ".join(map(str, com)),
        mass=str(props["mass_kg"]),
        diaginertia=" ".join(map(str, np.diag(inertia))),
    )
    visual = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    visual.export(assets / "object_visual.obj")
    ET.SubElement(
        arm.find("asset"),
        "mesh",
        name="object_visual_mesh",
        file="object_visual.obj",
        scale=f"{scale} {scale} {scale}",
    )
    ET.SubElement(
        body,
        "geom",
        name="object_visual",
        type="mesh",
        mesh="object_visual_mesh",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba=".2 .55 .85 1",
    )
    for i, component in enumerate(record["components"]):
        src = (colliders / component["path"]).resolve()
        if src.parent != colliders.resolve() or not src.is_file():
            raise ValueError("Unsafe or missing collider")
        filename = f"object_collision_{i:03d}.obj"
        shutil.copyfile(src, assets / filename)
        ET.SubElement(
            arm.find("asset"),
            "mesh",
            name=f"object_mesh_{i}",
            file=filename,
            scale=f"{scale} {scale} {scale}",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"object_collision_{i}",
            type="mesh",
            mesh=f"object_mesh_{i}",
            mass="0",
            friction=" ".join(map(str, config["object_friction"])),
            rgba="0 0 0 0",
        )
    xml = output / "scene.xml"
    ET.ElementTree(arm).write(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml.resolve()))
    return model, {
        "scene": str(xml.resolve()),
        "scale": scale,
        "dimensions_m": scaled.extents.tolist(),
        "object_initial_pos_m": [x, y, z],
        "mass_properties": props,
        "component_count": record["component_count"],
    }
