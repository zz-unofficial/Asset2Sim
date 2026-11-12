"""Build MuJoCo models with independent collision geoms and visual geometry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from asset2sim.geometry.prepare import load_geometry


def _numbers(values) -> str:
    return " ".join(str(v) for v in values)


def make_scene(
    collider_dir: Path,
    visual_path: Path,
    visual_obj: Path,
    config: dict,
    x_offset: float = 0.0,
    body_mode: str = "static",
    xml_path: Path | None = None,
    z_offset: float = 0.0,
    object_transform: np.ndarray | None = None,
):
    """Compile a scene and verify all mesh vertices in the physical coordinate frame."""
    record = json.loads((collider_dir / "colliders.json").read_text())
    components = record.get("components", [])
    if record.get("status") != "completed" or len(components) != record.get("component_count"):
        raise ValueError("Collision record is incomplete or component count differs")
    if not components or body_mode not in {"static", "dynamic"}:
        raise ValueError("Expected components and a static or dynamic body")
    visual, _ = load_geometry(visual_path)
    trimesh.Trimesh(vertices=visual.vertices, faces=visual.faces, process=False).export(visual_obj)
    root = ET.Element("mujoco", model="asset2sim_contact")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=str(config["timestep_s"]),
        gravity=_numbers(config["gravity_m_s2"]),
        solver=config["solver"],
        iterations=str(config["solver_iterations"]),
    )
    assets = ET.SubElement(root, "asset")
    sources = [("visual_mesh", visual_obj.resolve())]
    for i, component in enumerate(components):
        path = (collider_dir / component["path"]).resolve()
        if path.parent != collider_dir.resolve() or not path.is_file():
            raise ValueError(f"Invalid collision component path: {path}")
        sources.append((f"mesh_{i}", path))
    if xml_path is not None:
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        asset_dir = xml_path.parent / f"{xml_path.stem}_assets"
        asset_dir.mkdir(exist_ok=True)
        for name, source in sources:
            dest = asset_dir / f"{name}.obj"
            shutil.copyfile(source, dest)
            ET.SubElement(assets, "mesh", name=name, file=f"{asset_dir.name}/{dest.name}")
    else:
        for name, source in sources:
            ET.SubElement(assets, "mesh", name=name, file=str(source))
    transform = np.eye(4) if object_transform is None else np.asarray(object_transform, dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("object_transform must be a proper rigid transform")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).reshape(9))
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(
        world, "body", name="object", pos=_numbers(transform[:3, 3]), quat=_numbers(quat)
    )
    if body_mode == "dynamic":
        ET.SubElement(body, "freejoint", name="object_free")
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=str(config["body_mass_kg"]),
        diaginertia=_numbers(config["body_diagonal_inertia_kg_m2"]),
    )
    ET.SubElement(
        body,
        "geom",
        name="visual",
        type="mesh",
        mesh="visual_mesh",
        contype="0",
        conaffinity="0",
        mass="0",
        rgba="0.7 0.7 0.7 1",
    )
    for i in range(len(components)):
        ET.SubElement(
            body,
            "geom",
            name=f"collision_{i}",
            type="mesh",
            mesh=f"mesh_{i}",
            contype="1",
            conaffinity="1",
            mass="0",
            rgba="0 0 0 0",
        )
    probe = ET.SubElement(
        world,
        "body",
        name="probe",
        pos=_numbers([x_offset, config["trajectory_start_y_m"], z_offset]),
    )
    ET.SubElement(
        probe,
        "joint",
        name="probe_y",
        type="slide",
        axis="0 1 0",
        damping=str(config["probe_joint_damping"]),
    )
    ET.SubElement(
        probe,
        "geom",
        name="probe_geom",
        type="sphere",
        size=str(config["probe_radius_m"]),
        mass=str(config["probe_mass_kg"]),
        friction=_numbers(config["friction"]),
    )
    contacts = ET.SubElement(root, "contact")
    for i in range(len(components)):
        ET.SubElement(
            contacts,
            "pair",
            geom1=f"collision_{i}",
            geom2="probe_geom",
            condim="3",
            friction=_numbers(
                [
                    config["friction"][0],
                    config["friction"][0],
                    config["friction"][1],
                    config["friction"][2],
                    config["friction"][2],
                ]
            ),
            margin=str(config["contact_margin_m"]),
            gap=str(config["contact_gap_m"]),
            solref=_numbers(config["solref"]),
            solimp=_numbers(config["solimp"]),
        )
    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator, "position", name="probe_drive", joint="probe_y", kp=str(config["actuator_kp"])
    )
    xml = ET.tostring(root, encoding="unicode")
    if xml_path is not None:
        xml_path.write_text(xml + "\n")
    model = (
        mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        if xml_path is not None
        else mujoco.MjModel.from_xml_string(xml)
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual")
    if (
        model.ngeom != len(components) + 2
        or model.geom_contype[visual_id] != 0
        or model.geom_conaffinity[visual_id] != 0
    ):
        raise ValueError("Unexpected geom count or visual collision settings")
    validation = []
    for i, component in enumerate(components):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"collision_{i}")
        if geom_id < 0 or model.geom_bodyid[geom_id] != object_id:
            raise ValueError("Collision component was not attached to the common body")
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count]
        actual = local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        expected = np.asarray(
            trimesh.load(collider_dir / component["path"], force="mesh", process=False).vertices
        )
        expected = expected @ transform[:3, :3].T + transform[:3, 3]
        error = max(
            cKDTree(actual).query(expected)[0].max(), cKDTree(expected).query(actual)[0].max()
        )
        bounds_error = np.max(
            np.abs(
                np.array([actual.min(axis=0), actual.max(axis=0)])
                - np.array([expected.min(axis=0), expected.max(axis=0)])
            )
        )
        if error > 2e-6 or bounds_error > 2e-6:
            raise ValueError(f"Component {i} coordinate mismatch: {error:g} m")
        validation.append(
            {
                "geom_id": int(geom_id),
                "vertex_error_m": float(error),
                "aabb_error_m": float(bounds_error),
                "aabb_m": [actual.min(axis=0).tolist(), actual.max(axis=0).tolist()],
            }
        )
    return (
        model,
        data,
        {
            "expected_components": len(components),
            "actual_collision_geoms": len(validation),
            "body_mode": body_mode,
            "object_transform": transform.tolist(),
            "body_mass_kg": float(model.body_mass[object_id]),
            "body_inertia_kg_m2": model.body_inertia[object_id].tolist(),
            "components": validation,
            "visual_collision_enabled": False,
        },
    )
