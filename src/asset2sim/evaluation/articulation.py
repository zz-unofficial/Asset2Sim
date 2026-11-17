"""URDF reference structure and frame-consistent joint comparison."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(text: str | None, default: str) -> np.ndarray:
    values = np.array([float(value) for value in (text or default).split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite three-vector")
    return values


def read_urdf(path: Path, joint_state: dict[str, float] | None = None) -> dict[str, Any]:
    """Read links and joints; record world frames at the declared input pose."""
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("Expected URDF robot root")
    links = [link.attrib["name"] for link in root.findall("link")]
    if not links or len(links) != len(set(links)):
        raise ValueError("Missing or duplicate links")
    joints, children = [], set()
    names = set()
    for element in root.findall("joint"):
        if element.attrib["name"] in names:
            raise ValueError("Duplicate joint names")
        names.add(element.attrib["name"])
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        if parent not in links or child not in links or child in children:
            raise ValueError("Invalid parent/child relationship")
        children.add(child)
        origin = element.find("origin")
        xyz = _vector(origin.get("xyz") if origin is not None else None, "0 0 0")
        rpy = _vector(origin.get("rpy") if origin is not None else None, "0 0 0")
        axis_node = element.find("axis")
        axis = _vector(axis_node.get("xyz") if axis_node is not None else None, "1 0 0")
        kind = element.get("type")
        if kind not in {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}:
            raise ValueError(f"Unknown joint type: {kind}")
        if kind != "fixed" and np.linalg.norm(axis) == 0:
            raise ValueError("Joint axis has zero length")
        limit = element.find("limit")
        joints.append(
            {
                "name": element.attrib["name"],
                "type": kind,
                "parent": parent,
                "child": child,
                "origin_xyz": xyz.tolist(),
                "origin_rpy": rpy.tolist(),
                "axis": (axis / np.linalg.norm(axis)).tolist()
                if np.linalg.norm(axis)
                else [1, 0, 0],
                "range": [float(limit.get("lower")), float(limit.get("upper"))]
                if limit is not None
                and limit.get("lower") is not None
                and limit.get("upper") is not None
                else None,
            }
        )
        if joints[-1]["range"] is not None:
            low, high = joints[-1]["range"]
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Invalid joint limits")
    roots = set(links) - children
    if len(roots) != 1 or len(joints) != len(links) - 1:
        raise ValueError("URDF must be a single connected acyclic tree")
    frames = {next(iter(roots)): np.eye(4)}
    pending = list(joints)
    joint_state = joint_state or {}
    if set(joint_state) - {joint["name"] for joint in joints}:
        raise ValueError("Unknown joint in input pose")
    while pending:
        progress = False
        for joint in pending[:]:
            if joint["parent"] not in frames:
                continue
            frame = np.eye(4)
            frame[:3, :3] = Rotation.from_euler("xyz", joint["origin_rpy"]).as_matrix()
            frame[:3, 3] = joint["origin_xyz"]
            world = frames[joint["parent"]] @ frame
            axis = np.asarray(joint["axis"])
            joint["world_origin"] = world[:3, 3].tolist()
            joint["world_axis"] = (world[:3, :3] @ axis).tolist()
            value = float(joint_state.get(joint["name"], 0))
            if not np.isfinite(value):
                raise ValueError("Nonfinite joint pose")
            motion = np.eye(4)
            if joint["type"] in {"revolute", "continuous"}:
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint["type"] == "prismatic":
                motion[:3, 3] = axis * value
            elif joint["type"] in {"floating", "planar"} and value:
                raise ValueError("Scalar pose is unsupported for floating/planar joints")
            frames[joint["child"]] = world @ motion
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Cyclic or disconnected URDF")
    return {
        "links": links,
        "joints": joints,
        "root": next(iter(roots)),
        "link_to_world": {name: frame.tolist() for name, frame in frames.items()},
    }


def joint_errors(predicted: dict, reference: dict, extent: float) -> dict[str, Any]:
    """Compare already matched joints in one world frame, handling axis sign ambiguity."""
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Positive object extent required")
    a, b = np.asarray(predicted["world_axis"]), np.asarray(reference["world_axis"])
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or min(np.linalg.norm(a), np.linalg.norm(b)) == 0
    ):
        raise ValueError("Invalid joint axes")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    sign = 1 if a @ b >= 0 else -1
    delta = np.asarray(reference["world_origin"]) - np.asarray(predicted["world_origin"])
    # Distance from the reference anchor to the predicted axis, invariant along that axis.
    result = {
        "joint_type_correct": predicted["type"] == reference["type"],
        "axis_error_degrees": float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))),
        "axis_anchor_distance_normalized": float(np.linalg.norm(delta - (delta @ a) * a) / extent),
        "range_endpoint_mae": None,
        "range_unit": None,
    }
    if (
        predicted["type"] == reference["type"]
        and predicted.get("range") is not None
        and reference.get("range") is not None
    ):
        interval = np.asarray(predicted["range"])
        if sign < 0:
            interval = -interval[::-1]
        result["range_endpoint_mae"] = float(np.abs(interval - reference["range"]).mean())
        result["range_unit"] = "m" if reference["type"] == "prismatic" else "rad"
    return result
