"""Static geometry, export structure and property-transfer auditing."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from asset2sim.evaluation.geometry import scene_mesh
from asset2sim.geometry.prepare import inspect_mesh


def raw_parts(path: Path) -> dict[int, dict[str, Any]]:
    """Keep each original field, not just the upstream parser's numeric average."""
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*l_(\d+)\s*:\s*(.*)", line)
        if match:
            if int(match[1]) in result:
                raise ValueError(f"Duplicate raw part ID: {match[1]}")
            values = [value.strip() for value in match[2].split("|")]
            names = (
                "name",
                "priority_rank",
                "material",
                "density",
                "young",
                "poisson",
                "description",
            )
            result[int(match[1])] = dict(zip(names, values))
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def inertial_check(node: ET.Element, *, urdf: bool) -> dict:
    """Check explicitly stored inertia; inferred values remain an engine-stage concern."""
    if urdf:
        mass = float(node.find("mass").get("value"))
        origin, tensor = node.find("origin"), node.find("inertia")
        center = (
            [float(x) for x in origin.get("xyz", "0 0 0").split()]
            if origin is not None
            else [0, 0, 0]
        )
        values = [float(tensor.get(key)) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    else:
        mass = float(node.get("mass"))
        center = [float(x) for x in node.get("pos", "0 0 0").split()]
        values = [float(x) for x in node.get("fullinertia", node.get("diaginertia", "")).split()]
        if len(values) == 3:
            values += [0, 0, 0]
    if (
        len(values) != 6
        or len(center) != 3
        or not np.isfinite([mass, *values, *center]).all()
        or mass <= 0
    ):
        raise ValueError("Nonfinite or nonpositive explicit inertial")
    a, b, c, d, e, f = values
    eigen = np.linalg.eigvalsh([[a, d, e], [d, b, f], [e, f, c]])
    if eigen.min() <= 0 or eigen[0] + eigen[1] < eigen[2] - 1e-12:
        raise ValueError("Inertia must be positive definite and satisfy triangle inequality")
    return {
        "mass_kg": mass,
        "center": center,
        "principal_inertia": eigen.tolist(),
        "source": "explicit_file_value",
    }


def audit_asset(asset: Path, category: str) -> dict[str, Any]:
    """Inspect references, properties and category structure; compilation is separate."""
    if category not in {"rigid", "articulated", "deformable"}:
        raise ValueError(f"Unknown category: {category}")
    errors, warnings, transfers, structures = [], [], [], {}
    properties = json.loads((asset / "basic_info.json").read_text())
    parts = properties.get("parts", [])
    if not parts:
        errors.append("missing_physical_parts")
    raw = raw_parts(asset / "basic_info.txt")
    exported_ids = {int(part.get("label", index)) for index, part in enumerate(parts)}
    if len(exported_ids) != len(parts):
        errors.append("duplicate_exported_part_ids")
    if set(raw) != exported_ids:
        errors.append("raw_and_exported_part_ids_differ")
    roots = {}
    for name in ("basic.xml", "basic.urdf"):
        try:
            roots[name] = ET.parse(asset / name).getroot()
        except (ET.ParseError, OSError) as error:
            errors.append(f"{name}: {error}")
    mjcf = roots.get("basic.xml")
    if mjcf is not None and mjcf.tag != "mujoco":
        errors.append("invalid_mjcf_root")
    nodes = list(mjcf.iter()) if mjcf is not None else []
    compiler = mjcf.find("compiler") if mjcf is not None else None
    for node in nodes:
        if node.tag == "inertial":
            try:
                structures.setdefault("mjcf_inertials", []).append(inertial_check(node, urdf=False))
            except (ValueError, TypeError, AttributeError) as error:
                errors.append(f"invalid_mjcf_inertial:{error}")
    if not any(node.tag == "inertial" for node in nodes):
        transfers.append(
            {
                "part_id": None,
                "field": "mass_com_inertia",
                "status": "DEFERRED_TO_ENGINE",
                "source": "geom_density_and_compiler_mesh_inertia_mode",
            }
        )
    for node in nodes:
        if not node.get("file"):
            continue
        filename = node.get("file")
        prefix = ""
        if compiler is not None:
            key = (
                "texturedir"
                if node.tag == "texture"
                else "meshdir"
                if node.tag in {"mesh", "flexcomp"}
                else None
            )
            if key:
                prefix = compiler.get(key, compiler.get("assetdir", ""))
        path = asset / prefix / filename
        if not path.is_file():
            errors.append(f"missing_dependency:{node.tag}:{filename}")
        elif node.tag in {"mesh", "flexcomp"}:
            try:
                _, mesh = scene_mesh(path)
                diagnostic = inspect_mesh(mesh, diagnose_coincident=False)
                if not diagnostic["finite_vertices"] or diagnostic["degenerate_faces"]:
                    errors.append(f"invalid_mesh:{filename}")
                structures.setdefault("mesh_diagnostics", []).append(
                    {"file": filename, **diagnostic}
                )
            except (ValueError, OSError, TypeError) as error:
                errors.append(f"unreadable_mesh:{filename}:{error}")
    urdf = roots.get("basic.urdf")
    if urdf is not None:
        from asset2sim.evaluation.articulation import read_urdf

        try:
            structures["urdf"] = read_urdf(asset / "basic.urdf")
            for node in urdf.iter("mesh"):
                filename = node.get("filename", "")
                if filename.startswith("package://"):
                    warnings.append(f"package_uri_requires_mapping:{filename}")
                elif not (asset / filename).is_file():
                    errors.append(f"missing_urdf_mesh:{filename}")
        except (ValueError, KeyError, AttributeError) as error:
            errors.append(f"invalid_kinematic_tree:{error}")
        for inertial in urdf.iter("inertial"):
            try:
                structures.setdefault("urdf_inertials", []).append(
                    inertial_check(inertial, urdf=True)
                )
            except (ValueError, TypeError, AttributeError) as error:
                errors.append(f"invalid_urdf_inertial:{error}")
            mass, inertia = inertial.find("mass"), inertial.find("inertia")
            if mass is not None and inertia is not None:
                values = [inertia.get(key) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                if mass.get("value") == "1.0" and values == [
                    "1.0",
                    "1.0",
                    "1.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ]:
                    warnings.append("urdf_unit_mass_inertia_placeholder")
    if category == "articulated":
        movable = [
            joint
            for joint in structures.get("urdf", {}).get("joints", [])
            if joint["type"] not in {"fixed", "floating"}
        ]
        if not movable:
            errors.append("missing_articulated_joint")
    flexes = [node for node in nodes if node.tag in {"flexcomp", "flex"}]
    if category == "deformable" and not flexes:
        errors.append("deformable_branch_not_activated")
    structures["flex_count"] = len(flexes)
    for flex in flexes:
        elasticity = flex.find("elasticity")
        if elasticity is None:
            errors.append("missing_flex_elasticity")
        else:
            for key in ("thickness", "elastic2d"):
                if key not in elasticity.attrib:
                    warnings.append(f"flex_{key}_deferred_to_engine_default")
            if elasticity.get("damping") == "0.001":
                transfers.append(
                    {
                        "part_id": None,
                        "field": "damping",
                        "status": "DEFAULTED",
                        "export_value": 0.001,
                        "source": "official_export_constant",
                    }
                )
    for index, part in enumerate(parts):
        label = int(part.get("label", index))
        if label not in raw:
            errors.append(f"part_missing_from_raw_description:{label}")
        # Decoder IDs must cover every predicted part, including parts omitted by export.
        for extension in ("obj", "glb"):
            if not (asset / "objs" / str(label) / f"{label}.{extension}").is_file():
                errors.append(f"missing_part_geometry:{label}:{extension}")
        for field, json_key, multiplier, unit in (
            ("material", "material", 1, "category"),
            ("density", "density", 1000, "kg/m3"),
            ("young", "Young's Modulus (GPa)", 1e9, "Pa"),
            ("poisson", "Poisson's Ratio", 1, "dimensionless"),
        ):
            value = part.get(json_key)
            numeric = _number(value)
            row = {
                "part_id": label,
                "field": field,
                "raw_text": raw.get(label, {}).get(field),
                "json_value": value,
                "export_unit": unit,
                "status": "MISSING",
                "export_values": [],
                "raw_to_json": "unverified",
            }
            if field == "material":
                row["raw_to_json"] = (
                    "PRESERVED" if value == raw.get(label, {}).get(field) else "OVERRIDDEN"
                )
            elif row["raw_text"]:
                # Units are declared by the upstream template; keep the literal text for review.
                text = re.sub(
                    r"(?:g/cm(?:\^?3|³)|kg/m(?:\^?3|³)|GPa|Pa)",
                    "",
                    row["raw_text"],
                    flags=re.IGNORECASE,
                )
                text = re.sub(r"(?<=\d)\s*[-–~]\s*(?=\d)", " ", text)
                numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
                if numbers and numeric is not None:
                    row["raw_to_json"] = (
                        "PRESERVED" if np.isclose(np.mean(numbers), numeric) else "OVERRIDDEN"
                    )
            if row["raw_to_json"] == "OVERRIDDEN":
                errors.append(f"raw_to_json:{label}:{field}")
            required = field == "density" or category == "deformable"
            if (
                field != "material"
                and (required or numeric is not None)
                and (
                    numeric is None
                    or (field in {"density", "young"} and numeric <= 0)
                    or (field == "poisson" and not -1 < numeric < 0.5)
                )
            ):
                errors.append(f"invalid_physical_value:{label}:{field}")
            if field == "material":
                row["status"] = (
                    "PRESERVED"
                    if value is not None and raw.get(label, {}).get(field) == value
                    else "MISSING"
                )
                row["export_note"] = "semantic_material_is_not_an_engine_material_class"
                row["export_target"] = "basic_info.json"
            else:
                expected = numeric * multiplier if numeric is not None else None
                # Default geom classes carry density; flex name carries the part identity.
                matching = [
                    node
                    for node in nodes
                    if re.match(rf"l_{label}(?:_|$)", node.get("class", "") or node.get("name", ""))
                ]
                candidates = []
                for node in matching:
                    candidates.extend(node.iter())
                attr = {"density": "density", "young": "young", "poisson": "poisson"}[field]
                actual = [
                    float(node.get(attr)) for node in candidates if node.get(attr) is not None
                ]
                row["export_values"] = actual
                if actual and expected is not None:
                    row["status"] = (
                        ("TRANSFORMED" if multiplier != 1 else "PRESERVED")
                        if all(np.isfinite(x) and np.isclose(x, expected) for x in actual)
                        else "OVERRIDDEN"
                    )
                elif field == "density" and flexes:
                    row["status"] = "DERIVED"
                    row["export_note"] = (
                        "flex_mass_from_density_and_geometry_requires_formula_audit"
                    )
                elif category != "deformable" and field in {"young", "poisson"}:
                    row["status"] = "NOT_APPLICABLE"
                if row["status"] in {"MISSING", "OVERRIDDEN"}:
                    errors.append(f"property_transfer:{label}:{field}:{row['status']}")
            transfers.append(row)
    return {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "parts": len(parts),
        "transfers": transfers,
        "structure": structures,
        "validation": "static_only",
        "properties": properties,
    }
