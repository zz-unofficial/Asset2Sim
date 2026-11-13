"""Explicit, auditable mass and inertia estimates in the prepared mesh frame."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from asset2sim.geometry.prepare import inspect_mesh, physical_status


def estimate_properties(mesh: trimesh.Trimesh, profile: dict[str, Any]) -> dict[str, Any]:
    """Estimate homogeneous properties in meters/kg; never repair or modify mesh."""
    report = inspect_mesh(mesh)
    status = physical_status(report)
    if status["status"] == "invalid" or not report["longest_extent"]:
        raise ValueError("Invalid geometry cannot define physical properties")
    method = profile["method"]
    if method == "fixed":
        mass = float(profile["mass_kg"])
        center = np.asarray(profile["center_of_mass_m"], dtype=float)
        inertia = np.asarray(profile["inertia_kg_m2"], dtype=float)
        if center.shape != (3,) or inertia.shape != (3, 3):
            raise ValueError("Fixed COM and inertia require a 3-vector and 3x3 matrix")
        volume = None
        interpretation = "Explicit fixed mass, COM and inertia shared across edit conditions"
    elif method == "aabb":
        mass = float(profile["mass_kg"])
        extents = np.asarray(mesh.extents, dtype=float)
        center = np.asarray(mesh.bounds).mean(axis=0)
        inertia = np.diag(
            mass
            / 12
            * np.array(
                [
                    extents[1] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[1] ** 2,
                ]
            )
        )
        volume = None
        interpretation = "Fixed mass and homogeneous AABB solid; bounds-based COM and inertia"
    elif method in {"mesh_fixed_mass", "uniform_density"}:
        if status["status"] != "ready" or not report["valid_volume"]:
            raise ValueError(f"Mesh volume estimate requires ready closed topology: {status}")
        props = mesh.mass_properties
        volume = float(props["volume"])
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Mesh has nonpositive or nonfinite volume")
        mass = (
            float(profile["mass_kg"])
            if method == "mesh_fixed_mass"
            else volume * float(profile["density_kg_m3"])
        )
        center = np.asarray(props["center_mass"], dtype=float)
        original_mass = float(props["mass"])
        if not np.isfinite(original_mass) or original_mass <= 0:
            raise ValueError("Mesh integration returned invalid mass")
        inertia = np.asarray(props["inertia"], dtype=float) * (mass / original_mass)
        interpretation = (
            "Uniform solid density; volume-derived COM and inertia; no self-intersection guarantee"
        )
    else:
        raise ValueError(f"Unknown physical property method: {method}")
    if (
        not np.isfinite(mass)
        or mass <= 0
        or not np.isfinite(center).all()
        or not np.isfinite(inertia).all()
    ):
        raise ValueError("Mass, COM and inertia must be finite and mass positive")
    inertia = (inertia + inertia.T) / 2
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues.min() <= 1e-12 or eigenvalues[0] + eigenvalues[1] < eigenvalues[2] - 1e-10:
        raise ValueError(
            "Inertia must be positive definite and satisfy physical triangle inequality"
        )
    return {
        "method": method,
        "assumption": interpretation,
        "source": "mesh_and_declared_prior",
        "mass_kg": mass,
        "center_of_mass_m": center.tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "principal_inertia_kg_m2": eigenvalues.tolist(),
        "volume_m3": volume,
        "profile": profile,
        "mesh_physical_status": status,
        "mesh_bounds_m": mesh.bounds.tolist(),
        "warning": "Prior estimate only; scale, material, internal structure and self-intersections are not verified",
    }
