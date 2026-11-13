import numpy as np
import pytest
import trimesh

from asset2sim.geometry.point_edit import (
    displace_positive_x,
    shared_frame_transform,
    shrink_ring_hole,
)
from asset2sim.simulation.properties import estimate_properties


def test_local_point_edit_preserves_outside_region():
    points = np.column_stack((np.linspace(-1, 1, 101), np.zeros(101), np.ones(101)))
    edited, threshold = displace_positive_x(points, magnitude=0.12, quantile=0.8)
    assert np.array_equal(edited[points[:, 0] <= threshold], points[points[:, 0] <= threshold])
    assert edited[-1, 0] - points[-1, 0] == pytest.approx(0.12)
    assert np.array_equal(edited[:, 1:], points[:, 1:])


def test_ring_hole_radial_edit_preserves_outer_points():
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radii = np.repeat([0.5, 0.8, 1.0, 1.2], len(angles))
    phases = np.tile(angles, 4)
    points = np.column_stack([radii * np.cos(phases), radii * np.sin(phases), np.zeros_like(radii)])
    edited, info = shrink_ring_hole(points, factor=0.65)
    center = np.asarray(info["center"])
    before = np.linalg.norm(points - center, axis=1)
    after = np.linalg.norm(edited - center, axis=1)
    assert np.allclose(after[:200] / before[:200], 0.65)
    assert np.allclose(edited[-200:], points[-200:])
    with pytest.raises(ValueError):
        shrink_ring_hole(points, factor=1.1)


def test_shared_frame_and_fixed_mass():
    control = np.diag([0.2, 0.2, 0.2, 1.0])
    edited = np.diag([0.1, 0.1, 0.1, 1.0])
    assert np.allclose(shared_frame_transform(control, edited) @ edited, control)
    box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    profile = {
        "method": "fixed",
        "mass_kg": 0.1,
        "center_of_mass_m": [0, 0, 0],
        "inertia_kg_m2": [[0.0004, 0, 0], [0, 0.0004, 0], [0, 0, 0.0004]],
    }
    assert estimate_properties(box, profile)["mass_kg"] == 0.1
