import numpy as np
import pytest
import trimesh

from asset2sim.geometry.point_edit import displace_positive_x, shared_frame_transform
from asset2sim.simulation.properties import estimate_properties


def test_local_point_edit_preserves_outside_region():
    points = np.column_stack((np.linspace(-1, 1, 101), np.zeros(101), np.ones(101)))
    edited, threshold = displace_positive_x(points, magnitude=0.12, quantile=0.8)
    assert np.array_equal(edited[points[:, 0] <= threshold], points[points[:, 0] <= threshold])
    assert edited[-1, 0] - points[-1, 0] == pytest.approx(0.12)
    assert np.array_equal(edited[:, 1:], points[:, 1:])


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
