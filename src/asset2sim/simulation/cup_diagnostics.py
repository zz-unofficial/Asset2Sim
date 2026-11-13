"""Inspect closest sampled pusher-to-component distance in an existing Cup run."""

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    for name, coll in [
        ("control", args.control / "collisions/debug_cup_v0/C2"),
        ("edited", args.comparison / "collisions/debug_cup_v0/C2"),
    ]:
        path = args.comparison / "dynamics/debug_cup_v0" / name / "push/trajectory.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        points = []
        for row in rows:
            if float(row["time_s"]) < 0.6:
                continue
            position = np.array([float(row[k]) for k in ["x_m", "y_m", "z_m"]])
            quat = [float(row[k]) for k in ["quat_x", "quat_y", "quat_z", "quat_w"]]
            pusher = np.array([float(row[k]) for k in ["pusher_x_m", "pusher_y_m", "pusher_z_m"]])
            points.append(Rotation.from_quat(quat).inv().apply(pusher - position))
        mesh = trimesh.util.concatenate(
            [trimesh.load(path, force="mesh") for path in sorted(coll.glob("collision_*.obj"))]
        )
        distances = trimesh.proximity.closest_point(mesh, np.asarray(points))[1]
        print(
            f"{name}: min sampled center-to-component-surface={distances.min() * 1000:.2f} mm, samples={len(points)}, sphere radius=15 mm"
        )


if __name__ == "__main__":
    main()
