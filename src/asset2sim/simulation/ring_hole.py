"""Offline paired Ring aperture and fixed-probe diagnostic."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from asset2sim.collision import build_collision
from asset2sim.geometry.point_edit import shared_frame_transform
from asset2sim.geometry.prepare import inspect_mesh, load_geometry, physical_status
from asset2sim.simulation.evaluate import _probe_results, evaluate_trajectory
from asset2sim.simulation.scene import make_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--edited", type=Path, required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--probe-radius", type=float, default=0.044)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sample = "debug_ring_v0"
    cp = args.control / "prepared" / sample
    ep = args.edited / "prepared" / sample
    cm = json.loads((cp / "metadata.json").read_text())
    em = json.loads((ep / "metadata.json").read_text())
    control, _ = load_geometry(cp / "normalized.glb")
    edited, _ = load_geometry(ep / "normalized.glb")
    transform = shared_frame_transform(cm["scene_to_physical"], em["scene_to_physical"])
    edited.apply_transform(transform)
    dest = args.output / "prepared"
    dest.mkdir()
    edited.export(dest / "normalized.glb")
    em["operations"].append(
        {"operation": "use_control_physical_frame", "matrix": transform.tolist()}
    )
    em["scene_to_physical"] = cm["scene_to_physical"]
    em["physical_adaptation"] = physical_status(inspect_mesh(edited))
    (dest / "metadata.json").write_text(json.dumps(em, indent=2) + "\n")
    ec = args.output / "collision"
    build_collision(dest / "normalized.glb", ec, "C2")
    cloud = trimesh.load(args.points)
    points = np.asarray(cloud.vertices)
    center = np.median(points, axis=0)
    _, axes = np.linalg.eigh(np.cov((points - center).T))
    axis = axes[:, 0]
    norm = np.array(cm["operations"][2]["matrix"])
    center_phys = center * norm[0, 0] + norm[:3, 3]
    rot = Rotation.align_vectors([[0, 1, 0]], [axis])[0].as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = -rot @ center_phys
    relative = points - center
    rad = relative - np.outer(relative @ axis, axis)
    cutoff = np.quantile(np.linalg.norm(rad, axis=1), 0.65) * norm[0, 0]
    a, _ = trimesh.sample.sample_surface(control, 20000, seed=0)
    b, _ = trimesh.sample.sample_surface(edited, 20000, seed=1)
    ref, _ = trimesh.sample.sample_surface(control, 20000, seed=1)

    def distances(dst):
        dist = cKDTree(dst).query(a)[0]
        rel = a - center_phys
        radial = rel - np.outer(rel @ axis, axis)
        mask = np.linalg.norm(radial, axis=1) < cutoff
        return {
            "inside_mean_m": float(dist[mask].mean()),
            "outside_mean_m": float(dist[~mask].mean()),
            "inside_n": int(mask.sum()),
            "outside_n": int((~mask).sum()),
        }

    result = {
        "status": "running",
        "target": "shrink inner points to 65% radius",
        "probe_radius_m": args.probe_radius,
        "probe_selection": "chosen after inspecting reconstructed center clearance; exploratory",
        "axis": axis.tolist(),
        "center_m": center_phys.tolist(),
        "point_roi_outer_radius_m": float(cutoff),
        "geometry_change": distances(b),
        "same_mesh_sampling_baseline": distances(ref),
        "conditions": {},
        "failures": [],
    }
    cfg = yaml.safe_load(Path("configs/physics/debug.yaml").read_text())
    cfg["probe_radius_m"] = args.probe_radius
    for name, mesh, prepared, coll in [
        ("control", control, cp, args.control / "collisions" / sample / "C2"),
        ("edited", edited, dest, ec),
    ]:
        try:
            ys = np.linspace(-0.16, 0.16, 161)
            centers = center_phys + np.outer(ys, axis)
            signed = trimesh.proximity.signed_distance(mesh, centers)
            overlap = signed >= -args.probe_radius
            model, data, validation = make_scene(
                coll,
                prepared / "normalized.glb",
                args.output / f"{name}_visual.obj",
                cfg,
                xml_path=args.output / f"{name}_scene.xml",
                object_transform=pose,
            )
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_y")
            adr = model.jnt_qposadr[joint]
            hits = []
            for y in ys:
                data.qpos[adr] = y - cfg["trajectory_start_y_m"]
                mujoco.mj_forward(model, data)
                hits.append(_probe_results(model, data)["geometric_contact"])
            pred = np.asarray(hits)
            valid = np.abs(signed + args.probe_radius) > 1e-6
            samples, trials = evaluate_trajectory(
                model, cfg, "debug_ring", "C2", "center", 0, reference_enabled=False
            )
            result["conditions"][name] = {
                "center_clearance_m": float(
                    trimesh.proximity.closest_point(mesh, np.array([center_phys]))[1][0]
                ),
                "mesh_overlap_samples": int((overlap & valid).sum()),
                "collision_overlap_samples": int((pred & valid).sum()),
                "tp": int((pred & overlap & valid).sum()),
                "tn": int((~pred & ~overlap & valid).sum()),
                "fp": int((pred & ~overlap & valid).sum()),
                "fn": int((~pred & overlap & valid).sum()),
                "valid_samples": int(valid.sum()),
                "planned_samples": len(ys),
                "dynamics": trials,
                "scene_validation": validation,
            }
            (args.output / f"{name}_trajectory.json").write_text(
                json.dumps({"samples": samples, "trials": trials}, indent=2) + "\n"
            )
        except Exception as e:  # noqa: BLE001
            result["failures"].append({"condition": name, "error": str(e)})
    result["status"] = "completed_with_failures" if result["failures"] else "completed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
