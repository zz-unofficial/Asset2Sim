"""Compare point-conditioned reconstructions in the control mesh physical frame."""

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

from asset2sim.collision import build_collision
from asset2sim.geometry.point_edit import shared_frame_transform
from asset2sim.geometry.prepare import inspect_mesh, load_geometry, physical_status
from asset2sim.simulation.dynamics import _save_samples, make_dynamic_scene, simulate
from asset2sim.simulation.evaluate import _probe_results
from asset2sim.simulation.scene import make_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--edited", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("data/generated/spar3d/debug/physics_task4a_run01")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    physics = yaml.safe_load(Path("configs/physics/dynamics.yaml").read_text())
    contact = yaml.safe_load(Path("configs/physics/debug.yaml").read_text())
    fixed = {
        "mass_kg": 0.1,
        "center_of_mass_m": [0, 0, 0],
        "inertia_kg_m2": [[0.0004, 0, 0], [0, 0.0004, 0], [0, 0, 0.0004]],
    }
    results = []
    for sid in ["debug_ring_v0", "debug_cup_v0"]:
        item = {"sample_id": sid, "status": "running", "failures": []}
        try:
            cp = args.control / "prepared" / sid
            ep = args.edited / "prepared" / sid
            cm = json.loads((cp / "metadata.json").read_text())
            em = json.loads((ep / "metadata.json").read_text())
            control, _ = load_geometry(cp / "normalized.glb")
            edited, _ = load_geometry(ep / "normalized.glb")
            transform = shared_frame_transform(cm["scene_to_physical"], em["scene_to_physical"])
            edited.apply_transform(transform)
            dest = args.output / "prepared" / sid
            dest.mkdir(parents=True)
            edited.export(dest / "normalized.glb")
            em["operations"].append(
                {
                    "operation": "use_control_physical_frame",
                    "matrix": transform.tolist(),
                    "control_metadata": str((cp / "metadata.json").resolve()),
                }
            )
            em["scene_to_physical"] = cm["scene_to_physical"]
            em["output"] = "normalized.glb"
            report = inspect_mesh(edited)
            em["physical_adaptation"] = physical_status(report)
            (dest / "metadata.json").write_text(json.dumps(em, indent=2) + "\n")
            (dest / "mesh_report.json").write_text(json.dumps(report, indent=2) + "\n")
            cc = args.control / "collisions" / sid / "C2"
            ec = args.output / "collisions" / sid / "C2"
            build_collision(dest / "normalized.glb", ec, "C2")
            original = trimesh.load(args.source / sid / "points.ply")
            pc = np.asarray(original.vertices)
            cutoff = float(np.quantile(pc[:, 0], 0.8))
            normalize = np.asarray(cm["operations"][2]["matrix"])
            threshold = float(cutoff * normalize[0, 0] + normalize[0, 3])
            a, _ = trimesh.sample.sample_surface(control, 20000, seed=0)
            b, _ = trimesh.sample.sample_surface(edited, 20000, seed=1)
            distance = cKDTree(b).query(a)[0]
            roi = a[:, 0] >= threshold
            item["geometry"] = {
                "point_roi_threshold_model": cutoff,
                "physical_threshold_m": threshold,
                "roi_samples": int(roi.sum()),
                "non_roi_samples": int((~roi).sum()),
                "roi_mean_distance_m": float(distance[roi].mean()) if roi.any() else None,
                "non_roi_mean_distance_m": float(distance[~roi].mean()) if (~roi).any() else None,
                "distance_direction": "control surface samples to edited surface samples; sampling approximation",
                "common_frame_transform": transform.tolist(),
            }
            item["variants"] = {}
            for name, mesh, prepared, colliders in [
                ("control", control, cp, cc),
                ("edited", edited, dest, ec),
            ]:
                record = json.loads((colliders / "colliders.json").read_text())
                v = {
                    "components": record["component_count"],
                    "surface_error_m": record["surface_error"]["symmetric_mean_m"],
                }
                if physical_status(inspect_mesh(mesh))["status"] != "ready":
                    v["contact"] = {
                        "status": "unavailable",
                        "reason": "input mesh topology requires review",
                    }
                else:
                    counts = {
                        "status": "mesh_relative",
                        "valid": 0,
                        "planned": 0,
                        "tp": 0,
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                    }
                    for offset in (0.0, 0.07):
                        model, data, _ = make_scene(
                            colliders,
                            prepared / "normalized.glb",
                            args.output / f"{sid}_{name}_visual.obj",
                            contact,
                            offset,
                        )
                        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_y")
                        adr = model.jnt_qposadr[joint]
                        ys = np.linspace(-0.16, 0.16, 81)
                        centers = np.column_stack((np.full(len(ys), offset), ys, np.zeros(len(ys))))
                        signed = trimesh.proximity.signed_distance(mesh, centers)
                        scores = []
                        for y in ys:
                            data.qpos[adr] = y - contact["trajectory_start_y_m"]
                            mujoco.mj_forward(model, data)
                            scores.append(_probe_results(model, data)["geometric_contact"])
                        truth = signed >= -contact["probe_radius_m"]
                        pred = np.asarray(scores)
                        valid = np.abs(signed + contact["probe_radius_m"]) > 1e-6
                        counts["valid"] += int(valid.sum())
                        counts["planned"] += len(ys)
                        for key, mask in [
                            ("tp", truth & pred),
                            ("tn", ~truth & ~pred),
                            ("fp", ~truth & pred),
                            ("fn", truth & ~pred),
                        ]:
                            counts[key] += int(np.sum(valid & mask))
                    v["contact"] = counts
                v["dynamics"] = {}
                for task in ["drop", "push"]:
                    directory = args.output / "dynamics" / sid / name / task
                    model, data, _ = make_dynamic_scene(
                        prepared, colliders, fixed, physics, task, directory
                    )
                    trial = simulate(model, data, physics, task, directory / "simulation.mp4")
                    _save_samples(directory / "trajectory.csv", trial.pop("samples"))
                    v["dynamics"][task] = trial
                    if not trial["valid"] or trial["video_status"] != "completed":
                        item["failures"].append(
                            f"{name}/{task}: {trial['invalid_reason'] or trial['video_error']}"
                        )
                item["variants"][name] = v
            item["status"] = "completed_with_failures" if item["failures"] else "completed"
        except Exception as e:  # noqa: BLE001 - retain individual comparison failures
            item["status"] = "failed"
            item["failures"].append(f"{type(e).__name__}: {e}")
        results.append(item)
        (args.output / "comparison.json").write_text(
            json.dumps(
                {
                    "results": results,
                    "fixed_properties": fixed,
                    "warning": "Mesh-relative contact, not ground truth; shared control coordinate frame",
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    return 1 if any(r["status"] != "completed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
