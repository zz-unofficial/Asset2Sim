"""Regenerate two fixed samples with original or locally edited SPAR3D points."""

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from asset2sim.generation.spar3d import SPAR3DConfig, SPAR3DGenerator
from asset2sim.geometry.point_edit import displace_positive_x, shrink_ring_hole


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["control", "edited"], required=True)
    p.add_argument("--edit", choices=["x_shift", "ring_hole"], default="x_shift")
    p.add_argument("--sample-id", action="append")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    generator = SPAR3DGenerator(SPAR3DConfig())
    generator.load()
    run = {
        "model": "spar3d",
        "status": "running",
        "config": generator.parameters,
        "source_run": str(args.source.resolve()),
        "mode": args.mode,
        **generator.metadata(),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    failures = 0
    with (args.output / "results.jsonl").open("w") as stream:
        for sid in args.sample_id or ("debug_ring_v0", "debug_cup_v0"):
            row = {"sample_id": sid, "object_id": sid.rsplit("_", 1)[0], "view_id": "v0"}
            try:
                source = args.source / sid
                cloud = trimesh.load(source / "points.ply", process=False)
                xyz = np.asarray(cloud.vertices).copy()
                color = np.asarray(cloud.colors)[:, :3].astype(np.float32) / 255
                if args.edit == "ring_hole":
                    if sid != "debug_ring_v0":
                        raise ValueError("ring_hole edit requires debug_ring_v0")
                    edited_xyz, edit_metadata = shrink_ring_hole(xyz)
                    threshold = None
                else:
                    edited_xyz, threshold = displace_positive_x(xyz)
                    edit_metadata = {"method": "x_shift"}
                displacement = np.linalg.norm(edited_xyz - xyz, axis=1)
                if args.mode == "edited":
                    xyz = edited_xyz
                out = args.output / sid
                out.mkdir()
                trimesh.PointCloud(xyz, colors=cloud.colors).export(out / "points.ply")
                image = Image.open(source / "input.png").convert("RGBA")
                image.save(out / "input.png")
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                pc = np.column_stack((xyz, color)).astype(np.float32)
                start = time.perf_counter()
                with (
                    torch.no_grad(),
                    (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if torch.cuda.is_available()
                        else nullcontext()
                    ),
                ):
                    mesh, _ = generator.model.run_image(
                        image,
                        bake_resolution=1024,
                        pointcloud=pc,
                        remesh="none",
                        vertex_count=-1,
                        return_points=True,
                    )
                mesh.export(out / "mesh.glb", include_normals=True)
                row.update(
                    status="ok",
                    mesh=f"{sid}/mesh.glb",
                    point_cloud=f"{sid}/points.ply",
                    changed_points=int(np.count_nonzero(displacement))
                    if args.mode == "edited"
                    else 0,
                    roi_x_threshold=threshold,
                    max_point_displacement=float(displacement.max())
                    if args.mode == "edited"
                    else 0.0,
                    edit_method=args.edit,
                    edit_parameters=edit_metadata,
                    elapsed_seconds=time.perf_counter() - start,
                )
                generator.recover()
            except Exception as error:  # noqa: BLE001 - retain individual generation failures
                failures += 1
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                generator.recover()
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    run.update(status="completed_with_failures" if failures else "completed", failures=failures)
    (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
