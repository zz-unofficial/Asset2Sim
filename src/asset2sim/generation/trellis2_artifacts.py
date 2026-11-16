"""Persist TRELLIS.2 geometry and material volume for server-side deferred export."""

import json
import shutil
from pathlib import Path


def save_native_asset(mesh, output: Path, parameters: dict) -> None:
    import torch

    tensors = {
        name: getattr(mesh, name).detach().cpu().contiguous()
        for name in ("vertices", "faces", "attrs", "coords")
    }
    voxel_size = torch.as_tensor(mesh.voxel_size).cpu().reshape(-1)
    tensors["voxel_size"] = voxel_size.repeat(3) if voxel_size.numel() == 1 else voxel_size
    if not torch.isfinite(tensors["vertices"]).all() or not torch.isfinite(tensors["attrs"]).all():
        raise ValueError("Non-finite native geometry or material attributes")
    faces = tensors["faces"]
    if faces.numel() == 0 or faces.min() < 0 or faces.max() >= len(tensors["vertices"]):
        raise ValueError("Invalid native triangle indices")
    if len(tensors["coords"]) != len(tensors["attrs"]):
        raise ValueError("Material coordinates and attribute rows differ")
    if tensors["voxel_size"].shape != (3,) or not (tensors["voxel_size"] > 0).all():
        raise ValueError("Native voxel size must contain three positive lengths")
    required = sum(t.numel() * t.element_size() for t in tensors.values())
    if shutil.disk_usage(output).free < required + 8 * 1024**3:
        raise OSError("Insufficient disk space for native asset plus 8 GiB reserve")
    temporary = output / "native_asset.pt.tmp"
    torch.save(tensors, temporary)
    temporary.replace(output / "native_asset.pt")
    metadata = {
        "schema": "trellis2_native_v1",
        "attr_layout": {k: [v.start, v.stop, v.step] for k, v in mesh.layout.items()},
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "parameters": parameters,
        "status": "export_pending",
    }
    (output / "native_asset.json").write_text(json.dumps(metadata, indent=2))


def export_native_asset(asset: Path, output: Path, overrides: dict | None = None) -> dict:
    """Run official textured GLB export without loading generation networks."""
    import time

    import numpy as np
    import o_voxel
    import torch

    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads(asset.with_suffix(".json").read_text())
    data = torch.load(asset, map_location="cpu", weights_only=True)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "native_geometry.npz",
        vertices=data["vertices"].numpy(),
        faces=data["faces"].numpy(),
    )
    parameters = metadata["parameters"] | (overrides or {})
    torch.cuda.synchronize()
    start = time.perf_counter()
    exported = o_voxel.postprocess.to_glb(
        vertices=data["vertices"].cuda(),
        faces=data["faces"].cuda(),
        attr_volume=data["attrs"].cuda(),
        coords=data["coords"].cuda(),
        attr_layout={k: slice(*v) for k, v in metadata["attr_layout"].items()},
        voxel_size=data["voxel_size"].cuda(),
        aabb=metadata["aabb"],
        decimation_target=parameters["decimation_target"],
        texture_size=parameters["texture_size"],
        remesh=parameters["remesh"],
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    exported.export(output / "mesh.glb", extension_webp=parameters["extension_webp"])
    torch.cuda.synchronize()
    return {"export_seconds": time.perf_counter() - start, "export_parameters": parameters}


def main() -> None:
    """Export a native run into a new, separately recorded GLB run."""
    import argparse
    import sys

    from asset2sim.artifacts import write_json
    from asset2sim.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Explicit export parameter overrides")
    parser.add_argument("--sample-id", action="append", help="Select an already generated subset")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "third_party/TRELLIS.2"))
    import yaml

    overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
    if set(overrides) - {"remesh", "decimation_target", "texture_size", "extension_webp"}:
        parser.error("Unsupported export override")
    rows = [json.loads(s) for s in (args.run / "results.jsonl").read_text().splitlines()]
    if args.sample_id:
        if set(args.sample_id) - {r["sample_id"] for r in rows}:
            parser.error("Some requested samples have no generation record")
        rows = [r for r in rows if r["sample_id"] in args.sample_id]
    args.output.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.run / "run.json").read_text())
    run.update(
        status="exporting", source_native_run=str(args.run.resolve()), export_overrides=overrides
    )
    run["native_to_export"] = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    run["samples"] = [r["sample_id"] for r in rows]
    write_json(args.output / "run.json", run)
    if args.sample_id:
        inputs = [json.loads(s) for s in (args.run / "inputs.jsonl").read_text().splitlines()]
        (args.output / "inputs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in inputs if r["sample_id"] in args.sample_id)
        )
    else:
        shutil.copy2(args.run / "inputs.jsonl", args.output / "inputs.jsonl")
    failures = 0
    with (args.output / "results.jsonl").open("x") as stream:
        for row in rows:
            sid = row["sample_id"]
            row["raw_generation_status"] = (
                "generated" if row.get("status") == "export_pending" else "failed"
            )
            try:
                if row.get("status") != "export_pending":
                    raise ValueError("Native generation did not succeed: " + str(row.get("error")))
                folder = args.output / sid
                row.update(export_native_asset(args.run / row["raw_asset"], folder, overrides))
                shutil.copy2(args.run / sid / "input.png", folder / "input.png")
                row.update(
                    status="ok",
                    export_pending=False,
                    mesh=f"{sid}/mesh.glb",
                    source_native_run=str(args.run.resolve()),
                )
                print(sid, "exported", flush=True)
            except Exception as error:  # noqa: BLE001 - retain each failed export
                failures += 1
                row.update(
                    status="failed", failed_stage="export", error=f"{type(error).__name__}: {error}"
                )
                print(sid, row["error"], flush=True)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    run.update(
        status="completed_with_failures" if failures else "completed",
        failures=failures,
        pending_exports=0,
    )
    write_json(args.output / "run.json", run)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
