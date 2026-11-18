"""Offline evidence tracing from input pixels through RLE voxels to native mesh."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from asset2sim.artifacts import write_json
from asset2sim.evaluation.geometry import scene_mesh
from asset2sim.experiments.physx_validation import validate_rle


def trace_failures(bundle: Path, upstream: Path, output: Path, figures: Path) -> dict:
    """Replay saved coordinates and render diagnostic projections without inference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = bundle.resolve()
    output.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    items = json.loads((bundle / "core/asset_index.json").read_text())["samples"]
    evidence = []
    for item in items:
        sample = item["sample_id"]
        work = bundle / "runs" / item["source_run"] / sample
        asset = work / "ours_demo/asset"
        prepared = np.asarray(Image.open(bundle / item["input_image"]).convert("RGB"))
        actual = np.asarray(Image.open(work / "inputs/asset.png").convert("RGB"))
        conditional = np.asarray(Image.open(asset / "cond_img.png").convert("RGB"))
        coords = np.load(asset / "allind.npy", allow_pickle=False)
        description = (asset / "basic_info.txt").read_text()
        raw_name = re.search(r"^Name:\s*(.*)$", description, re.MULTILINE)
        raw_category = re.search(r"^Category:\s*(.*)$", description, re.MULTILINE)
        evidence.append(
            {
                "sample_id": sample,
                "prepared_equals_model_input": np.array_equal(prepared, actual),
                "conditional_equals_model_input": np.array_equal(actual, conditional),
                "predicted_name": raw_name[1] if raw_name else None,
                "predicted_category": raw_category[1] if raw_category else None,
                "voxel_count": len(coords),
                "unique_voxel_count": len(np.unique(coords, axis=0)),
                **validate_rle(asset, upstream, capture_parser_messages=True),
            }
        )

    selected = [
        item
        for item in items
        if item["sample_id"]
        in {
            "gso_box_v004",
            "gso_mug_v005",
            "gso_hammer_v003",
            "partnet_microwave_7236",
            "epfl_Lr_top_edge_2_0656",
        }
    ]
    fig = plt.figure(figsize=(12, 3.4 * len(selected)))
    colors = plt.get_cmap("tab10")
    rng = np.random.default_rng(0)
    for row, item in enumerate(selected):
        asset = bundle / item["asset_directory"]
        image_ax = fig.add_subplot(len(selected), 3, row * 3 + 1)
        image_ax.imshow(Image.open(bundle / item["input_image"]))
        image_ax.set_title(item["sample_id"])
        image_ax.axis("off")
        for column, kind in ((2, "VLM voxels"), (3, "Decoded GLB -> voxel axes")):
            ax = fig.add_subplot(len(selected), 3, row * 3 + column, projection="3d")
            for part in sorted(asset.glob("ind_*.npy")):
                label = int(part.stem.removeprefix("ind_"))
                if column == 2:
                    points = (np.load(part).astype(float) + 0.5) / 64 - 0.5
                else:
                    points = scene_mesh(asset / f"objs/{label}/{label}.glb")[1].vertices
                    # Invert the official to_glb Z-up -> Y-up export rotation.
                    points = points @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                if len(points) > 4000:
                    points = points[rng.choice(len(points), 4000, replace=False)]
                ax.scatter(*points.T, s=0.8, color=colors(label % 10), alpha=0.5)
            ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), zlim=(-0.55, 0.55))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=-60)
            ax.set_title(kind)
    fig.tight_layout()
    fig.savefig(figures / "input_voxels_mesh.png", dpi=150)
    plt.close(fig)

    mug = next(item for item in items if item["sample_id"] == "gso_mug_v005")
    folder = bundle / mug["asset_directory"]
    body, handle = [np.load(folder / f"ind_{i}.npy") for i in (0, 1)]
    grid = np.zeros((64, 64, 64), dtype=bool)
    grid[tuple(body.T)] = True
    hg = np.zeros_like(grid)
    hg[tuple(handle.T)] = True
    lo, hi = body.min(0), body.max(0)
    mid = int((lo[1] + hi[1]) // 2)
    panels = [
        (grid[:, :, lo[2]].T, f"Body bottom z={lo[2]}"),
        (grid[:, :, hi[2]].T, f"Body top z={hi[2]}"),
        (grid[:, mid, :].T, f"Body vertical slice y={mid}"),
        (hg.any(axis=0).T, "Handle YZ projection"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (mask, title) in zip(axes, panels):
        ax.imshow(
            mask,
            origin="lower",
            extent=(-0.5, 63.5, -0.5, 63.5),
            interpolation="nearest",
            cmap="gray_r",
        )
        ax.set_title(title)
        ax.set_xlabel("voxel grid index")
    fig.tight_layout()
    fig.savefig(figures / "mug_voxel_sections.png", dpi=160)
    plt.close(fig)
    mug_trace = {
        "body_bottom_z": int(lo[2]),
        "body_top_z": int(hi[2]),
        "bottom_occupied": int(grid[:, :, lo[2]].sum()),
        "top_occupied": int(grid[:, :, hi[2]].sum()),
        "central_y": mid,
        "bottom_central_row_occupied": int(grid[:, mid, lo[2]].sum()),
        "top_central_row_occupied": int(grid[:, mid, hi[2]].sum()),
        "coordinate_unit": "64-cube voxel grid; not meters",
        "interpretation": "Both extreme slices have broad occupancy; cap-like geometry exists before mesh decoding.",
    }
    result = {
        "samples": evidence,
        "mug_trace": mug_trace,
        "figure_coordinates": "Voxels mapped with (index+0.5)/64-0.5; GLB mapped back using inverse official Z-up -> Y-up rotation; no GT alignment and per-part colors, not original materials.",
    }
    write_json(output / "evidence.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("third_party/PhysX-Omni"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    result = trace_failures(args.bundle, args.upstream, args.output, args.figures)
    print(f"Traced {len(result['samples'])} saved samples; no inference")


if __name__ == "__main__":
    main()
