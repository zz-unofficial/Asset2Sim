"""Server identity gate: reference meshes traverse the complete render evaluator."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest, load_manifest
from asset2sim.evaluation.benchmark import evaluate
from asset2sim.experiments.config import load_parameters


def validate(manifest: Path, config_path: Path, output: Path) -> None:
    records = load_manifest(manifest)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "identity_run"
    run.mkdir()
    write_json(run / "run.json", {"model": "identity", "config": {"seed": 0}})
    export_manifest(records, run / "inputs.jsonl")
    # The evaluator only needs input identities from this snapshot; mesh paths stay absolute.
    (run / "results.jsonl").write_text(
        "".join(
            json.dumps(
                {"sample_id": record["sample_id"], "status": "ok", "mesh": record["reference_mesh"]}
            )
            + "\n"
            for record in records
        )
    )
    config = load_parameters(config_path)
    config["export_to_canonical"] = {"identity": np.eye(4).tolist()}
    # The identity export is already in the reference pose for each conditioning view.
    for record in records:
        camera = json.loads(Path(record["render_metadata"]).read_text())
        pose = np.asarray(camera["views"][camera["conditioning_view"]]["camera_to_world"])
        from asset2sim.data.synthetic import camera_pose

        expected = camera_pose(0, 20, 2.5)
        if not np.allclose(pose[:3, :3], expected[:3, :3]):
            raise ValueError(
                "Identity gate uses canonical view 000 only; prepare D0 with canonical_view=0"
            )
    profile = output / "identity.yaml"
    profile.write_text(yaml.safe_dump(config))
    evaluate(manifest, {"identity": run}, output / "evaluation", profile)
    with (output / "evaluation/per_sample.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["track"] == "export_pose"]
    passed = all(
        row["status"] == "ok"
        and float(row["chamfer_l1"]) < 0.03
        and float(row["texture_mask_iou"]) > 0.99999
        and float(row["texture_ssim"]) > 0.99999
        and abs(float(row["texture_lpips"])) < 1e-5
        for row in rows
    )
    write_json(
        output / "validation.json",
        {
            "passed": passed,
            "valid": sum(row["status"] == "ok" for row in rows),
            "planned": len(records),
            "cd_tolerance_normalized": 0.03,
            "psnr_cap_db": 120,
        },
    )
    if not passed:
        raise RuntimeError(
            "Identity gate failed; inspect validation.json and evaluation/failures.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/generation_benchmark.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.manifest, args.config, args.output)
