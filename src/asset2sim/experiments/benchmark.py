"""Freeze, generate and evaluate the paired benchmark across Conda environments."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest, load_manifest, select_samples
from asset2sim.evaluation.benchmark import evaluate
from asset2sim.paths import PROJECT_ROOT

MODELS = {"sf3d": "asset2sim-sf3d", "spar3d": "asset2sim-spar3d", "trellis2": "trellis2"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "pilot", "formal", "evaluate"])
    parser.add_argument("run_id")
    parser.add_argument("samples", nargs="*")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/generation_gso32.jsonl")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/generation_benchmark.yaml")
    )
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warm_repeats < 0 or not 0 <= args.seed < 2**32:
        parser.error("warm_repeats must be nonnegative; seed must fit uint32")
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_id
    ):
        parser.error("run_id must contain letters, digits, underscore or hyphen")
    if args.mode == "freeze":
        records = load_manifest(args.manifest)
        for record in records:
            for key in ("reference_mesh", "render_metadata"):
                if not Path(record.get(key, "")).is_file():
                    raise FileNotFoundError(f"{record['sample_id']}: {record.get(key)}")
            if not record.get("license") or record.get("category") in {None, "unverified"}:
                raise ValueError("Freeze requires audited category and license for every input")
            if record.get("pbr_eligible") and not {"baseColor", "metallic", "roughness"}.issubset(
                record.get("material_channels", {})
            ):
                raise ValueError(
                    "PBR eligibility requires declarations for all three material channels"
                )
        if any(r.get("pbr_eligible") for r in records):
            hdri = yaml.safe_load(args.config.read_text()).get("hdri")
            if not hdri or not Path(hdri).is_file():
                raise FileNotFoundError("PBR reference subset requires an accessible HDRI")
        target = args.freeze or Path(f"experiments/datasets/{args.run_id}_frozen.json")
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            target,
            {
                "manifest_text": args.manifest.read_text(),
                "evaluation_config": args.config.read_text(),
                "model_configs": {
                    name: (PROJECT_ROOT / "configs/models" / f"{name}.yaml").read_text()
                    for name in MODELS
                },
                "warm_repeats": args.warm_repeats,
                "objects": sorted({r["object_id"] for r in records}),
                "samples": [r["sample_id"] for r in records],
                "selection_rule": "reference audit completed before generation",
            },
        )
        print(
            f"Frozen {len(records)} inputs / {len({r['object_id'] for r in records})} objects: {target}"
        )
        return
    base = PROJECT_ROOT / "results/raw/generation_benchmark" / args.run_id
    runs = {
        model: PROJECT_ROOT / "data/generated" / model / "generation_core" / args.run_id
        for model in MODELS
    }
    if args.mode == "evaluate":
        plan = json.loads((base / "plan.json").read_text())
        if plan["evaluation_config"] != args.config.read_text():
            raise ValueError("Evaluation configuration differs from the recorded run plan")
        evaluate(base / "inputs.jsonl", runs, base / "evaluation", args.config)
        return
    if args.mode == "formal":
        if args.samples:
            parser.error("formal uses the entire frozen manifest; freeze a subset for seed repeats")
        if not args.freeze:
            parser.error("formal requires --freeze <frozen.json>")
        frozen = json.loads(args.freeze.read_text())
        if (
            frozen["manifest_text"] != args.manifest.read_text()
            or frozen["evaluation_config"] != args.config.read_text()
            or frozen["model_configs"]
            != {
                name: (PROJECT_ROOT / "configs/models" / f"{name}.yaml").read_text()
                for name in MODELS
            }
            or frozen["warm_repeats"] != args.warm_repeats
        ):
            raise ValueError("Manifest or evaluation settings differ from the frozen protocol")
    if base.exists() or any(path.exists() for path in runs.values()):
        raise FileExistsError("Use a new run_id; existing outputs are preserved")
    records = select_samples(load_manifest(args.manifest), args.samples or None)
    if args.mode == "pilot" and not args.samples:
        nominated = [record for record in records if record.get("pilot")]
        pool = nominated or records
        selected = {}
        for record in pool:
            selected.setdefault(record["object_id"], record)
        records = list(selected.values())[:3]
    for record in records:
        for field in ("reference_mesh", "render_metadata"):
            if not Path(record.get(field, "")).is_file():
                raise FileNotFoundError(record.get(field))
    base.mkdir(parents=True)
    export_manifest(records, base / "inputs.jsonl")
    plan = {
        "mode": args.mode,
        "seed": args.seed,
        "planned_inputs": len(records),
        "planned_objects": len({r["object_id"] for r in records}),
        "warm_repeats": args.warm_repeats,
        "models": MODELS,
        "status": {},
        "evaluation_config": args.config.read_text(),
    }
    write_json(base / "plan.json", plan)
    for model, environment in MODELS.items():
        config = base / f"{model}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "name": "generation_core",
                    "model": model,
                    "model_config": str(PROJECT_ROOT / "configs/models" / f"{model}.yaml"),
                    "manifest": str(base / "inputs.jsonl"),
                    "overrides": {
                        "seed": args.seed,
                        **({"low_vram": False} if model == "spar3d" else {}),
                    },
                }
            )
        )
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "asset2sim",
            "run",
            "--config",
            str(config),
            "--output",
            str(runs[model]),
            "--warm-repeats",
            str(args.warm_repeats),
        ]
        print(f"Starting {model}; log: {base / (model + '.log')}", flush=True)
        try:
            with (base / f"{model}.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            plan["status"][model] = {"exit_code": result.returncode, "command": command}
        except OSError as error:
            plan["status"][model] = {"error": str(error), "command": command}
        write_json(base / "plan.json", plan)
    if any(state.get("exit_code") != 0 for state in plan["status"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
