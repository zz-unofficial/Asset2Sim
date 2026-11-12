"""Shared experiment commands and the original SF3D shortcuts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asset2sim.experiments.config import ExperimentConfig, load_parameters
from asset2sim.paths import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(prog="asset2sim")
    commands = parser.add_subparsers(dest="command", required=True)
    physics = commands.add_parser("prepare-mesh", help="Prepare mesh geometry for physics")
    physics.add_argument("--mesh", type=Path, required=True)
    physics.add_argument("--kind", choices=("reference", "generated"), required=True)
    physics.add_argument("--run", type=Path, help="Generator run directory for generated meshes")
    physics.add_argument("--output", type=Path, required=True)
    physics.add_argument("--target-extent", type=float, default=0.20)
    physics.add_argument("--remove-degenerate", action="store_true")
    physics.add_argument("--remove-unreferenced", action="store_true")
    physics.add_argument("--merge-coincident", action="store_true")
    physics.add_argument("--min-triangle-area", type=float, default=0.0)
    collision = commands.add_parser("build-collision", help="Build convex collision components")
    collision.add_argument("--mesh", type=Path, required=True)
    collision.add_argument("--condition", choices=("C0", "C1", "C2", "C3"), required=True)
    collision.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/collision/default.yaml"
    )
    collision.add_argument("--output", type=Path, required=True)
    collision.add_argument("--allow-open", action="store_true")
    collision.add_argument("--assume-prepared", action="store_true")
    physics_eval = commands.add_parser("physics-eval", help="Evaluate MuJoCo contact behavior")
    physics_eval.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/physics/debug.yaml"
    )
    physics_eval.add_argument("--output", type=Path, required=True)
    physics_index = commands.add_parser("physics-index", help="Index generated assets for physics")
    physics_index.add_argument("--manifest", type=Path, required=True)
    physics_index.add_argument("--run", type=Path, required=True)
    physics_index.add_argument("--assets", type=Path, required=True)
    physics_index.add_argument("--condition", choices=("C0", "C1", "C2", "C3"), action="append")
    physics_index.add_argument("--output", type=Path, required=True)
    physics_batch = commands.add_parser(
        "physics-batch", help="Prepare and probe a small asset batch"
    )
    physics_batch.add_argument("--manifest", type=Path, required=True)
    source = physics_batch.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--reference", action="store_true")
    physics_batch.add_argument("--sample-id", action="append")
    physics_batch.add_argument(
        "--condition", choices=("C0", "C1", "C2", "C3"), action="append", required=True
    )
    physics_batch.add_argument("--output", type=Path, required=True)
    physics_batch.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/physics/debug.yaml"
    )
    physics_batch.add_argument("--merge-coincident", action="store_true")
    physics_batch.add_argument("--remove-degenerate", action="store_true")
    physics_batch.add_argument("--remove-unreferenced", action="store_true")
    oracle = commands.add_parser(
        "physics-oracle", help="Run isolated offline reference-fit diagnostic"
    )
    oracle.add_argument("--batch", type=Path, required=True)
    oracle.add_argument("--manifest", type=Path, required=True)
    oracle.add_argument("--sample-id", required=True)
    oracle.add_argument(
        "--condition", choices=("C0", "C1", "C2", "C3"), action="append", required=True
    )
    oracle.add_argument("--output", type=Path, required=True)
    paper_data = commands.add_parser("prepare-paper-data", help="Prepare public paper datasets")
    paper_data.add_argument("--config", type=Path, required=True)
    paper_data.add_argument("--output", type=Path, required=True)
    paper_data.add_argument("--manifest", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Measure paper geometry and image metrics")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("paper-report", help="Compare measured results with paper rows")
    report.add_argument("evaluations", type=Path, nargs="+")
    report.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run a model with an experiment configuration")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--sample-id", action="append")
    fetch = commands.add_parser("fetch", help="Cache a registered model's assets")
    fetch.add_argument("--model", required=True)
    fetch.add_argument("--config", type=Path)
    summary = commands.add_parser("summarize", help="Export run directories to a per-sample CSV")
    summary.add_argument("runs", type=Path, nargs="+")
    summary.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-manifest", help="Publish a split with rebased input paths")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--split", required=True)
    export.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-sf3d-data", help="Prepare references and official examples"
    )
    prepare.add_argument("--output", type=Path, default=Path("data/samples/sf3d"))
    legacy_fetch = commands.add_parser("fetch-sf3d", help="SF3D shortcut for fetch")
    legacy_fetch.add_argument("--config", type=Path, default=Path("configs/sf3d.yaml"))
    legacy_run = commands.add_parser("sf3d", help="SF3D shortcut with explicit input/output paths")
    legacy_run.add_argument("--config", type=Path, default=Path("configs/sf3d.yaml"))
    legacy_run.add_argument("--manifest", type=Path, required=True)
    legacy_run.add_argument("--output", type=Path, required=True)
    legacy_run.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    try:
        if args.command == "physics-oracle":
            from asset2sim.simulation.oracle import run_oracle_diagnostic

            result = run_oracle_diagnostic(
                args.batch, args.manifest, args.sample_id, args.condition, args.output
            )
            print(f"Offline oracle diagnostic {result['status']}: {args.output}")
            return 0 if result["status"] == "completed" else 1
        if args.command == "physics-batch":
            from asset2sim.simulation.batch import run_physics_batch

            result = run_physics_batch(
                args.manifest,
                args.output,
                args.condition,
                args.sample_id,
                run=args.run,
                reference=args.reference,
                merge_coincident=args.merge_coincident,
                remove_degenerate=args.remove_degenerate,
                remove_unreferenced=args.remove_unreferenced,
                config_path=args.config,
            )
            print(f"Physics batch {result['status']}: {args.output}")
            return 0 if result["status"] == "completed" else 1
        if args.command == "physics-index":
            import json

            from asset2sim.simulation.assets import index_generated_assets

            records = index_generated_assets(
                args.manifest,
                args.run,
                args.assets,
                tuple(args.condition or ("C0", "C1", "C2", "C3")),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(records, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            print(
                f"Indexed {sum(row['status'] == 'ready' for row in records)}/{len(records)} samples: {args.output}"
            )
            return 0 if all(row["status"] == "ready" for row in records) else 1
        if args.command == "physics-eval":
            from asset2sim.simulation.evaluate import run_physics_eval

            result = run_physics_eval(args.config, args.output)
            print(f"Physics evaluation {result['status']}: {args.output}")
            return 0 if result["status"] == "completed" else 1
        if args.command == "build-collision":
            from asset2sim.collision import build_collision

            result = build_collision(
                args.mesh,
                args.output,
                args.condition,
                args.config,
                allow_open=args.allow_open,
                assume_prepared=args.assume_prepared,
            )
            print(f"Built {result['component_count']} collision components: {args.output}")
            return 0
        if args.command == "prepare-mesh":
            from asset2sim.geometry.prepare import PreparationOptions, prepare_mesh

            options = PreparationOptions(
                args.kind,
                args.target_extent,
                args.remove_degenerate,
                args.remove_unreferenced,
                args.min_triangle_area,
                args.merge_coincident,
            )
            prepare_mesh(args.mesh, args.output, options, args.run)
            print(f"Prepared physics mesh: {args.output}")
            return 0
        if args.command == "prepare-paper-data":
            from asset2sim.data.paper import prepare_paper_data

            count = prepare_paper_data(args.config, args.output, args.manifest)
            print(f"Prepared {count} paper inputs: {args.manifest}")
            return 0
        if args.command == "evaluate":
            from asset2sim.evaluation.paper import evaluate_run

            result = evaluate_run(args.run, args.config, args.output)
            return (
                0
                if all(
                    group["planned"] == group["successful"] for group in result["datasets"].values()
                )
                else 1
            )
        if args.command == "paper-report":
            from asset2sim.evaluation.report import write_report

            write_report(args.evaluations, args.output)
            return 0
        if args.command == "prepare-sf3d-data":
            from asset2sim.data.sf3d import prepare_data

            prepare_data(args.output)
            return 0
        if args.command == "export-manifest":
            from asset2sim.data.manifest import export_manifest, load_manifest

            records = [row for row in load_manifest(args.source) if row.get("split") == args.split]
            if not records:
                raise ValueError(f"No samples in split {args.split!r}")
            export_manifest(records, args.output)
            print(f"Published {len(records)} inputs: {args.output}")
            return 0
        if args.command == "summarize":
            from asset2sim.experiments.summary import export_runs

            count = export_runs(args.runs, args.output)
            print(f"Exported {count} sample records: {args.output}")
            return 0
        from asset2sim.generation.registry import create_generator, fetch_assets

        if args.command in {"fetch", "fetch-sf3d"}:
            model = args.model if args.command == "fetch" else "sf3d"
            path = args.config or PROJECT_ROOT / "configs/models" / f"{model}.yaml"
            print(f"Model assets cached: {fetch_assets(model, load_parameters(path))}")
            return 0
        from asset2sim.experiments.runner import run_generation

        if args.command == "run":
            experiment = ExperimentConfig.load(args.config)
            generator = create_generator(experiment.model, experiment.parameters())
            selected = args.sample_id if args.sample_id is not None else experiment.sample_ids
            failures = run_generation(
                generator,
                experiment.manifest,
                args.output,
                selected,
                experiment={
                    "name": experiment.name,
                    "config": str(args.config.resolve()),
                    "model_config": str(experiment.model_config),
                },
            )
        else:
            generator = create_generator("sf3d", load_parameters(args.config))
            failures = run_generation(generator, args.manifest, args.output, args.sample_id)
        return 1 if failures else 0
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        print(f"asset2sim: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
