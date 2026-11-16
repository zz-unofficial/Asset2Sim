"""Partition objects across workers and combine their complete evaluation records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from asset2sim.artifacts import read_jsonl, write_json
from asset2sim.data.manifest import export_manifest, load_manifest
from asset2sim.evaluation.benchmark import generation_cost_records, write_csv, write_reports


def merge(manifest: Path, runs: dict[str, Path], shards: list[Path], output: Path) -> None:
    """Require exactly one result per planned model/input/track, including failures."""
    samples = load_manifest(manifest)
    expected = {
        (m, s["sample_id"], t)
        for m in runs
        for s in samples
        for t in ("export_pose", "aligned_diagnostic")
    }
    rows, views, keys = [], [], set()
    for shard in shards:
        for row in read_jsonl(shard / "progress.jsonl"):
            key = (row["model"], row["sample_id"], row["track"])
            if key not in expected or key in keys:
                raise ValueError(f"Unexpected or duplicate shard result: {key}")
            keys.add(key)
            rows.append(row)
        views.extend(read_jsonl(shard / "view_progress.jsonl"))
    if keys != expected:
        raise ValueError(f"Missing shard results: {expected - keys}")
    output.mkdir(parents=True, exist_ok=False)
    for shard in shards:
        for row in read_jsonl(shard / "progress.jsonl"):
            rel = Path("assets") / row["model"] / row["sample_id"] / row["track"]
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((shard / rel).resolve(), target_is_directory=True)
    for name, records in (("progress.jsonl", rows), ("view_progress.jsonl", views)):
        (output / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    for name, records in (
        ("per_sample.csv", rows),
        ("per_view.csv", views),
        ("failures.csv", [r for r in rows if r["status"] != "ok"]),
    ):
        write_csv(output / name, records)
    costs = generation_cost_records(samples, runs)
    write_csv(output / "generation_costs.csv", costs)
    audits = []
    for path in {s["reference_audit"] for s in samples if s.get("reference_audit")}:
        audits.extend(json.loads(Path(path).read_text()))
    write_json(
        output / "sources.json",
        {
            "manifest": str(manifest.resolve()),
            "shards": [str(p.resolve()) for p in shards],
            "runs": {m: str(p.resolve()) for m, p in runs.items()},
        },
    )
    write_reports(samples, runs, output, rows, views, costs, audits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    samples = load_manifest(args.manifest)
    groups = sorted({s["object_id"] for s in samples})
    if not 1 <= args.workers <= len(groups):
        parser.error("workers must be between 1 and the object count")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "execution.json",
        vars(args)
        | {
            "manifest": str(args.manifest),
            "config": str(args.config),
            "output": str(args.output),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
    )
    jobs, logs, shards = [], [], []
    try:
        for i in range(args.workers):
            ids = set(groups[i :: args.workers])
            manifest = args.output / f"shard_{i}.jsonl"
            export_manifest([s for s in samples if s["object_id"] in ids], manifest)
            shard = args.output / f"shard_{i}"
            command = [
                sys.executable,
                "-m",
                "asset2sim.evaluation.benchmark",
                "--manifest",
                str(manifest),
                "--config",
                str(args.config),
                "--output",
                str(shard),
            ]
            for run in args.run:
                command.extend(["--run", run])
            if args.checkpoint:
                command.extend(["--checkpoint", str(args.checkpoint)])
            log = (args.output / f"shard_{i}.log").open("x")
            logs.append(log)
            jobs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            shards.append(shard)
        exits = [job.wait() for job in jobs]
        write_json(args.output / "worker_exits.json", exits)
        if any(exits):
            raise RuntimeError(f"Evaluation worker failures: {exits}; inspect shard logs")
    finally:
        for log in logs:
            log.close()
    runs = {m: Path(p) for m, p in (v.split("=", 1) for v in args.run)}
    merge(args.manifest, runs, shards, args.output / "combined")


if __name__ == "__main__":
    main()
