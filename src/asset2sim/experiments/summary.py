"""Export per-sample measurements from current and earlier run directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from asset2sim.artifacts import read_jsonl

COLUMNS = [
    "run_id",
    "experiment",
    "model",
    "device",
    "seed",
    "sample_id",
    "object_id",
    "view_id",
    "split",
    "category",
    "status",
    "vertices",
    "faces",
    "background_mode",
    "preprocess_seconds",
    "inference_seconds",
    "export_seconds",
    "peak_allocated_mib",
    "mesh",
    "error",
]


def run_rows(directory: Path) -> list[dict[str, Any]]:
    directory = directory.resolve()
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    snapshot = directory / "inputs.jsonl"
    if not snapshot.exists():
        snapshot = Path(run.get("manifest", ""))
    inputs = read_jsonl(snapshot) if snapshot.is_file() else []
    by_id = {record["sample_id"]: record for record in inputs}
    experiment = run.get("experiment", {})
    name = experiment.get("name", "") if isinstance(experiment, dict) else str(experiment)
    rows = []
    results = directory / "results.jsonl"
    measured = (
        {record["sample_id"]: record for record in read_jsonl(results)} if results.exists() else {}
    )
    # Include pending/loading failures in the table, so row counts follow the run's input set.
    sample_ids = run.get("samples", list(measured))
    for sample_id in sample_ids:
        result = measured.get(sample_id, {})
        row = by_id.get(sample_id, {}) | result
        row.update(
            run_id=run.get("run_id", directory.name),
            experiment=name,
            model=run.get("model", run.get("config", {}).get("model", "")),
            device=run.get("device", ""),
            seed=run.get("config", {}).get("seed"),
            sample_id=sample_id,
            status=result.get("status", run.get("status", "pending")),
        )
        if not result:
            row["error"] = run.get("error", "")
        if result.get("mesh"):
            row["mesh"] = str(directory / result["mesh"])
        rows.append({key: row.get(key, "") for key in COLUMNS})
    return rows


def export_runs(directories: list[Path], output: Path) -> int:
    rows = [row for directory in directories for row in run_rows(directory)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
