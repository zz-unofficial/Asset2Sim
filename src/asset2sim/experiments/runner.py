"""Run the same input manifest through any registered generator."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asset2sim.artifacts import git_revision, write_json
from asset2sim.data.manifest import load_manifest, select_samples
from asset2sim.generation.base import Generator
from asset2sim.paths import PROJECT_ROOT


def run_generation(
    generator: Generator,
    manifest: Path,
    output: Path,
    sample_ids: list[str] | None = None,
    experiment: dict[str, Any] | None = None,
    warm_repeats: int = 0,
) -> int:
    """Save run metadata and per-input results; return the failed sample count."""
    records = select_samples(load_manifest(manifest), sample_ids)
    if warm_repeats < 0:
        raise ValueError("warm_repeats must be nonnegative")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "schema_version": 1,
        "run_id": output.name,
        "model": generator.model_name,
        "experiment": experiment or {},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "loading",
        "config": generator.parameters,
        "manifest": str(manifest.resolve()),
        "samples": [record["sample_id"] for record in records],
        "git_commit": git_revision(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "warm_repeats": warm_repeats,
    }
    write_json(output / "run.json", run)
    with (output / "inputs.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        start = time.perf_counter()
        generator.load()
        run["model_load_seconds"] = time.perf_counter() - start
        run.update(generator.metadata())
    except Exception as error:
        run.update(status="load_failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    run["status"] = "running"
    write_json(output / "run.json", run)
    failures = 0
    try:
        if warm_repeats:
            timing_dir = output / "timing"
            timing_dir.mkdir()
            timing = []
            # Warmup is explicit and excluded from both quality and timing statistics.
            for repeat in range(warm_repeats + 1):
                result = {
                    "repeat": repeat,
                    "warmup": repeat == 0,
                    "sample_id": records[0]["sample_id"],
                }
                try:
                    result.update(
                        generator.generate(
                            Path(records[0]["image"]), timing_dir / f"repeat_{repeat:03d}"
                        )
                    )
                    result["status"] = "ok"
                except Exception as error:  # noqa: BLE001 - retain failed timing attempts
                    result.update(status="failed", error=f"{type(error).__name__}: {error}")
                    generator.recover()
                timing.append(result)
                write_json(output / "timing.json", timing)
        with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                sample_id = record["sample_id"]
                result = {
                    "sample_id": sample_id,
                    "object_id": record["object_id"],
                    "view_id": record["view_id"],
                    "split": record.get("split"),
                    "category": record.get("category"),
                    "input": record["image"],
                    "reference_mesh": record.get("reference_mesh"),
                }
                try:
                    result.update(generator.generate(Path(record["image"]), output / sample_id))
                    if not (output / sample_id / "mesh.glb").is_file():
                        raise RuntimeError("Generator did not export mesh.glb")
                    result.update(status="ok", mesh=f"{sample_id}/mesh.glb")
                    print(
                        f"{sample_id}: {result['faces']} faces, {result['inference_seconds']:.2f}s"
                    )
                except Exception as error:  # noqa: BLE001 - record every failed sample
                    failures += 1
                    result.update(status="failed", error=f"{type(error).__name__}: {error}")
                    print(f"{sample_id}: {result['error']}", file=sys.stderr)
                    try:
                        generator.recover()
                    except Exception as recovery_error:  # noqa: BLE001 - retain recovery evidence
                        result["recovery_error"] = str(recovery_error)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        raise
    else:
        run["status"] = "completed" if not failures else "completed_with_failures"
    finally:
        run.update(failures=failures, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "run.json", run)
    return failures
