"""Shared single-image manifests and stable object/view identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from asset2sim.artifacts import read_jsonl


def load_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    records = read_jsonl(path)
    seen = set()
    for record in records:
        sample_id = record.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or Path(sample_id).name != sample_id
            or sample_id in {".", ".."}
        ):
            raise ValueError(f"Unsafe sample_id: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if not record.get("image"):
            raise ValueError(f"Missing image for {sample_id}")
        for key in ("image", "reference_mesh", "render_metadata", "reference_audit"):
            if record.get(key):
                resolved = (path.parent / record[key]).resolve()
                if key == "image" and not resolved.is_file():
                    raise FileNotFoundError(resolved)
                record[key] = str(resolved)
        record.setdefault("object_id", sample_id)
        record.setdefault("view_id", "provided")
    if not records:
        raise ValueError("The manifest contains no samples")
    return records


def select_samples(
    records: list[dict[str, Any]], sample_ids: list[str] | None
) -> list[dict[str, Any]]:
    if sample_ids is None:
        return records
    requested = set(sample_ids)
    available = {record["sample_id"] for record in records}
    if not requested or requested - available:
        raise ValueError(f"Unknown or empty sample selection: {sorted(requested - available)}")
    return [record for record in records if record["sample_id"] in requested]


def export_manifest(records: list[dict[str, Any]], output: Path) -> None:
    """Rebase resolved paths when publishing a reusable dataset manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for original in records:
            record = dict(original)
            for key in ("image", "reference_mesh", "render_metadata", "reference_audit"):
                if record.get(key):
                    record[key] = Path(os.path.relpath(record[key], output.parent)).as_posix()
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

