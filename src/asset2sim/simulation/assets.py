"""Index generated physics assets against the shared image manifest and generation run."""

from __future__ import annotations

import json
from pathlib import Path

from asset2sim.artifacts import read_jsonl
from asset2sim.data.manifest import load_manifest


def index_generated_assets(
    manifest: Path,
    run: Path,
    assets_root: Path,
    conditions: tuple[str, ...] = ("C0", "C1", "C2", "C3"),
) -> list[dict]:
    """Link sample IDs to verified Task A/B assets; never use reference geometry.

    Directory contract: prepared/<sample_id>/normalized.glb and
    collisions/<sample_id>/<condition>/colliders.json under assets_root.
    Missing or failed assets are returned with a reason for each sample.
    """
    run, assets_root = Path(run).resolve(), Path(assets_root).resolve()
    generated = read_jsonl(run / "results.jsonl")
    by_id = {}
    for row in generated:
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"Missing or duplicate generation sample_id: {sample_id!r}")
        by_id[sample_id] = row
    output = []
    for item in load_manifest(manifest):
        sid = item["sample_id"]
        record = {
            "sample_id": sid,
            "object_id": item["object_id"],
            "view_id": item["view_id"],
            "status": "failed",
            "reason": None,
            "prepared_mesh": None,
            "colliders": {},
        }
        try:
            generation = by_id.get(sid)
            if not generation or generation.get("status") != "ok" or not generation.get("mesh"):
                raise ValueError("generation result missing or unsuccessful")
            if any(generation.get(key, item[key]) != item[key] for key in ("object_id", "view_id")):
                raise ValueError("generation object/view identity differs from manifest")
            mesh = (run / generation["mesh"]).resolve()
            if not mesh.is_relative_to(run) or not mesh.is_file():
                raise ValueError("generated mesh path is missing or outside generation run")
            prepared_dir = assets_root / "prepared" / sid
            prepared = prepared_dir / "normalized.glb"
            metadata = json.loads((prepared_dir / "metadata.json").read_text())
            if (
                not prepared.is_file()
                or metadata.get("status") != "completed"
                or metadata.get("options", {}).get("kind") != "generated"
                or Path(metadata.get("source", "")).resolve() != mesh
                or metadata.get("physical_coordinates") != {"up_axis": "+Z", "units": "meters"}
            ):
                raise ValueError(
                    "prepared mesh provenance or coordinates do not match generation result"
                )
            colliders = {}
            for condition in conditions:
                if condition not in {"C0", "C1", "C2", "C3"}:
                    raise ValueError(f"Unknown collision condition: {condition}")
                path = assets_root / "collisions" / sid / condition / "colliders.json"
                collision = json.loads(path.read_text())
                if (
                    collision.get("status") != "completed"
                    or collision.get("condition") != condition
                    or Path(collision.get("input", "")).resolve() != prepared.resolve()
                    or len(collision.get("components", [])) != collision.get("component_count")
                    or not collision.get("components")
                ):
                    raise ValueError(
                        f"collision provenance or component count mismatch: {condition}"
                    )
                for component in collision["components"]:
                    component_path = (path.parent / component["path"]).resolve()
                    if (
                        component_path.parent != path.parent.resolve()
                        or not component_path.is_file()
                    ):
                        raise ValueError(f"collision component missing or unsafe: {component_path}")
                colliders[condition] = str(path.resolve())
            record.update(status="ready", prepared_mesh=str(prepared), colliders=colliders)
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["reason"] = str(error)
        output.append(record)
    return output
