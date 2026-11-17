"""Replay upstream voxel decoding without importing its model runtime."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rle(asset: Path, upstream: Path) -> dict:
    """Replay official pure RLE helpers and compare every saved part coordinate."""
    import re
    import typing

    tree = ast.parse((upstream / "1vlm_demo.py").read_text())
    helpers = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "addmessage":
                break
            if node.name != "get_logger":
                helpers.append(node)
    namespace = {**vars(typing), "np": np, "re": re}
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *helpers,
        ],
        type_ignores=[],
    )
    # Execute only pure helper definitions from the pinned, trusted upstream checkout.
    exec(compile(ast.fix_missing_locations(module), "<official-rle-helpers>", "exec"), namespace)  # noqa: S102
    checked = []
    for path in sorted(asset.glob("ind_*.npy")):
        index = path.stem.removeprefix("ind_")
        text = (asset / f"coord_{index}.txt").read_text()
        if not text.strip():
            raise ValueError(f"Empty RLE response: {index}")
        runs = namespace["string_to_runs_by_z_lossless_robust"](text, D=64)
        decoded = namespace["decode_voxel_2drle_by_z"](runs, shape=(64, 64, 64))
        saved = np.load(path, allow_pickle=False)
        if not decoded.size or not np.array_equal(decoded, saved):
            raise ValueError(f"RLE replay differs from saved part: {index}")
        checked.append(index)
    return {"rle_replayed_parts": checked, "rle_parser": "official_robust_parser"}
