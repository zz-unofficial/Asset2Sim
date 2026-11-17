"""Explicit, coordinate-preserving cleanup of triangular OBJ files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def remove_zero_area_obj_faces(source: Path, output: Path) -> dict:
    """Copy an OBJ without zero-area faces, preserving material and vertex records."""
    if source.resolve() == output.resolve():
        raise ValueError("Cleaned OBJ must not overwrite its source")
    vertices: list[np.ndarray] = []
    kept: list[str] = []
    removed: list[int] = []
    face_count = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = line.split()
            if tokens and tokens[0] == "v":
                point = np.asarray([float(value) for value in tokens[1:4]])
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError(f"Invalid vertex at OBJ line {line_number}")
                vertices.append(point)
            if tokens and tokens[0] == "f":
                face_count += 1
                if len(tokens) != 4:
                    raise ValueError(f"Expected triangular face at OBJ line {line_number}")
                points = []
                for token in tokens[1:]:
                    index = int(token.split("/", 1)[0])
                    index = index - 1 if index > 0 else len(vertices) + index
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"Invalid face index at OBJ line {line_number}")
                    points.append(vertices[index])
                if not np.any(np.cross(points[1] - points[0], points[2] - points[0])):
                    removed.append(line_number)
                    continue
            kept.append(line)
    if not face_count or face_count == len(removed):
        raise ValueError("OBJ has no remaining faces")
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.writelines(kept)
    return {
        "operation": "remove_zero_area_triangles",
        "source": str(source),
        "output": str(output),
        "vertices": len(vertices),
        "faces_before": face_count,
        "faces_after": face_count - len(removed),
        "removed_face_lines": removed,
    }
