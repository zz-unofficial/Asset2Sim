import numpy as np
import trimesh

from asset2sim.evaluation.benchmark import face_component_count, generation_cost_records, topology


def test_face_components_match_trimesh_split_with_isolated_face():
    box = trimesh.creation.box()
    other = box.copy()
    other.apply_translation([3, 0, 0])
    triangle = trimesh.Trimesh(
        vertices=np.array([[6, 0, 0], [7, 0, 0], [6, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    mesh = trimesh.util.concatenate([box, other, triangle])
    expected = len(mesh.split(only_watertight=False))

    assert expected == 3
    assert face_component_count(mesh) == expected
    assert topology(mesh)["raw"]["components"] == expected


def test_generation_cost_denominators_keep_export_failures_and_missing(tmp_path):
    import json

    mesh = tmp_path / "good/mesh.glb"
    mesh.parent.mkdir()
    mesh.write_bytes(b"fixture")
    records = [
        {
            "sample_id": "good",
            "status": "ok",
            "raw_generation_status": "generated",
            "mesh": "good/mesh.glb",
            "inference_seconds": 2,
            "export_seconds": 3,
        },
        {
            "sample_id": "bad_export",
            "status": "failed",
            "raw_generation_status": "generated",
            "failed_stage": "export",
            "inference_seconds": 4,
        },
    ]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    samples = [
        {"sample_id": s, "object_id": s, "view_id": "000"}
        for s in ("good", "bad_export", "missing")
    ]
    rows = generation_cost_records(samples, {"trellis2": tmp_path})
    assert len(rows) == 3
    assert sum(r["generation_valid"] for r in rows) == 2
    assert sum(r["export_valid"] for r in rows) == 1
    assert sum(r["result_missing"] for r in rows) == 1
    assert rows[1]["inference_seconds"] == 4
    assert rows[1]["export_seconds"] is None
