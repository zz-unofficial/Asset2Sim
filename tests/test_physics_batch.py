import json
from pathlib import Path

import pytest

from asset2sim.simulation.batch import run_physics_batch

MANIFEST = Path("data/manifests/debug.jsonl")


def test_missing_generation_is_recorded_for_each_condition(tmp_path):
    run = tmp_path / "generation"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"model": "sf3d"}))
    (run / "results.jsonl").write_text("")
    result = run_physics_batch(
        MANIFEST, tmp_path / "output", ["C0", "C2"], ["debug_box_v0"], run=run
    )
    assert result["status"] == "completed_with_failures"
    assert len(result["results"]) == 2
    assert all(
        row["failure_stage"] == "generation" and row["generation_status"] == "missing"
        for row in result["results"]
    )
    assert (tmp_path / "output/summary.csv").exists()
    with pytest.raises(FileExistsError):
        run_physics_batch(MANIFEST, tmp_path / "output", ["C0"], ["debug_box_v0"], run=run)


def test_reference_batch_loads_and_evaluates_contact(tmp_path):
    result = run_physics_batch(
        MANIFEST, tmp_path / "reference", ["C0"], ["debug_box_v0"], reference=True
    )
    row = result["results"][0]
    assert row["preparation_status"] == row["collision_status"] == row["mujoco_status"] == "ok"
    assert row["contact_status"] == "reference_evaluated"
    assert row["contact_valid_samples"] > 0
    assert row["contact_planned_samples"] == 162


def test_generated_batch_does_not_report_unaligned_accuracy(tmp_path):
    from asset2sim.simulation.oracle import run_oracle_diagnostic

    run = tmp_path / "generation"
    source = run / "debug_box_v0/mesh.glb"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path("data/samples/sf3d/reference/debug_box.glb").read_bytes())
    run_metadata = {
        "model": "synthetic",
        "raw_mesh_coordinates": {
            "up_axis": "+Z",
            "units": "test",
            "raw_to_model_z_up": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        },
    }
    (run / "run.json").write_text(json.dumps(run_metadata))
    (run / "results.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "debug_box_v0",
                "object_id": "debug_box",
                "view_id": "v0",
                "status": "ok",
                "mesh": "debug_box_v0/mesh.glb",
            }
        )
        + "\n"
    )
    result = run_physics_batch(MANIFEST, tmp_path / "generated", ["C0"], ["debug_box_v0"], run=run)
    row = result["results"][0]
    assert row["index_status"] == "ready" and row["mujoco_status"] == "ok"
    assert row["contact_status"] == "pose_unverified_no_accuracy"
    assert row["contact_valid_samples"] == 0 and row["contact_planned_samples"] == 162
    assert not any(key in row for key in ("fp", "fn", "behavior_consistency"))
    oracle = run_oracle_diagnostic(
        tmp_path / "generated", MANIFEST, "debug_box_v0", ["C0"], tmp_path / "oracle"
    )
    assert oracle["status"] == "completed"
    assert oracle["type"] == "offline_oracle_reference_rigid_fit"
    assert oracle["results"][0]["valid_samples"] > 0
    assert not any(
        key in json.loads((tmp_path / "generated/batch.json").read_text())["results"][0]
        for key in ("fp", "fn", "behavior_consistency")
    )
