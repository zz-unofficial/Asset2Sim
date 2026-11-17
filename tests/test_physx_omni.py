"""Official pipeline orchestration checks; no model weights or GPU required."""

import json
import sys

import numpy as np
import pytest
import yaml

from asset2sim.experiments.physx_omni import check_stage, run_physx_omni, stage_commands


def make_inputs(tmp_path):
    from PIL import Image

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for name in ("1vlm_demo.py", "decoder_each.py", "3jsongen_update.py"):
        (upstream / name).write_text("pass\n")
    for name in ("dataset", "trellis", "mjcf_source"):
        (upstream / name).mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"upstream": "upstream", "model_path": "weights"}))
    Image.new("RGB", (8, 8)).save(tmp_path / "input.png")
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(json.dumps({"sample_id": "cup", "image": "input.png"}) + "\n")
    return config, manifest, upstream


def test_plan_resolves_paths_and_refuses_overwrite(tmp_path):
    config, manifest, upstream = make_inputs(tmp_path)
    output = tmp_path / "plan"
    result = run_physx_omni(config, manifest, output, plan_only=True)
    assert result["status"] == "planned"
    assert result["n_planned"] == 1
    assert result["config"]["upstream"] == str(upstream)
    assert not (output / "cup").exists()
    assert [x["stage"] for x in result["samples"][0]["commands"]] == ["infer", "decode", "export"]
    with pytest.raises(FileExistsError):
        run_physx_omni(config, manifest, output, plan_only=True)


def test_swallowed_upstream_error_is_failure_and_batch_continues(tmp_path):
    config, manifest, upstream = make_inputs(tmp_path)
    manifest.write_text(
        "\n".join(json.dumps({"sample_id": s, "image": "input.png"}) for s in ("cup", "ring"))
    )
    result = run_physx_omni(config, manifest, tmp_path / "run", validate_mujoco=True)
    assert result["n_failed"] == 2
    assert result["n_success"] == 0
    for item in result["samples"]:
        failed = [entry for entry in item["stages"] if entry["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["returncode"] == 0
        assert failed[0]["stage"] == "infer"
        assert item["stages"][-1]["status"] == "blocked"
    assert not (upstream / "ours_demo").exists()


def test_voxel_validation_rejects_out_of_range(tmp_path):
    (tmp_path / "basic_info.txt").write_text("material prior")
    np.save(tmp_path / "ind_0.npy", [[1, 2, 3]])
    np.save(tmp_path / "allind.npy", [[64, 2, 3]])
    with pytest.raises(ValueError, match="64-cube"):
        check_stage("infer", tmp_path)


def test_decoder_checks_every_part(tmp_path):
    import trimesh

    np.save(tmp_path / "ind_0.npy", [[1, 2, 3]])
    np.save(tmp_path / "ind_1.npy", [[4, 5, 6]])
    part = tmp_path / "objs/0"
    part.mkdir(parents=True)
    for suffix in ("glb", "obj"):
        trimesh.creation.box().export(part / f"0.{suffix}")
    with pytest.raises((ValueError, OSError)):
        check_stage("decode", tmp_path)


def test_commands_preserve_space_paths_without_shell(tmp_path):
    commands = stage_commands(
        {
            "upstream": str(tmp_path / "source code"),
            "model_path": "/data/model files",
            "postprocess": False,
            "fixed_base": False,
            "deformable": False,
        },
        tmp_path / "run files",
    )
    assert commands[0][1][0] == sys.executable
    assert commands[0][1][-1] == "/data/model files"
    assert commands[1][1][1].endswith("decoder_each.py")
    assert "--process" in commands[2][1]


@pytest.mark.parametrize("load_failure", [False, True])
def test_complete_pipeline_records_load_result(tmp_path, monkeypatch, load_failure):
    import subprocess

    import trimesh

    from asset2sim.experiments import physx_omni

    config, manifest, _ = make_inputs(tmp_path)

    def execute(argv, *, cwd, **kwargs):
        asset = cwd / "ours_demo/asset"
        asset.mkdir(parents=True, exist_ok=True)
        if argv[1].endswith("1vlm_demo.py"):
            (asset / "basic_info.txt").write_text(
                "l_0: body | 1 | wood | 0.6 | 0.01 | 0.3 | fixture"
            )
            np.save(asset / "allind.npy", [[1, 2, 3]])
            np.save(asset / "ind_0.npy", [[1, 2, 3]])
        elif argv[1].endswith("decoder_each.py"):
            part = asset / "objs/0"
            part.mkdir(parents=True)
            for suffix in ("glb", "obj"):
                trimesh.creation.box().export(part / f"0.{suffix}")
        else:
            (asset / "basic_info.json").write_text(json.dumps({"parts": [{"material": "wood"}]}))
            (asset / "basic.xml").write_text("<mujoco/>")
            (asset / "basic.urdf").write_text('<robot name="asset"/>')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(physx_omni.subprocess, "run", execute)
    monkeypatch.setattr(physx_omni, "git_revision", lambda _: "test")

    def load(_):
        if load_failure:
            raise ValueError("invalid inertia")
        return {"geoms": 1}

    monkeypatch.setattr(physx_omni, "validate_mjcf", load)
    result = run_physx_omni(config, manifest, tmp_path / "run", validate_mujoco=True)
    if load_failure:
        assert result["status"] == "failed"
        assert result["n_success"] == 0
        assert result["n_failed"] == 1
        assert result["samples"][0]["stages"][-1]["stage"] == "mujoco_load"
        assert "invalid inertia" in result["samples"][0]["stages"][-1]["reason"]
        return
    assert result["status"] == "complete"
    assert result["n_success"] == 1
    stages = result["samples"][0]["stages"]
    assert len(stages) == 5
    assert all(stage["status"] == "ok" for stage in stages)
    assert stages[-1]["geoms"] == 1


def test_real_mujoco_compile_and_forward(tmp_path):
    pytest.importorskip("mujoco")
    from asset2sim.experiments.physx_omni import validate_mjcf

    (tmp_path / "basic.xml").write_text(
        '<mujoco><worldbody><body pos="0 0 0.2"><freejoint/>'
        '<geom type="box" size="0.1 0.1 0.1" mass="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    result = validate_mjcf(tmp_path)
    assert result["geoms"] == 1
    assert result["joints"] == 1
    assert result["body_mass_kg"][1] == pytest.approx(0.1)


def test_export_resume_reuses_geometry_and_does_not_load_mujoco(tmp_path, monkeypatch):
    import subprocess

    import trimesh
    from PIL import Image

    from asset2sim.experiments import physx_omni

    config, manifest, _ = make_inputs(tmp_path)
    source = tmp_path / "previous/cup"
    (source / "inputs").mkdir(parents=True)
    Image.open(tmp_path / "input.png").save(source / "inputs/asset.png")
    asset = source / "ours_demo/asset"
    (asset / "objs/0").mkdir(parents=True)
    (asset / "basic_info.txt").write_text("l_0: body | 1 | wood | 0.6 | 0.01 | 0.3 | fixture")
    for name in ("allind", "ind_0"):
        np.save(asset / f"{name}.npy", [[1, 2, 3]])
    for suffix in ("glb", "obj"):
        trimesh.creation.box().export(asset / f"objs/0/0.{suffix}")
    calls = []

    def execute(argv, *, cwd, **kwargs):
        calls.append(argv[1])
        target = cwd / "ours_demo/asset"
        (target / "basic_info.json").write_text('{"parts": [{"material": "wood"}]}')
        (target / "basic.xml").write_text("<mujoco/>")
        (target / "basic.urdf").write_text('<robot name="asset"/>')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(physx_omni.subprocess, "run", execute)
    monkeypatch.setattr(physx_omni, "git_revision", lambda _: "test")
    monkeypatch.setattr(
        physx_omni, "validate_mjcf", lambda _: pytest.fail("Static run imported MuJoCo")
    )
    result = run_physx_omni(
        config, manifest, tmp_path / "new", reuse_run=tmp_path / "previous", from_stage="export"
    )
    assert result["n_success"] == 1
    assert len(calls) == 1 and calls[0].endswith("3jsongen_update.py")
    assert not (asset / "basic.xml").exists()
    assert result["samples"][0]["stages"][1]["reused"]
    assert result["samples"][0]["stages"][2]["reused"]


def test_stage_timeout_is_recorded_and_later_stages_blocked(tmp_path, monkeypatch):
    import subprocess

    from asset2sim.experiments import physx_omni

    config, manifest, _ = make_inputs(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(physx_omni.subprocess, "run", timeout)
    monkeypatch.setattr(physx_omni, "git_revision", lambda _: "test")
    result = run_physx_omni(config, manifest, tmp_path / "run")
    assert result["n_failed"] == 1
    assert "TimeoutExpired" in result["samples"][0]["stages"][1]["reason"]
    assert result["samples"][0]["stages"][-1]["status"] == "blocked"


def test_category_export_override_and_explicit_baseline(tmp_path):
    config, manifest, _ = make_inputs(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["category_overrides"] = {"deformable": {"deformable": True}}
    config.write_text(yaml.safe_dump(value))
    record = json.loads(manifest.read_text())
    record["category"] = "deformable"
    manifest.write_text(json.dumps(record))
    activated = run_physx_omni(config, manifest, tmp_path / "activated", plan_only=True)
    baseline = run_physx_omni(
        config, manifest, tmp_path / "baseline", plan_only=True, deformable=False
    )
    assert activated["samples"][0]["commands"][-1]["argv"][-1] == "1"
    assert baseline["samples"][0]["commands"][-1]["argv"][-1] == "0"
