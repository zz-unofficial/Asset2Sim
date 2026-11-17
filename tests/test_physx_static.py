"""Category-aware exports, geometry, joint frames and full failure denominators."""

import json

import numpy as np
import pytest
import trimesh

from asset2sim.evaluation.articulation import joint_errors, read_urdf
from asset2sim.evaluation.physx_omni import (
    attribute_metrics,
    audit_asset,
    evaluate_run,
    geometry_metrics,
)


def make_asset(folder, *, flex=False, density=600):
    folder.mkdir(parents=True)
    part = folder / "objs/0"
    part.mkdir(parents=True)
    for suffix in ("obj", "glb"):
        trimesh.creation.box().export(part / f"0.{suffix}")
    (folder / "basic_info.txt").write_text("l_0: body | 1 | wood | 0.6 | 0.01 | 0.3 | fixture\n")
    (folder / "basic_info.json").write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "label": 0,
                        "material": "wood",
                        "density": 0.6,
                        "Young's Modulus (GPa)": 0.01,
                        "Poisson's Ratio": 0.3,
                    }
                ]
            }
        )
    )
    (folder / "basic.urdf").write_text('<robot name="asset"><link name="body"/></robot>')
    flexible = (
        '<flexcomp name="l_0_body" dim="2" file="objs/0/0.obj"><elasticity young="10000000" poisson="0.3" damping="0.001"/></flexcomp>'
        if flex
        else ""
    )
    (folder / "basic.xml").write_text(
        f'<mujoco><asset><mesh name="l_0_body" file="objs/0/0.obj"/></asset>'
        f'<default><default class="l_0_body"><geom density="{density}"/></default></default>'
        f"<worldbody>{flexible}</worldbody></mujoco>"
    )
    return folder


def test_density_conversion_and_disabled_soft_branch(tmp_path):
    folder = make_asset(tmp_path / "asset")
    rigid = audit_asset(folder, "rigid")
    assert rigid["status"] == "ok"
    assert (
        next(row for row in rigid["transfers"] if row["field"] == "density")["status"]
        == "TRANSFORMED"
    )
    assert "deformable_branch_not_activated" in audit_asset(folder, "deformable")["errors"]
    assert "missing_articulated_joint" in audit_asset(folder, "articulated")["errors"]


def test_soft_export_default_damping_and_property_override(tmp_path):
    folder = make_asset(tmp_path / "asset", flex=True)
    audit = audit_asset(folder, "deformable")
    assert audit["status"] == "ok"
    assert any(
        row["field"] == "damping" and row["status"] == "DEFAULTED" for row in audit["transfers"]
    )
    assert "flex_thickness_deferred_to_engine_default" in audit["warnings"]
    bad = make_asset(tmp_path / "bad", density=1000)
    assert any("density:OVERRIDDEN" in error for error in audit_asset(bad, "rigid")["errors"])


def test_missing_prediction_part_cannot_disappear_from_audit(tmp_path):
    folder = make_asset(tmp_path / "asset")
    with (folder / "basic_info.txt").open("a") as stream:
        stream.write("l_1: missing | 1 | wood | 0.6 | 0.01 | 0.3 | fixture\n")
    assert "raw_and_exported_part_ids_differ" in audit_asset(folder, "rigid")["errors"]


def test_attribute_scores_need_independent_unit_matched_truth(tmp_path):
    folder = make_asset(tmp_path / "asset")
    audit = audit_asset(folder, "rigid")
    assert all(row["status"] == "NA" for row in attribute_metrics({}, audit))
    metadata = tmp_path / "sample.json"
    metadata.write_text(
        json.dumps(
            {
                "physical_labels": {
                    "density": {
                        "value": 600,
                        "unit": "kg/m3",
                        "source": "prediction",
                        "ground_truth": True,
                        "evidence_type": "model_inference",
                    }
                }
            }
        )
    )
    assert all(
        row["status"] == "NA"
        for row in attribute_metrics({"sample_metadata": str(metadata)}, audit)
    )


def test_joint_parent_rotation_and_axis_sign_equivalence(tmp_path):
    urdf = tmp_path / "reference.urdf"
    urdf.write_text(
        '<robot name="test"><link name="base"/><link name="door"/>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="door"/>'
        '<origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="2"/></joint></robot>'
    )
    joint = read_urdf(urdf)["joints"][0]
    np.testing.assert_allclose(joint["world_axis"], [0, 1, 0], atol=1e-12)
    predicted = {**joint, "world_axis": [0, -1, 0], "world_origin": [0, 3, 0], "range": [-2, 1]}
    result = joint_errors(predicted, joint, 1)
    assert result["axis_error_degrees"] == pytest.approx(0)
    assert result["axis_anchor_distance_normalized"] == pytest.approx(0)
    assert result["range_endpoint_mae"] == pytest.approx(0)


def test_invalid_kinematic_tree_is_rejected(tmp_path):
    path = tmp_path / "bad.urdf"
    path.write_text('<robot name="bad"><link name="a"/><link name="b"/></robot>')
    with pytest.raises(ValueError, match="tree"):
        read_urdf(path)


def test_identity_geometry_reuses_common_metrics(tmp_path):
    folder = make_asset(tmp_path / "asset")
    reference = folder / "objs/0/0.glb"
    metrics = geometry_metrics(folder, reference, count=1000)
    assert metrics["chamfer_l1"] < 1e-8
    assert metrics["fscore@0.02"] == 1
    assert metrics["normal_consistency_unsigned"] == pytest.approx(1)


def test_absolute_scale_track_keeps_prediction_size_error(tmp_path):
    folder = make_asset(tmp_path / "asset")
    text = folder / "basic_info.txt"
    text.write_text("Dimension: 20*20*20\n" + text.read_text())
    reference = tmp_path / "reference.glb"
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(reference)
    result = geometry_metrics(folder, reference, count=1000, reference_unit_m=1)
    assert result["chamfer_l1"] < 1e-8
    assert result["absolute_chamfer_l1_m"] > 0.02
    assert result["absolute_scale_status"] == "ok"


def test_invalid_explicit_inertia_is_rejected(tmp_path):
    folder = make_asset(tmp_path / "asset")
    path = folder / "basic.xml"
    path.write_text(
        path.read_text().replace(
            "<worldbody>",
            '<worldbody><body><inertial mass="1" pos="0 0 0" diaginertia="1 1 4"/></body>',
        )
    )
    assert any("invalid_mjcf_inertial" in error for error in audit_asset(folder, "rigid")["errors"])


def test_raw_property_parser_change_is_audited(tmp_path):
    folder = make_asset(tmp_path / "asset")
    path = folder / "basic_info.txt"
    path.write_text(path.read_text().replace("0.6", "0.8"))
    assert "raw_to_json:0:density" in audit_asset(folder, "rigid")["errors"]


def test_failed_and_absent_cases_keep_category_denominators(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"samples": []}))
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "input.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sid,
                    "category": cat,
                    "image": "input.png",
                    "reference_mesh": "missing.glb",
                }
            )
            for sid, cat in (
                ("rigid", "rigid"),
                ("cloth_a", "deformable"),
                ("cloth_b", "deformable"),
            )
        )
    )
    summary = evaluate_run(run, manifest, tmp_path / "evaluation", sample_count=100)
    assert summary["n_planned"] == 3
    assert summary["categories"]["deformable"]["n_planned"] == 2
    assert summary["categories"]["deformable"]["end_to_end_success"] == 0
    assert len((tmp_path / "evaluation/failure_cases.csv").read_text().splitlines()) == 7


def test_full_static_evaluation_writes_success_and_na_attributes(tmp_path, monkeypatch):
    import builtins

    from PIL import Image

    run = tmp_path / "run"
    asset = make_asset(run / "box/ours_demo/asset")
    Image.new("RGB", (8, 8)).save(tmp_path / "input.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "box",
                "category": "rigid",
                "image": "input.png",
                "reference_mesh": str(asset / "objs/0/0.glb"),
            }
        )
        + "\n"
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "input": {"sample_id": "box"},
                        "status": "ok",
                        "stages": [
                            {"stage": stage, "status": "ok", "seconds": 1}
                            for stage in ("infer", "decode", "export")
                        ],
                    }
                ]
            }
        )
    )
    original = builtins.__import__

    def no_mujoco(name, *args, **kwargs):
        if name.startswith("mujoco"):
            pytest.fail("Static evaluation imported MuJoCo")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mujoco)
    result = evaluate_run(run, manifest, tmp_path / "evaluation", sample_count=1000)
    assert result["categories"]["rigid"]["end_to_end_success"] == 1
    assert result["categories"]["rigid"]["attribute_evaluable_pairs"]["density"] == 0
    assert len(list((tmp_path / "evaluation").glob("*.csv"))) == 8


def test_joint_metric_requires_correspondence_and_declared_frame(tmp_path):
    from asset2sim.evaluation.physx_omni import articulation_metrics

    joint = {
        "name": "hinge",
        "parent": "base",
        "child": "door",
        "type": "revolute",
        "world_origin": [0, 0, 0],
        "world_axis": [0, 0, 1],
        "range": [0, 1],
    }
    metadata = tmp_path / "sample.json"
    metadata.write_text(
        json.dumps(
            {
                "kinematics": {"joints": [joint]},
                "joint_matches": {"hinge": "hinge"},
                "reference_geometry": {"bounds": [[0, 0, 0], [1, 1, 1]]},
            }
        )
    )
    sample = {"sample_metadata": str(metadata)}
    audit = {"structure": {"urdf": {"joints": [joint]}}}
    result = articulation_metrics(sample, audit)[0]
    assert result["joint_type_correct"] and result["axis_status"] == "NA"
    value = json.loads(metadata.read_text())
    value["joint_prediction_to_reference"] = np.eye(4).tolist()
    metadata.write_text(json.dumps(value))
    result = articulation_metrics(sample, audit)[0]
    assert result["axis_error_degrees"] == pytest.approx(0)
    assert result["range_endpoint_mae"] == pytest.approx(0)


def test_part_comparison_keeps_shared_frame_and_unmatched_reference(tmp_path):
    from asset2sim.evaluation.physx_omni.parts import part_metrics

    asset = make_asset(tmp_path / "asset")
    metadata = {
        "part_references": {
            "body": {
                "mesh": str(asset / "objs/0/0.glb"),
                "part_to_reference": [[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            },
            "missing": {"mesh": "not_matched.glb"},
        },
        "part_matches": {"0": "body"},
    }
    geometry = {
        "prediction_transform": np.eye(4).tolist(),
        "reference_transform": np.eye(4).tolist(),
    }
    result = part_metrics(asset, metadata, geometry, tmp_path, sample_count=1000)
    assert result[0]["chamfer_l1"] > 0.04
    assert result[0]["alignment"] == "shared_object_transform"
    assert result[-1]["reference_parts"] == 2
    assert result[-1]["recall"] == 0.5
