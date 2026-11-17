"""The external-stage wrapper preserves measurements when an upstream script fails."""

import json
import os
import subprocess
import sys


def test_failed_worker_retains_cpu_resource_record(tmp_path):
    script = tmp_path / "failure.py"
    script.write_text('raise RuntimeError("fixture failure")\n')
    metrics = tmp_path / "resources.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "asset2sim.experiments.physx_omni_worker",
            "--metrics",
            str(metrics),
            "--",
            str(script),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode != 0
    record = json.loads(metrics.read_text())
    assert record["status"] == "failed"
    assert record["wall_seconds"] > 0
    assert record["cpu_peak_rss_mib"] > 0
    assert record["gpu"] == []


def test_official_cpu_export_does_not_initialize_torch(tmp_path):
    script = tmp_path / "3jsongen_update.py"
    script.write_text('import sys\nassert "torch" not in sys.modules\n')
    metrics = tmp_path / "export_resources.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "asset2sim.experiments.physx_omni_worker",
            "--metrics",
            str(metrics),
            "--",
            str(script),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0
    record = json.loads(metrics.read_text())
    assert record["status"] == "complete"
    assert record["gpu_scope"] == "cpu_export"
