"""Measure one official stage in its process; no model code is modified."""

from __future__ import annotations

import argparse
import json
import resource
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise ValueError("Missing official script")
    metrics = {
        "gpu": [],
        "status": "failed",
        "cpu_rss_unit": "MiB",
        "decode_scope": "official_decoder_including_texture_and_export",
    }
    torch = None
    metrics["gpu_scope"] = (
        "cpu_export" if Path(argv[0]).name == "3jsongen_update.py" else "torch_allocator"
    )
    if metrics["gpu_scope"] != "cpu_export":
        try:
            import torch

            if torch.cuda.is_available():
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
        except ImportError:
            pass
    started = time.perf_counter()
    try:
        sys.argv = argv
        runpy.run_path(argv[0], run_name="__main__")
        metrics["status"] = "complete"
    finally:
        metrics["wall_seconds"] = time.perf_counter() - started
        metrics["cpu_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if torch is not None and torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                metrics["gpu"].append(
                    {
                        "device": device,
                        "name": torch.cuda.get_device_name(device),
                        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                    }
                )
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
