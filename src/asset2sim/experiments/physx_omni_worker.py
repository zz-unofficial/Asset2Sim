"""Measure one official stage in its process; no model code is modified."""

from __future__ import annotations

import argparse
import json
import os
import resource
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--decoder-cpu-offload", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise ValueError("Missing official script")
    if args.decoder_cpu_offload and Path(argv[0]).name != "decoder_each.py":
        raise ValueError("Decoder CPU offload only applies to decoder_each.py")
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

            metrics["cuda_runtime"] = torch.version.cuda
            if torch.cuda.is_available():
                torch.cuda.init()
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
        except ImportError:
            pass
    started = time.perf_counter()
    try:
        if args.decoder_cpu_offload:
            from trellis.pipelines import TrellisImageTo3DPipeline

            from asset2sim.experiments.physx_decoder_offload import install_decoder_offload

            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("Decoder CPU offload requires CUDA")
            install_decoder_offload(TrellisImageTo3DPipeline, torch)
            metrics["decoder_cpu_offload"] = True
        dinov2_source = os.environ.get("ASSET2SIM_DINOV2_SOURCE")
        if dinov2_source and Path(argv[0]).name == "decoder_each.py":
            if torch is None or not Path(dinov2_source, "hubconf.py").is_file():
                raise ValueError("ASSET2SIM_DINOV2_SOURCE must be a local DINOv2 checkout")
            original_hub_load = torch.hub.load

            def load_local_dinov2(repo_or_dir, *args, **kwargs):
                if repo_or_dir == "facebookresearch/dinov2":
                    repo_or_dir = dinov2_source
                    kwargs["source"] = "local"
                return original_hub_load(repo_or_dir, *args, **kwargs)

            torch.hub.load = load_local_dinov2
            metrics["dinov2_source"] = dinov2_source
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
