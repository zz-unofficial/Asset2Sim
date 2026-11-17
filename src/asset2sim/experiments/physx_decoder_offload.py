"""Optional memory management for the official single-image decoder."""

from __future__ import annotations

import functools
import time
from typing import Any


def install_decoder_offload(pipeline_type: type, torch_module: Any) -> None:
    """Offload inference networks after latent sampling; keep texture baking untouched."""
    original_run = pipeline_type.run_decoder
    original_sample = pipeline_type.sample_slat

    @functools.wraps(original_sample)
    def sample_and_offload(self: Any, *args: Any, **kwargs: Any) -> Any:
        names = ("image_cond_model", "slat_flow_model")
        missing = [name for name in names if name not in self.models]
        if missing:
            raise RuntimeError(f"Cannot offload missing pipeline models: {missing}")
        latent = original_sample(self, *args, **kwargs)
        started = time.perf_counter()
        for name in names:
            self.models[name].to("cpu")
        torch_module.cuda.empty_cache()
        print(
            f"Decoder CPU offload: {', '.join(names)}; {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        return latent

    @functools.wraps(original_run)
    def run_without_grad(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The caller performs differentiable texture baking after this method returns.
        with torch_module.no_grad():
            return original_run(self, *args, **kwargs)

    pipeline_type.sample_slat = sample_and_offload
    pipeline_type.run_decoder = run_without_grad
