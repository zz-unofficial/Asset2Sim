"""SPAR3D single-image adapter with the shared mesh and point-cloud outputs."""

from __future__ import annotations

import gc
import importlib
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from asset2sim.artifacts import git_revision
from asset2sim.generation.sf3d import SF3D_RAW_TO_MODEL_Z_UP, SF3DGenerator
from asset2sim.paths import PROJECT_ROOT

UPSTREAM = PROJECT_ROOT / "third_party/stable-point-aware-3d"
os.environ.setdefault("ALPHA_CLIP_PATH", str(PROJECT_ROOT / ".local/alpha_clip"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class SPAR3DConfig:
    model: str = "stabilityai/stable-point-aware-3d"
    revision: str = "main"
    device: str = "cuda"
    precision: str = "bfloat16"
    texture_resolution: int = 1024
    foreground_ratio: float = 1.3
    remesh: str = "none"
    target_vertex_count: int = -1
    low_vram: bool = True
    seed: int = 0
    cpu_threads: int = 8

    def __post_init__(self):
        if self.device not in {"cuda", "cpu"} or self.precision not in {"bfloat16", "float32"}:
            raise ValueError("Invalid device or precision")
        if self.remesh not in {"none", "triangle", "quad"}:
            raise ValueError("Invalid remesh mode")
        if self.texture_resolution < 1 or self.foreground_ratio <= 0 or self.cpu_threads < 1:
            raise ValueError("Invalid texture resolution, crop ratio or thread count")


def fetch_assets(parameters: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError

    config = SPAR3DConfig(**parameters)
    local = Path(config.model).expanduser()
    if local.is_dir():
        for name in ("config.yaml", "model.safetensors"):
            if not (local / name).is_file():
                raise FileNotFoundError(local / name)
        return local.resolve()
    try:
        return Path(
            snapshot_download(
                config.model,
                revision=config.revision,
                allow_patterns=["config.yaml", "model.safetensors"],
            )
        )
    except GatedRepoError as error:
        raise RuntimeError(
            "Open access at https://huggingface.co/stabilityai/stable-point-aware-3d and log in"
        ) from error


class SPAR3DGenerator(SF3DGenerator):
    model_name = "spar3d"

    def __init__(self, config: SPAR3DConfig):
        super().__init__(config)
        self.remover = None

    @property
    def parameters(self) -> dict[str, Any]:
        return asdict(self.config)

    def load(self) -> None:
        import torch

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.config.device == "cpu":
            os.environ["SPAR3D_USE_CPU"] = "1"
        else:
            os.environ.pop("SPAR3D_USE_CPU", None)
        sys.path.insert(0, str(UPSTREAM))
        from spar3d.system import SPAR3D

        # Low-VRAM mode recreates the estimator for each input; reuse the first
        # successful upstream cache lookup rather than hashing the CLIP file each time.
        alpha_clip = importlib.import_module("alpha_clip.alpha_clip")
        if not hasattr(alpha_clip._download, "cache_info"):
            alpha_clip._download = lru_cache(maxsize=4)(alpha_clip._download)

        torch.set_num_threads(self.config.cpu_threads)
        self.weights_path = fetch_assets(self.parameters)
        self.model = SPAR3D.from_pretrained(
            str(self.weights_path),
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=self.config.low_vram,
        )
        self.model.to(self.config.device).eval()

    def metadata(self) -> dict[str, Any]:
        import torch

        return {
            "upstream_commit": git_revision(UPSTREAM),
            "weights_revision": self.weights_path.name,
            "model_structure": {
                "input_image_size": self.model.cfg.cond_image_size,
                "tetrahedral_resolution": self.model.cfg.isosurface_resolution,
                "diffusion_steps": self.model.cfg.inference_time_steps,
                "diffusion_schedule": self.model.cfg.diffu_sched,
                "guidance_scale": self.model.cfg.guidance_scale,
                "s_churn": self.model.cfg.s_churn,
            },
            "raw_mesh_coordinates": {
                "up_axis": "+Y",
                "units": "model-normalized; not meters",
                "raw_to_model_z_up": SF3D_RAW_TO_MODEL_Z_UP.tolist(),
            },
            "point_cloud_coordinates": {"up_axis": "+Z", "units": "model-normalized"},
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if self.config.device == "cuda" else "cpu",
        }

    def prepare_image(self, path: Path) -> Image.Image:
        from spar3d.utils import foreground_crop, remove_background

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        alpha = np.asarray(image.getchannel("A"))
        if not np.any(alpha > 127):
            raise ValueError("Input has an empty foreground mask")
        if alpha.min() == 255:
            from transparent_background import Remover

            if self.remover is None:
                self.remover = Remover(device=self.config.device)
            image = remove_background(image, self.remover)
            self.background_mode = "transparent_background"
        else:
            self.background_mode = "provided_alpha"
        return foreground_crop(image, self.config.foreground_ratio)

    def recover(self) -> None:
        """Release all staged modules after inference or an interrupted stage."""
        import torch

        if self.model is not None and getattr(self.model, "is_low_vram", False):
            self.model._unload_pdiff_modules()
            self.model._unload_main_modules()
            self.model._unload_estimator_modules()
            # Upstream retains the decoder for texture baking between stages.
            self.model.decoder = None
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

    def generate(self, image_path: Path, output_dir: Path) -> dict[str, Any]:
        import torch

        if self.model is None:
            raise RuntimeError("Call load() before generate()")
        # A previous exception can retain tensors until the caller exits its handler.
        self.recover()
        output_dir.mkdir(parents=True, exist_ok=False)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        start = time.perf_counter()
        image = self.prepare_image(image_path)
        image.save(output_dir / "input.png")
        preprocess = time.perf_counter() - start
        cuda = self.config.device == "cuda"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cuda and self.config.precision == "bfloat16"
            else nullcontext()
        )
        with torch.no_grad(), precision:
            mesh, auxiliary = self.model.run_image(
                image,
                bake_resolution=self.config.texture_resolution,
                remesh=self.config.remesh,
                vertex_count=self.config.target_vertex_count,
                return_points=True,
            )
        if cuda:
            torch.cuda.synchronize()
        inference = time.perf_counter() - start
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("SPAR3D generated an invalid mesh")
        start = time.perf_counter()
        mesh.export(output_dir / "mesh.glb", include_normals=True)
        export_seconds = time.perf_counter() - start
        start = time.perf_counter()
        auxiliary["point_clouds"][0].export(output_dir / "points.ply")
        result = {
            "background_mode": self.background_mode,
            "preprocess_seconds": preprocess,
            "inference_seconds": inference,
            "export_seconds": export_seconds,
            "point_cloud_export_seconds": time.perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if cuda else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if cuda else None,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "point_cloud": f"{output_dir.name}/points.ply",
            "points": len(auxiliary["point_clouds"][0].vertices),
        }
        del auxiliary
        self.recover()
        return result


def create_generator(parameters: dict[str, Any]) -> SPAR3DGenerator:
    try:
        return SPAR3DGenerator(SPAR3DConfig(**parameters))
    except TypeError as error:
        raise ValueError(f"Invalid SPAR3D parameters: {error}") from error
