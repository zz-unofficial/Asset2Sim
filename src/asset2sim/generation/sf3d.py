"""SF3D adapter: preserve upstream inference and use stable sample IDs."""

from __future__ import annotations

import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from asset2sim.artifacts import git_revision
from asset2sim.data.manifest import load_manifest
from asset2sim.experiments.config import load_parameters
from asset2sim.paths import PROJECT_ROOT as ROOT

__all__ = [
    "SF3D_RAW_TO_MODEL_Z_UP",
    "SF3DConfig",
    "SF3DGenerator",
    "create_generator",
    "fetch_assets",
    "fetch_weights",
    "load_manifest",
    "run_manifest",
]

UPSTREAM = ROOT / "third_party/stable-fast-3d"
# Keep large project caches writable without changing the user's login/token location.
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".local/huggingface/hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("U2NET_HOME", str(ROOT / ".local/u2net"))
# Inverse of the official export rotation: Ry(+90deg) @ Rx(-90deg).
SF3D_RAW_TO_MODEL_Z_UP = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
)


@dataclass
class SF3DConfig:
    model: str = "stabilityai/stable-fast-3d"
    revision: str = "main"
    device: str = "cuda"
    precision: str = "bfloat16"
    texture_resolution: int = 1024
    foreground_ratio: float = 0.85
    remesh: str = "none"
    target_vertex_count: int = -1
    seed: int = 0
    cpu_threads: int = 8

    def __post_init__(self) -> None:
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu")
        if self.precision not in {"bfloat16", "float32"}:
            raise ValueError("precision must be bfloat16 or float32")
        if self.remesh not in {"none", "triangle", "quad"}:
            raise ValueError("remesh must be none, triangle or quad")
        if self.texture_resolution < 1 or not 0 < self.foreground_ratio <= 1:
            raise ValueError("Invalid texture resolution or foreground ratio")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")

    @classmethod
    def load(cls, path: Path) -> SF3DConfig:
        return cls(**load_parameters(path))


def fetch_weights(config: SF3DConfig) -> Path:
    """Use the official repository and the user's normal Hugging Face cache/login."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError

    model_path = Path(config.model).expanduser()
    if model_path.is_dir():
        for name in ("config.yaml", "model.safetensors"):
            if not (model_path / name).is_file():
                raise FileNotFoundError(model_path / name)
        return model_path.resolve()
    try:
        result = snapshot_download(
            config.model,
            revision=config.revision,
            allow_patterns=["config.yaml", "model.safetensors"],
        )
    except GatedRepoError as error:
        raise RuntimeError(
            "SF3D weights require access approval. Accept the conditions at "
            "https://huggingface.co/stabilityai/stable-fast-3d and use "
            "huggingface-cli login in this Conda environment."
        ) from error
    return Path(result)


class SF3DGenerator:
    """One model instance reused across independently generated single images."""

    model_name = "sf3d"

    def __init__(self, config: SF3DConfig):
        self.config = config
        self.model = None
        self.rembg_session = None
        self.weights_path: Path | None = None
        self.background_mode: str | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return asdict(self.config)

    def metadata(self) -> dict[str, Any]:
        import torch

        return {
            "upstream_commit": git_revision(UPSTREAM),
            "model_structure": {
                "parameters_total": sum(p.numel() for p in self.model.parameters()),
                "input_image_size": self.model.cfg.cond_image_size,
                "tetrahedral_resolution": self.model.cfg.isosurface_resolution,
                "triplane_input_resolution": self.model.cfg.tokenizer["plane_size"],
                "triplane_output_resolution": self.model.cfg.tokenizer["plane_size"]
                * self.model.cfg.post_processor["scale_factor"],
                "triplane_channels": self.model.cfg.post_processor["out_channels"],
                "transformer_units": self.model.cfg.backbone["num_blocks"],
            },
            "weights_revision": self.weights_path.name if self.weights_path else None,
            "raw_mesh_coordinates": {
                "up_axis": "+Y",
                "units": "model-normalized; not meters",
                "raw_to_model_z_up": SF3D_RAW_TO_MODEL_Z_UP.tolist(),
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)
            if self.config.device == "cuda" and torch.cuda.is_available()
            else self.config.device,
            "dependencies": {
                name: version(name)
                for name in (
                    "numpy",
                    "trimesh",
                    "transformers",
                    "huggingface-hub",
                    "open_clip_torch",
                )
            },
        }

    def recover(self) -> None:
        if self.config.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    def load(self) -> None:
        import torch

        if not (UPSTREAM / "sf3d/system.py").is_file():
            raise RuntimeError("Initialize third_party/stable-fast-3d before inference")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no automatic CPU fallback")
        if self.config.device == "cpu":
            # Upstream geometry code consults this variable separately from model.device.
            os.environ["SF3D_USE_CPU"] = "1"
        else:
            os.environ.pop("SF3D_USE_CPU", None)
        sys.path.insert(0, str(UPSTREAM))
        from sf3d.system import SF3D

        torch.set_num_threads(self.config.cpu_threads)
        self.weights_path = fetch_weights(self.config)
        self.model = (
            SF3D.from_pretrained(
                str(self.weights_path), config_name="config.yaml", weight_name="model.safetensors"
            )
            .to(self.config.device)
            .eval()
        )

    def prepare_image(self, path: Path) -> Image.Image:
        from sf3d.utils import remove_background, resize_foreground

        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGBA")
        alpha = np.asarray(image.getchannel("A"))
        if not np.any(alpha > 127):
            raise ValueError("Input has an empty foreground mask")
        if alpha.min() == 255:
            self.background_mode = "rembg_cpu"
            import rembg

            if self.rembg_session is None:
                self.rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])
            image = remove_background(image, self.rembg_session)
        else:
            self.background_mode = "provided_alpha"
        return resize_foreground(image, self.config.foreground_ratio)

    def generate(self, image_path: Path, output_dir: Path) -> dict[str, Any]:
        import torch

        if self.model is None:
            raise RuntimeError("Call load() before generate()")
        output_dir.mkdir(parents=True, exist_ok=False)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        start = time.perf_counter()
        image = self.prepare_image(image_path)
        image.save(output_dir / "input.png")
        preprocess_seconds = time.perf_counter() - start
        use_cuda = self.config.device == "cuda"
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        inference_start = time.perf_counter()
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda and self.config.precision == "bfloat16"
            else nullcontext()
        )
        with torch.no_grad(), precision:
            mesh, _ = self.model.run_image(
                image,
                bake_resolution=self.config.texture_resolution,
                remesh=self.config.remesh,
                vertex_count=self.config.target_vertex_count,
            )
        if use_cuda:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("SF3D generated an empty mesh")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("SF3D generated non-finite vertices")
        export_start = time.perf_counter()
        mesh.export(output_dir / "mesh.glb", include_normals=True)
        return {
            "background_mode": self.background_mode,
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "export_seconds": time.perf_counter() - export_start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
        }


def run_manifest(
    config: SF3DConfig, manifest: Path, output: Path, sample_ids: list[str] | None = None
) -> int:
    """Compatibility entry point for existing SF3D callers."""
    from asset2sim.experiments.runner import run_generation

    return run_generation(SF3DGenerator(config), manifest, output, sample_ids)


def create_generator(parameters: dict[str, Any]) -> SF3DGenerator:
    try:
        return SF3DGenerator(SF3DConfig(**parameters))
    except TypeError as error:
        raise ValueError(f"Invalid SF3D parameters: {error}") from error


def fetch_assets(parameters: dict[str, Any]) -> Path:
    """Cache all model assets through the same entry point used by other adapters."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    config = create_generator(parameters).config
    weights = fetch_weights(config)
    model_config = yaml.safe_load((weights / "config.yaml").read_text())
    dino = model_config.get("image_tokenizer", {}).get(
        "pretrained_model_name_or_path", "facebook/dinov2-large"
    )
    hf_hub_download(dino, "config.json")
    try:
        hf_hub_download(dino, "model.safetensors")
    except EntryNotFoundError:
        hf_hub_download(dino, "pytorch_model.bin")
    import open_clip
    from open_clip.pretrained import download_pretrained

    estimator = model_config.get("image_estimator", {})
    clip_config = open_clip.get_pretrained_cfg(
        estimator.get("model", "ViT-B-32"), estimator.get("pretrain", "laion2b_s34b_b79k")
    )
    download_pretrained(clip_config)
    return weights
