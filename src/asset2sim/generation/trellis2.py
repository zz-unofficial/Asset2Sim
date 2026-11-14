"""TRELLIS.2 official 512 pipeline with the shared generation record format."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from asset2sim.artifacts import git_revision
from asset2sim.paths import PROJECT_ROOT

UPSTREAM = PROJECT_ROOT / "third_party/TRELLIS.2"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class TRELLIS2Config:
    model: str = "microsoft/TRELLIS.2-4B"
    pipeline_type: str = "512"
    seed: int = 0
    decimation_target: int = 1_000_000
    texture_size: int = 4096
    remesh: bool = True
    extension_webp: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_type != "512":
            raise ValueError("Benchmark adapter currently supports official 512 pipeline only")
        if self.decimation_target < 1 or self.texture_size < 1:
            raise ValueError("Export resolution and face budget must be positive")


class TRELLIS2Generator:
    model_name = "trellis2"

    def __init__(self, config: TRELLIS2Config):
        self.config = config
        self.pipeline = None

    @property
    def parameters(self) -> dict[str, Any]:
        return asdict(self.config)

    def load(self) -> None:
        import torch

        if not (UPSTREAM / "trellis2/pipelines/trellis2_image_to_3d.py").is_file():
            raise RuntimeError("Initialize third_party/TRELLIS.2 before inference")
        if not torch.cuda.is_available():
            raise RuntimeError("TRELLIS.2 official pipeline requires a CUDA GPU")
        sys.path.insert(0, str(UPSTREAM))
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.config.model)
        self.pipeline.cuda()

    def metadata(self) -> dict[str, Any]:
        import torch
        from huggingface_hub import hf_hub_download

        local = Path(self.config.model)
        weights_revision = (
            local.resolve().name
            if local.is_dir()
            else Path(
                hf_hub_download(self.config.model, "pipeline.json", local_files_only=True)
            ).parent.name
        )

        return {
            "upstream_commit": git_revision(UPSTREAM),
            "weights_repository": self.config.model,
            "weights_revision": weights_revision,
            "raw_mesh_coordinates": {
                "up_axis": "+Z (official render_utils and numeric GLB export)",
                "units": "model-normalized",
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "export": {
                "remesh": self.config.remesh,
                "decimation_target": self.config.decimation_target,
                "texture_size": self.config.texture_size,
                "extension_webp": self.config.extension_webp,
            },
        }

    def recover(self) -> None:
        import torch

        torch.cuda.empty_cache()

    def generate(self, image_path: Path, output_dir: Path) -> dict[str, Any]:
        import o_voxel
        import torch

        if self.pipeline is None:
            raise RuntimeError("Call load() before generate()")
        output_dir.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        image.save(output_dir / "input.png")
        preprocess_seconds = time.perf_counter() - start
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            mesh = self.pipeline.run(
                image, seed=self.config.seed, pipeline_type=self.config.pipeline_type
            )[0]
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        vertices = len(mesh.vertices)
        faces = len(mesh.faces)
        if vertices == 0 or faces == 0:
            raise ValueError("TRELLIS.2 generated an empty mesh")
        # Save the native mesh before official GLB postprocessing changes its topology.
        import numpy as np

        export_start = time.perf_counter()
        native_vertices = mesh.vertices.detach().cpu().numpy()
        native_faces = mesh.faces.detach().cpu().numpy()
        if not np.isfinite(native_vertices).all():
            raise ValueError("TRELLIS.2 generated non-finite vertices")
        np.savez_compressed(
            output_dir / "native_geometry.npz", vertices=native_vertices, faces=native_faces
        )
        native_archive_seconds = time.perf_counter() - export_start
        export_start = time.perf_counter()
        exported = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=self.config.decimation_target,
            texture_size=self.config.texture_size,
            remesh=self.config.remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        exported.export(output_dir / "mesh.glb", extension_webp=self.config.extension_webp)
        torch.cuda.synchronize()
        return {
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "export_seconds": time.perf_counter() - export_start,
            "native_archive_seconds": native_archive_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "vertices": vertices,
            "faces": faces,
            "native_geometry": "native_geometry.npz",
        }


def create_generator(parameters: dict[str, Any]) -> TRELLIS2Generator:
    return TRELLIS2Generator(TRELLIS2Config(**parameters))


def fetch_assets(parameters: dict[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    config = TRELLIS2Config(**parameters)
    local = Path(config.model).expanduser()
    return local.resolve() if local.is_dir() else Path(snapshot_download(config.model))
