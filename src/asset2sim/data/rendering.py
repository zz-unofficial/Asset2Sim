"""Common offscreen PBR renderer for dataset inputs and image metrics."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


class MeshRenderer:
    def __init__(self, image_size: int = 512):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender

        self.pyrender = pyrender
        self.renderer = pyrender.OffscreenRenderer(image_size, image_size)

    def render(
        self, mesh, camera_to_world: np.ndarray, yfov_degrees: float = 40, mode: str = "pbr"
    ) -> Image.Image:
        p = self.pyrender
        if mode not in {"pbr", "texture"}:
            raise ValueError(f"Unknown render mode: {mode}")
        scene = p.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.45] * 3)
        material = None
        if mode == "texture" and getattr(mesh.visual, "kind", None) == "texture":
            source = mesh.visual.material
            texture = getattr(source, "baseColorTexture", getattr(source, "image", None))
            factor = getattr(source, "baseColorFactor", getattr(source, "diffuse", None))
            # This pass measures stored texture RGB, independent of source Kd, metal and lights.
            if texture is not None:
                factor = [1.0, 1.0, 1.0, 1.0]
            elif factor is not None:
                factor = np.asarray(factor, dtype=float)
                if factor.max() > 1:
                    factor = factor / 255
            material = p.MetallicRoughnessMaterial(
                baseColorTexture=texture,
                baseColorFactor=factor,
                metallicFactor=0,
                roughnessFactor=1,
            )
        scene.add(p.Mesh.from_trimesh(mesh, smooth=False, material=material))
        scene.add(p.PerspectiveCamera(yfov=np.deg2rad(yfov_degrees)), pose=camera_to_world)
        scene.add(p.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera_to_world)
        flags = p.RenderFlags.RGBA | (p.RenderFlags.FLAT if mode == "texture" else 0)
        color, depth = self.renderer.render(scene, flags=flags)
        color = color.copy()
        color[:, :, 3] = np.where(depth > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(color)

    def close(self) -> None:
        self.renderer.delete()


def composite_rgb(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    return rgba[:, :, :3] * rgba[:, :, 3:] + (1 - rgba[:, :, 3:])
