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

    def render(self, mesh, camera_to_world: np.ndarray, yfov_degrees: float = 40) -> Image.Image:
        p = self.pyrender
        scene = p.Scene(bg_color=[1, 1, 1, 0], ambient_light=[0.45] * 3)
        scene.add(p.Mesh.from_trimesh(mesh, smooth=False))
        scene.add(p.PerspectiveCamera(yfov=np.deg2rad(yfov_degrees)), pose=camera_to_world)
        scene.add(p.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera_to_world)
        color, depth = self.renderer.render(scene, flags=p.RenderFlags.RGBA)
        color = color.copy()
        color[:, :, 3] = np.where(depth > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(color)

    def close(self) -> None:
        self.renderer.delete()


def composite_rgb(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    return rgba[:, :, :3] * rgba[:, :, 3:] + (1 - rgba[:, :, 3:])

