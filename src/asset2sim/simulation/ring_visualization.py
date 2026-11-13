"""Render paired fixed-probe simulations from saved MuJoCo scenes."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    models = [
        mujoco.MjModel.from_xml_path(str((a.comparison / f"{n}_scene.xml").resolve()))
        for n in ["control", "edited"]
    ]
    data = [mujoco.MjData(m) for m in models]
    renderers = [mujoco.Renderer(m, height=360, width=480) for m in models]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0]
    camera.distance = 0.53
    camera.azimuth = 90
    camera.elevation = -12
    for m in models:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
        m.geom_rgba[g] = [0.9, 0.1, 0.1, 1]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(a.output, fps=20, codec="libx264", macro_block_size=1) as out:
            for step in range(1200):
                for m, d in zip(models, data):
                    d.ctrl[0] = 0.32 * min((step + 1) / 900, 1)
                    mujoco.mj_step(m, d)
                if step % 25 == 0:
                    panels = []
                    for renderer, d, label in zip(renderers, data, ["CONTROL", "EDITED"]):
                        renderer.update_scene(d, camera=camera)
                        im = Image.fromarray(renderer.render())
                        ImageDraw.Draw(im).text((10, 10), label, fill="white")
                        panels.append(np.asarray(im))
                    out.append_data(np.concatenate(panels, axis=1))
    finally:
        for renderer in renderers:
            renderer.close()


if __name__ == "__main__":
    main()
