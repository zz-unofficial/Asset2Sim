"""Assemble reference inputs and upstream examples for SF3D development."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from asset2sim.artifacts import write_json
from asset2sim.data.synthetic import camera_pose as camera_pose
from asset2sim.data.synthetic import reference_meshes as reference_meshes
from asset2sim.paths import PROJECT_ROOT

UPSTREAM = PROJECT_ROOT / "third_party/stable-fast-3d"


def prepare_data(output: Path) -> None:
    """Create eight rendered inputs plus references to four upstream examples."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        _write_data(staging, output)
        staging.rename(output)
    print(f"Prepared 4 reference solids, 8 rendered inputs and 4 official examples: {output}")


def _write_data(output: Path, manifest_root: Path) -> None:
    examples = ["chair1.png", "axe.png", "stylized-rocks.png", "tree.png"]
    for name in examples:
        if not (UPSTREAM / "demo_files/examples" / name).is_file():
            raise FileNotFoundError(f"Initialize SF3D submodule; missing official example {name}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference").mkdir()
    (output / "images").mkdir()
    # Select the offscreen EGL backend before importing OpenGL through pyrender.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    renderer = pyrender.OffscreenRenderer(512, 512)
    records = []
    objects = []
    palette = [[52, 124, 189, 255], [215, 135, 67, 255], [93, 168, 123, 255], [174, 104, 169, 255]]
    try:
        for index, (object_id, (category, mesh)) in enumerate(reference_meshes().items()):
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
            mesh.apply_scale(0.20 / mesh.extents.max())
            if not mesh.is_watertight or not mesh.is_volume:
                raise ValueError(f"Reference solid is invalid: {object_id}")
            mesh.visual.face_colors = palette[index]
            reference = Path("reference") / f"{object_id}.glb"
            mesh.export(output / reference)
            objects.append({
                "object_id": object_id,
                "category": category,
                "reference_mesh": reference.as_posix(),
                "source": "Asset2Sim procedural debug geometry",
                "license": "CC0-1.0",
                "units": "meters",
                "longest_dimension_m": 0.20,
                "world_axes": "+Z up, right-handed",
            })
            for view, azimuth in enumerate([45.0, 135.0]):
                pose = camera_pose(azimuth, 20.0)
                scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45] * 3)
                scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
                scene.add(pyrender.PerspectiveCamera(yfov=math.radians(40)), pose=pose)
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=pose)
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.copy()
                color[:, :, 3] = np.where(depth > 0, 255, 0).astype(np.uint8)
                sample_id = f"{object_id}_v{view}"
                image = Path("images") / f"{sample_id}.png"
                Image.fromarray(color, "RGBA").save(output / image)
                records.append({
                    "sample_id": sample_id,
                    "object_id": object_id,
                    "view_id": f"v{view}",
                    "split": "debug",
                    "category": category,
                    "image": image.as_posix(),
                    "reference_mesh": reference.as_posix(),
                    "source": "Asset2Sim procedural render",
                    "license": "CC0-1.0",
                    "camera_to_world": pose.tolist(),
                    "camera_yfov_deg": 40.0,
                    "units": "meters",
                })
    finally:
        renderer.delete()
    for name in examples:
        image = UPSTREAM / "demo_files/examples" / name
        records.append({
            "sample_id": f"official_{image.stem}",
            "object_id": f"official_{image.stem}",
            "view_id": "provided",
            "split": "official_example",
            "image": Path(os.path.relpath(image, manifest_root)).as_posix(),
            "reference_mesh": None,
            "source": "Stability-AI/stable-fast-3d/demo_files/examples",
            "license": "Use under upstream terms; see third_party/stable-fast-3d/LICENSE.md",
        })
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    write_json(output / "objects.json", objects)
    (output / "README.md").write_text(
        "# SF3D 调试数据\n\n"
        "四个程序化参考实体，每个两个独立输入视角；另引用四张官方示例，不复制其文件。\n"
        "程序化网格及渲染图按 CC0-1.0 提供；官方示例遵循上游条款。\n"
        "manifest 路径相对本目录。图像为 512×512 RGBA；几何单位为米、Z 轴向上、最长边 0.20m。\n"
        "本集合用于接口、坐标与评测开发；正式实验使用独立对象清单。\n"
    )

