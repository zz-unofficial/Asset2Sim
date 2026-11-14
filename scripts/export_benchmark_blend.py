"""Run inside Blender: blender -b asset.blend --python this_file -- --output reference.glb."""

import argparse
import sys
from pathlib import Path

import bpy

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
output = args.output.resolve()
if output.exists():
    raise FileExistsError(output)
for item in bpy.data.objects:
    item.select_set(item.type == "MESH" and not item.hide_render)
if not any(item.select_get() for item in bpy.data.objects):
    raise ValueError("Blend scene has no visible mesh objects")
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(output), export_format="GLB", use_selection=True, export_yup=False
)
print(f"Exported {output}")
