"""Blender worker for shared unlit/PBR renders; invoked with --python."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def render_job(job: dict) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = job.get("samples", 32)
    scene.cycles.seed = 0
    scene.render.resolution_x = scene.render.resolution_y = job["image_size"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast" if job.get("contrast") else "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    bpy.ops.import_scene.gltf(filepath=job["mesh"])
    roots = [obj for obj in scene.objects if obj.parent is None]
    # glTF import converts numeric Y-up coordinates into Blender Z-up. Undo it:
    # benchmark meshes and camera matrices already share a numeric Z-up frame.
    undo = Matrix.Rotation(-math.pi / 2, 4, "X")
    for obj in roots:
        obj.matrix_world = undo @ obj.matrix_world
    if job["mode"] == "texture":
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue
            nodes, links = material.node_tree.nodes, material.node_tree.links
            shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
            output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
            if shader is None or output is None:
                continue
            color = shader.inputs["Base Color"]
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = color.default_value
            if color.is_linked:
                links.new(color.links[0].from_socket, emission.inputs["Color"])
            links.new(emission.outputs[0], output.inputs["Surface"])
    else:
        world = bpy.data.worlds.new("benchmark_hdri")
        world.use_nodes = True
        scene.world = world
        nodes = world.node_tree.nodes
        environment = nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(job["hdri"])
        world.node_tree.links.new(environment.outputs["Color"], nodes["Background"].inputs["Color"])
        nodes["Background"].inputs["Strength"].default_value = 1
    camera_data = bpy.data.cameras.new("benchmark_camera")
    camera = bpy.data.objects.new("benchmark_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "PERSP"
    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(job["yfov_degrees"])
    camera_data.clip_start = 0.001
    camera_data.clip_end = 100
    directory = Path(job["output"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "renderer.json").write_text(
        json.dumps(
            {
                "blender": bpy.app.version_string,
                "engine": scene.render.engine,
                "device": scene.cycles.device,
                "samples": scene.cycles.samples,
                "view_transform": scene.view_settings.view_transform,
                "exposure": scene.view_settings.exposure,
                "hdri": job.get("hdri"),
            },
            indent=2,
        )
    )
    for index, view in enumerate(job["views"]):
        camera.matrix_world = Matrix(view["camera_to_world"])
        scene.render.filepath = str(directory / f"{index:03d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for job in json.loads(args.jobs.read_text()):
        render_job(job)
