"""Public GSO/OmniObject3D subsets with sixteen rendered evaluation views."""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from asset2sim.artifacts import write_json
from asset2sim.data.manifest import export_manifest
from asset2sim.data.rendering import MeshRenderer
from asset2sim.data.synthetic import camera_pose
from asset2sim.evaluation.geometry import load_mesh, normalized_mesh
from asset2sim.experiments.config import load_parameters
from asset2sim.paths import PROJECT_ROOT


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.rename(path)


def extract_archive(archive: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".extract-") as folder:
        staging = Path(folder)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for item in source.infolist():
                    if not (staging / item.filename).resolve().is_relative_to(staging):
                        raise ValueError("Archive member is outside the data directory")
                source.extractall(staging)
        else:
            with tarfile.open(archive) as source:
                for item in source:
                    destination = (staging / item.name).resolve()
                    if not destination.is_relative_to(staging):
                        raise ValueError("Archive member is outside the data directory")
                    if item.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif item.isfile():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(item) as stream, destination.open("wb") as output:
                            shutil.copyfileobj(stream, output)
                    else:
                        raise ValueError(f"Unsupported archive link: {item.name}")
        staging.rename(target)


def gso_catalog() -> list[dict]:
    cache = PROJECT_ROOT / ".local/datasets/gso_catalog.json"
    if cache.exists():
        return json.loads(cache.read_text())
    items = {}
    for page in range(1, 21):
        url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=100&page={page}"
        with urllib.request.urlopen(url, timeout=60) as response:
            rows = json.load(response)
        previous = len(items)
        items.update({row["name"]: row for row in rows})
        if len(rows) < 100 or len(items) == previous:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(items.values(), key=lambda row: row["name"])
    write_json(cache, ordered)
    return ordered


def select_sources(config: dict, selection: Path) -> list[dict]:
    if selection.exists():
        return json.loads(selection.read_text())["objects"]
    count = config["sample_count"]
    rng = random.Random(config["seed"])
    if config["source"] == "gso":
        population = gso_catalog()
        rows = rng.sample(population, count)
        objects = [{
            "object_id": f"gso_{row['name']}",
            "name": row["name"],
            "url": f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{urllib.parse.quote(row['name'], safe='')}/tip/{urllib.parse.quote(row['name'], safe='')}.zip",
            "license": row["license_name"],
            "license_url": row["license_url"],
            "source": "GoogleResearch / Google Scanned Objects",
        } for row in rows]
    elif config["source"] == "omniobject3d":
        population = []
        archives = [PROJECT_ROOT / "data/raw/omniobject3d" / f"{item['name']}.tar.gz"
                    for item in config["archives"]]
        missing = [str(path) for path in archives if not path.is_file()]
        if missing:
            raise FileNotFoundError("Place the downloaded OmniObject3D archives at: " + ", ".join(missing))
        for source in config["archives"]:
            archive = PROJECT_ROOT / "data/raw/omniobject3d" / f"{source['name']}.tar.gz"
            folder = archive.parent / source["name"]
            extract_archive(archive, folder)
            for mesh in sorted(folder.rglob("Scan.obj")):
                population.append({
                    "object_id": f"omni_{source['name']}_{mesh.parent.parent.name}",
                    "name": mesh.parent.parent.name,
                    "mesh": str(mesh.relative_to(PROJECT_ROOT)),
                    "url": "https://drive.google.com/drive/folders/1ZpGmUlXYf0uc9xtN_w9iOhxmt9_gHZaP",
                    "license": "CC BY 4.0",
                    "source": "OmniObject3D author release",
                })
        if count > len(population):
            raise ValueError(f"Requested {count} objects but downloaded categories contain {len(population)}")
        objects = rng.sample(population, count)
    else:
        raise ValueError(f"Unknown dataset source: {config['source']}")
    selection.parent.mkdir(parents=True, exist_ok=True)
    write_json(selection, {
        "dataset": config["name"], "seed": config["seed"],
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "population_size": len(population), "objects": objects,
    })
    return objects


def source_mesh(source: dict, dataset: str) -> Path:
    if dataset == "omniobject3d":
        return PROJECT_ROOT / source["mesh"]
    name = source["name"]
    folder = PROJECT_ROOT / "data/raw/gso" / name
    archive = folder.parent / f"{name}.zip"
    download(source["url"], archive)
    extract_archive(archive, folder)
    return folder / "meshes/model.obj"


def load_source_mesh(path: Path, dataset: str):
    mesh = load_mesh(path)
    # GSO's MTL names texture.png, which lives outside the meshes directory.
    if dataset == "gso" and getattr(mesh.visual, "kind", None) == "texture":
        material = mesh.visual.material
        if getattr(material, "image", None) is None:
            texture = path.parent.parent / "materials/textures/texture.png"
            with Image.open(texture) as image:
                material.image = image.copy()
    return mesh


def prepare_paper_data(config_path: Path, output: Path, manifest: Path) -> int:
    config = load_parameters(config_path)
    output = output.resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Use new data and manifest paths: {output}, {manifest}")
    selection = (config_path.resolve().parent / config["selection_file"]).resolve()
    sources = select_sources(config, selection)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    renderer = MeshRenderer(config["image_size"])
    try:
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".paper-data-") as temporary:
            staging = Path(temporary) / "data"
            staging.mkdir()
            for source in sources:
                object_id = source["object_id"]
                directory = staging / object_id
                (directory / "views").mkdir(parents=True)
                mesh, transform = normalized_mesh(
                    load_source_mesh(source_mesh(source, config["source"]), config["source"]), config["normalization_extent"]
                )
                mesh.export(directory / "reference.glb")
                views = []
                foreground_areas = []
                for index in range(config["render_views"]):
                    pose = camera_pose(index * 360 / config["render_views"], config["elevation_degrees"], config["camera_distance"])
                    image = renderer.render(mesh, pose, config["yfov_degrees"])
                    filename = f"views/{index:03d}.png"
                    image.save(directory / filename)
                    foreground_areas.append(int(np.count_nonzero(np.asarray(image)[:, :, 3])))
                    views.append({"image": filename, "camera_to_world": pose.tolist()})
                conditioning = int(np.argmax(foreground_areas)) if config.get("conditioning_view") == "largest_silhouette" else 0
                write_json(directory / "render.json", {
                    "image_size": config["image_size"], "yfov_degrees": config["yfov_degrees"],
                    "source_to_reference": transform.tolist(), "views": views,
                    "conditioning_view": conditioning,
                    "conditioning_rule": config.get("conditioning_view", "first"),
                    "foreground_pixels": foreground_areas,
                })
                final = output / object_id
                records.append({
                    "sample_id": object_id, "object_id": object_id, "view_id": f"{conditioning:03d}",
                    "split": "paper_public", "dataset": config["source"],
                    "image": str(final / views[conditioning]["image"]),
                    "reference_mesh": str(final / "reference.glb"),
                    "render_metadata": str(final / "render.json"),
                    "source": source["source"], "source_url": source["url"],
                    "license": source["license"],
                })
                print(f"Prepared {object_id}", flush=True)
            write_json(staging / "dataset.json", config | {"selection": str(selection)})
            staging.rename(output)
    finally:
        renderer.close()
    export_manifest(records, manifest)
    return len(records)

