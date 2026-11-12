"""Place actual measurements and the published SF3D reference rows side by side."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from asset2sim.artifacts import read_jsonl, write_json


def qualitative_sheet(directory: Path, dataset: str, output: Path) -> Path | None:
    """Show four fixed CD ranks, each with its conditioning and opposite view."""
    from PIL import Image, ImageDraw, ImageFont

    rows = sorted(
        (
            row
            for row in read_jsonl(directory / "metrics.jsonl")
            if row["status"] == "ok" and row["dataset"] == dataset and row.get("evaluated_views")
        ),
        key=lambda row: row["chamfer_l1"],
    )
    if not rows:
        return None
    metadata = json.loads((directory / "evaluation.json").read_text())
    inputs = {row["sample_id"]: row for row in read_jsonl(Path(metadata["run"]) / "inputs.jsonl")}
    ranks = sorted({0, (len(rows) - 1) // 3, 2 * (len(rows) - 1) // 3, len(rows) - 1})
    width, height = 240, 296
    canvas = Image.new("RGB", (width * 4, 30 + height * len(ranks)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, title in enumerate(
        (
            "GT: input view",
            f"{metadata['model']}: input view",
            "GT: opposite view",
            f"{metadata['model']}: opposite view",
        )
    ):
        draw.text((index * width + 6, 8), title, font=font, fill="black")
    selection = []
    for index, rank in enumerate(ranks):
        row = rows[rank]
        sample = inputs[row["sample_id"]]
        render_path = Path(sample["render_metadata"])
        reference = json.loads(render_path.read_text())
        first = reference["conditioning_view"]
        opposite = (first + len(reference["views"]) // 2) % len(reference["views"])
        predicted = directory / "renders" / row["sample_id"]
        reference_folder = (
            predicted / "reference/texture" if (predicted / "reference/texture").is_dir() else None
        )
        paths = (
            reference_folder / f"{first:03d}.png"
            if reference_folder
            else render_path.parent / reference["views"][first]["image"],
            predicted / f"{first:03d}.png",
            reference_folder / f"{opposite:03d}.png"
            if reference_folder
            else render_path.parent / reference["views"][opposite]["image"],
            predicted / f"{opposite:03d}.png",
        )
        for column, path in enumerate(paths):
            with Image.open(path).convert("RGBA") as image:
                background = Image.new("RGBA", image.size, "white")
                background.alpha_composite(image)
                canvas.paste(
                    background.convert("RGB").resize((width, width)),
                    (column * width, 30 + index * height),
                )
        label = f"rank {rank + 1}/{len(rows)}  {row['sample_id']}"
        draw.text((6, 30 + index * height + width + 3), label, font=font, fill="black")
        draw.text(
            (6, 30 + index * height + width + 23),
            f"CD={row['chamfer_l1']:.4f}  FS@0.1={row['fscore@0.1']:.4f}",
            font=font,
            fill="black",
        )
        selection.append(
            {"sample_id": row["sample_id"], "cd_rank": rank + 1, "views": [first, opposite]}
        )
    target = output / f"{dataset}_qualitative.png"
    canvas.save(target)
    write_json(
        output / f"{dataset}_qualitative.json",
        {"selection_rule": "ranks 1, one-third, two-thirds, last by CD", "objects": selection},
    )
    return target


def write_report(evaluations: list[Path], output: Path) -> None:
    """Export strict geometry, texture appearance, and source-material PBR separately."""
    definitions = {
        json.loads((path / "evaluation.json").read_text()).get(
            "image_metrics_definition", "scaled_source_material"
        )
        for path in evaluations
    }
    if len(definitions) > 1:
        raise ValueError("Image settings differ; summarize them separately")
    rows = []
    for directory in evaluations:
        metadata = json.loads((directory / "evaluation.json").read_text())
        for dataset, metrics in metadata["datasets"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "model": metadata["model"],
                    "protocol": metadata["profile"]["name"],
                    "device": metadata["device"],
                    **{key: value for key, value in metrics.items() if key != "metric_counts"},
                }
            )
    if not rows:
        raise ValueError("No dataset measurements to report")
    output.mkdir(parents=True, exist_ok=False)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 公开子集统一评测",
        "",
        "主图像指标采用固定尺度、无光照贴图 RGB；PBR 材质渲染独立成表。各指标有效分母见 evaluation.json。",
        "",
    ]
    sections = [
        (
            "几何与轮廓",
            [
                ("CD↓", "chamfer_l1"),
                ("FS@1%↑", "fscore@0.01"),
                ("FS@2%↑", "fscore@0.02"),
                ("FS@5%↑", "fscore@0.05"),
                ("HD95↓", "hausdorff95"),
                ("IoU↑", "mask_iou"),
            ],
        ),
        (
            "贴图 RGB（主指标）",
            [
                ("PSNR↑", "psnr"),
                ("SSIM↑", "ssim"),
                ("LPIPS↓", "lpips"),
                ("新视角 PSNR↑", "novel_psnr"),
                ("新视角 LPIPS↓", "novel_lpips"),
                ("前景并集 PSNR↑", "psnr_foreground"),
            ],
        ),
        (
            "原始材质 PBR（辅助指标）",
            [
                ("PSNR↑", "pbr_psnr"),
                ("SSIM↑", "pbr_ssim"),
                ("LPIPS↓", "pbr_lpips"),
                ("新视角 LPIPS↓", "pbr_novel_lpips"),
            ],
        ),
    ]
    if any("bounded_scale" in row for row in rows):
        sections.append(
            (
                "尺度诊断（主指标仍固定尺度）",
                [
                    ("有界尺度", "bounded_scale"),
                    ("固定尺度 IoU↑", "mask_iou"),
                    ("有界尺度 IoU↑", "bounded_mask_iou"),
                    ("固定尺度 PSNR↑", "psnr"),
                    ("有界尺度 PSNR↑", "bounded_psnr"),
                ],
            )
        )
    for title, fields in sections:
        lines += [
            f"## {title}",
            "",
            "| 数据 | 有效/计划 | " + " | ".join(label for label, _ in fields) + " |",
            "| --- | --- | " + " | ".join("---" for _ in fields) + " |",
        ]
        for row in rows:
            values = [f"{row[key]:.5f}" if row.get(key) is not None else "—" for _, key in fields]
            lines.append(
                f"| {row['dataset']} | {row['successful']}/{row['planned']} | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")
    for directory in evaluations:
        metadata = json.loads((directory / "evaluation.json").read_text())
        for dataset in metadata["datasets"]:
            figure = qualitative_sheet(directory, dataset, output)
            if figure:
                lines += [f"[{dataset} 固定排名案例图]({figure.name})", ""]
    lines += ["论文表 1 的宽阈值保留在 CSV；不同图像设置的实验分别汇总。", ""]
    (output / "comparison.md").write_text("\n".join(lines))
    print(f"Standard comparison: {output / 'comparison.md'}")
