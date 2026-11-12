"""Place actual measurements and the published SF3D reference rows side by side."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from asset2sim.artifacts import read_jsonl, write_json
from asset2sim.paths import PROJECT_ROOT


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
        ("GT: input view", "SF3D: input view", "GT: opposite view", "SF3D: opposite view")
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
        paths = (
            render_path.parent / reference["views"][first]["image"],
            predicted / f"{first:03d}.png",
            render_path.parent / reference["views"][opposite]["image"],
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
    reference = json.loads((PROJECT_ROOT / "experiments/references/sf3d_paper.json").read_text())
    rows = []
    names = [
        "chamfer_l1",
        "fscore@0.1",
        "fscore@0.2",
        "fscore@0.5",
        "psnr",
        "ssim",
        "lpips",
        "image_to_mesh_seconds",
    ]
    for directory in evaluations:
        evaluation = json.loads((directory / "evaluation.json").read_text())
        for dataset, metrics in evaluation["datasets"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "source": "measured_public_subset",
                    "device": evaluation["device"],
                    "planned": metrics["planned"],
                    "successful": metrics["successful"],
                    **{name: metrics.get(name) for name in names},
                }
            )
            if dataset in reference["table_1"]:
                paper = reference["table_1"][dataset]
                rows.append(
                    {
                        "dataset": dataset,
                        "source": "paper_reported",
                        "device": reference["runtime"]["gpu"],
                        "planned": paper["objects"],
                        "successful": paper["objects"],
                        **{
                            name: paper.get(
                                name,
                                reference["table_a1"][dataset].get(
                                    name, reference["runtime"].get(name)
                                ),
                            )
                            for name in names
                        },
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SF3D 论文指标与公开子集实测",
        "",
        "| 数据 | 来源 | 设备 | 有效/计划 | CD | FS@0.1 | FS@0.2 | FS@0.5 | PSNR | SSIM | LPIPS | 图像→Mesh 秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        values = [f"{row[name]:.4f}" if row.get(name) is not None else "—" for name in names]
        lines.append(
            f"| {row['dataset']} | {row['source']} | {row['device']} | {row['successful']}/{row['planned']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "实测采用固定公开对象清单，数据与数值设置见各 evaluation.json。论文行来自 [SF3D 表 1 与表 A1](https://arxiv.org/abs/2408.00653)。运行时间对应各行记录的设备。",
        "",
    ]
    for directory in evaluations:
        metadata = json.loads((directory / "evaluation.json").read_text())
        for dataset in metadata["datasets"]:
            figure = qualitative_sheet(directory, dataset, output)
            if figure:
                lines += [
                    f"[查看 {dataset} 重建对照图]({figure.name})：按 CD 排名固定选择四个位置，展示输入与相反视角。",
                    "",
                ]
    (output / "comparison.md").write_text("\n".join(lines))
    print(f"Paper comparison: {output / 'comparison.md'}")
