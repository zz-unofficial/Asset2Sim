# Repository Guidelines

## 协作目标与文档入口

Asset2Sim 当前围绕 SF3D、SPAR3D、TRELLIS.2 开展预训练推理复现与统一评测，比较资产几何、碰撞表示、交互表现和运行成本。SF3D 已接入；其余模型按相同接口补充。实验范围、数据规模和阶段安排见 `docs/protocols/reproduction_experiment_framework.md`，课程交付要求见 `课程要求.md`。

README 由项目负责人维护。环境、实验命令和结果分析写入对应协议文档；协作结构与接口约定维护在本文件。开始修改前检查工作区，保留队友已有改动和实验输出。

## 目录与职责

| 路径 | 内容与维护方式 |
| --- | --- |
| `environments/<model>/` | 各模型独立的 `environment.yml` 与 `requirements.txt`，固定兼容的 Python、PyTorch、CUDA 和上游依赖 |
| `scripts/setup_<model>.sh` | 环境安装、上游扩展编译、项目包安装；从仓库根目录调用 |
| `third_party/<upstream>/` | 官方 Git 子模块，记录固定版本；适配写在项目源码中 |
| `configs/models/<model>.yaml` | 模型默认参数：权重标识、精度、生成与导出选项 |
| `configs/experiments/<experiment>.yaml` | 模型选择、共享数据清单、样例选择和本次参数覆盖 |
| `configs/datasets/`、`configs/evaluation/` | 对象选择与渲染配置；归一化、对齐、采样与指标参数 |
| `experiments/datasets/`、`experiments/references/` | 固定对象列表与来源；论文原始指标和表号 |
| `data/manifests/<dataset>.jsonl` | 模型间共用的输入清单；按对象与视角管理 |
| `data/samples/` | 小型参考网格、渲染图和来源说明；大数据放入被忽略的 `data/raw/` |
| `src/asset2sim/generation/` | 模型适配器、接口协议和注册表 |
| `src/asset2sim/data/` | 清单解析、程序化参考几何和数据准备 |
| `src/asset2sim/geometry/` | Mesh 物理准备、坐标变换、可选清理与几何报告 |
| `src/asset2sim/collision/` | 单凸包和 CoACD 碰撞组件构建、几何误差与失败记录 |
| `src/asset2sim/simulation/` | MuJoCo 碰撞组件加载、独立参考接触及受控轨迹实验 |
| `src/asset2sim/experiments/` | 配置解析、公共运行器、结果表导出 |
| `src/asset2sim/evaluation/` | 公共几何对齐、表面距离、图像指标与论文对照表 |
| `results/raw/<experiment>/<run_id>/` | 逐对象/视角指标、对齐矩阵与评测渲染 |
| `data/generated/<model>/<condition>/<run_id>/` | 每次生成的独立输出与运行记录 |
| `results/metrics/`、`results/figures/` | 从运行记录导出的 CSV 与比较图 |
| `docs/protocols/`、`tests/` | 关键操作与分析；公共接口、几何和评测逻辑测试 |

物理几何的输入声明与输出见 [Mesh 物理准备](docs/protocols/mesh_physics_preparation.md)；碰撞组件的输入、配置与输出见 [碰撞几何构建](docs/protocols/collision_geometry.md)，接触实验见 [MuJoCo 接触评测](docs/protocols/mujoco_contact_evaluation.md)。仿真须逐个加载导出的凸组件。物理实验使用独立的 `asset2sim-physics` Conda 环境；各模型的 GPU 依赖由自己的环境管理。

## 接入一个模型

1. 将官方代码作为子模块放入 `third_party/`，配套提交 `.gitmodules` 与子模块引用；在协议文档记录来源、权重访问方式和关键版本。
2. 新建 `environments/<model>/` 和 `scripts/setup_<model>.sh`，提供独立 Conda 环境，并在其中以 editable 方式安装本项目。
3. 实现 `generation/<model>.py`，遵循 `generation/base.py` 的 `Generator` 协议，提供 `create_generator(parameters)` 与 `fetch_assets(parameters)`；在 `generation/registry.py` 注册模型名和模块路径。
4. 新建模型配置，再复制已有实验配置，复用同一数据清单和样例 ID。只在 `overrides` 中填写本次变化的参数。
5. 用一个已有输入验证加载、生成、GLB 读取与结果汇总，再运行完整清单。将操作和观察写入 `docs/protocols/<model>_preparation.md`。

适配器负责 `load()`、`generate(image_path, output_dir)`、`metadata()` 与 `recover()`，`parameters` 返回解析后的参数字典。`generate()` 创建样例目录，保存实际输入 `input.png` 与原始 `mesh.glb`，返回 `vertices`、`faces`、`preprocess_seconds`、`inference_seconds`、`export_seconds`、`peak_allocated_mib`；其他测量可增加字段。`metadata()` 记录权重版本、软件/设备、原始坐标轴与单位。公共运行器负责清单选择、批量调度和记录，新增模型沿用这套逻辑。

时间单位为秒、显存为 MiB；推理时间覆盖网络、网格构建及贴图生成，加载与导出分别计时。模型保留原始输出朝向和尺度，评测阶段使用明确的变换统一坐标。输入参考几何由评测模块使用，生成接口接收单张输入图片。

## 配置与数据约定

实验配置示例：

```yaml
name: sf3d_debug
model: sf3d
model_config: ../models/sf3d.yaml
manifest: ../../data/manifests/debug.jsonl
# 相对路径从当前 YAML 所在目录解析。
overrides:
  seed: 0
```

实验中的 `model` 是注册名；模型配置中的 `model` 是官方权重标识或本地目录。参数优先级为模型默认值→实验 `overrides`。`sample_ids` 可选择清单子集；命令行 `--sample-id` 优先于该选择。`extends` 可继承同类 YAML，覆盖采用顶层字段替换。

数据清单每行包含 `sample_id`、`object_id`、`view_id`、`image`、`split`；有参考几何时填写 `reference_mesh`，同时记录类别、来源、许可和相机。文件路径相对清单所在目录。对象 ID 与视角 ID 跨模型保持一致，后续表格按这些字段配对。

论文评测数据额外填写 `dataset` 与 `render_metadata`，后者指向保存相机矩阵、参考尺度变换、输入视角及 16 张参考图路径的 JSON。公开对象列表固定后共用，准备阶段处理纹理和视角，生成阶段仅接收输入图。

共享开发输入使用 `data/manifests/debug.jsonl`（4 对象、8 张图）；SF3D 官方样例使用 `sf3d_examples.jsonl`（4 张图）。正式实验另建清单，按对象划分开发与评测集合。

## 运行与比较

在相应模型的 Conda 环境中，从仓库根目录执行：

```bash
asset2sim run --config configs/experiments/sf3d_debug.yaml \
  --output data/generated/sf3d/debug/run01
asset2sim summarize data/generated/sf3d/debug/run01 \
  --output results/metrics/sf3d_debug_run01.csv
```

`summarize` 可接收多个运行目录，导出统一的逐输入 CSV，包含实验、模型、对象/视角、状态、资源测量与资产路径。比较时使用同一清单、物性和任务配置；几何与成功率先按对象汇总，运行时间在统一硬件和计时口径下比较。

每轮运行保存 `run.json`（解析后参数与环境）、`inputs.jsonl`（实际输入快照）、`results.jsonl`（每个输入的测量与状态）及样例目录。新输出目录和 CSV 文件使用独立名称，已有结果保留原位。失败保留记录并计入对应分母。旧版 SF3D 命令与已有运行记录继续可读，`configs/sf3d.yaml` 指向新的模型配置。

## 论文指标复现

公开子集的模型比较使用 [SF3D 论文复现协议](docs/protocols/sf3d_paper_reproduction.md)、`data/manifests/sf3d_*_public.jsonl` 和 `configs/evaluation/sf3d_paper_public.yaml`。执行 `asset2sim evaluate --run <生成目录> --config configs/evaluation/sf3d_paper_public.yaml --output <新评测目录>`，再用 `asset2sim paper-report <评测目录> --output <新报告目录>` 汇总。

几何使用归一化、旋转搜索及刚体 ICP 后的 CD、严格 F-score 和距离分位数。图像主评测固定尺度，分开报告无光照贴图 RGB 与原始 PBR 材质结果，并区分输入/新视角和轮廓 IoU；受约束尺度仅用于校准诊断。数据准备显式声明源向上轴与渲染模式。各模型复用同一数据、评测配置与渲染器。修正实现时直接维护原模块与配置，只有开展独立对照实验才增加配置。已有实验结果保留，重跑使用新 run_id；不同图像设置分别汇总。分析同时给出每项指标的有效/计划对象数、数据选择规则和设备。

## 代码、验证与交付

Python 使用四空格缩进，函数/模块采用 `snake_case`，类采用 `PascalCase`，公共接口标注类型、输入输出与单位。pytest 和 Ruff 配置位于 `pyproject.toml`；测试命名为 `tests/test_<behavior>.py`，优先覆盖配置解析、坐标处理、失败记录和结果配对。

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
```

文档围绕“怎么运行、完成了什么、结果如何解释”组织，保留影响复现的关键命令和结论。临时 smoke 使用临时目录，分析后清理；正式实验保留原始记录。版本引用按关键运行记录一次，通常无需额外 SHA 校验。面向用户的说明关注行为、结果与实际问题。

提交建议使用 `feat:`、`fix:`、`test:`、`docs:`、`exp:`。PR 说明改动目的、配置变化与验证结果，实验或视觉变化附结果表或图。权重、大型资产和缓存由 `.gitignore` 管理；付费算力按已确认预算执行。
