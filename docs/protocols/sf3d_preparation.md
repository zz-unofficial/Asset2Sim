# SF3D 实验准备与运行

## 环境与权重

SF3D 通过官方子模块 `third_party/stable-fast-3d` 接入，适配代码位于 `src/asset2sim/generation/sf3d.py`。本机使用 RTX 5060 8GB，Conda 环境为 `asset2sim-sf3d`，配置 Python 3.10、PyTorch 2.8.0/CUDA 12.8 和 torchvision 0.23.0。

环境定义和 Python 依赖集中在 `environments/sf3d/`。安装入口编译官方 `texture_baker`、`uv_unwrapper` 扩展，自动识别 GPU 架构；交叉构建可设置 `SF3D_CUDA_ARCH_LIST`。

```bash
bash scripts/setup_sf3d.sh
conda activate asset2sim-sf3d
asset2sim fetch --model sf3d --config configs/models/sf3d.yaml
```

首次使用时，在 [官方模型页](https://huggingface.co/stabilityai/stable-fast-3d) 开通访问，并用 `huggingface-cli login` 登录。权重命令缓存 SF3D、DINOv2 和 CLIP，存放于 `.local/huggingface/hub/`；普通 RGB 输入使用的 U2NET 缓存在 `.local/u2net/`。

## 数据与配置

| 清单 | 内容 | 用途 |
| --- | --- | --- |
| `data/manifests/debug.jsonl` | 盒、杯、孔环、L 形支架，各两个视角，共 8 张图 | 各模型共用的参考几何开发输入 |
| `data/manifests/sf3d_examples.jsonl` | 官方椅子、斧头、岩石、树，共 4 张图 | 官方样例复现 |
| `data/samples/sf3d/manifest.jsonl` | 上述两组，共 12 张图 | 已有命令和运行记录沿用的清单 |

参考实体与渲染图片在 `data/samples/sf3d/`，图片为 512×512 RGBA，参考网格为 Z 轴向上、米制、最长边 0.20m。程序化数据按 CC0-1.0 提供；官方图片引用子模块中的文件。仓库已包含这些开发数据。

重建样例时使用 `asset2sim prepare-sf3d-data --output <新目录>`。从新清单发布某个分组可执行 `asset2sim export-manifest --source <清单> --split debug --output <新清单>`，文件路径会自动重设为相对新清单的位置。

模型默认参数在 `configs/models/sf3d.yaml`：种子 0、BF16、前景比例 0.85、纹理分辨率 1024、`remesh=none`。实验配置在 `configs/experiments/`，选择模型与数据清单，并通过 `overrides` 设置本次变化。

## 统一运行接口

参考开发集：

```bash
asset2sim run --config configs/experiments/sf3d_debug.yaml \
  --output data/generated/sf3d/debug/run01
```

官方样例：

```bash
asset2sim run --config configs/experiments/sf3d_examples.yaml \
  --output data/generated/sf3d/examples/run01
```

每张图独立推理，模型只加载一次。可用 `--sample-id official_chair1` 选择官方样例中的椅子，或重复此参数选择多张。原命令 `asset2sim sf3d --config configs/sf3d.yaml --manifest <清单> --output <新目录>` 继续可用。

每轮输出包含 `run.json`、实际输入快照 `inputs.jsonl`、逐输入记录 `results.jsonl`，以及各样例的 `input.png` 和 `mesh.glb`。公共运行器保留对象/视角 ID、成功或失败状态、配置与资源测量，后续模型复用同样的格式。

```bash
asset2sim summarize data/generated/sf3d/debug/run01 \
  --output results/metrics/sf3d_debug_run01.csv
```

`summarize` 支持一次读取多个运行目录，也可直接读取已有 `data/generated/sf3d/chair_run01`。表格每行对应一个输入，便于按对象和视角配对比较。

## 完成情况与效果分析

已有 `chair_run01` 在 RTX 5060 上完成 12/12 个输入，覆盖 8 张参考渲染图和 4 张官方示例；结果包含可读取的纹理 GLB，PyTorch 峰值分配约 6.0GiB。统一实验入口已完成椅子推理，旧版和新版记录可合并导出为同一 CSV，供后续几何和碰撞评测关联。

官方原始 GLB 为 Y 轴向上、模型归一化单位。运行记录保存 `raw_to_model_z_up` 旋转矩阵，评测时结合相机和尺度将生成 Mesh 与参考网格对齐。

RGB 背景处理使用 CPU ONNX Runtime，RGBA 输入沿用已有掩码。本次环孔 RGB 样例中，自动抠图保留了孔内白色背景，因此模型间定量比较使用统一 RGBA 输入，另行分析真实照片的背景处理影响。

论文表 1 与附录表 A1 的真实数据、指标参数和运行入口见 [SF3D 论文实验复现](sf3d_paper_reproduction.md)。按 [实验框架](reproduction_experiment_framework.md) 接入其余模型，并计算几何、碰撞和交互指标。新适配器的目录与接口约定见 [AGENTS.md](../../AGENTS.md)。

## 官方来源

- [SF3D 官方代码与许可](https://github.com/Stability-AI/stable-fast-3d)
- [SF3D 官方权重](https://huggingface.co/stabilityai/stable-fast-3d)
- [SF3D 论文](https://arxiv.org/abs/2408.00653)

Powered by Stability AI。SF3D 使用遵循 Stability AI Community License，许可副本保留在官方子模块中。
