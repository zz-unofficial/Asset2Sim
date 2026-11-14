# SF3D 与 SPAR3D 模型复现和统一评测

使用 SF3D 与 SPAR3D 生成三维资产，在统一的 GSO 和 OmniObject3D 公开子集上比较几何、外观与运行成本。

## 实验配置与运行

### SF3D 环境与生成

#### 环境与权重

SF3D 通过官方子模块 `third_party/stable-fast-3d` 接入，适配代码位于 `src/asset2sim/generation/sf3d.py`。本机使用 RTX 5060 8GB，Conda 环境为 `asset2sim-sf3d`，配置 Python 3.10、PyTorch 2.8.0/CUDA 12.8 和 torchvision 0.23.0。

环境定义和 Python 依赖集中在 `environments/sf3d/`。安装入口编译官方 `texture_baker`、`uv_unwrapper` 扩展，自动识别 GPU 架构；交叉构建可设置 `SF3D_CUDA_ARCH_LIST`。

```bash
bash scripts/setup_sf3d.sh
conda activate asset2sim-sf3d
asset2sim fetch --model sf3d --config configs/models/sf3d.yaml
```

首次使用时，在 [官方模型页](https://huggingface.co/stabilityai/stable-fast-3d) 开通访问，并用 `huggingface-cli login` 登录。权重命令缓存 SF3D、DINOv2 和 CLIP，存放于 `.local/huggingface/hub/`；普通 RGB 输入使用的 U2NET 缓存在 `.local/u2net/`。

#### 数据与配置

| 清单 | 内容 | 用途 |
| --- | --- | --- |
| `data/manifests/debug.jsonl` | 盒、杯、孔环、L 形支架，各两个视角，共 8 张图 | 各模型共用的参考几何开发输入 |
| `data/manifests/sf3d_examples.jsonl` | 官方椅子、斧头、岩石、树，共 4 张图 | 官方样例复现 |
| `data/samples/sf3d/manifest.jsonl` | 上述两组，共 12 张图 | 已有命令和运行记录沿用的清单 |

参考实体与渲染图片在 `data/samples/sf3d/`，图片为 512×512 RGBA，参考网格为 Z 轴向上、米制、最长边 0.20m。程序化数据按 CC0-1.0 提供；官方图片引用子模块中的文件。仓库已包含这些开发数据。

重建样例时使用 `asset2sim prepare-sf3d-data --output <新目录>`。从新清单发布某个分组可执行 `asset2sim export-manifest --source <清单> --split debug --output <新清单>`，文件路径会自动重设为相对新清单的位置。

模型默认参数在 `configs/models/sf3d.yaml`：种子 0、BF16、前景比例 0.85、纹理分辨率 1024、`remesh=none`。实验配置在 `configs/experiments/`，选择模型与数据清单，并通过 `overrides` 设置本次变化。

#### 统一运行接口

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

### SPAR3D 环境与生成

SPAR3D 先采样点云，再结合输入生成纹理网格；每个样例保存 `mesh.glb` 和 `points.ply`。官方代码位于 `third_party/stable-point-aware-3d`，适配器复用统一生成和评测接口。

#### 环境与权重

```bash
bash scripts/setup_spar3d.sh
conda activate asset2sim-spar3d
asset2sim fetch --model spar3d --config configs/models/spar3d.yaml
```

首次使用需要在 [官方模型页](https://huggingface.co/stabilityai/stable-point-aware-3d) 接受访问条件，并在本机登录 Hugging Face。项目缓存位于 `.local/huggingface/hub/` 和 `.local/alpha_clip/`。本机 8GB 显卡使用官方 `low_vram=true`；默认 BF16、纹理 1024、裁剪比 1.3、种子 0、无重网格化。官方权重配置使用 32 次点扩散采样，运行记录保存实际步数和引导参数。RGBA 输入直接使用已有掩码；普通 RGB 输入使用官方背景移除器。

#### 运行与评测

小样例共用 `data/manifests/debug.jsonl`，两组正式对象与 SF3D 使用相同的 GSO、OmniObject3D 清单。运行目录使用新的 `run_id`：

两组数据依次生成、评测并与已有 SF3D 结果汇总，可直接执行 `bash scripts/run_spar3d_paper.sh run01`。运行状态和日志位于 `results/raw/spar3d_paper/run01_launcher/`。
已完成一组数据时，可执行 `bash scripts/run_spar3d_paper.sh run02 omni` 单独运行另一组；已有输出保持原位。

```bash
asset2sim run --config configs/experiments/spar3d_gso_public.yaml \
  --output data/generated/spar3d/paper_gso/run01
asset2sim evaluate --run data/generated/spar3d/paper_gso/run01 \
  --config configs/evaluation/sf3d_paper_public.yaml \
  --output results/raw/spar3d_paper_gso/run01
```

Omni 使用 `configs/experiments/spar3d_omni_public.yaml`，输出路径对应改为 `paper_omni`。评测沿用 [SF3D 统一协议](model_reproduction.md)：CD、F-score、距离分位数、轮廓 IoU、贴图 RGB 和 PBR 图像指标，分别记录输入与新视角。

```bash
asset2sim paper-report results/raw/sf3d_paper_gso/run02 \
  results/raw/spar3d_paper_gso/run01 --output results/metrics/gso_model_comparison
```

报告列出模型名并核对对象 ID 和评测参数；各模型按相同对象配对。生成记录包含种子、低显存设置、耗时、面数、峰值分配显存及点云路径。

#### 论文参考

[SPAR3D 论文](https://arxiv.org/html/2501.04689v1) 表 1/2：GSO 的 CD 0.120、FS@0.1 0.584、PSNR 18.6、SSIM 0.836、LPIPS 0.139；Omni 的 CD 0.122、FS@0.1 0.587、PSNR 17.9、SSIM 0.832、LPIPS 0.140。论文每组约 250 对象，移除简单盒体/圆柱，并使用多仰角、随机 HDRI 和不同焦距。本项目首轮沿用固定 24 对象与统一渲染，适合模型间比较。

论文中 SF3D 和 SPAR3D 的完整参考行记录于 `experiments/references/spar3d_paper.json`。这些数字采用 SPAR3D 论文的实验设置，与 SF3D 论文自己的表格分别记录。

### 共享数据与评测协议

#### 目标与参考值

使用官方预训练权重，在固定的 GSO、OmniObject3D 公开子集上测量几何、外观和图像到 Mesh 的成本。[SF3D 表 1 与附录 A1](https://arxiv.org/abs/2408.00653) 的参考值如下：

| 数据集 | 对象数 | CD↓ | FS@0.1↑ | FS@0.2↑ | FS@0.5↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | 278 | 0.098 | 0.701 | 0.894 | 0.988 | 21.247 | 0.865 | 0.124 |
| OmniObject3D | 308 | 0.090 | 0.726 | 0.920 | 0.990 | 20.134 | 0.851 | 0.132 |

论文报告 H100 上图像到最终纹理 Mesh 为 0.5s，27.4K 三角形对应一个示例。原始数字保存在 `experiments/references/sf3d_paper.json`。作者未公开具体的 278/308 对象清单，本项目各固定 24 个公开对象；模型比较复用同一数据和参数。

#### 数据与坐标

对象列表位于 `experiments/datasets/`；GSO 按名称排序后以 seed=0 抽样，Omni 在下载类别中按路径排序后以同一种子抽样。Omni 实际包含书籍 16、腰包 6、电池 1、床 1，类别结果单列。

GSO 原始资产位于 `data/raw/gso/`；Omni 从 [作者 Google Drive raw_scans](https://drive.google.com/drive/folders/1ZpGmUlXYf0uc9xtN_w9iOhxmt9_gHZaP) 下载 `bed.tar.gz`、`bumbag.tar.gz`、`battery.tar.gz`、`book.tar.gz`，放到 `data/raw/omniobject3d/`。两组来源和许可写入清单，Omni 许可见 [作者仓库](https://github.com/omniobject3d/OmniObject3D)。

准备配置维护在原有 `configs/datasets/sf3d_*_public.yaml`：GSO 使用 Z-up，Omni OBJ 使用 Y-up，转换到右手 Z-up，保存源到参考的变换。Omni 的约定依据 [作者渲染脚本](https://github.com/omniobject3d/OmniObject3D/blob/main/tools/render_blender.py) 的默认 OBJ 导入及 [Blender 导入器](https://github.com/blender/blender-addons/blob/blender-v3.6-release/io_scene_obj/__init__.py) 的 Y-up 默认值。

参考网格居中，最长边缩放为 1。固定 512×512、16 个等间隔方位角、仰角 15°、距离 1.6、FOV 40°。输入选择轮廓面积最大的视角，平局取首个；这是几何正面近似。输入采用无光照贴图 RGB 和共享 RGBA 掩码。GSO 纹理从 `materials/textures/texture.png` 补全引用。

当前准备数据位于 `data/processed/gso/`、`data/processed/omniobject3d/`，共享清单仍为 `data/manifests/sf3d_gso_public.jsonl`、`sf3d_omni_public.jsonl`。`render.json` 保存相机、源向上轴、渲染方式、输入视角与尺度变换。原始准备数据和已完成运行保留，旧输入快照继续引用原位置。

#### 模型和几何指标

模型配置仍为 `configs/models/sf3d.yaml`：512 输入、1024 贴图、BF16、seed=0、前景比例 0.85、无重网格化。模型有四个双流单元，triplane 为 96→384、40 通道，四面体分辨率 160；本机加载参数总数为 1,006,026,407。

评测参数集中在原有 `configs/evaluation/sf3d_paper_public.yaml`：

| 项目 | 设置与定义 |
| --- | --- |
| 几何归一化 | 预测、参考分别居中并将最长边缩放为 1 |
| 表面采样 | 两侧各 20,000 点，按三角形面积均匀采样，种子 0/1 |
| 几何对齐 | 1,024 点、30° Euler 旋转网格，再做最多 50 次刚体 ICP，容差 1e-6 |
| CD | 双向最近邻欧氏距离均值的平均；平方距离另存 |
| F-score | 最长边的 1%、2%、5% 用于细部比较；保留论文的 10%、20%、50% 阈值 |
| 距离尾部 | 双向距离分别取 P50/P95；HD95 为两侧 P95 的较大值 |

阈值内的预测点比例为 precision，反向为 recall，F-score 为二者调和平均。几何对齐不使用反射、缩放或图像颜色择优。上述采样量、尺度和搜索间隔是本项目固定的实现参数。

#### 图像指标与尺度控制

主评测使用几何刚体对齐并恢复参考坐标，相对尺度固定为 1，直接取消原来无约束的尺度 ICP。所有对象使用相同规则，不按分数选择对齐方式。参考和预测由同一渲染器、相机与设置重新渲染。

- **贴图 RGB 为主指标：** 无光照读取存储纹理 RGB，有贴图时基础色乘数固定为 1，忽略金属度、粗糙度和法线贴图。该项评价表面颜色，不将扫描贴图解释为真实 albedo。
- **原始材质 PBR 为辅助指标：** 保留资产材质，固定环境光 0.45、相机方向光 2.5；记录两侧基础色、金属度、粗糙度。该结果包含源材质转换与模型材质预测的共同影响。

白色背景合成，RGB 范围 [0,1]；SSIM 使用 sigma=1.5 的高斯窗口，LPIPS 使用 AlexNet v0.1；PSNR 完全一致时封顶 120dB。alpha>127 定义轮廓，计算 IoU、预测/参考前景面积比；另在两侧前景并集上计算 PSNR，结合 IoU 解释白色物体和缺失区域。

分别保存输入视角、其余 15 个新视角和全部 16 视角的指标。先按对象平均，再对对象等权汇总。每项指标保留有效分母，图像评测失败不丢弃已完成的几何结果。近对称物体仍可能在几何对齐后交换纹理正反面，结合固定排名案例图分析；不使用测试视角 RGB 搜索姿态。

#### 运行与记录

```bash
conda activate asset2sim-sf3d
asset2sim run --config configs/experiments/sf3d_gso_public.yaml \
  --output data/generated/sf3d/paper_gso/run02
bash scripts/run_sf3d_omni_paper.sh run02
```

脚本沿用原入口，运行 Omni 生成、两组指标及合并报告。已有 GSO 同编号生成可复用；两组输出都按常规 `run_id` 管理。重跑选择新编号，已有结果保留。

单独评测与汇总仍使用原命令：

```bash
asset2sim evaluate --run <生成目录> \
  --config configs/evaluation/sf3d_paper_public.yaml --output <新评测目录>
asset2sim paper-report <GSO评测目录> <Omni评测目录> --output <新报告目录>
```

输出包含逐对象 CSV/JSONL、变换、参考与预测渲染、两侧材质和逐视角指标；汇总包含类别统计和每项指标的实际分母。图像到 Mesh 时间累加预处理、网络/网格/贴图生成及导出，模型加载另计。不同图像设置分别汇总，参数随实际运行记录保存。

已有 run01 的数值与问题定位见本文实验结论。修正后的 run02 使用同一对象列表，但输入颜色和 Omni 朝向已调整，结果变化反映数据与评测流程的整体调整。

### 官方来源

- [SF3D 官方代码与许可](https://github.com/Stability-AI/stable-fast-3d)
- [SF3D 官方权重](https://huggingface.co/stabilityai/stable-fast-3d)
- [SF3D 论文](https://arxiv.org/abs/2408.00653)

Powered by Stability AI。SF3D 使用遵循 Stability AI Community License，许可副本保留在官方子模块中。

## 实验结论

### SF3D 开发样例

已有 `chair_run01` 在 RTX 5060 上完成 12/12 个输入，覆盖 8 张参考渲染图和 4 张官方示例；结果包含可读取的纹理 GLB，PyTorch 峰值分配约 6.0GiB。统一实验入口已完成椅子推理，旧版和新版记录可合并导出为同一 CSV，供后续几何和碰撞评测关联。

官方原始 GLB 为 Y 轴向上、模型归一化单位。运行记录保存 `raw_to_model_z_up` 旋转矩阵，评测时结合相机和尺度将生成 Mesh 与参考网格对齐。

RGB 背景处理使用 CPU ONNX Runtime，RGBA 输入沿用已有掩码。本次环孔 RGB 样例中，自动抠图保留了孔内白色背景，因此模型间定量比较使用统一 RGBA 输入，另行分析真实照片的背景处理影响。

论文表 1 与附录表 A1 的参考值、指标参数和运行入口见本文。按 [实验框架](reproduction_experiment_framework.md) 接入其余模型，并计算几何、碰撞和交互指标。新适配器的目录与接口约定见 [AGENTS.md](../../AGENTS.md)。

### 评测尺度诊断

已用旧生成资产中的 6 个固定案例进行尺度诊断：GSO 的瓶罐、茶具组、Frozen 盒，Omni 的 book_011、bumbag_002、bed_002。对照缩放总范围为 [0.9,1.1]，双向拟合，仅接受对称 CD 降低的步骤，最多 20 次；该诊断不替代固定尺度主结果。三个 GSO 案例的平均 IoU 从固定尺度的 0.707 变为有界尺度的 0.770，三个 Omni 案例从 0.772 变为 0.778。两种设置结果均保存，主评测统一固定尺度，避免按对象择优。实际渲染验证相同资产 SSIM=1、LPIPS=0、IoU=1，贴图模式不受基础色 0.4/1.0 差异影响。

### SF3D 公开子集 run02

两组 run02 均完成 24/24 对象，主评测相对尺度全部为 1，每项主要指标的有效分母均为 24；两种图像口径各覆盖 768 个对象视角。完整数值与案例图见 [run02 报告](../../results/metrics/sf3d_paper_public/run02/comparison.md)。

| 数据 | CD↓ | FS@1%↑ | FS@2%↑ | FS@5%↑ | 轮廓 IoU↑ | 贴图 PSNR↑ | 贴图 LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | 0.03283 | 0.27176 | 0.47149 | 0.78450 | 0.77626 | 16.35562 | 0.30162 |
| OmniObject3D | 0.08403 | 0.13721 | 0.26053 | 0.44117 | 0.67200 | 11.69075 | 0.43085 |

严格阈值区分了原先宽阈值下难以观察的表面误差；Omni 的比例和厚度误差仍明显，统一朝向与颜色输入并未使其几何表现自动改善。固定尺度避免了通过大幅缩小资产来贴合参考表面的情况。后续模型沿用本配置、共享输入清单及报告字段进行比较。

### SF3D 公开子集 run01 历史结果

#### 完成情况与主要指标

两组均完成 24/24 对象的生成和评测，共 48 份带纹理 Mesh、768 个评测视角。生成成功表示资产已导出且评测完成；该轮评测覆盖生成和图像、几何指标。

| 数据 | CD↓ | FS@0.1↑ | FS@0.2↑ | FS@0.5↑ | PSNR↑ | SSIM↑ | LPIPS↓ | 图像→Mesh 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSO，24 对象 | 0.03273 | 0.94344 | 0.99749 | 1.00000 | 15.00895 | 0.77041 | 0.28108 | 0.66009 |
| OmniObject3D，24 对象 | 0.04523 | 0.95422 | 1.00000 | 1.00000 | 9.12035 | 0.73639 | 0.39717 | 0.66141 |

数值来自两组 `results/raw/sf3d_paper_*/run01/metrics.jsonl`；[合并对照表](../../results/metrics/sf3d_paper_public/run01/comparison.md) 同时保留论文参考值。运行设备均为 RTX 5060，PyTorch 峰值分配显存约 6.0GiB；GSO/Omni 平均面数分别为 25,629/22,404。去掉各轮首对象后，图像到 Mesh 平均耗时分别为 0.638/0.637s，说明稳态生成成本接近。

#### 几何表现与指标区分度

run01 采用最长边为 1 的归一化，FS@0.1 允许最大边长 10% 的表面距离。GSO 有 9/24 个对象、Omni 有 13/24 个对象在该阈值取得 1；FS@0.2/0.5 更接近完全饱和。因此高 F-score 主要支持整体形状接近；文字、薄壁、孔洞和小部件适合结合严格阈值与案例图检查。

GSO 的 CD 均值/中位数为 0.03273/0.02664，较差对象拉高均值。`Frozen_Scrabble_Jr` 的 CD=0.08412、FS@0.1=0.48685，重建厚度明显偏大；卷尺、家具组、文件架也是几何误差较大的对象。几何误差最小的 Wilton 长盒 CD=0.00431，但对照图仍可见贴图模糊和背面内容猜测，说明几何与纹理需要分别评价。

Omni 的 CD 均值/中位数为 0.04523/0.04683，两项统计值接近。Omni 的 FS@0.1 略高于 GSO，但 CD 更大：F-score 统计阈值内的点比例，CD 统计实际距离，分别反映不同的误差特征。

#### 外观差距包含评测流程影响

Omni 的 PSNR 范围为 7.46–11.57dB，整体偏低。查看 [Omni 对照图](../../results/metrics/sf3d_paper_public/run01/omniobject3d_qualitative.png) 与保存的变换、材质后，发现两项需要优先处理的流程问题。

1. **图像对齐出现明显缩小。** 相比归一化后的刚体对齐，图像阶段的尺度 ICP 在 GSO 上平均缩放至 0.894 倍，Omni 上为 0.721 倍；`omni_book_book_002` 缩放至 0.423 倍。对照图里的书因此明显小于参考图。当前 `icp(scale=True)` 使用单向最近邻拟合且未限制尺度，存在通过缩小预测来降低拟合误差的风险。该变换只用于图像指标，不影响本表 CD/F-score；它对图像分数的贡献需要固定尺度与受约束尺度的对照来量化。
2. **参考与预测材质存在亮度口径差异。** Omni 24 个参考 GLB 中，23 个的 `baseColorFactor` 为 `[102,102,102,255]`，即基础色系数约 0.4；参考贴图会再乘该系数。加载器的 OBJ→PBR 转换会继承 diffuse 参数，预测资产则使用模型输出材质。对照图中参考物体偏暗、重建偏亮，PSNR 因而混入材质转换、光照和模型去光照的差异。图像差距需结合这些材质和光照设置解释。

部分书籍封面和床的输入方向也不自然，需核对扫描数据的原始向上轴与统一相机约定。现有“最大轮廓面积”规则保证可重复，所选视角以轮廓面积为准。几何对齐还可能交换近对称对象的正反面；[SF3D 附录 A2](https://arxiv.org/html/2408.00653v1#A2) 也将这种纹理朝向歧义列为图像指标的限制。

#### 数据组成与论文对照

Omni 子集实际包含书籍 16、腰包 6、电池 1、床 1。书籍占 66.7%，平均 CD=0.0501、PSNR=8.5609dB、LPIPS=0.4590；腰包对应 0.0359、10.1986dB、0.2620。总体均值因此主要反映书籍案例，类别覆盖以这四类对象为准。

论文报告 GSO/Omni CD 为 0.098/0.090，PSNR 为 21.247/20.134dB。本次 CD 较小、图像分数较低；两者使用的对象清单、归一化和渲染参数不同，run01 的图像对齐与材质口径也存在上述差异。论文原值用于定位复现目标；后续跨模型结论来自统一数据和评测设置下的配对比较。

### SPAR3D 运行与模型对照

批量推理在对象之间释放低显存模式的阶段模块，并启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。AlphaCLIP 资源在进程内复用首次成功的缓存定位，避免逐对象重复读取并校验大文件。RTX 5060 的连续 6 对象验证全部成功，PyTorch 峰值分配约 6,440–6,494 MiB，平均推理约 10.84 秒；早期 `run01` 记录了 1/24 成功及显存不足问题，保留用于定位。

#### GSO 实测与运行情况

SPAR3D 的 `run02` 已完成 GSO、Omni 各 24/24 个对象的生成和各 384 个视角的评测。GSO 案例图文件名错误已修复；已有指标重新汇总到 [GSO 对照报告](../../results/metrics/spar3d_gso_comparison/run03/comparison.md)。Omni 随后单独续跑并正常完成，结果见 [Omni 对照报告](../../results/metrics/spar3d_omni_comparison/run02/comparison.md)。

| 指标 | SF3D | SPAR3D |
| --- | --- | --- |
| CD↓ | 0.03283 | 0.03127 |
| FS@1%↑ | 0.27176 | 0.31358 |
| HD95↓ | 0.08645 | 0.09755 |
| 轮廓 IoU↑ | 0.77626 | 0.76290 |
| 贴图 PSNR↑ | 16.35562 | 14.96879 |
| 贴图 LPIPS↓ | 0.30162 | 0.30132 |
| 图像到 Mesh / 秒 | 0.664 | 10.072 |

逐对象比较中，SPAR3D 的 CD 在 12/24 个对象上较低，FS@1% 在 16/24 个对象上较高，PSNR 在 5/24 个对象上较高。Frozen 游戏盒的 CD 明显改善，篮筐和背包案例变差；图像对照也可见几何对齐可能造成正反面或上下方向的纹理错位。当前数据只支持本机固定子集和单次种子的描述性比较。

两模型均在 RTX 5060 上运行；SPAR3D 使用低显存模式及阶段模块加载。Mesh 时间包含预处理、推理和 Mesh 导出，点云导出另行计时。计时结果对应本机硬件和上述流程。

#### Omni 实测与类别分析

| 指标 | SF3D | SPAR3D |
| --- | --- | --- |
| CD↓ | 0.08403 | 0.03241 |
| FS@1%↑ | 0.13721 | 0.21089 |
| HD95↓ | 0.15799 | 0.07990 |
| 轮廓 IoU↑ | 0.67200 | 0.79100 |
| 贴图 PSNR↑ | 11.69075 | 12.39446 |
| 前景并集 PSNR↑ | 7.61950 | 7.33876 |
| 贴图 LPIPS↓ | 0.43085 | 0.32097 |
| 图像到 Mesh / 秒 | 0.683 | 10.319 |

SPAR3D 在 17/24 个对象上获得更低 CD，在 18/24 个对象上获得更高整图 PSNR。该子集包含 16 本书、6 个腰包、1 个电池和 1 张床：书籍 16/16 的 CD 与整图 PSNR 均有改善，书籍平均 CD 从 0.1110 降至 0.0296；另外 8 个对象中只有 1 个的 CD 更低。总体差异主要由书籍构成驱动，不宜推断对所有类别都有优势。

SPAR3D 的整图 PSNR 和 LPIPS 较好，但前景并集 PSNR 略低；轮廓、白色背景面积与纹理颜色共同影响整图指标。案例中还存在贴图亮度差异和几何对齐后的正反面、上下方向错位，需结合参考图解释。

两组 GSO 和 Omni 各使用同一对象与评测配置、各一个推理种子。结果支持本机固定子集下的比较；扩大类别覆盖和重复随机种子后才能进一步判断稳定性。资产的碰撞和交互结果见 [受控接触与跨来源评测](mujoco_contact_evaluation.md)。
