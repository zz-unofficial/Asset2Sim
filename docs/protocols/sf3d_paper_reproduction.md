# SF3D 论文实验复现

## 目标与论文数值

复现 [SF3D](https://arxiv.org/abs/2408.00653) 表 1 的几何指标和附录表 A1 的图像指标，使用官方预训练权重生成带纹理 Mesh。论文基于 GSO 随机 278 个对象、OmniObject3D 随机 308 个对象，各渲染 16 个视角并选一张正面图作为输入。

| 数据集 | CD↓ | FS@0.1↑ | FS@0.2↑ | FS@0.5↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO（278） | 0.098 | 0.701 | 0.894 | 0.988 | 21.247 | 0.865 | 0.124 |
| OmniObject3D（308） | 0.090 | 0.726 | 0.920 | 0.990 | 20.134 | 0.851 | 0.132 |

论文图像到最终纹理 Mesh 的时间为 H100 上 0.5s。文中的 27.4K 三角形来自单个示例，不作为全数据集平均值。原始参考数字按表号保存在 `experiments/references/sf3d_paper.json`。

## 公开数据准备

首轮各选 24 个对象，用于跑通真实数据上的完整评测。作者未公开上述 278/308 对象的 ID 清单，本轮使用独立、固定的公开子集。GSO 从官方 GoogleResearch 目录按名称排序后以 seed=0 抽样；Omni 从指定类别压缩包内按路径排序后以同一种子抽样。选择结果保存在 `experiments/datasets/`，队友沿用同一份文件。

### GSO

```bash
conda activate asset2sim-sf3d
asset2sim prepare-paper-data --config configs/datasets/sf3d_gso_public.yaml \
  --output data/processed/sf3d_gso_public \
  --manifest data/manifests/sf3d_gso_public.jsonl
```

原始 ZIP 和解压文件缓存到 `data/raw/gso/`。GSO 材质的 `texture.png` 位于 `materials/textures/`，准备器补全该引用后导出带贴图参考 GLB。

### OmniObject3D

从作者提供的 [Google Drive 数据总目录](https://drive.google.com/drive/folders/1-hlkTqUJR_J2xht1qztmX2xAq_qNw4vn) 进入 [raw_scans](https://drive.google.com/drive/folders/1ZpGmUlXYf0uc9xtN_w9iOhxmt9_gHZaP)，下载 `bed.tar.gz`、`bumbag.tar.gz`、`battery.tar.gz`、`book.tar.gz`，放入 `data/raw/omniobject3d/`。四类压缩包合计约 446MiB；配置使用这些类别中的 24 个对象，属于类别受限的子集。来源和 CC BY 4.0 许可见 [作者仓库](https://github.com/omniobject3d/OmniObject3D)。

```bash
asset2sim prepare-paper-data --config configs/datasets/sf3d_omni_public.yaml \
  --output data/processed/sf3d_omni_public \
  --manifest data/manifests/sf3d_omni_public.jsonl
```

准备器直接读取本机压缩包。两组数据均输出参考 GLB、16 张 RGBA 图、相机与尺度变换记录，再发布共享 JSONL 清单。单张输入采用 16 个视角中轮廓面积最大的视角，作为固定的正面近似；选择在推理前完成，并记录在 `render.json`。

## 模型与评测参数

| 项目 | 本轮设置 |
| --- | --- |
| 模型输入 / 贴图 | 512×512 / 1024×1024 |
| 推理 | BF16、seed=0、前景比例 0.85、无重网格化 |
| 官方模型结构 | 四个双流单元；triplane 96→384、40 通道；四面体分辨率 160；本机加载参数总数 1,006,026,407 |
| 参考渲染 | 512×512、16 个等间隔方位角、仰角 15°、距离 1.6、垂直 FOV 40° |
| 归一化 | 包围盒居中，最长边缩放为 1；预测与参考分别处理 |
| 表面采样 | 各 20,000 点，按面积均匀采样，固定种子 |
| 朝向与位置 | 1,024 点、30° Euler 旋转网格选最小 CD，随后最多 50 次刚体 ICP，容差 1e-6 |
| 图像对齐 | 几何对齐后，最多 20 次带尺度 ICP，恢复参考坐标 |
| 图像指标 | 白色背景合成、RGB [0,1]；SSIM 高斯窗口 sigma=1.5；LPIPS AlexNet |

论文给出归一化、旋转搜索和 ICP 流程；表中采样量、归一化尺度、相机、搜索间隔及图像指标实现是本项目固定的实现参数，集中在 `configs/datasets/` 与 `configs/evaluation/`。因此公开子集实测适合在同一协议下比较模型，不能仅凭与论文表格的数值差判断复现是否成功。

CD 取预测到参考、参考到预测的最近邻欧氏距离均值的平均，另存平方距离版本便于核对定义。FS@τ 为距离小于 τ 的 precision 与 recall 的调和平均。图像指标覆盖全部 16 个视角（含输入视角），先逐视角计算，再按对象平均，最终各对象等权汇总。几何对齐不使用反射或尺度优化；尺度细化仅用于附录图像指标。

## 生成、评测与汇总

Omni 压缩包齐全后，一次启动数据准备、生成、评测和两组数据的合并报告：

```bash
bash scripts/run_sf3d_omni_paper.sh run01
```

该入口使用 `asset2sim-sf3d` Conda 环境，复用已有 GSO `run01` 指标，合并报告输出至 `results/metrics/sf3d_paper_public/run01/`。终端显示各阶段进度，日志与阶段状态保存在 `results/raw/sf3d_paper_omni/run01_launcher/`。重跑时更换运行编号。

```bash
asset2sim run --config configs/experiments/sf3d_gso_public.yaml \
  --output data/generated/sf3d/paper_gso/run01
asset2sim evaluate --run data/generated/sf3d/paper_gso/run01 \
  --config configs/evaluation/sf3d_paper_public.yaml \
  --output results/raw/sf3d_paper_gso/run01
asset2sim paper-report results/raw/sf3d_paper_gso/run01 \
  --output results/metrics/sf3d_paper_public
```

Omni 数据就绪后，使用 `configs/experiments/sf3d_omni_public.yaml`，对应输出目录命名为 `paper_omni`。汇总命令可以同时传入两组评测目录。已有目标目录保留，重跑更换 `run_id`。

生成记录保存实际模型参数、结构参数、硬件、逐对象耗时、面数与 PyTorch 峰值分配显存。评测记录包含 `metrics.jsonl/csv`、`evaluation.json`、逐对象对齐矩阵、16 张重建渲染和逐视角指标。图像到 Mesh 时间累加预处理、网络/网格/贴图生成和导出，模型加载另计。汇总表同时报告有效/计划对象数，失败对象保留错误记录。

## 本轮结果与分析

GSO 已完成 24/24 个对象的生成、几何评测与 384 张评测渲染，全部成功。原始记录位于 `data/generated/sf3d/paper_gso/run01/` 和 `results/raw/sf3d_paper_gso/run01/`；[对照表](../../results/metrics/sf3d_paper_public/comparison.md) 与 [重建图](../../results/metrics/sf3d_paper_public/gso_qualitative.png) 已导出。

| 实测数据 | CD↓ | FS@0.1↑ | FS@0.2↑ | FS@0.5↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO 公开子集（24） | 0.03273 | 0.94344 | 0.99749 | 1.00000 | 15.00895 | 0.77041 | 0.28108 |

RTX 5060 上平均图像到纹理 Mesh 为 0.660s，其中生成 0.612s、预处理 0.032s、导出 0.016s；模型加载 5.78s，单独记录。平均 25,629 个三角面，PyTorch 峰值分配范围为 6,127–6,167MiB。

几何得分较好的长方盒仍出现文字模糊与背面纹理猜测；`Wilton_PreCut...` 的 CD 为 0.00431，但较小表面误差没有保证贴图正确。`Frozen_Scrabble_Jr` 的 CD 为 0.08412、FS@0.1 为 0.48685，表现为重建厚度与比例偏差；卷尺的细带、茶具组与家具组的部件关系也是明显误差来源。对照图按 CD 排名 1、8、16、24 固定选择对象，展示输入方向与相反方向，选择记录另存 JSON。

本轮 F-score 在 0.2/0.5 阈值附近已饱和，区分效果主要看 CD、FS@0.1 和外观指标。公开子集的 CD 小于论文报告值，而图像指标较弱；对象组成、归一化尺度、条件视角和光照设置均影响数值，后续公平比较使用同一份数据与参数。几何对齐对近对称物体的纹理朝向存在歧义，图像指标结合对照图解释。

OmniObject3D 已完成 24/24 对象、384 个评测视角；两组 [合并报告](../../results/metrics/sf3d_paper_public/run01/comparison.md) 已生成。Omni 的 CD=0.04523、FS@0.1=0.95422、PSNR=9.12035dB、SSIM=0.73639、LPIPS=0.39717。进一步分析发现图像阶段尺度 ICP 明显缩小及参考材质基础色差异；具体证据、类别统计和后续处理顺序见 [run01 结果分析](sf3d_public_run01_analysis.md)。
