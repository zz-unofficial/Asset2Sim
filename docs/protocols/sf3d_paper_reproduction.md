# SF3D 论文实验复现

## 目标与参考值

使用官方预训练权重，在固定的 GSO、OmniObject3D 公开子集上测量几何、外观和图像到 Mesh 的成本。[SF3D 表 1 与附录 A1](https://arxiv.org/abs/2408.00653) 的参考值如下：

| 数据集 | 对象数 | CD↓ | FS@0.1↑ | FS@0.2↑ | FS@0.5↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | 278 | 0.098 | 0.701 | 0.894 | 0.988 | 21.247 | 0.865 | 0.124 |
| OmniObject3D | 308 | 0.090 | 0.726 | 0.920 | 0.990 | 20.134 | 0.851 | 0.132 |

论文报告 H100 上图像到最终纹理 Mesh 为 0.5s，27.4K 三角形对应一个示例。原始数字保存在 `experiments/references/sf3d_paper.json`。作者未公开具体的 278/308 对象清单，本项目各固定 24 个公开对象；模型比较复用同一数据和参数。

## 数据与坐标

对象列表位于 `experiments/datasets/`；GSO 按名称排序后以 seed=0 抽样，Omni 在下载类别中按路径排序后以同一种子抽样。Omni 实际包含书籍 16、腰包 6、电池 1、床 1，类别结果单列。

GSO 原始资产位于 `data/raw/gso/`；Omni 从 [作者 Google Drive raw_scans](https://drive.google.com/drive/folders/1ZpGmUlXYf0uc9xtN_w9iOhxmt9_gHZaP) 下载 `bed.tar.gz`、`bumbag.tar.gz`、`battery.tar.gz`、`book.tar.gz`，放到 `data/raw/omniobject3d/`。两组来源和许可写入清单，Omni 许可见 [作者仓库](https://github.com/omniobject3d/OmniObject3D)。

准备配置维护在原有 `configs/datasets/sf3d_*_public.yaml`：GSO 使用 Z-up，Omni OBJ 使用 Y-up，转换到右手 Z-up，保存源到参考的变换。Omni 的约定依据 [作者渲染脚本](https://github.com/omniobject3d/OmniObject3D/blob/main/tools/render_blender.py) 的默认 OBJ 导入及 [Blender 导入器](https://github.com/blender/blender-addons/blob/blender-v3.6-release/io_scene_obj/__init__.py) 的 Y-up 默认值。

参考网格居中，最长边缩放为 1。固定 512×512、16 个等间隔方位角、仰角 15°、距离 1.6、FOV 40°。输入选择轮廓面积最大的视角，平局取首个；这是几何正面近似。输入采用无光照贴图 RGB 和共享 RGBA 掩码。GSO 纹理从 `materials/textures/texture.png` 补全引用。

当前准备数据位于 `data/processed/gso/`、`data/processed/omniobject3d/`，共享清单仍为 `data/manifests/sf3d_gso_public.jsonl`、`sf3d_omni_public.jsonl`。`render.json` 保存相机、源向上轴、渲染方式、输入视角与尺度变换。原始准备数据和已完成运行保留，旧输入快照继续引用原位置。

## 模型和几何指标

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

## 图像指标与尺度控制

主评测使用几何刚体对齐并恢复参考坐标，相对尺度固定为 1，直接取消原来无约束的尺度 ICP。所有对象使用相同规则，不按分数选择对齐方式。参考和预测由同一渲染器、相机与设置重新渲染。

- **贴图 RGB 为主指标：** 无光照读取存储纹理 RGB，有贴图时基础色乘数固定为 1，忽略金属度、粗糙度和法线贴图。该项评价表面颜色，不将扫描贴图解释为真实 albedo。
- **原始材质 PBR 为辅助指标：** 保留资产材质，固定环境光 0.45、相机方向光 2.5；记录两侧基础色、金属度、粗糙度。该结果包含源材质转换与模型材质预测的共同影响。

白色背景合成，RGB 范围 [0,1]；SSIM 使用 sigma=1.5 的高斯窗口，LPIPS 使用 AlexNet v0.1；PSNR 完全一致时封顶 120dB。alpha>127 定义轮廓，计算 IoU、预测/参考前景面积比；另在两侧前景并集上计算 PSNR，结合 IoU 解释白色物体和缺失区域。

分别保存输入视角、其余 15 个新视角和全部 16 视角的指标。先按对象平均，再对对象等权汇总。每项指标保留有效分母，图像评测失败不丢弃已完成的几何结果。近对称物体仍可能在几何对齐后交换纹理正反面，结合固定排名案例图分析；不使用测试视角 RGB 搜索姿态。

已用旧生成资产中的 6 个固定案例进行尺度诊断：GSO 的瓶罐、茶具组、Frozen 盒，Omni 的 book_011、bumbag_002、bed_002。对照缩放总范围为 [0.9,1.1]，双向拟合，仅接受对称 CD 降低的步骤，最多 20 次；该诊断不替代固定尺度主结果。三个 GSO 案例的平均 IoU 从固定尺度的 0.707 变为有界尺度的 0.770，三个 Omni 案例从 0.772 变为 0.778。两种设置结果均保存，主评测统一固定尺度，避免按对象择优。实际渲染验证相同资产 SSIM=1、LPIPS=0、IoU=1，贴图模式不受基础色 0.4/1.0 差异影响。

## 运行与记录

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

已有 run01 的数值与问题定位见 [结果分析](sf3d_public_run01_analysis.md)。修正后的 run02 使用同一对象列表，但输入颜色和 Omni 朝向已调整，因此其变化反映整个数据与评测流程的修正，不能单独解释为模型性能提升。

## 修正后的实测结果

两组 run02 均完成 24/24 对象，主评测相对尺度全部为 1，每项主要指标的有效分母均为 24；两种图像口径各覆盖 768 个对象视角。完整数值与案例图见 [run02 报告](../../results/metrics/sf3d_paper_public/run02/comparison.md)。

| 数据 | CD↓ | FS@1%↑ | FS@2%↑ | FS@5%↑ | 轮廓 IoU↑ | 贴图 PSNR↑ | 贴图 LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | 0.03283 | 0.27176 | 0.47149 | 0.78450 | 0.77626 | 16.35562 | 0.30162 |
| OmniObject3D | 0.08403 | 0.13721 | 0.26053 | 0.44117 | 0.67200 | 11.69075 | 0.43085 |

严格阈值区分了原先宽阈值下难以观察的表面误差；Omni 的比例和厚度误差仍明显，统一朝向与颜色输入并未使其几何表现自动改善。固定尺度避免了通过大幅缩小资产来贴合参考表面的情况。后续模型沿用本配置、共享输入清单及报告字段进行比较。
