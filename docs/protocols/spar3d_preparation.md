# SPAR3D 实验准备与评测

SPAR3D 官方代码位于 `third_party/stable-point-aware-3d`，适配器与 SF3D 共用生成运行器和评测接口。模型先采样点云，再结合输入生成纹理网格；每个样例同时保存 `mesh.glb` 和 `points.ply`。

## 环境与权重

```bash
bash scripts/setup_spar3d.sh
conda activate asset2sim-spar3d
asset2sim fetch --model spar3d --config configs/models/spar3d.yaml
```

首次使用需要在 [官方模型页](https://huggingface.co/stabilityai/stable-point-aware-3d) 接受访问条件，并在本机登录 Hugging Face。项目缓存位于 `.local/huggingface/hub/` 和 `.local/alpha_clip/`。本机 8GB 显卡使用官方 `low_vram=true`；默认 BF16、纹理 1024、裁剪比 1.3、种子 0、无重网格化。官方权重配置使用 32 次点扩散采样，运行记录保存实际步数和引导参数。RGBA 输入直接使用已有掩码；普通 RGB 输入使用官方背景移除器。

批量推理在对象之间释放低显存模式的阶段模块，并启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。AlphaCLIP 资源在进程内复用首次成功的缓存定位，避免逐对象重复读取并校验大文件。RTX 5060 的连续 6 对象验证全部成功，PyTorch 峰值分配约 6,440–6,494 MiB，平均推理约 10.84 秒；早期 `run01` 记录了 1/24 成功及显存不足问题，保留用于定位。

## 运行与评测

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

Omni 使用 `configs/experiments/spar3d_omni_public.yaml`，输出路径对应改为 `paper_omni`。评测沿用 [SF3D 统一协议](sf3d_paper_reproduction.md)：CD、F-score、距离分位数、轮廓 IoU、贴图 RGB 和 PBR 图像指标，分别记录输入与新视角。

```bash
asset2sim paper-report results/raw/sf3d_paper_gso/run02 \
  results/raw/spar3d_paper_gso/run01 --output results/metrics/gso_model_comparison
```

报告列出模型名并核对对象 ID 和评测参数；各模型按相同对象配对。生成记录包含种子、低显存设置、耗时、面数、峰值分配显存及点云路径。

## 论文参考

[SPAR3D 论文](https://arxiv.org/html/2501.04689v1) 表 1/2：GSO 的 CD 0.120、FS@0.1 0.584、PSNR 18.6、SSIM 0.836、LPIPS 0.139；Omni 的 CD 0.122、FS@0.1 0.587、PSNR 17.9、SSIM 0.832、LPIPS 0.140。论文每组约 250 对象，移除简单盒体/圆柱，并使用多仰角、随机 HDRI 和不同焦距。本项目首轮沿用固定 24 对象与统一渲染，适合模型间比较。

论文中 SF3D 和 SPAR3D 的完整参考行记录于 `experiments/references/spar3d_paper.json`。这些数字采用 SPAR3D 论文的实验设置，与 SF3D 论文自己的表格分别记录。

## GSO 实测与运行情况

SPAR3D 的 `run02` 已完成 GSO、Omni 各 24/24 个对象的生成和各 384 个视角的评测。原流水线在生成 GSO 案例图时因文件名错误退出，错误已修复；已有指标重新汇总到 [GSO 对照报告](../../results/metrics/spar3d_gso_comparison/run03/comparison.md)。Omni 随后单独续跑并正常完成，结果见 [Omni 对照报告](../../results/metrics/spar3d_omni_comparison/run02/comparison.md)。

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

两模型均在 RTX 5060 上运行；SPAR3D 使用低显存模式及阶段模块加载。Mesh 时间包含预处理、推理和 Mesh 导出，点云导出另行计时。均值不能直接与论文使用的硬件和流程比较。

## Omni 实测与类别分析

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

两组 GSO 和 Omni 各使用同一对象与评测配置、各一个推理种子。结果支持本机固定子集下的比较；扩大类别覆盖和重复随机种子后才能进一步判断稳定性。资产的碰撞与仿真表现尚待后续实验。
