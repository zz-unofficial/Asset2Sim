# SPAR3D 实验准备与评测

SPAR3D 官方代码位于 `third_party/stable-point-aware-3d`，适配器与 SF3D 共用生成运行器和评测接口。模型先采样点云，再结合输入生成纹理网格；每个样例同时保存 `mesh.glb` 和 `points.ply`。

## 环境与权重

```bash
bash scripts/setup_spar3d.sh
conda activate asset2sim-spar3d
asset2sim fetch --model spar3d --config configs/models/spar3d.yaml
```

首次使用需要在 [官方模型页](https://huggingface.co/stabilityai/stable-point-aware-3d) 接受访问条件，并在本机登录 Hugging Face。项目缓存位于 `.local/huggingface/hub/` 和 `.local/alpha_clip/`。本机 8GB 显卡使用官方 `low_vram=true`；默认 BF16、纹理 1024、裁剪比 1.3、种子 0、无重网格化。官方权重配置使用 32 次点扩散采样，运行记录保存实际步数和引导参数。RGBA 输入直接使用已有掩码；普通 RGB 输入使用官方背景移除器。

## 运行与评测

小样例共用 `data/manifests/debug.jsonl`，两组正式对象与 SF3D 使用相同的 GSO、OmniObject3D 清单。运行目录使用新的 `run_id`：

两组数据依次生成、评测并与已有 SF3D 结果汇总，可直接执行 `bash scripts/run_spar3d_paper.sh run01`。运行状态和日志位于 `results/raw/spar3d_paper/run01_launcher/`。

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
