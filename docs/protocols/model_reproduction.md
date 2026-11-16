# SF3D、SPAR3D 与 TRELLIS.2 模型复现和统一评测

## 三模型单图生成基准：实验配置与运行

比较 SF3D、SPAR3D、TRELLIS.2 官方预训练单图生成的几何、外观、拓扑、视角敏感性和成本。D1 使用 GSO 的 32 个对象，其中 8 个增加第二输入视角，共 40 个单图输入、120 次三模型生成。D1-PBR 使用 ABO 的 6 个原始 GLB，增加 18 次生成。两套真实数据分别统计，共 138 次正式生成；D2 的 8 个程序化拓扑/PBR 参考为独立补充，再增加 24 次。预热及计时重复另计。D0 的 4 个开发对象用于验证评测器。

### 服务器环境

本轮系统盘 30GB、数据盘 150GB 的安装、缓存定位和本地权重上传步骤见 [服务器操作手册](../../AAA.md)，模型环境与所有大型文件放数据盘。

从同步后的仓库根目录执行。服务器需要 Conda、CUDA 开发工具、C/C++ 编译器和 Blender 4.2 LTS；正式比较通过 `generation_benchmark_gpu.yaml` 启用 Cycles CUDA 渲染。三个模型各用独立环境，数据准备、评测和批量调度复用 `asset2sim-sf3d`。

```bash
nvidia-smi
nvcc --version
blender --version
bash scripts/setup_sf3d.sh
bash scripts/setup_spar3d.sh
bash scripts/setup_trellis2.sh
conda run --no-capture-output -n asset2sim-sf3d huggingface-cli login
```

租卡前用同一 Hugging Face 账号开通 [SF3D](https://huggingface.co/stabilityai/stable-fast-3d)、[SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d)、[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 和 [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) 的权重访问。后两项由 TRELLIS.2 官方管线在加载阶段初始化，RGBA 输入也会触发加载。服务器终端登录该账号后，模型和辅助权重按需自动下载。

TRELLIS.2 环境定义在 `environments/trellis2/`，使用 PyTorch 2.6/CUDA 12.4，扩展安装调用固定子模块的官方脚本；已创建环境也会补装依赖。服务器优先选择 A100 80GB，pilot 记录实测显存后确定正式预算。首次安装和首次下载所用时间单独记入部署记录。系统另需 `build-essential`、`git`、`curl`、`wget`、`xz-utils`、`libegl1`、`libgl1`、`libxrender1`、`libxi6`、`libxkbcommon0` 和 `libjpeg-dev`。

Blender 可固定使用官方 4.2.0 Linux x64 包：

```bash
mkdir -p "$HOME/tools" "$HOME/downloads"
wget -c https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz \
  -O "$HOME/downloads/blender-4.2.0-linux-x64.tar.xz"
tar -xf "$HOME/downloads/blender-4.2.0-linux-x64.tar.xz" -C "$HOME/tools"
export PATH="$HOME/tools/blender-4.2.0-linux-x64:$PATH"
blender --version
```

### D0：渲染与指标自检

```bash
conda run -n asset2sim-sf3d python -m asset2sim.data.benchmark_synthetic --debug \
  --output data/raw/benchmark_d0 --selection experiments/datasets/benchmark_d0.jsonl
conda run -n asset2sim-sf3d python -m asset2sim.data.generation_benchmark \
  --selection experiments/datasets/benchmark_d0.jsonl \
  --output data/processed/benchmark_d0 --manifest data/manifests/benchmark_d0.jsonl
conda run --no-capture-output -n asset2sim-sf3d python -m asset2sim.evaluation.benchmark_identity \
  --manifest data/manifests/benchmark_d0.jsonl --output results/raw/benchmark_identity/run01
```

identity 将参考 Mesh 当作预测，经同一导出、渲染和指标流程，检查 CD 的采样容差、IoU/SSIM 接近 1、LPIPS 接近 0，PSNR 上限为 120dB。`validation.json` 保存判定及分母。D0 不计入正式比较。

### D1：GSO 32 对象数据准备

[GSO 官方介绍](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)提供带纹理的真实物体扫描，采用 CC BY 4.0。对象、选择理由、输入条件和官方单对象 ZIP 链接固定在 `experiments/datasets/generation_gso32.jsonl`。32 个 v1 下载链接已通过四字节范围请求确认可获取，总计约 194.88MiB。

| 分组 | 四个代表对象 | 观察重点 |
| --- | --- | --- |
| 包装与瓶罐 | Frozen 盒、Weston 瓶、Bifidus 包装、Crayola 盒 | 规则几何、标签与正反面纹理 |
| 容器 | 蓝色马克杯、深碗、烤盅、白色水壶 | 口沿、内壁、把手与壶嘴 |
| 厨具与花盆 | 煎锅、Bundt 模具、方盘、花盆 | 浅薄表面、环形凹腔与长柄 |
| 工具 | C 形夹、剪刀、锤子、螺丝刀 | 细杆、凹口、开口与非对称形体 |
| 家电与电子产品 | 手电筒、鼠标、键盘、咖啡机 | 轴对称体、密集按键与复合结构 |
| 鞋与软质物品 | 船鞋、背包、毛巾、靴子 | 鞋口、背带、褶皱与织物纹理 |
| 开口与薄结构 | 提篮、三格收纳篮、文件架、锅铲 | 隔板、狭缝、薄片与长柄 |
| 玩具 | 人偶、三角龙、蜜蜂、火车 | 多部件、遮挡、四肢和重复结构 |

杯、夹具、剪刀、鞋、水壶、Bundt 模具、文件架和人偶预选第二输入视角。主输入仍按参考轮廓面积最大规则选择，同面积时取最小编号；第二输入相隔 180°。输入选择只使用参考渲染。

服务器从仓库根目录执行，自动下载或复用 `data/raw/gso/` 缓存，恢复 OBJ/MTL/PNG，归一化并生成 16 个视角：

```bash
conda run --no-capture-output -n asset2sim-sf3d python -m asset2sim.data.generation_benchmark \
  --output data/processed/gso32 --manifest data/manifests/generation_gso32.jsonl
```

查看 `data/processed/gso32/reference_gallery.png` 与 `audit.json`，核对 32 个参考、40 个输入及结构可见性。源扫描的具体孔洞和薄壁状态由几何报告及参考图确认。随后冻结、试跑和正式生成：

```bash
bash scripts/run_generation_benchmark.sh freeze gso32
bash scripts/run_generation_benchmark.sh pilot gso32_pilot01
bash scripts/run_generation_benchmark.sh evaluate gso32_pilot01
bash scripts/run_generation_benchmark.sh formal gso32_run01 \
  --freeze experiments/datasets/gso32_frozen.json
bash scripts/run_generation_benchmark.sh evaluate gso32_run01
```

pilot 默认用盒、杯、剪刀各一个输入；末尾可指定 sample_id。新输出使用独立 run_id。下载、纹理、几何和渲染异常写入 `audit.json`，参考失败另表保留。已有八对象实验与缓存保留原位。GSO 基础色作为主外观结果，PBR 资格为 false。

### D1-PBR：ABO 六对象补充集

使用 [ABO 官方发布](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)的原始 GLB，保留原始高分辨率 PBR 贴图。`ABO-GLB` 浏览器派生包采用 WebP 和 meshopt 压缩；本协议通过官方 S3 按文件下载原件，沿用现有 Trimesh/Blender 接口。许可为 CC BY 4.0，数据署名为 Amazon.com。

选择清单为 `experiments/datasets/generation_abo6.jsonl`，共约 119.01MiB。六个文件的官方模型路径、GLB JSON 材质块及基础色/金属粗糙度纹理引用已经核验；完整网格和贴图值由服务器准备过程读取。

| ID | ABO model ID | 对象与材质先验 | 比较价值 |
| --- | --- | --- | --- |
| `abo_plastic` | `B000S6N026` | 黑色碎纸机，塑料外壳 | 低纹理表面与反射响应 |
| `abo_wood` | `B003DQ5B2U` | 松木套桌 | 木纹与多平面结构 |
| `abo_leather` | `B00IFHPVEU` | 皮革办公椅 | 软包与皮革表面 |
| `abo_fabric` | `B016ID34BO` | 织物办公椅 | 细纹理及粗糙表面 |
| `abo_metal` | `B072Y2MRY2` | 铬色支撑凳 | 金属与坐垫的混合材质 |
| `abo_ceramic` | `B075HR4ZDB` | 几何纹理陶瓷花瓶 | 曲面与粗糙度变化 |

材质组名来自商品描述，作为先验标签。GLB metallic-roughness 通道与参数另行记录。源资产为 Y-up，准备阶段显式转为 Z-up，再将最长边归一到 1。原始 GLB 保存在 `data/raw/abo/`，保持原样。天然正面 +Z 转换后对应参考视角 12；每个对象只生成一个输入。ABO 输入和 PBR 评测使用同一个 Blender/Cycles worker、固定 HDRI 和曝光；源材质缺少必需贴图时记录准备失败。

```bash
conda run --no-capture-output -n asset2sim-sf3d python -m asset2sim.data.generation_benchmark \
  --selection experiments/datasets/generation_abo6.jsonl \
  --output data/processed/abo6 --manifest data/manifests/generation_abo6.jsonl
bash scripts/run_generation_benchmark.sh freeze abo6 --manifest data/manifests/generation_abo6.jsonl
bash scripts/run_generation_benchmark.sh pilot abo6_pilot01 --manifest data/manifests/generation_abo6.jsonl
bash scripts/run_generation_benchmark.sh evaluate abo6_pilot01
bash scripts/run_generation_benchmark.sh formal abo6_run01 \
  --manifest data/manifests/generation_abo6.jsonl --freeze experiments/datasets/abo6_frozen.json
bash scripts/run_generation_benchmark.sh evaluate abo6_run01
```

ABO pilot 默认选木桌、金属凳和陶瓷瓶。准备前确认 Blender 与配置中的 HDRI 可用，使用 `--render-config` 可指定同类渲染配置。GSO32 与 ABO6 的图表和有效分母分别报告；ABO 同时展示 flat 与 PBR 多视图指标，结合几何误差解释材质重渲染差异。ABO 属于 TRELLIS.2 数据工具列出的公开训练来源之一，这组结果用于指定资产上的重建与材质表现分析。

每个数据集默认用一个共享输入执行一次预热和三次有效计时，参数为 `--warm-repeats 3`。这些输出只进入计时表，质量每个输入计一次。时间单位为秒、显存为 MiB，记录 allocated/reserved。SPAR3D 服务器覆盖 `low_vram=false`。主种子为 0；随机性扩展先冻结独立对象子集，再用新 run_id 和种子 1/2 执行。

### D2：受控拓扑实验

```bash
conda run -n asset2sim-sf3d python -m asset2sim.data.benchmark_synthetic \
  --output data/raw/benchmark_d2 --selection experiments/datasets/benchmark_d2.jsonl
conda run -n asset2sim-sf3d python -m asset2sim.data.generation_benchmark \
  --selection experiments/datasets/benchmark_d2.jsonl \
  --output data/processed/benchmark_d2 --manifest data/manifests/benchmark_d2.jsonl
bash scripts/run_generation_benchmark.sh freeze topology --manifest data/manifests/benchmark_d2.jsonl
bash scripts/run_generation_benchmark.sh formal topology01 \
  --manifest data/manifests/benchmark_d2.jsonl --freeze experiments/datasets/topology_frozen.json
bash scripts/run_generation_benchmark.sh evaluate topology01
```

规格在 `experiments/datasets/controlled_topology_specs_8.jsonl`，采用提供的 P01–P08：曲面开放片、三孔板、杆件框架、空心杯、嵌套壳、薄鳍、波纹布条及窄缝块。程序化实现使用 Trimesh/Shapely，并记录实际生成方式；基础色、金属度及粗糙度采用确定性纹理。三孔板与窄缝块采用高仰角输入，布条使用双面材质。嵌套壳的内部表面封闭不可见，元数据单独标记隐藏结构。D2 用来解释受控形状差异。

### 评测口径与结果文件

`configs/evaluation/generation_benchmark.yaml` 固定采样、旋转搜索、ICP、相机导出映射、渲染样本数与 HDRI。运行模块位于 `experiments/benchmark.py`，指标及统计位于 `evaluation/benchmark.py`，Blender worker 位于 `evaluation/benchmark_render.py`；Shell 脚本只负责进入环境与调用模块。

- `export_pose`：各自归一到单位最长边，按固定模型导出矩阵和输入相机旋转评价。变换不拟合参考形状，代表约定坐标下的归一化资产表现；配置中的映射在 pilot 检查后冻结。
- `aligned_diagnostic`：同一旋转搜索和刚体 ICP，只使用几何；尺度在归一化后固定。变换完整保存。
- 几何：两侧按面积各采样 20,000 点，CD-L1 是双向平均欧氏距离的均值，FS 阈值为最长边 1/2/5%，HD95 是两个方向 95% 距离分位数的较大者。采样种子固定。
- 图像：同一 Blender/Cycles 渲染器、512 RGBA、固定曝光与色彩管理。flat 模式使用基础色节点的无光照输出；PBR 使用合格参考子集和配置中的固定 HDRI。分别报告输入/新视角、全图、前景并集 PSNR、并集包围框裁剪 SSIM/LPIPS；裁剪指标名称带 `union_crop`。HDRI 相对仓库根目录解析，用户可替换配置后重新冻结。
- 拓扑：原始及仅按位置合并的诊断分别记录顶点/面、连通分量、边界边、非流形边、水密性、绕序和 Euler 数。位置合并只作用于副本，用来识别 UV seam。genus 留空并注明未验证顶点流形。TRELLIS.2 导出前的 `native_geometry.npz` 与官方重网格 GLB 分别统计。D2 另报告投影孔洞 IoU/recall；表面 recall 仍按距离阈值解释。
- 统计：视角先在对象内平均，再给对象等权中位数；主配对表使用该项全部计划视角有效的共同对象，报告配对数及固定种子的 2,000 次对象 bootstrap 区间。`robustness.csv` 保存同对象两个输入条件的差值。生成、几何、渲染和指标失败均有状态，参考审核排除另表保留。

每轮生成保存在 `data/generated/<model>/generation_core/<run_id>/`；计划、日志和评测在 `results/raw/generation_benchmark/<run_id>/`。重点查看 `evaluation/tables.md`、`summary.json`、`per_sample.csv`、`per_object.csv`、`per_view.csv`、`paired.csv`、`groups.csv`、`failures.csv` 和 `qualitative.png`。原始 Mesh、纹理、渲染图、环境日志由 Git 忽略；正式结论归入本文。

### TRELLIS.2 服务器分阶段运行

GPU 生成阶段使用同一输入、种子和 512 流水线，在实验 `overrides` 中设置
`defer_export: true`，运行 `asset2sim run --config <实验配置> --output <新目录>`，
重复计时设为 `--warm-repeats 0`。逐样例保存 `native_asset.pt` 和
`native_asset.json`，包含顶点、三角面、材质体素、体素坐标、尺寸、属性布局和参数。
该阶段记录 `export_pending`；模型加载一次后连续处理清单。

生成完成后在服务器独立导出，导出过程直接读取中间资产：

```bash
python -m asset2sim.generation.trellis2_artifacts \
  --run <中间资产运行目录> --output <新GLB运行目录> \
  --config configs/evaluation/trellis2_practical_export.yaml
```

`trellis2_practical_export.yaml` 定义 10 万面、1024 贴图和关闭重网格的实用导出条件。
原始生成数据保留，派生 GLB 单独记录导出参数和耗时。省略 `--config` 时沿用
中间资产记录的原导出参数。两种导出条件分别汇总几何和外观指标。

最终表的 `generation_*` 指标比较生成几何：TRELLIS.2 读取保存的原始网格，
SF3D/SPAR3D 读取模型原始 GLB。`chamfer_l1` 等原字段评价实际导出的 GLB；
`export_vs_native_*` 量化导出引入的形状变化。原始和导出几何各自保存归一化与对齐矩阵。
TRELLIS.2 官方导出的轴变换为 `(x,y,z)→(x,z,-y)`，部署坐标评测显式使用其逆变换。

评测配置 `generation_benchmark_gpu.yaml` 使用 Blender CUDA 渲染；渲染记录保存实际设备。
渲染器在 16 个视角之间保留场景缓存，共享同一样例的参考渲染。
配置将 Blender 和图像指标各限制为 3 个 CPU 线程。GSO 与 ABO 可在独立进程中
同时评测，分别指定清单和输出目录；生成模型的推理耗时沿用原运行记录。
同一配置、清单路径和生成运行的完整评测可直接复用；输出尚未完成或参数变化时使用新目录。

2026-11-15 在 A100 40GB、9 核 CPU 配额实例上启动 GSO/ABO 两个并行评测。
用 `gso_box_v004` 的同一预测 Mesh 核对缓存设置：16 张图的渲染日志累计时间
从 29.49 秒变为 27.03 秒，新测量时还有两个正式任务并行运行。
RGBA 图像最大差异为 1/255，平均差异为 0.000012 个 8 位灰度级。
该单例用于检查渲染结果与开销，整体吞吐量由完整评测耗时统计。

对象分片评测使用同一配置，由两个进程处理互不重叠的对象：

```bash
python -m asset2sim.evaluation.benchmark_parallel \
  --manifest <完整清单> --run trellis2=<生成目录> \
  --config configs/evaluation/generation_benchmark_gpu.yaml \
  --checkpoint <已暂停的评测目录> --workers 2 --output <新目录>
```

`--checkpoint` 可省略。复用前检查配置、生成运行与输入记录一致性；完整记录
（含失败）直接沿用，未完成的轨道重新计算，原目录保留。
同一对象的所有输入视角归入同一分片。每个分片保存日志和结果，全部结束后在
`combined/` 汇总原有指标、对象分母及案例图；缺失或重复的评测键会阻止合并。
合并目录通过链接引用分片资产，回传时一起保留原检查点和分片，或解引用链接。
2026-11-15 的 GSO 三路调度保留了切换前的 14 条完整记录，另一路为 ABO 评测。

运行成本区分模型加载、GPU 推理、中间数据保存和 GLB 导出。
取消重复计时的运行以逐样例时间和有效样本数汇总；重复计时统计单独使用已有记录。
`generation_costs.csv` 保留每个计划输入的状态和分阶段耗时；`coverage.csv` 按模型和
对齐方式列出计划对象/输入数、生成、导出、几何和图像指标的有效输入数。
`geometry_progress.jsonl` 和 `view_progress.jsonl` 逐条保存运行中的指标，完成后生成
对象汇总、配对差值、bootstrap 区间、视角鲁棒性表和案例图。

### 当前验证记录

#### 2026-11-16 本机续评

服务器数据保存在 `.local/server-backups/cfgpu-20261115/`（8.5GB），包含
三模型的 46 个生成资产、TRELLIS.2 原始中间资产、GSO/ABO 参考与评测记录。
本机复用 `asset2sim-sf3d` 环境；可携版 Blender 4.2.0 位于
`.local/tools/blender-4.2.0-linux-x64/`。渲染配置为
`configs/evaluation/generation_benchmark_local.yaml`，保持服务器的 16 视角、32 采样设置。

本地清单和检查点位于 `.local/local-evaluation-20261116/`，路径已适配到备份目录。
检查点保留来源协议，沿用 GSO 64 条、ABO 19 条完整记录，剩余分别为 96、5 条。
执行入口先完成 ABO，再将 GSO 按对象分为两路：

```bash
bash .local/local-evaluation-20261116/run.sh
```

输出为 `results/raw/generation_benchmark/abo6_local_20261116a/` 与
`results/raw/generation_benchmark/gso32_local_20261116a/combined/`。
已有结果保留原位，重跑时更换脚本中的输出目录。跨设备续评沿用服务器的生成耗时，
本机后处理耗时按本机硬件解释。

## 三模型单图生成基准：实验报告（2026-11-16）

### 摘要与完成范围

本实验在 GSO32（32 对象、40 输入）及 ABO6（6 对象、6 输入）上比较 SF3D、SPAR3D、TRELLIS.2 官方预训练模型，主比较覆盖 138 个生成资产、276 条双轨道记录。SF3D/SPAR3D 的八种 D2 结构及 SPAR3D 八对象三种子补充已完成，合计 178 个有效产物（含 8 个 seed=0 匹配重复对照）、记录 356 条、逐视角指标 6784 条。修正后的记录均完成几何及图像指标。统计单位是对象，轨道、视角和种子作为同一对象内的条件。

| 项目 | 完成情况 |
| --- | --- |
| 三模型 GSO32/ABO6 | 完成，逐输入生成与逐视角评测齐全 |
| 类别与第二输入视角 | 完成，每组 4 对象、8 对象具有第二输入 |
| SF3D/SPAR3D D2 | 完成，8 对象×2 模型 |
| SPAR3D seed=0/1/2 | 完成，8 对象×3 种子 |
| TRELLIS.2 D2、多种子、1024³ | 未执行，保留为扩展范围 |
| TRELLIS.2 官方默认导出完整对照 | 未执行，当前采用实用导出 |

本节为复核后的结果，替换复核前的质量表和外观排名。旧运行记录留作来源追踪。本报告聚焦单图生成；物理适配、编辑、抓取等实验见各自协议。

### 研究问题与配置

研究问题包括形状与外观保真、可用结构、输入视角及随机性的影响、坐标规范对指标的影响与资产处理成本。GSO 覆盖八类各四对象，8 对象增加第二输入；ABO 的六个对象覆盖塑料、木纹、皮革、织物、金属及陶瓷先验。对象、来源、许可、相机及准备命令见前文配置。ABO 是 TRELLIS.2 已知训练来源之一，具体对象及其他模型的训练重叠未知，结果解释为固定子集比较。

模型共享 512×512 RGBA 输入和掩码，沿用各自预处理。SF3D 前景比例 0.85，SPAR3D 1.3，两者为 BF16、1024 纹理、无重网格。本机复核使用 low_vram=true 的 SPAR3D，并显式关闭两个模型的 autocast 权重转换缓存；权重版本和上游代码与此前运行一致。TRELLIS.2 为官方 512³、默认 low_vram 模式，保留原生中间资产；实际外观 GLB 使用 remesh=false、100000 面目标、1024 纹理。

SF3D/SPAR3D 的有效质量资产在 RTX 5060 上重生成；TRELLIS.2 复用 A100 PCIe 40GB 上的既有生成。本轮没有新增 TRELLIS.2 推理。两基线为 PyTorch 2.8.0/CUDA 12.8，TRELLIS.2 为 2.6.0/CUDA 12.4。评测统一 Blender 4.2.0/Cycles CUDA、512×512、32 samples、16 相机；固定参考渲染复用原记录。

### 复核发现与处理

1. **条件相机基准。** SF3D/SPAR3D 的官方 `utils.default_cond_c2w` 是位于 +X 的水平 OpenGL 相机。原评测将参考常用仰角 20° 当作条件相机基准，造成额外旋转。现使用官方 0° 基准，再依据每个输入的参考相机旋转；受影响 export_pose 已重算。独立旋转搜索的 aligned_diagnostic 不依赖该基准。
2. **部署归一化顺序。** export_pose 先应用声明的相机/导出轴变换，再按预测自身在该坐标系中的 AABB 统一最长边和中心。变换矩阵保存为 `rotation @ prediction_normalization`，全程没有使用参考尺寸拟合。aligned_diagnostic 保留预声明的原生 AABB 归一化，另给出姿态后自身归一化的 Oracle 敏感性分析。
3. **原始贴图异常。** 审查发现旧服务器基线 GLB 中存在条纹状原始贴图。相同权重、相同 Frozen 盒体输入的本机 SF3D 对照恢复正常图案；SPAR3D 杯子对照亦显示差异。问题属于原资产生成/烘焙链，旧外观排名已撤回。新主表使用本机重生成的全部 92 个基线资产。由于硬件、低显存与缓存条件有差异，尚未将原因隔离到某个具体 CUDA 内核；这项底层归因作为未验证内容保留。
4. **TRELLIS.2 姿态。** 官方推理接口未提供每张输入的精确条件相机。当前 20° 约定保留为明确的姿态假设，export_pose 标记 `assumed_condition_camera`，作为该假设下的诊断；SF3D/SPAR3D 使用可核验的官方相机基准。三模型的主要形状比较使用单列的对齐诊断轨道。

![同权重同输入的源贴图核查](../../results/figures/source_texture_review_20261116.png)

### 指标与统计

两侧每个 Mesh 面积采样 20000 点，种子固定。CD 为两个方向最近距离均值的平均，FS 使用最长边 1%/2%/5% 阈值，HD95 为两方向 95% 距离分位数的较大者。对齐诊断采用统一旋转搜索和刚体 ICP，不执行参考尺度 ICP。flat 图像为无光照基础色，ABO 增加固定 HDRI 下的 PBR 重渲染。输入与 15 个新视角分开统计，前景指标及裁剪指标保存在原始表。

每对象先平均输入条件，再报告对象等权中位数；配对差值为对象 A−B 的中位数，使用排序后的对象 ID、2000 次 bootstrap、seed=0。下述区间为未作多重比较校正的探索性区间。图像成功分母与结构准确性分别记录。

### 主质量结果

下表为 aligned_diagnostic 的对象中位数。生成几何使用 SF3D/SPAR3D 原始 GLB 与 TRELLIS.2 原生几何；外观使用实际导出资产。

| 数据 | 模型 | 对象数 | 生成 CD ↓ | FS@2% ↑ | HD95 ↓ | 新视角 LPIPS ↓ | 新视角 IoU ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | SF3D | 32/32 | 0.03068 | 0.45916 | 0.10139 | 0.10685 | 0.79694 |
| GSO | SPAR3D | 32/32 | 0.02228 | 0.57544 | 0.07530 | 0.13622 | 0.78611 |
| GSO | TRELLIS.2 | 32/32 | 0.01542 | 0.72131 | 0.03523 | 0.07779 | 0.88529 |
| ABO | SF3D | 6/6 | 0.02205 | 0.57931 | 0.08390 | 0.16611 | 0.62223 |
| ABO | SPAR3D | 6/6 | 0.02599 | 0.54065 | 0.09681 | 0.17250 | 0.60973 |
| ABO | TRELLIS.2 | 6/6 | 0.01363 | 0.80753 | 0.03201 | 0.08357 | 0.80597 |

| ABO PBR | 对象数 | LPIPS ↓ | PSNR/dB ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- |
| SF3D | 6/6 | 0.14485 | 15.99045 | 0.87529 |
| SPAR3D | 6/6 | 0.15473 | 14.22816 | 0.85285 |
| TRELLIS.2 | 6/6 | 0.07439 | 19.34187 | 0.88867 |

| Flat RGB | 模型 | 对象数 | PSNR/dB ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- |
| GSO | SF3D | 32/32 | 20.64059 | 0.92580 |
| GSO | SPAR3D | 32/32 | 19.97513 | 0.89509 |
| GSO | TRELLIS.2 | 32/32 | 21.64812 | 0.93357 |
| ABO | SF3D | 6/6 | 14.82464 | 0.85408 |
| ABO | SPAR3D | 6/6 | 14.60942 | 0.84570 |
| ABO | TRELLIS.2 | 6/6 | 18.43260 | 0.87942 |

全图与前景指标分开汇总。裁剪 LPIPS 仅纳入全部计划新视角都有有效裁剪的对象，按对象等权；前景 PSNR 使用两侧掩码并集。

| 数据 | 模型 | 全图 LPIPS | 裁剪 LPIPS | 完整对象 | 前景 PSNR/dB | 完整对象 |
| --- | --- | --- | --- | --- | --- | --- |
| GSO | SF3D | 0.10685 | 0.37699 | 32/32 | 10.10822 | 32/32 |
| GSO | SPAR3D | 0.13622 | 0.42263 | 32/32 | 9.29788 | 32/32 |
| GSO | TRELLIS.2 | 0.07779 | 0.28351 | 32/32 | 11.84358 | 32/32 |
| ABO | SF3D | 0.16611 | 0.44423 | 6/6 | 7.20997 | 6/6 |
| ABO | SPAR3D | 0.17250 | 0.47333 | 6/6 | 5.16759 | 6/6 |
| ABO | TRELLIS.2 | 0.08357 | 0.27329 | 6/6 | 9.22368 | 6/6 |

### 三模型配对证据

差值 A−B；CD、LPIPS 越低越好，正差值倾向 B。配对差值中位数与总体中位数之差分别解释。

| 数据 | A−B | 指标 | 对象数 | 差值中位数 | 95% 区间 |
| --- | --- | --- | --- | --- | --- |
| GSO | sf3d-spar3d | generation_chamfer_l1 | 32 | 0.00392 | [-0.00168, 0.00725] |
| GSO | sf3d-spar3d | texture_novel_lpips | 32 | -0.01189 | [-0.01585, -0.00314] |
| GSO | sf3d-trellis2 | generation_chamfer_l1 | 32 | 0.01407 | [0.00523, 0.02447] |
| GSO | sf3d-trellis2 | texture_novel_lpips | 32 | 0.02449 | [0.00691, 0.04764] |
| GSO | spar3d-trellis2 | generation_chamfer_l1 | 32 | 0.00761 | [0.00057, 0.01636] |
| GSO | spar3d-trellis2 | texture_novel_lpips | 32 | 0.03509 | [0.01800, 0.06448] |
| ABO | sf3d-spar3d | generation_chamfer_l1 | 6 | 0.00036 | [-0.00869, 0.01058] |
| ABO | sf3d-spar3d | texture_novel_lpips | 6 | -0.00957 | [-0.03009, 0.01109] |
| ABO | sf3d-spar3d | pbr_novel_lpips | 6 | -0.00811 | [-0.02148, 0.01206] |
| ABO | sf3d-trellis2 | generation_chamfer_l1 | 6 | 0.00829 | [-0.00258, 0.01238] |
| ABO | sf3d-trellis2 | texture_novel_lpips | 6 | 0.04994 | [0.00056, 0.09124] |
| ABO | sf3d-trellis2 | pbr_novel_lpips | 6 | 0.04750 | [-0.01280, 0.07826] |
| ABO | spar3d-trellis2 | generation_chamfer_l1 | 6 | 0.00993 | [-0.00817, 0.01408] |
| ABO | spar3d-trellis2 | texture_novel_lpips | 6 | 0.06444 | [0.00397, 0.10190] |
| ABO | spar3d-trellis2 | pbr_novel_lpips | 6 | 0.05603 | [-0.01326, 0.08772] |

GSO 上，TRELLIS.2 相对两个基线的 CD 与贴图 LPIPS 区间均位于优势方向。SF3D 与 SPAR3D 的 CD 差异仍有不确定性；SF3D 的贴图 LPIPS 更低，配对区间支持这一差异。

ABO 上，TRELLIS.2 的贴图 LPIPS 有优势方向的区间；几何 CD 和 PBR LPIPS 的区间跨零。SF3D 与 SPAR3D 的上述差异区间也跨零，这六个对象适合作为材料条件的初步比较。

### 已声明或假设姿态的结果

SF3D/SPAR3D 使用官方水平条件相机，TRELLIS.2 使用标记的 20° 假设。以下数值描述各自已声明条件，TRELLIS.2 的绝对部署姿态准确性仍待独立相机证据。

| 数据 | 模型 | 对象数 | 声明/假设姿态 CD | 对齐 CD | 声明/假设姿态 IoU | 对齐 IoU |
| --- | --- | --- | --- | --- | --- | --- |
| GSO | SF3D | 32/32 | 0.03673 | 0.03068 | 0.78452 | 0.79694 |
| GSO | SPAR3D | 32/32 | 0.02947 | 0.02228 | 0.85114 | 0.78611 |
| GSO | TRELLIS.2 | 32/32 | 0.08616 | 0.01542 | 0.48808 | 0.88529 |
| ABO | SF3D | 6/6 | 0.01914 | 0.02205 | 0.77555 | 0.62223 |
| ABO | SPAR3D | 6/6 | 0.02626 | 0.02599 | 0.65973 | 0.60973 |
| ABO | TRELLIS.2 | 6/6 | 0.05867 | 0.01363 | 0.48953 | 0.80597 |

木桌案例展示了配准条件的影响：SF3D 几何拟合相对声明姿态旋转约 178.75°，输入 LPIPS 从 0.14923 变为 0.26342。两轨道同时包含位姿与自身归一化条件差异，图像分数反映整体重建相似性；位姿选择始终依据几何，RGB 用于事后观测。

### 类别和第二输入视角

GSO 每类别 4 对象；下表为对齐后生成 CD 中位数。

| 类别 | 每模型对象数 | SF3D | SPAR3D | TRELLIS.2 |
| --- | --- | --- | --- | --- |
| cookware | 4/4 | 0.04971 | 0.03424 | 0.01169 |
| electronics | 4/4 | 0.02942 | 0.02901 | 0.01950 |
| open_structure | 4/4 | 0.05205 | 0.05283 | 0.01774 |
| packaging | 4/4 | 0.02041 | 0.02096 | 0.01674 |
| soft_goods | 4/4 | 0.04140 | 0.01820 | 0.01680 |
| tools | 4/4 | 0.01129 | 0.01422 | 0.01779 |
| toys | 4/4 | 0.01967 | 0.02001 | 0.02879 |
| vessels | 4/4 | 0.03068 | 0.02276 | 0.01201 |

第二输入视角相隔 180°。先计算每对象两输入的绝对差，再取 8 对象中位数；该表描述视角敏感性。CD 使用导出 GLB 字段，与 TRELLIS.2 原生 CD 分开解释。

| 模型 | 配对对象 | CD 绝对变化 | LPIPS 绝对变化 |
| --- | --- | --- | --- |
| SF3D | 8/8 | 0.00352 | 0.01501 |
| SPAR3D | 8/8 | 0.00181 | 0.00767 |
| TRELLIS.2 | 8/8 | 0.00272 | 0.01497 |

### 拓扑及代表案例

位置合并只作用于诊断副本。表中计数先在对象内平均输入，再取中位数，故可出现小数。

| 数据 | 模型 | 对象数 | 面数 | 原始分量 | 位置合并分量 | 合并后边界边 | 合并后非流形边 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSO | SF3D | 32/32 | 26464.00 | 525.50 | 1.00 | 0.00 | 0.00 |
| GSO | SPAR3D | 32/32 | 31008.00 | 812.00 | 1.00 | 0.00 | 0.00 |
| GSO | TRELLIS.2 | 32/32 | 96483.50 | 1647.50 | 69.25 | 1918.75 | 66.50 |
| ABO | SF3D | 6/6 | 25418.00 | 670.00 | 1.00 | 0.00 | 0.00 |
| ABO | SPAR3D | 6/6 | 30662.00 | 735.00 | 1.00 | 0.00 | 0.00 |
| ABO | TRELLIS.2 | 6/6 | 97231.50 | 3929.50 | 451.00 | 9872.50 | 149.50 |

SF3D/SPAR3D 的原始分量包含 UV seam 的索引分离；位置合并后的诊断与原始计数一起解释。TRELLIS.2 导出资产中的开放面及离散部件按参考语义分析，水密性结合参考语义解释：封闭杯体与开放布条采用各自的结构目标。

高误差案例已检查输入及反向视图、原生/导出部件：

- TRELLIS.2 瓶子：原生输出最大平面占表面积约 60.0%，实用导出后约 67.4%。大平面在生成阶段已经存在，降低了瓶体在整体单位包围盒中的相对尺寸，并影响 CD 与多视图渲染；生成阶段的大平面是误差来源之一，减面影响另列。
- TRELLIS.2 蜜蜂：主体最大部件的最长边约为整体的 54.3%，另有分离部件拓展包围盒。保留这些原始部件，分别讨论结构完整性和归一化影响。
- SF3D 盒体：修正贴图后的几何 CD 仍约 0.08752，参考厚度约 0.2058，预测约 0.5291，长宽接近而厚度被高估。贴图执行异常与真实几何偏差分开解释。
- 木桌组：输入与反向视图显示多桌面及桌腿间遮挡关系的差别。视图内外观相似性和背面结构完整度需结合检查。
- 三孔板与窄缝块：位置合并后，参考与两个基线均水密、绕序一致、闭合顶点 fan 无异常，genus 分别为 3 和 1。窄缝较低的投影孔洞 IoU 反映孔的形状/投影不准，孔本身仍存在。
- 开放片：参考有开放边界，两个基线形成闭合表面。嵌套壳：参考包含两个闭合部件，输出只有外部一个；内部壳从输入不可见，列为观测不充分案例。

![复核后的输入及反向视角案例](../../results/figures/generation_case_review_20261116.png)

### 变换链与自身尺度敏感性审计

审计由原始 Mesh 和保存矩阵独立重建预测/参考视觉几何，核对渲染资产坐标；检查旋转正交性、行列式及原始顶点到视觉顶点的双向最近距离。部署轨道的已声明坐标变换在归一化之前应用，最长边由预测自身决定。

另给出 Oracle 诊断：使用已经 GT 对齐的坐标系，再按预测自身包围盒居中并归一化，最后只做刚体细化。该诊断不改变主结果，不拟合参考尺度，不进入物理适配流程。

全部 356 条记录的刚体无效数为 0，原始到视觉几何变换链不一致数为 0（距离容差为 1e-5 个参考最长边）。另外核验了 92 个原生几何对齐矩阵，均为有效刚体变换。

| 实验 | 模型 | 对象数 | 导出 GLB 原 CD | Oracle CD | 自身尺度倍率中位数 |
| --- | --- | --- | --- | --- | --- |
| gso32 | SF3D | 32 | 0.03068 | 0.02669 | 1.00922 |
| gso32 | SPAR3D | 32 | 0.02228 | 0.02139 | 0.99841 |
| gso32 | TRELLIS.2 | 32 | 0.01566 | 0.01553 | 0.99961 |
| abo6 | SF3D | 6 | 0.02205 | 0.01609 | 1.04971 |
| abo6 | SPAR3D | 6 | 0.02599 | 0.02031 | 1.05917 |
| abo6 | TRELLIS.2 | 6 | 0.01378 | 0.01375 | 0.99556 |
| d2 | SF3D | 8 | 0.06263 | 0.04329 | 1.00031 |
| d2 | SPAR3D | 8 | 0.05948 | 0.05480 | 0.99962 |
| seed0 | SPAR3D | 8 | 0.02270 | 0.02414 | 1.00083 |
| seed1 | SPAR3D | 8 | 0.02411 | 0.02154 | 1.00235 |
| seed2 | SPAR3D | 8 | 0.02322 | 0.02363 | 1.00085 |

对杆架和杯分开施加中心、尺度与刚体修正，使用同一批表面点，结果如下。所有尺度仍由预测自身包围盒决定。

| 模型 | 案例 | 原 CD | 仅居中 | 仅缩放 | 居中+缩放 | 再刚体细化 |
| --- | --- | --- | --- | --- | --- | --- |
| SF3D | P03_wire_frame_v000 | 0.14402 | 0.14386 | 0.04984 | 0.04845 | 0.04822 |
| SF3D | P04_hollow_cup_v000 | 0.08797 | 0.09633 | 0.06106 | 0.03918 | 0.03835 |
| SPAR3D | P03_wire_frame_v000 | 0.14196 | 0.14202 | 0.05266 | 0.05289 | 0.05252 |
| SPAR3D | P04_hollow_cup_v000 | 0.09336 | 0.10109 | 0.07744 | 0.07773 | 0.07155 |

旋转前 AABB 归一化随姿态变化，同一立方体旋转 45° 就能产生可复现的尺度偏差。杆架和杯的低分有这部分贡献，也包含真实形状、局部对齐和中心误差。以审计逐例表为准区分这些因素，避免将全部差异归因于模型结构恢复。

### 有效资产的运行成本

表中为 GSO 40 输入的逐输入中位数。SF3D/SPAR3D 使用同一 RTX 5060；TRELLIS.2 列为 A100 独立条件。PyTorch allocated/reserved 为 MiB，整卡还包括第三方 CUDA 和图形上下文开销；512³ 低显存模式单列，时间为秒，各阶段中位数分别计算。

| 模型 | 生成设备 | 有效输入 | 预处理/s | 推理/s | 导出/s | allocated/MiB | reserved/MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SF3D | RTX 5060 | 40/40 | 0.015 | 0.585 | 0.016 | 5219.1 | 5606.0 |
| SPAR3D | RTX 5060 | 40/40 | 0.014 | 8.934 | 0.018 | 4778.4 | 4982.0 |
| TRELLIS.2 | A100 40GB | 40/40 | 0.032 | 17.404 | 13.206 | 2685.1 | 2748.0 |

本地两个基线的速度同时受模型和低显存执行策略影响。TRELLIS.2 使用不同设备、软件栈及分阶段保存/导出，当前没有三模型同硬件有效资产的统一耗时排名。原服务器基线的计时作为历史工程记录保留，贴图质量异常使其不作为最终有效资产成本排序依据。推理列采用适配器实际边界，TRELLIS.2 包含官方 pipeline 内部预处理；各列记录对应复现开销。本轮没有补做多次热计时，质量按每输入一次统计；原有重复计时独立保存。

### 分母、复现证据与剩余范围

主结果 GSO 每模型 32/32 对象、40/40 输入，ABO 每模型 6/6 对象、6/6 输入，GLB、几何与图像记录齐全。GSO PBR 分母为 0，ABO PBR 每模型为 6/6。生成执行成功与结构保真分别统计。设备显存调试仅保留处理说明，不计入最终模型质量失败。

有效质量来源为 `data/generated/{sf3d,spar3d}/generation_core/{gso32,abo6}_quality_review_20261116/` 与备份中 TRELLIS.2 的实用导出；所有实际参数、权重修订、设备、输入快照及阶段时间见各 run.json/results.jsonl。复核使用这些命令：

```bash
bash .local/quality-review-20261116/generate.sh
# 生成完成后，脚本自动启动 main.sh / other.sh 的评测队列。
# 完成后由 finalize.py 校验退出码、分母、变换链并更新本报告。
/home/zhuze/miniconda3/envs/asset2sim-sf3d/bin/python .local/quality-review-20261116/finalize.py
```

首次执行后的目录已存在；独立复跑时更换输出 run_id。配置为 `configs/experiments/*_quality_review.yaml`，评测使用 `configs/evaluation/generation_benchmark_local.yaml`，原始数据保持不变。`source_camera_elevation_degrees` 记录条件相机，`export_normalization_frame` 声明部署归一化顺序。

结果根目录 `results/raw/generation_benchmark/quality_review_20261116/` 下包含 gso32/combined、abo6、d2、seed0/1/2、audit。每个数据集保留逐输入/逐视角/逐对象、配对区间、类别、视角鲁棒性、计时、分母、失败表和案例图；输入键缺失或重复会阻止分片合并。报告与图表从真实记录生成。相关回归测试 54 项通过，覆盖官方条件相机、矩阵链、归一化顺序、采样索引不变性、闭合顶点邻域及检查点复用。配置、清单、权重、许可和原始训练集重叠情况见前文。

- [GSO 三模型表](../../results/raw/generation_benchmark/quality_review_20261116/gso32/combined/tables.md)
- [ABO 三模型表](../../results/raw/generation_benchmark/quality_review_20261116/abo6/tables.md)
- [变换及尺度审计](../../results/raw/generation_benchmark/quality_review_20261116/audit/summary.json)

![复核后逐对象质量分布](../../results/figures/generation_benchmark_20261116.png)

剩余未验证内容为 TRELLIS.2 的 D2/多种子/1024³/官方默认导出扩展、其精确输入姿态、旧服务器贴图异常的底层实现根因及三模型同硬件有效资产成本。当前结论以已执行设置为范围；这些项目不写作已完成结果。PBR 图像相似性同时受几何和材质影响，未单独分离真实材质参数误差。

### D2 与多种子补充结果（2026-11-16 17:45）

D2 完成 8 对象×2 模型=16 次生成、32 条双轨道记录；SPAR3D 随机性实验完成
8 对象×3 种子=24 次生成、48 条记录。正式补跑均有效。以下结果使用
aligned_diagnostic、1024 纹理、autocast_cache_enabled=false，SPAR3D 采用 low_vram=true。

| D2 模型 | 有效/计划对象 | 生成 CD ↓ | FS@2% ↑ | 新视角 LPIPS ↓ | 新视角 IoU ↑ |
| --- | --- | --- | --- | --- | --- |
| SF3D | 8/8 | 0.06263 | 0.23090 | 0.16913 | 0.59824 |
| SPAR3D | 8/8 | 0.05948 | 0.32097 | 0.16615 | 0.69569 |

两模型配对差值的 95% bootstrap 区间均跨零：CD 为 [−0.00539, 0.00822]，
LPIPS 为 [−0.02147, 0.05302]，IoU 为 [−0.00437, 0.01886]。八种异质结构更适合逐例解释，
总体中位数的小差别不足以支持全面排名。

- **三孔板：** 新视角投影孔洞 IoU 为 SF3D 0.89682、SPAR3D 0.86592，
  recall 为 0.92389、0.94673（每模型 15/15 个新视角有效）。两者可恢复清晰的孔洞投影。
- **开放片与波纹布条：** 参考位置合并后各有 112 条边界边，两模型输出均为 0。
  可见轮廓与贴图相似性没有对应到参考的开放表面拓扑；布条新视角 IoU 为 0.33072、0.31418。
- **嵌套壳：** 两模型新视角 IoU 为 0.99158、0.97272，但参考有 2 个连通分量、输出均为 1。
  外表面图像无法验证内部壳恢复；该内部结构从单图不可见，作为观测不充分案例单列。
- **窄缝块：** SF3D 的生成 CD 为 0.01213，SPAR3D 为 0.05747；
  新视角投影孔洞 IoU 分别为 0.17993（14/15 有效）、0.15351（15/15 有效）。
  轮廓 IoU 较高并不保证窄缝投影在新视角保持。recall 仅在参考孔洞可见时定义，
  本例每模型为 5/15 个新视角，不能将这些分母混用。
- **杆架与杯：** 杆架两模型 FS@2% 均为 0，CD 约 0.14402、0.14196；
  杯 CD 约 0.08797、0.09336。图集中仍可见框架和杯形，低分应结合尺度、中心和朝向诊断解释。
  杆架参考 AABB 为 [1,1,1]，对齐后预测分别为 [0.824,0.820,0.732]、[0.835,0.830,0.700]；
  杯预测中心沿一轴偏移约 0.16/0.20 个参考最长边。旋转前 AABB 归一化本身随朝向变化，
  其尺度敏感性已由本报告的独立审计量化；主指标与 Oracle 诊断分列。

投影孔洞指标来自透明轮廓中封闭背景区域；三孔板与窄缝块已补充水密、绕序及顶点 fan 检查，genus 分别为 3 与 1。
没有参考孔洞像素的视角使用未定义值，空白不作为零分。D2 结果目录为
`results/raw/generation_benchmark/quality_review_20261116/d2/`，包含逐例图集和分母。

| SPAR3D 种子 | 有效/计划对象 | 生成 CD ↓ | FS@2% ↑ | 新视角 LPIPS ↓ | 新视角 IoU ↑ |
| --- | --- | --- | --- | --- | --- |
| 0 | 8/8 | 0.02270 | 0.64333 | 0.09869 | 0.81678 |
| 1 | 8/8 | 0.02411 | 0.58996 | 0.10328 | 0.79164 |
| 2 | 8/8 | 0.02322 | 0.58610 | 0.10376 | 0.78840 |

先对每对象计算三种子的最大值−最小值，再在 8 对象上取中位数：CD 范围为 0.00109，
FS@2% 为 0.02978，LPIPS 为 0.00474，IoU 为 0.01416。多数对象波动较小，但文件架
CD 极差为 0.01021、FS 极差为 0.11097；Bundt 模具 LPIPS 极差为 0.01179，
人偶 IoU 极差为 0.04407，说明结构复杂案例更值得多种子检查。

seed=1 相对 seed=0 的配对 CD 差值中位数为 0.00059，95% 区间为 [0.00001,0.00083]；
seed=2 相对 seed=0 为 0.00040，区间 [−0.00101,0.00109]。
这些是 8 对象、3 种子的探索性结果，主实验保持预先指定 seed=0；
当前结果没有用于按对象挑选最佳种子。补充集选择发生在主实验之后，并且只覆盖 SPAR3D，
不能据此推断 TRELLIS.2 的随机性。

三个种子结果位于 `results/raw/generation_benchmark/quality_review_20261116/seed{0,1,2}/`。
累计主实验和本机补充共 178 个正式产物（含 8 个 seed=0 匹配重复对照）、356 条双轨道记录、6,784 条逐视角指标记录，
样本按各实验原有对象/条件分母分别统计。

### 多种子结论的可重复性复核

同一 seed=0 的八对象重复运行点云完全一致，网格顶点集合最大差异为 0；当前主评测重复得到相同 CD 和 IoU。LPIPS 重复绝对差中位数约为 0.00006，最大约 0.00062，低于跨种子波动。UV 展开会重排索引，另按几何坐标排序三角形采样，以检查对索引排列的敏感性。

| 种子 | 对象数 | 几何排序采样 CD | FS@2% |
| --- | --- | --- | --- |
| 0 | 8 | 0.02394 | 0.60247 |
| 1 | 8 | 0.02437 | 0.57996 |
| 2 | 8 | 0.02356 | 0.58909 |

排序采样与主协议均显示三个种子的总体几何接近，但微小的种子排名随采样方案改变。维持预定 seed=0，并用对象内极差描述端到端敏感性，不按最好种子挑选样本。此项为几何诊断，不改变主图像指标。


补充拓扑与尺度解释以本报告复核章节为准；当前补充数据的 export_pose 已按相同修正规则重算，aligned_diagnostic 原记录复用。

使用 SF3D 与 SPAR3D 生成三维资产，在统一的 GSO 和 OmniObject3D 公开子集上比较几何、外观与运行成本。

## SF3D/SPAR3D 论文子集：实验配置与运行

### 上游代码

TRELLIS.2 官方代码位于 `third_party/TRELLIS.2/`。下载项目后使用 `git submodule update --init --recursive` 获取上游代码和 TRELLIS.2 的 Eigen 子模块。

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
