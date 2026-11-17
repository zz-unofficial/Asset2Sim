# 物理属性与动态仿真

本文包含 PhysX-Omni 三类别生成与静态评测规划，以及已有 Mesh 物性与动态实验。三类别规划使用原生生成流程；已有动态对照采用固定 **C2** 碰撞组件、米制坐标和显式物性，同一几何的不同物性条件复用组件、摩擦及求解器参数。原生导出与规范化实验副本分别保存。

## 实验配置与运行

### PhysX-Omni 三类别物理资产生成与评估任务配置

#### 1. 目标、范围与模型设置

复现 PhysX-Omni 官方预训练模型的单张 RGB → 物理理解与体素编码 → 部件 Mesh → URDF/MJCF 流程，比较 Rigid、Articulated、Deformable 三类案例的几何质量、属性表达及导出完整性。模型和几何解码器沿用上游 PhysX-Omni 与其预训练 TRELLIS Decoder，保存官方版本及实际参数。

本阶段开展生成、评价与静态资产审计。模型训练、TRELLIS.2 解码器替换、MuJoCo 加载/动力学、CoACD、跨模型物性迁移和 UR5e 交互留在既有协议或后续阶段。本节中的 Deformable 指所选 Cloth 形变表面；两个序列是两个序列级测试案例，可能来自同一块布。

规划已确认，当前进入接口与测试准备。新增数据尚未下载，关节实例、Cloth 序列/帧及最终运行清单待筛选。先用本地参考和最小样例验证接口，再接入真实新增数据，最后安排服务器三个类别的首例。

#### 2. 数据来源与实验规模

| 类别 | 来源 | 计划规模 | 新增下载 |
| --- | --- | --- | --- |
| Rigid | 本地已有 GSO32 | 2–3 对象 | 0 |
| Articulated | 第三方 PartNet-Mobility-USD | Microwave、FoldingChair 各 1 实例 | 仅两个类别 tar，合计约 80.5 MB |
| Deformable | EPFL Texture-less Deformable Surfaces | 2 个有 GT Mesh 的不同 Cloth 序列，各 1 帧 | 官方 zip，约 3.8 GB |

共 6–7 个核心输入，分别按对象和序列级案例统计。下载体量约 3.9 GB，实际解压空间按归档目录检查后记录。正式实验只整理最终选中的实例/序列依赖。

**Rigid。** 主选本地 `gso_box_v004`、`gso_mug_v005`，第三候选为 `gso_hammer_v003`；`gso_bowl_v008` 为技术筛选候补。以上输入图、Reference GLB 与渲染元数据已确认在本地存在。优先覆盖规则盒体、杯腔/把手和非对称多材料结构，按准备阶段的完整性审计冻结 2–3 个；Scissors 留在铰接语义范畴。原图、参考几何、尺寸/尺度变换、来源及已有 SF3D/SPAR3D/TRELLIS.2 结果复用原位。材料或质量标签缺失只影响对应属性指标的适用性。

本地 GSO 来源入口为 `.local/local-evaluation-20261116/gso32.jsonl`；目前路径指向已回传数据。数据准备时将实际所需文件与元数据整理为可搬迁清单，保留原有文件及运行记录。尺寸需区分原始物理尺度和评测归一化尺度，按有来源的字段使用。

**Articulated。** 来源明确标记 [AlbertS666/PartNet-Mobility-USD](https://huggingface.co/datasets/AlbertS666/PartNet-Mobility-USD)，是 PartNet-Mobility 的第三方 USD 转换。仅下载 `data/Microwave.tar` 和 `data/FoldingChair.tar`，每类选一个几何依赖完整、关节标注有效的实例。筛选并记录实际 `Object.usd`、`mobility.urdf`、per-link mesh、材质和纹理路径，以及原始对象 ID。

该仓库的 per-link 几何可能是 USD；筛选同时检查 URDF 引用是否实际可解析、USD 的米制/向上轴/嵌套变换及原始和转换后关节是否一致。用实际数据选择 USD 或 URDF 几何读取路径，渲染只依赖所需的轻量解析与 Blender 工具；不把 Isaac Sim 作为本次前置条件。标注记录来源层级，转换产生的差异单独记录。

服务器选用 Microwave `7236` 和 FoldingChair `100526`。URDF 引用的原始 OBJ 未随压缩包提供，参考几何从转换后的 USD 逐 link 提取：先移除 USD 根节点缩放，再按 URDF 关节状态变换。两者均保留原始 URDF、逐 link 网格、关节角度和渲染脚本；输入图使用中性灰材质。源文件空 base link 的 USD 引用未解析，其余视觉 mesh 数量与 URDF 相应 link 一致。物体的绝对实测尺寸尚未核实，尺寸评价以来源单位与可用的尺度标签为准。

**Deformable。** 来源为 [EPFL 官方数据页](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/)。从有三角网格的 7 个 Cloth 序列中选两个，每序列固定一帧。逐帧核对 RGB、Depth、Normal、GT Mesh、前景掩码、相机内参与共同三角连接文件的对应关系。网格由每帧顶点与共享连接关系组成；官方说明该子集网格为 961 顶点，准备时以实际文件核验。

EPFL 的几何/法线参考来自其采集与处理流程，法线由平滑深度计算，保留该来源说明和有效像素掩码。它们用于形状与表面评价，弹性参数需要独立来源。最终序列名称和帧号根据实际下载文件确定；配对不明确的候选在模型运行前更换，或只参加有依据的指标。

全部候选、技术排除理由及备用顺序在查看模型预测前冻结。新增数据仅限这两个 USD 类别包与 EPFL zip；GSO、PhysXVerse、PhysXNet、完整 PartNet-Mobility、CLOTH3D、DOT 和其他刚性集均不在本轮下载范围。

#### 3. 有限下载与文件整理

下列为审核后执行的下载命令。本轮文档修改不执行下载；使用已有工具时复用其环境，缺少工具再在对应准备环境安装。数据根目录可设置到容量充足的本地或服务器数据盘。

```bash
export DATASET_ROOT="$HOME/datasets"
mkdir -p "$DATASET_ROOT/PartNet-Mobility-USD" "$DATASET_ROOT/articulated"

hf download AlbertS666/PartNet-Mobility-USD data/Microwave.tar \
  --repo-type dataset --local-dir "$DATASET_ROOT/PartNet-Mobility-USD"
hf download AlbertS666/PartNet-Mobility-USD data/FoldingChair.tar \
  --repo-type dataset --local-dir "$DATASET_ROOT/PartNet-Mobility-USD"

# 先查看归档目录，确认实例名称及解压后的结构。
tar -tf "$DATASET_ROOT/PartNet-Mobility-USD/data/Microwave.tar"
tar -tf "$DATASET_ROOT/PartNet-Mobility-USD/data/FoldingChair.tar"
# 两个小类别包可解压供筛选；正式目录只整理选中的实例及依赖。
tar -xf "$DATASET_ROOT/PartNet-Mobility-USD/data/Microwave.tar" -C "$DATASET_ROOT/articulated"
tar -xf "$DATASET_ROOT/PartNet-Mobility-USD/data/FoldingChair.tar" -C "$DATASET_ROOT/articulated"

mkdir -p "$DATASET_ROOT/EPFL-Deformable"
gdown '1oZFiIVb5nSRhydepK9KMxB6Nlzi5pZia' \
  -O "$DATASET_ROOT/textureless_deformable_surfaces.zip"
unzip -l "$DATASET_ROOT/textureless_deformable_surfaces.zip"
```

EPFL 官方只提供约 3.8 GB 的整体 zip 下载入口。本轮下载该归档后按目录筛选，选择性解压两个 Cloth 序列及共享标定/连接文件；使用 `unzip <archive> <实际选定路径...> -d <目录>`。序列名核验前保持为待选，不把示例路径当作真实文件，也不执行全包展开和正式目录的全量复制。

`dataset_download_manifest.md` 记录实际下载文件、URL、仓库 revision、实际字节数、下载状态、归档/解压位置、选中内容和许可；GSO 条目标注本地复用。许可与原始引用分别保留，数据和权重存放于被忽略的数据目录。

#### 4. 单图输入准备与 Ground Truth 组织

所有模型输入为独立的 512×512 RGB PNG。保留原始图片，用等比例缩放、裁剪/补边构造画布，目标占比约 65%–80%。记录像素变换、裁剪框、掩码及背景处理；选帧优先满足完整性，调整以保留目标与原始状态为准。

| 类别 | 实际输入准备 | 配对证据 |
| --- | --- | --- |
| Rigid | 复用已有 GSO 输入；需要格式转换时另存副本 | 既有样例/视角 ID、Reference Mesh 与渲染元数据 |
| Articulated | 参考实例在明确开合状态下渲染，记录关节状态和相机 | 同姿态参考部件与整体网格、URDF/USD 变换 |
| Deformable | 使用 EPFL 原始 RGB 的预处理副本 | 原序列/帧、同状态顶点与三角连接、Depth/Normal 和内参 |

EPFL 发布的 RGB/Depth/Normal 为 224×224，原始采集分辨率与发布分辨率分开记录。缩放到 512×512 只改变图像采样，不增加真实细节。用二维变换 A 保存输入处理，针孔内参相应为 K'=A K；参考深度/法线及掩码按声明的插值和有效区域处理。几何仍保留原始状态及坐标。官方几何/法线采用 OpenCV 的 X-right、Y-down、Z-forward，进入评测坐标系时保存显式变换。

Articulated 的 URDF、关节状态、标注和参考部件仅由数据准备/评价器读取；生成阶段输入单张 RGB。Deformable 的参考深度、法线和网格也仅用于评价。

每个样例维护独立 GT 入口，实际目录名采用原始对象 ID 或序列+帧 ID；共享原始文件可用清单引用，以减少重复存储。

| 类别 | GT 入口内容 |
| --- | --- |
| Rigid | Reference Mesh、对应图片/相机、尺寸与尺度链、已有的材料/质量标签 |
| Articulated | Object.usd、URDF、部件几何和变换、link/joint/parent-child、关节状态与单位 |
| Deformable | 原 RGB、Depth、Normal、有效掩码、GT 顶点和三角连接、内参、序列与帧编号 |

每个物理字段记录值、单位、来源 URL/文件、获取方式及证据类型：实测、数据集标注、第三方转换、人工标注、材料先验、模型推断。缺少可靠物性标注时值为 NA 并保存原因。人工标签与模型预测独立存储；密度、杨氏模量、泊松比的数值合理性与真实误差分表。

数据准备阶段必须产出以下文件，当前仅定义其结构，待实际筛选后填写：

| 文件 | 位置与内容 |
| --- | --- |
| dataset_manifest.csv | 每次数据准备输出目录；sample_id、category、source、object_id、sequence_id、frame_id、image、reference_mesh、sample_metadata；冻结后的选择记录可整理至 `experiments/datasets/` |
| dataset_download_manifest.md | 同目录；实际下载/复用、大小、来源和选择内容 |
| sample_metadata.json | 每样例 GT 目录；尺寸/姿态、相机、像素/坐标变换、关节、序列帧号、物理标签来源及 NA 原因 |
| 推理 Manifest JSONL | 准备目录的 `manifest.jsonl`；复用项目清单接口，与 CSV 按 sample_id 对应；冻结后可通过既有清单导出接口发布 |

跨设备打包时重定位路径并检查所有依赖。最终清单只含真实存在且已配对的文件，保留旧数据和旧实验输出。

#### 5. 官方推理流程与过程记录

1. **VLM：** 官方 `1vlm_demo.py`，保存整体/部件描述、材料、密度、杨氏模量、泊松比、affordance rank、group_info 和逐部件 64³ RLE/体素。记录解析与体素有效性；当前提示词的可供性是部件级字段，空间区域按实际输出情况记录。
2. **Geometry Decoder：** 官方 `2infer_geo.py` 调用 `decoder_each.py` 完成部件解码。项目可直接调用该官方子入口以捕获返回状态，保持相同解码方法与参数。逐部件记录 ID、体素到 Mesh 的对应、耗时、缺失/空/异常 Mesh，保存原始输出和纹理。
3. **Asset Export：** 官方 `3jsongen_update.py`，保存 URDF/MJCF 和中间 JSON，核对类别分支的实际触发。Rigid/Articulated 使用对应官方配置；Deformable 比较默认与 `--deformable 1` 的导出，复用同一次推理与解码。官方开关属于配置条件；源码适配另存记录并与原生结果分组。

当前上游 VLM 使用 `do_sample=False`、`max_length=32768`，解码使用 `seed=1`，部件 GLB 导出为 `simplify=0.5`、`texture_size=1024`。基线保留这些设置并写入解析后配置；源码版本变化时核对实际调用参数。对失败部件保存记录；对象级成功要求必需部件和类别结构完整。上游可能捕获错误后返回 0，因此按实际阶段产物判定。

阶段判定同时记录 `execution_success`（子进程返回码为零）和 `output_valid`（独立产物检查通过）。正式成功要求两者均为真；复用阶段记录 `reused` 和来源运行，本轮执行状态为 NA。VLM 检查原始部件描述、连续部件 ID、非空有限整数体素、64³ 范围及合并数组一致性，并隔离加载官方纯 RLE 辅助函数，重放 `coord_<id>.txt` 与保存的体素逐项比较。Decoder 检查每个预测部件的 OBJ/GLB 可读、非空和有限顶点。Export 检查 JSON、URDF/MJCF、部件覆盖和文件引用；属性传递错误另存审计结果。

每阶段由独立子进程串行执行，RTX 4090D 24 GB 上先结束 VLM 再加载 Decoder。运行记录保存输入清单、实际 RGB、checkpoint 路径、上游 commit、环境版本、返回码、日志、资源峰值与耗时。

Cloth 导出对照使用同一轮冻结的 VLM 和 Decoder 结果，分别创建两个新 run 目录：

```bash
asset2sim physx-omni --manifest <统一清单> --sample-id <Cloth样本ID> \
  --reuse-run <冻结生成运行> --from-stage export --no-deformable \
  --output results/raw/physx_omni/<D0运行>
asset2sim physx-omni --manifest <统一清单> --sample-id <Cloth样本ID> \
  --reuse-run <冻结生成运行> --from-stage export --deformable \
  --output results/raw/physx_omni/<D1运行>
```

每个样本保存 `frozen_generation/`、D0 的 `export_default/` 或 D1 的 `export_deformable/`，以及本次日志、文件哈希和导出参数。复制的旧 basic.xml、basic.urdf、basic_info.json 在导出前移除，使本次判定对应新产物。

柔性审计分别报告 `group_info` 非根组的 A 标记、XML 中可观察的柔性分支结果、`dim="2"` 与 elasticity 数值。分支的观察结果来自最终文件；没有对应结构时记录 `not_observed`。原始 group_info 随生成结果保留。

Microwave `7236` 的官方基线在 Decoder 网格提取时发生 24 GB 显存不足，保留该次 VLM 及失败记录。其他对象运行结束后，可复用该 VLM，在新目录启用可选显存适配：

```bash
asset2sim physx-omni \
  --manifest data/processed/physx_omni/three_category_20261117.jsonl \
  --sample-id partnet_microwave_7236 \
  --reuse-run results/raw/physx_omni/server_pilot_partnet_microwave_7236_20261117 \
  --from-stage decode --decoder-cpu-offload \
  --output results/raw/physx_omni/<新的适配运行ID>
```

该模式仅在官方 `run_decoder` 生成调用内关闭梯度；潜变量采样后将图像编码器和潜变量流模型移至 CPU，后续贴图烘焙保持原有梯度行为。它保留相同体素、部件顺序、随机种子及导出参数，运行记录中的 `decoder_cpu_offload` 标识工程适配。先比较相同输入的基线与适配结果及时间、显存；24 GB 能否完成需通过实际运行确认。

EPFL `Lr_bottom_edge/0629` 的显存适配运行已完成 Decoder 和导出。模型将输入描述为床架、枕头、床垫和毯子；其中三个 A 组导出二维 flexcomp，根组床架使用刚体表示。原始 `objs/0/0.obj` 有 352,332 个面，其中 2 个零面积三角面；原始导出保留在 `server_cloth_offload_20261117/`。独立目录 `server_cloth_repair_20261117/asset/` 从 OBJ 副本显式移除这两个面，保留顶点、材质记录和原始 GLB；修复后静态审计无结构或属性传递错误。`repair_report.json` 保存操作、前后诊断和语义误判。混合刚体/柔性结果的审计只对柔性部件要求弹性字段；语义误判仍按原始预测分析。

#### 6. 几何质量评价

复用现有几何评价器的采样、CD-L1 与 F-score 实现，冻结点数、种子、阈值及单位。输出绝对尺度和归一化形状两条轨道：前者使用有来源的原始米制参考和刚体变换，后者显式归一化；三个方向尺寸误差独立报告。缺少可靠原始尺度时保留形状指标及尺度字段审计。

Rigid 评价 CD、F-score、BBox Error、几何完整性、连通、退化、空腔/把手/薄壁。Bowl/Cup 的几何开放和材料体积问题按实际参考解释。

Articulated 的主要指标为部件数量/召回、Parent-Child、Joint Type、Joint Axis 和 Joint Origin；Joint Range、部件 CD 及关节文本描述为辅助指标。单帧图像提供的是当前姿态，运动范围误差结合图像中的可辨识性解释。先按部件语义和几何建立匹配，再计算部件数量误差、召回、缺失/额外部件、部件 CD 和定义明确的表面或体素 Part IoU。全体部件采用同一整体对齐变换，按选定输入状态比较参考几何。

Cloth 先检查相机与 Mesh 坐标及显式 `prediction_to_reference` 变换，固定姿态轨道仅应用该声明变换；缺少可靠对应时状态为 `NA_coordinate_correspondence_unverified`。归一化刚体对齐单独作为形状诊断；评价不使用非刚性配准。法线主指标采用无向点积，适合薄片双面语义，前后面解释结合参考方向记录。Deformable 评价同帧表面 CD、F-score、Surface Completeness、Normal Consistency、连通、退化、自相交与薄片完整性。完整性采用参考表面到预测表面的阈值召回并声明阈值；法线区分几何表面匹配与像素投影匹配，保存有效区域和方向约定。合理开放薄片按开放参考评价；RGB/GT 配对是定量表面指标的前置条件。

#### 7. 类别专属属性与结构评价

| 类别 | 主要内容 | 参考与统计口径 |
| --- | --- | --- |
| Rigid | 材料、密度、尺寸、质量、功能描述 | 有可靠标签时算分类/回归误差；其余报告字段、来源和合理性；多材料按部件 |
| Articulated | 部件、关节类型/轴线/位置/范围、运动学树 | 使用 URDF/USD 的可追溯结构标注，排除仅为表示而增加的辅助 link/joint 后比较语义结构 |
| Deformable | 材料、密度、Young's Modulus、Poisson's Ratio、阻尼 | EPFL 主要提供形状参考，所选样例没有实测弹性标签；报告字段完整性、数值有效性和有来源的合理区间 |

有真值时，尺寸报 MAPE，密度/质量报相对误差，杨氏模量报正值对数误差，泊松比报绝对误差。没有预测、没有参考、解析失败和类别不适用分别记录 NA 原因及有效分母；人工从材料类别推定的参数作为先验。

关节轴统一到共同坐标系后计算无向夹角，内积裁剪到合法范围。轴线位置优先采用轴线距离或参考关节点到预测轴线距离，并按物体尺度归一化；关节点欧氏距离仅用于有一致原点定义的情况。范围分别比较上下限，旋转/平移单位分开，处理轴符号翻转和 continuous joint。

GSO32 与 PartNet-Mobility 可能与模型训练数据重叠；来源关系随结果记录，本轮定位为可复核的小规模专项复现。两个 Cloth 序列按两个案例报告，物体级类别均为 Cloth，序列间的关系保留原始数据说明。

#### 8. 静态资产与属性传递审计

检查文件存在、XML/URDF 可解析、相对路径和纹理依赖、Mesh 可读性与有限值、索引和退化，以及物性字段有效性。Rigid 审计 geometry/scale/density/mass/COM/inertia；Articulated 审计 link、joint、父子树、类型、轴、原点、范围；Deformable 审计 flexcomp/对应结构、质量/密度来源、Young、Poisson、阻尼、厚度和弹性模式。

逐字段保存 `source_type`：predicted、derived、default、constant、missing；原始预测来源与最终导出值的 `export_source_type` 分开记录。URDF 的固定质量/惯性占位值按该文件审计，MJCF 根据密度或编译器推导的量按 MJCF 审计。密度乘 1000 的转换明确记录 g/cm³ 假设及原文单位是否核实。逐字段追踪 VLM 原文 → JSON → 最终文件，状态为 PRESERVED、TRANSFORMED、OVERRIDDEN、MISSING，并补充 DERIVED、DEFAULTED、DEFERRED_TO_ENGINE。记录转换公式和单位，特别标明 URDF 惯性占位值、geom 密度推导、柔性分支触发与固定阻尼。

本阶段的审计终点是静态文件。引擎编译时才确定的量记录其推导方式，留待后续加载阶段核验。`asset2sim physx-omni` 默认执行三个官方阶段；`--validate-mujoco` 可单独启用加载检查。静态评价入口为 `physx-omni-evaluate`，不导入 MuJoCo。

#### 9. 参考 PhysX-Bench 的辅助评价

保留 RQS/MCS/DCS/DQS/APS 的研究维度，使用统一多视角渲染、独立参考描述、尺寸比较及有规则的人工部件/功能核验。自定义评价以实际指标名报告；采用官方评分流程时同时保存提示词、评审模型和证据。APS 有空间预测才做区域评价，其余使用部件级语义证据。KPS/MPS 的动态评分留在后续阶段。

#### 10. 成功率、运行成本与执行位置

按冻结后的实际规模分别统计：Rigid 2–3、Articulated 2、Deformable 2 个序列案例。各类 VLM、Decoder、Export、端到端对象/案例级成功率都以该类计划输入数为分母；另报阶段条件成功率以定位瓶颈。对象/案例级端到端成功要求必需部件完整及相应类别结构有效。

部件解码成功率为成功解码部件/预测部件，同时报告有标注时的参考部件召回，区分少预测与解码失败。每个指标同时给出 n_valid/n_planned 和 NA 原因。所有类别逐样例列出，不用不同规模类别的简单平均代替分层结果。

| 环节 | 地点 | 产物 |
| --- | --- | --- |
| 本地旧数据核对、有限下载/筛选、参考渲染、配对与评价器测试 | 本机已有环境；大归档也可先放服务器数据盘 | 冻结清单、输入、GT、下载记录 |
| PhysX-Omni 环境、权重、VLM、解码及官方导出 | GPU 服务器 | 完整中间输出、部件资产、属性和阶段资源记录 |
| 几何评价、结构/物性字段审计、图表和选择性渲染 | 本机已有环境 | 指标、失败分析、报告和案例图 |

每对象/部件记录加载、VLM、解码、贴图/导出墙钟、GPU allocated/reserved 峰值、CPU RSS 峰值、输出字节数、部件数和重试；冷启动与温运行分开，同类比较保留统一硬件/参数。CPU 导出与 GPU 工作细分计时后制定服务器调度。数据与模型准备不安装到本机生成环境。

#### 11. 结果组织与交付

每个新 run_id 输出 pipeline_summary.csv、geometry_metrics.csv、rigid_attributes.csv、articulated_metrics.csv、deformable_metrics.csv、export_audit.csv、runtime_cost.csv、failure_cases.csv。所有表按 sample_id、类别、原始对象/序列/帧、阶段/部件和参数条件关联。原始 VLM、体素、几何、纹理、JSON、XML/URDF 与日志一并保留。

正式产物放在被忽略的运行目录；代码与配置、对象选择和元数据规范入库。本文维护关键操作与结论，现有动态实验结果保留原位。

#### 12. 执行门槛与验收

1. 审阅本配置，确认 2–3+2+2 的范围。
2. 先准备读取、配对、清单、静态审计、计时和失败记录接口；用已有参考与合成最小样例测试，补齐静态/加载解耦。
3. 执行两 tar 和 EPFL zip 的有限下载，核验实际文件后冻结实例与帧；生成三个数据管理文件及公共 JSONL。
4. 本机先验证几何评价器、输入相机变换、部件/关节指标及属性追踪。关节和 Cloth 的标注配对具备可复核证据后再安排 GPU。
5. 服务器先跑每类一个完整样例，含解码、导出、资源统计和回传后静态审计。根据实测部件数与耗时估算剩余 3–4 个输入及费用，再展开正式批次。
6. 正式 6–7 个输入均保留终态；失败提供阶段与原因，成功输出可追溯的中间产物。原始预测与重新导出条件关联，导出参数对照复用推理结果。
7. 本机完成几何、类别属性、静态传递、成本及案例分析；分别报告三类计数和有效分母，交付复现命令、数据记录、表格与图。

当前已实现数据准备、分阶段运行和静态评价接口，并在已有 GSO 数据及最小三类别样例上验证。USD 原始资产的转换/渲染、真实 Cloth 的目录与帧配对将在下载指定数据后联调；服务器模型与 CUDA 环境尚未实测。

#### 接口使用与数据配置

本机使用已有 `asset2sim-physics` 环境。下列输出目录均取新名称：

```bash
conda activate asset2sim-physics
asset2sim prepare-physx-data --config configs/datasets/physx_omni.yaml \
  --output data/processed/physx_omni/<准备run_id>
asset2sim physx-omni \
  --manifest data/processed/physx_omni/<准备run_id>/manifest.jsonl \
  --plan --output results/raw/physx_omni/<计划run_id>
```

服务器已准备 3 个 GSO 刚体、2 个 PartNet-Mobility-USD 铰接对象和 2 个 EPFL Cloth 序列级案例。统一清单为 `data/processed/physx_omni/three_category_20261117.jsonl`；各类别的准备目录分别为 `interface_20261116/`、`articulated_prepare01/`、`epfl_prepare01/`。EPFL 样本保留原始 RGB、深度、法线、网格、相机内参及同帧核验记录。实际推理需要开卡后验证 CUDA 扩展和模型加载。

新数据选择写入 `samples`，路径相对 YAML。共同字段为 `sample_id/category/source/image/reference_mesh`，或通过 `source_manifest/source_sample_id` 复用旧清单。类别字段如下：

| 类别 | 数据接口必需字段及约定 |
| --- | --- |
| Articulated | 已渲染 RGB、同状态整体参考 Mesh、`urdf`、`joint_state`、`render_evidence`；输入 URDF 需有有效可动关节。`joint_matches/link_matches` 明确评价对应，`joint_prediction_to_reference` 声明统一坐标；未建立对应时对应指标为 NA |
| Articulated 部件 | `part_references` 记录每个参考部件的 `mesh/part_to_reference`，`part_matches` 指定一对一关系；指标沿用整体对齐，不单独拟合部件 |
| Deformable | `sequence_id/frame_id/image/depth/normal/mesh_vertices/triangles/intrinsics`，显式 `triangle_index_base`；`pairing_verified/pairing_evidence` 记录同帧核验，核心案例要求不同序列 |
| 物理标签 | `physical_labels` 中每字段保存 `value/unit/source/evidence_type/ground_truth`；模型推断和材料先验不能标记成真实 Ground Truth |
| 绝对尺度 | `absolute_scale_verified` 与正值 `reference_unit_m`；已知参考单位后，才计算模型 Dimension 声明对应的绝对尺度误差 |

数据准备保留原图并生成 512×512 RGB 副本、像素变换和更新内参；Cloth 深度/法线保留原始分辨率。参考数据和元数据采用相对路径组成可搬迁入口，原始来源路径作为追溯信息保留。URDF/USD 的完整外部几何依赖在真实实例接入时继续核验。

服务器无卡阶段已缓存模型权重及扩展源码。开卡后完成 CUDA 扩展安装，再先对一个刚体样本执行试运行：

```bash
cd /root/autodl-tmp/Asset2Sim
source /root/autodl-tmp/asset2sim-cache/env.sh
export PHYSX_BASE_READY=1
bash scripts/setup_physx_omni.sh --server
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asset2sim-physx-omni
asset2sim physx-omni \
  --manifest data/processed/physx_omni/three_category_20261117.jsonl \
  --sample-id gso_box_v004 \
  --output results/raw/physx_omni/server_pilot_01
```

核对该样本的三阶段产物后，再按同一清单运行其余对象。GPU 扩展编译和真实推理耗时依实际 GPU 及系统状态确定。服务器统一清单目录还包含 `dataset_manifest.csv` 与 `dataset_download_manifest.md`。

服务器安装与准备模型后，复用同一命令去掉 `--plan` 执行三个官方阶段。`configs/physics/physx_omni.yaml` 默认按 Deformable 类别启用柔性导出。导出对照复用旧产物，保存新目录：

```bash
# 服务器：仅重新导出默认刚体分支，用于柔性开关对照。
asset2sim physx-omni --manifest <准备目录>/manifest.jsonl \
  --sample-id <实际Cloth案例ID> --reuse-run <原运行目录> \
  --from-stage export --no-deformable --output <新导出运行目录>

# 本机：统一几何评价、三类别静态审计与分母汇总。
asset2sim physx-omni-evaluate --run <生成运行目录> \
  --manifest <准备目录>/manifest.jsonl --sample-count 10000 \
  --output results/raw/physx_omni/<评测run_id>
```

`--from-stage decode` 可复用 VLM 结果；复用前检查实际 RGB 一致，并重新验证前序产物。每次运行记录阶段状态和来源，不覆盖原目录。`stage_timeout_seconds` 默认 null，确认服务器预算后设置；资源包装器记录 CPU RSS、Torch allocated/reserved 峰值。当前解码计时包含官方贴图和导出，字段明确标注这一范围；独立导出脚本按 CPU 阶段记录。

静态评测输出八张约定 CSV、逐案例审计和 `summary.json`。归一化几何为刚体对齐诊断，保存变换；缺少部件的几何标记 partial。缺少独立物性参考、部件匹配或共同关节坐标时，字段附 NA 原因。未计算的自相交状态单独标记。

接口测试覆盖搬迁后的清单读取、RGB/Cloth 配对、错误 Ground Truth 来源、类别分支、单位转换、惯性占位/非法惯量、轴方向与范围符号、共同部件变换、缺失对象分母、静态运行、导出恢复和超时。使用既有环境执行：

```bash
python -m pytest -q tests/test_physx_omni.py tests/test_physx_data.py \
  tests/test_physx_static.py tests/test_physx_resources.py tests/test_experiments.py
```

#### 服务器候选环境与现有入口

保留现有 `environments/physx_omni/`、`scripts/setup_physx_omni.sh --server`、`configs/physics/physx_omni.yaml` 和 `asset2sim physx-omni` 基础封装。本机不安装模型环境。候选为 Linux、Python 3.10、PyTorch 2.4.0/cu121、CUDA Toolkit 12.1、Transformers 4.50.0；版本兼容与三类别静态入口在代码准备后验证。

设备建议单卡 RTX 4090 24GB，预算允许可选 48GB；实际可用 CPU 至少 12 核、RAM 64GB、数据盘 150–200GB。环境、Hub 缓存和归档均放数据盘，完整显存与耗时由三类别首例测定。安装与权重下载命令在接口冻结后给出；已有刚体与动态实验命令见下节。

来源：[PhysX-Omni 官方代码](https://github.com/ziangcao0312/PhysX-Omni)、[PhysX-Bench](https://github.com/ziangcao0312/PhysX-Omni/tree/main/benchmark)、[PartNet-Mobility-USD](https://huggingface.co/datasets/AlbertS666/PartNet-Mobility-USD)、[EPFL 官方数据](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/)。

### 环境、命令与物性

#### 安装和运行

先完成 [MuJoCo 协议](mujoco_contact_evaluation.md)中的 Task 4A 三个批次。物理环境不需要 CUDA；视频使用 MuJoCo EGL 离屏渲染及 imageio-ffmpeg 编码。

```bash
bash scripts/setup_physics.sh
conda activate asset2sim-physics
asset2sim dynamics-eval --config configs/physics/dynamics.yaml \
  --output results/physics/dynamics/<新run_id>
```

默认配置选择 Reference、SF3D、SPAR3D 的 `debug_box_v0`。可在配置的 `assets` 中更换已有 Task 4A 批次、样例及碰撞条件；同一资产的所有物性配置会复用指定条件。生成资产要求原批次已通过 `physics-index`。输出目录必须不存在：`run.json` 保存配置、物性、来源、分母、各次时间和失败原因，`summary.csv` 是逐试验汇总；各试验目录保存可重新加载的 `scene.xml`、碰撞组件副本、`trajectory.csv` 和 `simulation.mp4`。输出保存在被 Git 忽略的 `results/`，结论写入本文。

#### 物理参数与实验

- **baseline**：所有资产质量均为 0.10 kg；以准备后网格的轴对齐包围盒中心为质心，按均匀实心盒计算惯性。相同方法不意味着不同几何具有相同惯量。
- **mesh_fixed_mass**：仍为 0.10 kg，但从准备后原始网格的封闭体积计算均匀密度质心与完整惯性张量，惯性按质量缩放。与 baseline 的差异可用于观察质心/惯性假设。
- **material_prior**：同一网格体积、质心与惯性计算，假设均匀实心且密度为 **600 kg/m³**，作为类木材的演示先验；未根据纹理推断材料，也未测量真实密度、空腔或壁厚。

网格体积估计只接受 Task A 诊断为 `ready` 的水密、方向一致、正体积网格。条件不满足时记录 `property_estimation` 失败；AABB 基线独立记录其假设。惯性在质心处计算，使用 `fullinertia` 写入 MuJoCo，并校验编译后的质量、质心及完整惯性张量。网格体积来自原始网格；自相交情况需另行检查。[trimesh 质量属性](https://trimesh.org/trimesh.html)及 [MuJoCo 惯性参数](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html)对体积和惯性坐标有相应约定。

共同物理设置：重力 `(0, 0, -9.81)` m/s²、时间步 0.002 s、摩擦 `(0.5, 0.005, 0.0001)`、margin/gap 为 0、Newton 求解器迭代 100 次，三组均使用相同 `solref`/`solimp`。这些接触参数为实验设定值。

| 实验 | 初始条件与控制 | 主要记录 |
| --- | --- | --- |
| 自由落体 | 碰撞体最低点距地面 0.15 m，初速为零，模拟 2 s | 首次接地时间、位姿与速度、落地后运动 |
| 稳定放置 | 最低点距地面 0.001 m，初始姿态取准备后坐标，模拟 2 s | 接触、最终速度、最后 0.5 s 高度变化、相对初始转角 |
| 受控推动 | 至少等待 0.6 s，物体连续 0.2 s 接地并满足稳定速度及高度波动阈值后，按当前碰撞体世界 AABB 放置推杆；目标沿 +X 移动 0.12 m、持续 1 s | 稳定等待时间、推杆是否接触、开始后位移、转角、末态稳定性 |

球形推杆是带滑动关节和位置驱动的动态刚体。推杆半径 0.015 m、质量 0.20 kg、初始间隙 0.03 m，驱动和接触参数对所有组一致。稳定后按各组当前包围盒放置是统一的试验规则，不读取参考几何；因此实际推杆世界坐标可能不同，逐试验记录在 `run.json`。等待至 3 s 仍未稳定时记录 `settle_timeout`，不放置或启动推杆；无推杆接触、非有限状态、MuJoCo 警告及视频失败也记为无效/失败。视频标明等待、推动、保持或等待超时阶段。

`stable` 只表示末态接地、线速度 < 0.02 m/s、角速度 < 0.2 rad/s 且末尾 0.5 s 高度范围 < 0.002 m；初始与末态之间的姿态变化另行记录。轨迹每 0.02 s 记录一次 body 位置、质心、姿态四元数、自由关节速度、接触状态和推杆位置/驱动力。MuJoCo 接触状态来自最近一次步进的碰撞计算，时间分辨率为一个时间步；角速度分量遵循 MuJoCo 自由关节约定。推动位移从推杆开始时的物体位置计算，不把此前放置阶段位移计入。

每个试验另执行 3 次不渲染的重复仿真，记录每次 `mj_step` 总墙钟时间、实际步数及有效/计划次数；等待稳定的推动试验可能长于两秒，因此同时记录每步平均耗时。渲染/编码时间单独记录。该计时包含步进计算，受 CPU 负载、组件数、接触和模型形状共同影响。视频为 480×360、20 fps 的离屏视图，红球为推杆；视频显示简化视觉网格，碰撞组件保持独立。

## 实验结论

### 三个 box 资产的动态结果

修正计时后，运行 `results/physics/dynamics/debug_box_run05` 使用 Intel Core Ultra 7 265K、MuJoCo 3.3.5、trimesh 4.4.1。**27/27** 个试验当时按固定 0.6 s 等待规则记录为有效，**27/27** 个视频可解码（各 41 帧），无 MuJoCo 警告；每组不渲染计时均有 3/3 个实际保存的有效重复。该历史运行采用固定等待时间；现行实验增加了推动前的稳定门槛。不同来源的碰撞组件数为 Reference 1、SF3D 4、SPAR3D 52；组件沿用已有资产。

此前 `debug_box_run04` 的计时追加位置错误：每组虽然执行 3 次重复，只保存了最后 1 次，记录中的有效/计划数为 1/1。因此旧报告中该运行的“3/3 有效重复”及其作为三次平均的耗时表不成立；`run04` 的原始记录与视频保留，以下耗时改用修复后的 `run05`。物理轨迹和推杆接触的比较不依赖该计时均值。

| 来源 | baseline 质量 kg | mesh_fixed_mass 质量 kg | material_prior 质量 kg | 网格质心相对物体原点 mm |
| --- | ---: | ---: | ---: | --- |
| Reference | 0.100 | 0.100 | 1.800 | (0.00, 0.00, 0.00) |
| SF3D | 0.100 | 0.100 | 0.922 | (1.66, -0.34, -1.51) |
| SPAR3D | 0.100 | 0.100 | 0.656 | (6.08, 2.22, -2.29) |

按均匀密度估计的网格体积分别约为 0.003000、0.001537、0.001093 m³；估计质量由归一化网格体积和假设密度共同决定。完整张量及假设保存在 `run.json`。

| 来源 | 物性方案 | 推动后 X 位移 cm | 推杆接触步数 / 1000 | 推动末态 stable | 1000 步平均耗时 ms |
| --- | --- | ---: | ---: | --- | ---: |
| Reference | baseline | 8.93 | 506 / 1000 | 是 | 4.55 |
| Reference | mesh_fixed_mass | 8.93 | 506 / 1000 | 是 | 4.46 |
| Reference | material_prior | 1.58 | 581 / 1000 | 是 | 4.45 |
| SF3D | baseline | 4.46 | 305 / 1000 | 否 | 5.81 |
| SF3D | mesh_fixed_mass | 5.13 | 254 / 1000 | 是 | 5.93 |
| SF3D | material_prior | 2.52 | 485 / 1000 | 是 | 5.87 |
| SPAR3D | baseline | 4.82 | 262 / 1000 | 是 | 10.77 |
| SPAR3D | mesh_fixed_mass | 4.80 | 259 / 1000 | 是 | 11.27 |
| SPAR3D | material_prior | 3.33 | 485 / 1000 | 是 | 11.25 |

全部 9/9 落体试验首次接地均约 0.176 s，与 0.15 m 理想自由落体约 0.175 s 接近，与本设置的初始距离和重力相符。所有 9/9 放置试验在 2 s 末满足 `stable`，但 baseline 组相对初始姿态的转角约为 Reference 0°、SF3D 22.1°、SPAR3D 75.3°。SF3D baseline 推动末态未达到稳定阈值。

在此推杆和摩擦条件下，较大的密度先验质量对应较小的推动位移；SF3D 固定质量下改变质心和惯性也改变了动态结果。SPAR3D 的成本更高同时伴随更多凸组件和不同几何，耗时同时受到组件数和几何形状影响。初次 `run01` 的推杆按初始位置布置，SPAR3D 放置后偏移导致 3 组推杆均未接触，这三组标记为无效推动并保留记录。`run05` 按固定等待后的包围盒布置推杆，9/9 推动试验发生接触，但当时没有验证推动前稳定。

结果覆盖一个物体类别、三个几何来源和三组物性先验。进一步比较可结合真实质量、材料与摩擦测量，并验证生成资产相对参考的姿态。

自动化验证在 `asset2sim-physics` 环境执行 `python -m pytest -q tests/test_dynamics.py tests/test_physics_batch.py tests/test_physics.py tests/test_collision.py tests/test_geometry_prepare.py`，43 项通过，覆盖解析盒子物性、非水密拒绝、固定碰撞网格、落体时间、推杆接触、缺失资产记录，以及重复计时和失败计数。既有模型环境此前执行 `python -m pytest -q --ignore=tests/test_dynamics.py --ignore=tests/test_physics.py --ignore=tests/test_physics_batch.py`，59 项通过。
