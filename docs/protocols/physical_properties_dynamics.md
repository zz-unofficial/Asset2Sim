# 物理属性配置与动态仿真

本实验复用 Task 4A 的已准备 Mesh 和 **C2** 碰撞组件，比较物性配置，不重新生成资产、不调整 CoACD。每个物体的所有配置使用完全相同的组件、初始网格坐标、摩擦和求解器参数。质量、质心和惯性均为可追踪的假设，不代表真实物性；0.20 m 归一化尺寸也不是实物尺寸。

## 安装和运行

先完成 [MuJoCo 协议](mujoco_contact_evaluation.md)中的 Task 4A 三个批次。物理环境不需要 CUDA；视频使用 MuJoCo EGL 离屏渲染及 imageio-ffmpeg 编码。

```bash
bash scripts/setup_physics.sh
conda activate asset2sim-physics
asset2sim dynamics-eval --config configs/physics/dynamics.yaml \
  --output results/physics/dynamics/<新run_id>
```

默认配置选择 Reference、SF3D、SPAR3D 的 `debug_box_v0`。可在配置的 `assets` 中更换已有 Task 4A 批次、样例及碰撞条件；同一资产的所有物性配置会复用指定条件。生成资产要求原批次已通过 `physics-index`。输出目录必须不存在：`run.json` 保存配置、物性、来源、分母、各次时间和失败原因，`summary.csv` 是逐试验汇总；各试验目录保存可重新加载的 `scene.xml`、碰撞组件副本、`trajectory.csv` 和 `simulation.mp4`。输出保存在被 Git 忽略的 `results/`，结论写入本文。

## 物理参数与实验

- **baseline**：所有资产质量均为 0.10 kg；以准备后网格的轴对齐包围盒中心为质心，按均匀实心盒计算惯性。相同方法不意味着不同几何具有相同惯量。
- **mesh_fixed_mass**：仍为 0.10 kg，但从准备后原始网格的封闭体积计算均匀密度质心与完整惯性张量，惯性按质量缩放。与 baseline 的差异可用于观察质心/惯性假设。
- **material_prior**：同一网格体积、质心与惯性计算，假设均匀实心且密度为 **600 kg/m³**，作为类木材的演示先验；未根据纹理推断材料，也未测量真实密度、空腔或壁厚。

网格体积估计只接受 Task A 诊断为 `ready` 的水密、方向一致、正体积网格。拒绝时记录 `property_estimation` 失败，不能静默把包围盒结果冒充网格估计；AABB 基线单独明确标记其假设。惯性在质心处计算，使用 `fullinertia` 写入 MuJoCo，并校验编译后的质量、质心及完整惯性张量；不把多个重叠凸组件的体积简单相加。水密和正体积并不能证明不存在自相交。[trimesh 质量属性](https://trimesh.org/trimesh.html)及 [MuJoCo 惯性参数](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html)对体积和惯性坐标有相应约定。

共同物理设置：重力 `(0, 0, -9.81)` m/s²、时间步 0.002 s、摩擦 `(0.5, 0.005, 0.0001)`、margin/gap 为 0、Newton 求解器迭代 100 次，三组均使用相同 `solref`/`solimp`。不可将这些设置解释为测得的真实接触参数。

| 实验 | 初始条件与控制 | 主要记录 |
| --- | --- | --- |
| 自由落体 | 碰撞体最低点距地面 0.15 m，初速为零，模拟 2 s | 首次接地时间、位姿与速度、落地后运动 |
| 稳定放置 | 最低点距地面 0.001 m，初始姿态取准备后坐标，模拟 2 s | 接触、最终速度、最后 0.5 s 高度变化、相对初始转角 |
| 受控推动 | 至少等待 0.6 s，物体连续 0.2 s 接地并满足稳定速度及高度波动阈值后，按当前碰撞体世界 AABB 放置推杆；目标沿 +X 移动 0.12 m、持续 1 s | 稳定等待时间、推杆是否接触、开始后位移、转角、末态稳定性 |

球形推杆为带滑动关节和位置驱动的动态刚体，不是瞬移穿过物体的运动学探针。推杆半径 0.015 m、质量 0.20 kg、初始间隙 0.03 m，驱动和接触参数对所有组一致。稳定后按各组当前包围盒放置是统一的试验规则，不读取参考几何；因此实际推杆世界坐标可能不同，逐试验记录在 `run.json`。等待至 3 s 仍未稳定时记录 `settle_timeout`，不放置或启动推杆；无推杆接触、非有限状态、MuJoCo 警告及视频失败也记为无效/失败。视频标明等待、推动、保持或等待超时阶段。

`stable` 只表示末态接地、线速度 < 0.02 m/s、角速度 < 0.2 rad/s 且末尾 0.5 s 高度范围 < 0.002 m；它不表示物体保持初始姿态。轨迹每 0.02 s 记录一次 body 位置、质心、姿态四元数、自由关节速度、接触状态和推杆位置/驱动力。MuJoCo 接触状态来自最近一次步进的碰撞计算，时间分辨率为一个时间步；角速度分量遵循 MuJoCo 自由关节约定。推动位移从推杆开始时的物体位置计算，不把此前放置阶段位移计入。

每个试验另执行 3 次不渲染的重复仿真，记录每次 `mj_step` 总墙钟时间、实际步数及有效/计划次数；等待稳定的推动试验可能长于两秒，因此同时记录每步平均耗时。渲染/编码时间单独记录。该计时受 CPU 负载、组件数、接触和模型形状共同影响，不是单独的碰撞检测或 GPU 耗时。视频为 480×360、20 fps 的离屏视图，红球为推杆；视频显示简化视觉网格，碰撞组件保持独立。

## 实际结果：三个 box 资产

修正计时后，运行 `results/physics/dynamics/debug_box_run05` 使用 Intel Core Ultra 7 265K、MuJoCo 3.3.5、trimesh 4.4.1。**27/27** 个试验当时按固定 0.6 s 等待规则记录为有效，**27/27** 个视频可解码（各 41 帧），无 MuJoCo 警告；每组不渲染计时均有 3/3 个实际保存的有效重复。该历史运行未检查推动前是否稳定，不能与现行稳定门槛混为一谈。不同来源的碰撞组件数为 Reference 1、SF3D 4、SPAR3D 52；该数量差异来自既有资产，并非本实验修改分解参数。

此前 `debug_box_run04` 的计时追加位置错误：每组虽然执行 3 次重复，只保存了最后 1 次，记录中的有效/计划数为 1/1。因此旧报告中该运行的“3/3 有效重复”及其作为三次平均的耗时表不成立；`run04` 的原始记录与视频保留，以下耗时改用修复后的 `run05`。物理轨迹和推杆接触的比较不依赖该计时均值。

| 来源 | baseline 质量 kg | mesh_fixed_mass 质量 kg | material_prior 质量 kg | 网格质心相对物体原点 mm |
| --- | ---: | ---: | ---: | --- |
| Reference | 0.100 | 0.100 | 1.800 | (0.00, 0.00, 0.00) |
| SF3D | 0.100 | 0.100 | 0.922 | (1.66, -0.34, -1.51) |
| SPAR3D | 0.100 | 0.100 | 0.656 | (6.08, 2.22, -2.29) |

按均匀密度估计的网格体积分别约为 0.003000、0.001537、0.001093 m³；不同质量主要反映归一化网格体积和假设密度，不能说明真实物体质量。完整张量及假设保存在 `run.json`。

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

全部 9/9 落体试验首次接地均约 0.176 s，与 0.15 m 理想自由落体约 0.175 s 接近；该结果验证本设置下的初始距离和重力，不验证碰撞后行为。所有 9/9 放置试验在 2 s 末满足 `stable`，但 baseline 组相对初始姿态的转角约为 Reference 0°、SF3D 22.1°、SPAR3D 75.3°；稳定并不意味着保持原朝向。SF3D baseline 推动末态未达到稳定阈值，是观察结果而不是被丢弃的失败样本。

在此推杆和摩擦条件下，较大的密度先验质量对应较小的推动位移；SF3D 固定质量下改变质心和惯性也改变了动态结果。SPAR3D 的成本更高同时伴随更多凸组件和不同几何，不能仅凭本实验断定组件数是唯一原因。初次 `run01` 的推杆按初始位置布置，SPAR3D 放置后偏移导致 3 组推杆均未接触，不能计为有效推动；该记录保留。`run05` 按固定等待后的包围盒布置推杆，9/9 推动试验发生接触，但当时没有验证推动前稳定。

本结果只覆盖一个物体类别、三个几何来源和三组人为假设；尚未校验真实质量、材料、尺度、摩擦、自相交、长时间稳定性或真实机器人交互。Task 4A 中生成资产相对参考的姿态仍未验证，跨来源差异不能直接解释为生成模型质量排名。

自动化验证在 `asset2sim-physics` 环境执行 `python -m pytest -q tests/test_dynamics.py tests/test_physics_batch.py tests/test_physics.py tests/test_collision.py tests/test_geometry_prepare.py`，43 项通过，覆盖解析盒子物性、非水密拒绝、固定碰撞网格、落体时间、推杆接触、缺失资产记录，以及重复计时和失败计数。既有模型环境此前执行 `python -m pytest -q --ignore=tests/test_dynamics.py --ignore=tests/test_physics.py --ignore=tests/test_physics_batch.py`，59 项通过。

## SPAR3D 点云局部编辑：Ring / Cup 探索

官方 SPAR3D `run_image(..., pointcloud=...)` 支持输入 XYZRGB 点云；这次直接使用 Task 4A 的两个 `points.ply` 与当时保存的预处理输入图，未采用 Mesh 编辑，也未训练模型。对每个样例，保留原始点云重新生成一份 **control**，再把点云 X 坐标超过第 80 百分位的区域按平滑权重向 +X 移动，最大位移为模型坐标 0.12。两组使用相同图片、权重、随机种子 0 和生成设置；控制组用于区分点云条件重新生成本身与编辑影响。Ring、Cup 分别修改 103/512 个点；其余点坐标的最大变化均为 0。

复现（使用新目录名）：

```bash
conda activate asset2sim-spar3d
python -m asset2sim.generation.spar3d_edit --source data/generated/spar3d/debug/physics_task4a_run01 \
  --mode control --output data/generated/spar3d/editing/<control_run>
python -m asset2sim.generation.spar3d_edit --source data/generated/spar3d/debug/physics_task4a_run01 \
  --mode edited --output data/generated/spar3d/editing/<edited_run>

conda activate asset2sim-physics
asset2sim physics-batch --manifest data/manifests/debug.jsonl \
  --run data/generated/spar3d/editing/<control_run> --sample-id debug_ring_v0 \
  --sample-id debug_cup_v0 --condition C0 --condition C2 --merge-coincident \
  --remove-degenerate --remove-unreferenced --output results/physics/editing/<control_batch>
asset2sim physics-batch --manifest data/manifests/debug.jsonl \
  --run data/generated/spar3d/editing/<edited_run> --sample-id debug_ring_v0 \
  --sample-id debug_cup_v0 --condition C0 --condition C2 --merge-coincident \
  --remove-degenerate --remove-unreferenced --output results/physics/editing/<edited_batch>
python -m asset2sim.simulation.point_edit --control results/physics/editing/<control_batch> \
  --edited results/physics/editing/<edited_batch> --output results/physics/editing/<comparison_run>
```

物理比较将编辑后的准备结果通过已记录的两次归一化矩阵转换回 **control 的物理坐标系**，避免两组各自居中和缩放掩盖或放大编辑；不使用 Ground Truth 拟合。转换后重新构建 C2，原资产及 Task A/B 结果不覆盖。编辑区域用点云阈值通过 control 的轴变换和缩放映射到物理 X 轴。以下 Mesh 距离是 control 表面向 edited 表面的单向最近**采样点**距离，每侧 20,000 点，固定种子；区域按 control 采样点划分。它衡量变化而非精准的逐点对应：对同一未编辑 Mesh 使用两组独立采样时，Ring 的编辑/非编辑区域均值约 0.97/0.96 mm，Cup 约 1.36/1.37 mm。

| 样例 | 控制 → 编辑：C2 组件 | 编辑区域距离 mm（样本数） | 非编辑区域距离 mm（样本数） | 对自身输入 Mesh 的 C2 表面距离 mm |
| --- | --- | --- | --- | --- |
| Ring | 20 → 20 | 2.90（4471） | 2.00（15529） | 2.84 → 3.00 |
| Cup | 37 → 36 | 4.06（4471） | 3.57（15529） | 4.37 → 3.91 |

点云非编辑区域保持不代表 Mesh 非编辑区域完全不变：两个样例的非编辑区域距离均高于对应的独立采样基线，说明重建结果存在区域外变化；以上数据不足以证明变化只由局部编辑导致，也不能直接从采样距离推断碰撞误差。

Ring 使用 X=0 和 X=0.07 m 的两条球探针轨迹、共 162 个位置，以各自**输入 Mesh** 的有符号距离和球半径计算参考交叠，再与 MuJoCo C2 接触比较。control 的 TP/TN/FP/FN 为 15/147/0/0，edited 为 16/146/0/0；两组均为 162/162 有效点。这是相对自身 Mesh 的有限轨迹检查，不是相对真实物体的接触准确率。Cup 两组均有多个连通分量、状态为 `needs_repair`，符号距离不作为可靠真值，接触准确率标记不可用。

动态对照均使用质量 0.1 kg、质心 `(0,0,0)` m、惯性对角线 `(0.0004,0.0004,0.0004)` kg·m²，以及同一摩擦、推杆和求解器参数；这些是人为固定的控制参数，不代表真实物性。各样例/条件执行落体与推动，共 8 个试验和 8 个可解码视频，**7/8 动态试验有效**。Ring 推动两组分别位移 8.58/8.64 cm，推杆接触 499/440 步，末态均满足稳定阈值。Cup control 推动位移 7.79 cm、接触 454 步；edited 推杆接触 0 步，试验记为 `pusher_no_contact`，其 5.98 cm 位移不能解释为有效推动结果；edited 落体末态也未满足稳定阈值。初次 `comparison_run01` 在导入 MuJoCo 前未设置 EGL，渲染初始化异常退出；已修复导入顺序，记录保留。最终 `comparison_run04` 将 Cup 标为 `completed_with_failures`，退出码为 1，逐试验记录 `pusher_no_contact`；Ring 完成。

实际结果位于本地 `results/physics/editing/comparison_run04/`，含变换、C2 结果、碰撞计数、八份轨迹与八个视频；结果目录被 Git 忽略。自动化命令 `python -m pytest -q tests/test_point_edit.py tests/test_dynamics.py tests/test_physics.py` 共 18 项通过，覆盖局部点保持、共享坐标、固定物性及已有动力学接口。当前仅有两个样例、两条接触轨迹和一次动态演示，尚未证明编辑因果性、材料真实性或普遍的交互一致性。

## Ring 缩孔目标与固定探针对照

在上一节点云控制组的基础上，`ring_hole` 使用 Ring 点云自身 PCA 的最小方差轴作为孔轴，以点云坐标中位数为轴心。径向距离小于第 25 百分位的点缩至原半径的 65%；25%–65% 分位之间平滑过渡，65% 以上不移动。原始点云、原始 Mesh 和既有运行均保留。此操作是 SPAR3D 中间点云编辑后用官方接口重建，不是直接 Mesh 编辑。

```bash
conda activate asset2sim-spar3d
python -m asset2sim.generation.spar3d_edit --source data/generated/spar3d/debug/physics_task4a_run01 \
  --mode edited --edit ring_hole --sample-id debug_ring_v0 \
  --output data/generated/spar3d/editing/<ring_edit_run>
conda activate asset2sim-physics
asset2sim physics-batch --manifest data/manifests/debug.jsonl \
  --run data/generated/spar3d/editing/<ring_edit_run> --sample-id debug_ring_v0 \
  --condition C2 --merge-coincident --remove-degenerate --remove-unreferenced \
  --output results/physics/editing/<ring_batch>
python -m asset2sim.simulation.ring_hole --control results/physics/editing/control_run01 \
  --edited results/physics/editing/<ring_batch> \
  --points data/generated/spar3d/debug/physics_task4a_run01/debug_ring_v0/points.ply \
  --output results/physics/editing/<ring_comparison>
python -m asset2sim.simulation.ring_visualization --comparison results/physics/editing/<ring_comparison> \
  --output results/physics/editing/<ring_comparison>/probe_comparison.mp4
conda activate asset2sim-sf3d
PYOPENGL_PLATFORM=egl python -m asset2sim.evaluation.ring_visualization \
  --control-mesh data/generated/spar3d/editing/control_run01/debug_ring_v0/mesh.glb \
  --edited-mesh data/generated/spar3d/editing/<ring_edit_run>/debug_ring_v0/mesh.glb \
  --control-batch results/physics/editing/control_run01 \
  --comparison results/physics/editing/<ring_comparison> \
  --output results/physics/editing/<ring_comparison>/mesh_collision_comparison.png
```

控制组的生成和物理准备见上一节；实际新增运行分别为 `ring_hole_run01`、`ring_hole_batch_run01` 和 `ring_hole_comparison_run01`。点云内侧 128 个点的实际径向比例均值为 0.650000，最大比例误差约 3.6e-8；外侧 179 个点坐标未改变，中间过渡区按权重变化。两个点云共 512 点，目标是点云内侧比例，不保证 Mesh 孔径以相同比例变化。

Mesh 在共同坐标中，孔轴中心最近表面间隙由 49.05 mm 变为 39.26 mm，比例约 0.80；这不是严格测得的全孔径，也不能将点云 0.65 比例直接视为该间隙的精确目标。控制表面到编辑表面的近似采样距离：内侧 2.71 mm（16389 点）、外侧 1.37 mm（3611 点）；同一控制 Mesh 的独立采样基线分别为 0.96 和 0.95 mm。采样距离提示孔外区域亦有变化，但包含采样误差及 SPAR3D 的全局重建影响。

以点云自身孔轴将两个资产同时刚性摆到统一测试坐标，未使用参考几何拟合。相同球探针半径 **44 mm**、相同驱动、位姿和摩擦：控制组 3/3 次通过，缩孔组 3/3 次被阻挡。这个半径是在查看重建后中心间隙后选定的探索性设置，而非预注册结果。沿固定轴线 161 个有效位置，以各自输入 Mesh 的交叠标签为参考，控制组 TP/TN/FP/FN = 0/161/0/0；缩孔组为 23/132/6/0。两组 C2 凸组件分别为 20、21 个；缩孔组存在 6 个碰撞体额外阻挡点，故不能宣称碰撞完全正确。这些标签来自各自生成 Mesh，不代表真实物体真值。

`mesh_collision_comparison.png` 上排使用原始 GLB 的 PBR 材质和纹理，下排逐组件着色展示 C2；`probe_comparison.mp4` 并排展示同一探针的实际通过/阻挡。MuJoCo 交互视频使用为碰撞准备导出的简化灰色视觉网格，未保留原始纹理；不能把静态 PBR 效果当作仿真材质一致性。两个文件及场景、逐点和试验记录均在被忽略的 `results/physics/editing/ring_hole_comparison_run01/`。

### Cup 失败排查

上一节 Cup control 准备后有 2 个闭合分量，三角面数分别为 70,044 和 3,472；edited 有 3 个，分别为 73,136、3,784 和 24。较大分量不能仅凭坐标重复认定为纹理接缝并删除；未做自动拓扑修复，接触真值仍不可用。

根据 `comparison_run04` 推动轨迹，每 0.02 s 将球心变换到物体坐标，计算其到任一 C2 组件表面的最近距离：control 71 个采样中最小为 14.55 mm，edited 为 17.62 mm，球半径 15 mm；MuJoCo 全部 1000 步中 edited 的推杆接触数为 0。该诊断支持“轨迹未触及组件”，但采样距离不能证明所有连续时刻的最小距离；AABB 中心放置不保证推杆对准凹形实体，物体稳定过程的移动也会影响轨迹。复现：

视频与轨迹核对表明，edited Cup 在推杆出现前已经发生真实的模拟运动：约 0.40 s 时相对初始姿态转过 0.384 rad；0.60 s 推杆才从场景外移至测试起点，此时转角约 0.476 rad，线速度约 0.061 m/s、角速度约 0.606 rad/s，均超过现有稳定阈值。推杆随后始终未接触 Cup，因此视频中的倾倒不能归因于推杆。旧实现只是固定等待 0.6 s，未检查稳定且视频没有阶段标识；推杆开始前不可见是因为它位于场景外。此视频不能证明推动效果；尚未确定物体自行倾倒的唯一几何或数值原因。

修复后重新执行 `python -m asset2sim.simulation.point_edit --control results/physics/editing/control_run01 --edited results/physics/editing/edited_run01 --output results/physics/editing/comparison_settled_run01`：Ring control/edited 分别在 0.684/0.692 s 满足稳定条件并接触推杆；Cup control 在 1.784 s 后推动且有 473 个接触步，Cup edited 在 3.0 s 仍未稳定，标记 `settle_timeout`，推杆没有启动。四个推动试验中 3/4 有效；程序按失败返回非零退出码。新视频和轨迹位于该目录，旧运行不修改。

```bash
conda activate asset2sim-physics
python -m asset2sim.simulation.cup_diagnostics --comparison results/physics/editing/comparison_run04 \
  --control results/physics/editing/control_run01
```

Cup edited 推动仍为无效试验，未计入成功；尚未建立可信的 Cup 拓扑修复或固定接触位置。Ring 缩孔测试属于单样例、后选探针、单一编辑强度，尚需固定未参与选参的探针和更多样例复核。`python -m pytest -q tests/test_point_edit.py tests/test_dynamics.py tests/test_physics.py` 的 19 项测试通过。
