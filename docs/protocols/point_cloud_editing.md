# SPAR3D 点云局部编辑与物理交互

## 实验配置与运行

### Ring 与 Cup 局部位移

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

物理比较将编辑后的准备结果通过已记录的两次归一化矩阵转换回 **control 的物理坐标系**，避免两组各自居中和缩放掩盖或放大编辑；不使用 Ground Truth 拟合。转换后重新构建 C2，原资产及 Task A/B 结果不覆盖。编辑区域用点云阈值通过 control 的轴变换和缩放映射到物理 X 轴。以下 Mesh 距离是 control 表面向 edited 表面的单向最近**采样点**距离，每侧 20,000 点，固定种子；区域按 control 采样点划分。该指标衡量表面变化，采样基线为：对同一未编辑 Mesh 使用两组独立采样时，Ring 的编辑/非编辑区域均值约 0.97/0.96 mm，Cup 约 1.36/1.37 mm。

### Ring 缩孔与固定探针

在上一节点云控制组的基础上，`ring_hole` 使用 Ring 点云自身 PCA 的最小方差轴作为孔轴，以点云坐标中位数为轴心。径向距离小于第 25 百分位的点缩至原半径的 65%；25%–65% 分位之间平滑过渡，65% 以上不移动。原始点云、原始 Mesh 和既有运行均保留。该操作编辑 SPAR3D 中间点云，再通过官方接口重建 Mesh。

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

### Cup 推动诊断

```bash
conda activate asset2sim-physics
python -m asset2sim.simulation.cup_diagnostics --comparison results/physics/editing/comparison_run04 \
  --control results/physics/editing/control_run01
```

## 实验结论

### 局部位移与动力学

| 样例 | 控制 → 编辑：C2 组件 | 编辑区域距离 mm（样本数） | 非编辑区域距离 mm（样本数） | 对自身输入 Mesh 的 C2 表面距离 mm |
| --- | --- | --- | --- | --- |

| Ring | 20 → 20 | 2.90（4471） | 2.00（15529） | 2.84 → 3.00 |
| Cup | 37 → 36 | 4.06（4471） | 3.57（15529） | 4.37 → 3.91 |

两个样例的 Mesh 非编辑区域距离均高于独立采样基线，重建结果在编辑区域外也出现变化。采样距离反映表面变化，碰撞误差通过接触实验另行测量。

Ring 使用 X=0 和 X=0.07 m 的两条球探针轨迹、共 162 个位置，以各自**输入 Mesh** 的有符号距离和球半径计算参考交叠，再与 MuJoCo C2 接触比较。control 的 TP/TN/FP/FN 为 15/147/0/0，edited 为 16/146/0/0；两组均为 162/162 有效点。该结果衡量有限轨迹上碰撞组件与各自输入 Mesh 的一致性。Cup 两组均有多个连通分量、状态为 `needs_repair`，符号距离不作为可靠真值，接触准确率标记不可用。

动态对照均使用质量 0.1 kg、质心 `(0,0,0)` m、惯性对角线 `(0.0004,0.0004,0.0004)` kg·m²，以及同一摩擦、推杆和求解器参数；这些参数为实验控制值。各样例/条件执行落体与推动，共 8 个试验和 8 个可解码视频，**7/8 动态试验有效**。Ring 推动两组分别位移 8.58/8.64 cm，推杆接触 499/440 步，末态均满足稳定阈值。Cup control 推动位移 7.79 cm、接触 454 步；edited 推杆接触 0 步，试验记为 `pusher_no_contact`，记录的自行位移为 5.98 cm；edited 落体末态也未满足稳定阈值。初次 `comparison_run01` 在导入 MuJoCo 前未设置 EGL，渲染初始化异常退出；已修复导入顺序，记录保留。最终 `comparison_run04` 将 Cup 标为 `completed_with_failures`，退出码为 1，逐试验记录 `pusher_no_contact`；Ring 完成。

实际结果位于本地 `results/physics/editing/comparison_run04/`，含变换、C2 结果、碰撞计数、八份轨迹与八个视频；结果目录被 Git 忽略。自动化命令 `python -m pytest -q tests/test_point_edit.py tests/test_dynamics.py tests/test_physics.py` 共 18 项通过，覆盖局部点保持、共享坐标、固定物性及已有动力学接口。当前仅有两个样例、两条接触轨迹和一次动态演示，尚未证明编辑因果性、材料真实性或普遍的交互一致性。

### Ring 缩孔目标与交互

控制组的生成和物理准备见实验配置；实际新增运行分别为 `ring_hole_run01`、`ring_hole_batch_run01` 和 `ring_hole_comparison_run01`。点云内侧 128 个点的实际径向比例均值为 0.650000，最大比例误差约 3.6e-8；外侧 179 个点坐标未改变，中间过渡区按权重变化。两个点云共 512 点，目标针对点云内侧径向比例，Mesh 孔径变化见下文测量。

Mesh 在共同坐标中，孔轴中心最近表面间隙由 49.05 mm 变为 39.26 mm，比例约 0.80；该值描述孔轴中心最近表面间隙，点云目标比例为 0.65。控制表面到编辑表面的近似采样距离：内侧 2.71 mm（16389 点）、外侧 1.37 mm（3611 点）；同一控制 Mesh 的独立采样基线分别为 0.96 和 0.95 mm。采样距离提示孔外区域亦有变化，但包含采样误差及 SPAR3D 的全局重建影响。

以点云自身孔轴将两个资产同时刚性摆到统一测试坐标，未使用参考几何拟合。相同球探针半径 **44 mm**、相同驱动、位姿和摩擦：控制组 3/3 次通过，缩孔组 3/3 次被阻挡。探针半径根据重建后的中心间隙选定，用于探索性对照。沿固定轴线 161 个有效位置，以各自输入 Mesh 的交叠标签为参考，控制组 TP/TN/FP/FN = 0/161/0/0；缩孔组为 23/132/6/0。两组 C2 凸组件分别为 20、21 个；缩孔组有 6 个碰撞体额外阻挡点。参考标签来自各自生成 Mesh。

`mesh_collision_comparison.png` 上排使用原始 GLB 的 PBR 材质和纹理，下排逐组件着色展示 C2；`probe_comparison.mp4` 并排展示同一探针的实际通过/阻挡。MuJoCo 交互视频使用为碰撞准备导出的简化灰色视觉网格，未保留原始纹理，静态图与仿真视频分别展示原始材质和碰撞准备网格。两个文件及场景、逐点和试验记录均在被忽略的 `results/physics/editing/ring_hole_comparison_run01/`。

### Cup 拓扑和推动失败排查

上一节 Cup control 准备后有 2 个闭合分量，三角面数分别为 70,044 和 3,472；edited 有 3 个，分别为 73,136、3,784 和 24。较大分量需要结合对象结构检查；当前保留这些分量，接触真值标记为不可用。

根据 `comparison_run04` 推动轨迹，每 0.02 s 将球心变换到物体坐标，计算其到任一 C2 组件表面的最近距离：control 71 个采样中最小为 14.55 mm，edited 为 17.62 mm，球半径 15 mm；MuJoCo 全部 1000 步中 edited 的推杆接触数为 0。该诊断显示采样时刻的推杆与组件间距；推杆按 AABB 中心布置，凹形结构和物体稳定过程的移动会影响实际接触。

视频与轨迹核对表明，edited Cup 在推杆出现前已经发生真实的模拟运动：约 0.40 s 时相对初始姿态转过 0.384 rad；0.60 s 推杆才从场景外移至测试起点，此时转角约 0.476 rad，线速度约 0.061 m/s、角速度约 0.606 rad/s，均超过现有稳定阈值。推杆随后始终未接触 Cup，倾倒发生在推杆启动之前。旧实现只是固定等待 0.6 s，未检查稳定且视频没有阶段标识；推杆开始前不可见是因为它位于场景外。自行倾倒的具体几何或数值原因仍待分析。

修复后重新执行 `python -m asset2sim.simulation.point_edit --control results/physics/editing/control_run01 --edited results/physics/editing/edited_run01 --output results/physics/editing/comparison_settled_run01`：Ring control/edited 分别在 0.684/0.692 s 满足稳定条件并接触推杆；Cup control 在 1.784 s 后推动且有 473 个接触步，Cup edited 在 3.0 s 仍未稳定，标记 `settle_timeout`，推杆没有启动。四个推动试验中 3/4 有效；程序按失败返回非零退出码。新视频和轨迹位于该目录，旧运行不修改。

Cup edited 推动仍为无效试验，未计入成功；尚未建立可信的 Cup 拓扑修复或固定接触位置。Ring 缩孔测试属于单样例、后选探针、单一编辑强度，尚需固定未参与选参的探针和更多样例复核。`python -m pytest -q tests/test_point_edit.py tests/test_dynamics.py tests/test_physics.py` 的 19 项测试通过。
