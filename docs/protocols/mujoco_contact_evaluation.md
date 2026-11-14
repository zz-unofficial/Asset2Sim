# MuJoCo 受控接触与跨来源评测

## 实验配置与运行

### 基础接触实验

#### 环境与运行

独立 CPU 环境安装官方 `mujoco==3.3.5` 和 `coacd==1.0.14`，不需要 CUDA：

```bash
bash scripts/setup_physics.sh
conda activate asset2sim-physics
asset2sim physics-eval --config configs/physics/debug.yaml --output results/physics/debug/<新run_id>
```

一条命令从仓库 Debug 参考资产重新执行 Mesh 准备、C0–C3 碰撞构建及 MuJoCo 评测。输出目录必须不存在；它保存完整配置、各阶段资产、MJCF、`summary.json`、`summary.csv`、逐点 `contact_samples.csv` 和逐次 `trajectory_trials.csv`。失败记录写入 `summary.json`，不覆盖已有运行。修改参数时复制配置并为结果指定新的 run_id。

每个 `colliders.json` 中的凸组件分别加载为同一 body 的独立 mesh geom；视觉网格单独加载且禁用碰撞。编译后逐组件将 MuJoCo 局部顶点变回物理坐标，检查所有顶点与包围盒，核对实际 geom 数量。静态对象用于比较；每个条件还构建具有自由关节、统一质量和对角惯量的动态对象，验证编译和单步运行。动态对象尚未用于动态接触性能比较。

#### 参考与指标

参考标签由原始程序几何参数独立计算，不依赖碰撞近似：圆环为 64 边环形截面，内/外半径 0.055/0.10 m、厚度 0.035 m，孔轴为 Y；归一化盒子半尺寸为 `(0.075, 0.05, 0.10)` m。球半径 0.01 m，沿 Y 从 -0.16 到 0.16 m 移动。圆环使用中心孔和实体两条轨迹；盒子使用实体和外侧无碰撞轨迹。每条轨迹均匀取 81 个位置，参考球体与实体相切的 1 μm 范围标为无效，不计入混淆矩阵。

`reference_contact` 表示球与参考实体相交；`geometric_contact` 要求 MuJoCo 产生球与凸组件的接触且 `dist <= 0`。TP/TN/FP/FN 按有效采样位置计数；FPR = FP/(FP+TN)，FNR = FN/(FN+TP)，分母为零时记为空。逐点结果同时保留最小检测距离、纯缓冲区接触和求解器活跃接触。MuJoCo 3.3.5 使用 `margin=0.004 m`、`gap=0.002 m`；正距离接触属于缓冲范围；是否进入求解器以 `efc_address` 判定。具体语义见 [MuJoCo 3.3.5 接触参数](https://mujoco.readthedocs.io/en/3.3.5/XMLreference.html)。

轨迹行为采用固定参数的滑动球探针：位置驱动目标在 900 步中渐进移动至终点，共模拟 1200 步，时间步长 0.002 s。终点距目标不超过 0.01 m 记为通过；未通过且末态无活跃接触的试验记为无效。与独立参考轨迹的“无交叠应通过、存在交叠应阻挡”比较，每条轨迹重复 3 次。质量、惯量、摩擦、求解器及接触设置对所有条件一致；具体值均在配置和结果中记录。

构建时间为 Task 2 的构建耗时；查询时间为 `mj_forward` 的墙钟时间，包含动力学和碰撞计算，涵盖完整前向计算；仿真时间为 1200 次 `mj_step` 与驱动更新的墙钟时间。每个条件独立构建 3 次，每个位置查询重复 3 次，轨迹重复 3 次；构建的每次时间以及所有重复构建产物均保留。记录单位秒、CPU 与软件版本；实际运行负载会影响计时。

### 扩展探针与 margin/gap 消融

运行多轨迹对照：

```bash
asset2sim physics-eval --config configs/physics/contact_ablation.yaml \
  --output results/physics/debug/<新run_id>
```

配置继承 `debug.yaml`，比较 `buffered`（margin/gap = 0.004/0.002 m）和 `zero`（0/0 m），同一碰撞组件及探针条件配对。圆环使用 8 种不同的 X/Z 偏移和球半径：半径 0.005、0.010、0.020 m，覆盖中心孔、偏心孔、孔边、实体和外侧。每组 648 个位置、24 次动态试验；盒子保留两条路径。每条轨迹重复三次，用于计时与重复性评估。

三类判断分别记录：`geometric_contact` 仅表示 MuJoCo 产生距离非正的接触；`solver_active` 表示至少一个接触的 `efc_address >= 0`，接触力另行记录；`behavior_consistency` 是受限 Y 轴驱动试验中通过/阻挡与独立参考相交标签的一致率，点接触分类另行统计。`buffer_only` 是所有检测距离为正，可能仍有求解器活跃约束；`inactive_only` 是所有接触的 `efc_address < 0`。MuJoCo 未报告接触时，`geometric_contact` 为假；完全嵌入等情况需结合参考几何分析。

参考球体完全位于实体内部时标为 `fully_contained`，边界相交标为 `boundary_overlap`；两类都保留在主混淆矩阵中，并分开计算有效正样本和漏检。切线附近仍按容差标记无效。轨迹参考标签由路径采样得到，动态行为另外受驱动、求解器和有限时间影响。

### 生成资产来源索引

对于已有生成运行，先使用 Task A 和 Task B 产生以下目录：

```text
<资产根目录>/prepared/<sample_id>/normalized.glb
<资产根目录>/prepared/<sample_id>/metadata.json
<资产根目录>/collisions/<sample_id>/<C0...C3>/colliders.json
```

然后运行：

```bash
asset2sim physics-index --manifest data/manifests/debug.jsonl \
  --run <SF3D或SPAR3D生成运行目录> --assets <资产根目录> \
  --output results/metrics/<新索引名>.json
```

接口按 `sample_id` 和对象/视角关联 Manifest 与生成结果，检查 Task A 来源、Z-up/米制声明及 Task B 组件来源和文件；未完成的样例保留失败原因。可用重复的 `--condition C0` 等参数选择需要的条件。索引不读取参考网格，不进行 ICP、尺度拟合或其他 Ground Truth 对齐；索引用于准备待评测资产，接触参考标签由独立协议提供。正式生成资产评测仍需另行确定独立参考与探针协议。

### 三来源小规模批处理

选择 `debug_box_v0`、`debug_cup_v0`、`debug_ring_v0`、`debug_bracket_v0`，比较 C0 和 C2。SF3D 使用 `chair_run01`，SPAR3D 使用四个样例的 `physics_task4a_run01`。从仓库根目录运行：

```bash
conda activate asset2sim-spar3d
asset2sim run --config configs/experiments/spar3d_debug.yaml \
  --sample-id debug_box_v0 --sample-id debug_cup_v0 \
  --sample-id debug_ring_v0 --sample-id debug_bracket_v0 \
  --output data/generated/spar3d/debug/physics_task4a_run01

conda activate asset2sim-physics
asset2sim physics-batch --manifest data/manifests/debug.jsonl --reference \
  --sample-id debug_box_v0 --sample-id debug_cup_v0 \
  --sample-id debug_ring_v0 --sample-id debug_bracket_v0 \
  --condition C0 --condition C2 --output results/physics/task4a/reference_run01
asset2sim physics-batch --manifest data/manifests/debug.jsonl \
  --run data/generated/sf3d/chair_run01 \
  --sample-id debug_box_v0 --sample-id debug_cup_v0 \
  --sample-id debug_ring_v0 --sample-id debug_bracket_v0 \
  --condition C0 --condition C2 --merge-coincident --remove-degenerate \
  --remove-unreferenced --output results/physics/task4a/sf3d_run01
asset2sim physics-batch --manifest data/manifests/debug.jsonl \
  --run data/generated/spar3d/debug/physics_task4a_run01 \
  --sample-id debug_box_v0 --sample-id debug_cup_v0 \
  --sample-id debug_ring_v0 --sample-id debug_bracket_v0 \
  --condition C0 --condition C2 --merge-coincident --remove-degenerate \
  --remove-unreferenced --output results/physics/task4a/spar3d_run01
```

批处理不运行生成模型，按样例记录生成、准备、索引、碰撞构建、MuJoCo 加载与接触阶段。生成资产调用现有 `physics-index` 核验来源；既有结果目录不会覆盖。原始生成 Mesh、参考 Mesh 与每阶段的 Mesh、变换、组件和失败原因均保留；批次的 `summary.csv` 是逐样例/条件账本，`batch.json` 还包含配置与运行来源。需要复用已有生成 Mesh 时指定 `--run`；参考对象使用 `--reference`。生成资产的三项清理操作均由命令显式启用，操作记录与每项实际删除数量见准备结果的 `metadata.json`；两个模型各四个样例均合并了完全相同坐标的顶点，退化面及未引用顶点删除数均为 0。合并可能连接原本分离但恰好接触的部件。

### 离线 Oracle 对齐诊断

为了获得一个完整三来源配对的**离线诊断**，对 `debug_box_v0` 另行运行以下两条命令。它们使用参考网格进行 90° 旋转搜索和刚体 ICP，只改变诊断场景的 body 变换，不改变主流程网格和碰撞组件；不执行额外尺度拟合。该诊断使用参考拟合，结果与主流程指标分别记录。

```bash
asset2sim physics-oracle --batch results/physics/task4a/sf3d_run01 \
  --manifest data/manifests/debug.jsonl --sample-id debug_box_v0 \
  --condition C0 --condition C2 --output results/physics/task4a/oracle_sf3d_box_run01
asset2sim physics-oracle --batch results/physics/task4a/spar3d_run01 \
  --manifest data/manifests/debug.jsonl --sample-id debug_box_v0 \
  --condition C0 --condition C2 --output results/physics/task4a/oracle_spar3d_box_run01
```

## 实验结论

### 圆环与盒子基础接触

`results/physics/debug/run05` 在 Intel Core Ultra 7 265K 上完成。圆环各条件均为 162/162 有效接触点、6/6 有效轨迹试验；构建时间为 3 次均值：

| 条件 | 组件 | TP / TN / FP / FN | FPR / FNR | 轨迹一致率 | 构建时间（秒） |
| --- | ---: | --- | --- | ---: | ---: |
| C0 | 1 | 13 / 137 / 12 / 0 | 8.05% / 0% | 3/6 | 0.0011 |
| C1 | 7 | 13 / 149 / 0 / 0 | 0% / 0% | 6/6 | 1.06 |
| C2 | 12 | 13 / 149 / 0 / 0 | 0% / 0% | 6/6 | 1.27 |
| C3 | 16 | 13 / 149 / 0 / 0 | 0% / 0% | 6/6 | 1.40 |

C0 的凸包覆盖孔洞，使探针在孔内被错误阻挡；C1–C3 在所测轨迹上通过孔洞并阻挡实体。盒子的四个条件均为一个组件，160/162 个有效点中 TP/TN/FP/FN 为 28/131/0/1，FNR 为 1/29，轨迹一致率为 6/6。漏报发生在球心位于盒子中心、球完全嵌入盒子时，该采样位置未返回接触记录。圆环和盒子结果覆盖指定的探针轨迹。`run01` 使用过快的目标跳变导致探针穿透，已保留作调试记录；正式比较使用渐进驱动的 `run05`。

### 扩展探针与 margin/gap 消融

本机 `contact_ablation_run02` 的圆环各方案每种设置均有 648/648 个有效位置、24/24 次有效试验，逐轨迹数据保留在原始运行目录中：

| 方案 | 几何 FP / FN | FPR / FNR | 行为一致率 | buffered 活跃 / 仅非活跃位置 | zero 活跃 / 仅非活跃位置 |
| --- | --- | --- | --- | --- | --- |
| C0 | 64 / 0 | 10.29% / 0% | 9/24 | 102 / 2 | 90 / 0 |
| C1 | 13 / 0 | 2.09% / 0% | 24/24 | 43 / 13 | 39 / 0 |
| C2 | 0 / 0 | 0% / 0% | 21/24 | 39 / 4 | 26 / 0 |
| C3 | 0 / 0 | 0% / 0% | 21/24 | 39 / 4 | 26 / 0 |

这两个 margin/gap 设置在这批位置上几何分类与通过/阻挡结论相同，活跃和缓冲接触计数不同；其他轨迹和动力学设置可继续比较。C1 的 13 个误报来自斜向偏心孔轨迹，但驱动探针最后通过；C2/C3 的孔边轨迹与参考有相交，静态采样也检出接触，动态探针仍通过。这些结果展示了几何检测与特定驱动下的行为可能不同，探针通过的具体动力学原因仍需进一步分析。

盒子每种方案和设置有 160/162 个有效位置，其中完全嵌入 19 个、漏检 1 个；边界相交 10 个、漏检 0 个。中心漏检样本保留在原始运行的逐点记录中。这项结果对应当前盒子、探针和采样位置。

实验结论记录在本文；完整数据保留在 `results/` 下的原始运行目录，新增结果文件由 Git 忽略。保存的 MJCF 把各组件复制到与场景同目录的相对路径资源目录，整个场景目录移动后可重新加载。

### 三来源阶段结果

#### 全部样例的阶段结果

8/8 个生成网格存在且生成成功，12/12 个来源样例完成物理准备，24/24 个来源/条件完成碰撞构建和 MuJoCo 加载，生成资产的 16/16 个来源/条件通过索引校验；这些分母包含全部指定样例。下表每格为“组件数 / 相对自身输入网格的对称采样表面距离（mm）”。三个来源的拓扑为 `ready`，SPAR3D 杯子例外，标为 `needs_repair`（存在多个连通分量），虽然其两组碰撞体均能构建和加载，仍需人工检查。无生成、准备、碰撞或加载失败，亦无未找到的指定样例。

| 样例 | 来源 | 准备拓扑 | C0 组件 / 距离 | C2 组件 / 距离 | 索引与 MuJoCo |
| --- | --- | --- | ---: | ---: | --- |
| box | Reference | ready | 1 / 1.80 | 1 / 1.79 | 不需生成索引；加载成功 |
| box | SF3D | ready | 1 / 1.74 | 4 / 4.65 | 通过；加载成功 |
| box | SPAR3D | ready | 1 / 5.94 | 52 / 4.04 | 通过；加载成功 |
| cup | Reference | ready | 1 / 7.28 | 18 / 3.48 | 不需生成索引；加载成功 |
| cup | SF3D | ready | 1 / 7.28 | 31 / 4.98 | 通过；加载成功 |
| cup | SPAR3D | needs_repair | 1 / 7.88 | 39 / 4.53 | 通过；加载成功 |
| ring | Reference | ready | 1 / 3.93 | 12 / 2.54 | 不需生成索引；加载成功 |
| ring | SF3D | ready | 1 / 4.80 | 17 / 3.01 | 通过；加载成功 |
| ring | SPAR3D | ready | 1 / 4.28 | 19 / 2.87 | 通过；加载成功 |
| bracket | Reference | ready | 1 / 9.70 | 2 / 1.17 | 不需生成索引；加载成功 |
| bracket | SF3D | ready | 1 / 8.11 | 3 / 1.29 | 通过；加载成功 |
| bracket | SPAR3D | ready | 1 / 8.06 | 3 / 1.20 | 通过；加载成功 |

准备阶段 12/12 成功，其中 SPAR3D 杯子的拓扑状态为 `needs_repair`。每个生成样例执行 3 项显式清理，参考资产不做清理。局部表面误差按种子固定的 10,000 个点/侧计算，包含组件重叠内部面及采样误差，结果随组件内部面和采样变化。建造时间、MuJoCo 查询/仿真墙钟时间及条件参数在批次账本中；每个盒子/圆环、每个条件各有 162 个查询位置与 6 次轨迹试验，实际动态试验 72/72 有效且未报告求解器警告。这四个 Debug 样例没有配对的外观或论文几何指标，因此未分析外观关联。

#### 坐标、未评估项和独立 Oracle 诊断

主流程仅使用模型声明的原始轴约定及各自最长边 0.20 m 的独立归一化；未用参考几何执行 ICP、尺度拟合或其他对齐。生成资产与参考实体的相对姿态尚未校验，因此生成 box/ring 的主流程端到端 FP/FN 及行为一致率为每条件 **0 个有效 / 162 个计划采样点**，未计算准确率。其原始探针响应与 6/6 次动态试验仅作各资产自身坐标系的观测。参考 box/ring 有独立程序几何标签，各自 C0/C2 共 4/4 条件可评估：box 每条件 160/162 有效位置（C0 和 C2 均 TP/TN/FP/FN = 28/131/0/1，6/6 轨迹一致）；ring 每条件 162/162 有效位置（C0 为 13/137/12/0、3/6 一致；C2 为 13/149/0/0、6/6 一致）。cup 和 bracket 的参考实体与交互任务尚无经过验证的接触标签，三个来源的 12 个条件均标为 `reference_labels_unavailable`，未套用圆环真值。生成姿态未对齐与参考标签缺失是不同原因。

### Oracle 诊断结果

对齐拟合每侧 1,200 个表面点，固定种子 0/1；独立评估每侧 5,000 点、种子 2/3，距离单位为米。独立采样的对称表面均距：SF3D 从 18.57 mm 变为 15.78 mm，SPAR3D 从 20.60 mm 变为 21.10 mm。SPAR3D 的独立评估距离未改善，因此此拟合结果的可靠性尤其有限；该拟合结果适合结合更多对象检查。

| 来源 / 诊断 | 条件 | 凸组件 | TP / TN / FP / FN | 有效点 / 计划 | 行为一致 / 有效试验 |
| --- | --- | ---: | --- | --- | --- |
| Reference（主流程） | C0 | 1 | 28 / 131 / 0 / 1 | 160 / 162 | 6 / 6 |
| Reference（主流程） | C2 | 1 | 28 / 131 / 0 / 1 | 160 / 162 | 6 / 6 |
| SF3D（Oracle） | C0 | 1 | 27 / 131 / 0 / 2 | 160 / 162 | 6 / 6 |
| SF3D（Oracle） | C2 | 4 | 27 / 131 / 0 / 2 | 160 / 162 | 6 / 6 |
| SPAR3D（Oracle） | C0 | 1 | 29 / 124 / 7 / 0 | 160 / 162 | 6 / 6 |
| SPAR3D（Oracle） | C2 | 52 | 17 / 124 / 7 / 12 | 160 / 162 | 6 / 6 |

SF3D 的两个条件在这些点上相同，两个漏检均为边界相交；SPAR3D 的 C0/C2 共同出现 7 个误报，而 C2 比 C0 多出的 12 个漏报全部属于球完全嵌入参考盒子的 19 个采样点，边界相交的 10 个点没有漏报。MuJoCo 在完全嵌入时出现过漏报；分析这些差异时需结合生成几何、分解和碰撞查询。单凸包填充空间也可能掩盖生成几何缺失。所有 24/24 次 Oracle 动态试验有效且无求解器警告。结合可信姿态、更多物体与边界接触轨迹，可进一步分析生成几何、分解误差和物理行为之间的关系。

自动化验证：`asset2sim-physics` 环境执行 `python -m pytest -q tests/test_physics_batch.py tests/test_physics.py tests/test_collision.py tests/test_geometry_prepare.py`，37 项通过；`asset2sim-sf3d` 环境执行 `python -m pytest -q --ignore=tests/test_physics.py --ignore=tests/test_physics_batch.py`，59 项通过。前者覆盖缺失生成记录、参考接触、无参考对齐时禁止准确率、Oracle 场景及现有几何接口。
