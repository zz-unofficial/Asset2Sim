# 机械臂抓取与交互式资产仿真

## 实验配置与运行

### 固定抓取评测

#### 场景与复现

使用 MuJoCo Menagerie 的 UR5e 与 Robotiq 2F-85，分别遵循上游 BSD-3-Clause 和 BSD-2-Clause 许可。物体是自由刚体；手指驱动、接触和摩擦均由 MuJoCo 仿真，不以约束连接夹爪和物体。机械臂与夹爪之间的固定安装连接及夹爪内部联动约束保留。

在仓库根目录执行，前提是已按 Task 4A 准备 `configs/physics/grasp.yaml` 中列出的三个 batch 和 C2 碰撞组件：

```bash
conda activate asset2sim-physics
bash scripts/setup_menagerie.sh
asset2sim grasp-eval --config configs/physics/grasp.yaml \
  --output results/physics/grasp/debug_box_run03
```

输出目录必须不存在；重复实验请更换名称。安装脚本获取指定版本的上游两个模型到 `.local/mujoco_menagerie`。加载前核对模型版本、批次状态及生成资产的来源索引。每个来源保存可独立重载的 `model/scene.xml` 与相对路径资产、三次 `trajectory.csv`、首轮 `grasp.mp4`，以及公共 `run.json` 和 `summary.csv`。输入缺失或加载失败会写入失败记录，并保留计划次数作为成功率分母。原始生成网格、准备网格及碰撞组件均不改动。

#### 设置与判定

采用 `debug_box_v0` 的 Reference、SF3D、SPAR3D 的 Task 4A C2 结果，统一缩放 0.5，最长边均为 0.10 m；沿夹持方向的初始包围盒宽度分别为 50.0、71.9、70.9 mm，均小于 2F-85 的标称 85 mm 开口。不同来源保持各自生成朝向，不进行参考网格对齐。质量均设为 0.1 kg，摩擦参数一致；质心和惯量使用各自缩放网格的均匀 AABB 先验估计，采用实验物性先验。三个场景使用相同的物体初始水平位置和预设末端位姿。碰撞组件分别加载，数量为 1、4、52。

脚本先等待 1 s，再依次接近 2 s、闭合 2 s、抬升 2 s、保持 1 s、释放 1 s；使用逆运动学确定预设目标，采用预设抓取路径。仿真步长 0.002 s。保持阶段至少 80% 的 50 Hz 样本需同时满足：物体相对初始高度上升至少 0.06 m、离开桌面、两侧指垫各有至少 0.05 N 法向力。释放结束时物体需接触桌面且不接触指垫，且此前保持条件已满足；MuJoCo 数值警告导致该次判无效。日志包括各阶段位姿、接触、指垫法向力和 `mj_step` 耗时。时间只统计 4500 次 `mj_step`，不含场景构建、控制计算、采样及视频渲染。

### 键盘交互

在 Linux X11 桌面中运行以下命令；物理环境、Menagerie 和三个 Task 4A 资产与上文相同，输出目录需为新目录：

```bash
conda activate asset2sim-physics
asset2sim grasp-interactive --config configs/physics/grasp_interactive.yaml \
  --output results/physics/grasp/interactive_run01
```

在 [交互配置](../../configs/physics/grasp_interactive.yaml) 的 `sample_id` 和 `initial_source` 中设置默认物体与初始来源；也可以从命令行临时覆盖，例如：

```bash
asset2sim grasp-interactive --sample-id debug_cup_v0 --source spar3d \
  --output results/physics/grasp/interactive_cup_01
```

当前三个来源均有 `debug_box_v0`、`debug_cup_v0`、`debug_ring_v0`、`debug_bracket_v0` 的 C2 资产。`--sample-id` 对三个来源使用同一个样例 ID，`--source` 可选 `reference`、`sf3d`、`spar3d`；启动后仍可用数字键切换来源。命令行覆盖配置，选择和资产来源会记录在 `session.json` 中。物体在批次中缺失或对应碰撞资产不可用时，程序会说明失败的样例和条件。选择物体不改变其原始朝向；不同几何可能需要调整抓取姿态。

窗口启动后，机械臂通过执行器回到物体上方，终端出现 `Ready` 后开始操作。点击 MuJoCo 窗口使其获得焦点；仅在本进程的窗口拥有焦点时读取按键，失焦后不再接收移动输入。按住移动，松开停止改变目标；可同时按多个方向键。

| 按键 | 操作（世界坐标） |
| --- | --- |
| W / S | 末端 +X / −X |
| A / D | 末端 +Y / −Y |
| R / F | 末端 +Z / −Z |
| U / O | 末端绕参考姿态的 X 轴正 / 负旋转 |
| I / P | 末端绕参考姿态的 Y 轴正 / 负旋转 |
| H / L | 末端绕参考姿态的 Z 轴正 / 负旋转 |
| J / K | 闭合 / 张开夹爪 |
| 0 | 重置当前场景 |
| 1 / 2 / 3 | 切换 Reference / SF3D / SPAR3D |
| Q | 结束并保存会话 |

末端初始为向下姿态，旋转控制调整相对于该姿态的固定轴 XYZ 欧拉角；角速度及各轴角度范围由配置限制。IK 目标限制在配置的工作空间内；不可达或超关节限位的目标被拒绝并在终端及轨迹中记录。笛卡尔目标速度、关节**执行器目标**速度和夹爪命令变化速度均有上限；碰撞时的实际关节速度由仿真动力学决定。机械臂初始化、移动和夹爪动作均通过执行器，不直接写入仿真关节位置；物体仍为自由刚体。

切换和重置在同一个 Viewer 窗口内加载场景，避免反复关闭窗口造成闪屏。当前 MuJoCo 3.3.5 的公开被动 Viewer 接口不提供更换模型，加载使用该版本的内部接口；升级 MuJoCo 后需要重新验证。每次切换或重置产生独立 `segment_*`，包含 `trajectory.csv`、`segment.json` 和 `interaction.mp4`；会话目录另存使用的配置和各资产的可重载 MJCF。轨迹按 20 Hz 保存按键、目标末端位置与旋转、实际末端位置与姿态四元数、关节位置与命令、物体位置和四元数、桌面接触及双指法向力。视频使用固定观察视角，窗口中调整的视角不改变录像。关闭窗口后完成保存；录像帧率和仿真时间记录在配置中。绘制和 IK 计算可能使仿真时间慢于墙钟时间。

### 从图片或 GLB 进入交互

使用 `configs/physics/asset_interactive.yaml` 配置独立模型环境、贴图分辨率、物理尺寸、清理、碰撞和物性先验。在 `asset2sim-physics` 环境中运行，一条命令完成单图推理、物理准备和 Viewer 启动：

```bash
asset2sim asset-interactive --image /path/to/object.png --model sf3d \
  --device cpu --extent 0.10 --output results/physics/assets/my_object_01
```

`--model` 可选 `sf3d`、`spar3d`；省略 `--device` 使用模型配置的设备。模型推理由相应 Conda 环境执行，日志写入 `generation.log`。`--extent` 为物理最长边，单位米，不根据参考几何拟合。已有生成运行可以直接导入其原始 GLB：

```bash
asset2sim asset-interactive --mesh /path/to/run/sample/mesh.glb \
  --run /path/to/run --extent 0.10 --output results/physics/assets/import_01
```

无生成记录的外部 GLB 需用 `--up-axis y` 或 `--up-axis z` 声明朝向，程序记录为外部导入，并将未知原始尺度归一化到指定米制尺寸。Y-up 导入使用绕 X 轴 +90° 的变换；有生成记录时使用生成器声明的完整坐标变换。原始 GLB 不改动。Mesh 清理的每项操作写入准备记录；非水密输入允许 CoACD 显式内部预处理，配置固定为 `preprocess_mode: on`，此行为不表示原网格完全未变。

添加 `--prepare-only` 可只准备资产。之后直接打开准备好的资产，避免再次推理或分解：

```bash
asset2sim grasp-interactive --asset results/physics/assets/my_object_01 \
  --output results/physics/assets/my_object_01/manual_01
```

单资产模式支持原有位移、旋转、夹爪、0 重置及 Q 退出。数字键来源切换只用于原有三来源模式。每段交互记录双指受力、离桌抬升采样数及释放状态；抓取周期观察条件为两侧法向力均大于 0.05 N、离桌、物体原点比初始高至少 0.03 m，累计达到 0.5 s 后最终释放到桌面，且无数值警告。该指标统计单次交互中满足条件的累计采样。

输出包括 `raw/` 原图与原始 GLB、`generation/` 推理记录、`prepared/` 规范化几何与变换诊断、`collision/` 凸组件、`scene/` 可重载 MJCF 和视觉贴图、`physical_properties.json`、`asset.json`、`pipeline.json`，以及交互轨迹和视频。新运行另存解析后的 `configuration.yaml`，每阶段失败保留原因。当前物性默认使用 0.1 kg 固定质量与均匀 AABB 质心、惯量先验，作为实验中的物性估计。物体底部依据碰撞组件放到桌面上方 1 mm，检查接近位姿 IK、工作空间及 85 mm 标称开口；开口检查使用初始包围盒宽度，抓取姿态由操作人员选择。

视觉几何从原始 GLB 独立导出，应用同一物理变换，保留节点实例、UV、颜色贴图和基础颜色，近似转换粗糙度；法线贴图、金属度贴图、发光和透明混合尚未精确复现。视觉 geom 不参与碰撞，每个 CoACD 组件独立作为碰撞 geom 加载。导出后核对视觉与物理准备网格的包围盒，清理只作用于物理几何。

### 端到端验证的实际命令

输入官方示例 `third_party/stable-fast-3d/demo_files/examples/chair1.png`，实际运行命令：

```bash
asset2sim asset-interactive \
  --image third_party/stable-fast-3d/demo_files/examples/chair1.png \
  --model sf3d --device cpu --extent 0.10 --prepare-only \
  --output results/physics/assets/chair_cpu_run02
asset2sim grasp-interactive --asset results/physics/assets/chair_cpu_run02 \
  --output results/physics/assets/chair_cpu_run02/interaction_validation
```

## 实验结论

### 三来源 box 抓取

`debug_box_run03`，MuJoCo 3.3.5，Intel Core Ultra 7 265K。每个来源三次均使用完全相同的初始状态和控制，是确定性重复，并非三个独立对象或扰动试验。

| 来源 | 有效 / 计划 | 成功 / 计划 | 保持达标样本 / 次 | 最大抬升 | 接近前水平位移 | `mj_step` 平均耗时 / 4500 步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 3/3 | 3/3 | 50/50 | 108.6 mm | <0.1 mm | 0.121 s |
| SF3D | 3/3 | 3/3 | 50/50 | 103.5 mm | 18.4 mm | 0.164 s |
| SPAR3D | 3/3 | 0/3 | 0/50 | 未抬升 | 54.9 mm | 0.129 s |

Reference 和 SF3D 在保持阶段有双侧受力，释放后落回桌面。SPAR3D 在接近阶段开始前已倾倒并离开预设抓取位置，三次均未形成双侧指垫接触；该组记为抓取失败。这个结果对应当前固定的放置和抓取位姿，生成几何与碰撞表示共同影响结果。后续应增加独立对象、稳定放置与抓取位姿扰动，并在统一硬件上重复计时；本实验未验证真实材料、真机和泛化抓取能力。

### 键盘交互验证

已在本机 X11 Viewer 验证窗口启动、资产切换、重置及关闭保存。自动测试使用同一按键控制器驱动 Reference 盒子，观察到双指受力、离桌抬升、水平移动和释放落桌；验证输入由程序生成。当前键盘读取依赖 X11，原生 Wayland、Windows 和 macOS 尚未实现；SF3D/SPAR3D 的手动抓取效果也需实际操作验证。

自动输入演示保存在 `results/physics/grasp/keyboard_validation01/`，使用与交互相同的控制器、记录和渲染流程，输入由程序模拟，未经过人工键盘操作。参考盒子在 501 个采样中有 154 个样本处于双指受力且离桌状态，物体中心最高 0.168 m；移动后松开夹爪，最终重新接触桌面，双指法向力均为零。`tests/test_grasp_interactive.py` 可重复验证对应的接触和运动条件。控制键仅在本进程的 Viewer 窗口拥有焦点时由交互入口接收，避免同时触发 Viewer 的显示及几何分组快捷键。

### 新图片端到端验证

SF3D CPU 推理 32.24 s、512 贴图，生成 20,028 个三角面；C2 构建 12 个独立组件，夹持方向包围盒宽 52.5 mm，初始接近位姿 IK 通过。实际窗口显示了原始棕色纹理。交互验证采用自动 XTest 输入，轨迹和完整视频位于上述目录：有 109 个采样满足双指受力及离桌抬升条件，完成移动和释放，MuJoCo 无数值警告。椅子在等待期间发生姿态变化，随后由夹爪夹住；该实验未自动选择稳定姿态或抓取点，结果仅适用于这次交互。

失败记录也保留在 `results/physics/assets/`：SF3D 的 1024/512 GPU 尝试与 SPAR3D 低显存尝试均显存不足；早期非水密椅子分解因未显式启用内部预处理而失败。修正后已有 GLB 和新图片通路均已完成。SPAR3D 从新图片到交互的完整成功案例仍未验证，成功生成的案例使用 SF3D CPU。
