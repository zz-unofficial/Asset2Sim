# UR5e 双指抓取验证

## 场景与复现

使用 MuJoCo Menagerie 的 UR5e 与 Robotiq 2F-85，分别遵循上游 BSD-3-Clause 和 BSD-2-Clause 许可。物体是自由刚体；手指驱动、接触和摩擦均由 MuJoCo 仿真，不以约束连接夹爪和物体。机械臂与夹爪之间的固定安装连接及夹爪内部联动约束保留。

在仓库根目录执行，前提是已按 Task 4A 准备 `configs/physics/grasp.yaml` 中列出的三个 batch 和 C2 碰撞组件：

```bash
conda activate asset2sim-physics
bash scripts/setup_menagerie.sh
asset2sim grasp-eval --config configs/physics/grasp.yaml \
  --output results/physics/grasp/debug_box_run03
```

输出目录必须不存在；重复实验请更换名称。安装脚本获取指定版本的上游两个模型到 `.local/mujoco_menagerie`。加载前核对模型版本、批次状态及生成资产的来源索引。每个来源保存可独立重载的 `model/scene.xml` 与相对路径资产、三次 `trajectory.csv`、首轮 `grasp.mp4`，以及公共 `run.json` 和 `summary.csv`。输入缺失或加载失败会写入失败记录，并保留计划次数作为成功率分母。原始生成网格、准备网格及碰撞组件均不改动。

## 设置与判定

采用 `debug_box_v0` 的 Reference、SF3D、SPAR3D 的 Task 4A C2 结果，统一缩放 0.5，最长边均为 0.10 m；沿夹持方向的初始包围盒宽度分别为 50.0、71.9、70.9 mm，均小于 2F-85 的标称 85 mm 开口。不同来源保持各自生成朝向，不进行参考网格对齐。质量均设为 0.1 kg，摩擦参数一致；质心和惯量使用各自缩放网格的均匀 AABB 先验估计，属于假设而非实测物性。三个场景使用相同的物体初始水平位置和预设末端位姿。碰撞组件分别加载，数量为 1、4、52。

脚本先等待 1 s，再依次接近 2 s、闭合 2 s、抬升 2 s、保持 1 s、释放 1 s；使用逆运动学确定预设目标，并不进行抓取规划。仿真步长 0.002 s。保持阶段至少 80% 的 50 Hz 样本需同时满足：物体相对初始高度上升至少 0.06 m、离开桌面、两侧指垫各有至少 0.05 N 法向力。释放结束时物体需接触桌面且不接触指垫，且此前保持条件已满足；MuJoCo 数值警告导致该次判无效。日志包括各阶段位姿、接触、指垫法向力和 `mj_step` 耗时。时间只统计 4500 次 `mj_step`，不含场景构建、控制计算、采样及视频渲染。

## 小规模结果

`debug_box_run03`，MuJoCo 3.3.5，Intel Core Ultra 7 265K。每个来源三次均使用完全相同的初始状态和控制，是确定性重复，并非三个独立对象或扰动试验。

| 来源 | 有效 / 计划 | 成功 / 计划 | 保持达标样本 / 次 | 最大抬升 | 接近前水平位移 | `mj_step` 平均耗时 / 4500 步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 3/3 | 3/3 | 50/50 | 108.6 mm | <0.1 mm | 0.121 s |
| SF3D | 3/3 | 3/3 | 50/50 | 103.5 mm | 18.4 mm | 0.164 s |
| SPAR3D | 3/3 | 0/3 | 0/50 | 未抬升 | 54.9 mm | 0.129 s |

Reference 和 SF3D 在保持阶段有双侧受力，释放后落回桌面。SPAR3D 在接近阶段开始前已倾倒并离开预设抓取位置，三次均未形成双侧指垫接触；它落在桌面不计为释放成功。结果表明该固定放置与位姿下未能完成抓取，不能单独归因于模型或 CoACD，也不能推广为模型总体成功率。后续应增加独立对象、稳定放置与抓取位姿扰动，并在统一硬件上重复计时；本实验未验证真实材料、真机和泛化抓取能力。

## 键盘交互

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

末端初始为向下姿态，旋转控制调整相对于该姿态的固定轴 XYZ 欧拉角；角速度及各轴角度范围由配置限制。IK 目标限制在配置的工作空间内；不可达或超关节限位的目标被拒绝并在终端及轨迹中记录。笛卡尔目标速度、关节**执行器目标**速度和夹爪命令变化速度均有上限；这些限制不是对碰撞冲击下实际关节速度的硬性保证。机械臂初始化、移动和夹爪动作均通过执行器，不直接写入仿真关节位置；物体仍为自由刚体。

切换和重置在同一个 Viewer 窗口内加载场景，避免反复关闭窗口造成闪屏。当前 MuJoCo 3.3.5 的公开被动 Viewer 接口不提供更换模型，加载使用该版本的内部接口；升级 MuJoCo 后需要重新验证。每次切换或重置产生独立 `segment_*`，包含 `trajectory.csv`、`segment.json` 和 `interaction.mp4`；会话目录另存使用的配置和各资产的可重载 MJCF。轨迹按 20 Hz 保存按键、目标末端位置与旋转、实际末端位置与姿态四元数、关节位置与命令、物体位置和四元数、桌面接触及双指法向力。视频使用固定观察视角，窗口中调整的视角不改变录像。关闭窗口后完成保存；录像帧率和仿真时间记录在配置中。绘制和 IK 计算可能使仿真时间慢于墙钟时间。

已在本机 X11 Viewer 验证窗口启动、资产切换、重置及关闭保存。自动测试使用同一按键控制器驱动 Reference 盒子，观察到双指受力、离桌抬升、水平移动和释放落桌；这是自动输入验证，尚未代表用户实际手动操作的结果。当前键盘读取依赖 X11，原生 Wayland、Windows 和 macOS 尚未实现；SF3D/SPAR3D 的手动抓取效果也需实际操作验证。

自动输入演示保存在 `results/physics/grasp/keyboard_validation01/`，使用与交互相同的控制器、记录和渲染流程，输入由程序模拟，未经过人工键盘操作。参考盒子在 501 个采样中有 154 个样本处于双指受力且离桌状态，物体中心最高 0.168 m；移动后松开夹爪，最终重新接触桌面，双指法向力均为零。`tests/test_grasp_interactive.py` 可重复验证对应的接触和运动条件。控制键仅在本进程的 Viewer 窗口拥有焦点时由交互入口接收，避免同时触发 Viewer 的显示及几何分组快捷键。
