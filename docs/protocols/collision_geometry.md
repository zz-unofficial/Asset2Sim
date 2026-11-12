# 碰撞几何构建

碰撞模块接收 `prepare-mesh` 输出的 `normalized.glb`，沿用 Z-up、米制和已有坐标，不读取生成模型或 Ground Truth。CoACD 可在已有 SF3D 或 SPAR3D 环境中使用；MuJoCo 接触实验也提供独立的 `asset2sim-physics` 环境，见 [MuJoCo 接触评测](mujoco_contact_evaluation.md)。在模型环境安装碰撞依赖后运行：

```bash
conda activate asset2sim-sf3d
python -m pip install -e '.[collision]'
asset2sim build-collision \
  --mesh results/assets/physics/roundtrip_validation/debug_ring/normalized.glb \
  --condition C2 --output results/assets/collision/example_ring/C2
```

`C0` 用 trimesh / SciPy Qhull 构建单凸包；`C1`、`C2`、`C3` 使用官方 `coacd==1.0.14`，阈值分别为 0.10、0.05、0.02。默认公共设置见 `configs/collision/default.yaml`：种子 0，归一化凹度阈值（`real_metric: false`，阈值单位不是米），其余搜索参数及预处理设置显式记录。对比时保持配置、输入及计算资源一致。

每个结果写入新目录，包含独立的 `collision_000.obj` 等和 `colliders.json`。后者记录输入诊断、方法及设置、组件数、每个组件的顶点/三角面数和包围盒、总规模、分解时间、导出时间及表面误差。分解时间包括 CoACD 子进程启动和数据传递，不包括导出和误差计算。已有目录不会覆盖；失败时保留 JSON 和原因，使用前检查 `status: completed`。后续物理引擎应逐个加载所有 OBJ，而不是合并后再求单凸包。

默认拒绝非水密、非流形、绕向不一致或体积不可靠的输入；退化面、重复面与未引用顶点应先通过显式记录的步骤处理。UV 接缝造成的索引分离也不自动合并，参见 [Mesh 物理准备](mesh_physics_preparation.md)。如确需处理开放网格，复制配置并将 `coacd.preprocess_mode` 设为 `on`，然后传入 `--config <配置>` 和 `--allow-open`；内部预处理可能改变几何。`auto` 也可能触发内部预处理，官方接口不返回实际是否触发。C0 使用 `--allow-open` 时仅对输入顶点求凸包并记录警告。外部已准备网格可用 `--assume-prepared` 明确声明 Z-up 和米制。

误差对输入表面和**所有组件表面**分别按面积采样各 10,000 点，种子分别为 0 和 1，使用最近采样点估计双向距离的均值、95 分位和最大值，长度单位为米；对称均值为两个方向均值的平均。独立随机采样存在采样误差，即使两个表面相同也不一定为零。组件相互重叠时，此指标包括内部面，不能视为碰撞体并集边界误差，也不能直接当作物理接触误差。

## 示例结果

在 Intel Core Ultra 7 265K 上，对 Debug 圆环使用默认配置，结果保存在 `results/assets/collision/debug_ring/`：

| 条件 | 组件数 | 总顶点 / 面数 | 构建时间（秒） | 双向均值（mm） |
| --- | ---: | ---: | ---: | ---: |
| C0 | 1 | 128 / 252 | 0.001 | 3.93 |
| C1 | 7 | 184 / 340 | 1.08 | 2.48 |
| C2 | 12 | 224 / 400 | 1.26 | 2.54 |
| C3 | 16 | 256 / 448 | 1.35 | 2.73 |

单凸包填平圆环孔洞；CoACD 使用更多组件近似孔洞，但组件内部面与采样误差使该表面指标不保证随阈值单调下降。SF3D 和 SPAR3D 的卷尺生成资产在显式合并接缝后也完成 C2 分解，分别得到 25 和 27 个组件；这些只是接口示例，不是模型质量排名或接触实验。
