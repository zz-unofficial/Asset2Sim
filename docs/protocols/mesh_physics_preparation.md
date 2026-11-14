# Mesh 物理准备与碰撞几何

## 实验配置与运行

### Mesh 物理准备

`asset2sim prepare-mesh` 将 GLB、OBJ、STL 转为物理几何，输出到新目录。GLB 场景按所有节点实例展开，完整应用节点及父节点变换；源文件不修改。

参考网格声明为 Z-up、米制，保持原位置和 0.20m 最长边。生成网格必须提供对应生成运行目录，命令核对 `results.jsonl` 中成功的 Mesh，读取 `run.json` 声明的原始朝向及 `raw_to_model_z_up`，再居中并统一最长边至 0.20m。不读取参考几何，不执行 ICP 或依据参考对象调整尺度。

```bash
conda activate asset2sim-sf3d
asset2sim prepare-mesh --kind reference \
  --mesh data/samples/sf3d/reference/debug_ring.glb \
  --output data/processed/physics/reference_ring
asset2sim prepare-mesh --kind generated \
  --run data/generated/sf3d/paper_gso/run02 \
  --mesh data/generated/sf3d/paper_gso/run02/gso_SNAIL_MEASURING_TAPE/mesh.glb \
  --output data/processed/physics/sf3d_snail
```

`normalized.glb` 为 Z-up 米制三角网格；`metadata.json` 记录来源、生成器坐标声明、每个节点的完整变换、场景到物理坐标的矩阵与全部操作。`mesh_report.json` 保存 `before`（原始场景）、`before_cleaning`（转换后）、`after`（清理后）和 `after_export`（GLB 回读）的诊断，包括顶点/面数、AABB、有限值、退化面、边界边、非流形边、重复面、连通分量、水密性、绕向和体积。失败时保留两个 JSON、错误原因和已执行操作；已有输出目录不会覆盖。

导出后重新加载 GLB，检查 AABB、0.20m 尺度、双向顶点误差、三角面连接与绕向、表面积、拓扑及清理结果。坐标容差为 `max(1e-7m, 最长边×1e-6)`；未通过时标记失败，保留回读诊断。

`physical_adaptation.status` 表示物理准备状态：`ready` 为索引网格通过闭合、绕向和正体积等检查；`needs_repair` 需要进一步处理或人工检查；`invalid` 表示输入无效或导出回读失败。处理完成状态仍单独记录在 `status` 中。若索引不闭合、同坐标诊断闭合且存在不同 UV，会标记 `uv_seam_candidate`，该标记提供接缝证据，网格保留原状；无 UV 证据时标记 `coincident_seam_candidate`。非流形边等几何问题列为 `geometry_review`。

默认不清理、不填洞。根据需要显式选择：

- `--merge-coincident`：仅合并完全相同坐标的顶点；输出用于物理几何，不保留 UV/材质。相互接触但原本分离的部件也可能被合并。
- `--remove-degenerate`：删除零面积三角面；搭配 `--min-triangle-area <平方米>` 可指定面积阈值。
- `--remove-unreferenced`：删除没有面引用的顶点。

每次清理的阈值和数量写入 metadata，清理前后检查均保留。对有重复坐标的网格，报告另外提供仅在临时副本上合并坐标的拓扑诊断；诊断保留默认输出，重合坐标处的部件关系需结合对象结构判断。

### 碰撞几何构建

碰撞模块接收 `prepare-mesh` 输出的 `normalized.glb`，沿用 Z-up、米制和已有坐标，不读取生成模型或 Ground Truth。CoACD 可在已有 SF3D 或 SPAR3D 环境中使用；MuJoCo 接触实验也提供独立的 `asset2sim-physics` 环境，见 [MuJoCo 接触评测](mujoco_contact_evaluation.md)。在模型环境安装碰撞依赖后运行：

```bash
conda activate asset2sim-sf3d
python -m pip install -e '.[collision]'
asset2sim build-collision \
  --mesh results/assets/physics/roundtrip_validation/debug_ring/normalized.glb \
  --condition C2 --output results/assets/collision/example_ring/C2
```

`C0` 用 trimesh / SciPy Qhull 构建单凸包；`C1`、`C2`、`C3` 使用官方 `coacd==1.0.14`，阈值分别为 0.10、0.05、0.02。默认公共设置见 `configs/collision/default.yaml`：种子 0，归一化凹度阈值（`real_metric: false`，阈值为归一化数值），其余搜索参数及预处理设置显式记录。对比时保持配置、输入及计算资源一致。

每个结果写入新目录，包含独立的 `collision_000.obj` 等和 `colliders.json`。后者记录输入诊断、方法及设置、组件数、每个组件的顶点/三角面数和包围盒、总规模、分解时间、导出时间及表面误差。分解时间包括 CoACD 子进程启动和数据传递，不包括导出和误差计算。已有目录不会覆盖；失败时保留 JSON 和原因，使用前检查 `status: completed`。后续物理引擎逐个加载所有 OBJ。

默认拒绝非水密、非流形、绕向不一致或体积不可靠的输入；退化面、重复面与未引用顶点应先通过显式记录的步骤处理。UV 接缝造成的索引分离也不自动合并，参见上文 Mesh 物理准备。如确需处理开放网格，复制配置并将 `coacd.preprocess_mode` 设为 `on`，然后传入 `--config <配置>` 和 `--allow-open`；内部预处理可能改变几何。`auto` 也可能触发内部预处理，官方接口不返回实际是否触发。C0 使用 `--allow-open` 时仅对输入顶点求凸包并记录警告。外部已准备网格可用 `--assume-prepared` 明确声明 Z-up 和米制。

误差对输入表面和**所有组件表面**分别按面积采样各 10,000 点，种子分别为 0 和 1，使用最近采样点估计双向距离的均值、95 分位和最大值，长度单位为米；对称均值为两个方向均值的平均。独立随机采样存在采样误差，相同表面的采样结果也可能有非零距离。组件重叠时，指标包含内部面；接触行为通过物理实验另行测量。

## 实验结论

### 网格准备与回读

实际检查中，当前 SF3D 和 SPAR3D 卷尺 GLB 的纹理接缝分别造成 653 和 714 个按索引连接的分量；对完全相同坐标做诊断合并后均为单一水密分量。代表性输出位于 `results/assets/physics/`：`reference_ring` 保留参考坐标，`sf3d_snail` 和 `spar3d_snail` 显式使用了 `--merge-coincident`。碰撞适配时可结合接缝诊断和对象结构选择是否合并。水密、正体积和自相交分别检查，可靠闭合的网格用于体积估计。

完整回读结果保存在 `results/assets/physics/roundtrip_validation/`：四个项目 Debug 参考网格均为 `ready`；两模型卷尺默认输出均为 `needs_repair / uv_seam_candidate`，显式合并后均为 `ready`。最大顶点回读误差约为 4.2e-9m，均通过导出检查。合并坐标可能连接恰好接触的独立部件，因此 `ready` 仍需结合对象结构解释。

### 圆环及生成资产的碰撞组件

在 Intel Core Ultra 7 265K 上，对 Debug 圆环使用默认配置，结果保存在 `results/assets/collision/debug_ring/`：

| 条件 | 组件数 | 总顶点 / 面数 | 构建时间（秒） | 双向均值（mm） |
| --- | ---: | ---: | ---: | ---: |
| C0 | 1 | 128 / 252 | 0.001 | 3.93 |
| C1 | 7 | 184 / 340 | 1.08 | 2.48 |
| C2 | 12 | 224 / 400 | 1.26 | 2.54 |
| C3 | 16 | 256 / 448 | 1.35 | 2.73 |

单凸包填平圆环孔洞；CoACD 使用更多组件近似孔洞，表面距离同时包含组件内部面和采样误差，随阈值变化呈非单调关系。SF3D 和 SPAR3D 的卷尺生成资产在显式合并接缝后也完成 C2 分解，分别得到 25 和 27 个组件；两组结果展示了碰撞构建接口的输出规模。
