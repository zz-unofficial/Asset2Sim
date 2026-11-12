# Mesh 物理准备

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

`physical_adaptation.status` 表示物理准备状态：`ready` 为索引网格通过闭合、绕向和正体积等检查；`needs_repair` 需要进一步处理或人工检查；`invalid` 表示输入无效或导出回读失败。处理完成状态仍单独记录在 `status` 中。若索引不闭合、同坐标诊断闭合且存在不同 UV，会标记 `uv_seam_candidate`，这是接缝证据而非自动修复；无 UV 证据时标记 `coincident_seam_candidate`。非流形边等几何问题列为 `geometry_review`。

默认不清理、不填洞。根据需要显式选择：

- `--merge-coincident`：仅合并完全相同坐标的顶点；输出用于物理几何，不保留 UV/材质。相互接触但原本分离的部件也可能被合并。
- `--remove-degenerate`：删除零面积三角面；搭配 `--min-triangle-area <平方米>` 可指定面积阈值。
- `--remove-unreferenced`：删除没有面引用的顶点。

每次清理的阈值和数量写入 metadata，清理前后检查均保留。对有重复坐标的网格，报告另外提供仅在临时副本上合并坐标的拓扑诊断；它不会改变默认输出，也不能区分纹理接缝与恰好接触的不同部件。

实际检查中，当前 SF3D 和 SPAR3D 卷尺 GLB 的纹理接缝分别造成 653 和 714 个按索引连接的分量；对完全相同坐标做诊断合并后均为单一水密分量。代表性输出位于 `results/assets/physics/`：`reference_ring` 保留参考坐标，`sf3d_snail` 和 `spar3d_snail` 显式使用了 `--merge-coincident`。此结果提示后续碰撞适配需要明确选择是否合并，不能仅凭原始索引拓扑判断物理闭合性。水密和正体积检查不等同于自相交检测；未可靠闭合的对象不应把体积当作可信物理体积。

本轮完整回读结果保存在 `results/assets/physics/roundtrip_validation/`：四个项目 Debug 参考网格均为 `ready`；两模型卷尺默认输出均为 `needs_repair / uv_seam_candidate`，显式合并后均为 `ready`。最大顶点回读误差约为 4.2e-9m，均通过导出检查。合并坐标可能连接恰好接触的独立部件，因此 `ready` 仍需结合对象结构解释。
