# SF3D 论文指标与公开子集实测

| 数据 | 来源 | 设备 | 有效/计划 | CD | FS@0.1 | FS@0.2 | FS@0.5 | PSNR | SSIM | LPIPS | 图像→Mesh 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gso | measured_public_subset | NVIDIA GeForce RTX 5060 | 24/24 | 0.0327 | 0.9434 | 0.9975 | 1.0000 | 15.0089 | 0.7704 | 0.2811 | 0.6601 |
| gso | paper_reported | NVIDIA H100 | 278/278 | 0.0980 | 0.7010 | 0.8940 | 0.9880 | 21.2470 | 0.8650 | 0.1240 | 0.5000 |
| omniobject3d | measured_public_subset | NVIDIA GeForce RTX 5060 | 24/24 | 0.0452 | 0.9542 | 1.0000 | 1.0000 | 9.1204 | 0.7364 | 0.3972 | 0.6614 |
| omniobject3d | paper_reported | NVIDIA H100 | 308/308 | 0.0900 | 0.7260 | 0.9200 | 0.9900 | 20.1340 | 0.8510 | 0.1320 | 0.5000 |

实测采用固定公开对象清单，数据与数值设置见各 evaluation.json。论文行来自 [SF3D 表 1 与表 A1](https://arxiv.org/abs/2408.00653)。运行时间对应各行记录的设备。

[查看 gso 重建对照图](gso_qualitative.png)：按 CD 排名固定选择四个位置，展示输入与相反视角。

[查看 omniobject3d 重建对照图](omniobject3d_qualitative.png)：按 CD 排名固定选择四个位置，展示输入与相反视角。
