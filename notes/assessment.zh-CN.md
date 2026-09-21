# 2026-09-22 论文进展与给学长的清单

主稿已按 9 月 19 日 ImageNet 包和本地 24-run CIFAR 训练更新，并加入 P01--P12 可编译空槽，当前 PDF 29 页（`mergenet-main.pdf`）。CIFAR 图现为终点柱状图加三 seed 均值轨迹。结论仍然是：**空间路由有内部收益，完整系统还不是更强的压缩方法。**

## 现在可信的结论

| 问题 | 核验后的结果 |
| --- | --- |
| ImageNet 包 | 23 个有记录 run，21 个完整；3,365 行均连续且有限，其中 3,264 行与交付 EMA 日志一致，101 行因缺日志保留 summary-only 状态 |
| 空间约束（ImageNet，单 seed） | 150e、同 LR 下 R3 比 global 高 1.670 pp，比 flat 高 0.598 pp |
| 空间约束（CIFAR，3 seed） | 历史 selector：R3−global $+2.11\pm0.87$，R3−degree $+1.42\pm0.30$；逐样本 selector 同号。18 个逐 seed R3−对照差全为正 |
| 300e ImageNet 224 | MN 81.368%，dense p8 82.246%，差 −0.878 pp |
| 384 微调 | MN R3 82.582%，dense 82.976%，差 −0.394 pp |
| 512 微调 | MN $R=6.857$ 82.618%，dense 83.068%，差 −0.450 pp。384 不是“继续逼近”的趋势点 |
| 优化器/正则探针 | warmup −0.078、λ curriculum −0.006、lr 1e-3 +0.034、drop_path 0.07 +0.218，均未越过活动预先设定的 +0.40 pp 晋级门槛；这不是统计等价检验 |
| 是否真的更快 | H20 架构基准 224 较慢；384 R3/b64 推理 136.54 vs 170.61 ms，时间减少 20.0%。本机 A100 两个分辨率都没有超过 dense |
| ToMe | 精度在 20 个同最终 patch 预算上都更高；精度/计时的 prop_attn 不同，联合 Pareto 暂不成立 |
| DeiT-B | peak 80.442 @ep179，final 78.566；drop_path 0.1 过拟合，不能当 scale anchor |
| 未完成 | wd 0.03 停在 140/150（需要完整消融时再续）；DeiT-init 停在 105/150（优先续）；progressive latent 从未启动，且需要同最终预算对照 |

公司 ImageNet checkpoint 仍不在本地，也不在 tar 包里。CIFAR 24 个新训练的 last/best 在 `/liziqing/yukai/mergenet_local_campaign_20260915/runs/`。

发给学长的可转发清单：[`handoff-20260920.zh-CN.md`](handoff-20260920.zh-CN.md)。
