# 八卡 CIFAR 补实验已完成

2026-09-16 全部 24 个 200-epoch 任务结束。活动目录：`/liziqing/yukai/mergenet_local_campaign_20260915`。

主终点是 epoch-199 EMA top-1，已写入论文 `sections/results.tex` 与 `data/cifar_seeds/`。历史 selector 下 R3 相对 global / flat / degree 的三 seed 均值差为 +2.11 / +1.30 / +1.42 pp；逐样本 selector 为 +1.63 / +0.77 / +1.19 pp。12 个配对差全为正。

公司 ImageNet checkpoint 仍未参与。ImageNet 侧后续只缺学长机器上的 DeiT-init resume 与 progressive-latent 启动，见 `handoff-20260920.zh-CN.md`。
