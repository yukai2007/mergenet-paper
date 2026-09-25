# 本地效率测量（2026-09-25）

本目录是**随机权重架构探针**，不是训练后 checkpoint 的正式 E2。不得把以下效率数字与论文中已有的准确率拼成同一行，也不得用它们填充正式 E2 表。

## 测量设置

- 独占物理 GPU 2：NVIDIA A100-SXM4-80GB，UUID `GPU-93608e36-6ec1-0b48-c329-e0237065a231`；驱动 550.54.14；PyTorch 2.6.0+cu124、CUDA 12.4、cuDNN 90100、timm 0.9.11。
- MergeNet 代码为 `b1b2402e4859e11f6c5b4e44c2c2f4e28f2f19ec`。ToMe 使用该仓库的 adapter，而非 E2 手册指定的上游固定 commit；PiToMe 使用相邻 OpenToMe 工作树 `6c9640614a50e5e4d5b5ac91d6b5972839b1baff` 中未修改的 `opentome/timm/pitome.py`，其仓库 commit 也与 E2 手册指定的固定 commit 不同。
- 输入为设备驻留的随机 `B×3×224×224` tensor；随机初始化模型、eval、inference mode、fp16 autocast；不包含模型加载、数据解码或主机到设备传输。
- 每个方法和 batch 独立进程；预热 50 次；CUDA events 采样 5 轮，每轮 200 次；保留所有 1,000 个样本。每个 cell 记录延迟中位数、IQR、五轮中位数及其 CV、峰值 allocated 和 reserved 显存。Dense 有 784 个 patch，其余方法均经实际 forward 验证为 392 个 patch。
- eager 是正式 E2 约定的运行模式。编译版是额外的性能探索，MergeNet 和 ToMe 均编译 transformer blocks，且将 Dynamo specialization cache limit 设为 64；MergeNet 的 local encoder 保持 eager。两种模式不能混在论文同一比较中。

## Eager 结果

以下数字来自 [`summary_eager.csv`](summary_eager.csv)，每格对应的 JSON 中有完整的原始样本。

| 方法 | B1 延迟 (ms) | B1 峰值 allocated (GiB) | B64 延迟 (ms) | B64 吞吐 (图/秒) | B64 峰值 allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| Dense DeiT-S/8 | 4.382 | 0.142 | 30.469 | 2100.5 | 0.744 / 0.957 |
| ToMe | 12.035 | 0.143 | 34.717 | 1843.5 | 0.724 / 1.004 |
| PiToMe | 14.171 | 0.144 | 70.279 | 910.7 | 1.774 / 3.104 |
| MergeNet | 18.272 | 0.147 | 36.295 | 1763.3 | 0.776 / 0.836 |

MergeNet B1 的独立复测为 **20.998 ms**（[`mergenet_b1_rerun.json`](mergenet_b1_rerun.json)）；首次测量为 18.272 ms。两次的轮中位数 CV 分别为 2.44% 和 2.39%，均低于手册 3% 的 cell 门槛，但跨进程差异约 15%，且当时主机 load average 约 54。B1 对主机调度敏感，报告时应同时交代两次测量，不能只选较快的一次。B64 的四个 cell 的轮中位数 CV 均低于 0.3%。

## 编译版补充结果

以下数字来自 [`summary_compiled.csv`](summary_compiled.csv)。编译版 B64：MergeNet 27.242 ms、2349.3 图/秒、峰值 allocated 0.627 GiB；ToMe 27.219 ms、2351.3 图/秒、峰值 allocated 0.614 GiB。两者 B64 延迟差为 0.023 ms，不能据此宣称任一方显著更快。

编译版 B1：MergeNet 有效复测为 11.222 ms，ToMe 为 9.236 ms。MergeNet 首次 B1 测量的轮中位数 CV 为 3.98%，超过 3% 门槛，故保留 [`mergenet_b1_compiled.json`](mergenet_b1_compiled.json) 作为无效记录，汇总采用 [`mergenet_b1_compiled_rerun.json`](mergenet_b1_compiled_rerun.json)。

## 正式 E2 仍缺的输入

本地两个 repo 未找到 Dense、MergeNet 的训练后 EMA checkpoint，也没有 E1 的 matched DTEM 模型及其权重；本轮没有进行完整 ImageNet-val 准确率复评。因此上述数据只回答了当前实现的本地效率问题，正式五方法 accuracy/latency/memory 联合表仍需按 `EXPERIMENTS.md` 使用同语义 checkpoint、固定第三方 commit 和统一验证集完成。
