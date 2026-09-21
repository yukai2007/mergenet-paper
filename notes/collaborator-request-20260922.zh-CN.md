# 给合作者：MergeNet 最终只需要两项材料（2026-09-22）

公司侧 checkpoint **不需要传出**。论文已经把现有 ImageNet、CIFAR、分辨率扩展和
预算扫描全部用上；现在只缺下面两类可公开产物。请不要再整理整套历史日志，也不用
重跑已有的优化器扫描。

## 1. 一份补齐主表和两个组件消融的数字包

请在持有最终 checkpoint 的机器上，用**同一张空闲 GPU、同一套环境和同一个
评测脚本**输出一个 CSV/JSON。主表固定为 ImageNet-1K、224px、patch size 8、
最终 392 个 patch token（CLS 另计），包含四行：

1. `DeiT-S/8` dense；
2. `DTEM`，同 backbone、同最终 token 数；
3. `DeiT-S/8 + ToMe`，`prop_attn=true`；
4. `MergeNet R=3`，最终 300e EMA checkpoint。

每行请给：

```text
method, checkpoint_sha256, state_key, epoch, ema_or_raw,
image_size, input_patch_tokens, final_patch_tokens,
top1, top5, n_images,
batch_size, precision, gpu, driver, torch_cuda_flash_versions,
warmup, repeats, latency_samples_ms, median_ms, iqr_ms,
peak_allocated_gib, peak_reserved_gib, data_loading_included,
prop_attn, source_commit
```

准确率覆盖 50,000 张 val；计时前移除 token-count hook，用同步 CUDA event，模型执行
时间不含数据加载。四行必须共用 batch、precision、预处理和计时口径。现有论文已知
准确率是 dense 82.246、ToMe 81.932、MergeNet matched sweep 81.360（训练摘要 best
81.368）；如统一重跑有微小差异，请保留新值并附原始记录，不要手工对齐。真正缺的
分类分数只有 matched DTEM。

同一数字包再加两个**无需重训的 inference intervention**，都从最终 MergeNet
checkpoint 出发，只报 top-1/top-5：

- `mass_bias_off`：把 latent attention 的 `log(token_mass)` key bias 置零；
- `recovery_off`：跳过 gather 后的 residual cross-attention recovery。

这两个数字只用来说明已训练模型依赖哪些组件，论文会明确标成 inference-only，不会
写成重新训练的结构消融。如果 matched DTEM 在公司环境也没有，请明确回复
`DTEM matched checkpoint/run unavailable`；不要用论文里 79.85% 的 DeiT-S/16
公开结果代替 patch-8、392-token 单元。

## 2. 最终 MergeNet checkpoint 的路由可视化

本地没有最终 ImageNet checkpoint，所以当前只能保留空图。请在公司侧对
`s3_mn_r3_lr75_300e` 的 best EMA 权重导出 4--6 个 val 样本，至少含 2 个正确例和
2 个失败例。每个样本请同时给：

- 原图、ground truth、预测类别和置信度；
- 六个 routing step 的 donor/receiver partition 与非零 transport edge；
- 每一步的 token-mass heatmap；
- 最终 392 个 carrier 位置，并突出最高质量 carrier；
- gather 后 recovery 的 attention map（可选 1--2 个代表 carrier）；
- 原始 `npz/json` 和论文可直接用的 PDF/PNG。

可视化必须是 passive trace：同时记录 trace 开关前后最大 logit 差、top-k 是否完全
一致、每层 mass-conservation error。默认 center trace 不能还原完整累计 source
membership；如果没有开启完整 trace，就画“本层实际 transport edge + mass + carrier”，
不要把质心伪装成完整 token 分组。若 ImageNet 图片本身不能传出，可只给匿名 patch
网格和 raw trace，我们在本地排版。

## 回传目录

```text
mergenet_final_fill/
  metrics/main_and_interventions.csv
  metrics/timing_raw.json
  metrics/environment.txt
  visualizations/sample_*/figure.png
  visualizations/sample_*/trace.npz
  visualizations/index.json
  SHA256SUMS
```

以上两项到齐即可填论文主表、摘要中的三处空值和主可视化。旧的完整审计清单已存为
`notes/collaborator-request-full-audit-20260922.zh-CN.md`，这次无需按那份长清单逐项补。
