# 2026-09-22 MergeNet 投稿完成度与五天收口判断

## 当前状态

**论文工程完成度约 85%。** 主稿已切换到 ICLR 2027 官方样式，正文在参考文献前结束于第 7 页，低于 9 页硬上限；参考文献占第 7--9 页，附录从第 10 页开始，整份 PDF 17 页。原 19 页正文的证据审计内容没有删除，已移到附录，并另存 `mergenet-audit.tex` 和 `sections/audit/` 作为长稿快照。

主线已经统一为一个贡献：**连续、可微的空间质量路由之后接固定预算的物理 token bottleneck**。论文明确区分四件事：

1. Su Jianlin threshold Top-K 是采用的连续松弛，不宣称该算子本身原创；
2. 真正的新组合是 soft routing 与 exact-budget hard gather 同处一条训练 forward path；
3. 空间先验只限制原图网格上的可选边，edge weight、mass 和 carrier location 仍由模型学习；
4. `log(mass)` bias 可以用 rank-one Q/K 扩维兼容 FlashAttention，但现有 ImageNet checkpoint 实际走 PyTorch fused SDPA，因此 Flash 路径单独表述为经过 parity 验证的实现贡献。

## 已经足够可信的证据

| 论文主张 | 当前证据 |
| --- | --- |
| 二维空间支持优于无空间支持 | ImageNet 150e 同 recipe：R3 比 global +1.670 pp，比 flat +0.598 pp |
| 收益不只来自候选数 | CIFAR-100 三 seed 的 receiver-degree-matched control；历史 selector 下 +1.42±0.30 pp，逐样本 selector 下 +1.19±0.13 pp |
| 不依赖单一 selector | 两种 selector 下 R3 都排名第一；18 个 paired R3-control 差值全为正 |
| 确实产生短序列 | 784→392 hard gather 是 latent encoder 前第一个物理压缩点 |
| 完整系统边界 | 300e MN 81.368%，dense 82.246%；matched sweep MN 81.360%，ToMe 81.932%；ToMe 在 20/20 已记录预算上精度更高 |
| 分辨率行为 | MN-dense gap：224 -0.878 pp，384 -0.394 pp，512 -0.450 pp；不能写成单调逼近 |

论文因此可以可信地投稿为“新机制 + 受控空间路由证据”，但现在**不能写 SOTA**。现有完整模型没有超过 dense/ToMe，224px 历史随机权重基准也没有速度优势。摘要、Introduction、Results、Discussion 和 Conclusion 已全部按这个证据强度改写。

## 还缺的只有两项

1. **一个数字包：** matched DTEM top-1；dense/DTEM/ToMe/MergeNet 同协议 latency 与 peak memory；以及 `mass_bias_off`、`recovery_off` 两个 inference-only top-1/top-5。
2. **一张最终 checkpoint 路由图：** 4--6 个 ImageNet 样本的逐层 transport edge、mass、carrier 与失败例，并附 passive-trace 一致性检查。

可直接转发给合作者的格式见 `collaborator-request-20260922.zh-CN.md`。checkpoint 无需传出。

## 能否赶上

能形成合规投稿稿，但依赖合作者在截止前返回上述材料。ICLR 2027 官方 full-paper deadline 是 **2026-09-25 11:59 PM AOE**，正文严格最多 9 页，超页会 desk reject：<https://iclr.cc/Conferences/2027/AuthorGuidelines>。当前版已解决格式和页数风险。

建议收口顺序：

- 9 月 22--23 日：合作者跑统一数字包并导出 trace；本地继续做语言、引用和匿名性检查。
- 数字到达当天：自动填主表、摘要和组件消融，重新生成 PDF；若 DTEM 确实不可得，删掉 abstract 中的比较句而不是填公开的非匹配数字。
- 图到达当天：替换附录空图；若版面和信息质量足够，再将精简版调到主文。
- 截止前最后一天：冻结 PDF，逐项核对 OpenReview 元数据、作者列表、补充材料和匿名信息。

最大的投稿风险已经从“论文没写完”变成“核心结果强度有限”。即使两项都补齐，如果统一基准没有速度/内存优势，论文仍应坚持机制论文定位；不能用旧 H20 的非同协议数字包装成 Pareto 优势。
