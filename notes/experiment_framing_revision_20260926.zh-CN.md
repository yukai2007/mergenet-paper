# 实验叙述修订：2026-09-26 截止后本地准备稿

分支：revision/experiment-framing。原 main 的提交版本保留。本分支仅准备后续允许修订时使用的稿件，不能表示已更新 OpenReview。

## 提纲和段落角色

1. Introduction：将贡献与整体架构的分类能力、预算和执行成本联系起来；模块设计理由不改写为未经消融验证的准确率收益。
2. Experiments opening：先给出固定预算分类、预算/分辨率变化、执行成本三个问题，帮助读者理解已有实验的覆盖范围。
3. Main Results：先说明整体架构的比较对象，再解释主表。ToMe、PiToMe、dense 的真实差异仍在正文和原表中。
4. Budget and Resolution：解释同 checkpoint 的运行点和训练预算处峰值，不把预算变化包装为模块消融。
5. Efficiency：先解释同一实现 eager→compiled 的 25.0% 延迟变化，再保留 compiled ToMe 对比和完整模式表。
6. Discussion / Conclusion：收束联合流水线及不同运行点，不承诺未完成实验。

## 主张与证据

Claim: MN 在392 carriers达到81.360%，512微调82.618%。 | Evidence: 原 main/resolution 表和源训练、复评记录。 | Status: supported。
Claim: 解析MAC减少37.0%。 | Evidence: 原 analytic_cost.json，排除 indexing/data movement 的口径保留。 | Status: supported。
Claim: compiled A100 B64 MN latency 相对 eager 减少25.0%。 | Evidence: 原表36.30→27.24 ms；(36.30-27.24)/36.30=24.9587%。 | Status: supported，限定合成随机权重运行点。
Claim: 384预算扫中最佳MN点位于训练K=1152。 | Evidence: 原budget表，不新增归因。 | Status: supported。
Claim: 联合流水线可以在真实缩短的latent序列上训练/推理。 | Evidence: 模型结构、实现和物理token检查。 | Status: supported；不等于每个模块提升accuracy。

## 五维自查

- 贡献：强调可微传输与物理 gather 的连接；不新增SOTA或模块必要性声明。
- 写作：实验按问题组织，减少主表数字在引言中的重复；真实baseline排序仍保留。
- 实验强度：全部沿用已有数值；单run、随机权重测速与bestEMA/复评的口径仍说明。
- 完整性：缺少模块消融及DTEM实验的风险仍存在，本次文字修改没有补齐该证据。
- 方法可靠性：未把shape/parity测试当accuracy消融，未混合H20与A100排序。
