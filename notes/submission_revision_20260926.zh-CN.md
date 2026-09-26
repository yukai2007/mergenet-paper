# 2026-09-26 截稿版整理

## 叙述提纲与段落角色

1. Abstract / Introduction：问题是高密度 patch 的全局计算；贡献是可微空间质量路由与物理 carrier 瓶颈；用已完成准确率与执行成本作证据。
2. Method：解释 local encoder、mass routing、carrier selection / recovery、log-mass attention；不宣称未经验证的模块因果收益。
3. 4.1 Experimental Setup：交代训练、最终 token 对齐和结果来源。
4. 4.2 Main Results：展示 224-pixel 主表，训练时长结果作为补充证据。
5. 4.3 Token Budget and Resolution：先说明预算变化，再说明高分辨率微调，最后给出不微调的迁移结果。
6. 4.4 Efficiency：先给解析 MAC，再给 A100 eager / compiled 表图，最后概述 H20 高分辨率运行点。
7. Discussion / Conclusion：收束物理瓶颈的设计与已测运行点；明确准确率与随机权重测速的证据范围。

## 删除内容

- 正文和附录中的消融、CIFAR 控制实验、DTEM 待补比较。
- 未完成的 checkpoint evaluation、模块干预、路由可视化、占位符及索取后续数据的正文表述。
- 不再输入 draft-slots、pending-evaluations、visualization-placeholder；历史源文件留存但不参与编译。
- Related Work 中的 DTEM 文献背景保留，实验中无 DTEM 数值或待补行。

## 主张与证据

Claim: 224 pixels、392 carriers 下 top-1 为 81.360%。 | Evidence: data/results.csv；data/testtime_records.jsonl 中对应 checkpoint 全量 50,000 图像复评。 | Status: supported。

Claim: 512-pixel 微调达到 82.618%。 | Evidence: data/runs/ 对应训练 summary.csv 的 best EMA；tables/resolution.tex。 | Status: supported。

Claim: 224-pixel 解析 MAC 从 22.41 降为 14.13 GMAC，约减少 37%。 | Evidence: data/analytic_cost.json；解析计算不含 indexing / data movement。 | Status: supported。

Claim: 预算与分辨率改变带来不同准确率运行点。 | Evidence: data/testtime_accuracy_curves.csv；60 项准确率逐一与 latest 成功全验证原始记录核对。 | Status: supported。

Claim: compiled A100 B64 延迟 MN 27.24 ms、ToMe 27.22 ms。 | Evidence: experiments/local_gpu_probe_20260925_full_synthetic_gpu2/ 中记录；随机权重、合成输入，不宣称 0.02 ms 构成速度优势。 | Status: supported。

Claim: H20 384-pixel B64 延迟 MN 136.54 ms、dense 170.61 ms，但 MN 分配显存更高。 | Evidence: data/publishable_efficiency.csv 对应 clean 384 records；与 A100 条件分开报告。 | Status: supported。

## 五维自查

- 贡献：能否用未完成消融证明每个模块的必要性？不能，因此删除相关因果主张；保留架构与实现贡献。
- 清晰度：实验是否按读者问题组织？是：主准确率→预算/分辨率→运行成本；表格和图对应正文引用。
- 实验强度：是否虚构数据、把训练曲线最优与复评混为一谈？否；主表复评与 best EMA 在正文和表注中区分。
- 评价完整性：是否承诺缺失的 DTEM、可视化、模块干预？否；当前稿仅报告完成的数据；没有 SOTA 声明。
- 方法可靠性：解析 MAC 是否冒充 wall-clock speedup，或随机权重测时冒充训练 checkpoint 测试？否；两类证据单独标注；H20 与 A100 不混合排行。

## 复现入口

`python scripts/prepare_submission_results.py` 验证已有记录并生成结果表。
`python scripts/plot_a100_efficiency.py` 生成正文效率图。
`bash scripts/build.sh` 编译最终稿。
