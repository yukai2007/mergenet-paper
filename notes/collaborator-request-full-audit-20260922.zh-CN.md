# MergeNet 合作者补项清单与论文落位（2026-09-22）

这份清单面向持有公司侧 ImageNet 权重与运行环境的合作者。**不需要传出
checkpoint。** 权重可以始终留在公司机器上；论文需要的是允许分享的聚合指标、
运行参数、日志片段、哈希/状态收据和可视化。论文已建立 P01--P12 空槽，结果未到
之前所有数值格保持空白，不用中途最好值代替终点。

## 一、已有材料已经足够，不要重复提供

- ImageNet 224 像素 150/300 epoch 主结果，以及 384/512 像素完整微调结果。
- 四个已完成的 150 epoch 优化探针：warmup、$\lambda$ curriculum、学习率和
  drop-path。weight-decay 只有 140/150，仍按 partial 处理。
- 20 个预算点的 MergeNet/ToMe/PiToMe 准确率记录。
- H20 和本地 A100 的配置内效率测量；它们继续分别报告，不拼成联合 Pareto。
- 本地 24 个 CIFAR-100 三种子几何/选择器训练，以及 selector 和布局诊断。
- 已提供的配置、环境版本、源文件和启动记录。只有实际运行与这些材料不一致时才
  重新导出。

## 二、主线优先补项

### P01：完成 DeiT 初始化的 MergeNet 训练

**为什么需要：** 检验 224 像素准确率差距是否主要来自表示初始化，而不是路由
结构本身。当前只有 105/150，79.206 一类中途值不能作为终点。

**请提供：**

1. 从原 `last.pth.tar` 继续到 150 epoch 的完整 `summary.csv`；不要重新开一条
   训练冒充续跑。
2. resume 日志中恢复的 epoch、optimizer、scheduler、EMA 和 scaler 状态。
3. 初始化源 checkpoint 的文件 SHA-256、实际读取的 state key
   (`state_dict`/`state_dict_ema`/其他)、源 epoch 与是否 EMA。
4. 完整加载映射收据：loaded/missing/unexpected 的键名及形状、可训练参数统计。
   已有日志只确认 `loaded=152, missing=25, unexpected=0`，没有完整 missing 清单。
5. 最终 best/final EMA top-1、top-5、best epoch，及完整 resolved args。

**公平对照：** 主对照是同为 drop-path 0.07 的
`s5_mn_dp07_150e`（best EMA 79.534），不是 drop-path 0.1 的 79.316。另将源
dense DeiT 的 300 epoch 预训练成本单列，不能写成同训练预算比较。

**论文位置：** Results 的
`Representation initialization: reserved endpoint`，表 P01；协议身份写入 Setup，
映射明细放 Appendix P12。

### P02：渐进 latent 合并及同最终预算对照

**为什么需要：** 检验一次压缩后的 latent token 轨迹是否还可改进精度/效率。
它不是现有主模型的已完成结果。

**当前配置的真实含义：** 先保留原来的 `784 -> 392` gather，再在 6 个 latent
block 各删约 32 个 patch，名义轨迹是
`392 -> 360 -> 328 -> 296 -> 264 -> 232 -> 200`，CLS 另计。它没有缩短前
6 个 full-grid local block，也没有消除第一次 784->392 的突降。

**请提供：**

1. `s7_mn_proglatent_150e` 完整 summary、最终 args、preflight 和代码哈希。
2. 逐层 attention 输入、merge 输出、MLP 输入的 patch 数和 CLS 数；必须测实际
   数量，不能只抄配置。
3. best/final EMA 准确率、训练显存和与 P03 相同口径的推理时间。
4. 一条同为约 200 patches、同 drop-path/深度/初始化/训练配方的单次 gather
   对照。否则 progressive 与原生 392 模型的比较同时改变了轨迹和最终预算。
5. 同实际最终预算的 ToMe 点，用于部署比较；明确它是 dense checkpoint 上的
   post-training 方法。

**论文位置：** Results 的 `Progressive latent compression` 表 P02；Method
只在得到实际 token trace 后补候选分支；Systems 复用 P03 的效率协议。

### P03：同 checkpoint、同 harness 的准确率—延迟联合点

**为什么需要：** 现有 ToMe accuracy 开启 `prop_attn`，旧 latency 关闭；token
计数还混有 CLS 差异。两组记录不能组合成“同时更准、更快”的 Pareto 结论。

**每个 dense / MergeNet / ToMe / PiToMe 单元必须同时提供：**

- checkpoint SHA-256、实际 state key、EMA/raw、epoch；无需提供权重内容。
- 同一分辨率、crop/interpolation、batch 内容和顺序、world size、精度模式、
  `prop_attn` 及方法参数。
- 50,000 张 ImageNet val 的 top-1/top-5；实际每层 token 数，patch 与 CLS 分列。
- 移除计数 hook 后，在空闲单卡上测同步 CUDA-event 原始样本；GPU 型号、驱动、
  PyTorch/CUDA/FlashAttention 版本、warmup/重复次数、median/IQR。
- peak allocated 与 reserved memory 分列；说明是否只含 model execution。

建议先补 384px、约 1152 patches 的一个核心点，再扩展 224px 和其他预算。

**论文位置：** Systems 的 P03 空表和联合曲线空图；Setup 记录共同协议；原始
重复样本与设备信息放 Appendix/证据目录。结果到来前论文只保留“联合前沿未知”。

### P12：补齐最短的证据链缺口

本地重新核验了 23 个 run、3,365 个 summary epoch：全部从 0 连续，数值有限；
21 个达到目标 epoch，2 个是 partial。3,264 行 EMA loss/top-1/top-5 可与交付日志
匹配，仍缺 101 行日志：

- `s4_mn_lamcurr_150e`：epoch 64--149，共 86 行；
- `s4_mn_ft512_15e`：epoch 0--14，共 15 行。

如允许，请补这两段 EMA 日志及其文件哈希。不能补时，只需确认公司侧不可提供，
论文继续把对应终点标成 `summary-only`，不会删除结果，也不会声称日志复核通过。

**论文位置：** Setup 的证据审计段；Appendix P12。可复跑报告见
`research/collaborator-review-20260921/log-audit.json`。

## 三、只有要扩大论文主张时才补

### P04：ImageNet 多种子 geometry + degree-matched

若要把“空间几何超出候选数量的作用”从 CIFAR 推广到 ImageNet，需要 global、
flat8、R3、degree-matched 在完全相同配方和 selector 下做配对种子训练。degree
对照应保留每个 donor 的候选数和 receiver-degree multiset，改变 receiver 的空间
对应。输出每种子的完整终点与配对差值；不预写“全部为正”或“显著”。

**论文位置：** Results 的 P04 空表。没有这组结果时，ImageNet 结论继续限定为
R3 相对现有 global/flat 单种子对照，候选数控制只由 CIFAR 支撑。

### P05：rowwise selector 的 ImageNet 迁移

分成两个问题：同一旧权重替换 selector 只回答推理行为；匹配的重新训练才回答
学习结果。先固定一批图像，替换同批其他图像，报告 logits 变化、预测翻转、
selection-mass error 和 finite checks；若声称改善训练，再补 matched ImageNet
training。当前 rowwise 实现同时改变归一化和阈值求解，若要拆因果还要做
factorial control。不能把该问题泛化到未审计的其他 DTEM 实现。

**论文位置：** Systems 的 P05 空表，算子细节放 diagnostic appendix。

### P06：ImageNet routing 可视化

预先选定 val image IDs，并同时展示典型例和失败例。逐层输出实际 A/B partition、
eligible mask、transport weights、carrier、mass。默认 trace 只保存 flattened index
质心，不能反推完整成员集合；如果要展示累计来源，需在公司侧增加并验证 source
membership trace。空间局部边不能自动证明语义边界保留；若要做该主张，应再给
标注来源和定量指标。

**论文位置：** Method 后的 P06 空图；完整样本规则放附录。

### P07：训练时压缩强度的干净消融

保持 local/latent depth、window、recovery、selector 和优化配方不变，只改变实际
carrier budget，并使用配对 seed。必须实现并验证真正不物理缩短的 control；现有
`method=none` 仍输出 392 patches，不能作为 784-token dense readout。历史
$\lambda=4$ 同时改了深度和 window，也不能填这张表。

**论文位置：** Results 的 P07 空表。

### P08：真实 GPU 成本归因与完整训练步

只有要写具体瓶颈或训练加速时才需要。用 device-side profiler 分开 local
attention/MLP、selection、transport、gather、recovery 和 latent block；不要把可能
重叠的 region 时间直接相加。HBM 带宽需要设备 counter，逻辑 tensor bytes 不够。
BNHD/layout 需在相同 input/loss/backward/optimizer/AMP 下测完整 step，并记录实际
optimizer updates 与 skipped steps。现有约 5% 推理改善不能填训练效率格。

**论文位置：** Systems 的 P08 空表和 Appendix layout 小节。

## 四、低优先级/可选补项

- **P09 Base 尺度：** 先得到可靠 dense DeiT-B anchor，再做 matched MergeNet-B。
  现有 drop-path 0.1 运行出现 train loss 降、val loss 升，只能说明该锚点欠佳；
  不能断言 drop-path 是唯一原因，也不能预填 0.2 必然修复。
- **P10 512px 半径：** 若要隔离 radius，要让两个分支从相同 384px parent 开始。
  现有 512 MN 由 R5.143 parent 初始化；其父子链增益是 82.574->82.618，+0.044pp。
  相对另一条 R3 384 结果的展示差是 +0.036pp，不能称为同分支 fine-tune 增益。
- **P11 weight decay：** 140/150 尚未完成。最后十轮没有数学上界，不能称
  “不可能过门槛”。若论文需要完整 wd 消融就原状态续完；否则保留 partial，
  不写终点结论。

这些分别落在 Appendix P09、P10、P11。它们不阻止当前限定主张的论文成立。

## 五、每个新结果的统一交付格式

每个 run 建一个同名目录，至少包含：

```text
<run_id>/
  summary.csv
  args.yaml                 # 最终 resolved args，含 launcher override
  launch_receipt.txt        # 命令、resume/initial checkpoint 行、world size
  source_sha256.json        # 实际执行源码与配置的相对路径、SHA-256
  checkpoint_receipt.json   # 只含 digest/state key/epoch/EMA 标志，不含权重
  environment.txt           # GPU、driver、Python、torch/CUDA/timm/flash-attn
  token_trace.json          # 仅需要 token 方法/渐进实验时提供
  eval_or_timing.json       # 原始评测或 CUDA 重复样本
```

不要只发截图或手工汇总表。路径可脱敏，密钥、Cookie、用户名及权重不要包含。
若公司政策不允许某项，写明“不可提供”和边界即可，论文会保留相应限制，而不会
用推断补空值。

## 六、已纠正的旧文档表述

1. `$\pm0.40$ pp` 是活动预先设定的晋级门槛，不是统计估计的噪声带、零假设或
   等价区间。
2. 4 个已完成的新探针没有越过门槛；wd 是 partial。不能写“五个已完成探针”。
3. CIFAR 表有 2 selectors x 3 controls x 3 seeds = 18 个逐 seed 的 R3 配对差，
   不是 12 个。
4. fine-tune 的有效 global batch 是 1024：5004 是每 rank microsteps，约 1251
   optimizer steps；旧审计里的 batch 256 是漏算 `update_freq=4`。
5. ToMe 在现有 accuracy sweep 更准，但旧 latency 的 attention policy 不同；不能
   写成已验证联合支配。
6. `method=none` 仍是 392 patches，不能证明压缩本身只损失约 0.004pp。
7. 单个 DeiT-S/16 分数接近文献不是整个 pipeline 的 0.1pp 误差上界。
8. 150->300 gap 变宽支持“不优先盲目加长训练”，不能证明所有更长训练都无效。

论文正文已经按这些边界修订，所有待补结果均留空。
