# MergeNet：八卡本地补实验

2026-09-15启动。公司ImageNet权重不参与这批实验。所有正式模型从随机初始化独立训练；原发布目录、旧CIFAR权重和历史结果未改动。

当前进展请看 [PROGRESS.zh-CN.md](PROGRESS.zh-CN.md)，每分钟从原始CSV重新生成。最终主结果尚未得到，不能使用短跑或中途精度代替。

## 预先固定的矩阵

| 维度 | 取值 |
| --- | --- |
| 数据 | CIFAR-100，原生32px图像按旧协议增广/放大到224；不是原生高分辨率ImageNet |
| 网络 | MN λ2，patch8，6 local +6 latent，原始布局，历史dense空间评分后端 |
| Selector | 历史ThreTopK / 逐样本FP32二分selector |
| 几何 | Global / flat-window8 / R3 / R3候选数匹配的receiver置换 |
| Seeds | 42、43、44，每个配置全部重训；共24runs |
| 训练 | 200epochs，train/val batch200，accum1；每GPU一个独立任务 |
| 主终点 | epoch199 EMA top1；同seed差值、3seed均值和样本标准差 |
| 次终点 | 按旧saver规则保存的best EMA；不能用best替换主终点 |

配方沿用August CIFAR空间实验：AdamW lr0.001/wd0.05，cosine minlr0.0001，warm20，EMA0.9998，Mixup0.8/Cutmix1、RandAug、label smoothing0.1、drop_path0.1、clipnorm1。aux logit从epoch20开始20epochs升至0.05。除了预先指定的selector/geometry/seed，所有对比配置相同。8任务并发统一workers4（旧配方workers8），checkpoint_hist统一1；所以即使seed42也属于本次新run，不宣称逐位重现旧训练。

固定λ=2但保留旧trainer的`lambda_start=2, lambda_ramp_epochs=50`元数据。旧saver因此在epoch0–49对选优分数减1000；实际token数始终392patch。原`health.json: last_ema_metric`/调度器STATUS相应字段是传给saver的选优分数，早期不是准确率。**PROGRESS与reports始终读取summary.csv的真实eval_top1。** epoch199主终点不受该显示问题影响。

## 已实现的两个干预

`code/selector.py`保留原算子不变，通过进程内替换启用新算子：逐行极差归一化，FP32求解软质量总和为k的阈值；一个Triton program处理一行，40轮二分，反传用隐式VJP。大k时求缺额质量，避免FP32对接近N的总和产生消减误差。FP16/BF16输入先提升至FP32；FP64/CPU使用PyTorch参考路径。此处k质量约束在下游invalid-row mask之前成立，未另外改变历史empty-row语义。旧ImageNet/CIFAR结果均不标成新算子的结果。

`code/interventions.py`的degree对照从每次**真实donor/receiver划分后的R3掩码**出发，再置换receiver列。因此逐个donor候选数严格不变，receiver候选数的整体分布也不变。置换由固定seed20260915产生的receiver-ID优先级决定，重复调用可复现、不消耗网络的初始化/dropout/随机分组RNG。该对照保留列置换后的图相关结构，**不是独立随机边图**；它用来隔离“短距离邻接”与“相同候选数量”这两个因素。支持图seed固定，三seed描述的是训练随机性。

布局优化不加入本次矩阵，避免把系统改动和学习行为混在同一消融中。

## 已完成的门槛检查

- 120组GPU/FP64数值case：随机、常量、近常量、outlier，N=4/16/64/392/784，多个k；前向及梯度均有限。最大输出参考误差9.54e-7，最大质量误差6.10e-5。
- 3组非退化FP64 gradcheck；跨样本输出与梯度依赖为0；FP16/BF16有限性检查通过。
- 6组实际随机/交替分组的掩码检查：逐行候选数、receiver度数分布完全匹配；新旧边重合约2.5%–5.0%，网络RNG不变。
- 全部8配置真实数据短跑通过：每配置24个minibatch且24次实际optimizer更新、0次AMP跳步，随后完整10000张raw和EMA验证。
- 短跑有意将aux设为0.05且初始lr设为0.001进行压力测试；正式训练另起目录、从seed重新初始化，恢复原始warmup和aux日程。短跑checkpoint不作正式初始化。

## 调度与结果

`code/supervise.py`独立后台运行：先8短跑全通过，再分派24正式run；首批是seed42的8配置，其后43、44。每个GPU完成一个任务后自动领取下一个。禁止把其他人的进程当成本调度器任务，也不会终止外部进程。

- `protocol.json`：预先固定的比较协议。
- `runtime_manifest.json`：只读历史runtime副本的逐文件SHA256。
- `code_manifest.json`：实际启动的干预/训练包装器代码版本。
- `jobs/`：24个完整argv与干预标识。
- `smoke_jobs/`、`smoke/`、`smoke_gate.json`：独立短跑及通过凭据。
- `runs/<job>/`：配置、完整train.log、原始summary、health、last/best权重及完成凭据。
- `reports/progress.csv`：中间结果，明确标成partial。
- `reports/completed_endpoints.csv`：只有完成200epochs的正式终点。
- `reports/three_seed_contrasts.csv`：每组3seed都完成后才输出配对差值和均值/标准差。

发现非有限训练loss/验证指标、零实际更新或epoch1以后超过20%跳步时，该任务报错；出现任务失败后停止派发新任务，已在跑的任务继续，原始错误与中途权重保留。不会默默重试或替换seed。当前没有自动恢复未完成run；如后续显式resume，必须记录中断和恢复，因为原trainer未保存完整RNG状态，不能保证轨迹逐位一致。

源码/状态可随论文仓库保存；大型checkpoint与完整日志留在本机，不进入Overleaf。待正式终点完成，再更新论文结果表和结论。

## 操作入口

```bash
# 查看真实CSV进度并重新汇总，不修改训练
python /liziqing/yukai/mergenet_local_campaign_20260915/report.py

# 停止派发新任务，已在跑的任务继续
# 在campaign根目录创建STOP文件；移除后继续派发。

# 仅终止本调度器启动的任务
# 在campaign根目录创建CANCEL文件。
```

不要对活动目录重复启动调度器或删除run。首次启动命令和进程号在`supervisor_process.json`中；互斥锁阻止第二个调度器同时运行。
