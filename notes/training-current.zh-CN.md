# 八卡正式补实验已启动

2026-09-15，活动目录：`/liziqing/yukai/mergenet_local_campaign_20260915`。

本批补充已经进入正式训练，不再停留于计划：CIFAR-100/224、MN λ2，历史/逐样本selector ×global/flat8/R3/候选数匹配receiver置换 ×seeds42/43/44，合计24个200epoch任务。每GPU独立跑一项，八卡自动派发，训练/验证batch都固定为200。

选择该范围是为先回答三个问题：逐样本selector改变训练后的精度吗；R3相对global/flat收益能否跨seed重复；候选数量不变时，局部几何本身是否有贡献。布局优化和ImageNet新训练暂不加入，以保持消融可解释。

进入正式训练前，120组数值、3组gradcheck、6组候选数匹配测试通过；8配置各完成24次真实optimizer更新，无非有限loss或AMP跳步，并完成全测试集raw/EMA短跑验证。正式任务另起目录从随机初始化开始，恢复原始学习率和aux日程。

主终点预先固定为epoch199 EMA top1，报告同seed差值和三seed均值/样本标准差。中途精度与短跑结果不会写成论文终点；满足完整条件后才生成`three_seed_contrasts.csv`。

- 实时进度：本机`mergenet_local_campaign_20260915/PROGRESS.zh-CN.md`，每分钟读取真实CSV更新。
- 静态源码/协议与启动证据：[experiments/training_20260915](../experiments/training_20260915/README.md)。其中`status_snapshot.json`注明截取时间，不是实时查询。
- 数据、日志、完整last/best checkpoint保留在活动目录；论文GitHub只归档源码、协议、紧凑证据和状态快照。
- 任何任务失败会保留错误与已存权重，停止派发新任务，不自动换seed或静默重跑。

公司ImageNet checkpoint不参与，现有18页研究稿的历史实验表尚未被这些进行中的新训练替换。
