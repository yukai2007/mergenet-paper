# New local training campaign (in progress)

Source/evidence snapshot only. The active campaign is `/liziqing/yukai/mergenet_local_campaign_20260915`; read its `PROGRESS.zh-CN.md` for current progress. `status_snapshot.json` is timestamped and may be stale. Original run directories must not be relaunched or overwritten.

The frozen matrix is historical/rowwise selector ×global/flat8/R3/receiver-degree-preserving permutation ×seeds42/43/44, each200epochs on CIFAR-100 resized to224. Eight independent GPU jobs share the same batch200 and training recipe. Formal endpoints and seed statistics remain pending until the relevant runs finish.

`code/` and `runtime/` preserve exactly the executable source identities from the campaign. Runtime licensing and upstream headers remain intact. `jobs/` and `smoke_jobs/` record resolved commands; their output paths refer to the live campaign, so do not run them from this archive. `report.py` demonstrates the aggregation logic but reads a live campaign's state/logs. `validation/` and `smoke_gate.json` contain completed preflight evidence. Large weights and full machine logs are not copied.

`README.zh-CN.md` is the original campaign guide; its live relative progress links refer to the campaign directory, not this static archive. To refresh the source/progress snapshot only, run `python scripts/sync_training_campaign.py` from the paper project. It never launches training or changes paper result tables.
