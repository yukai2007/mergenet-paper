# New local training campaign (complete)

The 24 CIFAR-100 jobs finished on 2026-09-16. Endpoints used in the manuscript are epoch-199 EMA top-1, recomputed in `data/cifar_seeds/`. The live campaign directory is `/liziqing/yukai/mergenet_local_campaign_20260915`; this snapshot is not a second copy of the checkpoints.

`code/` and `runtime/` preserve exactly the executable source identities from the campaign. Runtime licensing and upstream headers remain intact. `jobs/` and `smoke_jobs/` record resolved commands; their output paths refer to the live campaign, so do not run them from this archive. `report.py` demonstrates the aggregation logic but reads a live campaign's state/logs. `validation/` and `smoke_gate.json` contain completed preflight evidence. Large weights and full machine logs are not copied.

`README.zh-CN.md` is the original campaign guide; its live relative progress links refer to the campaign directory, not this static archive. To refresh the source/progress snapshot only, run `python scripts/sync_training_campaign.py` from the paper project. It never launches training or changes paper result tables.
