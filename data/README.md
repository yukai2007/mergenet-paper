# Evidence subset

Primary ImageNet sources: `share_mergenet_campaign_20260919.tar.gz` and the incremental `share_mergenet_campaign_20260923.tar.gz` Stage-7 refresh (SHA-256 values in the corresponding manifests). The September 15 packet remains the source of the test-time sweep JSONL and efficiency CSVs. No model checkpoint is present in this repository or in the packets.

- `runs/*/summary.csv`: byte copies of source summaries, parsed by header name.
- `runs/*/args.json`: resolved arguments converted from YAML; machine-specific absolute paths are removed or shortened to checkpoint run/name.
- `launch_protocol.json`: protocol facts from `00_summary/tables/RUN_METADATA.csv` plus recomputed best/final metrics. Fine-tunes use `update_freq=4` and effective batch 1024.
- `cifar_seeds/`: 24 completed local CIFAR-100 trainings. `endpoints.csv` is recomputed from epoch-199 `eval_top1`; `contrasts.csv` matches the campaign's `three_seed_contrasts.csv`.
- `inference_merge_probe.json`: non-gate 224-pixel re-evaluations of the 300-epoch MergeNet checkpoint from `TESTTIME_SWEEP.csv`.
- `testtime_records.jsonl`: all 79 successful non-gate records from the September 15 packet. The rebuild uses the latest successful timestamp per key, never the highest score.
- `testtime_accuracy_curves.csv`: 20 matched patch-budget triplets, checked against the records by `scripts/rebuild.py`.
- `publishable_efficiency.csv` and `matched_latency_caveated.csv`: guarded H20 architecture timings. Random weights, synthetic input. Not training-log throughput. The later 384 px ToMe timings have proportional attention off and must not be paired with the accuracy sweep.
- `followup/`: independent A100 whole-model timing and August CIFAR EMA replay.

`results.csv` and `sweep_latest.csv` are regenerated products. Source summary line numbers in the older manifest are one-based including the header; epoch indices are zero-based. The training summary's `eval_top1` is EMA because the trainer overwrites its raw evaluation metrics before writing the row. For the $\lambda$ curriculum run, best/final reporting uses `eval_top1_full_compression` once that column is nonzero.

The read-only audit in `research/collaborator-review-20260921/`, refreshed on
September 23, checks all 3,560 delivered summary rows and matches 3,354 to
supplied EMA logs. The 206 missing log rows include the curriculum run's
epochs 64--149, all 15 MergeNet 512-pixel epochs, and the first 105 epochs of
the host-migrated DeiT-initialized run; those rows remain summary-only evidence.
Checkpoint receipts remain evidence of remote checks, not an independent
load. Training-box throughput and speculative GPU bandwidth/launch
interpretations are excluded.
