# Evidence subset

The source is `share_mergenet_campaign_20260915(1).tar.gz`, captured on September 15, 2026. Its SHA256 and individual source hashes are recorded in `manifest.json`. The complete packet was read locally; no model checkpoint is present in this repository or in that packet.

- `runs/*/summary.csv`: byte-identical source summaries, parsed by header name.
- `runs/*/args.json`: resolved arguments converted from YAML; machine-specific absolute paths are removed or shortened to checkpoint run/name.
- `launch_protocol.json`: independently audited protocol facts from launch banners and resolved arguments, including world size, microbatch, accumulation and optimizer-step count. **Fine-tunes use update_freq=4 and effective batch 1024.** The packet's own derived analysis missed accumulation.
- `testtime_records.jsonl`: all 79 successful non-gate records, including repeated evaluations. The rebuild uses the latest successful timestamp per key, never the highest score; errors and gates do not enter result curves. Each retained record includes its original-line SHA256.
- `testtime_accuracy_curves.csv`: 20 matched patch-budget triplets, checked against the records by `scripts/rebuild.py`.
- `publishable_efficiency.csv`: independent audit of guarded H20 benchmark JSONs. These cells use random weights and synthetic input. They are not training-log throughput.
- `matched_latency_caveated.csv`: later 384 px measurements, with corrected patch counts, random weights, ToMe proportional attention off, timing hooks present, and reserved-memory semantics. No pairing with the accuracy sweep is authorized by these data.
- `token_trajectories.csv` and `analytic_cost.json`: code-derived shapes and arithmetic estimates, not logged per-layer shapes or measured GPU traffic.
- `diagnostics/`: new CPU soft-selection probes and an accepted A100 single-module layout benchmark. None changes or revalidates historical ImageNet accuracy.

`results.csv` and `sweep_latest.csv` are regenerated products. Source summary line numbers are one-based including the header; epoch indices are zero-based. Percentages are reported in percentage points when subtracted. The training summary's `eval_top1` is EMA because the trainer overwrites its raw evaluation metrics before writing the row.

The complete source logs were independently checked: 2,419/2,419 summary top-1/top-5 rows match final per-epoch EMA log lines at printed precision. Checkpoint verification receipts remain evidence of remote checks, not an independent load here. Training-box throughput and speculative GPU bandwidth/launch interpretations are excluded.

`followup/` contains the independent A100 whole-model timing, local CIFAR EMA replay, per-image predicted classes (no images), selector intervention and complete-model parity records. Its manifest is independent of the company packet manifest. `scripts/rebuild_followup.py` validates these artifacts and regenerates their tables; no company ImageNet weights are used.
