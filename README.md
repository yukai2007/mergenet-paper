# MergeNet manuscript and reproducible evidence

**Complete English research draft, September 15, 2026.** The manuscript studies spatial routing gains and deployment tradeoffs. It does not claim a state-of-the-art compression system or a verified joint ToMe accuracy–latency frontier.

- Read the paper: [mergenet-main.pdf](mergenet-main.pdf).
- Edit the paper: [mergenet-main.tex](mergenet-main.tex); prose is in `sections/`.
- Chinese assessment and next experiments: [notes/assessment.zh-CN.md](notes/assessment.zh-CN.md).
- Source provenance: [data/README.md](data/README.md), [research/review-plan.md](research/review-plan.md).

## Overleaf

Import this repository as a new project with [Overleaf GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization) if your account has that premium feature, or [upload the supplied ZIP](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project). Select **mergenet-main.tex** as the main document and **pdfLaTeX** as the compiler. All figures and generated tables are committed, so Python, GPUs, checkpoints, and external shell execution are unnecessary for compilation. The manuscript uses a standard article layout; venue formatting and author order are not yet specified.

## Rebuild locally

```bash
python -m pip install -r requirements-analysis.txt
python scripts/rebuild.py
python scripts/verify_cifar.py
python scripts/rebuild_followup.py
bash scripts/build.sh
```

`build.sh` uses `latexmk` when available, otherwise Tectonic (`TECTONIC_BIN` can specify its path). The canonical compiled PDF is copied to the repository root. Python rebuilds the numerical tables and four main vector figures from the curated records; it never launches training.

The CIFAR appendix has its own provenance and endpoint table. It is supporting resized-image evidence, not native high-resolution ImageNet evidence.

## What is established

- All 2,419 recorded ImageNet epochs match the corresponding EMA log lines; 16 schedules are complete.
- At 150 epochs and the same base learning rate, radius-3 routing improves best top-1 by 1.670 pp over global routing and 0.598 pp over flattened routing.
- Main best EMA accuracy: 81.368% at 224 px and 82.582% after 384 px fine-tuning, versus dense DeiT-S/8 at 82.246% and 82.976%.
- Guarded synthetic H20 measurements show a 224 px latency penalty and a 384 px dense-relative latency advantage. These are random-initialization architecture benchmarks.
- ToMe wins recorded accuracy at all 20 matched final-patch budgets. Its accuracy and latency harnesses differ in proportional attention; no combined Pareto claim is made.
- Local replay of two CIFAR EMA checkpoints preserves a 1.04–1.14 pp R3 advantage under rebatching; prediction changes and a selector intervention are archived per image.
- Unified A100 architecture timing finds no dense-relative MergeNet speedup at 224 or 384 on the local stack. Layout optimization improves R3 inference by about 5%; complete-model loss-scaled parity tests pass.

## What remains open

Company policy prevents transfer of the ImageNet checkpoints. Their recorded metrics are audited, but cannot be independently replayed here. This is a fixed availability restriction, not a request for weights. Local CIFAR checkpoints and random-initialization architecture tests support independent follow-up experiments; see [the follow-up report](notes/followup.zh-CN.md). Additional training seeds and degree-matched controls remain open. Partial curriculum and MergeNet 512 px runs are not presented as completed results.

This private repository is a writing and evidence workspace. Large checkpoints, full machine logs, original datasets and historical project trees remain outside it. Source snippets retain explicit origin information; there is no blanket relicensing of upstream code.
