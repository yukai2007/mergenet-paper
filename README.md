# MergeNet manuscript and reproducible evidence

**English research draft, September 22, 2026.** The manuscript presents MergeNet as a differentiable spatial-routing architecture with an explicit physical token bottleneck. The core claim is the joint design: learned continuous mass transport on the original image grid, followed by an exact-budget gather before global latent reasoning. Deployment superiority remains conditional on a common trained-checkpoint benchmark.

- Read the paper: [mergenet-main.pdf](mergenet-main.pdf).
- Edit the paper: [mergenet-main.tex](mergenet-main.tex); prose is in `sections/`.
- Chinese submission assessment and collaborator list: [notes/assessment.zh-CN.md](notes/assessment.zh-CN.md), [notes/collaborator-request-20260922.zh-CN.md](notes/collaborator-request-20260922.zh-CN.md).
- Final collaborator request: [notes/collaborator-request-20260922.zh-CN.md](notes/collaborator-request-20260922.zh-CN.md). It contains only two deliverables: missing numeric cells and final-checkpoint routing visualizations. Blank fields are never treated as results.
- Source provenance: [data/README.md](data/README.md), [research/review-plan.md](research/review-plan.md).

## Overleaf

Import this repository as a new project with [Overleaf GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization) if your account has that premium feature, or [upload the supplied ZIP](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project). Select **mergenet-main.tex** as the main document and **pdfLaTeX** as the compiler. All figures and generated tables are committed, so Python, GPUs, checkpoints, and external shell execution are unnecessary for compilation. The repository bundles the official ICLR 2027 style files. The current anonymous submission draft has seven pages of main text before references, within the nine-page limit.

## Rebuild locally

```bash
python -m pip install -r requirements-analysis.txt
python scripts/rebuild.py
python scripts/verify_cifar.py
python scripts/rebuild_followup.py
bash scripts/build.sh
```

`scripts/ingest_20260919.py` copies new ImageNet summaries and the 24 CIFAR seed runs into `data/`. `build.sh` uses `latexmk` when available, otherwise Tectonic (`TECTONIC_BIN` can specify its path). The canonical compiled PDF is copied to the repository root. Python rebuilds the numerical tables and figures from the curated records; it never launches training.

The CIFAR appendix has two layers: a historical single-seed resized-image sweep, and a later three-seed selector/geometry campaign. Neither is native high-resolution ImageNet evidence.

## What is established

- 21 of 23 recorded ImageNet runs in the September 19 packet are complete. Endpoints are recomputed from `summary.csv`. A September 21 audit checks all 3,365 rows for continuity and finite values and matches 3,264 rows to supplied EMA logs; 101 rows are explicitly summary-only because their log segments are absent.
- At 150 epochs and the same base learning rate, radius-3 routing improves best top-1 by 1.670 pp over global routing and 0.598 pp over flattened routing (single ImageNet seed).
- CIFAR-100, three seeds: radius 3 beats global, flat, and a degree-matched control under both selectors. Historical-selector mean R3−global is $+2.11\pm0.87$ pp; R3−degree is $+1.42\pm0.30$ pp.
- Main best EMA accuracy: 81.368% at 224 px, 82.582% after 384 px fine-tuning, 82.618% after 512 px fine-tuning, versus dense DeiT-S/8 at 82.246%, 82.976% and 83.068%. The gap narrows at 384 px and does not keep closing at 512 px.
- Four completed 150-epoch probes (warmup 20, $\lambda$ curriculum, lr $10^{-3}$, drop-path 0.07) do not exceed the campaign-defined $+0.40$ pp promotion threshold. This is not an equivalence test; the weight-decay run is partial.
- Guarded synthetic H20 measurements show a 224 px latency penalty and a 384 px dense-relative latency advantage for radius 3. These are random-initialization architecture benchmarks.
- ToMe wins recorded accuracy at all 20 matched final-patch budgets. Its accuracy and latency harnesses differ in proportional attention; no combined Pareto claim is made.
- Local replay of two CIFAR EMA checkpoints preserves a 1.04–1.14 pp R3 advantage under rebatching; prediction changes and a selector intervention are archived per image.
- Unified A100 architecture timing finds no dense-relative MergeNet speedup at 224 or 384 on the local stack. Layout optimization improves R3 inference by about 5%; complete-model loss-scaled parity tests pass.

## What remains open for the submission

Company policy prevents transfer of the ImageNet checkpoints. The paper now reserves only two collaborator-side outputs: a matched DTEM/top-1 plus common-protocol latency and memory packet, including two inference-only component interventions; and passive routing visualizations from the final MergeNet checkpoint. Interrupted initialization, progressive compression, weight-decay completion, and Base-scale work remain follow-up ideas rather than submission blockers.

This private repository is a writing and evidence workspace. Large checkpoints, full machine logs, original datasets and historical project trees remain outside it. Source snippets retain explicit origin information; there is no blanket relicensing of upstream code.
