# MergeNet manuscript

**Post-deadline working revision:** branch `revision/experiment-framing` prepares clearer experimental framing for a later permitted revision. The original submission is preserved on `main`; this branch does not update OpenReview.

**Completed manuscript, September 26, 2026.** This version uses the available ImageNet-1K accuracy records and completed synthetic efficiency measurements. It does not depend on outstanding collaborator experiments.

- Paper: [mergenet-main.pdf](mergenet-main.pdf).
- Main source: [mergenet-main.tex](mergenet-main.tex); prose is in `sections/`.
- Data provenance: [data/README.md](data/README.md).
- Revision and evidence map: [notes/submission_revision_20260926.zh-CN.md](notes/submission_revision_20260926.zh-CN.md).

## Contents

The main experiments cover ImageNet accuracy at 224 pixels, inference token budgets, higher-resolution fine-tuning, and A100 inference latency and memory. The appendix reports implementation settings, complete accuracy budgets, training duration, resolution transfer, and a separate H20 benchmark at 384 pixels. Architecture benchmarks use random weights and synthetic inputs; their results are reported separately from trained-checkpoint accuracy. Ablation results, DTEM experimental comparisons, and routing visualizations are excluded from this version.

Historical experiment plans and audits remain in the repository for provenance; they are not part of the compiled manuscript.

## Anonymous code supplement

Upload [mergenet-anonymous-code.zip](mergenet-anonymous-code.zip) to the supplementary-material field separately from the paper PDF. This is the implementation package, not the Overleaf manuscript source ZIP. It contains unchanged model and trainer sources, configuration, dependency pins, implementation checks, third-party notices, and file checksums. Git history, repository-owner handles, checkpoints, datasets, and internal experiment plans are excluded.

Rebuild the package with `python scripts/package_supplementary_code.py`. The manuscript references this supplement; upload it together with the PDF.

## Compile in Overleaf

Upload the submission source ZIP, select **mergenet-main.tex** as the main document, and use **pdfLaTeX**. The source package contains the required figures, tables, bibliography, and conference style files. Python and GPU access are unnecessary for compilation.

## Rebuild locally

```bash
python -m pip install -r requirements-analysis.txt
python scripts/prepare_submission_results.py
python scripts/plot_a100_efficiency.py
bash scripts/build.sh
```

The result preparation script validates the published budget accuracy values against the archived full-validation records and generates the completed-result tables. The plotting script reads the recorded A100 benchmark results. Neither script launches training or benchmarks. The build script copies the compiled PDF to the repository root.

Large checkpoints and original datasets remain outside this writing repository. Historical scripts and source snippets retain their original provenance.
