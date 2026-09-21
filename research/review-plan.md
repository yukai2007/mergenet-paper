# Review and manuscript contract

- Working title: MergeNet: Spatial Routing Gains and Deployment Tradeoffs in Learned Token Compression.
- Question: Does original-grid spatial support improve a DTEM-derived, fixed-slot soft router, and do those internal gains translate into competitive ImageNet accuracy and measured latency?
- Reader: computer-vision and efficient-learning researchers.
- Retrieval cutoff: 2026-09-20.
- Unit: a versioned method, concrete implementation, or individually identified experimental run.
- “All”: all 23 recorded ImageNet runs in the September 19 campaign packet, plus the 24 completed local CIFAR seed runs; not an exhaustive survey of every token-compression paper.

## Scope

Include primary papers for DTEM, ToMe, PiToMe, geometric merging and compressed/dense exchange; raw ImageNet summaries, launch configurations, valid test-time sweeps and guarded timing records. Use the August true-2D CIFAR campaign only as explicitly separate supporting evidence if its source chain can be carried into the manuscript.

Exclude quarantined historical results, partial experiments from completed-run tables, contaminated throughput, unverified checkpoint reproduction, acceptance claims without official confirmation, causal claims based only on endpoint slopes, and scores copied across incompatible recipes. No new weights are present in the delivered packet.

## Comparison axes

1. Matching support: global, flattened local, original-grid radius.
2. Training: same-recipe internal controls versus independently trained backbones; no added training for post-hoc merging.
3. Representation: fixed-slot soft transport, physical selection, recovery, layerwise token trajectory.
4. Outcome: best and final EMA ImageNet accuracy; observed single-seed differences.
5. Cost: actual batched inference latency at fixed device/batch/precision, distinct from MAC estimates and final tokens.

## Search map

| Family | Primary evidence | Status |
| --- | --- | --- |
| DTEM / ToMe / PiToMe | versioned papers and official code | refreshed by literature audit |
| ALGM / DSM / GTP-ViT / CubistMerge / NAP | versioned spatial-method papers | refreshed by literature audit |
| LookupViT / token pruning | official papers | inherited and rechecked |
| MergeNet | release commit plus September 19 packet and local CIFAR seed campaign | independent numerical and implementation audits |

## Planned synthesis and limits

Internal spatial constraints improve one-seed ImageNet results over MergeNet's unconstrained router. A three-seed CIFAR campaign reproduces the sign against global, flat, and degree-matched controls. The composed architecture does not establish a superior deployment frontier. Report negative comparisons and separate suggested mechanisms from measured causes. Company restrictions prevent ImageNet checkpoint transfer; ImageNet remains single-seed. Independent local CIFAR replay, three-seed CIFAR training, and synthetic architecture timing provide bounded follow-up evidence.

## Deliverables

- Anonymous, complete English research manuscript, standard LaTeX article, modular sections.
- Data-derived figures/tables with reproducible CPU analysis; claim and source ledgers.
- Canonical source: mergenet-main.tex; canonical PDF: mergenet-main.pdf.
- Private GitHub repository: yukai2007/mergenet-paper; Overleaf entry point mergenet-main.tex.
- Separate Chinese assessment and prioritized remaining experiments.

## Workflow note

The available literature-review skill references paper-writing/general-writing siblings that are absent from the installed skill directories. Direct editorial, citation, build-log and rendered-page audits substitute for these unavailable helpers. Tectonic is used for local compilation because latexmk/TeX Live are not installed; the sources remain compatible with standard pdfLaTeX/Overleaf.

## September 20 update

The September 19 ImageNet packet completes the 512-pixel MergeNet fine-tune, the $\lambda$ curriculum, drop-path and learning-rate probes, and a DeiT-B/16 run that overfits. The local eight-GPU CIFAR campaign completed all 24 seed/geometry/selector jobs on 2026-09-16. Interrupted: weight-decay 140/150 (do not resume) and DeiT-init 105/150 (resume). Progressive latent merging was never started. Company ImageNet weights remain unavailable.
