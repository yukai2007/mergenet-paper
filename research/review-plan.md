# Review and manuscript contract

- Working title: MergeNet: Spatial Routing Gains and Deployment Tradeoffs in Learned Token Compression.
- Question: Does original-grid spatial support improve a DTEM-derived, fixed-slot soft router, and do those internal gains translate into competitive ImageNet accuracy and measured latency?
- Reader: computer-vision and efficient-learning researchers.
- Retrieval cutoff: 2026-09-15.
- Unit: a versioned method, concrete implementation, or individually identified experimental run.
- “All”: all 18 runs in the September 15 campaign packet; not an exhaustive survey of every token-compression paper.

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
| MergeNet | release commit plus September 15 packet | independent numerical and implementation audits |

## Planned synthesis and limits

Internal spatial constraints improve one-seed ImageNet results over MergeNet's unconstrained router. The composed architecture does not establish a superior deployment frontier. Report negative comparisons and separate suggested mechanisms from measured causes. Missing transferred checkpoints and absent repeated seeds limit independent replication and statistical conclusions.

## Deliverables

- Anonymous, complete English research manuscript, standard LaTeX article, modular sections.
- Data-derived figures/tables with reproducible CPU analysis; claim and source ledgers.
- Canonical source: mergenet-main.tex; canonical PDF: mergenet-main.pdf.
- Private GitHub repository: yukai2007/mergenet-paper; Overleaf entry point mergenet-main.tex.
- Separate Chinese assessment and prioritized remaining experiments.

## Workflow note

The available literature-review skill references paper-writing/general-writing siblings that are absent from the installed skill directories. Direct editorial, citation, build-log and rendered-page audits substitute for these unavailable helpers. Tectonic is used for local compilation because latexmk/TeX Live are not installed; the sources remain compatible with standard pdfLaTeX/Overleaf.
