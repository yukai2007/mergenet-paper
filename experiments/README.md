# Independent implementation diagnostics

These experiments are separate from the model that produced the ImageNet results. They neither load nor modify a campaign checkpoint. The canonical campaign release is commit `4b4945afd93986d94ebb8bf4f44b2e0dba83d230`.

## Soft selection (CPU)

```bash
python experiments/soft_topk/probe_batch_dependence.py
python experiments/soft_topk/test_thretopk_bisection_reference.py
```

Requires PyTorch. Results go to ignored `build/diagnostics/`; the captured results live in `data/diagnostics/`. `historical_thretopk.py` is a byte-identical copy of the release's `opentome/utils/thetopk.py` for the probe. The separate bisection function supplies per-sample normalization, an exact-cardinality threshold reference and implicit backward. It passed 120/120 bounded checks and three nondegenerate gradchecks. It is slow, is not integrated, and has no trained-model accuracy evidence. A failed closed-form correction remains in the local review archive and is not offered as an implementation to adopt.

## Attention layout (single GPU)

`layout/microbenchmark.py` compares the released LocalAttention with an isolated layout variant. It requires the release attention file and a compatible PyTorch/FlashAttention environment. A device is selected explicitly; its process set is checked before, between and after rounds.

```bash
timeout 180s python experiments/layout/microbenchmark.py \
  --gpu 7 \
  --source /path/to/release/opentome/timm/bias_local_attn.py \
  --outdir build/layout
```

The captured experiment used one A100-SXM4-80GB, PyTorch 2.6.0+cu124 and FlashAttention 2.7.4.post1. It tests zero dropout, selected shapes, and random weights. The accepted run's samples and numerical comparisons are in `data/diagnostics/layout_results.json`. Per-round variation must be retained; no full-model or H20 speedup follows from this module measurement. The source module is not vendored as an entire framework, and `--deps` can optionally point to a dependency overlay.
