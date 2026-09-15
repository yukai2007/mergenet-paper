# Follow-up without company ImageNet checkpoints

The ImageNet weights cannot leave the collaborator's company. These experiments need only the canonical local source, the frozen local CIFAR runtime and its existing CIFAR EMA weights. Original source trees and checkpoints are never modified.

- `whole_model.py`: synthetic A100 B64 inference at 224 and 384; final patch budgets 392 and 1152; dense, ToMe with proportional attention on/off, R3 MergeNet, and R5.143 at384. Each cell has 10 warmups and 30 CUDA-event samples, three rounds with reversed order in round2. Token hooks are removed after checking actual shapes. Source tracing is disabled for ToMe; it does not contribute to classification. Random weights mean these measurements do not establish a trained-checkpoint accuracy–latency frontier. Complete-model layout training-graph comparisons report fixed, strict thresholds even if they fail, including repeat-original controls.
- `cifar_batch.py`: strict epoch199 EMA loads with fresh checkpoint hashes and frozen protocol validation, using only local August spatial campaign λ2/224 global and R3 weights. CIFAR-100 has 10000 test images; transforms reproduce the frozen evaluator (bicubic resize/crop, crop0.9, CIFAR statistics). Original-order B200 and B64 and permuted-order B200 predictions are restored to dataset order and saved. A separate equal-shape probe keeps images0–15 fixed and replaces48 companions; its rowwise selector intervention is a correctness experiment, not a re-trained model.

Both runners require an initially empty selected GPU and the same sole host process throughout measured cells. Model code may use nondeterministic reductions; a clean GPU does not imply deterministic arithmetic. Run imports and model setup outside timing. No dataset loading is included in synthetic inference timings. Forward timing has no optimizer or backward pass.

Example paths for this workspace:

```bash
python experiments/followup/whole_model.py \
  --source /liziqing/yukai/MergeNet_true2d_release_20260827/deliverables/imagenet_longtrain_v1 \
  --deps /liziqing/yukai/.deps_mergenet_resize20260810 --gpu 7 \
  --out build/followup/whole_model.json
python experiments/followup/cifar_batch.py \
  --campaign /liziqing/yukai/otm_worktree_mncifar/work_dirs/classification/cifar_dtem_spatial_20260814 \
  --deps /liziqing/yukai/.deps_mergenet_resize20260810 --gpu 6 \
  --data /liziqing/yukai/data --outdir build/followup/cifar
```

GPU indices must be changed if occupied. The dependency directory is a Python overlay, not a virtual environment. Archived scalar results live in `data/followup`; original per-image prediction arrays remain in the local ignored build directory, with hashes in the result manifest.

Two follow-up modes avoid repeating the main timing/evaluation campaign:

- Add `--parity-only --parity-backend dense` to `whole_model.py` to test the dense routing reference; choose a distinct `--out` file.
- Add `--parity-only --loss-scale 1024` to test the sparse training graph with fixed loss scaling and explicitly unscaled gradients. Without this flag the loss scale is1; the archived unscaled failures and unchanged-model repeat controls are retained.
- After the base CIFAR run has saved its original-order B200 logits, add `--reference-only` to `cifar_batch.py`. This evaluates only the rowwise selector intervention, saving `reference_eval.json` and its own prediction arrays; it preserves `results.json`.

The accepted results include two dense-backend parity passes and two loss-scaled sparse-backend passes. The failed unscaled sparse checks do not isolate a layout error: unchanged-model repeats exhibit comparable noise. No training update or long-run equivalence claim follows from these small training-graph probes.
