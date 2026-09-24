#!/usr/bin/env python3
"""Architecture-only ImageNet-scale GPU latency and memory probe.

Weights are random. This is not the trained-checkpoint E2 benchmark.
Run one process per method/batch; the timed path has no tracing hooks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def gpu_apps() -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader"], text=True
    )
    apps = [line.strip() for line in output.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.isdecimal():
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
             f"--id={visible}"], text=True
        ).strip()
        apps = [line for line in apps if line.startswith(f"{uuid},")]
    return apps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dense", "tome", "pitome", "mergenet"),
                        required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 64), required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--compile-local-encoder", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compile_local_encoder and (
        not args.compile_blocks or args.method != "mergenet"
    ):
        parser.error("--compile-local-encoder requires --compile-blocks and MergeNet")

    before = gpu_apps()
    if before:
        raise RuntimeError(f"GPU has compute processes before probe: {before}")
    os.environ.setdefault("OPENTOME_SKIP_OPTIONAL_NLP", "1")
    sys.path.insert(0, str(args.code_root.resolve()))

    import torch
    import timm

    if args.compile_blocks:
        # ToMe changes token length in each block; the default limit of eight
        # specializations otherwise leaves later blocks uncompiled.
        torch._dynamo.config.cache_size_limit = 64

    torch.manual_seed(42)
    if args.method == "mergenet":
        import opentome.models.mergenet.model  # noqa: F401
        model = timm.create_model(
            "mergenet_small_cls", pretrained=False, img_size=224,
            patch_size=8, num_classes=1000,
        )
    else:
        model = timm.create_model(
            "deit_small_patch16_224.fb_in1k", pretrained=False,
            img_size=224, patch_size=8, num_classes=1000,
        )
        if args.method == "tome":
            from opentome.timm.tome import tome_apply_patch
            tome_apply_patch(model, prop_attn=True, trace_source=False)
            model.r = [32] * 11 + [40]
        elif args.method == "pitome":
            from opentome.timm.pitome import pitome_apply_patch
            pitome_apply_patch(model, prop_attn=True, trace_source=False)
            model.r = [32] * 11 + [40]

    model = model.cuda().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_loaded_allocated = torch.cuda.memory_allocated()
    model_loaded_reserved = torch.cuda.memory_reserved()
    if args.compile_blocks:
        if args.method == "mergenet":
            from opentome.models.mergenet.inference import compile_transformer_blocks
            compile_transformer_blocks(
                model, compile_local_encoder=args.compile_local_encoder
            )
        else:
            if args.compile_local_encoder:
                raise ValueError("--compile-local-encoder requires MergeNet")
            for block in model.blocks:
                block.forward = torch.compile(block.forward, mode="default")
    x = torch.randn(args.batch, 3, 224, 224, device="cuda")
    torch.cuda.synchronize()
    own_count = len(gpu_apps())
    if own_count != 1:
        raise RuntimeError(f"Expected only this GPU process; saw {gpu_apps()}")

    observed = []
    if args.method in ("tome", "pitome"):
        hook = model.blocks[-1].register_forward_hook(
            lambda _mod, _inp, out: observed.append(out.shape[1])
        )
    else:
        hook = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        first = model(x)
    if hook:
        hook.remove()
    torch.cuda.synchronize()
    if args.method == "mergenet":
        actual_patches = int(first[1]["retained_tokens"])
    elif args.method == "dense":
        actual_patches = int(model.patch_embed.num_patches)
    else:
        actual_patches = int(observed[-1] - 1)
    target = 784 if args.method == "dense" else 392
    if actual_patches != target:
        raise RuntimeError(f"{args.method}: achieved {actual_patches}, expected {target}")
    del first

    def forward() -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            model(x)

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    forward()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    rounds = []
    for _ in range(args.rounds):
        events = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        rounds.append([start.elapsed_time(end) for start, end in events])
        if len(gpu_apps()) != own_count:
            raise RuntimeError("GPU process count changed during the probe")

    values = [v for run in rounds for v in run]
    ordered = sorted(values)
    round_medians = [statistics.median(run) for run in rounds]
    def percentile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    result = {
        "kind": "random_weight_architecture_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "checkpoint_provenance": "random_initialization",
        "code_root": str(args.code_root.resolve()),
        "batch": args.batch,
        "resolution": 224,
        "final_patches": actual_patches,
        "parameter_count": parameter_count,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm.__version__,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "precision": "fp16 autocast",
        "compiled_blocks": args.compile_blocks,
        "compiled_local_encoder": args.compile_local_encoder,
        "input": "device-resident random tensor",
        "prop_attn": args.method in ("tome", "pitome"),
        "warmup": args.warmup,
        "iters_per_round": args.iters,
        "rounds": args.rounds,
        "latency_median_ms": statistics.median(values),
        "latency_iqr_ms": ordered[3 * len(ordered) // 4] - ordered[len(ordered) // 4],
        "latency_p10_ms": percentile(0.1),
        "latency_p90_ms": percentile(0.9),
        "round_medians_ms": round_medians,
        "round_median_cv": (
            statistics.pstdev(round_medians) / statistics.mean(round_medians)
            if len(round_medians) > 1 else 0.0
        ),
        "throughput_images_s": args.batch * 1000 / statistics.median(values),
        "model_loaded_allocated_gib": model_loaded_allocated / 2**30,
        "model_loaded_reserved_gib": model_loaded_reserved / 2**30,
        "baseline_allocated_gib": baseline_allocated / 2**30,
        "peak_allocated_gib": peak_allocated / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
        "raw_latency_ms": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_latency_ms"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
