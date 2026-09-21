#!/usr/bin/env python3
"""Curate the 2026-09-19 ImageNet packet and completed CIFAR seed campaign."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHARE = Path("/liziqing/yukai/MergeNet/share_20260919/share")
CIFAR = Path("/liziqing/yukai/mergenet_local_campaign_20260915")
DATA = ROOT / "data"
RUNS = DATA / "runs"
META = pd.read_csv(SHARE / "00_summary/tables/RUN_METADATA.csv")
META = META.set_index("run")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sanitize_args(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith("/"):
            out[k] = "/".join(Path(v).parts[-2:]) if "checkpoint" in k else "<host-path-omitted>"
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def metric_from_summary(csv_path: Path) -> dict:
    d = pd.read_csv(csv_path)
    assert d.epoch.tolist() == list(range(len(d))), csv_path
    col = "eval_top1"
    if "eval_top1_full_compression" in d.columns:
        full = d["eval_top1_full_compression"].astype(float)
        # Curriculum rows log 0 until the ramp finishes.
        if (full > 0).any():
            # Prefer full-compression metric when it is defined; otherwise keep eval_top1.
            use = d.copy()
            use["report_top1"] = [
                float(fc) if fc > 0 else float(t1)
                for fc, t1 in zip(full, d.eval_top1)
            ]
            # Best among fully compressed epochs only, if any exist.
            compressed = use[full > 0]
            b = compressed.loc[compressed.report_top1.idxmax()]
            last = use.iloc[-1]
            return {
                "epochs_done": int(len(d)),
                "best_top1": float(b.report_top1),
                "best_epoch": int(b.epoch),
                "best_top5_at_best_top1": float(b.eval_top5),
                "final_top1": float(last.report_top1),
                "final_epoch": int(last.epoch),
                "metric_column": "eval_top1_full_compression_when_defined",
            }
    ix = d.eval_top1.idxmax()
    b = d.loc[ix]
    last = d.iloc[-1]
    return {
        "epochs_done": int(len(d)),
        "best_top1": float(b.eval_top1),
        "best_epoch": int(b.epoch),
        "best_top5_at_best_top1": float(b.eval_top5),
        "final_top1": float(last.eval_top1),
        "final_epoch": int(last.epoch),
        "metric_column": "eval_top1",
    }


def find_run_dir(run: str) -> Path:
    hits = list((SHARE / "10_runs").glob(f"*/{run}"))
    assert len(hits) == 1, (run, hits)
    return hits[0]


def protocol_row(run: str, summary: dict, args: dict, src_summary: str) -> dict:
    m = META.loc[run]
    target = int(m.epochs_target)
    done = int(summary["epochs_done"])
    if done >= target:
        status = "COMPLETE"
    elif done == 0:
        status = "NO_COMPLETED_EPOCH"
    else:
        status = "PARTIAL_SNAPSHOT"
    init = str(m.init_from)
    if init in {"scratch", "nan"}:
        init = ""
    elif init.endswith(".pth.tar") or init.endswith(".pth"):
        init = init
    return {
        "run": run,
        "model": str(m.model),
        "resolution": str(int(m.resolution)),
        "patch_size": str(int(m.patch)),
        "epochs_done": str(done),
        "epochs_target": str(target),
        "status": status,
        "best_top1": str(summary["best_top1"]),
        "best_epoch": str(summary["best_epoch"]),
        "best_top5_at_best_top1": str(summary["best_top5_at_best_top1"]),
        "final_top1": str(summary["final_top1"]),
        "final_epoch": str(summary["final_epoch"]),
        "seed": str(int(m.seed)),
        "world_size": str(int(m.world_size)),
        "batch_per_gpu": str(int(m.batch_per_gpu)),
        "update_freq": str(int(m.update_freq)),
        "effective_global_batch": str(int(m.global_batch)),
        "microsteps_per_epoch": str(int(m.micro_steps_per_epoch)),
        "optimizer_steps_per_epoch": str(int(m.opt_steps_per_epoch)),
        "lr": str(m.lr),
        "warmup_epochs": str(int(m.warmup_epochs)),
        "ema_decay": str(m.ema_decay),
        "crop_pct": str(m.crop_pct),
        "drop_path_rate": str(m.drop_path),
        "weight_decay": str(m.weight_decay),
        "dtem_spatial_radius": str(m.dtem_spatial_radius) if pd.notna(m.dtem_spatial_radius) else "",
        "initial_checkpoint": init,
        "source_summary": src_summary,
        "metric_column": summary["metric_column"],
        "img_size": str(args.get("img_size", m.resolution)),
    }


def ingest_imagenet(manifest: dict) -> dict:
    protocol = json.loads((DATA / "launch_protocol.json").read_text())
    checks = {}
    for run, m in META.iterrows():
        d = find_run_dir(run)
        out = RUNS / run
        out.mkdir(parents=True, exist_ok=True)
        summary_src = d / "summary.csv"
        shutil.copyfile(summary_src, out / "summary.csv")
        rel = str(summary_src.relative_to(SHARE))
        manifest["sources"].append(
            {"source": rel, "sha256": sha256(summary_src), "curated": f"runs/{run}/summary.csv"}
        )
        args_src = d / "args.yaml"
        cfg = load_yaml(args_src)
        (out / "args.json").write_text(json.dumps(sanitize_args(cfg), indent=2) + "\n")
        manifest["sources"].append(
            {"source": str(args_src.relative_to(SHARE)), "sha256": sha256(args_src), "curated": f"runs/{run}/args.json"}
        )
        summary = metric_from_summary(out / "summary.csv")
        protocol[run] = protocol_row(run, summary, cfg, rel)
        # Cross-check metadata table to three decimals.
        checks[run] = {
            "csv_best": summary["best_top1"],
            "meta_best": float(m.best_top1),
            "csv_final": summary["final_top1"],
            "meta_final": float(m.final_top1),
            "epochs": summary["epochs_done"],
            "meta_epochs": int(m.epochs_done),
        }
        assert abs(summary["best_top1"] - float(m.best_top1)) < 5e-4, (run, checks[run])
        assert abs(summary["final_top1"] - float(m.final_top1)) < 5e-4, (run, checks[run])
        assert summary["epochs_done"] == int(m.epochs_done), (run, checks[run])
    (DATA / "launch_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (DATA / "ingest_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return protocol


def ingest_sweep_diagnosis() -> dict:
    sweep = pd.read_csv(SHARE / "00_summary/TESTTIME_SWEEP.csv")
    sub = sweep[
        (sweep.model == "mergenet")
        & (sweep.resolution == 224)
        & (sweep.ckpt_set == "e300_224")
        & (~sweep.is_gate.astype(bool))
    ]
    recs = {}
    for _, r in sub.iterrows():
        recs[f"{r.method}_{int(r.achieved_tokens)}"] = {
            "method": r.method,
            "requested_tokens": int(r.requested_tokens),
            "achieved_tokens": int(r.achieved_tokens),
            "top1": float(r.top1),
            "top5": float(r.top5),
            "n_images": int(r.n_images),
            "tokens_match": bool(r.tokens_match),
        }
    (DATA / "inference_merge_probe.json").write_text(json.dumps(recs, indent=2) + "\n")
    return recs


def ingest_cifar(manifest: dict) -> None:
    dst = DATA / "cifar_seeds"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in sorted((CIFAR / "runs").iterdir()):
        summary = job / "summary.csv"
        assert summary.exists(), job
        d = pd.read_csv(summary)
        assert len(d) == 200 and d.epoch.tolist() == list(range(200)), job
        last = d.iloc[199]
        selector, geometry, seed = job.name.split("_")
        seed = int(seed[1:])
        rows.append(
            {
                "job": job.name,
                "selector": selector,
                "geometry": geometry,
                "seed": seed,
                "epoch199_ema_top1": float(last.eval_top1),
                "best_ema_top1": float(d.eval_top1.max()),
                "best_epoch": int(d.eval_top1.idxmax()),
            }
        )
        out = dst / "summaries" / job.name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary, out / "summary.csv")
        manifest["sources"].append(
            {
                "source": str(summary),
                "sha256": sha256(summary),
                "curated": f"cifar_seeds/summaries/{job.name}/summary.csv",
            }
        )
    ep = pd.DataFrame(rows).sort_values(["selector", "geometry", "seed"])
    ep.to_csv(dst / "endpoints.csv", index=False)
    # Recompute three-seed contrasts from epoch-199 only.
    contrasts = []
    for selector in ["historical", "rowwise"]:
        g = ep[ep.selector == selector].pivot(index="seed", columns="geometry", values="epoch199_ema_top1")
        for a, b, name in [("r3", "global", "r3-global"), ("r3", "flat8", "r3-flat8"), ("r3", "degree", "r3-degree")]:
            delta = g[a] - g[b]
            contrasts.append(
                {
                    "contrast": f"{selector}:{name}",
                    "n_seeds": int(delta.count()),
                    "mean_delta_pp": float(delta.mean()),
                    "std_delta_pp": float(delta.std(ddof=1)),
                    "seed42_delta": float(delta.loc[42]),
                    "seed43_delta": float(delta.loc[43]),
                    "seed44_delta": float(delta.loc[44]),
                }
            )
    for geometry in ["global", "flat8", "r3", "degree"]:
        h = ep[(ep.selector == "historical") & (ep.geometry == geometry)].set_index("seed").epoch199_ema_top1
        r = ep[(ep.selector == "rowwise") & (ep.geometry == geometry)].set_index("seed").epoch199_ema_top1
        delta = r - h
        contrasts.append(
            {
                "contrast": f"{geometry}:rowwise-historical",
                "n_seeds": int(delta.count()),
                "mean_delta_pp": float(delta.mean()),
                "std_delta_pp": float(delta.std(ddof=1)),
                "seed42_delta": float(delta.loc[42]),
                "seed43_delta": float(delta.loc[43]),
                "seed44_delta": float(delta.loc[44]),
            }
        )
    pd.DataFrame(contrasts).to_csv(dst / "contrasts.csv", index=False)
    means = (
        ep.groupby(["selector", "geometry"], as_index=False)
        .agg(
            mean_epoch199=("epoch199_ema_top1", "mean"),
            std_epoch199=("epoch199_ema_top1", lambda s: float(s.std(ddof=1))),
            n=("epoch199_ema_top1", "count"),
        )
    )
    means.to_csv(dst / "group_means.csv", index=False)
    shutil.copyfile(CIFAR / "protocol.json", dst / "protocol.json")
    # Confirm the campaign's published contrast CSV to 1e-8.
    published = pd.read_csv(CIFAR / "reports/three_seed_contrasts.csv").set_index("contrast")
    recomputed = pd.read_csv(dst / "contrasts.csv").set_index("contrast")
    for key in published.index:
        assert abs(published.loc[key, "mean_delta_pp"] - recomputed.loc[key, "mean_delta_pp"]) < 1e-10, key
        assert abs(published.loc[key, "std_delta_pp"] - recomputed.loc[key, "std_delta_pp"]) < 1e-10, key


def main() -> None:
    manifest = {
        "imagenet_archive": str(Path("/liziqing/yukai/MergeNet/share_mergenet_campaign_20260919.tar.gz")),
        "imagenet_archive_sha256": sha256(Path("/liziqing/yukai/MergeNet/share_mergenet_campaign_20260919.tar.gz")),
        "cifar_campaign": str(CIFAR),
        "note": "Summaries are byte copies. Host paths are stripped from args. No checkpoints are stored.",
        "sources": [],
        "cutoff": "2026-09-19",
    }
    protocol = ingest_imagenet(manifest)
    probe = ingest_sweep_diagnosis()
    ingest_cifar(manifest)
    (DATA / "manifest_20260919.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("runs", len(protocol))
    print("complete", sum(p["status"] == "COMPLETE" for p in protocol.values()))
    print("partial", [k for k, p in protocol.items() if p["status"] != "COMPLETE"])
    print("merge-none", probe.get("none_392"))
    print("merge-native", {k: probe[k] for k in probe if k.startswith("mergenet_")})


if __name__ == "__main__":
    main()
