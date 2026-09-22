#!/usr/bin/env python3
"""Ingest the two completed Stage-7 records from the 2026-09-23 packet.

The DeiT-initialized run resumed on another host.  Its resumed ``args.yaml`` no
longer names the initialization checkpoint, so this script deliberately keeps
the already curated pre-resume args and records that evidence boundary in a
separate provenance file.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = DATA / "runs"
EXPECTED_ARCHIVE_SHA256 = "e672d28937e56bb545556c7ee875c555004287d74d2d1ac2d378e8af06e92211"
PREFIX = "share/10_runs/stage7_150e_224"


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_member(archive: tarfile.TarFile, member: str) -> bytes:
    info = archive.getmember(member)
    if not info.isfile():
        raise ValueError(f"not a regular file: {member}")
    handle = archive.extractfile(info)
    if handle is None:
        raise ValueError(f"cannot read: {member}")
    return handle.read()


def sanitize_args(config: dict) -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("/"):
            out[key] = "/".join(Path(value).parts[-2:]) if "checkpoint" in key else "<host-path-omitted>"
        else:
            out[key] = value
    return out


def summarize(blob: bytes) -> dict:
    frame = pd.read_csv(io.BytesIO(blob))
    assert frame.epoch.tolist() == list(range(len(frame)))
    assert frame.eval_top1.notna().all()
    best = frame.loc[frame.eval_top1.astype(float).idxmax()]
    final = frame.iloc[-1]
    return {
        "epochs_done": len(frame),
        "best_top1": float(best.eval_top1),
        "best_epoch": int(best.epoch),
        "best_top5_at_best_top1": float(best.eval_top5),
        "final_top1": float(final.eval_top1),
        "final_epoch": int(final.epoch),
    }


def protocol_row(run: str, summary: dict, config: dict, source: str, *, initial_checkpoint: str = "") -> dict:
    done = int(summary["epochs_done"])
    target = int(config["epochs"])
    return {
        "run": run,
        "model": str(config["model"]),
        "resolution": str(int(config["img_size"])),
        "patch_size": str(int(config["patch_size"])),
        "epochs_done": str(done),
        "epochs_target": str(target),
        "status": "COMPLETE" if done >= target else "PARTIAL_SNAPSHOT",
        "best_top1": str(summary["best_top1"]),
        "best_epoch": str(summary["best_epoch"]),
        "best_top5_at_best_top1": str(summary["best_top5_at_best_top1"]),
        "final_top1": str(summary["final_top1"]),
        "final_epoch": str(summary["final_epoch"]),
        "seed": str(int(config["seed"])),
        "world_size": "4",
        "batch_per_gpu": str(int(config["batch_size"])),
        "update_freq": str(int(config["update_freq"])),
        "effective_global_batch": str(4 * int(config["batch_size"]) * int(config["update_freq"])),
        "microsteps_per_epoch": "1251",
        "optimizer_steps_per_epoch": "1251",
        "lr": str(config["lr"]),
        "warmup_epochs": str(int(config["warmup_epochs"])),
        "ema_decay": str(config["model_ema_decay"]),
        "crop_pct": str(config["crop_pct"]),
        "drop_path_rate": str(config["drop_path_rate"]),
        "weight_decay": str(config["weight_decay"]),
        "dtem_spatial_radius": str(config["dtem_spatial_radius"]),
        "initial_checkpoint": initial_checkpoint,
        "source_summary": source,
        "metric_column": "eval_top1",
        "img_size": str(int(config["img_size"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    archive_sha = sha256_file(archive_path)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit(f"unexpected archive SHA-256: {archive_sha}")

    records = {}
    source_manifest = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for run in ["s7_mn_deitinit_150e", "s7_mn_proglatent_150e"]:
            summary_member = f"{PREFIX}/{run}/summary.csv"
            args_member = f"{PREFIX}/{run}/args.yaml"
            summary_blob = read_member(archive, summary_member)
            args_blob = read_member(archive, args_member)
            config = yaml.safe_load(args_blob)
            records[run] = (summary_blob, config)
            source_manifest.extend([
                {"source": summary_member, "sha256": sha256_bytes(summary_blob), "curated": f"runs/{run}/summary.csv"},
                {"source": args_member, "sha256": sha256_bytes(args_blob),
                 "curated": f"runs/{run}/args.json" if run.endswith("proglatent_150e") else "provenance only"},
            ])

    protocol = json.loads((DATA / "launch_protocol.json").read_text())
    for run, (summary_blob, config) in records.items():
        out = RUNS / run
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.csv").write_bytes(summary_blob)
        summary = summarize(summary_blob)
        source = f"10_runs/stage7_150e_224/{run}/summary.csv"
        if run == "s7_mn_deitinit_150e":
            prior = protocol[run]
            protocol[run] = protocol_row(
                run, summary, config, source,
                initial_checkpoint=prior["initial_checkpoint"],
            )
            provenance = {
                "archive": archive_path.name,
                "archive_sha256": archive_sha,
                "summary_sha256": sha256_bytes(summary_blob),
                "pre_migration_prefix_lines": 106,
                "pre_migration_prefix_md5": "9651b0d14ccf28bb0d1b31955f5effa7",
                "resume_started_at_epoch": 105,
                "curated_args_policy": "Preserve the pre-resume args.json because resumed args clear initial_checkpoint.",
                "evidence_limit": "The original first-launch checkpoint-loading log was not delivered; initialization is supported by the contemporaneous receipt and early learning curve, not independently replayed here.",
            }
            (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        else:
            (out / "args.json").write_text(json.dumps(sanitize_args(config), indent=2) + "\n")
            protocol[run] = protocol_row(run, summary, config, source)

    (DATA / "launch_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = {
        "archive": archive_path.name,
        "archive_sha256": archive_sha,
        "cutoff": "2026-09-23",
        "note": "Incremental Stage-7 refresh. Summaries are byte copies; no checkpoint is stored.",
        "sources": source_manifest,
    }
    (DATA / "manifest_20260923.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({run: summarize(blob) for run, (blob, _) in records.items()}, indent=2))


if __name__ == "__main__":
    main()
