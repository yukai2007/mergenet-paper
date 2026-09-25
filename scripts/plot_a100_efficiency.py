#!/usr/bin/env python3
"""Render the paper's A100 synthetic efficiency figure from probe output."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/local_gpu_probe_20260925_full_synthetic_gpu2/summary_eager.csv"
OUTPUT = ROOT / "figures/a100_efficiency.pdf"
ORDER = ("dense", "tome", "pitome", "mergenet")
LABELS = ("Dense", "ToMe", "PiToMe", "MergeNet")
COLORS = ("#81909e", "#47759c", "#72a89d", "#ca7d34")


def main() -> None:
    with SOURCE.open(newline="") as handle:
        records = {row["method"]: row for row in csv.DictReader(handle)}
    if set(records) != set(ORDER):
        raise ValueError(f"Expected methods {ORDER}, got {tuple(records)}")
    rows = [records[name] for name in ORDER]
    for name, row in zip(ORDER, rows):
        target = 784 if name == "dense" else 392
        if int(row["patches"]) != target:
            raise ValueError(f"{name} has {row['patches']} patches; expected {target}")

    latency = [float(row["b64_median_ms"]) for row in rows]
    memory = [float(row["b64_peak_allocated_gib"]) for row in rows]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.85))
    for ax, values, xlabel, limit, fmt in (
        (axes[0], latency, "B64 latency (ms)  ↓", 82, "{:.2f}"),
        (axes[1], memory, "B64 peak allocated (GiB)  ↓", 2.08, "{:.3f}"),
    ):
        bars = ax.barh(range(4), values, color=COLORS, edgecolor="#263238", linewidth=0.45, height=0.62)
        ax.set_yticks(range(4), LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color="#d8dce0", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for bar, value in zip(bars, values):
            ax.text(value + limit * 0.018, bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", ha="left", fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.29, top=0.98, wspace=0.36)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
