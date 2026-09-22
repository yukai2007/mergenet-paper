#!/usr/bin/env python3
"""Recreate the project model diagram supplied as slide 3 of 模型.pptx.

The source deck uses a very large square canvas and many individually drawn
PowerPoint shapes.  This script keeps the slide's block structure and labels in
a compact vector figure that can be included in the anonymous manuscript.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parents[1] / "figures" / "model_architecture_slide3.pdf"


def rounded(ax, x, y, w, h, color, text, *, fs=10, lw=0.8, ec="#606060",
            radius=0.04, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color="#1f1f1f", linespacing=1.0)


def arrow(ax, x0, y0, x1, y1, color="#303030", lw=0.9, style="-|>",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
    ))


def token_row(ax, x, y, w, h, n=16, selected=(), edge="#a7a7a7"):
    gap = w * 0.006
    cell = (w - gap * (n - 1)) / n
    selected = set(selected)
    for i in range(n):
        color = "#ffc319" if i in selected else "#fffbea"
        ax.add_patch(FancyBboxPatch(
            (x + i * (cell + gap), y), cell, h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=color, edgecolor=edge, linewidth=0.35,
        ))


def main():
    # Coordinates follow slide 3's central crop: x=10.2--25.7, y=15.6--24.0.
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Coordinate helpers for the slide crop.
    x0, x1 = 10.2, 25.7
    y0, y1 = 15.55, 24.05
    sx = lambda x: (x - x0) / (x1 - x0)
    sy = lambda y: 1.0 - (y - y0) / (y1 - y0)
    box = lambda x, y, w, h: (sx(x), sy(y + h), sx(x + w) - sx(x), sy(y) - sy(y + h))

    # Right-hand stages from slide 3.
    ux, uy, uw, uh = box(16.75, 15.96, 8.86, 4.78)
    lx, ly, lw, lh = box(16.75, 20.63, 8.86, 3.05)
    ax.add_patch(Rectangle((ux, uy), uw, uh, facecolor="#fbf3f3",
                           edgecolor="#925c5c", linewidth=0.9))
    ax.add_patch(Rectangle((lx, ly), lw, lh, facecolor="#fff9e1",
                           edgecolor="#9b8e62", linewidth=0.9))

    # Left local encoder / latent / decoder column.
    ex, ey, ew, eh = box(10.28, 16.30, 6.16, 1.264)
    ax.add_patch(Polygon([(ex, ey + eh * 0.08), (ex + ew, ey),
                          (ex + ew, ey + eh * 0.92), (ex, ey + eh)],
                         closed=True, facecolor="#fbf3f3",
                         edgecolor="#9a7777", linewidth=0.9))
    ax.text(ex + ew / 2, ey + eh / 2, "Local Encoder", ha="center", va="center",
            fontsize=11, weight="bold")
    dx, dy, dw, dh = box(10.29, 21.98, 6.16, 1.264)
    ax.add_patch(Polygon([(dx, dy + dh * 0.08), (dx + dw, dy),
                          (dx + dw, dy + dh * 0.92), (dx, dy + dh)],
                         closed=True, facecolor="#fff9e1",
                         edgecolor="#9b8e62", linewidth=0.9))
    ax.text(dx + dw / 2, dy + dh / 2, "Local Decoder", ha="center", va="center",
            fontsize=11, weight="bold")
    gx, gy, gw, gh = box(11.84, 18.73, 3.06, 2.04)
    rounded(ax, gx, gy, gw, gh, "#e7eef8", "Latent GPT", fs=11, lw=0.8,
            ec="#6f8da8", radius=0.04, weight="bold")

    # Input and output byte-level token strips.
    ix, iy, iw, ih = box(10.2, 15.72, 6.28, 0.39)
    token_row(ax, ix, iy, iw, ih, n=17, selected=(1, 6, 12))
    ox, oy, ow, oh = box(10.2, 23.34, 6.28, 0.39)
    token_row(ax, ox, oy, ow, oh, n=17, selected=(2, 9, 15))
    ax.text(ix + iw / 2, iy + ih + 0.013, "byte-level sentence",
            ha="center", va="bottom", fontsize=7, color="#555555")
    ax.text(ox + ow / 2, oy - 0.014, "byte-level sentence",
            ha="center", va="top", fontsize=7, color="#555555")

    # Upper branch: local transformer, dynamic merging, and cross-attention.
    bx, by, bw, bh = box(17.46, 16.55, 6.24, 0.45)
    ax.add_patch(Polygon([(bx + 0.01, by + bh), (bx + bw, by + bh),
                          (bx + bw - 0.02, by), (bx, by)],
                         closed=True, facecolor="#f1d4c8",
                         edgecolor="#9b7770", linewidth=0.7))
    ax.text(bx + bw / 2, by + bh / 2, "Local Transformer", ha="center", va="center",
            fontsize=8.3, weight="bold")
    mx, my, mw, mh = box(17.46, 17.45, 6.24, 0.45)
    ax.add_patch(Polygon([(mx + 0.01, my + mh), (mx + mw, my + mh),
                          (mx + mw - 0.02, my), (mx, my)],
                         closed=True, facecolor="#f5d6bd",
                         edgecolor="#9b7770", linewidth=0.7))
    ax.text(mx + mw / 2, my + mh / 2, "Dynamic merging module", ha="center", va="center",
            fontsize=8.0, weight="bold")
    token_row(ax, *box(17.44, 18.06, 6.28, 0.37), n=18, selected=(0, 6, 11))
    rounded(ax, *box(17.46, 19.03, 6.24, 0.90), "#f4c7c0",
            "Cross attention", fs=10, lw=0.8, ec="#9b7770", radius=0.04,
            weight="bold")

    # Lower branch: local transformer and cross-attention recovery.
    bx2, by2, bw2, bh2 = box(17.45, 21.17, 6.24, 0.45)
    ax.add_patch(Polygon([(bx2 + 0.01, by2 + bh2), (bx2 + bw2, by2 + bh2),
                          (bx2 + bw2 - 0.02, by2), (bx2, by2)],
                         closed=True, facecolor="#f4dca0",
                         edgecolor="#9b8e62", linewidth=0.7))
    ax.text(bx2 + bw2 / 2, by2 + bh2 / 2, "Local Transformer", ha="center", va="center",
            fontsize=8.3, weight="bold")
    token_row(ax, *box(17.44, 20.77, 6.28, 0.37), n=18, selected=())
    rounded(ax, *box(17.46, 22.14, 6.24, 0.90), "#f8e2a6",
            "Cross attention", fs=10, lw=0.8, ec="#9b8e62", radius=0.04,
            weight="bold")

    # Hard selection stage.
    tx, ty, tw, th = box(24.28, 18.99, 1.20, 0.99)
    rounded(ax, tx, ty, tw, th, "#d8beec", "Top-K\nselection", fs=9.2,
            lw=0.9, ec="#76598b", radius=0.05, weight="bold")

    # Main information-flow arrows, matching the slide's block-level routing.
    arrow(ax, ex + ew, ey + eh / 2, bx - 0.01, by + bh / 2)
    arrow(ax, mx + mw, my + mh / 2, tx - 0.01, ty + th / 2)
    arrow(ax, tx + tw / 2, ty, mx + mw * 0.90, my + mh * 0.05)
    arrow(ax, bx + bw / 2, by + bh, mx + mw / 2, my + mh)
    arrow(ax, mx + mw / 2, my, token_row_x := sx(20.8), token_row_y := sy(18.06),
          color="#bd6d24", lw=0.8)
    arrow(ax, token_row_x, token_row_y, gx + gw, gy + gh * 0.55,
          color="#4b779e", lw=0.8)
    arrow(ax, gx + gw * 0.5, gy, bx2 + bw2 * 0.5, by2 + bh2,
          color="#4b779e", lw=0.8)
    arrow(ax, bx2 + bw2 / 2, by2, lx + lw * 0.52, ly + lh,
          color="#9b8e62", lw=0.8)
    arrow(ax, lx + lw * 0.52, ly, gx + gw * 0.48, gy,
          color="#9b8e62", lw=0.8)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
