#!/usr/bin/env python3
"""Create the raster reference figure supplied for the MergeNet architecture.

This is intentionally a directly usable image asset.  A later pass can replace
it with a fully editable vector drawing without changing the manuscript input.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H = 2306, 1248
OUT = Path(__file__).resolve().parents[1] / "figures" / "model_architecture_reference.png"
FONT_DIR = Path("/liziqing/yukai/.local/opt/libreoffice-26.2.5/opt/libreoffice26.2/share/fonts/truetype")
REGULAR = FONT_DIR / "LiberationSerif-Regular.ttf"
BOLD = FONT_DIR / "LiberationSerif-Bold.ttf"
BLUE = "#3561a4"
PALE = "#fffdfc"


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size)


def text_center(draw, xy, text, size, fill="#050505", bold=False):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor="mm", align="center")


def rounded(draw, xy, fill, outline=BLUE, width=3, radius=12):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def gradient_box(im, xy, colors, outline=BLUE, width=3, radius=12):
    x0, y0, x1, y1 = xy
    cols = np.asarray([tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in colors], dtype=float)
    n = max(1, x1 - x0)
    xs = np.linspace(0, 1, n)
    stops = np.linspace(0, 1, len(cols))
    row = np.stack([np.interp(xs, stops, cols[:, i]) for i in range(3)], axis=1)
    arr = np.repeat(row[None, :, :], max(1, y1 - y0), axis=0).astype(np.uint8)
    patch = Image.fromarray(arr, mode="RGB")
    mask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, patch.width - 1, patch.height - 1), radius=radius, fill=255)
    im.paste(patch, (x0, y0), mask)
    ImageDraw.Draw(im).rounded_rectangle(xy, radius=radius, outline=outline, width=width)


def trapezoid(draw, xy, fill, outline=BLUE, width=3, inset=95):
    x0, y0, x1, y1 = xy
    pts = [(x0 + inset, y0), (x1 - inset, y0), (x1, y1), (x0, y1)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=width, joint="curve")


def dashed_rect(draw, xy, dash=18, gap=15, fill=BLUE, width=5):
    x0, y0, x1, y1 = xy
    for x in range(x0, x1, dash + gap):
        draw.line((x, y0, min(x + dash, x1), y0), fill=fill, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=fill, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=fill, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=fill, width=width)


def arrow(draw, points, fill="#111111", width=3, head=16):
    draw.line(points, fill=fill, width=width, joint="curve")
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    norm = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / norm, (y1 - y0) / norm
    px, py = -uy, ux
    tri = [(x1, y1), (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55),
           (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)]
    draw.polygon(tri, fill=fill)


def token_row(draw, x, y, text, cell_w=40, cell_h=54, gap=7, fill="#ffffff", selected=()):
    selected = set(selected)
    for i, ch in enumerate(text):
        x0 = x + i * (cell_w + gap)
        rounded(draw, (x0, y, x0 + cell_w, y + cell_h), fill="#fffefa", width=3, radius=8)
        if i in selected:
            draw.rounded_rectangle((x0 + 3, y + 3, x0 + cell_w - 3, y + cell_h - 3), radius=6, fill="#ffc21c")
            draw.rounded_rectangle((x0, y, x0 + cell_w, y + cell_h), radius=8, outline=BLUE, width=3)
        text_center(draw, (x0 + cell_w / 2, y + cell_h / 2 + 1), ch, 28)


def small_cells(draw, x, y, n=18, cell_w=29, cell_h=40, gap=10, selected=(), fill="#fff8d9"):
    selected = set(selected)
    for i in range(n):
        x0 = x + i * (cell_w + gap)
        rounded(draw, (x0, y, x0 + cell_w, y + cell_h), fill="#ffc21c" if i in selected else fill,
                width=3, radius=6)


def main():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    # Left local-to-latent path.
    token_row(d, 32, 38, "#Byte_level_sentence", cell_w=40, cell_h=55, gap=7)
    trapezoid(d, (38, 108, 830, 271), "#fcf3f3", inset=145)
    text_center(d, (434, 183), "Local Encoder", 62, bold=True)
    for i, colors in enumerate((("#ceb5ee", "#fafafa"), ("#ffbcb9", "#fffefe"),
                                ("#d8f2c0", "#ffffff"), ("#92cdf4", "#ffffff"))):
        x = 265 + i * 94
        gradient_box(im, (x, 287, x + 58, 404), colors, radius=12)
    gradient_box(im, (238, 418, 708, 680), ("#d8a6dc", "#f6d5ba", "#e2f4cf", "#99ccff", "#e8a2f1"), radius=0)
    text_center(d, (473, 550), "Latent GPT", 59, bold=True)
    for i, colors in enumerate((("#ffbcb9", "#ffffff"), ("#d8f2c0", "#ffffff"),
                                ("#92cdf4", "#ffffff"), ("#f0b0ee", "#ffffff"))):
        x = 271 + i * 95
        gradient_box(im, (x, 700, x + 58, 814), colors, radius=12)
    trapezoid(d, (38, 834, 830, 997), "#fff9e1", inset=145)
    text_center(d, (434, 915), "Local Decoder", 62, bold=True)
    token_row(d, 32, 1010, "Byte_level_sentence", cell_w=40, cell_h=55, gap=7)

    # Right-hand routed/latent computation path.
    dashed_rect(d, (866, 63, 2003, 662), width=5)
    dashed_rect(d, (866, 661, 2003, 1053), width=5)
    token_row(d, 955, 87, "#Byte_level_sentence", cell_w=28, cell_h=39, gap=8)
    trapezoid(d, (955, 139, 1754, 197), "#ffe6a0", inset=115)
    text_center(d, (1355, 168), "Local Transformer", 30)
    small_cells(d, 958, 208, n=18, cell_w=28, cell_h=39, gap=10, selected=())
    trapezoid(d, (955, 254, 1754, 313), "#adc2e5", inset=115)
    text_center(d, (1355, 284), "Dynamic merging module", 30)
    # Grouped candidate tokens and the selected subset.
    small_cells(d, 957, 326, n=18, cell_w=28, cell_h=39, gap=10, selected=(0, 1, 8, 13))
    d.rounded_rectangle((948, 318, 1159, 375), radius=12, outline="#111111", width=4)
    d.rounded_rectangle((1167, 318, 1416, 375), radius=12, outline="#111111", width=4)
    d.rounded_rectangle((1425, 318, 1761, 375), radius=12, outline="#111111", width=4)
    small_cells(d, 958, 401, n=18, cell_w=28, cell_h=39, gap=10, selected=())
    text_center(d, (1778, 421), "KV", 31)
    rounded(d, (957, 455, 1754, 570), "#d9efca", width=3, radius=0)
    text_center(d, (1355, 512), "Cross attention", 36)
    for x, colors in ((960, ("#ceb5ee", "#ffffff")), (1063, ("#ffbcb9", "#ffffff")),
                      (1274, ("#d8f2c0", "#ffffff")), (1583, ("#92cdf4", "#ffffff"))):
        gradient_box(im, (x, 580, x + 28, 643), colors, radius=6)
    rounded(d, (1828, 450, 1985, 578), "#d8beec", width=3, radius=25)
    text_center(d, (1906, 514), "Top-K\nselection", 31)
    # lower half
    token_row(d, 957, 681, "Byte_level_sentence", cell_w=28, cell_h=40, gap=8)
    trapezoid(d, (955, 730, 1754, 788), "#ffe6a0", inset=115)
    text_center(d, (1355, 760), "Local Transformer", 30)
    small_cells(d, 958, 800, n=18, cell_w=28, cell_h=39, gap=10, fill="#f7c8aa")
    rounded(d, (957, 854, 1754, 969), "#d9efca", width=3, radius=0)
    text_center(d, (1355, 912), "Cross attention", 36)
    token_row(d, 957, 988, "Byte_level_sentence", cell_w=28, cell_h=40, gap=8)
    # KV legend.
    rounded(d, (1778, 828, 1862, 1038), "#efefef", outline="#eeeeee", width=1, radius=8)
    for i, c in enumerate(("#ffbcb9", "#d8f2c0", "#92cdf4", "#f0b0ee")):
        d.rounded_rectangle((1790, 858 + i * 34, 1850, 879 + i * 34), radius=4, fill=c, outline=BLUE, width=2)
    text_center(d, (1820, 1012), "KV", 29)
    # Flow arrows.
    arrow(d, [(830, 185), (913, 185), (949, 225)], width=3)
    arrow(d, [(1761, 352), (1904, 352), (1904, 450)], width=3)
    arrow(d, [(1770, 421), (1810, 421), (1810, 450)], width=3)
    arrow(d, [(953, 344), (921, 344), (953, 420)], width=3)
    arrow(d, [(952, 420), (921, 420), (921, 700), (958, 700)], width=3)
    arrow(d, [(1754, 571), (1754, 590), (1585, 590)], width=3)
    arrow(d, [(1355, 570), (1355, 730)], width=3)
    arrow(d, [(1355, 969), (1355, 988)], width=3)

    im.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
