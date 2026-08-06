#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Regenerate README/paper schematic SVGs in a publication color style.

Muted ColorBrewer-style palette (blue = frozen/baseline, orange = L2/outcome),
thin strokes, Arial labels. Run from repository root:

  python scripts/make_schematic_figures.py
  python scripts/make_schematic_figures.py --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

FONT = "Arial, Helvetica, sans-serif"
INK = "#222222"
MUTED = "#555555"
SW = "0.85"

# Frozen / baseline (blue)
BLUE = "#3182bd"
BLUE_FILL = "#deebf7"
# L2 / outcome (orange)
ORANGE = "#e6550d"
ORANGE_FILL = "#fee6ce"
# Neutral process blocks
NEUTRAL_FILL = "#ffffff"
NEUTRAL_STROKE = "#969696"
RULED_FILL = "#f7f7f7"
RULED_STROKE = "#bdbdbd"


def _svg_open(w: int, h: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="{FONT}" font-size="9" fill="{INK}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        "<defs>",
        f'<marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8 Z" fill="#666666"/></marker>',
        "</defs>",
    ]


def _t(x, y, text, *, size=9, weight="normal", anchor="start", fill=INK) -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(text)}</text>'
    )


def _box(x, y, w, h, lines: list[str], *, fill=NEUTRAL_FILL, stroke=NEUTRAL_STROKE, bold_first=True) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{SW}"/>'
    ]
    cy = y + 16
    for i, line in enumerate(lines):
        parts.append(
            _t(
                x + w / 2,
                cy + i * 13,
                line,
                size=8.5 if i else 9,
                weight="bold" if i == 0 and bold_first else "normal",
                anchor="middle",
                fill=INK if i == 0 else MUTED,
            )
        )
    return "\n".join(parts)


def _arrow(x1, y1, x2, y2) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#666666" '
        f'stroke-width="{SW}" marker-end="url(#arr)"/>'
    )


def fig_architecture() -> str:
    w, h = 920, 210
    lines = _svg_open(w, h)
    steps = [
        (("CT volume", ["RibFrac CT", "RibSeg seg/cl"]), BLUE_FILL, BLUE),
        (("Projection", ["Orthographic AP", "and lateral"]), NEUTRAL_FILL, NEUTRAL_STROKE),
        (("Detector", ["U-Net x2", "heatmaps"]), NEUTRAL_FILL, NEUTRAL_STROKE),
        (("Peaks", ["NMS + floor", "per view"]), NEUTRAL_FILL, NEUTRAL_STROKE),
        (("Pair graph", ["SI gate", "|dSI| <= gate"]), BLUE_FILL, BLUE),
        (("Assignment", ["One-to-one", "with abstention"]), BLUE_FILL, BLUE),
        (("Triangulation", ["Back-project", "3D points"]), ORANGE_FILL, ORANGE),
    ]
    bw, bh, gap = 108, 58, 14
    x0, y = 24, 72
    xs = []
    for i, ((title, sub), fill, stroke) in enumerate(steps):
        x = x0 + i * (bw + gap)
        xs.append(x + bw / 2)
        lines.append(_box(x, y, bw, bh, [title, *sub], fill=fill, stroke=stroke))
        if i:
            lines.append(_arrow(x - gap + 2, y + bh / 2, x - 2, y + bh / 2))
    lines.append(_t(w / 2, 28, "Biplanar detection and 3D localization pipeline", size=10, weight="bold", anchor="middle"))
    lines.append(_t(xs[3], y + bh + 18, "Stages D0-D1", size=8, anchor="middle", fill=BLUE))
    lines.append(_t(xs[5], y + bh + 18, "frozen policy", size=8, anchor="middle", fill=BLUE))
    lines.append("</svg>")
    return "\n".join(lines)


def fig_bottleneck() -> str:
    w, h = 680, 520
    lines = _svg_open(w, h)
    lines.append(_t(w / 2, 26, "Hypothesis elimination (development diagnostics)", size=10, weight="bold", anchor="middle"))
    rows = [
        ("1", "Projection geometry", "Stage A: 0.0 mm round-trip; triangulation exact.", False),
        ("2", "Detector localization", "Stage B: median ~4 mm given correct correspondence.", False),
        ("3", "Correspondence method", "D1 and D2: 0% recall@10 at <=1 false/case on frozen detections.", False),
        ("4", "Extraction calibration", "L1 sweep: cleaner field, frontier still 0%.", False),
        ("!", "Lateral availability", "L2 retraining raises dual-view availability and first commits.", True),
    ]
    y = 48
    for num, title, detail, highlight in rows:
        rh = 58 if highlight else 50
        if highlight:
            fill, stroke, sw = ORANGE_FILL, ORANGE, "1.4"
            num_fill = ORANGE
        else:
            fill, stroke, sw = RULED_FILL, RULED_STROKE, SW
            num_fill = BLUE
        lines.append(
            f'<rect x="40" y="{y}" width="600" height="{rh}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
        lines.append(
            f'<circle cx="58" cy="{y + 18}" r="11" fill="{num_fill}" stroke="none"/>'
        )
        lines.append(_t(58, y + 22, num, size=9, weight="bold", anchor="middle", fill="#ffffff"))
        lines.append(_t(78, y + 18, title, size=9.5, weight="bold"))
        lines.append(_t(78, y + 34, detail, size=8.5, fill=MUTED))
        if not highlight:
            lines.append(_t(610, y + 22, "ruled out", size=8, anchor="end", fill=MUTED))
        y += rh + 8
    lines.append(
        f'<rect x="40" y="{y + 4}" width="600" height="44" fill="{BLUE_FILL}" '
        f'stroke="{BLUE}" stroke-width="{SW}"/>'
    )
    lines.append(
        _t(
            w / 2,
            y + 22,
            "Sealed test: recall@10 2.50% (15/601) at 0.436 false-3D/case; CI 0.69%-4.48%",
            size=9,
            weight="bold",
            anchor="middle",
            fill=BLUE,
        )
    )
    lines.append(_t(w / 2, y + 36, "55 cases, 601 fractures; policy frozen on development", size=8, anchor="middle", fill=MUTED))
    lines.append("</svg>")
    return "\n".join(lines)


def fig_2x2_attribution() -> str:
    w, h = 520, 380
    lines = _svg_open(w, h)
    lines.append(
        _t(w / 2, 22, "Recall@10 at <=1 false-3D/case (development OOF)", size=10, weight="bold", anchor="middle")
    )
    lines.append(_t(248, 52, "D1 deterministic", size=9, weight="bold", anchor="middle", fill=BLUE))
    lines.append(_t(398, 52, "D2 learned", size=9, weight="bold", anchor="middle", fill=MUTED))
    lines.append(_t(78, 118, "Frozen", size=9, weight="bold", anchor="middle", fill=BLUE))
    lines.append(_t(78, 218, "L2", size=9, weight="bold", anchor="middle", fill=ORANGE))
    cells = [
        (188, 88, "0.0%", "0 / 492", BLUE_FILL, BLUE, False),
        (338, 88, "0.2%", "1 / 492", RULED_FILL, RULED_STROKE, False),
        (188, 188, "2.44%", "12 / 492", ORANGE_FILL, ORANGE, True),
        (338, 188, "0.0%", "0 / 492", RULED_FILL, RULED_STROKE, False),
    ]
    for x, y, val, sub, fill, stroke, hi in cells:
        sw = "1.4" if hi else SW
        val_color = ORANGE if hi else INK
        lines.append(
            f'<rect x="{x}" y="{y}" width="120" height="72" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
        lines.append(_t(x + 60, y + 32, val, size=14, weight="bold", anchor="middle", fill=val_color))
        lines.append(_t(x + 60, y + 50, sub, size=8, anchor="middle", fill=MUTED))
    lines.append(_t(w / 2, 290, "Detector change (frozen to L2, D1): 0.0% to 2.44%", size=8.5, anchor="middle", fill=ORANGE))
    lines.append(_t(w / 2, 308, "Correspondence change (D1 to D2, L2): 2.44% to 0.0%", size=8.5, anchor="middle", fill=MUTED))
    lines.append(
        _t(
            w / 2,
            330,
            "Sealed L2-D1: 2.50% (15/601); learned scorer does not beat detector confidence",
            size=8,
            anchor="middle",
            fill=BLUE,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _layer_row(
    lines: list[str],
    y: int,
    blocks: list[tuple[str, str, str, str, str]],
    x_start=40,
    bw=95,
    gap=12,
) -> int:
    x = x_start
    for title, shape, params, fill, stroke in blocks:
        lines.append(_box(x, y, bw, 46, [title, shape, params], fill=fill, stroke=stroke))
        x += bw + gap
    return y + 46


def fig_model_layers() -> str:
    w, h = 980, 620
    lines = _svg_open(w, h)
    lines.append(_t(40, 24, "(a) Per-view detector U-Net (base channels = 32)", size=10, weight="bold"))
    unet = [
        ("Input", "1 x 256^2", "projection", BLUE_FILL, BLUE),
        ("inc", "32 x 256^2", "9.7k", NEUTRAL_FILL, NEUTRAL_STROKE),
        ("d1", "64 x 128^2", "55.7k", BLUE_FILL, BLUE),
        ("d2", "128 x 64^2", "222k", NEUTRAL_FILL, NEUTRAL_STROKE),
        ("d3", "256 x 32^2", "886k", BLUE_FILL, BLUE),
        ("u3", "128 x 64^2", "591k", NEUTRAL_FILL, NEUTRAL_STROKE),
        ("u2", "64 x 128^2", "148k", BLUE_FILL, BLUE),
        ("u1", "32 x 256^2", "37k", NEUTRAL_FILL, NEUTRAL_STROKE),
        ("out", "1 x 256^2", "33", ORANGE_FILL, ORANGE),
    ]
    x = 40
    for i, trip in enumerate(unet):
        lines.append(_box(x, 38, 88, 50, list(trip[:3]), fill=trip[3], stroke=trip[4]))
        if i:
            lines.append(_arrow(x - 10, 63, x - 1, 63))
        x += 98
    lines.append(_t(40, 108, "Two independent copies (AP and lateral); ~1.95M params each.", size=8, fill=MUTED))

    lines.append(_t(40, 138, "(b) Addressing network (48.8k params)", size=10, weight="bold"))
    y = 152
    stream_blocks = [
        ("AP proj", "256^2", "", BLUE_FILL, BLUE),
        ("Conv pool x3", "64-d", "", NEUTRAL_FILL, NEUTRAL_STROKE),
        ("GAP", "64-d", "", NEUTRAL_FILL, NEUTRAL_STROKE),
    ]
    for label, tint in (("AP stream", BLUE), ("Lat stream", ORANGE)):
        lines.append(_t(40, y + 12, label, size=8.5, weight="bold", fill=tint))
        blocks = list(stream_blocks)
        if tint == ORANGE:
            blocks[0] = ("Lat proj", "256^2", "", ORANGE_FILL, ORANGE)
        _layer_row(lines, y + 18, blocks, x_start=110, bw=100, gap=10)
        y += 78
    lines.append(_box(520, 188, 120, 46, ["concat", "128-d", ""], fill=RULED_FILL, stroke=RULED_STROKE))
    lines.append(_box(660, 170, 130, 30, ["side head", "sigmoid", ""], fill=ORANGE_FILL, stroke=ORANGE))
    lines.append(_box(660, 205, 130, 30, ["rib head", "softmax 12", ""], fill=ORANGE_FILL, stroke=ORANGE))
    lines.append(_box(660, 240, 130, 30, ["quality head", "sigmoid", ""], fill=ORANGE_FILL, stroke=ORANGE))
    lines.append(_arrow(430, 211, 518, 211))
    lines.append(_arrow(640, 211, 658, 185))
    lines.append(_arrow(640, 211, 658, 220))
    lines.append(_arrow(640, 211, 658, 255))

    lines.append(_t(40, 320, "(c) Learned pair-scorer, D2b (12.6k params)", size=10, weight="bold"))
    _layer_row(
        lines,
        334,
        [
            ("AP crop", "40^2", "", BLUE_FILL, BLUE),
            ("Lat crop", "40^2", "", ORANGE_FILL, ORANGE),
            ("Tower", "32-d ea.", "shared", NEUTRAL_FILL, NEUTRAL_STROKE),
            ("Combine", "102-d", "geom+emb", NEUTRAL_FILL, NEUTRAL_STROKE),
            ("MLP", "64 hidden", "drop 0.4", NEUTRAL_FILL, NEUTRAL_STROKE),
            ("Score", "scalar", "", ORANGE_FILL, ORANGE),
        ],
        bw=88,
        gap=10,
    )
    lines.append(
        _t(
            40,
            400,
            "Pair-scorer used only for attribution; at the operational budget it does not beat detector confidence.",
            size=8,
            fill=MUTED,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


FIGURES = {
    "fig_architecture.svg": fig_architecture,
    "fig_bottleneck.svg": fig_bottleneck,
    "fig_2x2_attribution.svg": fig_2x2_attribution,
    "fig_model_layers.svg": fig_model_layers,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        path = args.out_dir / name
        path.write_text(builder() + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
