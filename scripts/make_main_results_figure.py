#!/usr/bin/env python3
"""RibAssist 3D MAIN RESULTS figure — sealed frozen-vs-L2, operational outcome + candidate-graph mechanism.

Reads the four sealed JSONs (frozen/L2 x D0/D1) and renders one publication figure as SMALL MULTIPLES — one
mini-panel per metric, each an independent scale (the metrics span very different ranges, so a shared axis would
be a dual-scale error). Two groups: the operational outcome (recall@10 at <=1 false/case, false-3D/case) and the
candidate-graph mechanism that produced it (lateral compatible recall, dual-view availability, candidate
ceiling@10). Numbers come from the JSONs, not hardcoded, so the figure is reproducible and provenance-linked.

Colors: frozen=#2a78d6 (blue), L2=#eb6834 (orange) — a validator-passing categorical pair (avoids status green);
identity is carried by a legend + direct value labels on every bar, never color alone.

Usage (from RibAssist 3D ROOT):
  python scripts/make_main_results_figure.py --sealed-dir outputs/sealed --out outputs/figures/main_results.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FROZEN_C = "#2a78d6"; L2_C = "#eb6834"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e6e3"


def load(p): return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr1 = load(a.sealed_dir / "frozen_sealed_D1.json"); l21 = load(a.sealed_dir / "L2_sealed_D1.json")
    fr0 = load(a.sealed_dir / "frozen_sealed_D0.json"); l20 = load(a.sealed_dir / "L2_sealed_D0.json")

    hF, hL = fr1["operational_headline"], l21["operational_headline"]
    ciL = l21["bootstrap95_case_level"]["recall10"]
    avF = fr0["correct_pair_availability_existential"]; avL = l20["correct_pair_availability_existential"]
    ngt = l21["gt_fractures"]; ncase = l21["val_cases"]

    # (title, frozen, L2, formatter, group, extra)  — extra: ("ci", lo, hi) for the recall panel
    def pct(x): return f"{x*100:.2f}%"
    def f2(x): return f"{x:.3f}"
    def per(x): return f"{x:.2f}"
    panels = [
        ("Recall@10\n(≤1 false-3D/case)", hF["recall10"], hL["recall10"], pct, "OPERATIONAL OUTCOME",
         {"ci": ciL, "note": f"{hL['n_matched_within10']}/{ngt} GT"}),
        ("False 3D pts / case\n@10 mm", hF["false_3d_per_case_at_10mm"], hL["false_3d_per_case_at_10mm"], per, "OPERATIONAL OUTCOME", {}),
        ("Lateral compatible\nrecall", avF["compatible_lat_recall"], avL["compatible_lat_recall"], f2, "CANDIDATE-GRAPH MECHANISM (why)", {}),
        ("Dual-view\navailability", avF["existential_dual_view_frac"], avL["existential_dual_view_frac"], f2, "CANDIDATE-GRAPH MECHANISM (why)", {}),
        ("Candidate ceiling\n@10 mm", fr1["candidate_ceiling_at_policy_gate"]["within10"], l21["candidate_ceiling_at_policy_gate"]["within10"], f2, "CANDIDATE-GRAPH MECHANISM (why)", {}),
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 5.3)); fig.patch.set_facecolor("white")
    for ax, (title, vF, vL, fmt, group, extra) in zip(axes, panels):
        vmax = max(vF, vL, extra.get("ci", [0, 0])[1] if "ci" in extra else 0) or 1.0
        top = vmax * 1.35 + 1e-9
        bars = ax.bar([0, 1], [vF, vL], width=0.62, color=[FROZEN_C, L2_C], zorder=3)
        if "ci" in extra:
            lo, hi = extra["ci"]
            ax.errorbar([1], [vL], yerr=[[vL - lo], [hi - vL]], fmt="none", ecolor=INK2, elinewidth=1.4, capsize=4, zorder=4)
        ax.text(0, vF + top * 0.03, fmt(vF), ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        lbl_y = (extra["ci"][1] if "ci" in extra else vL) + top * 0.03
        ax.text(1, lbl_y, fmt(vL), ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        if extra.get("note"):
            ax.text(1, lbl_y + top * 0.07, extra["note"], ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_ylim(0, top); ax.set_xlim(-0.7, 1.7); ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen", "L2"], fontsize=9, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_yticks([]); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.68, bottom=0.20, wspace=0.42)
    # title + group labels (well above the panel titles, which sit just over the axes at top=0.68)
    fig.text(0.5, 0.955, "Sealed-test result — frozen detector → L2 retraining",
             ha="center", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.5, 0.905, f"{ncase} sealed cases, {ngt} GT fractures · frozen policy, no test-set reselection",
             ha="center", fontsize=10, color=INK2)
    # group labels centered over their panel spans (panels 1–2 vs 3–5), with an underline rule
    ax12 = 0.035 + (0.985 - 0.035) * (2 / n) / 2
    ax345 = 0.035 + (0.985 - 0.035) * (2 / n + 3 / n) / 2
    fig.text(ax12, 0.80, "OPERATIONAL OUTCOME", ha="center", fontsize=10, color=INK, fontweight="bold")
    fig.text(ax345, 0.80, "CANDIDATE-GRAPH MECHANISM  (why it moved)", ha="center", fontsize=10, color=INK, fontweight="bold")
    fig.add_artist(plt.Line2D([0.035, 0.035 + (0.985 - 0.035) * 2 / n - 0.02], [0.775, 0.775], color=GRID, lw=1.2))
    fig.add_artist(plt.Line2D([0.035 + (0.985 - 0.035) * 2 / n + 0.01, 0.985], [0.775, 0.775], color=GRID, lw=1.2))
    # legend (identity carried here AND on x-ticks — never color-alone)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=FROZEN_C), Patch(color=L2_C)],
               ["frozen detector  (lat nms3/floor0.06, abstain policy)", "L2 detector  (lat nms3/floor0.10, gate30/geomean/u=0.417)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.065))
    fig.text(0.5, 0.02, f"recall@10 case-bootstrap 95% CI {ciL} (excludes 0)   ·   data_sha {l21['data_sha256'][:12]}..",
             ha="center", fontsize=8, color=INK2)
    a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {a.out}")
    print(f"  recall@10 {pct(hF['recall10'])} -> {pct(hL['recall10'])} | false/case {per(hF['false_3d_per_case_at_10mm'])} -> {per(hL['false_3d_per_case_at_10mm'])}")
    print(f"  lat_rec {avF['compatible_lat_recall']} -> {avL['compatible_lat_recall']} | dual {avF['existential_dual_view_frac']} -> {avL['existential_dual_view_frac']} "
          f"| ceiling@10 {fr1['candidate_ceiling_at_policy_gate']['within10']} -> {l21['candidate_ceiling_at_policy_gate']['within10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
