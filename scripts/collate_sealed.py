#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Paired sealed-test comparison: frozen detector vs L2 detector, deterministic D1, fixed policy.

Reads the two sealed D1 apply-policy JSONs (+ their sealed D0 JSONs for candidate-side diagnostics) and prints
the primary comparison the sealed pass exists to answer: does the L2 operational recall@10 at the frozen
≤1-false policy survive on untouched data, and by how much, with case-level bootstrap CIs and a paired per-case
change from frozen to L2.

Usage:
  python collate_sealed.py --sealed-dir outputs/sealed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None


def d0diag(d0):
    if not d0: return {}
    pg = d0.get("pair_graph", {}); av = d0.get("correct_pair_availability_existential", {})
    comp = "-"
    for r in d0.get("gate_sensitivity_sweep", {}).get("rows", []):
        if abs(float(r.get("gate_vox", -1)) - float(d0.get("si_tol_voxels", 6.0))) < 1e-6:
            comp = r.get("mean_competitors_per_dualview_fracture", "-")
    return {"ap_pk": pg.get("mean_ap_peaks_per_case"), "lat_pk": pg.get("mean_lat_peaks_per_case"),
            "ap_rec": av.get("compatible_ap_recall"), "lat_rec": av.get("compatible_lat_recall"),
            "dual": av.get("existential_dual_view_frac"), "comp": comp,
            "pos_prev": next((r.get("positive_prevalence") for r in d0.get("gate_sensitivity_sweep", {}).get("rows", [])
                              if abs(float(r.get("gate_vox", -1)) - float(d0.get("si_tol_voxels", 6.0))) < 1e-6), "-")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed-dir", type=Path, required=True)
    a = ap.parse_args()
    fr = load(a.sealed_dir / "frozen_sealed_D1.json"); l2 = load(a.sealed_dir / "L2_sealed_D1.json")
    fr0 = load(a.sealed_dir / "frozen_sealed_D0.json"); l20 = load(a.sealed_dir / "L2_sealed_D0.json")
    if fr is None or l2 is None:
        print(f"missing sealed D1 JSONs in {a.sealed_dir}"); return 1

    def row(tag, j, j0):
        h = j["operational_headline"]; ci = j["bootstrap95_case_level"]; pc = j["per_case"]
        cc = sum(1 for c in pc if pc[c]["matched10"] > 0)
        dg = d0diag(j0)
        print(f"  [{tag}] policy gate {int(j['frozen_policy']['gate'])} {j['frozen_policy']['cost']}"
              f"{'+mb' if j['frozen_policy']['mutual_best'] else ''} u={j['frozen_policy']['u']:.4f}")
        print(f"     recall 5/10/15mm: {h['recall5']}/{h['recall10']}/{h['recall15']} | matched@10 {h['n_matched_within10']}/{j['gt_fractures']} GT")
        print(f"     recall@10 bootstrap95 {ci['recall10']} | false@10 {h['false_3d_per_case_at_10mm']}/case {ci['false_3d_per_case_at_10mm']} | cases w/≥1 correct {cc}/{j['val_cases']}")
        print(f"     matched-point quality: median/p90 mm {h['median_mm_matched_within10']}/{h['p90_mm_matched_within10']} | rib-exact {h['rib_exact_matched_within10']} rib±1 {h['rib_within1_matched_within10']}")
        print(f"     candidate-side: ap_pk {dg['ap_pk']} lat_pk {dg['lat_pk']} | ap_rec {dg['ap_rec']} lat_rec {dg['lat_rec']} | dual {dg['dual']} | comp/dv {dg['comp']} | cand-ceiling@10 {j['candidate_ceiling_at_policy_gate']['within10']}")

    print("\nSEALED-TEST CONFIRMATION — frozen vs L2 (deterministic D1, frozen policy, untouched cohort)")
    print("=" * 100)
    row("frozen", fr, fr0)
    print("-" * 100)
    row("L2", l2, l20)
    print("=" * 100)

    # paired per-case change frozen -> L2 (matched@10)
    pf = fr["per_case"]; pl = l2["per_case"]; cases = sorted(set(pf) & set(pl))
    imp = wor = same = 0; net = 0
    for c in cases:
        dm = pl[c]["matched10"] - pf[c]["matched10"]; net += dm
        if dm > 0: imp += 1
        elif dm < 0: wor += 1
        else: same += 1
    print(f"paired per-case (matched@10, frozen→L2 over {len(cases)} cases): improved {imp}, worse {wor}, unchanged {same} | "
          f"net Δ matched {net:+d}")
    fr_tot = sum(pf[c]['matched10'] for c in cases); l2_tot = sum(pl[c]['matched10'] for c in cases)
    print(f"  total matched@10: frozen {fr_tot} → L2 {l2_tot}  (of {fr['gt_fractures']} GT)")
    print("\nDecision: L2 nonzero recall@10 at ≤1 false-3D/case with CI excluding a frozen tie ⇒ detector-direction")
    print("          hypothesis CONFIRMED on sealed data (still far from usable). Both zero ⇒ dev gain did not generalize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
