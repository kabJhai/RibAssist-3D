#!/usr/bin/env python3
"""Collate the L1 extraction-policy correspondence sweep into one comparison table.

Reads each policy's D0 (pair graph) + D1 (deterministic operational frontier) JSON from --sweep-dir and prints,
side by side, whether shrinking the lateral candidate field lifts the operational 3D-reconstruction frontier at
a controlled false-3D rate. Read-only; derives nothing new — it only surfaces the recorded headline fields.

The decisive columns: lateral peaks/case + lateral availability (what recalibration costs), broad-graph size +
positive prevalence + competitors/dual-view fracture (how much the flood shrank), and the D1 out-of-fold
operational recall@10 at <=1 false-3D/case vs uncapped (whether that shrink converts into reconstruction).

Usage:
  python collate_L1_sweep.py --sweep-dir outputs/L1_sweep
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

TAGS = ["deployed", "latN5_f055", "latN5_f060", "latN5_f065", "latN3_f060"]


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def cap_entry(d1, cap):
    for cs in d1.get("operational_cap_sweep_out_of_fold", []):
        if cs.get("false_3d_cap_per_case_at_10mm") == cap:
            return cs
    return None


def gate_row(d0, gate):
    for r in d0.get("gate_sensitivity_sweep", {}).get("rows", []):
        if abs(float(r.get("gate_vox", -1)) - gate) < 1e-6:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--gate", type=float, default=None, help="gate (vox) for competitor/prevalence column; default deployed si_tol")
    a = ap.parse_args()

    rows = []
    for tag in TAGS:
        d0 = load(a.sweep_dir / f"{tag}_D0.json"); d1 = load(a.sweep_dir / f"{tag}_D1.json")
        if d0 is None and d1 is None: continue
        rows.append((tag, d0, d1))
    if not rows:
        print(f"no policy JSONs found in {a.sweep_dir}"); return 1

    si_tol = None
    for _, d0, _ in rows:
        if d0: si_tol = float(d0.get("si_tol_voxels", 6.0)); break
    gate = a.gate if a.gate is not None else (si_tol if si_tol is not None else 6.0)

    print(f"\nL1 CORRESPONDENCE SWEEP — collation (competitor/prevalence @ gate {gate} vox; D1 out-of-fold operational)")
    print("=" * 118)
    hdr = ("policy", "lat_pk", "lat_rec", "dual%", "broad", "pos", "comp/dv", "R@10|c1", "fp@10", "R@10|none", "corr@10")
    print("{:<11} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>8} {:>6} {:>10} {:>8}".format(*hdr))
    print("-" * 118)
    for tag, d0, d1 in rows:
        lat_pk = lat_rec = dual = broad = pos = compdv = "-"
        if d0:
            pg = d0.get("pair_graph", {}); av = d0.get("correct_pair_availability_existential", {})
            lat_pk = pg.get("mean_lat_peaks_per_case", "-")
            lat_rec = av.get("compatible_lat_recall", "-")
            dual = av.get("existential_dual_view_frac", "-")
            broad = pg.get("pairs_in_broad_graph_npz", "-")
            gr = gate_row(d0, gate)
            if gr:
                pos = gr.get("positive_prevalence", "-"); compdv = gr.get("mean_competitors_per_dualview_fracture", "-")
        r10c1 = fp10 = r10none = corr10 = "-"
        if d1:
            h = d1.get("operational_headline_out_of_fold", {})
            r10c1 = h.get("recall10", "-"); fp10 = h.get("false_3d_per_case_at_10mm", "-")
            none = cap_entry(d1, None)
            r10none = none.get("recall10", "-") if none else "-"
            hc = d1.get("correspondence_diagnostic_out_of_fold", {})
            corr10 = hc.get("correct_iid_within10", "-")
        print("{:<11} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>8} {:>6} {:>10} {:>8}".format(
            tag, str(lat_pk), str(lat_rec), str(dual), str(broad), str(pos), str(compdv),
            str(r10c1), str(fp10), str(r10none), str(corr10)))
    print("=" * 118)
    print("legend: lat_pk=mean lateral peaks/case | lat_rec=compatible lateral recall | dual%=existential dual-view frac")
    print("        broad=edges in broad graph | pos=positive prevalence@gate | comp/dv=mean competitors per dual-view fracture")
    print("        R@10|c1=D1 out-of-fold operational recall@10mm at <=1 false-3D/case | fp@10=realized false-3D/case")
    print("        R@10|none=uncapped operational recall@10 | corr@10=correct-iid credited within 10mm (correspondence diagnostic)")
    print("\nDecisive read: if R@10|c1 stays ~flat while comp/dv drops sharply, real-vs-real ambiguity dominates -> L2/global.")
    print("               if R@10|c1 rises materially as the field shrinks, calibration was the binding lever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
