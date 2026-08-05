#!/usr/bin/env python3
"""Bounded, DEVELOPMENT-ONLY calibration of the biplanar fusion association layer — NO retraining,
NO new seed, NO sealed-test access. The detector weights are FIXED; only the unmatched-lateral score
gate (lat_gate) of the fusion candidate set is swept.

Motivation: with the ungated union (lat_gate=0), unmatched lateral single-view candidates add recall
at high FP budgets but pollute the low-FP ranking, so fusion can dip BELOW AP at the 1 FP target and
thereby FAIL the pre-declared primary-condition rule (fusion must be >= AP at every FROC target AND
have higher AUPRC or recall). This sweeps a fixed lat_gate grid over the SAME frozen dev-internal
validation slice and picks, by the pre-declared objective, the gate that:

    maximizes fusion AUPRC  SUBJECT TO  fusion sensitivity >= AP sensitivity at EVERY FROC target.

That enforces the already-written rule instead of inventing a post-hoc criterion. It reuses
train_detector's model/peaks/matching/FROC verbatim (identical fusion semantics as the sealed test).

Usage:
  python calibrate_fusion.py --dev-run outputs/detector_dev_scratch_c32_both \
      --data outputs/det_out_v2/det_dev.npz
  # (optional custom grid)  --gates 0,0.05,0.07,0.08,0.09,0.10,0.11,0.12
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import train_detector as T

DEFAULT_GATES = (0.0, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-run", type=Path, required=True, help="a --views both dev run (detector_ap.pt + detector_lat.pt)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gates", default=None, help="comma-separated lat_gate grid (default 0,0.05,0.07,0.08,0.09,0.10,0.11,0.12)")
    a = ap.parse_args()
    if torch is None: print("pip install torch scipy", file=sys.stderr); return 1
    gates = tuple(float(x) for x in a.gates.split(",")) if a.gates else DEFAULT_GATES
    if 0.0 not in gates: gates = (0.0,) + gates   # always include the ungated baseline

    rec = json.loads((a.dev_run / "detector_dev_run.json").read_text())
    ep = rec["eval_params"]; base_ch, si_tol, boot, seed = ep.get("base_ch"), ep["si_tol"], ep["bootstrap"], ep["seed"]
    for v in ("ap", "lat"):
        if not (a.dev_run / f"detector_{v}.pt").exists():
            print(f"calibrate_fusion needs BOTH views; missing detector_{v}.pt in {a.dev_run}", file=sys.stderr); return 1
    d, man, _ = T.load_dev(a.data); dev = T.device()
    arch = T.arch_from_record(rec)
    nets = {}
    for v in ("ap", "lat"):
        net = T.build_detector(arch, pretrained=False).to(dev)
        net.load_state_dict(torch.load(a.dev_run / f"detector_{v}.pt", map_location=dev)); net.eval(); nets[v] = net
    val_ids = rec["split"]["val_case_ids"]; c2i = {str(c): i for i, c in enumerate(d["case"])}
    va_idx = np.array([c2i[c] for c in val_ids])
    ap_g, lat_g = T.group_instances(d, "ap"), T.group_instances(d, "lat")
    cache = T.peak_cache(nets, d, va_idx, dev)

    # AP is the fixed reference the pre-declared rule compares against.
    ap_res = T.eval_condition("ap", cache, ap_g, lat_g, va_idx, si_tol, boot, seed)
    print(f"AP reference (c32): " + "  ".join(f"@{t}FP {ap_res['sens'][t]:.3f}" for t in T.FP_TARGETS)
          + f"  AUPRC {ap_res['auprc']:.3f}  recall {ap_res['case_recall']:.3f}\n")

    hdr = ("lat_gate | " + "".join(f"@{t}FP  " for t in T.FP_TARGETS) + "| AUPRC | recall | >=AP@all | dAUPRC_vs_gate0")
    print(hdr); print("-" * len(hdr))
    rows = []
    for g in gates:
        r = T.eval_condition("fusion", cache, ap_g, lat_g, va_idx, si_tol, boot, seed, lat_gate=g)
        meets = all(r["sens"][t] >= ap_res["sens"][t] - 1e-9 for t in T.FP_TARGETS)  # fusion not worse than AP anywhere
        rows.append({"gate": g, "sens": r["sens"], "auprc": r["auprc"], "recall": r["case_recall"], "meets": meets})
    base_auprc = rows[0]["auprc"]
    for row in rows:
        g = row["gate"]
        cells = "".join(f"{row['sens'][t]:.3f} " for t in T.FP_TARGETS)
        flag = "  YES  " if row["meets"] else "   no  "
        print(f"{g:8.3f} | {cells}| {row['auprc']:.3f} | {row['recall']:.3f} | {flag}  | {row['auprc']-base_auprc:+.3f}")

    # Pre-declared selection: among gates meeting the rule, maximize AUPRC (tie-break: higher recall, then lower gate).
    eligible = [r for r in rows if r["meets"]]
    print()
    if not eligible:
        print("NO lat_gate makes fusion >= AP at every FROC target on this dev slice.")
        print("=> By the pre-declared rule, AP-only is the primary condition; fusion stays a documented "
              "secondary (it still helps at >=2 FP and in AUPRC/recall). Do NOT invent a new rule to rescue fusion.")
        return 0
    best = max(eligible, key=lambda r: (r["auprc"], r["recall"], -r["gate"]))
    print(f"SELECTED lat_gate = {best['gate']:.3f}  (max fusion AUPRC among rule-satisfying gates)")
    print(f"  fusion @ tau*: " + "  ".join(f"@{t}FP {best['sens'][t]:.3f}" for t in T.FP_TARGETS)
          + f"  AUPRC {best['auprc']:.3f}  recall {best['recall']:.3f}")
    print(f"  vs AP:         " + "  ".join(f"@{t}FP {ap_res['sens'][t]:.3f}" for t in T.FP_TARGETS)
          + f"  AUPRC {ap_res['auprc']:.3f}  recall {ap_res['case_recall']:.3f}")
    print(f"  vs ungated fusion (gate 0): AUPRC {rows[0]['auprc']:.3f} -> {best['auprc']:.3f}, "
          f"@1FP {rows[0]['sens'][1.0]:.3f} -> {best['sens'][1.0]:.3f}")
    print("\nNOTE: chosen on the 65-case dev slice (the 1 FP AP/fusion gap is within the bootstrap CIs, so this is "
          "partly tuning to noise). It is a candidate to BAKE INTO the frozen fusion algorithm; the sealed test "
          "confirms it. Nothing is frozen or opened here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
