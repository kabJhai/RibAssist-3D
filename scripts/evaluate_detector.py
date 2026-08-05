#!/usr/bin/env python3
"""Protocol stage: SCORE the saved detector weights with the current evaluation code (no retraining).

Pipeline: train_detector.py -> evaluate_detector.py -> freeze_detector.py -> eval_sealed_test.py.
Optimization (training) and evaluation are deliberately separated so freezing promotes exact,
already-scored artifacts. This loads the selected checkpoints, re-runs the current FROC on the SAME
recorded validation cases, and writes a fresh detector_dev_run.json with corrected metrics (weights
copied bit-identically). It never optimizes the network.

SAFEGUARD: AP-only evaluation is unchanged by the fusion refactor, so this asserts the recomputed
AP metrics (FROC targets, AUPRC, operating threshold, case recall) match the input dev run within
float tolerance. A mismatch means something OTHER than fusion changed (e.g. a different val split) —
it fails loudly rather than silently freezing a drifted evaluation.

Usage:
  python evaluate_detector.py --dev-run outputs/detector_dev_e80 --data outputs/det_out_v2/det_dev.npz \
      --out outputs/detector_dev_e80_scored
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import train_detector as T

AP_TOL = 1e-6   # AP eval is deterministic + unchanged; recompute must match the input to float noise


def _check_ap_identity(old_ap, new):
    """Compare recomputed AP metrics to the input dev run's AP. Returns (ok, diffs)."""
    if not old_ap: return True, ["(input had no AP metrics to compare)"]
    diffs = []
    for t in T.FP_TARGETS:
        o = old_ap.get("sens_at_targets", {}).get(str(t)); n = new["sens"][t]
        if o is not None and abs(o - n) > AP_TOL: diffs.append(f"sens@{t}FP {o:.6f} vs {n:.6f}")
    for key, nv in (("op_threshold", new["op_threshold"]), ("case_recall", new["case_recall"]),
                    ("count_mae", new["count_mae"]), ("auprc", new["auprc"])):
        o = old_ap.get(key)
        if o is not None and nv is not None and abs(o - nv) > max(AP_TOL, 1e-4 * abs(o)):
            diffs.append(f"{key} {o} vs {nv}")
    return (len(diffs) == 0), diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-ap-mismatch", action="store_true", help="proceed despite AP drift (NOT recommended)")
    ap.add_argument("--lat-gate", type=float, default=0.0,
                    help="bake the calibrated unmatched-lateral score gate into the fusion condition (calibrate_fusion.py "
                         "picks it). 0.0 = ungated union. Only fusion is affected; AP/lat/paired are unchanged.")
    a = ap.parse_args()
    if torch is None: print("pip install torch scipy", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new dir.")
    rec = json.loads((a.dev_run / "detector_dev_run.json").read_text())
    lp = rec["learning_procedure"]; ep = rec["eval_params"]; base_ch, si_tol = ep["base_ch"], ep["si_tol"]
    boot, seed = ep["bootstrap"], ep["seed"]
    d, man, dev_sha = T.load_dev(a.data); dev = T.device()
    for v, want in rec["detector_sha256"].items():
        got = T.sha256_file(a.dev_run / f"detector_{v}.pt")
        if got != want: raise ValueError(f"detector_{v}.pt hash {got[:12]}.. != record {want[:12]}..")
    val_ids = rec["split"]["val_case_ids"]; case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    va_idx = np.array([case_to_idx[c] for c in val_ids])

    arch = T.arch_from_record(rec)   # rebuild the SAME architecture (from-scratch U-Net or pretrained-encoder)
    views = [v for v in ("ap", "lat") if (a.dev_run / f"detector_{v}.pt").exists()]   # support single-view (ap-only) runs
    nets = {}
    for v in views:
        net = T.build_detector(arch, pretrained=False).to(dev)
        net.load_state_dict(torch.load(a.dev_run / f"detector_{v}.pt", map_location=dev)); net.eval(); nets[v] = net
    ap_g, lat_g = T.group_instances(d, "ap"), T.group_instances(d, "lat")
    cache = T.peak_cache(nets, d, va_idx, dev)
    conds = ["ap", "lat", "fusion", "paired"] if len(views) == 2 else list(views)   # fusion/paired need both views
    # lat_gate affects ONLY the fusion candidate set (build_case_candidates gates unmatched lateral);
    # passing it to ap/lat/paired is a no-op, so the AP-identity guard below still holds.
    results = {c: T.eval_condition(c, cache, ap_g, lat_g, va_idx, si_tol, boot, seed, lat_gate=a.lat_gate) for c in conds}

    # ---- AP-identity safeguard ----
    ok, diffs = _check_ap_identity(rec.get("dev_internal_froc", {}).get("ap"), results["ap"])
    if ok: print("AP-identity check: PASS (recomputed AP matches the input dev run within tolerance).")
    else:
        print(f"AP-identity check: FAIL — AP metrics drifted: {diffs}", file=sys.stderr)
        if not a.allow_ap_mismatch:
            raise ValueError("AP metrics changed although AP evaluation should be unchanged. Something other than the "
                             "fusion refactor differs (e.g. split, weights, data). Investigate before freezing "
                             "(or pass --allow-ap-mismatch if you understand why).")

    name = {"ap": "AP-only", "lat": "lateral-only", "fusion": "biplanar-FUSION", "paired": "paired-confirmed"}
    rankable = [c for c in conds if results[c]["auprc"] is not None]
    leader = max(rankable, key=lambda c: results[c]["auprc"]) if rankable else None
    print(f"\n====== SCORED (current algorithm) FROC — {a.dev_run.name}"
          + (f"  [fusion lat_gate={a.lat_gate}]" if a.lat_gate > 0 else "") + " ======")
    print(f"{'condition':18}{'unit':16}" + "".join(f"{'sens@'+str(t)+'FP':>18}" for t in T.FP_TARGETS))
    for c in conds:
        r = results[c]; row = f"{(name[c]+' *' if c == leader else name[c]):18}{T.UNIT[c]:16}"
        for t in T.FP_TARGETS:
            lo, hi = r["ci"][t]; row += f"{(format(r['sens'][t],'.3f')+' ['+format(lo,'.2f')+','+format(hi,'.2f')+']'):>18}"
        print(row)
    print(f"{'condition':18}{'op_thr':>8}{'case-recall':>13}{'count-MAE':>11}{'AUPRC':>8}")
    for c in conds:
        r = results[c]
        print(f"{name[c]:18}{r['op_threshold']:>8.3f}{r['case_recall']:>13.3f}{r['count_mae']:>11.3f}"
              f"{('%.3f'%r['auprc'] if r['auprc'] is not None else '   n/a'):>8}")

    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    det_sha = {}
    for v in views:
        shutil.copy2(a.dev_run / f"detector_{v}.pt", work / f"detector_{v}.pt"); det_sha[v] = T.sha256_file(work / f"detector_{v}.pt")
    new = dict(rec); new["detector_sha256"] = det_sha; new["det_dev_sha256"] = dev_sha
    new["scored_from"] = str(a.dev_run)
    new["scoring_note"] = ("weights unchanged (selected checkpoints); dev_internal_froc / frozen_froc_grids / operating "
                           "thresholds recomputed with the CURRENT evaluation algorithm; AP-identity checked."
                           + (f" Fusion uses calibrated unmatched-lateral score gate = {a.lat_gate}." if a.lat_gate > 0 else ""))
    new["ap_identity_check"] = {"pass": bool(ok), "diffs": diffs}
    new["frozen_froc_grids"] = {c: results[c]["grid"] for c in conds}
    lp2 = dict(lp); lp2["operating_threshold_per_condition"] = {c: results[c]["op_threshold"] for c in conds}
    # bake the calibrated fusion gate into the persisted procedure so freeze_detector + eval_sealed_test reuse it verbatim
    bf = dict(lp.get("biplanar_fusion", {})); bf["unmatched_lateral_score_gate"] = float(a.lat_gate)
    if a.lat_gate > 0:
        bf["gate_selection"] = ("maximum development AUPRC among gates satisfying fusion >= AP at 0.5/1/2/4 FP "
                                "(calibrate_fusion.py, dev-internal val slice; sealed test confirms)")
    lp2["biplanar_fusion"] = bf
    new["learning_procedure"] = lp2
    new["dev_internal_froc"] = {c: {"sens_at_targets": {str(k): v for k, v in results[c]["sens"].items()},
                                    "ci": {str(k): v for k, v in results[c]["ci"].items()},
                                    "froc_curve": results[c]["froc"], "unit": T.UNIT[c],
                                    "op_threshold": results[c]["op_threshold"], "case_recall": results[c]["case_recall"],
                                    "count_mae": results[c]["count_mae"], "unmatched_fp_at_op": results[c]["unmatched_fp_at_op"],
                                    "auprc": results[c]["auprc"], "clean_case_fp_at~1fp": results[c]["clean_case_fp_at~1fp"],
                                    "n_negatives": results[c]["n_neg"]} for c in conds}
    (work / "detector_dev_run.json").write_text(json.dumps(new, indent=2))
    work.rename(a.out)
    print(f"\nwrote scored dev run to {a.out}/ (weights bit-identical; eval recomputed). Promote with freeze_detector.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
