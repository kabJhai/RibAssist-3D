#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""INTEGRATION GATE: prove run_ribassist.py's fused candidate set is IDENTICAL to the established
evaluation path (evaluate_detector.py / train_detector.eval_condition) on every validation case.

Both paths must call the SAME train_detector primitives (peaks_from_hm, build_case_candidates) on the
SAME rebuilt detector, so the fused candidate scores are identical by construction. This script asserts
that construction still holds — it is a regression guard, so a future edit that silently forks the demo
peak/fusion logic away from the evaluated logic fails LOUDLY here instead of in a portfolio figure.

For each validation case it compares, at BOTH the extraction floor (full candidate set) and the frozen
fusion operating threshold (the deployed set):
  * per-view peak arrays (rows, cols, scores);
  * the multiset of fusion candidate (score, source) tuples.

It reads GT only to locate the recorded val split — GT footprints are NOT used to build candidates and
are never touched by run_ribassist's inference path; this script is a development-time check, not the
deployed path.

Usage:
  python verify_candidate_identity.py --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch
    import train_detector as T
    from run_ribassist import load_detector, fused_candidates_at_op
except Exception:  # noqa: BLE001
    torch = None


def _source(cd):
    return "paired" if (cd["ap"] is not None and cd["lat"] is not None) else ("ap_only" if cd["lat"] is None else "lat_only")


def _cand_multiset(cands):
    # (rounded score, source) multiset — identity of the fused candidate set independent of list order
    return sorted((round(float(c["score"]), 6), _source(c)) for c in cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    if torch is None: print("pip install torch scipy", file=sys.stderr); return 1
    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = load_detector(a.detector_run, dev)

    d, man, _ = T.load_dev(a.data)
    val_ids = rec["split"]["val_case_ids"]; case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    va_idx = np.array([case_to_idx[c] for c in val_ids])

    # reference path: exactly what evaluate_detector.py runs (peak_cache -> build_case_candidates)
    cache = T.peak_cache(nets, d, va_idx, dev)

    n_ok = 0; mismatches = []; tot_floor = 0; tot_op = 0
    for k, i in enumerate(va_idx):
        cid = str(d["case"][i])
        # --- reference candidates (floor + op) ---
        ref_ap = cache[k].get("ap", np.zeros((0, 3))); ref_lat = cache[k].get("lat", np.zeros((0, 3)))
        ref_floor = T.build_case_candidates("fusion", ref_ap, ref_lat, cache[k]["ap_geo"], cache[k]["lat_geo"], si_tol, lat_gate)
        ref_op = [c for c in ref_floor if c["score"] >= op_thr]
        # --- demo path (run_ribassist) rebuilds peaks from the images itself ---
        demo_op, demo_ap, demo_lat = fused_candidates_at_op(
            nets, d["ap"][i].astype(np.float32), d["lat"][i].astype(np.float32),
            d["ap_geo"][i], d["lat_geo"][i], si_tol, lat_gate, op_thr, dev)
        demo_floor = T.build_case_candidates("fusion", demo_ap, demo_lat, d["ap_geo"][i], d["lat_geo"][i], si_tol, lat_gate)

        problems = []
        if not (np.array_equal(np.round(demo_ap, 6), np.round(ref_ap, 6))): problems.append("AP peaks differ")
        if not (np.array_equal(np.round(demo_lat, 6), np.round(ref_lat, 6))): problems.append("lat peaks differ")
        if _cand_multiset(demo_floor) != _cand_multiset(ref_floor): problems.append("floor candidate set differs")
        if _cand_multiset(demo_op) != _cand_multiset(ref_op): problems.append("op-point candidate set differs")
        tot_floor += len(ref_floor); tot_op += len(ref_op)
        if problems: mismatches.append((cid, problems))
        else: n_ok += 1

    print(f"validation cases checked: {len(va_idx)} | identical: {n_ok} | mismatched: {len(mismatches)}")
    print(f"reference fused candidates: floor={tot_floor}, at op(thr={op_thr:.4f}, gate={lat_gate:.4f})={tot_op}")
    if mismatches:
        for cid, ps in mismatches[:20]: print(f"  MISMATCH {cid}: {', '.join(ps)}", file=sys.stderr)
        print("\nFAIL: run_ribassist candidates diverge from the evaluation path.", file=sys.stderr); return 1
    print("PASS: run_ribassist's fused candidates are identical to the evaluation path on every val case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
