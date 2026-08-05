#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""END-TO-END addressing DEVELOPMENT INTEGRATION DIAGNOSTIC (NOT an unbiased generalization estimate).

Prior addressing metrics were measured on crops centered at GROUND-TRUTH fracture centers. Deployment
addresses crops centered at PREDICTED detector peaks — a different distribution. This runs the frozen
detector + deployed addressing model exactly as run_ribassist does (REUSING those functions), derives GT
addresses per fracture from rib-seg + centerlines via the SAME shared convention the training set used,
matches predicted addressed detections to GT fractures with the detector's own matching rule, and scores
addressing CONDITIONAL ON a correct detection.

BIAS CAVEAT — read before comparing models: the deployment addressing checkpoint was refit on ALL 305
development addressing cases, which INCLUDE the cases represented in the detector-validation split used
here. So this is NOT a held-out addressing estimate and MUST NOT be used to reselect AP-pos vs AP-no-pos
as if these cases were unseen by addressing. It IS valid for: verifying detector-centered crop behaviour,
measuring degradation from oracle to predicted centers, finding duplicate detections and plausible-looking
addressed false positives, and validating the end-to-end implementation. The SEALED cohort is the first
unbiased end-to-end read for the final full-development checkpoint.

GT is read ONLY to score; it is never fed to the detector or addressing model (run_ribassist reads images
only). Fail-closed provenance + crop geometry are shared with run_ribassist so mixed artifacts cannot be
scored.

Usage:
  python eval_address_e2e.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --address-model outputs/addressing_model_ap_pos \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/eval_address_e2e_ap_pos.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import torch, nibabel as nib
    from scipy.optimize import linear_sum_assignment
    import train_detector as T
    import run_ribassist as RR
    from make_address_data_detframe import anchor_canonical
    from make_rib_targets import canonical_voxels, _find
    from rib_labeling import side_num_from_seg
except Exception:  # noqa: BLE001
    torch = None


def gt_addresses(cid, image_dirs, seg_dir, cl_dir, iids):
    """GT (side, rib, s) per requested fracture-instance label, derived in the canonical frame EXACTLY as
    make_address_data_detframe builds training labels. Returns (addr {iid:(side,rib,s)}, skipped {iid:reason}).
    A whole-case input problem marks every requested iid skipped with that reason (per-INSTANCE accounting)."""
    segp = _find([seg_dir], f"{cid}-rib-seg.nii.gz", f"{cid}.nii.gz", f"{cid}*.nii.gz", f"{cid}*.nii")
    imp = _find(image_dirs, f"{cid}-label.nii.gz", f"{cid}-label.nii", f"{cid}-label.nii*")
    clp = Path(cl_dir) / f"{cid}.npz"
    iids = [int(i) for i in iids]
    if segp is None or imp is None or not clp.exists():
        return {}, {i: "missing seg/label/cl" for i in iids}
    lab_nii = nib.load(str(imp)); A_o = lab_nii.affine
    fl_c = nib.as_closest_canonical(lab_nii); rs_c = nib.as_closest_canonical(nib.load(str(segp)))
    if fl_c.shape != rs_c.shape or not np.allclose(fl_c.affine, rs_c.affine, rtol=0, atol=1e-4):
        return {}, {i: "seg/label frame mismatch" for i in iids}
    fl_can = np.asarray(fl_c.get_fdata()).astype(np.int32); rs_can = np.asarray(rs_c.get_fdata()).astype(np.int32)
    info, _ = side_num_from_seg(rs_can, lr_axis=0, si_axis=2)
    cl = np.load(clp)["cl"]; R = cl.shape[0]
    cl_can = [canonical_voxels(cl[r].astype(np.float64), A_o, fl_c.affine) for r in range(R)]
    addr, skipped = {}, {}
    for lb in iids:
        vox = np.array(np.nonzero(fl_can == lb))
        if vox.size == 0: skipped[lb] = "label absent in canonical volume"; continue
        fc = vox.mean(1); at = rs_can[vox[0], vox[1], vox[2]]; nz = at[at != 0]
        if nz.size == 0: skipped[lb] = "no rib-seg overlap"; continue
        rl = int(np.bincount(nz).argmax())
        if rl not in info or not (1 <= rl <= R): skipped[lb] = "rib id out of range"; continue
        clc = anchor_canonical(cl_can[rl - 1])
        s = int(np.linalg.norm(clc - fc[None], axis=1).argmin()) / (len(clc) - 1)
        addr[lb] = (info[rl]["side"], int(info[rl]["num"]), float(s))
    return addr, skipped


def match(peaks_rc, footprints, radius):
    """One-to-one Hungarian assignment of point positions (Nx2 row,col) to GT footprints (each Mx2) within
    `radius`, using the detector's own _min_dist. Returns [(peak_idx, gt_idx, dist), ...]."""
    n, m = len(peaks_rc), len(footprints)
    if n == 0 or m == 0: return []
    BIG = 1e6; cost = np.full((n, m), BIG, np.float32)
    for i in range(n):
        for j in range(m):
            dm = T._min_dist(np.asarray(peaks_rc[i], np.float32), footprints[j])
            if dm <= radius: cost[i, j] = dm
    ri, ci = linear_sum_assignment(cost)
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(ri, ci) if cost[r, c] < BIG]


def build_instance_records(d):
    """Per-case GT instance records built DIRECTLY from the fp arrays in global fp order, so each footprint
    carries its explicit fracture-instance id (iid). Asserted to reproduce T.group_instances exactly, so
    downstream matching columns are proven to correspond to the right iid (no positional-alignment assumption)."""
    fp_case, fp_iid = d["fp_case"], d["fp_iid"]
    ap_pts, ap_ptr = d["ap_fp_pts"], d["ap_fp_ptr"]; lat_pts, lat_ptr = d["lat_fp_pts"], d["lat_fp_ptr"]
    recs = {}
    for gi in range(len(fp_case)):
        ci = int(fp_case[gi])
        recs.setdefault(ci, []).append({"iid": int(fp_iid[gi]),
                                        "ap_foot": ap_pts[ap_ptr[gi]:ap_ptr[gi + 1]].astype(np.int32),
                                        "lat_foot": lat_pts[lat_ptr[gi]:lat_ptr[gi + 1]].astype(np.int32)})
    ap_g, lat_g = T.group_instances(d, "ap"), T.group_instances(d, "lat")
    for ci, rs in recs.items():   # identity-critical: prove the direct slicing == the detector's grouping
        assert len(rs) == len(ap_g[ci]) == len(lat_g[ci]), f"instance-count mismatch for case {ci}"
        for k, r in enumerate(rs):
            assert np.array_equal(r["ap_foot"], ap_g[ci][k]) and np.array_equal(r["lat_foot"], lat_g[ci][k]), \
                f"footprint ordering mismatch for case {ci} instance {k}"
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True); ap.add_argument("--address-model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if torch is None: print("pip install torch nibabel scipy scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)
    cfg, views, use_pos, crop, half_ds, man = RR.load_addressing(a.address_model)
    half = RR.resolve_half(cfg, half_ds)                       # FAIL CLOSED (point 2)
    net = RR.TA.Net(views, use_pos).to(dev)
    net.load_state_dict(torch.load(a.address_model / "addressing_model.pt", map_location=dev)); net.eval()
    RR.verify_provenance(a.data, a.address_model, cfg, rec)    # FAIL CLOSED (point 2)
    need = RR.needed_views(views)

    d = np.load(a.data, allow_pickle=False)
    case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    val_ids = rec["split"]["val_case_ids"]
    recs = build_instance_records(d)                          # explicit-iid records (point 4)
    RADIUS = T.MATCH_RADIUS_PX
    if "ap" not in need:
        raise ValueError(f"this evaluator matches ADDRESSED detections by their AP peak; the addressing model "
                         f"consumes {list(need)} (no AP peak). Extend the matching before evaluating such a model.")

    # cascade counters — names reflect VIEW-LEVEL matching semantics (a GT is counted if a peak of an active
    # candidate matches its footprint IN THAT VIEW; a paired candidate could in principle match different GT in
    # each view, so 'either view' is NOT the same as 'one valid fused candidate').
    C = {"gt_fractures_total": 0, "gt_fractures_with_derivable_address": 0,
         "gt_with_matching_ap_peak_in_active_candidate": 0,
         "gt_with_matching_lateral_peak_in_active_candidate": 0,
         "gt_with_matching_peak_in_either_view": 0,
         "gt_matched_by_addressed_detection": 0, "matched_detections_with_scorable_gt_address": 0}
    tp = []                # scorable matched pairs
    fp_addr = 0            # addressed detections not matching any GT
    lost_no_ap_peak = 0    # GT with a matching peak in some view but NONE in AP -> AP model cannot address it
    dup_addressed = 0      # duplicate ADDRESSED detections (incl. FP) at one predicted rib level, per case
    skipped_instances = Counter()

    for cid in val_ids:
        ci = case_to_idx[cid]
        ap_img = d["ap"][ci].astype(np.float32); lat_img = d["lat"][ci].astype(np.float32)
        active, ap_peaks, lat_peaks = RR.fused_candidates_at_op(
            nets, ap_img, lat_img, d["ap_geo"][ci], d["lat_geo"][ci], si_tol, lat_gate, op_thr, dev)
        enriched = [RR.enrich_candidate(cd, ap_peaks, lat_peaks) for cd in active]
        addressed = RR.address_candidates(enriched, ap_img, lat_img, net, views, use_pos, crop, half, dev)
        sites = [x for x in addressed if x["address_status"] == "addressed"]

        gt = recs.get(ci, []); C["gt_fractures_total"] += len(gt)
        gt_ap = [g["ap_foot"] for g in gt]; gt_lat = [g["lat_foot"] for g in gt]
        gt_addr, skipped = gt_addresses(cid, a.image_dirs, a.seg_dir, a.cl_dir, [g["iid"] for g in gt])
        for r in skipped.values(): skipped_instances[r] += 1
        C["gt_fractures_with_derivable_address"] += len(gt_addr)

        ap_cand_peaks = [e["ap_rc"] for e in enriched if e["ap_rc"] is not None]
        lat_cand_peaks = [e["lat_rc"] for e in enriched if e["lat_rc"] is not None]
        det_ap = {g for _, g, _ in match(ap_cand_peaks, gt_ap, RADIUS)}          # GT with an AP peak in some active cand
        det_lat = {g for _, g, _ in match(lat_cand_peaks, gt_lat, RADIUS)}       # GT with a lateral peak in some active cand
        C["gt_with_matching_ap_peak_in_active_candidate"] += len(det_ap)
        C["gt_with_matching_lateral_peak_in_active_candidate"] += len(det_lat)
        C["gt_with_matching_peak_in_either_view"] += len(det_ap | det_lat)
        lost_no_ap_peak += len((det_ap | det_lat) - det_ap)                      # matched in a view but NOT in AP

        m = match([s["ap_xy"] for s in sites], gt_ap, RADIUS)     # addressed detections -> GT
        matched_gt = {g for _, g, _ in m}
        C["gt_matched_by_addressed_detection"] += len(matched_gt)
        fp_addr += len(sites) - len(m)
        # duplicate ADDRESSED detections (TP and FP) sharing a predicted (side,rib) level, this case
        site_rib = Counter((s["side"], int(s["rib"])) for s in sites)
        dup_addressed += sum(c - 1 for c in site_rib.values() if c > 1)
        for si, gi, dist in m:
            g = gt_addr.get(gt[gi]["iid"])
            if g is None: continue                                 # matched but GT address not derivable
            C["matched_detections_with_scorable_gt_address"] += 1
            ps = sites[si]
            tp.append({"case": cid, "loc_err_px": round(dist, 2), "pred": (ps["side"], int(ps["rib"]), ps["s"]),
                       "gt": g, "side_ok": ps["side"] == g[0], "rib_exact": int(ps["rib"]) == g[1],
                       "rib_within1": abs(int(ps["rib"]) - g[1]) <= 1, "s_abs": abs(ps["s"] - g[2])})

    n = len(tp)
    def mean(k): return round(float(np.mean([t[k] for t in tp])), 4) if n else None
    strat = None
    if n >= 4:
        errs = np.array([t["loc_err_px"] for t in tp]); med = float(np.median(errs))
        lo = [t for t in tp if t["loc_err_px"] <= med]; hi = [t for t in tp if t["loc_err_px"] > med]
        strat = {"median_loc_err_px": round(med, 2),
                 "rib_exact_low_locerr": round(float(np.mean([t["rib_exact"] for t in lo])), 4) if lo else None,
                 "rib_exact_high_locerr": round(float(np.mean([t["rib_exact"] for t in hi])), 4) if hi else None,
                 "s_mae_low_locerr": round(float(np.mean([t["s_abs"] for t in lo])), 4) if lo else None,
                 "s_mae_high_locerr": round(float(np.mean([t["s_abs"] for t in hi])), 4) if hi else None}
    dup_tp = Counter((t["case"], t["pred"][0], int(t["pred"][1])) for t in tp)
    duplicate_tp_same_rib = sum(c - 1 for c in dup_tp.values() if c > 1)

    # ---- cascade invariants (point 7): each stage is a subset of the previous ----
    assert C["gt_fractures_with_derivable_address"] <= C["gt_fractures_total"]
    assert C["gt_with_matching_ap_peak_in_active_candidate"] <= C["gt_with_matching_peak_in_either_view"]
    assert C["gt_matched_by_addressed_detection"] <= C["gt_with_matching_ap_peak_in_active_candidate"], \
        "AP-model: addressed detections match GT via AP, so must be a subset of GT-with-AP-peak"
    assert C["matched_detections_with_scorable_gt_address"] <= C["gt_matched_by_addressed_detection"]
    assert len(tp) == C["matched_detections_with_scorable_gt_address"]

    result = {
        "diagnostic": "END-TO-END DEVELOPMENT INTEGRATION DIAGNOSTIC — not an unbiased addressing "
                      "generalization estimate. Detector split is validation-held-out, but the deployment "
                      "addressing checkpoint was refit on ALL development addressing cases, including cases "
                      "represented in the detector-validation split. Do NOT use to reselect the addressing "
                      "model; the sealed cohort is the first unbiased end-to-end read.",
        "n_val_cases": len(val_ids), "addressing_model": str(a.address_model),
        "addressing_consumes_views": list(need), "detector_run": str(a.detector_run),
        "fusion_op_threshold": op_thr, "lat_gate": lat_gate,
        "coverage_cascade": C,
        "conditional_addressing_on_true_positive_detections": {
            "n_scored": n, "side_accuracy": mean("side_ok"), "rib_exact": mean("rib_exact"),
            "rib_within1": mean("rib_within1"), "s_mae_descriptive": mean("s_abs"),
            "note": "detector-centered crops (deployment distribution), conditional on the addressed detection "
                    "matching a GT fracture within the detector matching radius AND the GT address being derivable."},
        "address_error_by_detector_localization_error": strat,
        "failure_modes": {"false_positive_detections_with_address": fp_addr,
                          "true_fractures_detected_but_no_ap_candidate": lost_no_ap_peak,
                          "duplicate_addressed_detections_same_predicted_rib": dup_addressed,
                          "duplicate_true_positive_detections_same_predicted_rib": duplicate_tp_same_rib},
        "gt_address_derivation_skipped_instances": dict(skipped_instances),
        "along_rib_s_status": "exploratory — reported descriptively; not a supported capability",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    c = result["conditional_addressing_on_true_positive_detections"]
    print("END-TO-END DEVELOPMENT INTEGRATION DIAGNOSTIC (biased for model selection — see JSON caveat)")
    print(f"  model {a.address_model.name} consumes {list(need)} | {len(val_ids)} val cases")
    print(f"  cascade: GT {C['gt_fractures_total']} -> AP-peak-match {C['gt_with_matching_ap_peak_in_active_candidate']} "
          f"/ either-view {C['gt_with_matching_peak_in_either_view']} -> addressed {C['gt_matched_by_addressed_detection']} "
          f"-> scorable {C['matched_detections_with_scorable_gt_address']} (derivable GT {C['gt_fractures_with_derivable_address']})")
    print(f"  conditional (n={c['n_scored']}): side {c['side_accuracy']} | rib-exact {c['rib_exact']} "
          f"| rib±1 {c['rib_within1']} | s-MAE {c['s_mae_descriptive']} (descriptive)")
    print(f"  failure modes: FP-with-address {fp_addr} | detected-but-no-AP-peak {lost_no_ap_peak} "
          f"| dup-addressed-same-rib {dup_addressed} (of which TP {duplicate_tp_same_rib})")
    if strat: print(f"  rib-exact by loc-err (<= vs > {strat['median_loc_err_px']}px): "
                    f"{strat['rib_exact_low_locerr']} vs {strat['rib_exact_high_locerr']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
