#!/usr/bin/env python3
"""STAGE C — PREDICTED centers + DEPLOYABLE correspondence, through the EXACT FROZEN FUSION PATH.

Stage A: the projection geometry supports accurate 3D reconstruction (oracle centers + oracle pairing).
Stage B: detector-predicted centers preserve rib-level accuracy UNDER ORACLE correspondence (median ~4mm
to fracture, 93.6% rib-exact, 100% rib±1 on the dual-view subset). Stage C removes the correspondence
oracle: AP<->lateral peaks are paired by the deployed matcher and the cost of that matcher is measured.

The pairing logic is NOT changed while measuring. Candidates come from train_detector.peaks_from_hm +
build_case_candidates with the RECORDED si_tol, unmatched-lateral gate, and fusion operating threshold —
i.e. run_ribassist's deployment fusion, reused verbatim. Only PAIRED fusion candidates (a real peak in
BOTH views) are triangulated; AP-only and lateral-only candidates are reported as explicitly
NON-triangulable, never silently dropped.

EVERY validation case is analyzed, INCLUDING GT-negative cases (no fractures) — those are where fully-false
phantom pairs concentrate, so excluding them would inflate pair precision and understate phantom output.
On a negative case every peak maps to no GT, so every paired candidate is a fully-false pair by construction.

Correspondence CLASSIFICATION is separated from geometry SCORING. Classification uses only the per-peak GT
identity (coordinate-independent): all AP peaks are matched to AP GT footprints and all lateral peaks to
lateral GT footprints (the Stage-B one-to-one matcher), giving each peak a GT instance id (iid) or None.
Each paired candidate is classified by (ap_gt_iid, lat_gt_iid):
    correct correspondence : both non-null and EQUAL
    cross-instance mispair : both non-null and UNEQUAL   (phantom: AP fracture joined to a DIFFERENT lat)
    one-sided false pair   : exactly one non-null        (a real peak paired to a non-GT peak)
    fully false pair       : both null                   (two non-GT peaks paired; ALL negative-case pairs)
These class COUNTS never depend on CT geometry. Geometry (case_gt) only decides whether a pair's 3D
distance / anatomical metrics can be COMPUTED; a correct correspondence whose fracture volume is
unavailable still counts as correct correspondence (it just isn't geometry-scored).

3D error is stratified by class (a mispair has no single correct target, so it is NEVER folded into one
"all-pair" statistic): correct pairs get the full Stage-B metric suite + the oracle comparison on the SAME
correct instances; cross-instance pairs report distance to the AP-associated AND the lat-associated
fracture separately plus the min distance to ANY GT fracture in the case; one-sided pairs report the
phantom-to-known-fracture distance; fully-false pairs report only the min distance to any GT fracture
(positive cases only — a negative case has no GT to measure against, counted under negative_case_accounting).

Everything is reported at BOTH the per-view extraction floor and the frozen fused operating point.

DIAGNOSTIC STATUS: development diagnostic on the detector-VALIDATION split (champion checkpoint /
architecture / lateral gate / operating config selected on development validation). The sealed test is the
first confirmatory read. The Stage-B floor dual-view availability is an assignment-dependent estimate of the
practical availability ceiling; Stage C measures survival through thresholding and deployable pairing.

Usage:
  python eval_biplanar_geometry_stageC.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/eval_biplanar_geometry_stageC.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch  # noqa: F401
    import train_detector as T
    import run_ribassist as RR
    from eval_address_e2e import build_instance_records, match
    from eval_biplanar_geometry import back_project, case_gt, fracture_metrics
    from eval_biplanar_geometry_stageB import verify_detector_data_provenance
    from nibabel.affines import apply_affine
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e


def voxel_center_mm(p_rec, vox, aff):
    """Min distance (world mm) from reconstructed point p_rec to any voxel CENTER of a fracture group."""
    p_w = apply_affine(aff, p_rec); mask_w = apply_affine(aff, vox.T.astype(np.float64))
    return float(np.linalg.norm(mask_w - p_w[None], axis=1).min())


def min_any_gt_mm(p_rec, g):
    """Min distance (world mm) from p_rec to the nearest voxel center of ANY GT fracture in the case."""
    if g is None or not g["fl_groups"]: return None
    return min(voxel_center_mm(p_rec, v, g["aff"]) for v in g["fl_groups"].values())


def new_collector():
    return {
        # candidate-level
        "n_active": 0, "n_paired": 0, "n_ap_only": 0, "n_lat_only": 0,
        # correspondence-class COUNTS (geometry-independent)
        "n_correct_correspondence": 0, "n_cross_instance_mispair": 0, "n_one_sided_false": 0, "n_fully_false": 0,
        "n_paired_either_gt": 0, "n_paired_both_gt": 0,
        # geometry-scored subsets
        "n_correct_geometry_scorable": 0, "n_incorrect_geometry_scorable": 0,
        # GT-level cascade (threshold-specific dual-view, correspondence, then geometry within-X)
        "both_views_available_at_threshold": 0,
        "dual_view_gt_touched_by_paired_candidate": 0,
        "correctly_paired_threshold_gt": 0, "correctly_paired_floor_gt": 0,
        "correct_within_5": 0, "correct_within_10": 0, "correct_within_15": 0,
        # geometry-scored metric records
        "correct": [], "cross": [], "one_sided": [], "fully_false": [],
        # case-level flags
        "cases_with_correct": set(), "cases_with_phantom": set(),
        # negative-case accounting
        "neg_cases_with_active": set(), "neg_cases_with_paired": set(),
        "paired_false_3d_in_neg": 0, "single_view_in_neg": 0,
    }


def analyze_case(col, thr, cands_all, ap_pk, lat_pk, ap_geo_ci, lat_geo_ci, gt,
                 floor_both_available_iids, g, cid, ci, gj, ap_ctr, lat_ctr):
    """Accumulate one case at threshold `thr`. Peak->GT identity AND dual-view availability are BOTH derived
    THRESHOLD-SPECIFICALLY from the unique peaks that participate in this threshold's PAIRED candidates (a
    floor-only peak can otherwise steal a GT assignment from an operating-point peak, and floor availability
    can drop an instance the threshold matching recovers). Geometry `g` (may be None) affects ONLY whether
    distances are computed. Negative cases (gt == []) run fully -> every pair is fully-false."""
    is_neg = (len(gt) == 0)
    active = [c for c in cands_all if c["score"] >= thr]
    paired = [c for c in active if c["ap"] is not None and c["lat"] is not None]
    n_ap_only = sum(1 for c in active if c["ap"] is not None and c["lat"] is None)
    n_lat_only = sum(1 for c in active if c["ap"] is None and c["lat"] is not None)
    col["n_active"] += len(active); col["n_paired"] += len(paired)
    col["n_ap_only"] += n_ap_only; col["n_lat_only"] += n_lat_only
    if is_neg:
        if active: col["neg_cases_with_active"].add(cid)
        if paired: col["neg_cases_with_paired"].add(cid)
        col["single_view_in_neg"] += n_ap_only + n_lat_only

    # THRESHOLD-SPECIFIC identity: one-to-one GT matching among ONLY the peaks participating in paired
    # candidates at this threshold (indices into the original peak arrays). Empty on negative cases.
    active_ap_idx = sorted({int(c["ap"]) for c in paired})
    active_lat_idx = sorted({int(c["lat"]) for c in paired})
    ap_iid_thr, lat_iid_thr = {}, {}
    if gt and active_ap_idx:
        am = match([ap_pk[i, :2] for i in active_ap_idx], [x["ap_foot"] for x in gt], T.MATCH_RADIUS_PX)
        ap_iid_thr = {active_ap_idx[si]: int(gt[gi]["iid"]) for si, gi, _ in am}
    if gt and active_lat_idx:
        lm = match([lat_pk[i, :2] for i in active_lat_idx], [x["lat_foot"] for x in gt], T.MATCH_RADIUS_PX)
        lat_iid_thr = {active_lat_idx[si]: int(gt[gi]["iid"]) for si, gi, _ in lm}
    # THRESHOLD-SPECIFIC dual-view availability: GT with BOTH a participating AP peak and a participating
    # lateral peak at THIS threshold (from the threshold-conditioned maps, NOT the floor set — the floor
    # Hungarian can drop an instance a low-score stealer took, which the threshold matching recovers).
    both_available_iids_thr = set(ap_iid_thr.values()) & set(lat_iid_thr.values())
    col["both_views_available_at_threshold"] += len(both_available_iids_thr)

    represented_iids = set(); correct_corr_iids = set(); correct_scored_iid = {}
    has_correct = has_phantom = False
    for c in paired:
        ai, li = c["ap"], c["lat"]
        ap_iid = ap_iid_thr.get(ai); lat_iid = lat_iid_thr.get(li)
        for x in (ap_iid, lat_iid):                       # threshold-dual-view GT touched by a paired candidate
            if x is not None and x in both_available_iids_thr: represented_iids.add(int(x))
        ng = (ap_iid is not None) + (lat_iid is not None)
        if ng >= 1: col["n_paired_either_gt"] += 1
        if ng == 2: col["n_paired_both_gt"] += 1
        p_rec, si_dis = back_project(ap_pk[ai, :2], lat_pk[li, :2], ap_geo_ci, lat_geo_ci)  # coords only, no g
        if ap_iid is not None and lat_iid is not None and ap_iid == lat_iid:      # CORRECT correspondence
            col["n_correct_correspondence"] += 1; iid = int(ap_iid); correct_corr_iids.add(iid); has_correct = True
            vox = g["fl_groups"].get(iid) if g is not None else None
            if vox is not None and vox.shape[1] > 0:
                m, reason = fracture_metrics(p_rec, si_dis, vox, g, cid, iid)
                if m is not None:
                    j = gj[(ci, iid)]; p_or, si_or = back_project(ap_ctr[j], lat_ctr[j], ap_geo_ci, lat_geo_ci)
                    mo, _ = fracture_metrics(p_or, si_or, vox, g, cid, iid)
                    m["oracle_mask_center_mm"] = mo["mask_center_mm"] if mo else None
                    m["oracle_rib_exact"] = mo["rib_exact"] if mo else None
                    m["oracle_along_mm"] = mo["along_mm"] if mo else None
                    m["cand_score"] = float(c["score"])
                    col["correct"].append(m); col["n_correct_geometry_scorable"] += 1
                    correct_scored_iid[iid] = m["mask_center_mm"]
        elif ap_iid is not None and lat_iid is not None:                          # CROSS-INSTANCE mispair
            col["n_cross_instance_mispair"] += 1; has_phantom = True
            va = g["fl_groups"].get(int(ap_iid)) if g is not None else None
            vl = g["fl_groups"].get(int(lat_iid)) if g is not None else None
            if va is not None and vl is not None:   # both distances reportable
                col["cross"].append({"case": cid, "ap_iid": int(ap_iid), "lat_iid": int(lat_iid),
                    "dist_ap_fracture_mm": voxel_center_mm(p_rec, va, g["aff"]),
                    "dist_lat_fracture_mm": voxel_center_mm(p_rec, vl, g["aff"]),
                    "min_any_gt_mm": min_any_gt_mm(p_rec, g), "cand_score": float(c["score"])})
                col["n_incorrect_geometry_scorable"] += 1
        elif ap_iid is not None or lat_iid is not None:                           # ONE-SIDED false pair
            col["n_one_sided_false"] += 1; has_phantom = True
            known = int(ap_iid) if ap_iid is not None else int(lat_iid)
            side = "ap" if ap_iid is not None else "lat"
            vk = g["fl_groups"].get(known) if g is not None else None
            if vk is not None:                      # known-fracture distance reportable
                col["one_sided"].append({"case": cid, "known_side": side, "known_iid": known,
                    "phantom_to_known_fracture_mm": voxel_center_mm(p_rec, vk, g["aff"]),
                    "min_any_gt_mm": min_any_gt_mm(p_rec, g), "cand_score": float(c["score"])})
                col["n_incorrect_geometry_scorable"] += 1
        else:                                                                     # FULLY-FALSE pair
            col["n_fully_false"] += 1; has_phantom = True
            if is_neg: col["paired_false_3d_in_neg"] += 1
            dist = min_any_gt_mm(p_rec, g)          # positive case with two non-GT peaks -> dist to nearest real fx
            if dist is not None:
                col["fully_false"].append({"case": cid, "min_any_gt_mm": dist, "cand_score": float(c["score"])})
                col["n_incorrect_geometry_scorable"] += 1

    # GT-level cascade over THRESHOLD dual-view-available GT (distinct iids; empty on negative cases). Two
    # DISTINCT correct-pair numerators so each recall divides a numerator by its OWN cohort denominator:
    #   threshold cohort -> correct ∩ threshold-dual-view (recall vs peaks participating at this threshold)
    #   floor cohort     -> correct ∩ Stage-B floor-dual-view (survival of the SAME floor cohort)
    col["dual_view_gt_touched_by_paired_candidate"] += len(represented_iids)   # restricted to threshold set
    correct_threshold_here = correct_corr_iids & both_available_iids_thr
    correct_floor_here = correct_corr_iids & floor_both_available_iids
    col["correctly_paired_threshold_gt"] += len(correct_threshold_here)
    col["correctly_paired_floor_gt"] += len(correct_floor_here)
    for iid in correct_threshold_here:            # within-distance cascade stays on the threshold cohort
        if iid in correct_scored_iid:
            mc = correct_scored_iid[iid]
            if mc <= 5: col["correct_within_5"] += 1
            if mc <= 10: col["correct_within_10"] += 1
            if mc <= 15: col["correct_within_15"] += 1
    if has_correct: col["cases_with_correct"].add(cid)
    if has_phantom: col["cases_with_phantom"].add(cid)


def stat(vals):
    v = np.asarray([x for x in (vals or []) if x is not None], np.float64)
    if v.size == 0: return None
    return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2)}


def within(vals, ts=(5, 10, 15, 20)):
    v = np.asarray([x for x in (vals or []) if x is not None], np.float64)
    if v.size == 0: return None
    return {f"{t}mm": round(float((v <= t).mean()), 4) for t in ts}


def ratio(a, b):
    return round(a / b, 4) if b else None


def summarize(col, gt_total, both_available, n_cases, n_gt_negative_cases):
    cor = col["correct"]; n_cor_corr = col["n_correct_correspondence"]; n_paired = col["n_paired"]
    n_cross = col["n_cross_instance_mispair"]; mc = [r["mask_center_mm"] for r in cor]
    n_phantom = n_cross + col["n_one_sided_false"] + col["n_fully_false"]
    return {
        "availability": {"n_active_candidates": col["n_active"], "n_paired": n_paired,
                         "n_ap_only": col["n_ap_only"], "n_lat_only": col["n_lat_only"]},
        "correspondence": {
            "n_correct_correspondence": n_cor_corr, "n_cross_instance_mispair": n_cross,
            "n_one_sided_false": col["n_one_sided_false"], "n_fully_false": col["n_fully_false"],
            "n_paired_with_either_gt_peak": col["n_paired_either_gt"], "n_paired_with_two_gt_peaks": col["n_paired_both_gt"],
            "correct_pair_precision_among_all_paired": ratio(n_cor_corr, n_paired),
            "correct_pair_precision_when_either_peak_matches_gt": ratio(n_cor_corr, col["n_paired_either_gt"]),
            "cross_instance_rate_among_pairs_with_two_gt_peaks": ratio(n_cross, n_cor_corr + n_cross),
            "correct_pair_recall_among_floor_dual_view_gt": ratio(col["correctly_paired_floor_gt"], both_available),
            "correct_pair_recall_among_threshold_dual_view_gt": ratio(col["correctly_paired_threshold_gt"], col["both_views_available_at_threshold"]),
            "note": "precision_among_all_paired includes fully-false detections (operational); precision_when_either_"
                    "peak_matches_gt removes pure false-positive pairs; cross_instance_rate = cross/(correct+cross) "
                    "isolates matcher errors on pairs where BOTH peaks are real fractures. FLOOR recall (numerator = "
                    "correct ∩ floor cohort) = survival of the SAME Stage-B floor cohort through thresholding + pairing; "
                    "THRESHOLD recall (numerator = correct ∩ threshold cohort) = correspondence selection among GT "
                    "whose participating peaks survive at this threshold. Each numerator divides its OWN cohort."},
        "correct_pair_reconstruction": {
            "n_correspondence_correct": n_cor_corr, "n_geometry_scored": col["n_correct_geometry_scorable"],
            "distance_to_nearest_fracture_voxel_center_mm": {**(stat(mc) or {}), "within": within(mc)},
            "distance_to_centroid_mm": stat([r["centroid_mm"] for r in cor]),
            "anatomical_nearest_rib_exact": round(float(np.mean([r["rib_exact"] for r in cor])), 4) if cor else None,
            "anatomical_nearest_rib_within1": round(float(np.mean([r["rib_within1"] for r in cor])), 4) if cor else None,
            "along_rib_arc_length_error_mm": stat([r["along_mm"] for r in cor]),
            "inter_view_SI_disagreement_mm": stat([r["si_disagree_mm"] for r in cor]),
            "oracle_on_same_correct_instances": {
                "distance_to_nearest_fracture_voxel_center_mm": stat([r["oracle_mask_center_mm"] for r in cor]),
                "anatomical_nearest_rib_exact": round(float(np.mean([r["oracle_rib_exact"] for r in cor])), 4) if cor else None,
                "along_rib_arc_length_error_mm": stat([r["oracle_along_mm"] for r in cor]),
                "note": "Stage-B oracle-center reconstruction on EXACTLY these geometry-scored correct instances; the "
                        "gap is the residual localization cost, separate from the correspondence cost."}},
        "incorrect_pair_reconstruction": {
            "cross_instance": {"n_total": n_cross, "n_geometry_scored": len(col["cross"]),
                "dist_to_ap_fracture_mm": stat([r["dist_ap_fracture_mm"] for r in col["cross"]]),
                "dist_to_lat_fracture_mm": stat([r["dist_lat_fracture_mm"] for r in col["cross"]]),
                "min_dist_to_any_gt_mm": stat([r["min_any_gt_mm"] for r in col["cross"]])},
            "one_sided_false": {"n_total": col["n_one_sided_false"], "n_geometry_scored": len(col["one_sided"]),
                "phantom_to_known_fracture_mm": stat([r["phantom_to_known_fracture_mm"] for r in col["one_sided"]]),
                "min_dist_to_any_gt_mm": stat([r["min_any_gt_mm"] for r in col["one_sided"]])},
            "fully_false": {"n_total": col["n_fully_false"], "n_geometry_scored_positive_cases": len(col["fully_false"]),
                "min_dist_to_any_gt_mm_positive_cases": stat([r["min_any_gt_mm"] for r in col["fully_false"]])},
            "note": "mispairs have no single correct target and are NOT folded into correct-pair statistics. "
                    "fully-false distances are for positive cases only; negative-case fully-false are under "
                    "negative_case_accounting (no GT to measure against)."},
        "operational_end_to_end": {
            "frac_all_gt_correctly_paired_within_5mm": ratio(col["correct_within_5"], gt_total),
            "frac_all_gt_correctly_paired_within_10mm": ratio(col["correct_within_10"], gt_total),
            "frac_all_gt_correctly_paired_within_15mm": ratio(col["correct_within_15"], gt_total),
            "false_3d_points_total": n_phantom, "false_3d_points_per_case": round(n_phantom / n_cases, 3) if n_cases else None,
            "cases_with_at_least_one_correct_reconstruction": len(col["cases_with_correct"]),
            "cases_with_at_least_one_phantom_reconstruction": len(col["cases_with_phantom"]),
            "single_view_candidates_retained_not_triangulated": col["n_ap_only"] + col["n_lat_only"]},
        "negative_case_accounting": {
            "n_gt_negative_cases": n_gt_negative_cases,
            "n_negative_cases_with_active_candidate": len(col["neg_cases_with_active"]),
            "n_negative_cases_with_paired_candidate": len(col["neg_cases_with_paired"]),
            "paired_false_3d_points_in_negative_cases": col["paired_false_3d_in_neg"],
            "single_view_candidates_in_negative_cases": col["single_view_in_neg"]},
        "gt_level_cascade": {
            "floor_dual_view_available": both_available,
            "threshold_dual_view_available": col["both_views_available_at_threshold"],
            "dual_view_gt_touched_by_paired_candidate": col["dual_view_gt_touched_by_paired_candidate"],
            "correctly_paired_threshold": col["correctly_paired_threshold_gt"],
            "correctly_paired_floor": col["correctly_paired_floor_gt"],
            "correctly_paired_within_5mm": col["correct_within_5"],
            "correctly_paired_within_10mm": col["correct_within_10"], "correctly_paired_within_15mm": col["correct_within_15"],
            "note": "floor_dual_view_available is the Stage-B extraction-floor availability ESTIMATE. "
                    "threshold_dual_view_available may differ because one-to-one GT assignment is recomputed among "
                    "participating peaks; it does NOT imply the threshold created additional detections. Cascade "
                    "(threshold cohort): threshold dual-view -> touched by a paired candidate -> correctly_paired_threshold "
                    "-> within Xmm; each a subset of the previous (asserted). correctly_paired_* is correspondence "
                    "(geometry-independent); within_Xmm also requires geometry. correctly_paired_floor is the same correct "
                    "set intersected with the floor cohort (the numerator for FLOOR recall)."},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)   # FAIL-CLOSED weight hashes
    data_sha = verify_detector_data_provenance(a.data, rec)                             # FAIL-CLOSED data<->detector
    d = np.load(a.data, allow_pickle=False)

    cases = [str(c) for c in d["case"]]; case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    fp_case, fp_iid = d["fp_case"], d["fp_iid"]; ap_ctr, lat_ctr = d["ap_ctr"], d["lat_ctr"]
    gj = {}
    for j in range(len(fp_case)):
        key = (int(fp_case[j]), int(fp_iid[j]))
        if key in gj: raise ValueError(f"duplicate fracture identity in fp arrays: case {key[0]} iid {key[1]}")
        gj[key] = j
    recs = build_instance_records(d)
    val_ids = [str(c) for c in rec["split"]["val_case_ids"]]
    used_val = [c for c in val_ids if c in case_to_idx]
    va_idx = np.array([case_to_idx[c] for c in used_val], dtype=int)
    if va_idx.size == 0: raise RuntimeError("no validation cases from the dev-run record are present in --data")
    RADIUS = T.MATCH_RADIUS_PX; FLOOR = float(T.MIN_PEAK_SCORE)
    thresholds = {"operating_point": float(op_thr), "extraction_floor": FLOOR}
    print(f"Stage C: champion detector on {len(used_val)} detector-validation cases (development diagnostic; ALL "
          f"cases incl. GT-negative). FROZEN fusion si_tol={si_tol}, lat_gate={lat_gate}, op_thr={op_thr:.4f}. "
          f"match radius {RADIUS}px. thresholds: op {op_thr:.4f} + floor {FLOOR:g}.", flush=True)
    print("  running detector (peak_cache), building frozen fusion candidates, classifying + triangulating ...", flush=True)
    cache = T.peak_cache(nets, d, va_idx, dev)

    cols = {name: new_collector() for name in thresholds}
    gt_total = 0; ap_matched = lat_matched = both_available = 0; n_gt_negative_cases = 0
    C = len(va_idx)
    for c in range(C):
        ci = int(va_idx[c]); cid = cases[ci]; gt = recs.get(ci, [])
        if c % 25 == 0 or c == C - 1:
            op = cols["operating_point"]
            print(f"  [{c+1}/{C}] cases | correct-corr(op) {op['n_correct_correspondence']} "
                  f"| phantom(op) {op['n_cross_instance_mispair']+op['n_one_sided_false']+op['n_fully_false']}", flush=True)
        gt_total += len(gt)
        if not gt: n_gt_negative_cases += 1
        entry = cache[c]; ap_pk = T._peaks(entry, "ap"); lat_pk = T._peaks(entry, "lat")
        ap_geo_ci, lat_geo_ci = entry["ap_geo"], entry["lat_geo"]
        gt_ap = [x["ap_foot"] for x in gt]; gt_lat = [x["lat_foot"] for x in gt]
        ap_m = match([p[:2] for p in ap_pk], gt_ap, RADIUS)      # [] when negative
        lat_m = match([p[:2] for p in lat_pk], gt_lat, RADIUS)
        ap_by_gt = {gi: (pi, dd) for pi, gi, dd in ap_m}; lat_by_gt = {gi: (pi, dd) for pi, gi, dd in lat_m}
        ap_matched += len(ap_by_gt); lat_matched += len(lat_by_gt)
        # floor-level dual-view AVAILABILITY (the intentional recall ceiling denominator); as a set of iids
        both_gis = set(ap_by_gt) & set(lat_by_gt)
        floor_both_available_iids = {int(gt[gi]["iid"]) for gi in both_gis}   # Stage-B floor availability cohort (per case)
        both_available += len(floor_both_available_iids)                      # floor ceiling total (reporting/denominator)
        cands_all = T.build_case_candidates("fusion", ap_pk, lat_pk, ap_geo_ci, lat_geo_ci, si_tol, lat_gate)
        g = case_gt(cid, a.image_dirs, a.seg_dir, a.cl_dir) if gt else None  # geometry only needed for scoring
        for name, thr in thresholds.items():
            analyze_case(cols[name], thr, cands_all, ap_pk, lat_pk, ap_geo_ci, lat_geo_ci, gt,
                         floor_both_available_iids, g, cid, ci, gj, ap_ctr, lat_ctr)

    # invariants per threshold: candidate partition, class/geometry separation, cascade monotonicity
    for name, col in cols.items():
        assert col["n_paired"] == (col["n_correct_correspondence"] + col["n_cross_instance_mispair"]
                                   + col["n_one_sided_false"] + col["n_fully_false"]), name
        assert col["n_paired_both_gt"] == col["n_correct_correspondence"] + col["n_cross_instance_mispair"], name
        assert col["n_paired_either_gt"] >= col["n_paired_both_gt"], name
        assert col["n_correct_geometry_scorable"] <= col["n_correct_correspondence"], name
        assert col["correctly_paired_threshold_gt"] <= col["n_correct_correspondence"], name
        assert col["correctly_paired_floor_gt"] <= col["n_correct_correspondence"], name
        assert col["correctly_paired_floor_gt"] <= both_available, name
        assert col["dual_view_gt_touched_by_paired_candidate"] <= col["both_views_available_at_threshold"], name
        assert col["correctly_paired_threshold_gt"] <= col["dual_view_gt_touched_by_paired_candidate"], name
        assert col["correct_within_5"] <= col["correct_within_10"] <= col["correct_within_15"] <= col["correctly_paired_threshold_gt"], name

    result = {
        "stage": "C — predicted centers + deployable (frozen-fusion) correspondence",
        "diagnostic": "DEVELOPMENT diagnostic on the detector-VALIDATION split (champion selected on development "
                      "validation; SEALED test is the first confirmatory read). ALL val cases analyzed, including "
                      "GT-negative. The Stage-B floor dual-view availability is an assignment-dependent estimate of the "
                      "practical availability ceiling; Stage C measures survival through thresholding and deployable pairing.",
        "detector_run": str(a.detector_run), "data_sha256": data_sha, "val_cases_used": len(used_val),
        "correspondence_identity_assignment": "one-to-one GT matching among unique peaks participating in paired "
                                              "candidates at the evaluated candidate threshold",
        "frozen_fusion": {"si_tol": si_tol, "unmatched_lateral_gate": lat_gate, "fusion_operating_threshold": op_thr,
                          "match_radius_px": RADIUS},
        "shared_availability": {
            "gt_fractures": gt_total, "gt_negative_cases": n_gt_negative_cases,
            "ap_matched": ap_matched, "lat_matched": lat_matched, "both_views_available": both_available,
            "ap_recall": ratio(ap_matched, gt_total), "lat_recall": ratio(lat_matched, gt_total),
            "dual_view_availability": ratio(both_available, gt_total),
            "note": "peak extraction-floor availability ESTIMATE (threshold-independent, but one-to-one-assignment "
                    "dependent), continuing Stage B. Practical ceiling on triangulable recall: a correspondence method "
                    "can only pair GT detected in both views, and the floor peak set is maximal."},
        "at_operating_point": summarize(cols["operating_point"], gt_total, both_available, len(used_val), n_gt_negative_cases),
        "at_extraction_floor": summarize(cols["extraction_floor"], gt_total, both_available, len(used_val), n_gt_negative_cases),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    sa = result["shared_availability"]
    print(f"\nSTAGE C — dev correspondence diagnostic (detector-validation split; biased, see JSON) | "
          f"detector={a.detector_run.name}")
    print(f"  availability (floor): AP {sa['ap_recall']} | lat {sa['lat_recall']} | dual-view {sa['dual_view_availability']} "
          f"({both_available}/{gt_total}) = floor availability estimate | {n_gt_negative_cases} GT-negative cases")
    for name in ("operating_point", "extraction_floor"):
        s = result[f"at_{name}"]; cp = s["correspondence"]; cr = s["correct_pair_reconstruction"]
        op = s["operational_end_to_end"]; nc = s["negative_case_accounting"]
        mc = cr["distance_to_nearest_fracture_voxel_center_mm"]; orc = cr["oracle_on_same_correct_instances"]
        print(f"  [{name}] paired {s['availability']['n_paired']} -> correct {cp['n_correct_correspondence']} "
              f"(prec-all {cp['correct_pair_precision_among_all_paired']}, prec-either {cp['correct_pair_precision_when_either_peak_matches_gt']}, "
              f"recall floor/thr {cp['correct_pair_recall_among_floor_dual_view_gt']}/{cp['correct_pair_recall_among_threshold_dual_view_gt']}) "
              f"| cross {cp['n_cross_instance_mispair']} one-sided {cp['n_one_sided_false']} false {cp['n_fully_false']} "
              f"| cross-rate(2GT) {cp['cross_instance_rate_among_pairs_with_two_gt_peaks']}")
        print(f"       correct-pair 3D (n={cr['n_geometry_scored']}): median {mc.get('median')} p90 {mc.get('p90')} within {mc.get('within')} "
              f"| rib-exact {cr['anatomical_nearest_rib_exact']} (oracle {orc['anatomical_nearest_rib_exact']})")
        print(f"       end-to-end: all-GT correct within 5/10/15mm = "
              f"{op['frac_all_gt_correctly_paired_within_5mm']}/{op['frac_all_gt_correctly_paired_within_10mm']}/"
              f"{op['frac_all_gt_correctly_paired_within_15mm']} | phantoms/case {op['false_3d_points_per_case']} "
              f"| neg-case phantoms {nc['paired_false_3d_points_in_negative_cases']} in {nc['n_negative_cases_with_paired_candidate']} neg cases")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
