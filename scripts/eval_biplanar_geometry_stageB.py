#!/usr/bin/env python3
"""STAGE B — PREDICTED centers + ORACLE correspondence. A DEVELOPMENT LOCALIZATION DIAGNOSTIC on the
detector-validation split (NOT an unbiased detector-generalization estimate — see the bias note below).

Stage A proved the geometry: given the CORRECT paired AP+lateral centers, orthographic back-projection
lands on the fracture (median 0 mm to nearest fracture voxel, ~98.6% correct rib, along-rib s ~0.02).
Stage B changes exactly ONE variable — it replaces the oracle centers with the CHAMPION DETECTOR's own
predicted peaks — to isolate detector localization error. Correspondence stays oracle: each predicted
peak is matched to its KNOWN GT fracture INDEPENDENTLY in each view, oracle correspondence is the
intersection of matched GT INSTANCE IDs (iid), and the AP-peak and lateral-peak of the SAME instance are
triangulated together. The AP<->lateral matching problem is deliberately NOT solved here — that is Stage C.

BIAS NOTE (do not overclaim): the cases are held out from detector FITTING, but the champion checkpoint,
architecture, lateral gate, and operating configuration were SELECTED using development-validation
results. This is therefore a biased (optimistic) development localization diagnostic on the detector-
validation split, not a held-out generalization estimate. The SEALED test is the first confirmatory read.
It IS valid for: measuring 3D localization CONDITIONAL ON dual-view detection, isolating the detector-
localization cost against the oracle centers on the identical fractures, and attributing 3D degradation to
its source (AP vs lateral 2D error, SI disagreement, confidence). It must NOT be used to reselect the
detector.

Reuse, not reimplementation:
  * Detector peaks: train_detector.peak_cache / peaks_from_hm on the champion checkpoints; the detector is
    rebuilt with run_ribassist.load_detector (FAIL-CLOSED per-view weight hashes).
  * Data provenance: FAIL-CLOSED sha256(--data) == detector_dev_run.json['det_dev_sha256'] (missing = error),
    the same required check run_ribassist enforces — so the champion weights cannot score a different npz.
  * Explicit-iid GT + matching: eval_address_e2e.build_instance_records (footprints sliced in global fp
    order, ASSERTED to reproduce T.group_instances so every Hungarian column carries the right iid) and
    eval_address_e2e.match (the detector's own T._min_dist + T.MATCH_RADIUS_PX one-to-one assignment).
  * Scoring: the shared eval_biplanar_geometry.fracture_metrics — rib side/number and along-rib position
    are DERIVED after reconstruction (never predicted before). Every dual-view fracture is scored TWICE,
    predicted-center and oracle-center, on the IDENTICAL instance set (asserted equal), so the gap is the
    clean localization cost.

Peaks are taken at the per-view EXTRACTION FLOOR (score >= MIN_PEAK_SCORE) because Stage B measures
localization CONDITIONAL ON a per-view detection. --research-per-view-peak-floor only RAISES that per-view
floor for research; it is NOT a deployment operating point. The fusion operating threshold (~0.1583) is a
threshold on the FUSED candidate score from build_case_candidates, not on individual AP/lat peaks —
applying it per-view would define a different algorithm. Deployment operating-point behaviour belongs in
Stage C, through the frozen fusion path.

Usage:
  python eval_biplanar_geometry_stageB.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/eval_biplanar_geometry_stageB.json
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
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e


def verify_detector_data_provenance(data_path, det_rec):
    """FAIL-CLOSED: sha256(--data) must equal detector_dev_run.json['det_dev_sha256']. Missing field is an
    ERROR, not a skipped check (same required guard run_ribassist.verify_provenance enforces for the
    detector<->data link). Returns the data sha256."""
    exp = det_rec.get("det_dev_sha256")
    if not exp:
        raise ValueError("detector_dev_run.json is missing required provenance field 'det_dev_sha256'")
    data_sha = T.sha256_file(data_path)
    if data_sha != exp:
        raise ValueError(f"--data hash mismatch: current {data_sha[:12]}.. != detector run det_dev_sha256 {exp[:12]}..")
    return data_sha


def stat(vals):
    if not vals: return None
    v = np.asarray(vals, np.float64)
    return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "p95": round(float(np.percentile(v, 95)), 2)}


def within(vals, ts=(5, 10, 15, 20, 30)):
    if not vals: return None
    v = np.asarray(vals, np.float64); return {f"{t}mm": round(float((v <= t).mean()), 4) for t in ts}


def corr(xs, ys):
    x = np.asarray(xs, np.float64); y = np.asarray(ys, np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0: return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--research-per-view-peak-floor", type=float, default=None,
                    help="RESEARCH ONLY: raise the per-view peak score floor above MIN_PEAK_SCORE. This is NOT "
                         "a deployment operating point (the fusion op-threshold is on FUSED candidate scores, "
                         "not per-view peaks); deployment thresholding belongs in Stage C.")
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")

    base_floor = float(T.MIN_PEAK_SCORE)
    if a.research_per_view_peak_floor is None:
        floor = base_floor; floor_is_research = False
    else:
        floor = float(a.research_per_view_peak_floor)
        if floor < base_floor:
            raise ValueError(f"--research-per-view-peak-floor {floor} is below MIN_PEAK_SCORE {base_floor}; "
                             "Stage B cannot recover peaks that peak_cache already discarded.")
        floor_is_research = True   # an override AT the floor is still 'applied' (a no-op filter), so it is recorded honestly
    dev = T.device()
    # FAIL-CLOSED detector load (per-view weight hashes) — reuse the deployed loader verbatim
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)
    data_sha = verify_detector_data_provenance(a.data, rec)   # FAIL-CLOSED data<->detector provenance, BEFORE opening the npz
    d = np.load(a.data, allow_pickle=False)

    cases = [str(c) for c in d["case"]]
    case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    fp_case, fp_iid = d["fp_case"], d["fp_iid"]
    ap_ctr, lat_ctr = d["ap_ctr"], d["lat_ctr"]; ap_geo, lat_geo = d["ap_geo"], d["lat_geo"]
    # (case, iid) -> GLOBAL fp index, for the oracle center + the per-view 2D-error target. Built explicitly
    # so a duplicate fracture identity FAILS rather than silently overwriting (preserves the iid guarantee).
    gj = {}
    for j in range(len(fp_case)):
        key = (int(fp_case[j]), int(fp_iid[j]))
        if key in gj:
            raise ValueError(f"duplicate fracture identity in fp arrays: case index {key[0]}, iid {key[1]}")
        gj[key] = j
    recs = build_instance_records(d)                          # explicit-iid records, ASSERTED == group_instances

    val_ids = [str(c) for c in rec["split"]["val_case_ids"]]
    used_val = [c for c in val_ids if c in case_to_idx]
    dropped_val = [c for c in val_ids if c not in case_to_idx]
    va_idx = np.array([case_to_idx[c] for c in used_val], dtype=int)
    if va_idx.size == 0: raise RuntimeError("no validation cases from the dev-run record are present in --data")
    RADIUS = T.MATCH_RADIUS_PX
    print(f"Stage B: champion detector on {len(used_val)} detector-validation cases "
          f"(development diagnostic; model/config selected using development validation)"
          + (f" [{len(dropped_val)} val ids not in --data]" if dropped_val else "")
          + f". per-view floor score>={floor:g}, match radius {RADIUS}px, ORACLE correspondence by iid.", flush=True)
    print("  running detector (peak_cache) then per-case matching + triangulation ...", flush=True)
    cache = T.peak_cache(nets, d, va_idx, dev)

    # instance-level coverage cascade (each stage a strict subset of the previous — asserted below)
    casc = {"gt_fractures_in_scored_cases": 0, "ap_matched": 0, "lat_matched": 0,
            "both_views_matched_same_iid": 0, "fracture_geometry_available": 0, "anatomically_scorable": 0}
    # case-level denominators
    cc = {"total_val_cases": len(used_val), "cases_with_fractures": 0, "cases_with_gt_geometry": 0,
          "cases_with_dual_view_detection": 0, "cases_contributing_reconstruction": 0}
    skipped = {"gt_geometry_missing": 0, "label_absent": 0, "no_rib_overlap": 0, "rib_out_of_range": 0}
    PRED, ORACLE = [], []

    C = len(va_idx)
    for c in range(C):
        ci = int(va_idx[c]); cid = cases[ci]; gt = recs.get(ci, [])
        if c % 25 == 0 or c == C - 1:
            print(f"  [{c+1}/{C}] cases | both-view {casc['both_views_matched_same_iid']} | scored {len(PRED)}", flush=True)
        casc["gt_fractures_in_scored_cases"] += len(gt)
        if not gt: continue
        cc["cases_with_fractures"] += 1
        entry = cache[c]
        ap_pk = T._peaks(entry, "ap"); lat_pk = T._peaks(entry, "lat")
        if floor_is_research:
            if len(ap_pk): ap_pk = ap_pk[ap_pk[:, 2] >= floor]
            if len(lat_pk): lat_pk = lat_pk[lat_pk[:, 2] >= floor]
        gt_ap = [g["ap_foot"] for g in gt]; gt_lat = [g["lat_foot"] for g in gt]
        ap_by_gt = {gi: (pi, dd) for pi, gi, dd in match([p[:2] for p in ap_pk], gt_ap, RADIUS)}
        lat_by_gt = {gi: (pi, dd) for pi, gi, dd in match([p[:2] for p in lat_pk], gt_lat, RADIUS)}
        casc["ap_matched"] += len(ap_by_gt); casc["lat_matched"] += len(lat_by_gt)
        both = sorted(set(ap_by_gt) & set(lat_by_gt))         # GT instances matched in BOTH views (same iid)
        casc["both_views_matched_same_iid"] += len(both)
        if both: cc["cases_with_dual_view_detection"] += 1
        # load GT geometry once per fracture-bearing case (clean geometry denominator + reused for reconstruction)
        g = case_gt(cid, a.image_dirs, a.seg_dir, a.cl_dir)
        if g is not None: cc["cases_with_gt_geometry"] += 1
        contributed = False
        for gi in both:
            iid = int(gt[gi]["iid"])
            if g is None: skipped["gt_geometry_missing"] += 1; continue
            vox = g["fl_groups"].get(iid)
            if vox is None or vox.shape[1] == 0: skipped["label_absent"] += 1; continue
            casc["fracture_geometry_available"] += 1
            api, ad = ap_by_gt[gi]; li, ld = lat_by_gt[gi]
            ap_rc = ap_pk[api, :2].astype(np.float64); lat_rc = lat_pk[li, :2].astype(np.float64)
            ap_sc = float(ap_pk[api, 2]); lat_sc = float(lat_pk[li, 2])
            # PREDICTED-center reconstruction
            p_pred, si_pred = back_project(ap_rc, lat_rc, ap_geo[ci], lat_geo[ci])
            mp, reason = fracture_metrics(p_pred, si_pred, vox, g, cid, iid)
            if mp is None: skipped[reason] += 1; continue
            # ORACLE-center reconstruction on the SAME instance (must succeed; else accounting defect)
            j = gj[(ci, iid)]
            p_or, si_or = back_project(ap_ctr[j], lat_ctr[j], ap_geo[ci], lat_geo[ci])
            mo, mo_reason = fracture_metrics(p_or, si_or, vox, g, cid, iid)
            if mo is None:
                raise RuntimeError(f"accounting defect: predicted metric succeeded but oracle failed "
                                   f"({mo_reason}) for case {cid} iid {iid}")
            e_ap = float(np.linalg.norm(ap_rc - np.asarray(ap_ctr[j], np.float64)))
            e_lat = float(np.linalg.norm(lat_rc - np.asarray(lat_ctr[j], np.float64)))
            mp.update({"ap_score": ap_sc, "lat_score": lat_sc, "ap_match_px": ad, "lat_match_px": ld,
                       "ap2d_px": e_ap, "lat2d_px": e_lat, "max2d_px": max(e_ap, e_lat)})
            casc["anatomically_scorable"] += 1
            PRED.append(mp); ORACLE.append(mo); contributed = True
        if contributed: cc["cases_contributing_reconstruction"] += 1

    if not PRED: raise RuntimeError("no dual-view reconstructable fractures on the validation split")

    # ---- cascade + same-set invariants (subset monotonicity; predicted/oracle on identical instances) ----
    assert casc["ap_matched"] <= casc["gt_fractures_in_scored_cases"]
    assert casc["lat_matched"] <= casc["gt_fractures_in_scored_cases"]
    assert casc["both_views_matched_same_iid"] <= min(casc["ap_matched"], casc["lat_matched"])
    assert casc["fracture_geometry_available"] <= casc["both_views_matched_same_iid"]
    assert casc["anatomically_scorable"] <= casc["fracture_geometry_available"]
    assert casc["anatomically_scorable"] == len(PRED) == len(ORACLE)
    assert [(x["case"], x["iid"]) for x in PRED] == [(x["case"], x["iid"]) for x in ORACLE], \
        "predicted and oracle metrics are not on the identical fracture set/order"

    n = len(PRED)
    mc = [r["mask_center_mm"] for r in PRED]; mv = [r["mask_volume_lower_bound_mm"] for r in PRED]
    cen = [r["centroid_mm"] for r in PRED]
    mc_or = [r["mask_center_mm"] for r in ORACLE]
    max2d = [r["max2d_px"] for r in PRED]
    strat = {}
    for lo, hi, name in [(0, 2, "<=2px"), (2, 4, "2-4px"), (4, 8, "4-8px"), (8, 1e9, ">8px")]:
        sel = [mc[i] for i in range(n) if lo <= max2d[i] < hi]
        strat[name] = {"n": len(sel), "mask_center_mm_median": (round(float(np.median(sel)), 2) if sel else None),
                       "within": within(sel)}
    worst = sorted(PRED, key=lambda r: -r["mask_center_mm"])[:15]

    result = {
        "stage": "B — predicted centers + oracle correspondence",
        "diagnostic": "DEVELOPMENT LOCALIZATION DIAGNOSTIC on the detector-VALIDATION split — NOT an unbiased "
                      "detector-generalization estimate. Cases are held out from detector FITTING, but the champion "
                      "checkpoint/architecture/lateral-gate/operating config were SELECTED on development-validation "
                      "results, so this is optimistically biased. The SEALED test is the first confirmatory read. "
                      "Valid for: 3D localization conditional on dual-view detection, isolating the detector-"
                      "localization cost vs oracle centers, and failure-source attribution. Not for reselecting the detector.",
        "detector_run": str(a.detector_run), "data_sha256": data_sha,
        "per_view_peak_floor": floor, "per_view_peak_floor_is_research_override": bool(floor_is_research),
        "match_radius_px": RADIUS, "correspondence": "oracle (intersection of matched GT instance ids)",
        "val_cases_used": len(used_val), "val_cases_in_record_missing_from_data": dropped_val,
        "coverage_cascade": {**casc,
            "note": "instance-level; each downstream stage is a subset of the applicable upstream stage (nested/"
                    "non-increasing, asserted). Chain: GT -> AP matched -> lateral matched -> both views matched to "
                    "the same iid -> fracture geometry available -> anatomically scorable. Coverage is at the per-view "
                    "extraction floor (availability ceiling); deployment thresholding + AP<->lat matching are Stage C. "
                    "Accuracy below is on the anatomically_scorable subset only."},
        "case_denominators": {**cc,
            "note": "cases_with_gt_geometry counts fracture-bearing val cases whose CT-derived GT inputs loaded."},
        "skipped_instances": skipped,
        "localization_predicted_centers": {
            "n": n,
            "distance_to_nearest_fracture_voxel_center_mm": {**stat(mc), "within": within(mc)},
            "distance_to_fracture_volume_lower_bound_mm": {**stat(mv), "within": within(mv),
                "interpretation": "LOWER bound (nearest voxel-center distance minus max center-to-corner radius, "
                                  "floored at 0); zero does NOT prove the point is inside the mask."},
            "distance_to_centroid_mm": {**stat(cen), "within": within(cen)},
            "component_error_mm": {"LR": stat([r["lr_mm"] for r in PRED]), "AP": stat([r["ap_mm"] for r in PRED]),
                                   "SI": stat([r["si_mm"] for r in PRED])},
            "inter_view_SI_disagreement_mm": stat([r["si_disagree_mm"] for r in PRED]),
            "anatomical_nearest_rib_exact": round(float(np.mean([r["rib_exact"] for r in PRED])), 4),
            "anatomical_nearest_rib_within1": round(float(np.mean([r["rib_within1"] for r in PRED])), 4),
            "distance_to_correct_rib_centerline_mm": stat([r["dist_correct_cl_mm"] for r in PRED]),
            "along_rib_normalized_s_error": stat([r["s_err"] for r in PRED]),
            "along_rib_arc_length_error_mm": stat([r["along_mm"] for r in PRED])},
        "oracle_center_reference_same_fractures": {
            "distance_to_nearest_fracture_voxel_center_mm": {**stat(mc_or), "within": within(mc_or)},
            "anatomical_nearest_rib_exact": round(float(np.mean([r["rib_exact"] for r in ORACLE])), 4),
            "along_rib_arc_length_error_mm": stat([r["along_mm"] for r in ORACLE]),
            "note": "Stage A's oracle-center metric recomputed on EXACTLY the Stage-B scored fractures (asserted "
                    "identical set/order); the predicted-vs-oracle gap is the isolated detector-localization cost."},
        "source_attribution": {
            "ap_2d_localization_error_px": stat([r["ap2d_px"] for r in PRED]),
            "lat_2d_localization_error_px": stat([r["lat2d_px"] for r in PRED]),
            "max_2d_localization_error_px": stat(max2d),
            "ap_peak_score": stat([r["ap_score"] for r in PRED]), "lat_peak_score": stat([r["lat_score"] for r in PRED]),
            "corr_3Dmm_with": {
                "ap_2d_px": corr([r["ap2d_px"] for r in PRED], mc), "lat_2d_px": corr([r["lat2d_px"] for r in PRED], mc),
                "max_2d_px": corr(max2d, mc), "si_disagree_mm": corr([r["si_disagree_mm"] for r in PRED], mc),
                "ap_score": corr([r["ap_score"] for r in PRED], mc), "lat_score": corr([r["lat_score"] for r in PRED], mc)},
            "mask_center_mm_by_max2d_error_bucket": strat,
            "note": "2D error = matched predicted peak vs the stored per-view target center (image px; match radius "
                    "8px). Correlations/buckets show whether AP error, lateral error, SI disagreement, or low peak "
                    "confidence dominates the 3D degradation."},
        "worst_by_nearest_fracture_voxel_distance": [
            {"case": w["case"], "iid": w["iid"], "mask_center_mm": round(w["mask_center_mm"], 1),
             "centroid_mm": round(w["centroid_mm"], 1), "ap2d_px": round(w["ap2d_px"], 1),
             "lat2d_px": round(w["lat2d_px"], 1), "ap_score": round(w["ap_score"], 3),
             "lat_score": round(w["lat_score"], 3), "si_disagree_mm": round(w["si_disagree_mm"], 1)} for w in worst],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    L = result["localization_predicted_centers"]; O = result["oracle_center_reference_same_fractures"]
    S = result["source_attribution"]; mc_ = L["distance_to_nearest_fracture_voxel_center_mm"]
    gtot = casc["gt_fractures_in_scored_cases"]
    print(f"\nSTAGE B — dev localization diagnostic (detector-validation split; biased, see JSON) | "
          f"detector={a.detector_run.name}")
    print(f"  cascade: GT {gtot} -> AP {casc['ap_matched']} / lat {casc['lat_matched']} "
          f"-> both-iid {casc['both_views_matched_same_iid']} -> geom {casc['fracture_geometry_available']} "
          f"-> scored {casc['anatomically_scorable']}")
    print(f"  dual-view coverage {round(casc['both_views_matched_same_iid']/gtot,4) if gtot else None} "
          f"| cases: {cc['cases_with_fractures']} frac / {cc['cases_with_gt_geometry']} geom / "
          f"{cc['cases_contributing_reconstruction']} contributing (of {cc['total_val_cases']})")
    print(f"  PREDICTED nearest fracture VOXEL-CENTER mm: median {mc_['median']} | p90 {mc_['p90']} | within {mc_['within']}")
    print(f"    ORACLE (same {n} fractures)   voxel-center mm: median "
          f"{O['distance_to_nearest_fracture_voxel_center_mm']['median']} | "
          f"p90 {O['distance_to_nearest_fracture_voxel_center_mm']['p90']}")
    print(f"  component mm (mean): LR {L['component_error_mm']['LR']['mean']} | "
          f"AP {L['component_error_mm']['AP']['mean']} | SI {L['component_error_mm']['SI']['mean']}")
    print(f"  rib exact {L['anatomical_nearest_rib_exact']} (oracle {O['anatomical_nearest_rib_exact']}) "
          f"| rib±1 {L['anatomical_nearest_rib_within1']} | along-rib arc mm "
          f"{L['along_rib_arc_length_error_mm']['mean']} (oracle {O['along_rib_arc_length_error_mm']['mean']})")
    print(f"  2D loc err px (mean): AP {S['ap_2d_localization_error_px']['mean']} | lat {S['lat_2d_localization_error_px']['mean']} "
          f"| corr(3Dmm~max2Dpx) {S['corr_3Dmm_with']['max_2d_px']} corr(3Dmm~SIdisagree) {S['corr_3Dmm_with']['si_disagree_mm']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
