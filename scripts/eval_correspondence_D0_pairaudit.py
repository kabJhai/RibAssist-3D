#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""STAGE D0 — CANDIDATE-PAIR DATA AUDIT + BROAD PAIR GRAPH for biplanar correspondence (no learned model).

Stage C showed the frozen SI-only fusion matcher (train_detector.form_pairs — a one-to-one Hungarian
assignment whose cost/reject model is inadequate for correspondence, NOT a greedy chooser) fails as a
triangulation correspondence mechanism. Before building any correspondence model, D0 characterizes the
candidate-pair DATA and emits a self-contained pair dataset so we neither train a scorer to solve a problem
candidate generation already made unsolvable, nor mis-diagnose which stage loses the correct pair.

IDENTITY is MANY-TO-ONE (graph existence is a many-peaks-to-one-fracture question, unlike Stage B/C's
one-to-one detection scoring, kept here only as a secondary continuity audit): every peak within
MATCH_RADIUS_PX of a GT footprint is COMPATIBLE with that fracture. A pair is classified by the INTERSECTION
of its compatible-iid sets: positive_capable (shared nonempty), cross_instance_only (both match GT, disjoint),
one_sided (one has any iid), fully_false (neither). This never calls a duplicate true peak spurious.

BROAD GRAPH: the deployed gate si_tol was tuned for 2D fusion, not correspondence, so a gate SWEEP is needed
to see whether correct pairs sit just past it. The emitted NPZ therefore contains EVERY AP x lateral edge with
|dSI| <= --audit-gate-max (default 30 vox, deliberately conservative), each with coordinates + geometry so D1
can filter to any narrower gate AND reconstruct accepted pairs WITHOUT rerunning the detector. The JSON
headline decomposition is still at the DEPLOYED si_tol; a gate_sensitivity_sweep reports, per gate,
existential correct-edge recall, edges, class composition, contamination, and competitors.

CORRECT-PAIR AVAILABILITY is EXISTENTIAL per fracture (any admissible same-iid combination), giving the true
loss decomposition: detection-availability (no compatible peak in a view -> fix detector/lateral recall) ->
SI-gate removal (compatible peaks both views but no same-iid combo within the gate -> redesign candidate
generation; a scorer cannot recover these) -> selection loss (a same-iid combo is in-graph but the frozen
assignment did not pick a compatible pair -> fixable by a better scorer/assignment). SELECTION is split into
unique (maps to exactly one iid = actionable) vs ambiguous (overlapping iids = uncertainty bucket).

DIAGNOSTIC STATUS: development audit on the detector-validation split (biased; sealed test first confirmatory).
Provenance fails closed. Governing objective is 3D RECONSTRUCTION: D0 only diagnoses candidate generation for it.

Usage:
  python eval_correspondence_D0_pairaudit.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --audit-gate-max 30 \
      --out outputs/correspondence_D0_pairaudit.json
"""
from __future__ import annotations
import argparse, json, sys
from itertools import product
from pathlib import Path
import numpy as np

try:
    import torch  # noqa: F401
    import nibabel as nib
    import train_detector as T
    import run_ribassist as RR
    from eval_address_e2e import build_instance_records, match
    from eval_biplanar_geometry_stageB import verify_detector_data_provenance
    from make_rib_targets import _find
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

CLS = ["positive_capable", "cross_instance_only", "one_sided", "fully_false"]
GRID_CANDIDATES = (6, 8, 10, 12, 16, 20, 30)


def stat(vals):
    v = np.asarray([x for x in (vals if vals is not None else []) if x is not None and np.isfinite(x)], np.float64)
    if v.size == 0: return None
    return {"n": int(v.size), "mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
            "p10": round(float(np.percentile(v, 10)), 3), "p90": round(float(np.percentile(v, 90)), 3),
            "max": round(float(v.max()), 3)}


def ratio(a, b):
    return round(a / b, 4) if b else None


def si_mm_per_vox(cid, image_dirs):
    """Canonical superior-inferior voxel size (mm) from the label NIfTI HEADER only (no data materialized)."""
    imp = _find(image_dirs, f"{cid}-label.nii.gz", f"{cid}-label.nii", f"{cid}-label.nii*")
    if imp is None: return None
    can = nib.as_closest_canonical(nib.load(str(imp)))
    return float(nib.affines.voxel_sizes(can.affine)[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="*", default=None)
    ap.add_argument("--audit-gate-max", type=float, default=30.0,
                    help="emit EVERY AP x lateral edge with |dSI(vox)| <= this into the NPZ (broad graph for the "
                         "D1 gate sweep). Default 30. The JSON headline decomposition stays at the deployed si_tol.")
    # ---- L1 extraction policy (per view). Defaults reproduce the deployed extraction exactly. ----
    ap.add_argument("--ap-nms", type=int, default=T.NMS_RADIUS_PX, help="AP peak NMS radius (px). Default deployed.")
    ap.add_argument("--ap-floor", type=float, default=T.MIN_PEAK_SCORE, help="AP peak score floor. Default deployed.")
    ap.add_argument("--lat-nms", type=int, default=T.NMS_RADIUS_PX, help="lateral peak NMS radius (px). Default deployed.")
    ap.add_argument("--lat-floor", type=float, default=T.MIN_PEAK_SCORE, help="lateral peak score floor. Default deployed.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-data-sha256", default=None,
                    help="SEALED mode: bind to this external data digest instead of the detector's dev data hash, "
                         "and evaluate ALL cases in --data (not the dev-run val split). Frozen policy; no selection.")
    a = ap.parse_args()
    for nm, val in (("--ap-nms", a.ap_nms), ("--lat-nms", a.lat_nms)):
        if val < 1: raise ValueError(f"{nm} must be >= 1")
    for nm, val in (("--ap-floor", a.ap_floor), ("--lat-floor", a.lat_floor)):
        if not (0.0 < val < 1.0): raise ValueError(f"{nm} must be in (0,1)")
    pol = {"ap": (int(a.ap_nms), float(a.ap_floor)), "lat": (int(a.lat_nms), float(a.lat_floor))}
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)   # FAIL-CLOSED weight hashes
    sealed = a.expected_data_sha256 is not None
    if sealed:                                                                         # SEALED external-anchor binding
        data_sha = T.sha256_file(a.data)
        if data_sha != a.expected_data_sha256:
            raise ValueError(f"--data sha256 {data_sha[:12]}.. != --expected-data-sha256 {a.expected_data_sha256[:12]}..")
    else:
        data_sha = verify_detector_data_provenance(a.data, rec)                         # FAIL-CLOSED data<->detector (dev)
    d = np.load(a.data, allow_pickle=False)
    cases = [str(c) for c in d["case"]]; case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    recs = build_instance_records(d)
    val_ids = cases if sealed else [str(c) for c in rec["split"]["val_case_ids"]]        # sealed = ALL cases in --data
    used_val = [c for c in val_ids if c in case_to_idx]
    va_idx = np.array([case_to_idx[c] for c in used_val], dtype=int)
    if va_idx.size == 0: raise RuntimeError("no validation cases from the dev-run record are present in --data")
    RADIUS = T.MATCH_RADIUS_PX; want_mm = bool(a.image_dirs); GMAX = float(a.audit_gate_max)
    if not np.isfinite(GMAX) or GMAX <= 0:
        raise ValueError("--audit-gate-max must be a finite positive number")
    if GMAX < float(si_tol):
        raise ValueError(f"--audit-gate-max {GMAX} must be >= deployed si_tol {si_tol}; otherwise the broad graph "
                         "would omit deployed-gate edges.")
    GRID = sorted({float(si_tol), GMAX} | {float(g) for g in GRID_CANDIDATES if g <= GMAX})
    deployed_pol = (pol["ap"] == (T.NMS_RADIUS_PX, T.MIN_PEAK_SCORE) and pol["lat"] == (T.NMS_RADIUS_PX, T.MIN_PEAK_SCORE))
    print(f"D0 pair audit: champion detector on {len(used_val)} detector-validation cases. si_tol={si_tol} vox "
          f"(deployed headline), broad graph |dSI|<= {GMAX} vox, match radius {RADIUS}px, MANY-TO-ONE.", flush=True)
    print(f"  extraction policy: ap nms{pol['ap'][0]}/floor{pol['ap'][1]}  lat nms{pol['lat'][0]}/floor{pol['lat'][1]}"
          f"  ({'DEPLOYED' if deployed_pol else 'L1-recalibrated'})", flush=True)

    def policy_peak_cache(nets, d, idx, dev, pol):
        """peak_cache with per-view (NMS radius, floor). Same primitive (peaks_from_hm) as deployed; params only."""
        cache = []
        for n in nets.values(): n.eval()
        with torch.no_grad():
            for i in idx:
                entry = {"ap_geo": d["ap_geo"][i], "lat_geo": d["lat_geo"][i]}
                for v in ("ap", "lat"):
                    if v in nets:
                        hm = nets[v](torch.from_numpy(d[v][i].astype(np.float32))[None, None].to(dev))[0, 0]
                        r, f = pol[v]; entry[v] = T.peaks_from_hm(hm, radius=r, thresh=f)
                cache.append(entry)
        return cache
    cache = policy_peak_cache(nets, d, va_idx, dev, pol)

    P = {k: [] for k in ("case_global_idx", "case_val_idx", "case_id", "ap_idx", "lat_idx",
                         "ap_row", "ap_col", "lat_row", "lat_col", "ap_score", "lat_score",
                         "dsi_vox", "dsi_mm", "cls", "n_shared_iids", "shared_iid", "ap_single_iid", "lat_single_iid",
                         "identity_ambiguous", "ap_is_dup", "lat_is_dup", "selected_by_form_pairs")}
    amb_sidecar = {}
    cls_count = {k: 0 for k in CLS}          # DEPLOYED-gate (<= si_tol) headline class counts
    potential_before = 0; after_gate = 0; ap_pc = []; lat_pc = []; positives_per_case = []
    identity_ambiguous_pairs = 0
    # gate sweep accumulators
    sweep_edges = {g: 0 for g in GRID}; sweep_cls = {g: {k: 0 for k in CLS} for g in GRID}
    sweep_comp = {g: 0 for g in GRID}
    # existential correct-pair availability
    gt_total = 0; exist_ap = exist_lat = exist_both = 0
    correct_exists_after_gate = 0; removed_by_gate = 0; lost_in_assignment = 0
    correct_selected_compatible = correct_selected_unique = correct_selected_ambiguous = 0
    n_ambiguous = 0; ap_comp = []; lat_comp = []; dsi_correct_min = []
    s1_dual = 0; s1_in_graph = 0; s1_selected = 0

    C = len(va_idx)
    for c in range(C):
        ci = int(va_idx[c]); cid = cases[ci]; gt = recs.get(ci, [])
        entry = cache[c]; ap_pk = T._peaks(entry, "ap"); lat_pk = T._peaks(entry, "lat")
        n_ap, n_lat = len(ap_pk), len(lat_pk); ap_pc.append(n_ap); lat_pc.append(n_lat)
        potential_before += n_ap * n_lat; gt_total += len(gt)
        if c % 25 == 0 or c == C - 1:
            print(f"  [{c+1}/{C}] cases | broad edges {len(P['case_global_idx'])} | pos-capable(<=si_tol) {cls_count['positive_capable']}", flush=True)
        smm = si_mm_per_vox(cid, a.image_dirs) if want_mm else None
        ap_p2i = [set() for _ in range(n_ap)]; lat_p2i = [set() for _ in range(n_lat)]
        iid_ap = {}; iid_lat = {}
        for gi, x in enumerate(gt):
            iid = int(x["iid"])
            for i in range(n_ap):
                if T._min_dist(ap_pk[i, :2], x["ap_foot"]) <= RADIUS: ap_p2i[i].add(iid); iid_ap.setdefault(iid, set()).add(i)
            for j in range(n_lat):
                if T._min_dist(lat_pk[j, :2], x["lat_foot"]) <= RADIUS: lat_p2i[j].add(iid); iid_lat.setdefault(iid, set()).add(j)
        ap_dup = [any(len(iid_ap.get(iid, ())) > 1 for iid in s) for s in ap_p2i]
        lat_dup = [any(len(iid_lat.get(iid, ())) > 1 for iid in s) for s in lat_p2i]
        si_ap = T.si_voxel(ap_pk[:, 0], entry["ap_geo"]) if n_ap else np.empty(0, np.float64)
        si_lat = T.si_voxel(lat_pk[:, 0], entry["lat_geo"]) if n_lat else np.empty(0, np.float64)
        dsi = np.abs(si_ap[:, None] - si_lat[None, :])
        after_gate += int((dsi <= si_tol).sum())
        pairs = []
        if n_ap and n_lat:
            pairs, _ua, _ul = T.form_pairs(ap_pk, lat_pk, entry["ap_geo"], entry["lat_geo"], si_tol)
        selected = {(int(x), int(y)) for x, y in pairs}
        # iterate the BROAD graph (<= GMAX); classify + store every edge; count deployed + sweep
        ii, jj = np.nonzero(dsi <= GMAX); pos_this = 0
        for i, j in zip(ii.tolist(), jj.tolist()):
            dvox = float(dsi[i, j]); aset, lset = ap_p2i[i], lat_p2i[j]; shared = aset & lset
            if aset and lset:
                cname = "positive_capable" if shared else "cross_instance_only"
            else:
                cname = "one_sided" if (aset or lset) else "fully_false"
            ident_amb = (len(shared) > 1) or (len(aset) > 1) or (len(lset) > 1)
            for g in GRID:
                if dvox <= g: sweep_edges[g] += 1; sweep_cls[g][cname] += 1
            if dvox <= si_tol:
                cls_count[cname] += 1
                if cname == "positive_capable": pos_this += 1
                if ident_amb: identity_ambiguous_pairs += 1
            row = len(P["case_global_idx"])
            P["case_global_idx"].append(ci); P["case_val_idx"].append(c); P["case_id"].append(cid)
            P["ap_idx"].append(i); P["lat_idx"].append(j)
            P["ap_row"].append(float(ap_pk[i, 0])); P["ap_col"].append(float(ap_pk[i, 1]))
            P["lat_row"].append(float(lat_pk[j, 0])); P["lat_col"].append(float(lat_pk[j, 1]))
            P["ap_score"].append(float(ap_pk[i, 2])); P["lat_score"].append(float(lat_pk[j, 2]))
            P["dsi_vox"].append(dvox); P["dsi_mm"].append(float(dvox * smm) if smm else np.nan)
            P["cls"].append(CLS.index(cname)); P["n_shared_iids"].append(len(shared))
            P["shared_iid"].append(next(iter(shared)) if len(shared) == 1 else -1)
            P["ap_single_iid"].append(next(iter(aset)) if len(aset) == 1 else -1)
            P["lat_single_iid"].append(next(iter(lset)) if len(lset) == 1 else -1)
            P["identity_ambiguous"].append(bool(ident_amb))
            P["ap_is_dup"].append(bool(ap_dup[i])); P["lat_is_dup"].append(bool(lat_dup[j]))
            P["selected_by_form_pairs"].append((i, j) in selected)
            if ident_amb:
                amb_sidecar[row] = {"ap_iids": sorted(int(x) for x in aset), "lat_iids": sorted(int(x) for x in lset)}
        positives_per_case.append(pos_this)

        # EXISTENTIAL correct-pair availability (per fracture) + competitor sweep
        for iid in {int(x["iid"]) for x in gt}:
            aps = iid_ap.get(iid, set()); lats = iid_lat.get(iid, set())
            if aps: exist_ap += 1
            if lats: exist_lat += 1
            if not (aps and lats): continue
            exist_both += 1
            mdc = min(dsi[aa, ll] for aa in aps for ll in lats); dsi_correct_min.append(float(mdc))
            if mdc <= si_tol: correct_exists_after_gate += 1
            else: removed_by_gate += 1
            sel_compat = sel_unique = False
            for aa, ll in product(aps, lats):
                if (aa, ll) not in selected: continue
                shared_sel = ap_p2i[aa] & lat_p2i[ll]
                if iid not in shared_sel: continue
                sel_compat = True
                if shared_sel == {iid}: sel_unique = True
            if sel_compat: correct_selected_compatible += 1
            if sel_unique: correct_selected_unique += 1
            if sel_compat and not sel_unique: correct_selected_ambiguous += 1
            if mdc <= si_tol and not sel_compat: lost_in_assignment += 1
            # competitors at the DEPLOYED gate (headline) + at every grid gate (sweep)
            comp_a = sum(1 for aa in aps for ll in range(n_lat) if dsi[aa, ll] <= si_tol and iid not in lat_p2i[ll])
            comp_l = sum(1 for ll in lats for aa in range(n_ap) if dsi[aa, ll] <= si_tol and iid not in ap_p2i[aa])
            ap_comp.append(comp_a); lat_comp.append(comp_l)
            if mdc <= si_tol and (comp_a > 0 or comp_l > 0): n_ambiguous += 1
            for g in GRID:
                ca = sum(1 for aa in aps for ll in range(n_lat) if dsi[aa, ll] <= g and iid not in lat_p2i[ll])
                cl = sum(1 for ll in lats for aa in range(n_ap) if dsi[aa, ll] <= g and iid not in ap_p2i[aa])
                sweep_comp[g] += ca + cl

        # SECONDARY one-to-one continuity audit (Stage B/C convention; NOT authoritative for existence)
        ap_m = match([p[:2] for p in ap_pk], [x["ap_foot"] for x in gt], RADIUS)
        lat_m = match([p[:2] for p in lat_pk], [x["lat_foot"] for x in gt], RADIUS)
        ap_by_gt = {gi: pi for pi, gi, _ in ap_m}; lat_by_gt = {gi: pi for pi, gi, _ in lat_m}
        for gi in set(ap_by_gt) & set(lat_by_gt):
            s1_dual += 1; astar, lstar = ap_by_gt[gi], lat_by_gt[gi]
            if dsi[astar, lstar] <= si_tol: s1_in_graph += 1
            if (astar, lstar) in selected: s1_selected += 1

    total_deployed = sum(cls_count.values()); broad_total = len(P["case_global_idx"])
    npz_path = a.out.with_name(a.out.stem + "_pairs.npz")
    np.savez_compressed(npz_path,
        case_global_idx=np.asarray(P["case_global_idx"], np.int32), case_val_idx=np.asarray(P["case_val_idx"], np.int32),
        case_id=np.asarray(P["case_id"]), ap_idx=np.asarray(P["ap_idx"], np.int32), lat_idx=np.asarray(P["lat_idx"], np.int32),
        ap_row=np.asarray(P["ap_row"], np.float32), ap_col=np.asarray(P["ap_col"], np.float32),
        lat_row=np.asarray(P["lat_row"], np.float32), lat_col=np.asarray(P["lat_col"], np.float32),
        ap_score=np.asarray(P["ap_score"], np.float32), lat_score=np.asarray(P["lat_score"], np.float32),
        dsi_vox=np.asarray(P["dsi_vox"], np.float32), dsi_mm=np.asarray(P["dsi_mm"], np.float32),
        cls=np.asarray(P["cls"], np.int8), n_shared_iids=np.asarray(P["n_shared_iids"], np.int16),
        shared_iid=np.asarray(P["shared_iid"], np.int32), ap_single_iid=np.asarray(P["ap_single_iid"], np.int32),
        lat_single_iid=np.asarray(P["lat_single_iid"], np.int32), identity_ambiguous=np.asarray(P["identity_ambiguous"], np.bool_),
        ap_is_dup=np.asarray(P["ap_is_dup"], np.bool_), lat_is_dup=np.asarray(P["lat_is_dup"], np.bool_),
        selected=np.asarray(P["selected_by_form_pairs"], np.bool_),
        class_names=np.asarray(CLS), all_case_ids=np.asarray(cases), val_case_ids=np.asarray(used_val),
        all_ap_geo=np.asarray(d["ap_geo"]), all_lat_geo=np.asarray(d["lat_geo"]),
        si_tol=np.float32(si_tol), audit_gate_max=np.float32(GMAX),
        ap_nms=np.int32(pol["ap"][0]), ap_floor=np.float32(pol["ap"][1]),
        lat_nms=np.int32(pol["lat"][0]), lat_floor=np.float32(pol["lat"][1]),
        data_sha256=np.asarray(data_sha), detector_run=np.asarray(str(a.detector_run)),
        # detector checkpoint provenance so a downstream stage can prove the graph came from THIS checkpoint
        detector_ap_sha256=np.asarray(rec["detector_sha256"]["ap"]),
        detector_lat_sha256=np.asarray(rec["detector_sha256"]["lat"]),
        detector_record_sha256=np.asarray(T.sha256_file(a.detector_run / "detector_dev_run.json")))
    if amb_sidecar:
        amb_path = a.out.with_name(a.out.stem + "_ambiguous_pairs.json"); amb_path.write_text(json.dumps(amb_sidecar))
    else:
        amb_path = None

    dep_rows = [k for k in range(broad_total) if P["dsi_vox"][k] <= si_tol]

    def cls_feature(code):   # DEPLOYED-gate separability (D1 can recompute at any gate from the NPZ)
        idx = [k for k in dep_rows if P["cls"][k] == code]
        return {"ap_score": stat([P["ap_score"][k] for k in idx]), "lat_score": stat([P["lat_score"][k] for k in idx]),
                "min_score": stat([min(P["ap_score"][k], P["lat_score"][k]) for k in idx]),
                "max_score_fused": stat([max(P["ap_score"][k], P["lat_score"][k]) for k in idx]),
                "prod_score": stat([P["ap_score"][k] * P["lat_score"][k] for k in idx]),
                "dsi_vox": stat([P["dsi_vox"][k] for k in idx]),
                "dsi_mm": stat([P["dsi_mm"][k] for k in idx]) if want_mm else None,
                "frac_selected_by_form_pairs": ratio(sum(P["selected_by_form_pairs"][k] for k in idx), len(idx))}

    dcm = np.asarray(dsi_correct_min, np.float64); base6 = int((dcm <= si_tol).sum())
    gate_sensitivity = []
    for g in GRID:
        rn = int((dcm <= g).sum()); comp = sweep_cls[g]; edges = sweep_edges[g]
        gate_sensitivity.append({"gate_vox": g, "correct_edge_recall_fractures": rn,
            "correct_edge_recall_frac_all_gt": ratio(rn, gt_total), "recovered_vs_si_tol": rn - base6,
            "gated_edges": edges, "composition": dict(comp), "positive_prevalence": ratio(comp["positive_capable"], edges),
            "mean_competitors_per_dualview_fracture": round(sweep_comp[g] / exist_both, 2) if exist_both else None,
            "pre_assignment_pairability_ceiling_frac_all_gt": ratio(rn, gt_total)})

    result = {
        "audit": "D0 — candidate-pair data audit + BROAD pair graph (frozen detector, det-validation split, many-to-one)",
        "diagnostic_status": "development audit on the detector-validation split (biased; sealed test first confirmatory). "
                             "Extraction-floor peaks; SI in voxel units; mm via canonical NIfTI header when --image-dirs given. "
                             "NPZ contains the BROAD graph (<= audit_gate_max) for the D1 gate sweep; JSON headline is at the "
                             "deployed si_tol. Governing objective is 3D reconstruction candidate generation.",
        "detector_run": str(a.detector_run), "data_sha256": data_sha, "val_cases": len(used_val),
        "extraction_policy": {"ap_nms": pol["ap"][0], "ap_floor": pol["ap"][1],
                              "lat_nms": pol["lat"][0], "lat_floor": pol["lat"][1],
                              "is_deployed": deployed_pol},
        "si_tol_voxels": si_tol, "audit_gate_max_voxels": GMAX, "match_radius_px": RADIUS,
        "pair_dataset_npz": str(npz_path), "pair_dataset_ambiguous_sidecar": (str(amb_path) if amb_path else None),
        "pair_graph": {
            "potential_pairs_before_si_gating": potential_before, "pairs_after_deployed_si_gating": after_gate,
            "pairs_in_broad_graph_npz": broad_total, "si_gating_retention_frac": ratio(after_gate, potential_before),
            "mean_ap_peaks_per_case": round(float(np.mean(ap_pc)), 1), "mean_lat_peaks_per_case": round(float(np.mean(lat_pc)), 1),
            "identity_ambiguous_gated_pairs_deployed": identity_ambiguous_pairs},
        "graph_class_counts_deployed": {**cls_count, "total": total_deployed},
        "graph_class_fraction_deployed": {k: ratio(cls_count[k], total_deployed) for k in CLS},
        "positives_capable_per_case_deployed": stat(positives_per_case),
        "per_class_features_deployed": {k: cls_feature(CLS.index(k)) for k in CLS},
        "correct_pair_availability_existential": {
            "gt_fractures": gt_total, "with_compatible_ap_peak": exist_ap, "with_compatible_lat_peak": exist_lat,
            "compatible_ap_recall": ratio(exist_ap, gt_total), "compatible_lat_recall": ratio(exist_lat, gt_total),
            "detection_loss_no_ap": gt_total - exist_ap, "detection_loss_no_lat": gt_total - exist_lat,
            "existential_dual_view": exist_both, "existential_dual_view_frac": ratio(exist_both, gt_total),
            "correct_pair_exists_after_gate": correct_exists_after_gate,
            "correct_pair_exists_after_gate_frac_of_dualview": ratio(correct_exists_after_gate, exist_both),
            "removed_by_si_gating": removed_by_gate, "removed_by_si_gating_frac_of_dualview": ratio(removed_by_gate, exist_both),
            "fractures_with_frozen_selected_compatible_pair": correct_selected_compatible,
            "fractures_with_frozen_selected_unique_pair": correct_selected_unique,
            "fractures_with_only_ambiguous_frozen_selection": correct_selected_ambiguous,
            "unique_selected_frac_of_dualview": ratio(correct_selected_unique, exist_both),
            "compatible_selected_frac_of_dualview": ratio(correct_selected_compatible, exist_both),
            "lost_in_one_to_one_assignment": lost_in_assignment,
            "current_graph_pre_assignment_pairability_ceiling_all_gt": ratio(correct_exists_after_gate, gt_total),
            "note": "EXISTENTIAL (many-to-one): removed_by_si_gating = compatible peaks in BOTH views but NO same-iid "
                    "combo within the deployed gate -> UPSTREAM loss no scorer recovers (redesign the graph; see the "
                    "gate_sensitivity_sweep). lost_in_one_to_one_assignment = a same-iid combo IS in-graph but frozen "
                    "form_pairs selected no compatible pair -> DOWNSTREAM, fixable by a better scorer/assignment. "
                    "SELECTION split: unique (exactly one iid = actionable, AUTHORITATIVE) vs ambiguous (overlapping "
                    "iids = uncertainty). pre_assignment_pairability_ceiling = fraction of ALL GT for which a same-"
                    "fracture edge merely EXISTS on this gate; an UPPER BOUND before assignment, abstention, and "
                    "reconstruction localization error (D1d turns it into actual accepted 3D recall within 5/10/15 mm). "
                    "Widening the gate (sweep) or improving lateral recall raises it."},
        "assignment_ambiguity_existential_deployed": {
            "n_dualview_with_competitor_within_gate": n_ambiguous, "frac_dualview_ambiguous": ratio(n_ambiguous, exist_both),
            "ap_side_competitors": stat(ap_comp), "lat_side_competitors": stat(lat_comp),
            "min_dsi_correct_combination_vox": stat(dsi_correct_min)},
        "gate_sensitivity_sweep": {
            "grid_vox": GRID,
            "note": "per gate: existential correct-edge recall (pre-assignment pairability ceiling, an UPPER BOUND before "
                    "assignment/abstention/localization), recovered vs deployed si_tol, broad-graph edges + class composition + positive prevalence, and mean "
                    "competitors per dual-view fracture. Widening recovers true edges (Loss 2) but grows the negative-edge "
                    "load and competitors -> judge each gate jointly, not by recall alone. Achievable accepted-3D recall / "
                    "phantoms-per-case require assignment + reconstruction and are D1c/D1d, not D0.",
            "rows": gate_sensitivity},
        "secondary_one_to_one_continuity_audit": {
            "note": "Stage B/C one-to-one Hungarian identity; continuity ONLY, NOT authoritative for existence.",
            "dual_view_gt_one_to_one": s1_dual, "correct_pair_in_graph_one_to_one": s1_in_graph,
            "correct_pair_selected_one_to_one": s1_selected},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    pg = result["pair_graph"]; cc = result["graph_class_counts_deployed"]; cpa = result["correct_pair_availability_existential"]
    amb = result["assignment_ambiguity_existential_deployed"]; pcf = result["per_class_features_deployed"]
    print(f"\nD0 PAIR AUDIT — {len(used_val)} val cases | deployed si_tol {si_tol} vox | broad graph <= {GMAX} vox | MANY-TO-ONE")
    print(f"  graph: deployed {pg['pairs_after_deployed_si_gating']} / broad-npz {pg['pairs_in_broad_graph_npz']} "
          f"of {pg['potential_pairs_before_si_gating']} potential | peaks/case AP {pg['mean_ap_peaks_per_case']} lat {pg['mean_lat_peaks_per_case']}")
    print(f"  deployed classes: pos-capable {cc['positive_capable']} | cross-only {cc['cross_instance_only']} | "
          f"one-sided {cc['one_sided']} | fully-false {cc['fully_false']}")
    pv = pcf["positive_capable"]["dsi_vox"]; cv = pcf["cross_instance_only"]["dsi_vox"]
    print(f"  |dSI| vox median: pos {pv['median'] if pv else None} | cross {cv['median'] if cv else None} (overlap => SI can't separate real-vs-real)")
    pmin = pcf["positive_capable"]["min_score"]; ffmin = pcf["fully_false"]["min_score"]
    print(f"  min-score median: pos {pmin['median'] if pmin else None} | fully-false {ffmin['median'] if ffmin else None}")
    print(f"  compatible recall (many-to-one): AP {cpa['compatible_ap_recall']} | lat {cpa['compatible_lat_recall']} "
          f"(vs Stage-B one-to-one AP 0.746 / lat 0.516) | no-AP {cpa['detection_loss_no_ap']} no-lat {cpa['detection_loss_no_lat']}")
    print(f"  EXISTENTIAL: gt {cpa['gt_fractures']} | dual-view {cpa['existential_dual_view']} ({cpa['existential_dual_view_frac']}) "
          f"| correct-in-graph {cpa['correct_pair_exists_after_gate']} ({cpa['correct_pair_exists_after_gate_frac_of_dualview']}) "
          f"| removed-by-gate {cpa['removed_by_si_gating']} | frac-unique-sel {cpa['fractures_with_frozen_selected_unique_pair']} "
          f"| lost-in-assign {cpa['lost_in_one_to_one_assignment']}")
    print(f"  current-graph pre-assignment pairability ceiling: {cpa['current_graph_pre_assignment_pairability_ceiling_all_gt']} of all GT "
          f"(UPPER BOUND; pre assignment/abstention/localization)")
    print("  gate sweep (vox: recall_frac / recovered / edges / pos-prevalence / mean-competitors):")
    for r in gate_sensitivity:
        print(f"    {int(r['gate_vox']):>3}: {r['correct_edge_recall_frac_all_gt']} / +{r['recovered_vs_si_tol']} "
              f"/ {r['gated_edges']} / {r['positive_prevalence']} / {r['mean_competitors_per_dualview_fracture']}")
    print(f"wrote {a.out} and {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
