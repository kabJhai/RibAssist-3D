#!/usr/bin/env python3
"""STAGE D1 — DETERMINISTIC correspondence: one-to-one assignment WITH EXPLICIT ABSTENTION over a gate sweep,
selected by OUT-OF-FOLD case-level configuration selection on the OPERATIONAL 3D-RECONSTRUCTION frontier.

Consumes the self-contained D0 broad pair graph (every AP x lateral edge with |dSI| <= audit_gate_max, plus
coordinates + per-case projection geometry + the ambiguous-identity sidecar). Every accepted pair is
back-projected from the NPZ coordinates via back_project (NO detector rerun).

TWO SCOREBOARDS (the deployed output is a 3D POINT, not a GT id):
  * OPERATIONAL 3D scoreboard (PRIMARY; drives selection). Back-project EVERY accepted edge regardless of
    correspondence class, compute its distance to every GT fracture VOLUME in the case, and run an INDEPENDENT
    case-level one-to-one prediction<->GT Hungarian matching AT EACH of 5 / 10 / 15 mm (each the maximum valid
    matching at that tolerance). The primary endpoint and the false-3D-point budget are at 10 mm (predictions
    unmatched in the 10 mm assignment are FALSE 3D points, incl. displaced duplicates and 10-15 mm points);
    15 mm is a secondary coarse endpoint; unmatched GT are MISSED. Hidden GT correspondence identity is NOT
    used to decide which fracture a point satisfies — geometry is.
  * CORRESPONDENCE-conditioned scoreboard (DIAGNOSTIC only). The IID-labelled view: correct-iid /
    cross-instance / one-sided / fully-false / ambiguous, and localization of correctly-corresponding pairs.

Method:
  * ABSTENTION IS IN THE ASSIGNMENT. Each case's bipartite matrix is augmented with dummy unmatched nodes at
    per-node cost u (AP->dummy = u, dummy->lateral = u, dummy->dummy = 0). Matching a real AP-lateral edge
    REPLACES an AP->dummy plus a dummy->lateral (two unmatched penalties), so a real edge is favoured only
    when roughly c_ij < 2u; u is a PER-NODE unmatched cost, not a direct edge-rejection threshold. The matrix
    is re-solved per u. Mutual-best is a SEPARATE graph restricted BEFORE assignment (independent optimum).
  * OUT-OF-FOLD SELECTION (not nested CV — there is no inner split): for each outer fold, the full
    configuration (gate, cost, mutual-best, u) is selected on the outer-TRAIN cases by the operational
    objective and applied ONCE to the held-out outer-TEST cases; the headline concatenates only outer-test
    outputs. Full-data frontiers are reported separately and labelled explicitly optimistic.
  * PREDECLARED per-cost u-grid (no held-out leakage). Feasible/fallback two-stage config selection.
  * ONE CREDIT PER FRACTURE by GEOMETRY: case-level one-to-one matching already credits <=1 prediction per
    GT (the globally assigned prediction under minimum-distance one-to-one matching); other accepted points
    near it fall out as false.

Ceilings per gate (UPPER bounds, pre assignment): OPERATIONAL geometric candidate ceilings at 5/10/15 mm
(match ALL candidate points to GT geometrically with the SAME helper as the realized metric — aligned with
the primary endpoint; within10 is used for the D1-vs-D2 gap) and, for DIAGNOSIS ONLY, correspondence-identity
ceilings (existential-compatible incl. ambiguous via sidecar | unique-identity | reconstructable-unique-within-tol).

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory). The
4 GT-negative cases cannot validate negative-scan safety; their false-point counts are DESCRIPTIVE.
Provenance fails closed: data hash == sha256(--data) == detector det_dev_sha256; NPZ records its detector run
(+ checkpoint SHAs when present).

Usage:
  python eval_correspondence_D1_assign.py \
      --pairs-npz outputs/correspondence_D0_broad_pairs.npz \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/correspondence_D1_assign.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from nibabel.affines import apply_affine
    import train_detector as T
    from eval_address_e2e import build_instance_records
    from eval_biplanar_geometry import back_project, case_gt, fracture_metrics
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

GRID_CANDIDATES = (6, 8, 10, 12, 16, 20, 30)
COSTS = ("si", "min_conf", "prod_conf", "geomean_conf", "si_norm_invconf", "lat_weighted")
K_FOLDS = 5
CORRECT_MM = 10.0     # primary credit tolerance (recall also reported at 5 and 15)
MATCH_TOL = 15.0      # a prediction can only MATCH a GT within this distance (else it is a false 3D point)
SCORE_BOUND = 30.0    # precompute prediction->GT distances up to here (all matching happens well within)


def edge_cost(name, aps, lts, dsi, gate):
    minc = np.minimum(aps, lts); prod = np.clip(aps * lts, 0, None)
    if name == "si": return dsi.astype(np.float64)
    if name == "min_conf": return 1.0 - minc
    if name == "prod_conf": return 1.0 - prod
    if name == "geomean_conf": return 1.0 - np.sqrt(prod)
    if name == "si_norm_invconf": return dsi / max(gate, 1e-6) + (1.0 - minc)
    if name == "lat_weighted": return 1.0 - np.cbrt(np.clip(aps * lts * lts, 0, None))
    raise ValueError(name)


def u_grid(cost, gate):
    hi = gate if cost == "si" else (2.0 if cost == "si_norm_invconf" else 1.0)
    return [float(x) for x in np.linspace(0.0, hi, 13)]


def assign_abstain(na, nl, cost_of, u):
    if na == 0 or nl == 0: return []
    BIG = 1e9; N = na + nl; M = np.full((N, N), BIG)
    for (al, ll), (row, cost) in cost_of.items(): M[al, ll] = cost
    for al in range(na): M[al, nl + al] = u
    for ll in range(nl): M[na + ll, ll] = u
    M[na:, nl:] = 0.0
    ri, ci = linear_sum_assignment(M)
    return [cost_of[(i, j)][0] for i, j in zip(ri, ci) if i < na and j < nl and M[i, j] < BIG]


def ratio(a, b):
    return round(a / b, 4) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-npz", type=Path, required=True); ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--ambiguous-sidecar", type=Path, default=None)
    ap.add_argument("--false-3d-cap-at10", "--phantom-cap", dest="false_cap", type=float, default=1.0,
                    help="max FALSE 3D points at 10mm per case when selecting the configuration (alias: --phantom-cap)")
    ap.add_argument("--freeze-policy-out", type=Path, default=None,
                    help="DEV mode: select ONE deployment config on ALL val cases (no folds) at --false-3d-cap-at10, "
                         "write it as a frozen-policy JSON, and exit. Use to freeze the config BEFORE the sealed run.")
    ap.add_argument("--apply-policy", type=Path, default=None,
                    help="SEALED mode: apply a frozen-policy JSON (gate,cost,mutual_best,u) to ALL cases with NO "
                         "reselection; emit metrics + case-level bootstrap + per-case counts.")
    ap.add_argument("--expected-data-sha256", default=None,
                    help="SEALED mode: bind to this external data digest instead of the detector's dev data hash.")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel scikit-learn", file=sys.stderr); return 1
    if a.freeze_policy_out is None and a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    if not np.isfinite(a.false_cap) or a.false_cap < 0: raise ValueError("--false-3d-cap-at10 must be finite and non-negative")

    rec = json.loads((a.detector_run / "detector_dev_run.json").read_text())
    det_sha = rec.get("det_dev_sha256")
    if not det_sha: raise ValueError("detector_dev_run.json missing det_dev_sha256")
    data_sha = T.sha256_file(a.data)
    sealed = a.expected_data_sha256 is not None
    if sealed:
        if data_sha != a.expected_data_sha256:
            raise ValueError(f"--data sha256 {data_sha[:12]}.. != --expected-data-sha256 {a.expected_data_sha256[:12]}..")
    elif data_sha != det_sha:
        raise ValueError("--data hash != detector det_dev_sha256")
    z = np.load(a.pairs_npz, allow_pickle=False)
    if str(z["data_sha256"]) != data_sha: raise ValueError("pairs NPZ data hash != --data")
    if Path(str(z["detector_run"])).resolve() != a.detector_run.resolve():
        raise ValueError(f"pairs NPZ detector run {str(z['detector_run'])} != --detector-run {a.detector_run}")
    prov_level = "detector_run_path"
    if "detector_ap_sha256" in z.files:
        for v, key in (("ap", "detector_ap_sha256"), ("lat", "detector_lat_sha256")):
            if T.sha256_file(a.detector_run / f"detector_{v}.pt") != str(z[key]):
                raise ValueError(f"detector_{v}.pt hash != NPZ record")
        prov_level = "detector_checkpoint_sha256"
    GMAX = float(z["audit_gate_max"]); si_tol = float(z["si_tol"])
    GRID = sorted({float(g) for g in GRID_CANDIDATES if g <= GMAX} | {si_tol, GMAX})

    d = np.load(a.data, allow_pickle=False)
    recs = build_instance_records(d); case_to_globalidx = {str(c): i for i, c in enumerate(d["case"])}
    val_ids = [str(c) for c in z["val_case_ids"]]
    gt_per_case = {c: len(recs.get(case_to_globalidx[c], [])) for c in val_ids}
    gt_total = sum(gt_per_case.values()); neg_set = {c for c in val_ids if gt_per_case[c] == 0}
    all_ap_geo = z["all_ap_geo"]; all_lat_geo = z["all_lat_geo"]

    cgi = z["case_global_idx"]; cid_arr = np.array([str(x) for x in z["case_id"]]); apx = z["ap_idx"]; ltx = z["lat_idx"]
    ap_row = z["ap_row"]; ap_col = z["ap_col"]; lat_row = z["lat_row"]; lat_col = z["lat_col"]
    ap_s = z["ap_score"].astype(np.float64); lt_s = z["lat_score"].astype(np.float64)
    dsi = z["dsi_vox"].astype(np.float64); cls = z["cls"]; shared_iid = z["shared_iid"]; ident_amb = z["identity_ambiguous"]
    class_names = [str(x) for x in z["class_names"]]; POS = class_names.index("positive_capable")
    n_edge = len(cgi)
    sp = a.ambiguous_sidecar
    if sp is None:
        stem = a.pairs_npz.stem; base = stem[:-6] if stem.endswith("_pairs") else stem
        cand = a.pairs_npz.with_name(base + "_ambiguous_pairs.json"); sp = cand if cand.exists() else None
    sidecar = json.loads(Path(sp).read_text()) if sp and Path(sp).exists() else {}
    print(f"D1: {n_edge} edges, {len(val_ids)} val cases (gt {gt_total}, neg {len(neg_set)}). gates {GRID}, "
          f"costs {list(COSTS)}, out-of-fold {K_FOLDS}-fold selection, cap {a.false_cap}, provenance {prov_level}.", flush=True)

    # ---- PRECOMPUTE per edge: 3D point -> distance to EVERY GT, with anatomical metrics cached per (edge, GT)
    #      for the GT assigned during operational matching; and the
    #      correspondence-labelled reconstruction (vs shared_iid). Load case_gt ONCE per case. ----
    geo_dist = [None] * n_edge          # dict {iid: mm} for GTs within SCORE_BOUND
    geo_metrics = [None] * n_edge       # row -> {iid: (rib_exact, rib_within1, along_mm)} for GTs within MATCH_TOL
    corr_iid = np.full(n_edge, -1, np.int64); corr_mm = np.full(n_edge, np.inf, np.float64)
    corr_rib = np.zeros(n_edge, bool); corr_w1 = np.zeros(n_edge, bool); corr_along = np.full(n_edge, np.nan, np.float64)
    gts_by_case = {}
    rows_by_case = {}
    for r in range(n_edge): rows_by_case.setdefault(cid_arr[r], []).append(r)
    for k, (cid, rows) in enumerate(rows_by_case.items()):
        if k % 10 == 0: print(f"  reconstruct+score case {k+1}/{len(rows_by_case)} ...", flush=True)
        g = case_gt(cid, a.image_dirs, a.seg_dir, a.cl_dir)
        if g is None: gts_by_case[cid] = []; continue
        aff = g["aff"]; iids = [int(i) for i in g["fl_groups"].keys()]; gts_by_case[cid] = iids
        trees = {iid: cKDTree(apply_affine(aff, g["fl_groups"][iid].T.astype(np.float64))) for iid in iids}
        gci = int(cgi[rows[0]]); ap_geo = all_ap_geo[gci]; lat_geo = all_lat_geo[gci]
        prec = {}; siw = {}; pts = np.empty((len(rows), 3))
        for idx, r in enumerate(rows):
            p_rec, si_dis = back_project((float(ap_row[r]), float(ap_col[r])), (float(lat_row[r]), float(lat_col[r])), ap_geo, lat_geo)
            prec[r] = p_rec; siw[r] = si_dis; pts[idx] = apply_affine(aff, p_rec)
        for iid in iids:                                   # nearest fracture-voxel distance via KD-tree (low peak memory)
            dmin, _ = trees[iid].query(pts, k=1, workers=-1)
            for idx, r in enumerate(rows):
                dd = float(dmin[idx])
                if dd <= SCORE_BOUND:
                    if geo_dist[r] is None: geo_dist[r] = {}
                    geo_dist[r][iid] = dd
        for r in rows:
            dm = geo_dist[r] or {}
            for iid, dd in dm.items():                     # anatomical cached PER (row, iid) within MATCH_TOL (assigned-GT accurate)
                if dd <= MATCH_TOL:
                    m, _ = fracture_metrics(prec[r], siw[r], g["fl_groups"][iid], g, cid, iid)
                    if m:
                        if geo_metrics[r] is None: geo_metrics[r] = {}
                        geo_metrics[r][iid] = (bool(m["rib_exact"]), bool(m["rib_within1"]), float(m["along_mm"]))
            if cls[r] == POS and int(shared_iid[r]) >= 0:  # correspondence-labelled recon (diagnostic)
                iid = int(shared_iid[r]); vox = g["fl_groups"].get(iid)
                if vox is not None and vox.shape[1] > 0:
                    mc, _ = fracture_metrics(prec[r], siw[r], vox, g, cid, iid)
                    if mc: corr_iid[r] = iid; corr_mm[r] = mc["mask_center_mm"]; corr_rib[r] = mc["rib_exact"]; corr_w1[r] = mc["rib_within1"]; corr_along[r] = mc["along_mm"]

    def match_predictions(preds, gts, tol):
        """Independent case-level one-to-one prediction<->GT Hungarian using ONLY distances <= tol (shared by
        tally_geo and op_ceiling so a realized metric and its ceiling have identical matching semantics)."""
        if not preds or not gts: return []
        # BIG dominates every possible SUM of valid distances (each <= tol), so linear_sum_assignment first
        # MAXIMIZES the number of valid <=tol matches, then minimizes their total distance.
        BIG = 1e9; assert BIG > tol * (min(len(preds), len(gts)) + 1)
        M = np.full((len(preds), len(gts)), BIG)
        for pi, r in enumerate(preds):
            dm = geo_dist[r] or {}
            for gi, iid in enumerate(gts):
                if iid in dm and dm[iid] <= tol: M[pi, gi] = dm[iid]
        ri, ci = linear_sum_assignment(M)
        return [(preds[pi], gts[gi], float(M[pi, gi])) for pi, gi in zip(ri, ci) if M[pi, gi] <= tol]

    def tally_geo(accepted, cohort):
        """OPERATIONAL: SEPARATE geometric matching at 5/10/15 mm (each the maximum valid matching at that
        tolerance). Primary endpoint + false-point budget are at 10 mm; 15 mm is a secondary coarse endpoint."""
        cohort = list(cohort); gt_c = sum(gt_per_case[c] for c in cohort); nca = len(cohort)
        by_case = {}
        for row, _ in accepted: by_case.setdefault(cid_arr[row], []).append(row)
        m5 = []; m10 = []; m15 = []; false10 = 0; false15 = 0
        neg_false10 = 0; neg_raw10 = {c: 0 for c in cohort if c in neg_set}
        for cid, preds in by_case.items():
            gts = gts_by_case.get(cid, [])
            a5 = match_predictions(preds, gts, 5.0); a10 = match_predictions(preds, gts, CORRECT_MM); a15 = match_predictions(preds, gts, MATCH_TOL)
            assert len(a5) <= len(a10) <= len(a15)   # nested graphs -> monotone max matching cardinality
            m5 += a5; m10 += a10; m15 += a15
            false10 += len(preds) - len(a10); false15 += len(preds) - len(a15)
            if cid in neg_set: neg_false10 += len(preds); neg_raw10[cid] = neg_raw10.get(cid, 0) + len(preds)
        c5 = len({(cid_arr[r], iid) for r, iid, _ in m5}); c10 = len({(cid_arr[r], iid) for r, iid, _ in m10})
        c15 = len({(cid_arr[r], iid) for r, iid, _ in m15})
        assert c5 <= c10 <= c15 and false10 >= false15   # aggregate tolerance monotonicity
        within10 = [(r, iid) for r, iid, _ in m10]; dists10 = np.array([dd for _, _, dd in m10], np.float64)
        rx = [geo_metrics[r][iid][0] for (r, iid) in within10 if geo_metrics[r] and iid in geo_metrics[r]]
        rw = [geo_metrics[r][iid][1] for (r, iid) in within10 if geo_metrics[r] and iid in geo_metrics[r]]
        al = [geo_metrics[r][iid][2] for (r, iid) in within10 if geo_metrics[r] and iid in geo_metrics[r]]
        return {"cohort_gt": gt_c, "cohort_cases": nca, "cohort_neg_cases": len([c for c in cohort if c in neg_set]),
                "correct5": c5, "correct10": c10, "correct15": c15,
                "recall5": ratio(c5, gt_c), "recall10": ratio(c10, gt_c), "recall15": ratio(c15, gt_c),
                "additional_matches_at15_vs10": c15 - c10, "fraction_gt_missed_within10": ratio(gt_c - c10, gt_c),
                "false_3d_points_at_10mm": false10, "false_3d_per_case_at_10mm": round(false10 / nca, 3) if nca else None,
                "false_3d_points_at_15mm": false15, "false_3d_per_case_at_15mm": round(false15 / nca, 3) if nca else None,
                "neg_case_false_points_at10_total": neg_false10, "neg_case_false_points_at10_raw": neg_raw10,
                "median_mm_matched_within10": round(float(np.median(dists10)), 2) if dists10.size else None,
                "p90_mm_matched_within10": round(float(np.percentile(dists10, 90)), 2) if dists10.size else None,
                "rib_exact_matched_within10": round(float(np.mean(rx)), 4) if rx else None,
                "rib_within1_matched_within10": round(float(np.mean(rw)), 4) if rw else None,
                "along_rib_mm_matched_within10": round(float(np.mean(al)), 2) if al else None,
                "n_accepted_3d": len(accepted), "n_matched_within10": c10,
                "anatomical_note": "independent Hungarian matching per tolerance; anatomical from the 10mm-ASSIGNED GT"}

    def tally_corr(accepted, cohort):
        """DIAGNOSTIC: correspondence-labelled view (credit MINIMUM GEOMETRIC ERROR per corr-iid, correspondence
        cost as tie-break; classes from cls)."""
        cohort = set(cohort); acc = [(r, c) for r, c in accepted if cid_arr[r] in cohort]
        by_gt = {}; cross = one = full = amb = 0
        for r, c in acc:
            if corr_iid[r] >= 0: by_gt.setdefault((cid_arr[r], int(corr_iid[r])), []).append((r, c))
            else:
                cn = class_names[cls[r]]
                if cn == "positive_capable": amb += 1     # positive-capable but not uniquely identified
                elif cn == "cross_instance_only": cross += 1
                elif cn == "one_sided": one += 1
                else: full += 1
        credited = [min(v, key=lambda x: (corr_mm[x[0]], x[1]))[0] for v in by_gt.values()]
        cm = np.array([corr_mm[r] for r in credited], np.float64)
        c10 = int((cm <= CORRECT_MM).sum()) if cm.size else 0
        return {"n_correct_iid_credited": len(credited), "correct_iid_within10": c10,
                "cross_instance_accepted": cross, "one_sided_accepted": one, "fully_false_accepted": full,
                "ambiguous_identity_accepted": amb,
                "rib_exact_correct_iid_within10": round(float(np.mean([corr_rib[r] for r in credited if corr_mm[r] <= CORRECT_MM])), 4)
                    if any(corr_mm[r] <= CORRECT_MM for r in credited) else None}

    # ---- ceilings ----
    def op_ceiling(gate):
        """OPERATIONAL geometric candidate ceiling at 5/10/15 mm: ALL edges at this gate as candidate 3D points,
        case-level prediction<->GT matching (SAME helper as tally_geo) at each tolerance, distinct matched GT /
        all GT. Ignores AP/lateral peak reuse and the one-to-one peak constraint -> an intentionally OPTIMISTIC
        upper bound over candidate points BEFORE correspondence assignment/abstention. within10 is the field for
        the D1-vs-D2 gap."""
        gm = dsi <= gate; by_case = {}
        for r in np.nonzero(gm)[0]: by_case.setdefault(cid_arr[r], []).append(int(r))
        mg = {5.0: set(), CORRECT_MM: set(), MATCH_TOL: set()}
        for cid, preds in by_case.items():
            gts = gts_by_case.get(cid, [])
            for tol in mg:
                for r, iid, _ in match_predictions(preds, gts, tol): mg[tol].add((cid, iid))
        return {"within5": ratio(len(mg[5.0]), gt_total), "within10": ratio(len(mg[CORRECT_MM]), gt_total),
                "within15": ratio(len(mg[MATCH_TOL]), gt_total)}

    def corr_ceilings(gate):
        gm = dsi <= gate
        uniq = {(int(cgi[r]), int(shared_iid[r])) for r in np.nonzero(gm & (cls == POS) & (shared_iid >= 0))[0]}
        recon = {(int(cgi[r]), int(corr_iid[r])) for r in np.nonzero(gm & (cls == POS))[0] if corr_iid[r] >= 0 and corr_mm[r] <= CORRECT_MM}
        exist = set(uniq)
        for r in np.nonzero(gm & (cls == POS) & ident_amb)[0]:
            sc = sidecar.get(str(int(r)))
            if sc:
                for iid in set(sc["ap_iids"]) & set(sc["lat_iids"]): exist.add((int(cgi[r]), int(iid)))
        return {"existential_compatible_frac_all_gt": ratio(len(exist), gt_total),
                "unique_identity_frac_all_gt": ratio(len(uniq), gt_total),
                "reconstructable_unique_within_tol_frac_all_gt": ratio(len(recon), gt_total)}

    # ---- precompute accepted rows per (gate,cost,mb,u,case) ----
    fold_of = {c: i % K_FOLDS for i, c in enumerate(sorted(val_ids))}
    ACC = {}
    for gate in GRID:
        gmask = dsi <= gate
        for cost in COSTS:
            costs_all = edge_cost(cost, ap_s, lt_s, dsi, gate); ug = u_grid(cost, gate)
            for cid, rows in rows_by_case.items():
                rr = [r for r in rows if gmask[r]]
                if not rr:
                    for mb in (False, True):
                        for u in ug: ACC[(gate, cost, mb, u, cid)] = []
                    continue
                aps_u = sorted({int(apx[r]) for r in rr}); lts_u = sorted({int(ltx[r]) for r in rr})
                ai = {v: i for i, v in enumerate(aps_u)}; lj = {v: i for i, v in enumerate(lts_u)}
                na, nl = len(aps_u), len(lts_u); BIG = 1e9; Mreal = np.full((na, nl), BIG); cost_of = {}
                for r in rr:
                    i, j = ai[int(apx[r])], lj[int(ltx[r])]
                    if costs_all[r] < Mreal[i, j]: Mreal[i, j] = costs_all[r]; cost_of[(i, j)] = (r, float(costs_all[r]))
                rowmin = Mreal.min(1); colmin = Mreal.min(0)
                mb_edge = {(i, j): (Mreal[i, j] <= rowmin[i] + 1e-12 and Mreal[i, j] <= colmin[j] + 1e-12) for (i, j) in cost_of}
                cost_of_mb = {k: v for k, v in cost_of.items() if mb_edge[k]}
                for u in ug:
                    ACC[(gate, cost, False, u, cid)] = [(r, float(costs_all[r])) for r in assign_abstain(na, nl, cost_of, u)]
                    ACC[(gate, cost, True, u, cid)] = [(r, float(costs_all[r])) for r in assign_abstain(na, nl, cost_of_mb, u)]

    # ---- full-data frontiers (OPTIMISTIC) ----
    frontiers = {}
    for gate in GRID:
        for cost in COSTS:
            for mb in (False, True):
                fr = []
                for u in u_grid(cost, gate):
                    t = tally_geo([p for c in val_ids for p in ACC[(gate, cost, mb, u, c)]], val_ids)
                    fr.append({"u": round(u, 4), "recall10": t["recall10"], "false_3d_per_case_at_10mm": t["false_3d_per_case_at_10mm"],
                               "false_3d_per_case_at_15mm": t["false_3d_per_case_at_15mm"], "recall5": t["recall5"], "recall15": t["recall15"]})
                frontiers[f"gate{int(gate)}:{cost}{'+mb' if mb else ''}"] = fr

    # ---- OUT-OF-FOLD configuration selection, swept over a range of false-3D@10 caps ----
    # Cache each fold's TRAIN tally per configuration ONCE; the cap sweep is then pure re-selection.
    CONFIGS = [(gate, cost, mb, u) for gate in GRID for cost in COSTS for mb in (False, True) for u in u_grid(cost, gate)]
    train_tally = {}
    for f in range(K_FOLDS):
        train = [c for c in val_ids if fold_of[c] != f]
        for cfg in CONFIGS:
            g_, c_, mb_, u_ = cfg
            train_tally[(f, cfg)] = tally_geo([p for cc in train for p in ACC[(g_, c_, mb_, u_, cc)]], train)

    def select(cands, cap):
        feasible = [(cfg, t) for cfg, t in cands if t["false_3d_per_case_at_10mm"] is not None and t["false_3d_per_case_at_10mm"] <= cap]
        if feasible:   # within cap: maximize 10mm recall, then fewer 10mm false points, then fewer emitted
            cfg, t = max(feasible, key=lambda x: (x[1]["correct10"], -x[1]["false_3d_points_at_10mm"], -x[1]["n_accepted_3d"])); return cfg, t, True
        cfg, t = min(cands, key=lambda x: (x[1]["false_3d_per_case_at_10mm"] if x[1]["false_3d_per_case_at_10mm"] is not None else 1e18, -x[1]["correct10"])); return cfg, t, False

    # ================= FROZEN-POLICY PATHS (freeze on dev, apply on sealed; NO out-of-fold reselection) =========
    def pooled(cfg, cohort):
        g_, c_, mb_, u_ = cfg; return [p for cc in cohort for p in ACC[(g_, c_, mb_, u_, cc)]]

    if a.freeze_policy_out is not None:
        # select ONE deployment config on ALL val cases (no folds) at the cap — the exact grid u is preserved.
        full_cands = [(cfg, tally_geo(pooled(cfg, val_ids), val_ids)) for cfg in CONFIGS]
        (g_, c_, mb_, u_), t, feas = select(full_cands, float(a.false_cap))
        pol = {"gate": float(g_), "cost": c_, "mutual_best": bool(mb_), "u": float(u_),
               "false_3d_cap_per_case_at_10mm": float(a.false_cap),
               "selection": "single config on ALL dev val cases (no folds) at the cap",
               "feasible_within_cap": bool(feas), "dev_recall10": t["recall10"],
               "dev_false_3d_per_case_at_10mm": t["false_3d_per_case_at_10mm"], "dev_correct10": t["correct10"],
               "detector_run": str(a.detector_run), "dev_data_sha256": data_sha, "pairs_npz": str(a.pairs_npz),
               "gates_grid": GRID, "costs_grid": list(COSTS)}
        a.freeze_policy_out.parent.mkdir(parents=True, exist_ok=True)
        a.freeze_policy_out.write_text(json.dumps(pol, indent=2))
        print(f"\nFROZEN DEPLOYMENT POLICY (all-dev selection @ cap {a.false_cap}): gate {int(g_)} {c_}"
              f"{'+mb' if mb_ else ''} u={u_:.6f}  (feasible {feas}; dev recall10 {t['recall10']} @ "
              f"{t['false_3d_per_case_at_10mm']}/case)\nwrote {a.freeze_policy_out}")
        return 0

    if a.apply_policy is not None:
        pj = json.loads(a.apply_policy.read_text())
        g_ = float(pj["gate"]); c_ = str(pj["cost"]); mb_ = bool(pj["mutual_best"]); u_want = float(pj["u"])
        if g_ not in GRID: raise ValueError(f"policy gate {g_} not in graph GRID {GRID}")
        if c_ not in COSTS: raise ValueError(f"policy cost {c_} not in {list(COSTS)}")
        ug = list(u_grid(c_, g_)); u_ = min(ug, key=lambda x: abs(x - u_want))
        if abs(u_ - u_want) > 1e-4: raise ValueError(f"policy u={u_want} not on grid {ug}")
        cfg = (g_, c_, mb_, u_)
        accepted = pooled(cfg, val_ids)
        head = tally_geo(accepted, val_ids); hcorr = tally_corr(accepted, val_ids)
        # per-case counts (same matching semantics as tally_geo)
        by_case = {}
        for row, _ in accepted: by_case.setdefault(cid_arr[row], []).append(row)
        per_case = {}
        for cid in val_ids:
            preds = by_case.get(cid, []); gts = gts_by_case.get(cid, [])
            a5 = match_predictions(preds, gts, 5.0); a10 = match_predictions(preds, gts, CORRECT_MM); a15 = match_predictions(preds, gts, MATCH_TOL)
            m10 = len({(cid, iid) for _, iid, _ in a10})
            per_case[cid] = {"gt": gt_per_case[cid], "n_pred": len(preds),
                             "matched5": len({(cid, iid) for _, iid, _ in a5}), "matched10": m10,
                             "matched15": len({(cid, iid) for _, iid, _ in a15}), "false10": len(preds) - len(a10),
                             "is_negative": cid in neg_set}
        # case-level bootstrap (seeded) of recall@{5,10,15} and false/case at 10mm
        rng = np.random.RandomState(0); B = 2000; cohort = list(val_ids); n = len(cohort)
        bs = {"recall5": [], "recall10": [], "recall15": [], "false_3d_per_case_at_10mm": []}
        for _ in range(B):
            samp = [cohort[i] for i in rng.randint(0, n, n)]
            gt = sum(per_case[c]["gt"] for c in samp)
            bs["recall5"].append(sum(per_case[c]["matched5"] for c in samp) / gt if gt else 0.0)
            bs["recall10"].append(sum(per_case[c]["matched10"] for c in samp) / gt if gt else 0.0)
            bs["recall15"].append(sum(per_case[c]["matched15"] for c in samp) / gt if gt else 0.0)
            bs["false_3d_per_case_at_10mm"].append(sum(per_case[c]["false10"] for c in samp) / n)
        ci95 = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in bs.items()}
        out = {
            "stage": "SEALED fixed-policy 3D reconstruction (frozen config; NO reselection)",
            "diagnostic_status": "SEALED confirmatory pass — frozen extraction policy + frozen D1 config selected on dev.",
            "mode": "apply-policy-sealed", "expected_data_sha256": a.expected_data_sha256,
            "detector_run": str(a.detector_run), "data_sha256": data_sha, "pairs_npz": str(a.pairs_npz),
            "provenance_level": prov_level, "frozen_policy": {"gate": g_, "cost": c_, "mutual_best": mb_, "u": u_},
            "val_cases": len(val_ids), "gt_fractures": gt_total, "gt_negative_cases": len(neg_set),
            "correct_within_mm": CORRECT_MM, "match_tol_mm": MATCH_TOL,
            "operational_headline": head, "bootstrap95_case_level": ci95,
            "correspondence_diagnostic": hcorr,
            "candidate_ceiling_at_policy_gate": op_ceiling(g_),
            "gate_ceilings": {str(g): op_ceiling(g) for g in GRID},
            "per_case": per_case,
        }
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=2))
        h = head
        print(f"\nSEALED FIXED-POLICY HEADLINE — gate {int(g_)} {c_}{'+mb' if mb_ else ''} u={u_:.4f} | {len(val_ids)} cases, gt {gt_total}")
        print(f"  recall 5/10/15mm: {h['recall5']}/{h['recall10']}/{h['recall15']} | matched@10 {h['n_matched_within10']} "
              f"| false@10 {h['false_3d_per_case_at_10mm']}/case (total {h['false_3d_points_at_10mm']})")
        print(f"  bootstrap95 recall10 {ci95['recall10']} | false@10/case {ci95['false_3d_per_case_at_10mm']}")
        print(f"  cases with >=1 correct@10: {sum(1 for c in val_ids if per_case[c]['matched10']>0)}/{len(val_ids)} | "
              f"median/p90 mm {h['median_mm_matched_within10']}/{h['p90_mm_matched_within10']} | rib-exact {h['rib_exact_matched_within10']}")
        print(f"  candidate ceiling@10 (gate {int(g_)}): {op_ceiling(g_)['within10']}")
        print(f"wrote {a.out}")
        return 0
    # ================= END frozen-policy paths; default = out-of-fold dev evaluation ==========================

    caps = sorted({0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float(a.false_cap)}) + [float("inf")]
    cap_sweep = []; headline = None; headline_corr = None; fold_configs = None
    for cap in caps:
        oof = []; fcfgs = []
        for f in range(K_FOLDS):
            test = [c for c in val_ids if fold_of[c] == f]
            cands = [(cfg, train_tally[(f, cfg)]) for cfg in CONFIGS]
            cfg, t, feas = select(cands, cap); g_, c_, mb_, u_ = cfg
            oof += [p for cc in test for p in ACC[(g_, c_, mb_, u_, cc)]]
            fcfgs.append({"fold": f, "gate": g_, "cost": c_, "mutual_best": mb_, "u": round(u_, 4),
                          "false3d_cap_feasible_on_training_fold": feas, "train_recall10": t["recall10"],
                          "train_false_3d_per_case_at_10mm": t["false_3d_per_case_at_10mm"]})
        ht = tally_geo(oof, val_ids)
        cap_sweep.append({"false_3d_cap_per_case_at_10mm": (None if cap == float("inf") else cap),
                          "recall5": ht["recall5"], "recall10": ht["recall10"], "recall15": ht["recall15"],
                          "realized_false_3d_per_case_at_10mm": ht["false_3d_per_case_at_10mm"], "n_matched_within10": ht["n_matched_within10"]})
        if cap == float(a.false_cap): headline = ht; headline_corr = tally_corr(oof, val_ids); fold_configs = fcfgs

    out = {
        "stage": "D1 — deterministic assignment with abstention; OPERATIONAL 3D scoreboard (out-of-fold selection)",
        "diagnostic_status": "development on the detector-validation split (biased; sealed test first confirmatory). "
                             "PRIMARY = operational geometric prediction<->GT matching (out-of-fold). Correspondence view "
                             "is diagnostic. Full-data frontiers are explicitly OPTIMISTIC. Negative-scan safety NOT "
                             "validated (4 negative cases); false-point counts descriptive.",
        "pairs_npz": str(a.pairs_npz), "data_sha256": data_sha, "detector_run": str(a.detector_run), "provenance_level": prov_level,
        "val_cases": len(val_ids), "gt_fractures": gt_total, "gt_negative_cases": len(neg_set),
        "correct_within_mm": CORRECT_MM, "match_tol_mm": MATCH_TOL, "false_3d_cap_per_case_at_10mm": a.false_cap, "k_folds": K_FOLDS,
        "gates": GRID, "costs": list(COSTS),
        "operational_headline_out_of_fold": headline,
        "correspondence_diagnostic_out_of_fold": headline_corr,
        "operational_cap_sweep_out_of_fold": cap_sweep,
        "out_of_fold_fold_configs": fold_configs,
        "gate_ceilings": {str(g): {"operational_geometric_candidate_ceiling": op_ceiling(g),
                                   "operational_ceiling_note": "upper bound over ALL candidate 3D points at this gate "
                                       "(case-level geometric pred<->GT match at 5/10/15mm) BEFORE correspondence "
                                       "assignment, abstention, and peak-level one-to-one constraints; intentionally "
                                       "optimistic. within10 is the field for the D1-vs-D2 gap.",
                                   "correspondence_identity_ceilings": corr_ceilings(g)} for g in GRID},
        "operational_full_data_frontiers_optimistic": frontiers,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=2))

    h = headline; hc = headline_corr
    print(f"\nD1 OPERATIONAL HEADLINE (out-of-fold; geometric pred<->GT match; correct<= {CORRECT_MM}mm, false@10-cap {a.false_cap}/case)")
    print(f"  recall 5/10/15mm: {h['recall5']}/{h['recall10']}/{h['recall15']} | matched@10 {h['n_matched_within10']} (additional@15-vs10 {h['additional_matches_at15_vs10']}) of {gt_total} GT")
    print(f"  false 3D/case @10 {h['false_3d_per_case_at_10mm']} (total {h['false_3d_points_at_10mm']}) | @15 {h['false_3d_per_case_at_15mm']} | neg-case false@10 {h['neg_case_false_points_at10_total']} (descriptive)")
    print(f"  median/p90 mm within10 {h['median_mm_matched_within10']}/{h['p90_mm_matched_within10']} | rib-exact {h['rib_exact_matched_within10']} rib±1 {h['rib_within1_matched_within10']} | along mm {h['along_rib_mm_matched_within10']}")
    print(f"  [corr diagnostic] correct-iid credited {hc['n_correct_iid_credited']} (within10 {hc['correct_iid_within10']}) | "
          f"cross {hc['cross_instance_accepted']} one-sided {hc['one_sided_accepted']} fully-false {hc['fully_false_accepted']} ambiguous {hc['ambiguous_identity_accepted']}")
    print(f"  out-of-fold CAP SWEEP (false@10 cap -> realized recall@10 / recall@5 / recall@15 @ realized false@10/case):")
    for cs in cap_sweep:
        capname = "none" if cs["false_3d_cap_per_case_at_10mm"] is None else cs["false_3d_cap_per_case_at_10mm"]
        print(f"    cap {str(capname):>5}: recall10 {cs['recall10']} (5 {cs['recall5']} / 15 {cs['recall15']}) @ false {cs['realized_false_3d_per_case_at_10mm']}/case | matched {cs['n_matched_within10']}")
    print(f"  per-fold config:")
    for fc in fold_configs:
        feas = "" if fc["false3d_cap_feasible_on_training_fold"] else "  [CAP INFEASIBLE->fallback]"
        print(f"    fold {fc['fold']}: gate {int(fc['gate'])} {fc['cost']}{'+mb' if fc['mutual_best'] else ''} u={fc['u']}{feas}")
    print(f"  gate ceilings [operational geometric candidate 5/10/15mm | correspondence-identity existential/unique/reconstructable]:")
    for g in GRID:
        cc = out["gate_ceilings"][str(g)]; oc = cc["operational_geometric_candidate_ceiling"]; ci = cc["correspondence_identity_ceilings"]
        print(f"    {int(g):>3}: op {oc['within5']}/{oc['within10']}/{oc['within15']} | "
              f"id {ci['existential_compatible_frac_all_gt']}/{ci['unique_identity_frac_all_gt']}/{ci['reconstructable_unique_within_tol_frac_all_gt']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
