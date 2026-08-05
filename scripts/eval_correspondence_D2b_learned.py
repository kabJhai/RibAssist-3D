#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""STAGE D2b — LEARNED cross-view pair-compatibility scorer, evaluated through D1's FROZEN operational
3D-reconstruction scoreboard. Replaces ONLY D1's deterministic edge cost with a learned one; the candidate
graph (D0), assignment-with-abstention, independent 5/10/15 mm geometric matching, false-3D@10 budget,
out-of-fold case-level selection, cap sweep, and operational candidate ceilings are all UNCHANGED (the
assignment + ratio helpers are imported from D1 verbatim; the geometric scoreboard is the same code).

Model: a lightweight two-tower CNN (shared encoders by default, given only ~180 positives) — encode the AP
crop and the lateral crop, combine [ap_emb, lat_emb, ap_emb*lat_emb] with scalar covariates (|dSI|, AP/lat
confidence, min/prod/asymmetry, duplicate flags), one compatibility logit. Appearance is the principal new
signal; scalars are auxiliary so D2 tests whether appearance ADDS value beyond D1's SI+confidence.

Training discipline (severe imbalance: 180 pos / 286 cross / 7292 one-sided / 34715 fully-false):
  * CASE-level folds only (from D2a); never split edges of one case across train/test.
  * Per epoch: ALL positives, ALL cross-instance HARD negatives, a controlled REFRESHED sample of one-sided
    and fully-false negatives. BCE with a positive-class weight. Strong regularization (dropout, weight decay).
  * Nested honestly: per OUTER fold, train on outer-train cases, score EVERY edge with that model, select the
    (gate, mutual-best, unmatched-cost u) configuration on outer-train by the operational objective, apply
    ONCE to outer-test. The reported frontier concatenates only outer-test.
  * Evaluation is on the COMPLETE unsampled candidate graph. Pair-separation AUROC (pos-vs-cross / -one-sided
    / -fully-false) is DIAGNOSTIC ONLY and never selects the model.

The decisive output is the out-of-fold recall@10 vs false-3D@10 cap sweep, compared to D1's, and the
conversion of the operational candidate ceiling@10 into realized recall.

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory).
Provenance fails closed across D2a crops, D0 graph, --data, and the detector run.

Usage:
  python eval_correspondence_D2b_learned.py \
      --crops-npz outputs/correspondence_D2a_crops.npz \
      --pairs-npz outputs/correspondence_D0_broad_pairs.npz \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/correspondence_D2b_learned.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch, torch.nn as nn
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from nibabel.affines import apply_affine
    import train_detector as T
    from eval_address_e2e import build_instance_records
    from eval_biplanar_geometry import back_project, case_gt, fracture_metrics
    from eval_correspondence_D1_assign import assign_abstain, ratio   # EXACT D1 assignment + helper
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

GRID_CANDIDATES = (6, 8, 10, 12, 16, 20, 30)
K_FOLDS = 5
CORRECT_MM = 10.0; MATCH_TOL = 15.0; SCORE_BOUND = 30.0
N_U = 13
SCALAR_KEYS = ("dsi_norm", "ap_score", "lat_score", "conf_min", "conf_prod", "conf_asym")


def u_grid_learned():
    return [float(x) for x in np.linspace(0.0, 1.0, N_U)]   # learned cost = 1 - P(same) in [0,1]


class Tower(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(32, emb)

    def forward(self, x): return self.fc(self.net(x).flatten(1))


class Scorer(nn.Module):
    def __init__(self, n_scalar, emb=32, shared=True, p=0.4):
        super().__init__()
        self.ap = Tower(emb); self.lat = self.ap if shared else Tower(emb)
        self.head = nn.Sequential(nn.Linear(emb * 3 + n_scalar, 64), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, ap, lat, sc):
        ea = self.ap(ap); el = self.lat(lat)
        return self.head(torch.cat([ea, el, ea * el, sc], 1)).squeeze(1)


def norm_crop(x):
    m = x.mean(); s = x.std()
    return (x - m) / (s + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-npz", type=Path, required=True); ap.add_argument("--pairs-npz", type=Path, required=True)
    ap.add_argument("--detector-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--false-3d-cap-at10", "--phantom-cap", dest="false_cap", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--neg-sample", type=int, default=3000)
    ap.add_argument("--view-specific", action="store_true", help="separate AP/lat encoders (default: shared)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    if not np.isfinite(a.false_cap) or a.false_cap < 0: raise ValueError("--false-3d-cap-at10 must be finite and non-negative")
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = T.device()

    # ---- FAIL-CLOSED provenance across D2a crops, D0 graph, --data, detector ----
    rec = json.loads((a.detector_run / "detector_dev_run.json").read_text())
    det_sha = rec.get("det_dev_sha256")
    if not det_sha: raise ValueError("detector_dev_run.json missing det_dev_sha256")
    data_sha = T.sha256_file(a.data)
    if data_sha != det_sha: raise ValueError("--data hash != detector det_dev_sha256")
    z = np.load(a.pairs_npz, allow_pickle=False)
    if str(z["data_sha256"]) != data_sha: raise ValueError("pairs NPZ data hash != --data")
    cz = np.load(a.crops_npz, allow_pickle=False)
    if str(cz["data_sha256"]) != data_sha: raise ValueError("crops NPZ data hash != --data")
    if int(len(cz["label"])) != int(len(z["dsi_vox"])): raise ValueError("crops/pairs NPZ row count mismatch (not aligned)")

    GMAX = float(z["audit_gate_max"]); si_tol = float(z["si_tol"])
    GRID = sorted({float(g) for g in GRID_CANDIDATES if g <= GMAX} | {si_tol, GMAX})
    d = np.load(a.data, allow_pickle=False)
    recs = build_instance_records(d); case_to_globalidx = {str(c): i for i, c in enumerate(d["case"])}
    val_ids = [str(c) for c in z["val_case_ids"]]
    gt_per_case = {c: len(recs.get(case_to_globalidx[c], [])) for c in val_ids}
    gt_total = sum(gt_per_case.values()); neg_set = {c for c in val_ids if gt_per_case[c] == 0}
    all_ap_geo = z["all_ap_geo"]; all_lat_geo = z["all_lat_geo"]

    # edge columns (D0) + appearance (D2a), row-aligned
    cgi = z["case_global_idx"]; cid_arr = np.array([str(x) for x in z["case_id"]]); apx = z["ap_idx"]; ltx = z["lat_idx"]
    ap_row = z["ap_row"]; ap_col = z["ap_col"]; lat_row = z["lat_row"]; lat_col = z["lat_col"]
    dsi = z["dsi_vox"].astype(np.float64); n_edge = len(cgi)
    ap_crops = cz["ap_crops"].astype(np.float32); lat_crops = cz["lat_crops"].astype(np.float32)
    e_apb = cz["edge_ap_bank_idx"]; e_latb = cz["edge_lat_bank_idx"]; label = cz["label"]; fold = cz["fold"]
    # scalar covariates (normalized dsi to [0,1] by broad-gate max)
    scal = np.stack([dsi / GMAX, cz["ap_score"], cz["lat_score"], cz["conf_min"], cz["conf_prod"], cz["conf_asym"]], 1).astype(np.float32)
    print(f"D2b: {n_edge} edges, {len(val_ids)} cases (gt {gt_total}). crop {ap_crops.shape[-1]}px, "
          f"pos {(label==0).sum()} cross {(label==1).sum()} one {(label==2).sum()} full {(label==3).sum()}. "
          f"{'view-specific' if a.view_specific else 'shared'} towers, {a.epochs} epochs, neg-sample {a.neg_sample}/class.", flush=True)

    # ================= geometric scoreboard (SAME semantics as D1) =================
    geo_dist = [None] * n_edge; geo_metrics = [None] * n_edge; gts_by_case = {}
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
        for iid in iids:
            dmin, _ = trees[iid].query(pts, k=1, workers=-1)
            for idx, r in enumerate(rows):
                dd = float(dmin[idx])
                if dd <= SCORE_BOUND:
                    if geo_dist[r] is None: geo_dist[r] = {}
                    geo_dist[r][iid] = dd
        for r in rows:
            for iid, dd in (geo_dist[r] or {}).items():
                if dd <= MATCH_TOL:
                    m, _ = fracture_metrics(prec[r], siw[r], g["fl_groups"][iid], g, cid, iid)
                    if m:
                        if geo_metrics[r] is None: geo_metrics[r] = {}
                        geo_metrics[r][iid] = (bool(m["rib_exact"]), bool(m["rib_within1"]), float(m["along_mm"]))

    def match_predictions(preds, gts, tol):
        if not preds or not gts: return []
        BIG = 1e9; assert BIG > tol * (min(len(preds), len(gts)) + 1)
        M = np.full((len(preds), len(gts)), BIG)
        for pi, r in enumerate(preds):
            dm = geo_dist[r] or {}
            for gi, iid in enumerate(gts):
                if iid in dm and dm[iid] <= tol: M[pi, gi] = dm[iid]
        ri, ci = linear_sum_assignment(M)
        return [(preds[pi], gts[gi], float(M[pi, gi])) for pi, gi in zip(ri, ci) if M[pi, gi] <= tol]

    def tally_geo(accepted, cohort):
        cohort = list(cohort); gt_c = sum(gt_per_case[c] for c in cohort); nca = len(cohort)
        by_case = {}
        for row, _ in accepted: by_case.setdefault(cid_arr[row], []).append(row)
        m5 = []; m10 = []; m15 = []; false10 = 0; false15 = 0; neg_false10 = 0; neg_raw10 = {c: 0 for c in cohort if c in neg_set}
        for cid, preds in by_case.items():
            gts = gts_by_case.get(cid, [])
            a5 = match_predictions(preds, gts, 5.0); a10 = match_predictions(preds, gts, CORRECT_MM); a15 = match_predictions(preds, gts, MATCH_TOL)
            assert len(a5) <= len(a10) <= len(a15)
            m5 += a5; m10 += a10; m15 += a15
            false10 += len(preds) - len(a10); false15 += len(preds) - len(a15)
            if cid in neg_set: neg_false10 += len(preds); neg_raw10[cid] = neg_raw10.get(cid, 0) + len(preds)
        c5 = len({(cid_arr[r], iid) for r, iid, _ in m5}); c10 = len({(cid_arr[r], iid) for r, iid, _ in m10}); c15 = len({(cid_arr[r], iid) for r, iid, _ in m15})
        assert c5 <= c10 <= c15 and false10 >= false15
        within10 = [(r, iid) for r, iid, _ in m10]; dists10 = np.array([dd for _, _, dd in m10], np.float64)
        rx = [geo_metrics[r][iid][0] for (r, iid) in within10 if geo_metrics[r] and iid in geo_metrics[r]]
        rw = [geo_metrics[r][iid][1] for (r, iid) in within10 if geo_metrics[r] and iid in geo_metrics[r]]
        return {"cohort_gt": gt_c, "cohort_cases": nca, "correct5": c5, "correct10": c10, "correct15": c15,
                "recall5": ratio(c5, gt_c), "recall10": ratio(c10, gt_c), "recall15": ratio(c15, gt_c),
                "additional_matches_at15_vs10": c15 - c10,
                "false_3d_points_at_10mm": false10, "false_3d_per_case_at_10mm": round(false10 / nca, 3) if nca else None,
                "false_3d_points_at_15mm": false15, "false_3d_per_case_at_15mm": round(false15 / nca, 3) if nca else None,
                "neg_case_false_points_at10_total": neg_false10, "neg_case_false_points_at10_raw": neg_raw10,
                "median_mm_matched_within10": round(float(np.median(dists10)), 2) if dists10.size else None,
                "p90_mm_matched_within10": round(float(np.percentile(dists10, 90)), 2) if dists10.size else None,
                "rib_exact_matched_within10": round(float(np.mean(rx)), 4) if rx else None,
                "rib_within1_matched_within10": round(float(np.mean(rw)), 4) if rw else None,
                "n_accepted_3d": len(accepted), "n_matched_within10": c10}

    def op_ceiling(gate):
        gm = dsi <= gate; by_case = {}
        for r in np.nonzero(gm)[0]: by_case.setdefault(cid_arr[r], []).append(int(r))
        mg = {5.0: set(), CORRECT_MM: set(), MATCH_TOL: set()}
        for cid, preds in by_case.items():
            gts = gts_by_case.get(cid, [])
            for tol in mg:
                for r, iid, _ in match_predictions(preds, gts, tol): mg[tol].add((cid, iid))
        return {"within5": ratio(len(mg[5.0]), gt_total), "within10": ratio(len(mg[CORRECT_MM]), gt_total), "within15": ratio(len(mg[MATCH_TOL]), gt_total)}

    # ================= learned scorer: train per outer fold, score all edges =================
    ap_t = torch.from_numpy(ap_crops)[:, None].to(dev); lat_t = torch.from_numpy(lat_crops)[:, None].to(dev)
    ap_t = torch.stack([norm_crop(ap_t[i]) for i in range(ap_t.shape[0])]) if ap_t.shape[0] else ap_t
    lat_t = torch.stack([norm_crop(lat_t[i]) for i in range(lat_t.shape[0])]) if lat_t.shape[0] else lat_t
    scal_t = torch.from_numpy(scal).to(dev)

    def predict(model):
        model.eval(); out = np.empty(n_edge, np.float32)
        with torch.no_grad():
            for s in range(0, n_edge, 4096):
                idx = np.arange(s, min(s + 4096, n_edge))
                a_ = ap_t[e_apb[idx]]; l_ = lat_t[e_latb[idx]]; sc_ = scal_t[torch.from_numpy(idx).to(dev)]
                out[idx] = torch.sigmoid(model(a_, l_, sc_)).cpu().numpy()
        return out

    def train_fold(train_cases):
        tc = set(train_cases)
        pos = np.array([r for r in range(n_edge) if cid_arr[r] in tc and label[r] == 0])
        cross = np.array([r for r in range(n_edge) if cid_arr[r] in tc and label[r] == 1])
        one = np.array([r for r in range(n_edge) if cid_arr[r] in tc and label[r] == 2])
        full = np.array([r for r in range(n_edge) if cid_arr[r] in tc and label[r] == 3])
        model = Scorer(len(SCALAR_KEYS), shared=not a.view_specific).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        pos_w = torch.tensor(8.0, device=dev)   # upweight the rare positive class
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        rng = np.random.RandomState(a.seed)
        for ep in range(a.epochs):
            model.train()
            neg_o = one[rng.randint(0, len(one), min(a.neg_sample, len(one)))] if len(one) else np.array([], int)
            neg_f = full[rng.randint(0, len(full), min(a.neg_sample, len(full)))] if len(full) else np.array([], int)
            rows = np.concatenate([pos, cross, neg_o, neg_f]).astype(int)
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(cross) + len(neg_o) + len(neg_f))]).astype(np.float32)
            perm = rng.permutation(len(rows)); rows = rows[perm]; y = y[perm]
            for s in range(0, len(rows), 256):
                b = rows[s:s + 256]; yb = torch.from_numpy(y[s:s + 256]).to(dev)
                a_ = ap_t[e_apb[b]]; l_ = lat_t[e_latb[b]]; sc_ = scal_t[torch.from_numpy(b).to(dev)]
                opt.zero_grad(); loss = bce(model(a_, l_, sc_), yb); loss.backward(); opt.step()
        return model

    def select(cands, cap):
        feasible = [(cfg, t) for cfg, t in cands if t["false_3d_per_case_at_10mm"] is not None and t["false_3d_per_case_at_10mm"] <= cap]
        if feasible:
            cfg, t = max(feasible, key=lambda x: (x[1]["correct10"], -x[1]["false_3d_points_at_10mm"], -x[1]["n_accepted_3d"])); return cfg, t, True
        cfg, t = min(cands, key=lambda x: (x[1]["false_3d_per_case_at_10mm"] if x[1]["false_3d_per_case_at_10mm"] is not None else 1e18, -x[1]["correct10"])); return cfg, t, False

    CONFIGS = [(gate, mb, u) for gate in GRID for mb in (False, True) for u in u_grid_learned()]
    fold_of = {c: i % K_FOLDS for i, c in enumerate(sorted(val_ids))}
    fold_ACC = []; fold_scores = []
    for f in range(K_FOLDS):
        train = [c for c in val_ids if fold_of[c] != f]
        print(f"  training fold {f} on {len(train)} cases ...", flush=True)
        model = train_fold(train); sc = predict(model); lcost = 1.0 - sc; fold_scores.append(sc)
        ACC = {}
        for gate in GRID:
            gmask = dsi <= gate
            for cid, rows in rows_by_case.items():
                rr = [r for r in rows if gmask[r]]
                if not rr:
                    for mb in (False, True):
                        for u in u_grid_learned(): ACC[(gate, mb, u, cid)] = []
                    continue
                aps_u = sorted({int(apx[r]) for r in rr}); lts_u = sorted({int(ltx[r]) for r in rr})
                ai = {v: i for i, v in enumerate(aps_u)}; lj = {v: i for i, v in enumerate(lts_u)}
                na, nl = len(aps_u), len(lts_u); BIG = 1e9; Mreal = np.full((na, nl), BIG); cost_of = {}
                for r in rr:
                    i, j = ai[int(apx[r])], lj[int(ltx[r])]
                    if lcost[r] < Mreal[i, j]: Mreal[i, j] = lcost[r]; cost_of[(i, j)] = (r, float(lcost[r]))
                rowmin = Mreal.min(1); colmin = Mreal.min(0)
                mb_edge = {(i, j): (Mreal[i, j] <= rowmin[i] + 1e-12 and Mreal[i, j] <= colmin[j] + 1e-12) for (i, j) in cost_of}
                cost_of_mb = {k: v for k, v in cost_of.items() if mb_edge[k]}
                for u in u_grid_learned():
                    ACC[(gate, False, u, cid)] = [(r, float(lcost[r])) for r in assign_abstain(na, nl, cost_of, u)]
                    ACC[(gate, True, u, cid)] = [(r, float(lcost[r])) for r in assign_abstain(na, nl, cost_of_mb, u)]
        fold_ACC.append(ACC)

    # per-fold train tallies (cache) then cap sweep (out-of-fold)
    train_tally = [{} for _ in range(K_FOLDS)]
    for f in range(K_FOLDS):
        train = [c for c in val_ids if fold_of[c] != f]
        for cfg in CONFIGS:
            gate, mb, u = cfg
            train_tally[f][cfg] = tally_geo([p for c in train for p in fold_ACC[f][(gate, mb, u, c)]], train)
    caps = sorted({0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float(a.false_cap)}) + [float("inf")]
    cap_sweep = []; headline = None; fold_configs = None
    for cap in caps:
        oof = []; fcfgs = []
        for f in range(K_FOLDS):
            test = [c for c in val_ids if fold_of[c] == f]
            cfg, t, feas = select([(cfg, train_tally[f][cfg]) for cfg in CONFIGS], cap)
            gate, mb, u = cfg; oof += [p for c in test for p in fold_ACC[f][(gate, mb, u, c)]]
            fcfgs.append({"fold": f, "gate": gate, "mutual_best": mb, "u": round(u, 4), "feasible": feas,
                          "train_recall10": t["recall10"], "train_false_3d_per_case_at_10mm": t["false_3d_per_case_at_10mm"]})
        ht = tally_geo(oof, val_ids)
        cap_sweep.append({"false_3d_cap_per_case_at_10mm": (None if cap == float("inf") else cap),
                          "recall5": ht["recall5"], "recall10": ht["recall10"], "recall15": ht["recall15"],
                          "realized_false_3d_per_case_at_10mm": ht["false_3d_per_case_at_10mm"], "n_matched_within10": ht["n_matched_within10"]})
        if cap == float(a.false_cap): headline = ht; fold_configs = fcfgs

    # ---- DIAGNOSTIC pair-separation AUROC (out-of-fold scores), never used for selection ----
    def auroc(pos_scores, neg_scores):
        if len(pos_scores) == 0 or len(neg_scores) == 0: return None
        allv = np.concatenate([pos_scores, neg_scores]); order = allv.argsort()
        ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv) + 1)
        rp = ranks[:len(pos_scores)].sum(); n1 = len(pos_scores); n2 = len(neg_scores)
        return round(float((rp - n1 * (n1 + 1) / 2) / (n1 * n2)), 4)
    oof_score = np.empty(n_edge, np.float32)
    for f in range(K_FOLDS):
        test_mask = np.array([fold_of[cid_arr[r]] == f for r in range(n_edge)])
        oof_score[test_mask] = fold_scores[f][test_mask]
    def sc_of(lab): return oof_score[label == lab]
    diag = {"auroc_pos_vs_cross": auroc(sc_of(0), sc_of(1)), "auroc_pos_vs_one_sided": auroc(sc_of(0), sc_of(2)),
            "auroc_pos_vs_fully_false": auroc(sc_of(0), sc_of(3)),
            "note": "DIAGNOSTIC ONLY (out-of-fold scores); never used to select the model. pos-vs-cross is the "
                    "scientifically decisive separation (both peaks are real fractures)."}

    out = {
        "stage": "D2b — learned pair-compatibility through D1's frozen operational scoreboard",
        "diagnostic_status": "development on the detector-validation split (biased; sealed test first confirmatory). "
                             "Learned cost replaces ONLY D1's deterministic cost; scoreboard identical. AUROC diagnostic only.",
        "crops_npz": str(a.crops_npz), "pairs_npz": str(a.pairs_npz), "data_sha256": data_sha, "detector_run": str(a.detector_run),
        "val_cases": len(val_ids), "gt_fractures": gt_total, "correct_within_mm": CORRECT_MM, "match_tol_mm": MATCH_TOL,
        "false_3d_cap_per_case_at_10mm": a.false_cap, "k_folds": K_FOLDS, "gates": GRID, "towers": ("view_specific" if a.view_specific else "shared"),
        "epochs": a.epochs, "neg_sample_per_class": a.neg_sample,
        "operational_headline_out_of_fold": headline, "operational_cap_sweep_out_of_fold": cap_sweep,
        "out_of_fold_fold_configs": fold_configs, "pair_separation_auroc_diagnostic": diag,
        "gate_ceilings_operational_within_5_10_15": {str(g): op_ceiling(g) for g in GRID},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=2))

    h = headline
    print(f"\nD2b LEARNED — out-of-fold operational headline (false@10-cap {a.false_cap}/case)")
    print(f"  recall 5/10/15mm: {h['recall5']}/{h['recall10']}/{h['recall15']} | matched@10 {h['n_matched_within10']} of {gt_total} GT | false@10 {h['false_3d_per_case_at_10mm']}/case")
    print(f"  cap sweep (false@10 cap -> recall10 @ realized false@10/case):")
    for cs in cap_sweep:
        cn = "none" if cs["false_3d_cap_per_case_at_10mm"] is None else cs["false_3d_cap_per_case_at_10mm"]
        print(f"    cap {str(cn):>5}: recall10 {cs['recall10']} (5 {cs['recall5']} / 15 {cs['recall15']}) @ false {cs['realized_false_3d_per_case_at_10mm']}/case | matched {cs['n_matched_within10']}")
    print(f"  pair-separation AUROC (diagnostic): pos-vs-cross {diag['auroc_pos_vs_cross']} | pos-vs-one-sided {diag['auroc_pos_vs_one_sided']} | pos-vs-fully-false {diag['auroc_pos_vs_fully_false']}")
    print(f"  op ceilings within10: " + " ".join(f"g{int(g)} {op_ceiling(g)['within10']}" for g in GRID))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
