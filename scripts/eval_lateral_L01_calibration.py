#!/usr/bin/env python3
"""STAGE L0.1 — LATERAL CALIBRATION + EXTRACTION AUDIT (read-only; no retraining).

L0 established that the lateral head is the binding systems problem (~188 peaks/case, ~97.5% spurious) but
ALSO exposed that the coarse floor grid {0.05, 0.10, 0.15, 0.20} is invalid for this checkpoint: lateral
scores are compressed just above the 0.05 extraction floor, so the sweep jumped straight from "everything"
(floor 0.05 -> 187.6 peaks/case) to "nothing" (floor 0.10 -> 0), and the reported FROC never characterized
the actual precision-recall tradeoff. L0.1 fixes that BEFORE any L1 intervention or L2 retraining is chosen.

Two things make this audit correct where L0 was coarse:
  1. peaks_from_hm is a pure local-max test — keep = (x == max_pool(x)) & (x > thresh) — so the max-pool
     structure is threshold-INDEPENDENT. The set of peaks with score >= t is identical whether we extract at
     floor t or extract ONCE at a low floor and filter post-hoc. L0.1 therefore extracts once per
     (case, view, NMS radius) at a low floor and sweeps a DENSE grid (0.04..0.14 by 0.005) exactly — the grid
     is dense enough to resolve the 0.05-0.10 range where all the action is.
  2. Threshold OPERATING POINTS are chosen OUT-OF-FOLD: for a target spurious-per-case budget, the threshold
     is derived on out-of-fold calibration cases and applied to the held-out fold, so no operating point is
     tuned on the same cases it is reported on (the L0 FROC took the whole val cohort at once).

Reported per view (lateral + AP for comparison):
  * heatmap AMPLITUDE — pooled-pixel percentiles (min/median/p90/p95/p99/max) and the per-case max/p99.9
    distribution: the decisive test of "does the lateral heatmap physically max near ~0.1" (real -> L2) vs a
    coarse-grid artifact. AP amplitude is the control (it MUST exceed op_threshold to deploy).
  * peak-score distribution (p10/p25/median/p75/p90/p95/p99) for compatible vs spurious peaks.
  * DENSE FROC per NMS radius {3,5,8}: at each threshold — compatible fracture recall, total/spurious/
    compatible peaks per case, precision among peaks, cases with zero candidates, duplicate-compatible peaks
    per fracture, and boundary / background / clustered spurious counts SEPARATELY (categories overlap; never
    summed), plus the fraction of compatible and spurious peaks exceeding the threshold.
  * OUT-OF-FOLD budget operating points: for spurious/case budgets {5,10,20,30}, the smallest OOF-derived
    threshold meeting the budget, applied to held-out folds; pooled held-out recall / spurious-per-case /
    precision / compatible-per-case. Answers "at a controlled false rate, what lateral recall survives?"

Attribution flags are per-peak: boundary (within EDGE_PX of any border), right-edge (cols 247-255 specifically),
background (local mean intensity < BG_INTENSITY), clustered (another retained peak within 2*NMS AND above the
same threshold — assessed exactly per threshold via a precomputed neighbor-max, not against a fixed field).

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory).
Provenance fails closed: sha256(--data) == detector det_dev_sha256; per-view weight hashes checked on load.

Usage:
  python eval_lateral_L01_calibration.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --out outputs/lateral_L01_calibration.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import torch
    import train_detector as T
    import run_ribassist as RR
    from eval_address_e2e import build_instance_records
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

EDGE_PX = 8            # a peak within this many px of any image border is "boundary"
BG_INTENSITY = 0.05   # local-patch mean below this (of the [0,1] image) is "background/padding"
BG_HALF = 3           # half-window for the local-intensity probe
EXTRACT_FLOOR = 0.04  # single low extraction floor; dense grid starts at 0.05 (deployed), sweeps up
GRID = np.round(np.arange(0.05, 0.1401, 0.005), 3)          # dense threshold grid over the compressed range
NMS_RADII = (3, 5, 8)
REF_NMS = 5           # deployed
BUDGETS = (5, 10, 20, 30)  # spurious-per-case budgets for OOF operating points
K_FOLDS = 5
PEAK_PCTL = (10, 25, 50, 75, 90, 95, 99)


def stat(v):
    v = np.asarray(v, np.float64)
    if v.size == 0: return None
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p10": round(float(np.percentile(v, 10)), 4), "p90": round(float(np.percentile(v, 90)), 4)}


def pctl_block(v, pctls=PEAK_PCTL):
    v = np.asarray(v, np.float64)
    if v.size == 0: return None
    out = {"n": int(v.size)}
    for p in pctls: out[f"p{p}"] = round(float(np.percentile(v, p)), 4)
    return out


def ratio(a, b):
    return round(a / b, 4) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch scipy nibabel", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)   # FAIL-CLOSED weight hashes
    data_sha = T.sha256_file(a.data)
    if data_sha != rec.get("det_dev_sha256"): raise ValueError("--data hash != detector det_dev_sha256")
    d = np.load(a.data, allow_pickle=False)
    cases = [str(c) for c in d["case"]]; case_to_idx = {c: i for i, c in enumerate(cases)}
    recs = build_instance_records(d); val_ids = [str(c) for c in rec["split"]["val_case_ids"] if str(c) in case_to_idx]
    RADIUS = T.MATCH_RADIUS_PX; H = W = T.PROTOCOL_SIZE; ncase = len(val_ids)
    views = ["lat", "ap"]
    fold_of = {c: i % K_FOLDS for i, c in enumerate(sorted(val_ids))}
    print(f"L0.1: lateral (+AP) calibration audit on {ncase} val cases. match radius {RADIUS}px, image {H}x{W}, "
          f"extract floor {EXTRACT_FLOOR}, grid {GRID[0]}..{GRID[-1]} step 0.005, nms {NMS_RADII}, OOF budgets {BUDGETS}.", flush=True)

    def run_hm(view, ci):
        with torch.no_grad():
            return nets[view](torch.from_numpy(d[view][ci].astype(np.float32))[None, None].to(dev))[0, 0].cpu().numpy()

    # heatmap amplitude collectors
    hm_pool_pct = {v: {p: [] for p in (0, 50, 90, 95, 99, 100)} for v in views}   # per-case pooled-pixel percentiles
    hm_case_max = {v: [] for v in views}; hm_case_p999 = {v: [] for v in views}
    # per (view, nms): case -> peak table [row,col,score,compat,boundary,rightedge,bg,neighbor_max]; case -> foot-hit score lists
    peaks_by = {v: {r: {} for r in NMS_RADII} for v in views}
    foothits_by = {v: {r: {} for r in NMS_RADII} for v in views}
    ngt = {v: 0 for v in views}
    # compatible/spurious peak scores at reference NMS + deployed floor, for the distribution block
    comp_scores = {v: [] for v in views}; spur_scores = {v: [] for v in views}

    for k, cid in enumerate(val_ids):
        ci = case_to_idx[cid]; gt = recs.get(ci, [])
        if k % 15 == 0: print(f"  case {k+1}/{ncase} ...", flush=True)
        for v in views:
            raw = d[v][ci].astype(np.float32); img = raw / max(float(raw.max()), 1e-6)   # [0,1] for intensity probe
            foots = [g[f"{v}_foot"] for g in gt]
            hm = run_hm(v, ci)
            hm_case_max[v].append(float(hm.max())); hm_case_p999[v].append(float(np.percentile(hm, 99.9)))
            for p, key in [(0, 0), (50, 50), (90, 90), (95, 95), (99, 99), (100, 100)]:
                hm_pool_pct[v][key].append(float(hm.min()) if p == 0 else (float(hm.max()) if p == 100 else float(np.percentile(hm, p))))
            for r in NMS_RADII:
                pk = T.peaks_from_hm(torch.from_numpy(hm), radius=r, thresh=EXTRACT_FLOOR)
                M = len(pk)
                tab = np.zeros((M, 8), np.float64)
                nb_max = np.full(M, -1.0)
                for i in range(M):
                    for j in range(i + 1, M):
                        if np.hypot(pk[i, 0] - pk[j, 0], pk[i, 1] - pk[j, 1]) <= 2 * r:
                            nb_max[i] = max(nb_max[i], pk[j, 2]); nb_max[j] = max(nb_max[j], pk[i, 2])
                foot_scores = [[] for _ in foots]
                for i in range(M):
                    row, col, sc = float(pk[i, 0]), float(pk[i, 1]), float(pk[i, 2])
                    dmin = min((T._min_dist(pk[i, :2], f) for f in foots), default=1e9)
                    compat = 1.0 if dmin <= RADIUS else 0.0
                    boundary = 1.0 if (row < EDGE_PX or row >= H - EDGE_PX or col < EDGE_PX or col >= W - EDGE_PX) else 0.0
                    rightedge = 1.0 if 247 <= col <= 255 else 0.0
                    r0, r1 = max(0, int(row) - BG_HALF), min(H, int(row) + BG_HALF + 1)
                    c0, c1 = max(0, int(col) - BG_HALF), min(W, int(col) + BG_HALF + 1)
                    bg = 1.0 if float(img[r0:r1, c0:c1].mean()) < BG_INTENSITY else 0.0
                    tab[i] = [row, col, sc, compat, boundary, rightedge, bg, nb_max[i]]
                    for fi, f in enumerate(foots):
                        if T._min_dist(pk[i, :2], f) <= RADIUS: foot_scores[fi].append(sc)
                    if r == REF_NMS and sc >= T.MIN_PEAK_SCORE:
                        (comp_scores if compat else spur_scores)[v].append(sc)
                peaks_by[v][r][cid] = tab
                foothits_by[v][r][cid] = [np.asarray(s, np.float64) for s in foot_scores]
            ngt[v] += len(foots)

    # -------- metric evaluation at a threshold over a set of cases --------
    def eval_at(v, r, t, case_subset):
        tot = spur = comp = boundary = bg = clustered = zero_cases = 0
        hits = dups = ngt_sub = 0
        for cid in case_subset:
            tab = peaks_by[v][r][cid]; fh = foothits_by[v][r][cid]
            sel = tab[:, 2] >= t if len(tab) else np.zeros(0, bool)
            n_sel = int(sel.sum()); tot += n_sel
            if n_sel == 0: zero_cases += 1
            if n_sel:
                sub = tab[sel]
                cflag = sub[:, 3] == 1.0
                comp += int(cflag.sum()); sflag = ~cflag; spur += int(sflag.sum())
                boundary += int((sflag & (sub[:, 4] == 1.0)).sum())
                bg += int((sflag & (sub[:, 6] == 1.0)).sum())
                clustered += int((sflag & (sub[:, 7] >= t)).sum())
            ngt_sub += len(fh)
            for s in fh:
                nabove = int((s >= t).sum()) if s.size else 0
                if nabove >= 1: hits += 1; dups += (nabove - 1)
        nc = len(case_subset)
        return {"threshold": round(float(t), 3),
                "recall": ratio(hits, ngt_sub), "peaks_per_case": round(tot / nc, 2) if nc else None,
                "spurious_per_case": round(spur / nc, 2) if nc else None,
                "compatible_per_case": round(comp / nc, 2) if nc else None,
                "precision_among_peaks": ratio(comp, tot),
                "zero_candidate_cases": zero_cases,
                "duplicate_compatible_per_fracture": ratio(dups, ngt_sub),
                "boundary_spurious_per_case": round(boundary / nc, 2) if nc else None,
                "background_spurious_per_case": round(bg / nc, 2) if nc else None,
                "clustered_spurious_per_case": round(clustered / nc, 2) if nc else None,
                "_raw": {"tot": tot, "spur": spur, "comp": comp, "hits": hits, "ngt": ngt_sub, "nc": nc}}

    # -------- fraction of compatible/spurious peaks exceeding each threshold (ref NMS) --------
    def frac_exceed(v, t):
        c = np.asarray(comp_scores[v]); s = np.asarray(spur_scores[v])
        return (round(float((c >= t).mean()), 4) if c.size else None,
                round(float((s >= t).mean()), 4) if s.size else None)

    # -------- OOF budget operating points --------
    def oof_operating(v, r):
        out = []
        for B in BUDGETS:
            per_fold_thr = []; held = {"tot": 0, "spur": 0, "comp": 0, "hits": 0, "ngt": 0, "nc": 0}
            feasible_folds = 0
            for fold in range(K_FOLDS):
                oof = [c for c in val_ids if fold_of[c] != fold]; heldout = [c for c in val_ids if fold_of[c] == fold]
                if not heldout: continue
                tstar = None
                for t in GRID:                                   # smallest threshold meeting the budget on OOF
                    m = eval_at(v, r, t, oof)
                    if m["spurious_per_case"] is not None and m["spurious_per_case"] <= B: tstar = t; break
                if tstar is None:                                # budget infeasible even at grid max
                    per_fold_thr.append(None); continue
                feasible_folds += 1; per_fold_thr.append(round(float(tstar), 3))
                hm = eval_at(v, r, tstar, heldout)["_raw"]
                for kk in held: held[kk] += hm[kk]
            thr_vals = [x for x in per_fold_thr if x is not None]
            out.append({"budget_spurious_per_case": B,
                        "feasible_folds": feasible_folds, "per_fold_threshold": per_fold_thr,
                        "threshold_median": round(float(np.median(thr_vals)), 3) if thr_vals else None,
                        "held_out_recall": ratio(held["hits"], held["ngt"]),
                        "held_out_spurious_per_case": round(held["spur"] / held["nc"], 2) if held["nc"] else None,
                        "held_out_compatible_per_case": round(held["comp"] / held["nc"], 2) if held["nc"] else None,
                        "held_out_precision_among_peaks": ratio(held["comp"], held["tot"]),
                        "held_out_cases": held["nc"]})
        return out

    result = {
        "stage": "L0.1 — lateral calibration + extraction audit (frozen; read-only)",
        "diagnostic_status": "development on the detector-validation split (biased; sealed test first confirmatory).",
        "detector_run": str(a.detector_run), "data_sha256": data_sha, "val_cases": ncase,
        "match_radius_px": RADIUS, "image_size": H, "extract_floor": EXTRACT_FLOOR,
        "deployed_nms_radius": T.NMS_RADIUS_PX, "deployed_floor": T.MIN_PEAK_SCORE, "op_threshold": op_thr, "lat_gate": lat_gate,
        "threshold_grid": [round(float(x), 3) for x in GRID], "nms_radii": list(NMS_RADII), "ref_nms": REF_NMS,
        "oof_budgets": list(BUDGETS), "k_folds": K_FOLDS,
        "note": "peaks_from_hm is a pure local-max test; the >=t peak set equals extract-at-floor-t, so the dense "
                "grid is swept exactly from one low-floor extraction. Attribution categories (boundary/background/"
                "clustered) OVERLAP and are never summed. OOF operating points derive the threshold on out-of-fold "
                "calibration cases and report on the held-out fold. compatible recall is many-to-one existential.",
        "per_view": {}}
    for v in views:
        froc = {r: [] for r in NMS_RADII}
        for r in NMS_RADII:
            for t in GRID:
                m = eval_at(v, r, float(t), val_ids)
                if r == REF_NMS:
                    fc, fs = frac_exceed(v, float(t)); m["frac_compatible_ge_thr"] = fc; m["frac_spurious_ge_thr"] = fs
                m.pop("_raw"); froc[r].append(m)
        result["per_view"][v] = {
            "gt_fractures": ngt[v],
            "heatmap_amplitude": {
                "pooled_pixel_min": stat(hm_pool_pct[v][0]), "pooled_pixel_median": stat(hm_pool_pct[v][50]),
                "pooled_pixel_p90": stat(hm_pool_pct[v][90]), "pooled_pixel_p95": stat(hm_pool_pct[v][95]),
                "pooled_pixel_p99": stat(hm_pool_pct[v][99]), "per_case_max": stat(hm_case_max[v]),
                "per_case_p999": stat(hm_case_p999[v])},
            "peak_score_distribution_ref_nms": {"compatible": pctl_block(comp_scores[v]), "spurious": pctl_block(spur_scores[v])},
            "availability_ceiling_at_extract_floor": {
                f"nms{r}": eval_at(v, r, EXTRACT_FLOOR, val_ids)["recall"] for r in NMS_RADII},
            "dense_froc": {f"nms{r}": froc[r] for r in NMS_RADII},
            "oof_budget_operating_points": {f"nms{r}": oof_operating(v, r) for r in NMS_RADII},
        }
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2))

    # -------- console --------
    print(f"\nL0.1 LATERAL CALIBRATION + EXTRACTION AUDIT — {ncase} val cases")
    for v in views:
        pv = result["per_view"][v]; amp = pv["heatmap_amplitude"]
        cm = amp["per_case_max"]; pm = amp["pooled_pixel_median"]
        pd_ = pv["peak_score_distribution_ref_nms"]
        print(f"  [{v}] heatmap amplitude: per-case MAX median {cm['median'] if cm else None} (p10 {cm['p10'] if cm else None} p90 {cm['p90'] if cm else None}) | "
              f"pooled-pixel median {pm['median'] if pm else None} p99 {amp['pooled_pixel_p99']['median'] if amp['pooled_pixel_p99'] else None}")
        cc, ss = pd_["compatible"], pd_["spurious"]
        print(f"       peak score (ref nms{REF_NMS}) compatible p50 {cc['p50'] if cc else None}/p90 {cc['p90'] if cc else None} "
              f"vs spurious p50 {ss['p50'] if ss else None}/p90 {ss['p90'] if ss else None}")
    for v in views:
        print(f"  [{v}] dense FROC @ nms{REF_NMS} (thr -> recall / spur-per-case / precision):")
        for m in result["per_view"][v]["dense_froc"][f"nms{REF_NMS}"]:
            print(f"    thr {m['threshold']:.3f}: {m['recall']} / {m['spurious_per_case']} / {m['precision_among_peaks']}")
    for v in views:
        av = result["per_view"][v]["availability_ceiling_at_extract_floor"]
        print(f"  [{v}] availability ceiling @ extract floor {EXTRACT_FLOOR} (recall, ANY compatible peak): "
              + " ".join(f"nms{r}={av[f'nms{r}']}" for r in NMS_RADII))
    for v in views:
        for r in NMS_RADII:
            print(f"  [{v}] OOF budget operating points @ nms{r} (budget -> thr_median, held-out recall @ spur/case):")
            for op in result["per_view"][v]["oof_budget_operating_points"][f"nms{r}"]:
                print(f"    budget {op['budget_spurious_per_case']}: thr {op['threshold_median']} (feasible {op['feasible_folds']}/{K_FOLDS}) "
                      f"-> recall {op['held_out_recall']} @ {op['held_out_spurious_per_case']} spur/case, prec {op['held_out_precision_among_peaks']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
