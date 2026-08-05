#!/usr/bin/env python3
"""STAGE L0 — LATERAL DETECTOR DIAGNOSIS (read-only; no retraining).

The A->D2 program established that cross-view correspondence is not solvable at the pair-scoring layer on the
current detections, and that the binding lever is lateral candidate quality: ~188 lateral peaks/case create an
unresolvable competitor field, and lateral dual-view availability caps recall at 51.8%. L0 characterizes WHY
the lateral head is so spurious-peak-heavy, BEFORE any intervention (L1) or retraining (L2).

It runs the frozen detector's LATERAL (and AP, for comparison) heatmaps on the detector-validation cases,
extracts peaks with the deployed extraction (train_detector.peaks_from_hm), classifies each peak as
TRUE-COMPATIBLE (within the detector's match radius of some GT footprint for that view — the D0 many-to-one
criterion) vs SPURIOUS, and reports:

  * peaks/case and compatible vs spurious counts (AP vs lateral);
  * FROC: per-view fracture recall vs SPURIOUS peaks/case over a score-threshold sweep;
  * score distributions for compatible vs spurious peaks (can a threshold separate them?);
  * spurious-peak ATTRIBUTION: image-boundary (esp. columns 247-255), near-zero-intensity background/padding,
    near-extraction-floor score, and repeated/clustered local maxima;
  * heatmap-score CALIBRATION: fraction-compatible per score bin (does score ~ P(true)?);
  * per-fracture stratification: fractures with NO compatible lateral peak (the availability loss) and the best
    compatible-peak score for those that have one;
  * NMS-radius and extraction-floor SENSITIVITY: peaks/case, recall, and spurious/case as (radius, floor) vary,
    to see whether cheap extraction changes (L1) can thin the flood without costing compatible recall.

This determines whether the lateral problem is primarily model learning, projection construction, peak
extraction, or boundary artifacts — routing L1 (cheap precision) vs L2 (retraining).

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory).
Provenance fails closed: sha256(--data) == detector det_dev_sha256; per-view weight hashes checked on load.

Usage:
  python eval_lateral_L0_diagnosis.py \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --out outputs/lateral_L0_diagnosis.json
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

EDGE_PX = 8          # a peak within this many px of any image border is "boundary"
BG_INTENSITY = 0.05  # local-patch mean below this (of the [0,1] image) is "background/padding"
BG_HALF = 3          # half-window for the local-intensity probe


def stat(v):
    v = np.asarray(v, np.float64)
    if v.size == 0: return None
    return {"n": int(v.size), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
            "p10": round(float(np.percentile(v, 10)), 4), "p90": round(float(np.percentile(v, 90)), 4)}


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
    RADIUS = T.MATCH_RADIUS_PX; H = W = T.PROTOCOL_SIZE
    print(f"L0: lateral (+AP) diagnosis on {len(val_ids)} val cases. match radius {RADIUS}px, image {H}x{W}, "
          f"deployed NMS {T.NMS_RADIUS_PX} floor {T.MIN_PEAK_SCORE}.", flush=True)

    def run_hm(view, ci):
        with torch.no_grad():
            return nets[view](torch.from_numpy(d[view][ci].astype(np.float32))[None, None].to(dev))[0, 0].cpu().numpy()

    views = ["lat", "ap"]
    # per-view collectors
    peaks_per_case = {v: [] for v in views}; comp_scores = {v: [] for v in views}; spur_scores = {v: [] for v in views}
    best_comp_score = {v: [] for v in views}      # per GT footprint: max compatible-peak score (or -1)
    spur_boundary = {v: 0 for v in views}; spur_bg = {v: 0 for v in views}; spur_clustered = {v: 0 for v in views}
    spur_total = {v: 0 for v in views}; spur_col_hist = {v: np.zeros(W // 16 + 1, int) for v in views}
    spur_rightedge = {v: 0 for v in views}        # cols 247..255 specifically
    all_scores = {v: [] for v in views}; all_comp = {v: [] for v in views}   # for calibration
    # FROC accumulators: per GT best-compatible score (recall) + per-case spurious score lists
    case_spur_scores = {v: [] for v in views}
    n_gt = {v: 0 for v in views}
    # NMS/floor sensitivity
    SENS = [(r, f) for r in (3, 5, 8) for f in (0.05, 0.1, 0.15, 0.2)]
    sens_acc = {v: {rf: {"peaks": 0, "spur": 0, "gt_hit": 0} for rf in SENS} for v in views}
    hm_max = {v: [] for v in views}; hm_p999 = {v: [] for v in views}   # heatmap AMPLITUDE per case

    for k, cid in enumerate(val_ids):
        ci = case_to_idx[cid]; gt = recs.get(ci, [])
        if k % 15 == 0: print(f"  case {k+1}/{len(val_ids)} ...", flush=True)
        for v in views:
            img = d[v][ci].astype(np.float32); img = img / max(float(img.max()), 1e-6)  # [0,1] for intensity probe
            foots = [g[f"{v}_foot"] for g in gt]; n_gt[v] += len(foots)
            hm = run_hm(v, ci)
            hm_max[v].append(float(hm.max())); hm_p999[v].append(float(np.percentile(hm, 99.9)))  # AMPLITUDE
            pk = T.peaks_from_hm(torch.from_numpy(hm))                       # deployed extraction
            peaks_per_case[v].append(len(pk))
            # per-GT best compatible score
            for f in foots:
                best = -1.0
                for r in range(len(pk)):
                    if T._min_dist(pk[r, :2], f) <= RADIUS: best = max(best, float(pk[r, 2]))
                best_comp_score[v].append(best)
            # per-peak classification
            case_spur = []
            # precompute clustering: peaks within 2*NMS of another peak
            near = np.zeros(len(pk), bool)
            for i in range(len(pk)):
                for j in range(i + 1, len(pk)):
                    if np.hypot(pk[i, 0] - pk[j, 0], pk[i, 1] - pk[j, 1]) <= 2 * T.NMS_RADIUS_PX: near[i] = near[j] = True
            for r in range(len(pk)):
                row, col, sc = float(pk[r, 0]), float(pk[r, 1]), float(pk[r, 2])
                comp = any(T._min_dist(pk[r, :2], f) <= RADIUS for f in foots)
                all_scores[v].append(sc); all_comp[v].append(1 if comp else 0)
                if comp: comp_scores[v].append(sc)
                else:
                    spur_scores[v].append(sc); spur_total[v] += 1; case_spur.append(sc)
                    boundary = (row < EDGE_PX or row >= H - EDGE_PX or col < EDGE_PX or col >= W - EDGE_PX)
                    r0, r1 = max(0, int(row) - BG_HALF), min(H, int(row) + BG_HALF + 1)
                    c0, c1 = max(0, int(col) - BG_HALF), min(W, int(col) + BG_HALF + 1)
                    bg = float(img[r0:r1, c0:c1].mean()) < BG_INTENSITY
                    if boundary: spur_boundary[v] += 1
                    if bg: spur_bg[v] += 1
                    if near[r]: spur_clustered[v] += 1
                    spur_col_hist[v][min(int(col) // 16, W // 16)] += 1
                    if 247 <= col <= 255: spur_rightedge[v] += 1
            case_spur_scores[v].append(case_spur)
            # NMS/floor sensitivity
            for (rr, ff) in SENS:
                pk2 = T.peaks_from_hm(torch.from_numpy(hm), radius=rr, thresh=ff)
                sens_acc[v][(rr, ff)]["peaks"] += len(pk2)
                sp = sum(1 for r in range(len(pk2)) if not any(T._min_dist(pk2[r, :2], f) <= RADIUS for f in foots))
                sens_acc[v][(rr, ff)]["spur"] += sp
                hit = sum(1 for f in foots if any(T._min_dist(pk2[r, :2], f) <= RADIUS for r in range(len(pk2))))
                sens_acc[v][(rr, ff)]["gt_hit"] += hit

    ncase = len(val_ids)

    def froc(v):
        best = np.asarray(best_comp_score[v], np.float64)   # per GT: max compat score or -1
        thr = np.unique(np.concatenate([[T.MIN_PEAK_SCORE], np.linspace(T.MIN_PEAK_SCORE, 1.0, 40)]))
        rows = []
        for t in thr:
            recall = float((best >= t).mean()) if best.size else 0.0
            spc = np.mean([sum(1 for s in cs if s >= t) for cs in case_spur_scores[v]]) if case_spur_scores[v] else 0.0
            rows.append({"score_thr": round(float(t), 3), "recall": round(recall, 4), "spurious_per_case": round(float(spc), 2)})
        return rows

    def calib(v):
        s = np.asarray(all_scores[v], np.float64); c = np.asarray(all_comp[v], np.float64)
        out = []
        for lo in np.linspace(0, 0.9, 10):
            hi = lo + 0.1; m = (s >= lo) & (s < hi + (1e-9 if hi >= 1 else 0))
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "frac_compatible": round(float(c[m].mean()), 4) if m.any() else None})
        return out

    result = {
        "stage": "L0 — lateral detector diagnosis (frozen; read-only)",
        "diagnostic_status": "development on the detector-validation split (biased; sealed test first confirmatory).",
        "detector_run": str(a.detector_run), "data_sha256": data_sha, "val_cases": ncase,
        "match_radius_px": RADIUS, "image_size": H, "deployed_nms_radius": T.NMS_RADIUS_PX, "deployed_floor": T.MIN_PEAK_SCORE,
        "per_view": {}, "note": "TRUE-COMPATIBLE = peak within match radius of some GT footprint (many-to-one, D0 "
                                "criterion). SPURIOUS attribution categories (boundary / background / clustered) are "
                                "non-exclusive. FROC recall is many-to-one existential per-view fracture recall."}
    for v in views:
        result["per_view"][v] = {
            "peaks_per_case": stat(peaks_per_case[v]),
            "gt_fractures": n_gt[v], "spurious_peaks_total": spur_total[v], "spurious_per_case": round(spur_total[v] / ncase, 1),
            "heatmap_amplitude_per_case_max": stat(hm_max[v]), "heatmap_amplitude_per_case_p999": stat(hm_p999[v]),
            "compatible_recall_at_floor": ratio(int((np.asarray(best_comp_score[v]) >= 0).sum()), n_gt[v]),
            "score_compatible": stat(comp_scores[v]), "score_spurious": stat(spur_scores[v]),
            "spurious_attribution": {
                "boundary_frac": ratio(spur_boundary[v], spur_total[v]),
                "right_edge_cols_247_255_frac": ratio(spur_rightedge[v], spur_total[v]),
                "background_padding_frac": ratio(spur_bg[v], spur_total[v]),
                "clustered_repeated_frac": ratio(spur_clustered[v], spur_total[v]),
                "spurious_col_histogram_bins_of_16": [int(x) for x in spur_col_hist[v]]},
            "froc_recall_vs_spurious_per_case": froc(v),
            "heatmap_score_calibration": calib(v),
            "nms_floor_sensitivity": [
                {"nms_radius": rr, "floor": ff, "peaks_per_case": round(sens_acc[v][(rr, ff)]["peaks"] / ncase, 1),
                 "spurious_per_case": round(sens_acc[v][(rr, ff)]["spur"] / ncase, 1),
                 "compatible_recall": ratio(sens_acc[v][(rr, ff)]["gt_hit"], n_gt[v])} for (rr, ff) in SENS],
        }
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2))

    print(f"\nL0 LATERAL DETECTOR DIAGNOSIS — {ncase} val cases")
    for v in views:
        pv = result["per_view"][v]; att = pv["spurious_attribution"]
        amx = pv["heatmap_amplitude_per_case_max"]; ap999 = pv["heatmap_amplitude_per_case_p999"]
        print(f"  [{v}] peaks/case {pv['peaks_per_case']['mean']} | spurious/case {pv['spurious_per_case']} | "
              f"compatible recall@floor {pv['compatible_recall_at_floor']}")
        print(f"       heatmap amplitude/case: max median {amx['median'] if amx else None} (p10 {amx['p10'] if amx else None} "
              f"p90 {amx['p90'] if amx else None}) | p99.9 median {ap999['median'] if ap999 else None}")
        print(f"       score median: compatible {pv['score_compatible']['median'] if pv['score_compatible'] else None} "
              f"vs spurious {pv['score_spurious']['median'] if pv['score_spurious'] else None}")
        print(f"       spurious attribution: boundary {att['boundary_frac']} (right-edge247-255 {att['right_edge_cols_247_255_frac']}) "
              f"| background {att['background_padding_frac']} | clustered {att['clustered_repeated_frac']}")
    print(f"  lateral FROC (recall @ spurious/case): " + " ".join(
        f"{r['recall']}@{r['spurious_per_case']}" for r in result["per_view"]["lat"]["froc_recall_vs_spurious_per_case"][::8]))
    for v in views:
        print(f"  [{v}] NMS/floor sensitivity (nms,floor -> peaks/case, spur/case, recall):")
        for row in result["per_view"][v]["nms_floor_sensitivity"]:
            print(f"    nms{row['nms_radius']} floor{row['floor']}: {row['peaks_per_case']} / {row['spurious_per_case']} / {row['compatible_recall']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
