#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""RibAssist 3D END-TO-END DEMO figures on the SEALED cohort — presentation-ready, FAIL-CLOSED, no retraining.

Replays the FROZEN L2 deployment policy on the sealed pair graph to recover the EXACT committed AP<->lateral
correspondences, matches each to GT by nearest-fracture-voxel distance at 10 mm (identical semantics to the D1
operational scoreboard), and REFUSES TO RENDER unless that replay reproduces the sealed D1 matched@10 count. Only
then does it emit figures — so no presentation figure is ever built from a replay that merely "looks close".

Peak overlays come from the FROZEN sealed D0 graph coordinates (not a fresh peak re-extraction); the detector is
rerun only to draw the heatmap backgrounds. Rib identity is the AUTHORITATIVE rib-seg label under the fracture
voxels + fracture_metrics' rib_exact/rib_within1 — never an unverified nearest-centerline guess.

Composition (4 figures): best-localization success, typical success (error near the median), challenging success
(near the 10 mm boundary but still correct), and an ABSTENTION failure (an eligible correct pair present under the
frozen policy that the assignment did not commit — the honest high-confidence bottleneck). Failure panels label
the correct pair as "not committed" and show NO triangulated point (no counterfactual reconstruction).

Usage (from RibAssist 3D ROOT):
  python scripts/make_demo_figures.py \
      --detector-run outputs/detector_L2_lateral_hnm --data outputs/det_out_v2/det_test.npz \
      --pairs-npz outputs/sealed/L2_sealed_D0_pairs.npz --policy outputs/sealed/L2_policy.json \
      --sealed-d1-json outputs/sealed/L2_sealed_D1.json \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --expected-data-sha256 <det_test sha> --out-dir outputs/demo
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

try:
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.spatial import cKDTree
    from scipy.optimize import linear_sum_assignment
    import train_detector as T
    import run_ribassist as RR
    from eval_address_e2e import build_instance_records
    from eval_biplanar_geometry import back_project, case_gt, fracture_metrics, apply_affine
    from eval_correspondence_D1_assign import edge_cost, u_grid, assign_abstain
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

CORRECT_MM = 10.0


def gt_rib_label(g, iid):
    """AUTHORITATIVE GT rib for a fracture: majority nonzero rib-seg label under its fracture voxels (the rib the
    fracture is annotated on — the same overlap fracture_metrics uses). Returns (label or None)."""
    vox = g["fl_groups"].get(iid)
    if vox is None or vox.shape[1] == 0: return None
    labs = g["rs"][vox[0], vox[1], vox[2]]; labs = labs[labs > 0]
    if labs.size == 0: return None
    lab = int(Counter(labs.tolist()).most_common(1)[0][0])
    return lab if lab in g["info"] else None


def rib_name(g, label):
    return f"{g['info'][label]['side']}{int(g['info'][label]['num'])}" if label in g["info"] else "?"


def replay_accepted(z, gate, cost, mb, u):
    """{cid: [accepted global-row ints]} for a single (gate,cost,mb,u) — identical to D1's ACC construction."""
    cid_arr = np.array([str(x) for x in z["case_id"]]); apx = z["ap_idx"]; ltx = z["lat_idx"]
    ap_s = z["ap_score"].astype(np.float64); lt_s = z["lat_score"].astype(np.float64); dsi = z["dsi_vox"].astype(np.float64)
    gmask = dsi <= gate; costs_all = edge_cost(cost, ap_s, lt_s, dsi, gate)
    rows_by_case = {}
    for r in range(len(cid_arr)): rows_by_case.setdefault(cid_arr[r], []).append(r)
    out = {}
    for cid, rows in rows_by_case.items():
        rr = [r for r in rows if gmask[r]]
        if not rr: out[cid] = []; continue
        aps_u = sorted({int(apx[r]) for r in rr}); lts_u = sorted({int(ltx[r]) for r in rr})
        ai = {v: i for i, v in enumerate(aps_u)}; lj = {v: i for i, v in enumerate(lts_u)}
        na, nl = len(aps_u), len(lts_u); BIG = 1e9; Mreal = np.full((na, nl), BIG); cost_of = {}
        for r in rr:
            i, j = ai[int(apx[r])], lj[int(ltx[r])]
            if costs_all[r] < Mreal[i, j]: Mreal[i, j] = costs_all[r]; cost_of[(i, j)] = (r, float(costs_all[r]))
        if mb:
            rowmin = Mreal.min(1); colmin = Mreal.min(0)
            cost_of = {k: v for k, v in cost_of.items()
                       if Mreal[k] <= rowmin[k[0]] + 1e-12 and Mreal[k] <= colmin[k[1]] + 1e-12}
        out[cid] = list(assign_abstain(na, nl, cost_of, u))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pairs-npz", type=Path, required=True); ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--sealed-d1-json", type=Path, default=None, help="sealed D1 apply JSON; its matched@10 is the expected replay count")
    ap.add_argument("--expected-matched-at10", type=int, default=15, help="fallback expected matched@10 if no --sealed-d1-json")
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--expected-data-sha256", default=None)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--distinct-cases", action="store_true",
                    help="force best/typical/challenging onto DISTINCT patient cases (default off: pick the scientifically "
                         "ideal example per category even if two share a multi-fracture case)")
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch matplotlib scipy nibabel", file=sys.stderr); return 1

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not a.overwrite and any(a.out_dir.glob("demo_*.png")):
        raise FileExistsError(f"{a.out_dir} already contains demo_*.png; pass --overwrite to regenerate.")

    dev = T.device()
    nets, arch, si_tol, op_thr, lat_gate, rec = RR.load_detector(a.detector_run, dev)   # FAIL-CLOSED weight hashes
    data_sha = T.sha256_file(a.data)
    if a.expected_data_sha256 and data_sha != a.expected_data_sha256:
        raise ValueError(f"--data sha {data_sha[:12]}.. != --expected-data-sha256 {a.expected_data_sha256[:12]}..")
    z = np.load(a.pairs_npz, allow_pickle=False)
    if str(z["data_sha256"]) != data_sha: raise ValueError("pairs NPZ data hash != --data")
    # provenance: pair graph must come from THIS detector run + checkpoints
    if Path(str(z["detector_run"])).resolve() != a.detector_run.resolve():
        raise ValueError(f"pairs NPZ detector_run {str(z['detector_run'])} != --detector-run")
    if "detector_ap_sha256" in z.files:
        for v, key in (("ap", "detector_ap_sha256"), ("lat", "detector_lat_sha256")):
            if T.sha256_file(a.detector_run / f"detector_{v}.pt") != str(z[key]):
                raise ValueError(f"detector_{v}.pt hash != NPZ record (graph came from a different checkpoint)")
    pol = json.loads(a.policy.read_text())
    if "detector_run" in pol and Path(str(pol["detector_run"])).resolve() != a.detector_run.resolve():
        raise ValueError(f"policy detector_run {pol['detector_run']} != --detector-run (frozen policy is for a different detector)")
    gate = float(pol["gate"]); cost = str(pol["cost"]); mb = bool(pol["mutual_best"]); u_want = float(pol["u"])
    ug = list(u_grid(cost, gate)); u = min(ug, key=lambda x: abs(x - u_want))
    if abs(u - u_want) > 1e-4: raise ValueError(f"policy u={u_want} not on grid {ug}")

    lat_nms = int(z["lat_nms"]) if "lat_nms" in z.files else T.NMS_RADIUS_PX
    lat_floor = float(z["lat_floor"]) if "lat_floor" in z.files else T.MIN_PEAK_SCORE
    ap_nms = int(z["ap_nms"]) if "ap_nms" in z.files else T.NMS_RADIUS_PX
    ap_floor = float(z["ap_floor"]) if "ap_floor" in z.files else T.MIN_PEAK_SCORE
    expected = a.expected_matched_at10
    if a.sealed_d1_json and a.sealed_d1_json.exists():
        expected = int(json.loads(a.sealed_d1_json.read_text())["operational_headline"]["n_matched_within10"])

    d = np.load(a.data, allow_pickle=False); cases = [str(c) for c in d["case"]]
    case_to_idx = {c: i for i, c in enumerate(cases)}; recs = build_instance_records(d)
    cid_arr = np.array([str(x) for x in z["case_id"]])
    ap_row = z["ap_row"]; ap_col = z["ap_col"]; lat_row = z["lat_row"]; lat_col = z["lat_col"]
    apx = z["ap_idx"]; ltx = z["lat_idx"]; cgi = z["case_global_idx"]; cls = z["cls"]; shared = z["shared_iid"]
    ap_s = z["ap_score"].astype(np.float64); lt_s = z["lat_score"].astype(np.float64); dsi_v = z["dsi_vox"].astype(np.float64)
    class_names = [str(x) for x in z["class_names"]]; POS = class_names.index("positive_capable")
    all_ap_geo = z["all_ap_geo"]; all_lat_geo = z["all_lat_geo"]
    all_costs = edge_cost(cost, ap_s, lt_s, dsi_v, gate)   # frozen-policy edge cost, computed once
    accepted = replay_accepted(z, gate, cost, mb, u)
    accepted_rows = {cid: set(rows) for cid, rows in accepted.items()}
    print(f"demo: policy gate {int(gate)} {cost}{'+mb' if mb else ''} u={u:.4f} | lat nms{lat_nms}/{lat_floor} ap nms{ap_nms}/{ap_floor} | expected matched@10 {expected}", flush=True)

    rows_by_case = {}
    for r in range(len(cid_arr)): rows_by_case.setdefault(cid_arr[r], []).append(r)

    # ---- match accepted edges to GT (nearest fracture-voxel dist, Hungarian @10mm) ----
    matched = []                    # (cid, row, iid, dist_mm, metrics)
    matched_iids_by_case = {}
    correct_row_by_case = {}        # cid -> {iid: lowest-cost positive-capable, gate-eligible row}
    gcache = {}
    for cid in cases:
        g = case_gt(cid, a.image_dirs, a.seg_dir, a.cl_dir)
        if g is None: matched_iids_by_case[cid] = set(); continue
        gcache[cid] = g
        # lowest-cost, gate-eligible, positive-capable correct pair per iid (NOT the first encountered)
        cr = {}
        for r in rows_by_case.get(cid, []):
            if cls[r] != POS or int(shared[r]) < 0 or dsi_v[r] > gate: continue
            iid = int(shared[r])
            if iid not in cr or all_costs[r] < all_costs[cr[iid]]: cr[iid] = r
        correct_row_by_case[cid] = cr
        rows = accepted.get(cid, [])
        iids = [int(k) for k in g["fl_groups"].keys()]
        trees = {iid: cKDTree(apply_affine(g["aff"], g["fl_groups"][iid].T.astype(np.float64)))
                 for iid in iids if g["fl_groups"][iid].shape[1] > 0}
        if not rows: matched_iids_by_case[cid] = set(); continue
        pts = {}
        for r in rows:
            p_rec, si_dis = back_project((float(ap_row[r]), float(ap_col[r])), (float(lat_row[r]), float(lat_col[r])),
                                         all_ap_geo[cgi[r]], all_lat_geo[cgi[r]])
            pts[r] = (p_rec, si_dis, apply_affine(g["aff"], p_rec))
        BIG = 1e9; M = np.full((len(rows), len(iids)), BIG)
        for pi, r in enumerate(rows):
            for gi, iid in enumerate(iids):
                if iid in trees:
                    dd, _ = trees[iid].query(pts[r][2][None], k=1); dd = float(dd[0])
                    if dd <= CORRECT_MM: M[pi, gi] = dd
        ri, ci = linear_sum_assignment(M); mi = set()
        for pi, gi in zip(ri, ci):
            if M[pi, gi] <= CORRECT_MM:
                r = rows[pi]; iid = iids[gi]; p_rec, si_dis, _ = pts[r]
                m, _ = fracture_metrics(p_rec, si_dis, g["fl_groups"][iid], g, cid, iid)
                matched.append((cid, r, iid, float(M[pi, gi]), m)); mi.add(iid)
        matched_iids_by_case[cid] = mi
    n_match = len(matched)

    # ---- FAIL CLOSED: refuse to render unless the replay reproduces the sealed matched@10 ----
    if n_match != expected:
        raise AssertionError(f"Policy replay produced {n_match} matched fractures at 10 mm; expected {expected}. "
                             "Refusing to generate demo figures.")
    print(f"  replay reproduced sealed matched@10 = {n_match} (== expected). Proceeding.", flush=True)

    # ---- select demo fractures AFTER validation (DISTINCT patient cases across the 3 successes) ----
    succ = [m for m in matched if m[4]]
    dists = sorted(s[3] for s in succ); med = dists[len(dists) // 2] if dists else 0.0
    picks, used_pairs, used_cases = [], set(), set()
    def take(cands, tag, distinct_case=None):
        dc = a.distinct_cases if distinct_case is None else distinct_case
        for m in cands:
            if m is None or (m[0], m[2]) in used_pairs: continue
            if dc and m[0] in used_cases: continue
            picks.append((tag, m)); used_pairs.add((m[0], m[2])); used_cases.add(m[0]); return True
        return False
    take(sorted(succ, key=lambda x: x[3]), "success_best")                                   # smallest fracture-volume distance
    take(sorted((s for s in succ if s[4]["rib_exact"]), key=lambda x: -x[3]), "success_challenging")  # largest, still correct
    take(sorted(succ, key=lambda x: abs(x[3] - med)), "success_typical")                      # nearest the median
    if len([p for p in picks if p[0].startswith("success")]) < 3:   # backfill if a category collided
        take(sorted(succ, key=lambda x: x[3]), "success_extra", distinct_case=False)

    # failure: prefer a PURE ABSTENTION — an eligible correct pair whose AP and lat peaks were BOTH left uncommitted
    fail = None; fail_fallback = None
    for cid, cr in correct_row_by_case.items():
        ar = accepted_rows.get(cid, set())
        acc_ap = {int(apx[r]) for r in ar}; acc_lat = {int(ltx[r]) for r in ar}
        for iid, r in cr.items():
            if iid in matched_iids_by_case.get(cid, set()): continue
            pure_abstain = (r not in ar) and (int(apx[r]) not in acc_ap) and (int(ltx[r]) not in acc_lat)
            fclass = "abstained" if r not in ar else "accepted_but_not_correct_at_10mm"
            cand = (cid, iid, r, fclass, pure_abstain)
            if pure_abstain and fail is None: fail = cand
            if fail_fallback is None: fail_fallback = cand
    fail = fail or fail_fallback
    if fail: picks.append(("failure", fail))
    print(f"  selected: {[(t, m[0]) for t, m in picks]}", flush=True)

    # ---- rendering ----
    def hm_bg(cid):
        ci = case_to_idx[cid]; out = {}
        with torch.no_grad():
            for v in ("ap", "lat"):
                out[v] = (d[v][ci].astype(np.float32),
                          nets[v](torch.from_numpy(d[v][ci].astype(np.float32))[None, None].to(dev))[0, 0].cpu().numpy())
        return out

    def graph_peaks(cid, view):   # unique candidate coords from the FROZEN graph (not re-extracted)
        rows = rows_by_case.get(cid, [])
        rc = np.column_stack(((ap_row if view == "ap" else lat_row)[rows], (ap_col if view == "ap" else lat_col)[rows]))
        return np.unique(rc, axis=0) if len(rc) else rc

    def render(kind, cid, iid, row, dist, m):
        g = gcache[cid]; bg = hm_bg(cid); recmap = {r["iid"]: r for r in recs[case_to_idx[cid]]}
        is_succ = kind.startswith("success")
        col = "#1a7f37" if is_succ else "#b3261e"
        fig = plt.figure(figsize=(12.5, 12.5)); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45], hspace=0.32, wspace=0.14, top=0.93, bottom=0.04)
        pair_label = "committed peak" if is_succ else "correct candidate — NOT committed"
        pair_marker = "o" if is_succ else "X"
        for ri_, v, vlabel in ((0, "ap", "AP"), (1, "lat", "Lateral")):
            img, hm = bg[v]; pk = graph_peaks(cid, v)
            axL = fig.add_subplot(gs[ri_, 0]); axL.imshow(img, cmap="gray")
            axL.set_title(f"{vlabel} projection + sealed-graph candidates", fontsize=10)
            if len(pk): axL.scatter(pk[:, 1], pk[:, 0], s=12, facecolors="none", edgecolors="#f6c343", linewidths=0.8, label="candidate peaks")
            foot = recmap[iid][f"{v}_foot"]; axL.scatter(foot[:, 1].mean(), foot[:, 0].mean(), s=130, marker="+", c="#2f81f7", linewidths=1.7, label="GT center")
            ar, ac = (ap_row, ap_col) if v == "ap" else (lat_row, lat_col)
            axL.scatter([ac[row]], [ar[row]], s=130, marker=pair_marker, facecolors="none", edgecolors=col, linewidths=2.2, label=pair_label)
            axL.set_xticks([]); axL.set_yticks([])
            if ri_ == 0: axL.legend(loc="upper right", fontsize=7, framealpha=0.9)
            axH = fig.add_subplot(gs[ri_, 1]); im = axH.imshow(hm, cmap="magma", vmin=0)
            axH.set_title(f"{vlabel} detector heatmap (max {hm.max():.3f})", fontsize=10)
            axH.scatter([ac[row]], [ar[row]], s=90, marker=pair_marker, facecolors="none", edgecolors="white", linewidths=1.6)
            axH.set_xticks([]); axH.set_yticks([]); fig.colorbar(im, ax=axH, fraction=0.046, pad=0.02)
        # 3D reconstruction on rib anatomy
        ax3 = fig.add_subplot(gs[2, :], projection="3d")
        gl = gt_rib_label(g, iid)
        vox = g["fl_groups"][iid].T.astype(np.float64); gtw = apply_affine(g["aff"], vox)
        if len(gtw):
            sub = gtw[np.random.RandomState(0).choice(len(gtw), min(400, len(gtw)), replace=False)]
            ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=6, c="#2f81f7", alpha=0.4, label="GT fracture voxels")
        if gl: cl = g["cl_world"][gl - 1]; ax3.plot(cl[:, 0], cl[:, 1], cl[:, 2], c="#8250df", lw=2, label=f"GT rib {rib_name(g, gl)} centerline")
        if is_succ:
            p_rec, si_dis = back_project((float(ap_row[row]), float(ap_col[row])), (float(lat_row[row]), float(lat_col[row])),
                                         all_ap_geo[cgi[row]], all_lat_geo[cgi[row]])
            p_w = apply_affine(g["aff"], p_rec)
            ax3.scatter([p_w[0]], [p_w[1]], [p_w[2]], s=170, marker="*", c=col, depthshade=False, label="triangulated 3D point")
            nn = gtw[np.argmin(np.linalg.norm(gtw - p_w[None], axis=1))]
            ax3.plot([p_w[0], nn[0]], [p_w[1], nn[1]], [p_w[2], nn[2]], c=col, ls="--", lw=1.4)
        ax3.set_xlabel("LR (mm)"); ax3.set_ylabel("AP (mm)"); ax3.set_zlabel("SI (mm)")
        ax3.view_init(elev=16, azim=-70); ax3.legend(loc="upper left", fontsize=8)
        gmean = float(np.sqrt(max(ap_s[row], 0) * max(lt_s[row], 0)))   # geomean detector confidence of this pair
        thr = max(0.0, 1.0 - 2.0 * u)                                    # geomean-conf commit threshold (=1-2u)
        if is_succ:
            ribtxt = f"rib exact ✓ ({rib_name(g, gl)})" if m["rib_exact"] else (f"rib ±1 (GT {rib_name(g, gl)})" if m["rib_within1"] else f"rib ✗ (GT {rib_name(g, gl)})")
            title = f"{kind.replace('success_','').upper()} SUCCESS — {cid}  |  fracture-volume distance {dist:.2f} mm  |  {ribtxt}  |  along-rib error {m['along_mm']:.1f} mm"
            sub = (f"fracture-volume distance = distance to nearest fracture voxel (0 = inside mask; the operational endpoint)   ·   "
                   f"committed pair detector confidence (geomean) {gmean:.3f} ≥ commit threshold {thr:.3f}")
        else:
            title = f"ABSTENTION FAILURE — {cid}, GT {rib_name(g, gl)}: correct eligible pair detected, but assignment emitted no 3D point"
            sub = (f"correct pair detector confidence (geomean) {gmean:.3f} < commit threshold {thr:.3f} (=1−2u): the frozen "
                   f"assignment abstains rather than commit a low-confidence edge — the residual bottleneck")
        fig.suptitle(title, fontsize=12.5, color=col, y=0.995)
        fig.text(0.5, 0.955, sub, ha="center", fontsize=9, color="#57606a")
        outp = a.out_dir / f"demo_{kind}_{cid}_iid{iid}.png"
        fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig)
        return outp

    outs = []; recorded = []
    for tag, payload in picks:
        if tag.startswith("success"):
            cid, row, iid, dist, m = payload
            outs.append(render(tag, cid, iid, row, dist, m))
            recorded.append({"kind": tag, "case": cid, "iid": int(iid), "accepted_row": int(row), "ap_idx": int(apx[row]),
                             "lat_idx": int(ltx[row]), "ap_score": round(float(ap_s[row]), 4), "lat_score": round(float(lt_s[row]), 4),
                             "dsi_vox": round(float(dsi_v[row]), 3), "dist_mm": round(dist, 2),
                             "rib_exact": bool(m["rib_exact"]), "rib_within1": bool(m["rib_within1"]), "along_mm": round(float(m["along_mm"]), 2)})
        else:
            cid, iid, row, fclass, pure = payload
            outs.append(render("failure", cid, iid, row, None, None))
            recorded.append({"kind": "failure", "failure_class": fclass, "pure_abstention": bool(pure), "case": cid, "iid": int(iid),
                             "correct_candidate_row": int(row), "ap_idx": int(apx[row]), "lat_idx": int(ltx[row]),
                             "ap_score": round(float(ap_s[row]), 4), "lat_score": round(float(lt_s[row]), 4), "dsi_vox": round(float(dsi_v[row]), 3)})

    summary = {"note": "POST-EVALUATION illustrative examples selected AFTER the sealed matched@10 was reproduced; not used for any selection.",
               "policy": {"gate": gate, "cost": cost, "mutual_best": mb, "u": u}, "data_sha256": data_sha,
               "expected_matched_at10": expected, "replayed_matched_at10": n_match,
               "extraction_policy": {"ap_nms": ap_nms, "ap_floor": ap_floor, "lat_nms": lat_nms, "lat_floor": lat_floor},
               "figures": [str(o) for o in outs], "demo_fractures": recorded}
    (a.out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(outs)} demo figures + demo_summary.json to {a.out_dir}")
    for o in outs: print(f"  {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
