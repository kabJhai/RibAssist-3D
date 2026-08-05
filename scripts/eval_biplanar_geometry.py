#!/usr/bin/env python3
"""STAGE A of the biplanar 3D-reconstruction test — the geometry gate for the central RibAssist 3D claim
(a fracture seen in both orthographic projections can be triangulated to a 3D location). NO learned model
and NO detector: pure orthographic back-projection, scored against GT geometry in WORLD millimetres.

It runs TWO explicitly separate oracle tests so a target-selection artifact cannot masquerade as a
geometry failure:

  A0 — geometric round-trip oracle: forward-project the TRUE 3D fracture centroid into AP+lateral with the
       stored transforms, back-project, compare to the original. Must be ~numerical zero; validates the
       transform implementation + metadata.
  A1 — detector-target oracle: back-project the STORED GT AP+lateral centers (ap_ctr / lat_ctr — the
       distance-transform interiors of the projected footprints, i.e. the detector's own targets) under
       ORACLE correspondence (same instance), and compare to the fracture centroid AND to the nearest
       voxel anywhere in the true fracture mask. This measures the floor from the target definition.

Geometry (orthographic, canonical RAS+): AP=(row=SI, col=LR), lateral=(row=SI, col=AP), shared SI.
Back-projection to canonical voxel (lr, ap, si) with ap_geo=[sa,pta,pla], lat_geo=[sl,ptl,pll]:
    LR=(ap_col-pla)/sa, AP=(lat_col-pll)/sl, SI=(ap_row-pta)/sa  (lat gives SI too; disagreement is the
    correspondence basis). ALL distances are computed in WORLD mm via the canonical NIfTI affine
    (nibabel.affines.apply_affine), NOT reconstructed from ap_sp/lat_sp — those are only AUDITED against
    the affine's voxel sizes.

Reports (over all development fractures): A0 error; A1 3D error to centroid AND to fracture mask (mean/
median/p90/p95 + fractions within 5/10/15/20/30 mm); per-axis LR/AP/SI error; inter-view SI disagreement;
ANATOMICAL nearest-rib exact/±1 (side+number via side_num_from_seg, never raw label arithmetic); distance
to the correct rib centerline; along-rib normalized-s and mm error; and the worst cases for inspection.

Usage:
  python eval_biplanar_geometry.py --data outputs/det_out_v2/det_dev.npz \
      --image-dirs data/ribfrac_train data/ribfrac \
      --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
      --out outputs/eval_biplanar_geometry_stageA.json
"""
from __future__ import annotations
import argparse, json, sys
from itertools import product
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    from nibabel.affines import apply_affine
    from make_address_data_detframe import anchor_canonical
    from make_rib_targets import canonical_voxels, _find
    from rib_labeling import side_num_from_seg
except Exception:  # noqa: BLE001
    nib = None


def back_project(ap_ctr, lat_ctr, ap_geo, lat_geo):
    """Orthographic back-projection of two 2D centers to a canonical voxel point (lr, ap, si).
    Returns (point, si_disagreement_voxels)."""
    sa, pta, pla = [float(x) for x in ap_geo]; sl, ptl, pll = [float(x) for x in lat_geo]
    (row_ap, col_ap), (row_lat, col_lat) = ap_ctr, lat_ctr
    si_ap = (row_ap - pta) / sa; si_lat = (row_lat - ptl) / sl
    return np.array([(col_ap - pla) / sa, (col_lat - pll) / sl, 0.5 * (si_ap + si_lat)], np.float64), abs(si_ap - si_lat)


def forward_project(fc, ap_geo, lat_geo):
    """Forward-project a canonical voxel point (lr, ap, si) into AP (row=SI,col=LR) and lateral
    (row=SI,col=AP) 2D centers using the stored transforms — the inverse of back_project."""
    sa, pta, pla = [float(x) for x in ap_geo]; sl, ptl, pll = [float(x) for x in lat_geo]
    lr, apx, si = fc
    return (si * sa + pta, lr * sa + pla), (si * sl + ptl, apx * sl + pll)


def centerline_position(pt_world, cl_world):
    """Position of the nearest centerline vertex to `pt_world`, using CUMULATIVE world-space arc length
    (no uniform-spacing assumption). Returns (idx, normalized_s, arc_length_mm, distance_mm)."""
    seg = np.linalg.norm(np.diff(cl_world, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    dists = np.linalg.norm(cl_world - pt_world[None], axis=1)
    idx = int(dists.argmin()); total = float(cum[-1])
    return idx, (float(cum[idx] / total) if total > 0 else 0.0), float(cum[idx]), float(dists[idx])


def fracture_metrics(p_rec, si_dis_vox, vox, g, cid, iid):
    """Post-triangulation metrics for ONE reconstructed 3D point against a GT fracture instance, in WORLD
    mm. Rib side/number and along-rib position are DERIVED here from the reconstructed coordinate + GT
    geometry (never predicted before reconstruction). Shared by Stage A (oracle centers) and Stage B
    (detector-predicted centers) so the two stages differ ONLY in the center source, not the scoring.

    `vox` is the (3, N) canonical-voxel group of the true fracture; `g` is a case_gt() dict. Returns
    (metrics_dict, None) on success, or (None, skip_reason) if the instance has no rib overlap or maps to
    an out-of-range rib label."""
    aff = g["aff"]; fc = vox.mean(1)
    at = g["rs"][vox[0], vox[1], vox[2]]; nz = at[at != 0]
    if nz.size == 0: return None, "no_rib_overlap"
    rl = int(np.bincount(nz).argmax())
    if rl not in g["info"] or not (1 <= rl <= g["R"]): return None, "rib_out_of_range"
    fc_w = apply_affine(aff, fc); p_w = apply_affine(aff, p_rec)
    centroid_mm = float(np.linalg.norm(p_w - fc_w))
    mask_w = apply_affine(aff, vox.T.astype(np.float64))              # every fracture voxel CENTER -> world
    mask_center_mm = float(np.linalg.norm(mask_w - p_w[None], axis=1).min())
    # LOWER bound on distance to the occupied voxel VOLUME: nearest voxel-center distance minus the max
    # center-to-corner radius, floored at 0. Zero does NOT prove the point lies inside the mask.
    mask_volume_lower_bound_mm = max(0.0, mask_center_mm - g["corner_radius"])
    err_vec = p_w - fc_w
    dists = {lbl: float(np.linalg.norm(g["cl_world"][lbl - 1] - p_w[None], axis=1).min()) for lbl in g["info"]}
    near_lbl = min(dists, key=dists.get)
    gt_sd, gt_nm = g["info"][rl]["side"], int(g["info"][rl]["num"])
    pr_sd, pr_nm = g["info"][near_lbl]["side"], int(g["info"][near_lbl]["num"])
    gt_cl_w = g["cl_world"][rl - 1]
    _, s_gt, arc_gt, _ = centerline_position(fc_w, gt_cl_w)           # cumulative arc length, no uniform assumption
    _, s_rec, arc_rec, dcl = centerline_position(p_w, gt_cl_w)
    return {"case": cid, "iid": iid, "centroid_mm": centroid_mm,
            "mask_center_mm": mask_center_mm, "mask_volume_lower_bound_mm": mask_volume_lower_bound_mm,
            "lr_mm": abs(float(err_vec[0])), "ap_mm": abs(float(err_vec[1])), "si_mm": abs(float(err_vec[2])),
            "si_disagree_mm": float(si_dis_vox * g["vsz"][2]),
            "rib_exact": (pr_sd == gt_sd and pr_nm == gt_nm),
            "rib_within1": (pr_sd == gt_sd and abs(pr_nm - gt_nm) <= 1),
            "dist_correct_cl_mm": dcl, "s_err": abs(s_rec - s_gt), "along_mm": abs(arc_rec - arc_gt)}, None


def case_gt(cid, image_dirs, seg_dir, cl_dir):
    """Canonical-frame GT for a case: fracture label volume, rib-seg, label->{side,num} map, apex-anchored
    rib centerlines (voxel + world), the affine, and its per-axis voxel sizes. None on missing inputs."""
    segp = _find([seg_dir], f"{cid}-rib-seg.nii.gz", f"{cid}.nii.gz", f"{cid}*.nii.gz", f"{cid}*.nii")
    imp = _find(image_dirs, f"{cid}-label.nii.gz", f"{cid}-label.nii", f"{cid}-label.nii*")
    clp = Path(cl_dir) / f"{cid}.npz"
    if segp is None or imp is None or not clp.exists(): return None
    lab_nii = nib.load(str(imp)); A_o = lab_nii.affine
    fl_c = nib.as_closest_canonical(lab_nii); rs_c = nib.as_closest_canonical(nib.load(str(segp)))
    if fl_c.shape != rs_c.shape or not np.allclose(fl_c.affine, rs_c.affine, rtol=0, atol=1e-4): return None
    aff = fl_c.affine
    # native-dtype label reads (dataobj, not get_fdata) — these are integer masks; get_fdata would
    # materialize a float64 copy (~2x-8x larger) of each full CT-sized volume
    fl_can = np.asarray(fl_c.dataobj).astype(np.int32); rs_can = np.asarray(rs_c.dataobj).astype(np.int32)
    # rib info from a SINGLE nonzero pass over the rib-seg (NOT 24 full-volume scans). Uses the shared
    # assign_side_num convention (lr=axis0, si=axis2), identical result to side_num_from_seg.
    from rib_labeling import assign_side_num
    rnz = np.array(np.nonzero(rs_can))
    if rnz.size == 0: return None
    rlab = rs_can[rnz[0], rnz[1], rnz[2]]; spine_lr = float(rnz[0].mean())
    ulabs = [int(x) for x in np.unique(rlab)]
    cent = {lb: rnz[:, rlab == lb].mean(1) for lb in ulabs}
    sn = assign_side_num([cent[lb][0] for lb in ulabs], [cent[lb][2] for lb in ulabs], spine_lr, keys=ulabs)
    info = {lb: {"c": cent[lb], "side": sn[lb][0], "num": sn[lb][1]} for lb in ulabs}
    # fracture-voxel groups from a SINGLE nonzero pass over the label volume (NOT one scan per instance)
    fnz = np.array(np.nonzero(fl_can))
    if fnz.size:
        flab = fl_can[fnz[0], fnz[1], fnz[2]]
        fl_groups = {int(lb): fnz[:, flab == lb] for lb in np.unique(flab)}
    else:
        fl_groups = {}
    cl = np.load(clp)["cl"]; R = cl.shape[0]
    # PROVE, don't assume, the rib-label -> centerline-slot relationship (cl_world[lbl-1]) before using it
    for lbl in info:
        if not 1 <= int(lbl) <= R:
            raise ValueError(f"{cid}: RibSeg label {lbl} has no corresponding centerline slot among 1..{R}")
    cl_vox = [anchor_canonical(canonical_voxels(cl[r].astype(np.float64), A_o, aff)) for r in range(R)]
    cl_world = [apply_affine(aff, c) for c in cl_vox]
    vsz = nib.affines.voxel_sizes(aff)   # mm/voxel along canonical lr, ap, si (for the ap_sp/lat_sp audit)
    # max center-to-corner radius of one voxel, from the affine's linear part (robust to obliquity)
    lin = np.asarray(aff[:3, :3], np.float64)
    corner_radius = float(max(np.linalg.norm(lin @ np.array(c, np.float64)) for c in product((-0.5, 0.5), repeat=3)))
    return {"fl_groups": fl_groups, "rs": rs_can, "info": info, "cl_world": cl_world, "R": R, "aff": aff,
            "vsz": vsz, "corner_radius": corner_radius, "n_ribs": len(info)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if nib is None: print("pip install nibabel scipy", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    d = np.load(a.data, allow_pickle=False)
    cases = [str(c) for c in d["case"]]
    ap_ctr, lat_ctr = d["ap_ctr"], d["lat_ctr"]; ap_geo, lat_geo = d["ap_geo"], d["lat_geo"]
    ap_sp, lat_sp = d["ap_sp"], d["lat_sp"]; fp_case, fp_iid = d["fp_case"], d["fp_iid"]
    per_case = {}
    for j in range(len(fp_case)): per_case.setdefault(int(fp_case[j]), []).append(j)

    A0, A1 = [], []
    audit_sp = []   # (ap_sp/lat_sp) vs affine voxel-size discrepancies (mm/pixel), an audit only
    vox_dims = []; rib_counts = []   # per-case affine voxel dims + count of valid anatomical rib labels
    skipped = {"missing_inputs": 0, "label_absent": 0, "no_rib_overlap": 0, "rib_out_of_range": 0}
    n_cases = len(per_case); done = 0
    print(f"Stage A over {len(fp_case)} fractures in {n_cases} cases (A0 round-trip + A1 stored-target oracle).",
          flush=True)
    print("  loading + reconstructing per case (each case reads 2 CT-derived NIfTIs — this is the slow part) ...",
          flush=True)

    for ci, cid in enumerate(cases):
        if ci not in per_case: continue
        done += 1
        if done % 25 == 0 or done == n_cases:
            print(f"  [{done}/{n_cases}] cases | {len(A1)} fractures reconstructed so far", flush=True)
        # load THIS case only; do NOT cache (each case is processed exactly once — caching all cases' full
        # label + seg volumes was an OOM leak). g is dropped when the next iteration rebinds it.
        g = case_gt(cid, a.image_dirs, a.seg_dir, a.cl_dir)
        if g is not None:
            sa, sl = float(ap_geo[ci][0]), float(lat_geo[ci][0]); vsz = g["vsz"]
            audit_sp.append(max(abs(float(ap_sp[ci][1]) * sa - vsz[0]), abs(float(lat_sp[ci][1]) * sl - vsz[1]),
                                abs(float(ap_sp[ci][0]) * sa - vsz[2])))
            vox_dims.append([round(float(x), 3) for x in vsz]); rib_counts.append(g["n_ribs"])
        for j in per_case[ci]:
            if g is None: skipped["missing_inputs"] += 1; continue
            lb = int(fp_iid[j]); vox = g["fl_groups"].get(lb)   # precomputed voxel group (no per-instance scan)
            if vox is None or vox.shape[1] == 0: skipped["label_absent"] += 1; continue
            # ---- A1: back-project the STORED GT AP+lat centers under oracle correspondence ----
            p_rec, si_dis_vox = back_project(ap_ctr[j], lat_ctr[j], ap_geo[ci], lat_geo[ci])
            m, reason = fracture_metrics(p_rec, si_dis_vox, vox, g, cid, lb)
            if m is None: skipped[reason] += 1; continue
            A1.append(m)
            # ---- A0: forward-project the TRUE centroid then back-project (transform round-trip) ----
            # Same passing set as A1 (instances with a valid anatomical rib); independent of the target.
            fc = vox.mean(1); aff = g["aff"]; fc_w = apply_affine(aff, fc)
            ap0, lat0 = forward_project(fc, ap_geo[ci], lat_geo[ci])
            p0, _ = back_project(ap0, lat0, ap_geo[ci], lat_geo[ci])
            A0.append(float(np.linalg.norm(apply_affine(aff, p0) - fc_w)))

    if not A1: raise RuntimeError("no reconstructable fractures — check inputs")

    def stat(vals):
        v = np.asarray(vals, np.float64)
        return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                "p90": round(float(np.percentile(v, 90)), 2), "p95": round(float(np.percentile(v, 95)), 2)}
    def within(vals, ts=(5, 10, 15, 20, 30)):
        v = np.asarray(vals, np.float64); return {f"{t}mm": round(float((v <= t).mean()), 4) for t in ts}
    cen = [r["centroid_mm"] for r in A1]; mc = [r["mask_center_mm"] for r in A1]
    mv = [r["mask_volume_lower_bound_mm"] for r in A1]
    worst = sorted(A1, key=lambda r: -r["mask_center_mm"])[:15]
    uniq_dims = sorted({tuple(v) for v in vox_dims})

    result = {
        "n_fractures": len(A1), "n_skipped": skipped,
        "geometry_audit": {
            "A0_transform_roundtrip_error_mm": {**stat(A0), "max": round(float(np.max(A0)), 6),
                "interpretation": "must be ~0; validates the projection transforms + metadata implementation"},
            "ap_sp_latsp_vs_affine_voxelsize_max_discrepancy_mm_per_pixel": round(float(np.max(audit_sp)), 4) if audit_sp else None,
            "affine_voxel_dims_mm_unique": [list(u) for u in uniq_dims[:10]],
            "valid_anatomical_rib_labels_per_case": {"min": int(min(rib_counts)), "max": int(max(rib_counts)),
                                                     "mean": round(float(np.mean(rib_counts)), 1)} if rib_counts else None},
        "A1_stored_target_oracle": {
            "distance_to_centroid_mm": {**stat(cen), "within": within(cen)},
            "distance_to_nearest_fracture_voxel_center_mm": {**stat(mc), "within": within(mc),
                "interpretation": "min distance to any fracture VOXEL CENTER — did triangulation land on the fracture volume? "
                                  "(distance is to voxel centers; a point inside an occupied voxel can still be up to ~half a "
                                  "voxel diagonal away)"},
            "distance_to_fracture_volume_lower_bound_mm": {**stat(mv), "within": within(mv),
                "interpretation": "LOWER bound on distance to the occupied fracture-voxel volume: nearest voxel-center "
                                  "distance minus the max center-to-corner radius, floored at 0. A value of zero does "
                                  "NOT prove the reconstructed point lies inside the fracture mask."},
            "component_error_mm": {"LR": stat([r["lr_mm"] for r in A1]), "AP": stat([r["ap_mm"] for r in A1]),
                                   "SI": stat([r["si_mm"] for r in A1])},
            "inter_view_SI_disagreement_mm": stat([r["si_disagree_mm"] for r in A1]),
            "anatomical_nearest_rib_exact": round(float(np.mean([r["rib_exact"] for r in A1])), 4),
            "anatomical_nearest_rib_within1": round(float(np.mean([r["rib_within1"] for r in A1])), 4),
            "distance_to_correct_rib_centerline_mm": stat([r["dist_correct_cl_mm"] for r in A1]),
            "along_rib_normalized_s_error": stat([r["s_err"] for r in A1]),
            "along_rib_arc_length_error_mm": stat([r["along_mm"] for r in A1]),
            "worst_by_nearest_fracture_voxel_distance": [{"case": w["case"], "iid": w["iid"],
                                        "mask_center_mm": round(w["mask_center_mm"], 1),
                                        "mask_volume_lower_bound_mm": round(w["mask_volume_lower_bound_mm"], 1),
                                        "centroid_mm": round(w["centroid_mm"], 1),
                                        "si_disagree_mm": round(w["si_disagree_mm"], 1)} for w in worst]},
        "note": "A0 isolates the transform math; A1 adds the detector-target definition (stored centers are DT "
                "interiors of the projected footprints, not exact projected centroids). Distances are WORLD mm via "
                "the canonical NIfTI affine; ap_sp/lat_sp are audited, not used, for distance. Along-rib uses "
                "cumulative arc length. Fracture distance is to voxel CENTERS (+ a conservative lower bound on "
                "distance to the voxelized fracture volume; the lower bound reaching 0 does not prove overlap).",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    au = result["geometry_audit"]; A = au["A0_transform_roundtrip_error_mm"]; B = result["A1_stored_target_oracle"]
    mc_ = B["distance_to_nearest_fracture_voxel_center_mm"]; mv_ = B["distance_to_fracture_volume_lower_bound_mm"]
    print(f"\nSTAGE A over {len(A1)} development fractures ({sum(skipped.values())} skipped)")
    print(f"  A0 transform round-trip error mm: mean {A['mean']} | max {A['max']}   (must be ~0)")
    print(f"  ap_sp/lat_sp vs affine audit (max mm/pixel discrepancy): {au['ap_sp_latsp_vs_affine_voxelsize_max_discrepancy_mm_per_pixel']}"
          f"  | rib labels/case {au['valid_anatomical_rib_labels_per_case']}")
    print(f"  A1 dist to CENTROID mm: mean {B['distance_to_centroid_mm']['mean']} | median {B['distance_to_centroid_mm']['median']} "
          f"| p90 {B['distance_to_centroid_mm']['p90']}")
    print(f"  A1 nearest fracture VOXEL-CENTER mm: median {mc_['median']} | p90 {mc_['p90']} | within {mc_['within']}")
    print(f"  A1 fracture-volume distance LOWER BOUND mm: median {mv_['median']} | p90 {mv_['p90']} | within {mv_['within']}")
    print(f"  component mm (mean): LR {B['component_error_mm']['LR']['mean']} | AP {B['component_error_mm']['AP']['mean']} | SI {B['component_error_mm']['SI']['mean']}")
    print(f"  inter-view SI disagreement mm (mean/median): {B['inter_view_SI_disagreement_mm']['mean']} / {B['inter_view_SI_disagreement_mm']['median']}")
    print(f"  anatomical nearest-rib exact {B['anatomical_nearest_rib_exact']} | rib±1 {B['anatomical_nearest_rib_within1']}")
    print(f"  along-rib error: s {B['along_rib_normalized_s_error']['mean']} | arc-mm {B['along_rib_arc_length_error_mm']['mean']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
