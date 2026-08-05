#!/usr/bin/env python3
"""Det-frame addressing dataset for the DEPLOYABLE RibAssist 3D addressing model.

The original address_dataset.npz (extract_crops.py) lives in a DIFFERENT projection frame than the
champion detector, so an addressing model trained on it cannot be applied to detector-frame peaks
without silent crop misalignment. This builds an addressing dataset ENTIRELY in the detector's
canonical 256x256 frame:

  * crops from det_dev.npz's OWN AP/lateral projections, centered on the detector's stored GT centers
    (ap_ctr / lat_ctr), with EDGE PADDING (never clamping) so the fracture stays at the crop center —
    identical semantics to a crop at a PREDICTED peak at inference;
  * (side, rib, s) labels in the CANONICAL (RAS+) frame via the SHARED rib-labeling convention
    (rib_labeling.side_num_from_seg — the same convention build_rib_atlas.py / reconstruct_3d.py use),
    so an address "R7" here is the same rib slot the atlas reconstructs; s = nearest point on the
    apex-anchored canonical rib centerline (AP-axis anchor, atlas-consistent);
  * 2D coords in the 256 frame; AP horizontal is relative to an image-derived midline proxy (sign => side).

Output address_dataset_detframe.npz has the SAME keys train_address.py consumes
(ap, lat, ap_xy, lat_xy, side, rib, s, case, fclass). fclass is a -1 PLACEHOLDER (the real fracture
class is not wired here and the addressing trainer does not use it) — NOT a rib label.

Usage:
  python make_address_data_detframe.py --dev outputs/det_out_v2/det_dev.npz \
      --ribfrac-dir data/ribfrac_train data/ribfrac --seg-dir data/ribseg/ribseg_v2/seg \
      --cl-dir data/ribseg/ribseg_v2/cl --crop 96 --half 24 \
      --out outputs/det_out_v2/address_dataset_detframe.npz --overlays outputs/address_crop_overlays
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    from scipy.ndimage import zoom
    from make_rib_targets import canonical_voxels, _find
    from rib_labeling import side_num_from_seg
except Exception:  # noqa: BLE001
    nib = None

PROTOCOL_SIZE = 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def validate_center(r, c, height, width, name):
    """Guard before cropping: the fixed half-width padding only covers coordinates INSIDE the image,
    so a non-finite or out-of-bounds center is a hard error (would otherwise crop the wrong region)."""
    if not np.isfinite([r, c]).all():
        raise ValueError(f"{name}: non-finite center {(r, c)}")
    if not (0 <= r < height and 0 <= c < width):
        raise ValueError(f"{name}: center {(r, c)} outside image bounds [0,{height})x[0,{width})")


def crop_at(img, r, c, half, out):
    """Fixed crop centered at (r,c) via EDGE PADDING (never clamping) so the center pixel is always the
    requested coordinate — train-time GT centers and inference-time predicted peaks share semantics.
    Returns (crop[out,out] float16, padding_used: bool)."""
    r = int(round(r)); c = int(round(c)); H, W = img.shape
    padding_used = bool(r < half or r >= H - half or c < half or c >= W - half)
    fill = float(np.min(img))
    p = np.pad(img, ((half, half), (half, half)), mode="constant", constant_values=fill)
    rp, cp = r + half, c + half
    patch = p[rp - half:rp + half + 1, cp - half:cp + half + 1]
    if patch.shape != (2 * half + 1, 2 * half + 1):
        raise RuntimeError(f"unexpected crop shape {patch.shape} at coord {(r, c)}")
    return zoom(patch, (out / patch.shape[0], out / patch.shape[1]), order=1)[:out, :out].astype(np.float16), padding_used


def image_spine_col(ap_img):
    """IMAGE-ONLY midline PROXY column from the AP projection, so the spine-relative coordinate is
    reproducible at INFERENCE (no CT / no RibSeg). Attenuation-weighted column centroid AFTER
    subtracting a background floor (5th percentile), so constant padding/background cannot dominate.
    This is a midline PROXY (center of attenuation), NOT a validated spine detector — its agreement
    with the RibSeg midline is AUDITED on development data (manifest.midline_audit). Used identically
    at training and inference; only the side LABEL may use RibSeg."""
    x = np.asarray(ap_img, np.float64)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError("AP image must be a finite 2D array")
    floor = np.percentile(x, 5); w = np.clip(x - floor, 0.0, None)
    colmass = w.sum(axis=0); tot = float(colmass.sum())
    if tot <= 1e-12: return (x.shape[1] - 1) / 2.0
    return float(np.dot(np.arange(x.shape[1], dtype=np.float64), colmass) / tot)


def anchor_canonical(cl_can):
    """Apex-anchor a canonical centerline by AP axis (axis1 = world-y), matching extract_crops.py /
    build_rib_atlas so s aligns with the atlas."""
    if cl_can[:50, 1].mean() > cl_can[-50:, 1].mean(): return cl_can[::-1]
    return cl_can


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--image-dirs", "--ribfrac-dir", dest="image_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--seg-dir", type=Path, required=True); ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--crop", type=int, default=96); ap.add_argument("--half", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--overlays", type=Path, default=None)
    ap.add_argument("--n-overlay", type=int, default=20)
    a = ap.parse_args()
    if nib is None: print("pip install nibabel scipy matplotlib", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")

    dev_sha = sha256_file(a.dev); d = np.load(a.dev, allow_pickle=False)
    cases = [str(c) for c in d["case"]]; S = int(d["ap"].shape[-1])
    if S != PROTOCOL_SIZE: raise ValueError(f"det_dev images {S}px != {PROTOCOL_SIZE}")
    ap_geo = d["ap_geo"]; fp_case, fp_iid = d["fp_case"], d["fp_iid"]; ap_ctr, lat_ctr = d["ap_ctr"], d["lat_ctr"]
    per_case = {}
    for j in range(len(fp_case)): per_case.setdefault(int(fp_case[j]), []).append(j)
    requested = int(len(fp_case))

    APc, LATc, side, rib, Sv, case_out, fcls, APxy, LATxy = [], [], [], [], [], [], [], [], []
    audit = {"requested": requested, "affine_mismatch": 0, "no_rib_overlap": 0, "rib_out_of_range": 0,
             "label_not_in_volume": 0, "missing_inputs": 0, "padding_used": 0}
    spine_audit = []; sign_agree = []   # training-only midline audit: image proxy vs RibSeg reference
    sign_records = []                   # per-fracture {distance-from-reference-midline, side-sign-agree} for stratified audit
    side_error_records = []             # per-fracture {true side, signed proxy error} for side-stratified midline error
    skipped = {}
    print(f"building det-frame address crops for {len(per_case)} positive cases ({requested} instances) ...",
          file=sys.stderr, flush=True)
    for ci, cid in enumerate(cases):
        if ci not in per_case: continue
        segp = _find([a.seg_dir], f"{cid}-rib-seg.nii.gz", f"{cid}.nii.gz", f"{cid}*.nii.gz", f"{cid}*.nii")
        imp = _find(a.image_dirs, f"{cid}-label.nii.gz", f"{cid}-label.nii", f"{cid}-label.nii*")
        clp = a.cl_dir / f"{cid}.npz"
        if segp is None or imp is None or not clp.exists():
            skipped[cid] = "missing seg/label/cl"; audit["missing_inputs"] += len(per_case[ci]); continue
        lab_nii = nib.load(str(imp)); A_o = lab_nii.affine
        fl_c_nii = nib.as_closest_canonical(lab_nii); rs_c_nii = nib.as_closest_canonical(nib.load(str(segp)))
        A_c = fl_c_nii.affine
        if fl_c_nii.shape != rs_c_nii.shape:
            skipped[cid] = "seg/label shape mismatch"; audit["affine_mismatch"] += len(per_case[ci]); continue
        if not np.allclose(fl_c_nii.affine, rs_c_nii.affine, rtol=0, atol=1e-4):
            skipped[cid] = "seg/label canonical affine mismatch"; audit["affine_mismatch"] += len(per_case[ci]); continue
        fl_can = np.asarray(fl_c_nii.get_fdata()).astype(np.int32); rs_can = np.asarray(rs_c_nii.get_fdata()).astype(np.int32)
        info, spine_lr_ref = side_num_from_seg(rs_can, lr_axis=0, si_axis=2)   # LABELS + AUDIT-reference midline only
        cl = np.load(clp)["cl"]; R = cl.shape[0]
        cl_can = [canonical_voxels(cl[r].astype(np.float64), A_o, A_c) for r in range(R)]
        ap_img = d["ap"][ci].astype(np.float32); lat_img = d["lat"][ci].astype(np.float32)
        sa, pta, pla = map(float, ap_geo[ci])
        spine_256 = image_spine_col(ap_img)                # DEPLOYABLE coordinate (image-only, inference-reproducible)
        ribseg_spine_256 = spine_lr_ref * sa + pla         # training-only AUDIT reference (NOT a feature)
        spine_audit.append((cid, float(spine_256), float(ribseg_spine_256)))
        for j in per_case[ci]:
            lb = int(fp_iid[j]); vox = np.array(np.nonzero(fl_can == lb))
            if vox.size == 0: audit["label_not_in_volume"] += 1; continue
            fc = vox.mean(1)
            at = rs_can[vox[0], vox[1], vox[2]]; nz = at[at != 0]
            if nz.size == 0: audit["no_rib_overlap"] += 1; continue
            rl = int(np.bincount(nz).argmax())
            if rl not in info or not (1 <= rl <= R): audit["rib_out_of_range"] += 1; continue
            clc = anchor_canonical(cl_can[rl - 1])
            s = int(np.linalg.norm(clc - fc[None], axis=1).argmin()) / (len(clc) - 1)
            ar, ac = float(ap_ctr[j][0]), float(ap_ctr[j][1]); lrr, lcc = float(lat_ctr[j][0]), float(lat_ctr[j][1])
            # side classification uses the SAME >= convention as the labels (equality -> Right), so the audit
            # measures exactly the operational side decision (avoids np.sign(0)==0's spurious third state)
            proxy_side_right = ac >= spine_256
            reference_side_right = ac >= ribseg_spine_256
            agree = bool(proxy_side_right == reference_side_right)
            sign_agree.append(agree)
            sign_records.append({"distance": float(abs(ac - ribseg_spine_256)), "agree": agree})
            side_error_records.append({"side": info[rl]["side"], "error": float(spine_256 - ribseg_spine_256)})
            validate_center(ar, ac, S, S, f"{cid}#{lb} AP"); validate_center(lrr, lcc, S, S, f"{cid}#{lb} lat")
            ca, pad_a = crop_at(ap_img, ar, ac, a.half, a.crop); cl2, pad_l = crop_at(lat_img, lrr, lcc, a.half, a.crop)
            if pad_a or pad_l: audit["padding_used"] += 1
            APc.append(ca); LATc.append(cl2)
            APxy.append([ar / S, (ac - spine_256) / S]); LATxy.append([lrr / S, lcc / S])
            side.append(0 if info[rl]["side"] == "L" else 1); rib.append(int(info[rl]["num"])); Sv.append(float(s))
            case_out.append(cid); fcls.append(-1)   # honest placeholder: NOT a fracture class, NOT a rib label
        if (ci + 1) % 50 == 0: print(f"[{ci+1}/{len(cases)}] crops so far {len(APc)}", file=sys.stderr, flush=True)

    if not APc: raise RuntimeError("No crops produced — check --seg-dir / --cl-dir / --ribfrac-dir.")
    rib = np.array(rib); Sv = np.array(Sv, np.float32); APxy = np.array(APxy, np.float32); LATxy = np.array(LATxy, np.float32)
    side = np.array(side)
    # ---- hard invariants before writing the definitive dataset ----
    assert set(int(x) for x in np.unique(rib)).issubset(set(range(1, 13))), "rib labels outside 1..12"
    assert np.isfinite(Sv).all() and np.all((Sv >= 0) & (Sv <= 1)), "s out of [0,1] or non-finite"
    assert np.isfinite(APxy).all() and np.isfinite(LATxy).all(), "non-finite 2D coords"
    assert set(int(x) for x in np.unique(side)).issubset({0, 1}), "side not in {0,1}"
    # accounting identity: every requested instance is either a crop or attributed to exactly one skip reason
    accounted = len(APc) + audit["missing_inputs"] + audit["affine_mismatch"] + audit["label_not_in_volume"] \
        + audit["no_rib_overlap"] + audit["rib_out_of_range"]
    assert accounted == requested, f"accounting mismatch: crops+skips {accounted} != requested {requested}"

    # ---- training-only midline audit: image proxy vs RibSeg reference (RibSeg is audit-only here) ----
    img_s = np.array([x[1] for x in spine_audit]); ref_s = np.array([x[2] for x in spine_audit])
    signed = img_s - ref_s; absr = np.abs(signed)
    worst = sorted(spine_audit, key=lambda x: -abs(x[1] - x[2]))[:10]
    # error DIRECTION: how often, and how far, is the image proxy left/right of the RibSeg reference? (this
    # is by error sign, NOT by anatomical side — a case-level description of which way the proxy displaces)
    neg = signed[signed < 0]; pos = signed[signed > 0]
    error_direction = {
        "fraction_proxy_left_of_reference": round(float((signed < 0).mean()), 4) if len(signed) else None,
        "mean_negative_error_px": round(float(neg.mean()), 2) if neg.size else None,
        "mean_positive_error_px": round(float(pos.mean()), 2) if pos.size else None}
    # error by TRUE fracture side: does the proxy behave differently for anatomically left- vs right-sided
    # fractures? (fracture-weighted: repeated per fracture in a case, appropriate for operational side risk;
    # the case-level `signed` stats above remain the unbiased case-level midline-accuracy description)
    left_errors = np.array([x["error"] for x in side_error_records if x["side"] == "L"])
    right_errors = np.array([x["error"] for x in side_error_records if x["side"] == "R"])
    error_by_true_side = {
        sd: {"n": int(e.size),
             "mean_signed_error_px": round(float(e.mean()), 2) if e.size else None,
             "median_abs_error_px": round(float(np.median(np.abs(e))), 2) if e.size else None}
        for sd, e in (("L", left_errors), ("R", right_errors))}
    # side-sign agreement STRATIFIED by true distance from the reference midline: overall agreement is
    # dominated by far-from-midline fractures (which are trivially on the correct side); the fractures that
    # can actually flip side are the near-midline ones. Report agreement AND sample count per band.
    dist = np.array([r["distance"] for r in sign_records]); agr = np.array([r["agree"] for r in sign_records], bool)

    def agreement_within(limit):
        m = dist <= limit if limit is not None else np.ones(len(dist), bool)
        n = int(m.sum())
        return {"n": n, "agreement": round(float(agr[m].mean()), 4) if n else None}

    midline_audit = {"n_cases": len(spine_audit),
                     "mean_signed_error_px": round(float(signed.mean()), 2),
                     "median_abs_error_px": round(float(np.median(absr)), 2),
                     "p90_abs_error_px": round(float(np.percentile(absr, 90)), 2),
                     "p95_abs_error_px": round(float(np.percentile(absr, 95)), 2),
                     "frac_within_px": {p: round(float((absr <= p).mean()), 4) for p in (3, 5, 10, 15)},
                     "side_sign_agreement": round(float(np.mean(sign_agree)), 4) if sign_agree else None,
                     "error_direction": error_direction,
                     "error_by_true_fracture_side": error_by_true_side,
                     "side_sign_agreement_by_distance": {
                         "all": agreement_within(None), "within_5px": agreement_within(5),
                         "within_10px": agreement_within(10), "within_20px": agreement_within(20)},
                     "corr": round(float(np.corrcoef(img_s, ref_s)[0, 1]), 4) if len(img_s) > 1 else None,
                     "worst10": [{"case": c, "image": round(i, 1), "ribseg": round(r, 1), "abs_err": round(abs(i - r), 1)}
                                 for c, i, r in worst]}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, ap=np.array(APc), lat=np.array(LATc), side=side, rib=rib, s=Sv,
                        case=np.array(case_out), fclass=np.array(fcls, np.int64), ap_xy=APxy, lat_xy=LATxy)
    rib_hist = {int(k): int((rib == k).sum()) for k in range(1, 13)}
    s_pct = [round(float(np.percentile(Sv, p)), 3) for p in (5, 50, 95)]
    man = {"address_frame": "detector canonical 256 (crops from det_dev AP/lat at stored GT centers; labels canonical)",
           "aligned_to": {"det_dev": str(a.dev), "det_dev_sha256": dev_sha},
           "crop": a.crop, "half": a.half, "rib_labeling": "shared rib_labeling.side_num_from_seg (RAS-anchored, matches build_rib_atlas)",
           "ap_xy_horizontal": "AP-image attenuation centroid used as an inference-reproducible midline proxy; "
                               "agreement with the training-only RibSeg reference audited on development data (midline_audit). "
                               "RibSeg is used only for the side LABEL, never as an inference feature.",
           "midline_audit": midline_audit,
           "fclass": "-1 placeholder (NOT wired; addressing trainer does not use it)",
           "counts": {"crops": len(APc), "cases": len(set(case_out)), "requested_instances": requested,
                      "skipped_instances": requested - len(APc), **audit},
           "distribution": {"side_L": int((side == 0).sum()), "side_R": int((side == 1).sum()),
                            "rib_histogram": rib_hist, "s_p5_p50_p95": s_pct},
           "skipped_cases": skipped, "address_data_sha256": sha256_file(a.out),
           "software": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__}}
    (a.out.with_name(a.out.stem + "_manifest.json")).write_text(json.dumps(man, indent=2))

    print(f"\n{'fracture instances requested:':34}{requested}")
    print(f"{'crops produced:':34}{len(APc)}")
    print(f"{'instances skipped:':34}{requested - len(APc)}")
    print(f"{'cases represented:':34}{len(set(case_out))}")
    print(f"{'side L/R:':34}{int((side==0).sum())}/{int((side==1).sum())}")
    print(f"{'rib 1-12 histogram:':34}{rib_hist}")
    print(f"{'s median (p5,p95):':34}{s_pct[1]} ({s_pct[0]}, {s_pct[2]})")
    print(f"{'crops requiring padding:':34}{audit['padding_used']}")
    print(f"{'label/seg affine mismatches:':34}{audit['affine_mismatch']}")
    print(f"{'fractures w/ no RibSeg overlap:':34}{audit['no_rib_overlap']}")
    print(f"\n--- MIDLINE AUDIT (image proxy vs RibSeg reference; training-only) ---")
    print(f"{'mean signed error (px):':34}{midline_audit['mean_signed_error_px']}")
    print(f"{'median abs error (px):':34}{midline_audit['median_abs_error_px']}")
    print(f"{'p90 / p95 abs error (px):':34}{midline_audit['p90_abs_error_px']} / {midline_audit['p95_abs_error_px']}")
    print(f"{'within 3/5/10/15 px:':34}{midline_audit['frac_within_px']}")
    print(f"{'fracture side-sign agreement:':34}{midline_audit['side_sign_agreement']}")
    ed = midline_audit["error_direction"]
    print(f"{'proxy left-of-reference frac:':34}{ed['fraction_proxy_left_of_reference']}  "
          f"(mean err <0 {ed['mean_negative_error_px']} px / >0 {ed['mean_positive_error_px']} px)")
    ets = midline_audit["error_by_true_fracture_side"]
    print(f"{'error by true side (fx-wt):':34}"
          f"L mean {ets['L']['mean_signed_error_px']} / med|.| {ets['L']['median_abs_error_px']} (n={ets['L']['n']}) | "
          f"R mean {ets['R']['mean_signed_error_px']} / med|.| {ets['R']['median_abs_error_px']} (n={ets['R']['n']})")
    sd = midline_audit["side_sign_agreement_by_distance"]
    print(f"{'side-sign agree by dist:':34}"
          f"all {sd['all']['agreement']} (n={sd['all']['n']}) | "
          f"<=5px {sd['within_5px']['agreement']} (n={sd['within_5px']['n']}) | "
          f"<=10px {sd['within_10px']['agreement']} (n={sd['within_10px']['n']}) | "
          f"<=20px {sd['within_20px']['agreement']} (n={sd['within_20px']['n']})")
    print(f"{'worst case abs err (px):':34}{midline_audit['worst10'][0]['abs_err']} ({midline_audit['worst10'][0]['case']})")
    print(f"wrote {a.out}")

    if a.overlays is not None:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            print(f"[overlay skip] {e}", file=sys.stderr); return 0
        a.overlays.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(0); pick = rng.choice(len(APc), size=min(len(APc), a.n_overlay), replace=False)
        for k in pick:
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.7))
            for x, crop, lab in ((ax[0], APc[k], "AP"), (ax[1], LATc[k], "lat")):
                x.imshow(crop.astype(np.float32), cmap="gray", origin="lower")
                x.plot(a.crop / 2, a.crop / 2, "r+", ms=12, mew=2)   # crop center = the (unshifted) detector coordinate
                x.set_title(lab, fontsize=9); x.axis("off")
            fig.suptitle(f"{case_out[k]} -> {'L' if side[k]==0 else 'R'}{rib[k]} s={Sv[k]:.2f}", fontsize=10)
            fig.tight_layout(); fig.savefig(a.overlays / f"crop_{k:04d}_{case_out[k]}.png", dpi=90); plt.close(fig)
        print(f"wrote {len(pick)} crop overlays to {a.overlays}/ — verify each crop is centered on a fracture (red +) "
              "and the L/R rib s label is anatomically sensible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
