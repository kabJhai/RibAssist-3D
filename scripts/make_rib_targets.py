#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Rib-region AUXILIARY training target for the Phase-2 multi-task detector — a TRAINING-ONLY
side-car for det_dev.npz, aligned to its EXACT images via each case's stored ap_geo/lat_geo (the
frozen det_dev.npz is left byte-identical; the sealed test is untouched; the mask is never an
inference input).

TARGET HIERARCHY (recorded per case in target_source):
  1. "segmentation" (PRIMARY): the 3D RibSeg rib label volume, canonicalized to RAS+, MAX-projected
     to AP/lateral, resized+padded with the case's exact geometry, then a small binary closing/dilate.
     This is a genuine rib-CAGE region prior, not a line.
  2. "centerline" (FALLBACK): RibSeg centerlines projected to points and dilated into a corridor of
     radius --corridor-dilate px. Weaker (a corridor, not a filled region); use for the ablation.
  3. "missing" / "skipped_*": no usable source, or the target failed a sanity gate.

CORRECTNESS FIXES over the prototype:
  * out-of-frame projected points are FILTERED, never clipped onto borders (clipping paints fake rib
    bands along image edges). retained_fraction (points kept / points total) is recorded; a case with
    too little anatomy inside the frame is skipped, not silently marked valid.
  * sanity gates reject all-zero, border-only, and near-full-frame masks.
  * exact case-order alignment to det_dev is asserted; det_dev sha256 is recorded for the training
    harness to verify.

Alignment (matches make_det_data.py): RibSeg is in ORIGINAL-image voxel space; make_det_data projects
in CANONICAL (RAS+) voxel space. Points: original-voxel --(A_orig)--> world --(inv A_can)--> canonical
voxel, then row=canvox[S]*scale+pad_top, col=canvox[{R for AP, A for lat}]*scale+pad_left (identical
to footprint_padded). Volumes: canonicalize, MAX-project along the view axis, then make_det_data's
resize_pad (same scale/pad as the CT projection — asserted against stored ap_geo/lat_geo).

Usage (segmentation primary, centerline fallback):
  python make_rib_targets.py --dev outputs/det_out_v2/det_dev.npz \
      --cl-dir data/ribseg/ribseg_v2/cl --seg-dir data/ribseg/ribseg_v2/seg \
      --image-dirs data/ribfrac_train data/ribfrac \
      --out outputs/det_out_v2/det_dev_rib_seg.npz --overlays outputs/rib_target_overlays_seg --n-overlay 24
  # centerline-only variant for the ablation:  --prefer centerline  (omit --seg-dir)
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
    import make_det_data as MD   # reuse resize_pad so the volume projection matches the CT projection exactly
except Exception:  # noqa: BLE001
    nib = None

RIB_TARGET_VERSION = "ribassist-rib-aux-2"
PROTOCOL_SIZE = 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _find(dirs, *pats):
    for d in dirs:
        for pat in pats:
            p = Path(d) / pat
            if p.exists(): return p
    for d in dirs:
        for pat in pats:
            for p in Path(d).rglob(pat):
                return p
    return None


def canonical_voxels(pts, A_orig, A_can):
    world = pts @ A_orig[:3, :3].T + A_orig[:3, 3]
    return (world - A_can[:3, 3]) @ np.linalg.inv(A_can[:3, :3]).T


def border_only(mask):
    """True if the mask has pixels but ALL of them lie on the outer 1px frame (an alignment artifact)."""
    if not mask.any(): return False
    interior = mask[1:-1, 1:-1]
    return not interior.any()


def project_volume_mask(seg_can, view, geo, S):
    """MAX-project a canonical (RAS+) binary rib volume to a view, then resize_pad with the case
    geometry. Returns (mask[S,S], recomputed (scale,pt,pl)). AP: sum axis=A(1), rows=S(2), cols=R(0);
    lat: sum axis=R(0), rows=S(2), cols=A(1)."""
    if view == "ap":
        line = seg_can.max(axis=1); img2d = np.transpose(line, (1, 0))          # (S, R)
    else:
        line = seg_can.max(axis=0); img2d = np.transpose(line, (1, 0))          # (S, A)
    out, scale, pt, pl = MD.resize_pad(img2d.astype(np.float32), S)
    return (out > 0.5), (scale, pt, pl)


def rasterize_points(rows, cols, S):
    """Filter out-of-frame points (NEVER clip), rasterize hits. Returns (mask, retained_fraction).
    Rounds to integer indices FIRST, then bounds-checks — so a point at 255.7 (which rint would push
    to 256) is correctly rejected rather than landing on / overflowing the border."""
    rows = np.asarray(rows, np.float32); cols = np.asarray(cols, np.float32)
    tot = len(rows)
    finite = np.isfinite(rows) & np.isfinite(cols)
    r = np.rint(np.where(finite, rows, -1)).astype(np.int64); c = np.rint(np.where(finite, cols, -1)).astype(np.int64)
    ok = (r >= 0) & (r < S) & (c >= 0) & (c < S)
    r, c = r[ok], c[ok]
    m = np.zeros((S, S), bool)
    if len(r): m[r, c] = True
    return m, (float(ok.mean()) if tot else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True); ap.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--cl-dir", type=Path, default=None, help="RibSeg centerline dir (<cid>.npz, key 'cl') — fallback source")
    ap.add_argument("--seg-dir", type=Path, default=None, help="RibSeg segmentation dir (<cid>*.nii(.gz)) — PRIMARY source")
    ap.add_argument("--prefer", choices=["seg", "centerline"], default="seg",
                    help="seg = segmentation primary + centerline fallback; centerline = centerline only (ablation)")
    ap.add_argument("--corridor-dilate", type=float, default=6.0, help="centerline corridor radius (px)")
    ap.add_argument("--seg-close", type=int, default=2, help="binary closing/dilation iterations on the projected seg mask")
    ap.add_argument("--min-retained", type=float, default=0.50, help="skip a centerline case if < this fraction of points fall in-frame")
    ap.add_argument("--min-coverage", type=float, default=0.005); ap.add_argument("--max-coverage", type=float, default=0.60)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--overlays", type=Path, default=None)
    ap.add_argument("--n-overlay", type=int, default=24)
    a = ap.parse_args()
    if nib is None: print("pip install nibabel scipy matplotlib", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    if a.prefer == "seg" and a.seg_dir is None and a.cl_dir is None:
        raise ValueError("Provide --seg-dir and/or --cl-dir.")

    dev_sha = sha256_file(a.dev); d = np.load(a.dev, allow_pickle=False)
    cases = [str(c) for c in d["case"]]; S = int(d["ap"].shape[-1]); N = len(cases)
    if S != PROTOCOL_SIZE: raise ValueError(f"det_dev images {S}px != {PROTOCOL_SIZE}")
    ap_geo, lat_geo = d["ap_geo"], d["lat_geo"]
    nfrac = d["nfrac"] if "nfrac" in d else np.zeros(N, int)
    ap_rib = np.zeros((N, S, S), np.float32); lat_rib = np.zeros((N, S, S), np.float32)
    source = np.array(["missing"] * N, dtype=object); has_rib = np.zeros(N, bool)
    retained = np.full(N, np.nan); apcov = np.full(N, np.nan); latcov = np.full(N, np.nan)
    skipped = {}; border_hits = 0
    seg_dirs = [a.seg_dir] if a.seg_dir else []

    print(f"generating rib targets for {N} dev cases (prefer={a.prefer}) ...", file=sys.stderr, flush=True)
    for i, cid in enumerate(cases):
        sa, pta, pla = map(float, ap_geo[i]); sl, ptl, pll = map(float, lat_geo[i])
        made = None  # ("segmentation"|"centerline", ap_mask, lat_mask, retained_frac)

        # ---- PRIMARY: projected segmentation volume ----
        if a.prefer == "seg" and a.seg_dir is not None:
            segp = _find(seg_dirs, f"{cid}.nii.gz", f"{cid}.nii", f"{cid}-rib-seg.nii.gz", f"{cid}*.nii.gz", f"{cid}*.nii")
            imp = _find(a.image_dirs, f"{cid}-image.nii.gz", f"{cid}-image.nii", f"{cid}-image.nii*")
            if segp is not None and imp is not None:
                try:
                    segc = nib.as_closest_canonical(nib.load(str(segp)))
                    seg = (np.asarray(segc.get_fdata()) > 0)
                    apm, (rsa, rpt, rpl) = project_volume_mask(seg, "ap", ap_geo[i], S)
                    latm, _ = project_volume_mask(seg, "lat", lat_geo[i], S)
                    # geometry must match the stored CT geometry (same canonical grid) — else alignment is off
                    if abs(rsa - sa) > 1e-3 or abs(rpt - pta) > 1 or abs(rpl - pla) > 1:
                        skipped[cid] = f"seg geometry mismatch (scale {rsa:.4f} vs {sa:.4f})"
                    else:
                        if a.seg_close > 0:
                            st = np.ones((3, 3), bool)
                            apm = binary_dilation(binary_closing(apm, st), st, iterations=a.seg_close)
                            latm = binary_dilation(binary_closing(latm, st), st, iterations=a.seg_close)
                        made = ("segmentation", apm.astype(np.float32), latm.astype(np.float32), 1.0)
                except Exception as e:  # noqa: BLE001
                    skipped[cid] = f"seg error: {e}"

        # ---- FALLBACK (or --prefer centerline): projected centerline corridor ----
        if made is None and a.cl_dir is not None:
            clp = a.cl_dir / f"{cid}.npz"; imp = _find(a.image_dirs, f"{cid}-image.nii.gz", f"{cid}-image.nii", f"{cid}-image.nii*")
            if clp.exists() and imp is not None:
                nii = nib.load(str(imp)); A_o = nii.affine; A_c = nib.as_closest_canonical(nii).affine
                cl = np.load(clp)["cl"]; R = cl.shape[0]; valid = ~np.all(cl.reshape(R, -1) == 0, axis=1)
                if valid.sum() >= 6:
                    ar, ac, lr_, lc = [], [], [], []
                    for r in range(R):
                        if not valid[r]: continue
                        cv = canonical_voxels(cl[r].astype(np.float64), A_o, A_c)
                        ar.append(cv[:, 2] * sa + pta); ac.append(cv[:, 0] * sa + pla)
                        lr_.append(cv[:, 2] * sl + ptl); lc.append(cv[:, 1] * sl + pll)
                    apm, ret_a = rasterize_points(np.concatenate(ar), np.concatenate(ac), S)
                    latm, ret_l = rasterize_points(np.concatenate(lr_), np.concatenate(lc), S)
                    ret = float(min(ret_a, ret_l))
                    apm = distance_transform_edt(~apm) <= a.corridor_dilate if apm.any() else apm
                    latm = distance_transform_edt(~latm) <= a.corridor_dilate if latm.any() else latm
                    made = ("centerline", apm.astype(np.float32), latm.astype(np.float32), ret)
                elif cid not in skipped:
                    skipped[cid] = "too few valid ribs"
            elif cid not in skipped:
                skipped[cid] = "no centerline/affine"

        if made is None:
            if cid not in skipped: skipped[cid] = "no source"
            continue
        src, apm, latm, ret = made
        ca, cl_cov = float(apm.mean()), float(latm.mean())
        # ---- sanity gates ----
        if src == "centerline" and ret < a.min_retained:
            source[i] = "skipped_retained"; skipped[cid] = f"retained {ret:.2f} < {a.min_retained}"; retained[i] = ret; continue
        cov = max(ca, cl_cov)
        if cov < a.min_coverage: source[i] = "skipped_small"; skipped[cid] = f"coverage {cov:.4f} < {a.min_coverage}"; continue
        if cov > a.max_coverage: source[i] = "skipped_large"; skipped[cid] = f"coverage {cov:.3f} > {a.max_coverage}"; continue
        if border_only(apm) or border_only(latm):
            border_hits += 1; source[i] = "skipped_border"; skipped[cid] = "border-only mask"; continue
        ap_rib[i], lat_rib[i] = apm, latm; source[i] = src; has_rib[i] = True
        retained[i] = ret; apcov[i] = ca; latcov[i] = cl_cov
        if (i + 1) % 50 == 0 or i + 1 == N: print(f"[{i+1}/{N}] processed", file=sys.stderr, flush=True)

    assert list(cases) == [str(c) for c in d["case"]]   # exact case-order alignment to det_dev
    n_seg = int((source == "segmentation").sum()); n_cl = int((source == "centerline").sum())
    n_have = int(has_rib.sum())
    if n_have == 0: raise RuntimeError("No case produced a valid rib target — check --seg-dir / --cl-dir / --image-dirs.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, ap_rib_mask=ap_rib.astype(np.float16), lat_rib_mask=lat_rib.astype(np.float16),
                        case=np.array(cases), has_rib=has_rib, target_source=source.astype("U16"),
                        retained_fraction=retained.astype(np.float32),
                        ap_coverage=apcov.astype(np.float32), lat_coverage=latcov.astype(np.float32))
    def _med(x): v = x[np.isfinite(x)]; return round(float(np.median(v)), 4) if len(v) else None
    man = {"rib_target_version": RIB_TARGET_VERSION,
           "rib_auxiliary_target": {"source": f"projected RibSeg ({a.prefer} primary): segmentation volume max-projection "
                                    "or centerline corridor", "resolution": S, "corridor_dilate_px": a.corridor_dilate,
                                    "seg_close_iters": a.seg_close, "usage": "training-only auxiliary supervision",
                                    "inference_input": False},
           "aligned_to": {"det_dev": str(a.dev), "det_dev_sha256": dev_sha,
                          "note": "masks reuse det_dev per-case ap_geo/lat_geo; det_dev.npz unchanged; case order asserted equal"},
           "counts": {"n_dev": N, "segmentation": n_seg, "centerline": n_cl, "with_rib": n_have,
                      "skipped": len(skipped), "border_only_rejected": border_hits},
           "distribution": {"median_ap_coverage": _med(apcov), "median_lat_coverage": _med(latcov),
                            "median_retained_fraction": _med(retained)},
           "gates": {"min_retained": a.min_retained, "min_coverage": a.min_coverage, "max_coverage": a.max_coverage},
           "skipped": skipped, "rib_target_sha256": sha256_file(a.out),
           "software": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__}}
    (a.out.with_name(a.out.stem + "_manifest.json")).write_text(json.dumps(man, indent=2))

    print(f"\n{'Cases in detector dev set:':40}{N}")
    print(f"{'Exact case-order matches:':40}{N}/{N}")
    print(f"{'Full RibSeg segmentation targets:':40}{n_seg}")
    print(f"{'Centerline fallback targets:':40}{n_cl}")
    print(f"{'Missing/invalid/skipped targets:':40}{N - n_have}")
    print(f"{'Median AP target coverage:':40}{_med(apcov)}")
    print(f"{'Median lateral target coverage:':40}{_med(latcov)}")
    print(f"{'Median retained source fraction:':40}{_med(retained)}")
    print(f"{'Border-only masks rejected:':40}{border_hits}")
    print(f"wrote {a.out}  (+ manifest). ")

    if a.overlays is not None:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            print(f"[overlay skip] {e}", file=sys.stderr); return 0
        a.overlays.mkdir(parents=True, exist_ok=True)
        have = np.where(has_rib)[0]
        cov = np.nan_to_num(np.maximum(np.nan_to_num(apcov), np.nan_to_num(latcov)))
        pick = set()
        order = have[np.argsort(cov[have])]
        for j in list(order[:3]) + list(order[-3:]): pick.add(int(j))                 # coverage extremes
        ret_order = have[np.argsort(np.nan_to_num(retained[have], nan=1.0))]
        for j in ret_order[:3]: pick.add(int(j))                                       # worst retained
        for j in have[nfrac[have] > 0][:4]: pick.add(int(j))                           # positive cases
        for j in have[nfrac[have] == 0][:3]: pick.add(int(j))                          # negative cases
        rng = np.random.RandomState(0)
        for j in rng.choice(have, size=min(len(have), a.n_overlay), replace=False): pick.add(int(j))
        pick = sorted(pick)[: max(a.n_overlay, 20)]
        for i in pick:
            fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
            for x, v, rib in ((ax[0], "ap", ap_rib[i]), (ax[1], "lat", lat_rib[i])):
                x.imshow(d[v][i].astype(np.float32), cmap="gray", origin="lower")
                x.imshow(np.ma.masked_where(rib < 0.5, rib), cmap="cool", alpha=0.5, origin="lower")
                x.set_title(f"{cases[i]} {v} [{source[i]}] cov {rib.mean():.3f} nfx {int(nfrac[i])}"); x.axis("off")
            fig.tight_layout(); fig.savefig(a.overlays / f"{cases[i]}_rib.png", dpi=100); plt.close(fig)
        print(f"wrote {len(pick)} rib-target overlays to {a.overlays}/ — verify: (1) band follows the rib cage not the "
              "whole thorax, (2) NO lines along image borders, (3) fractures lie inside/adjacent to the rib region.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
