#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Build the CT-derived, fracture-centered crop dataset for the RibAssist 3D addressing model
(localization engine). For each annotated fracture, render AP + lateral DRRs once and save
a fixed-size crop centered on the fracture's projected location in each view, with the
ground-truth address label (side, rib number 1-12, normalized position s).

Raw grayscale crops ONLY — no markers, masks, ids, or position channels (leakage-safe).
Output: <out>/address_dataset.npz  with arrays:
  ap [N,C,C] float16, lat [N,C,C] float16, side [N] (0=L,1=R), rib [N] (1-12),
  s [N] float, case [N] str, fclass [N] int.
Case strings enable case-level (grouped) splitting downstream.

Usage: python extract_crops.py --ribfrac-dir data/ribfrac --ribseg-dir data/ribseg \
    --out . --crop 96
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    from scipy.ndimage import zoom
except Exception:  # noqa: BLE001
    nib = None


def _num(n): m = re.search(r"RibFrac[_-]?0*(\d+)", n, re.I); return int(m.group(1)) if m else None
def _nii(n): return n.endswith(".nii") or n.endswith(".nii.gz")


def find_cases(rfd, rsd):
    rf = {}
    for p in Path(rfd).rglob("*"):
        if p.is_file() and _nii(p.name) and (k := _num(p.name)) is not None:
            rf.setdefault(k, {})["image" if "image" in p.name.lower() else "label"] = p
    rs, rc = {}, {}
    for p in Path(rsd).rglob("*"):
        if p.is_file() and (k := _num(p.name)) is not None:
            if p.name.endswith(".npz"): rc.setdefault(k, p)
            elif _nii(p.name): rs.setdefault(k, p)
    return [{"cid": f"RibFrac{k}", "image": rf[k]["image"], "flabel": rf[k]["label"], "rseg": rs[k], "cl": rc[k]}
            for k in sorted(rf) if "image" in rf[k] and "label" in rf[k] and k in rs and k in rc]


def world_axes(A):
    R = A[:3, :3]; lr, ap, si = (int(np.argmax(np.abs(R[i]))) for i in range(3))
    if len({lr, ap, si}) < 3: lr, ap, si = 0, 1, 2
    return lr, ap, si, np.sqrt((R ** 2).sum(0))


def project(mu, pa, ra, ca, sp):
    line = (mu * sp[pa]).sum(axis=pa); rem = [a for a in (0, 1, 2) if a != pa]
    img = np.transpose(line, (rem.index(ra), rem.index(ca)))
    img = img - img.min(); return (img / (img.max() + 1e-8)).astype(np.float32)


def rib_geometry(rs):
    labels = [int(v) for v in np.unique(rs) if v != 0]
    allv = np.array(np.nonzero(rs)); mid = allv[0].mean() if allv.size else rs.shape[0] / 2
    info = {}
    for lb in labels:
        c = np.array(np.nonzero(rs == lb)).mean(1)
        info[lb] = {"c": c, "side": "L" if c[0] < mid else "R"}
    for side in ("L", "R"):
        for i, lb in enumerate(sorted([l for l in labels if info[l]["side"] == side],
                                      key=lambda l: info[l]["c"][2], reverse=True), 1):
            info[lb]["num"] = min(12, i)
    return info


def crop_at(img, r, c, half, out):
    H, W = img.shape
    r = min(max(r, half), H - half - 1); c = min(max(c, half), W - half - 1)
    p = img[r - half:r + half + 1, c - half:c + half + 1]
    return zoom(p, (out / p.shape[0], out / p.shape[1]), order=1)[:out, :out].astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ribfrac-dir", type=Path, required=True); ap.add_argument("--ribseg-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(".")); ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--half", type=int, default=24); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    if nib is None: print("pip install nibabel scipy", file=sys.stderr); return 1
    cases = find_cases(a.ribfrac_dir, a.ribseg_dir)
    if a.n: cases = cases[: a.n]
    APc, LATc, side, rib, S, case, fcls, APxy, LATxy = [], [], [], [], [], [], [], [], []
    print(f"extracting crops from {len(cases)} cases ...", file=sys.stderr, flush=True)
    n_in, excl = 0, {}
    for i, cs in enumerate(cases, 1):
        fl = np.asarray(nib.load(str(cs["flabel"])).get_fdata()).astype(int)
        if not (fl != 0).any(): continue
        rs = np.asarray(nib.load(str(cs["rseg"])).get_fdata()).astype(int)
        ct = nib.load(str(cs["image"])); A = ct.affine
        lr, apx, si, sp = world_axes(A); mu = np.clip(1 + ct.get_fdata() / 1000, 0, None).astype(np.float32)
        im_ap, im_lat = project(mu, apx, si, lr, sp), project(mu, lr, si, apx, sp)
        spine_lr = float(np.nonzero(rs)[lr].mean())  # rib-cage horizontal midline (~spine) for side
        z = np.load(cs["cl"]); cl = z[list(z.keys())[0]]
        info = rib_geometry(rs)
        for lb in np.unique(fl):
            if lb == 0: continue
            n_in += 1
            v = np.array(np.nonzero(fl == lb)); fc = v.mean(1)
            at = rs[v[0], v[1], v[2]]; nz = at[at != 0]
            rl = int(np.bincount(nz).argmax()) if nz.size else 0
            if not (1 <= rl <= cl.shape[0]) or rl not in info:
                key = "rib_unassigned (fracture voxels overlap no rib label)" if rl == 0 else "rib_id_out_of_range"
                excl[key] = excl.get(key, 0) + 1; continue
            pts = cl[rl - 1]
            wy = (pts @ A[:3, :3].T + A[:3, 3])[:, 1]
            if wy[:50].mean() > wy[-50:].mean(): pts = pts[::-1]
            d = np.linalg.norm(pts - fc[None], axis=1); idx = int(d.argmin())
            s = idx / (len(pts) - 1); p = pts[idx]
            ar, ac = int(round(p[si])), int(round(p[lr])); lrr, lcc = int(round(p[si])), int(round(p[apx]))
            APc.append(crop_at(im_ap, ar, ac, a.half, a.crop)); LATc.append(crop_at(im_lat, lrr, lcc, a.half, a.crop))
            Hap, Wap = im_ap.shape; Hl, Wl = im_lat.shape
            # AP coords: vertical normalized; horizontal is SPINE-RELATIVE (sign => side)
            APxy.append([ar / Hap, (ac - spine_lr) / Wap]); LATxy.append([lrr / Hl, lcc / Wl])
            side.append(0 if info[rl]["side"] == "L" else 1); rib.append(int(info[rl]["num"])); S.append(float(s))
            case.append(cs["cid"]); fcls.append(int(nz.max()) if nz.size else -1)  # placeholder class
        print(f"[{i}/{len(cases)}] {cs['cid']}: total crops={len(APc)}", file=sys.stderr, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    fp = a.out / "address_dataset.npz"
    np.savez_compressed(fp, ap=np.array(APc), lat=np.array(LATc), side=np.array(side),
                        rib=np.array(rib), s=np.array(S, np.float32), case=np.array(case), fclass=np.array(fcls),
                        ap_xy=np.array(APxy, np.float32), lat_xy=np.array(LATxy, np.float32))
    print(f"\nwrote {fp}: {len(APc)} crops, {a.crop}x{a.crop}, from {len(set(case))} cases")
    print(f"input fractures: {n_in}   crops written: {len(APc)}   excluded: {n_in - len(APc)}")
    for k, v in excl.items():
        print(f"  - {k}: {v}")
    print(f"side L/R: {(np.array(side)==0).sum()}/{(np.array(side)==1).sum()}   "
          f"rib range {min(rib) if rib else '-'}..{max(rib) if rib else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
