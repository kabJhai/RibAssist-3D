#!/usr/bin/env python3
"""Build the ground-truth 'fracture address' label table for the rib-fracture
localization project, and resolve the anterior/lateral/posterior segment convention.

For every joinable case it emits one row per fracture with:
  side (L/R), rib_number (superior->inferior), segment (posterior/lateral/anterior),
  centerline_pos (0=posterior .. 1=anterior along the rib arc), fracture_class,
  and the fracture centroid in RAS world mm.

Segment is the dimension the feasibility audit flagged as convention-dependent, so it
is computed robustly here: everything is mapped to RAS world coordinates via the NIfTI
affine (anterior = +y), the centerline direction is fixed against world-y (not assumed),
and the centerline-arc segment is cross-checked against an independent world-y-thirds
segment. The printed report shows their agreement and the corrected distribution.

Usage:
  python build_labels.py --ribfrac-dir data/ribfrac --ribseg-dir data/ribseg \
      --info data/ribfrac/ribfrac-val-info.csv --out labels_val.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except Exception:  # noqa: BLE001
    nib = None

SEG_NAMES = ["posterior", "lateral", "anterior"]


def _cid_num(name: str):
    m = re.search(r"RibFrac[_-]?0*(\d+)", name, re.I)
    return int(m.group(1)) if m else None


def _is_nii(n: str) -> bool:
    return n.endswith(".nii") or n.endswith(".nii.gz")


def find_cases(ribfrac_dir: Path, ribseg_dir: Path):
    rf = {}
    for p in Path(ribfrac_dir).rglob("*"):
        if not p.is_file() or not _is_nii(p.name):
            continue
        num = _cid_num(p.name)
        if num is None:
            continue
        low = p.name.lower()
        if "image" in low:
            rf.setdefault(num, {})["image"] = p
        elif "label" in low:
            rf.setdefault(num, {})["label"] = p
    rseg, rcl = {}, {}
    for p in Path(ribseg_dir).rglob("*"):
        if not p.is_file():
            continue
        num = _cid_num(p.name)
        if num is None:
            continue
        if p.name.endswith(".npz"):
            rcl.setdefault(num, p)
        elif _is_nii(p.name):
            rseg.setdefault(num, p)
    out = []
    for num in sorted(rf):
        d = rf[num]
        if "image" in d and "label" in d and num in rseg:
            out.append({"cid": f"RibFrac{num}", "num": num, "flabel": d["label"],
                        "rseg": rseg[num], "cl": rcl.get(num)})
    return out


def vox_to_world(vox_xyz, affine):
    """vox_xyz: [...,3] index coords -> RAS world mm."""
    v = np.asarray(vox_xyz, float)
    return v @ affine[:3, :3].T + affine[:3, 3]


def rib_geometry(rseg, affine):
    labels = [int(v) for v in np.unique(rseg) if v != 0]
    allvox = np.array(np.nonzero(rseg))
    mid_x = allvox[0].mean() if allvox.size else rseg.shape[0] / 2
    info = {}
    for lb in labels:
        vox = np.array(np.nonzero(rseg == lb))  # [3,k]
        c = vox.mean(axis=1)
        world = vox_to_world(vox.T, affine)  # [k,3]
        info[lb] = {"centroid_vox": c, "side": "L" if c[0] < mid_x else "R",
                    "vox": vox, "world": world}
    for side in ("L", "R"):
        sib = sorted([lb for lb in labels if info[lb]["side"] == side],
                     key=lambda lb: info[lb]["centroid_vox"][2], reverse=True)
        for i, lb in enumerate(sib, 1):
            info[lb]["number"] = i
    return info


def oriented_centerline(cl_row_vox, affine):
    """Return centerline points (500,3 vox) oriented so index 0 = posterior (min world-y),
    plus a bool 'flipped'."""
    world_y = vox_to_world(cl_row_vox, affine)[:, 1]
    # posterior end = lower world-y; ensure start (first 10%) is more posterior than end
    if world_y[:50].mean() > world_y[-50:].mean():
        return cl_row_vox[::-1], True
    return cl_row_vox, False


def seg_from_pos(s):
    return SEG_NAMES[min(2, int(s * 3))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ribfrac-dir", type=Path, required=True)
    ap.add_argument("--ribseg-dir", type=Path, required=True)
    ap.add_argument("--info", type=Path)
    ap.add_argument("--out", type=Path, default=Path("labels.csv"))
    ap.add_argument("--n", type=int, default=0, help="cap cases (0=all)")
    args = ap.parse_args()
    if nib is None:
        print("pip install nibabel numpy pandas", file=sys.stderr)
        return 1
    import pandas as pd

    cls_map = {}
    if args.info and args.info.exists():
        idf = pd.read_csv(args.info)
        cols = {c.lower(): c for c in idf.columns}
        pid, lid, code = cols.get("public_id"), cols.get("label_id"), cols.get("label_code")
        if pid and lid and code:
            for _, r in idf.iterrows():
                cls_map[(str(r[pid]), int(r[lid]))] = int(r[code])

    cases = find_cases(args.ribfrac_dir, args.ribseg_dir)
    if args.n:
        cases = cases[: args.n]
    if not cases:
        print("No joinable cases found.", file=sys.stderr)
        return 1

    rows, geo_fail, n_flip, agree, seg_total = [], 0, 0, 0, 0
    print(f"joining {len(cases)} cases ...", file=sys.stderr, flush=True)
    for i, cs in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {cs['cid']} loading ...", file=sys.stderr, flush=True)
        fl_img, rs_img = nib.load(str(cs["flabel"])), nib.load(str(cs["rseg"]))
        if fl_img.shape != rs_img.shape or not np.allclose(fl_img.affine, rs_img.affine, atol=1e-2):
            geo_fail += 1
            print(f"  [GEO-FAIL] {cs['cid']}", file=sys.stderr, flush=True)
            continue
        aff = fl_img.affine
        fl = np.asarray(fl_img.get_fdata()).astype(int)
        rs = np.asarray(rs_img.get_fdata()).astype(int)
        cl = None
        if cs["cl"] is not None:
            z = np.load(cs["cl"]); cl = z[list(z.keys())[0]]
        nfrac = int((np.unique(fl) != 0).sum())
        print(f"  {cs['cid']}: {nfrac} fractures, {'centerlines ok' if cl is not None else 'NO centerlines'}",
              file=sys.stderr, flush=True)
        ribinfo = rib_geometry(rs, aff)
        allw = np.concatenate([ri["world"] for ri in ribinfo.values()], 0) if ribinfo else None
        spine_x = float(np.median(allw[:, 0])) if allw is not None else 0.0

        for lb in np.unique(fl):
            if lb == 0:
                continue
            fvox = np.array(np.nonzero(fl == lb))
            at = rs[fvox[0], fvox[1], fvox[2]]
            nz = at[at != 0]
            rib = int(np.bincount(nz).argmax()) if nz.size else \
                min(ribinfo, key=lambda k: np.linalg.norm(ribinfo[k]["centroid_vox"] - fvox.mean(1)))
            ri = ribinfo.get(rib, {})
            fc_w = vox_to_world(fvox.mean(1), aff)
            s, seg_cl, seg_anat = float("nan"), "unknown", "unknown"
            if cl is not None and 0 <= rib - 1 < cl.shape[0]:
                pts, flipped = oriented_centerline(cl[rib - 1], aff)
                n_flip += int(flipped)
                pw = vox_to_world(pts, aff)
                d = np.linalg.norm(pw - fc_w[None], axis=1)
                s = int(d.argmin()) / (len(pts) - 1)
                seg_cl = seg_from_pos(s)
                # truncation-robust: anchor 'lateral' to the rib's true lateral apex
                # (max |x - spine_x|), preserved even when costal cartilage is unsegmented
                apex = int(np.abs(pw[:, 0] - spine_x).argmax()) / (len(pts) - 1)
                seg_anat = "lateral" if abs(s - apex) <= 0.2 else \
                    ("posterior" if s < apex else "anterior")
            # independent cross-check: world-y thirds within the rib's y-range
            seg_wy = "unknown"
            if "world" in ri:
                ry = ri["world"][:, 1]
                p = (fc_w[1] - ry.min()) / (ry.max() - ry.min() + 1e-6)
                seg_wy = seg_from_pos(min(0.999, max(0.0, p)))
            if seg_cl != "unknown" and seg_wy != "unknown":
                seg_total += 1
                agree += int(seg_cl == seg_wy)
            rows.append({"case": cs["cid"], "frac_id": int(lb), "side": ri.get("side"),
                         "rib_number": ri.get("number"), "segment": seg_anat,
                         "segment_arc_thirds": seg_cl, "segment_worldy": seg_wy,
                         "centerline_pos": round(s, 3) if s == s else None,
                         "fracture_class": cls_map.get((cs["cid"], int(lb))),
                         "world_x": round(float(fc_w[0]), 1), "world_y": round(float(fc_w[1]), 1),
                         "world_z": round(float(fc_w[2]), 1)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("==================== LABEL BUILD SUMMARY ====================")
    print(f"cases: {len(cases)}   geometry-join failures: {geo_fail}   fractures: {len(df)}")
    print(f"centerlines flipped to posterior->anterior: {n_flip}")
    if seg_total:
        print(f"segment agreement (centerline-arc vs world-y): {agree}/{seg_total} "
              f"({agree/seg_total:.0%})")
    if len(df):
        print("segment (apex-anchored, PRIMARY):", df["segment"].value_counts().to_dict())
        print("segment (arc-thirds, biased)     :", df["segment_arc_thirds"].value_counts().to_dict())
        print("segment (world-y check)          :", df["segment_worldy"].value_counts().to_dict())
        print("side:", df["side"].value_counts().to_dict(),
              " rib#:", int(df["rib_number"].min()), "..", int(df["rib_number"].max()))
        if df["fracture_class"].notna().any():
            print("fracture_class:", df["fracture_class"].value_counts().to_dict())
    print(f"\nwrote {args.out}")
    print("PRIMARY targets = centerline_pos (continuous) + world coords (robust). The "
          "'segment' column is now apex-anchored (lateral = near the rib's max-|x| apex), "
          "which should restore a clinically sensible lateral share vs the arc-thirds "
          "version. Compare the three segment lines above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
