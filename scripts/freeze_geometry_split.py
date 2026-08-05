#!/usr/bin/env python3
"""Freeze the deterministic ATLAS-BUILD / GEOMETRY-VALIDATION split for the reconstruction stage.

Mirrors the detector discipline: one written-down, hash-based, label-blind split. The atlas is
later built from the BUILD cases only, and the projection fitter is tuned/compared exclusively on
the VALIDATION cases — so the atlas never contains the case it is scored against (leakage control).

Split is ~80/20 by md5(case_id) WITHIN strata (source cohort x rib-centerline completeness), with
>=1 build and >=1 val guaranteed per stratum of size>1. The 55 sealed detector-test cases are
excluded. Cases without a RibSeg centerline are recorded and excluded. LOO is a later robustness
analysis, not this split.

Output: geometry_split.json {build:[...], val:[...], val_pct, split_md5, strata, excluded_no_cl}.

Usage:
  python freeze_geometry_split.py --det-manifest outputs/det_out_v2/det_manifest.json \
      --cl-dir data/ribseg/ribseg_v2/cl --val-pct 20 --out geometry_split.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-manifest", type=Path, required=True)
    ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("geometry_split.json"))
    ap.add_argument("--seed-tag", default="ribassist-geometry-split")
    a = ap.parse_args()
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; the geometry split is frozen — remove explicitly to re-freeze.")
    man = json.loads(a.det_manifest.read_text())
    src = man.get("dev_case_sources", {})
    dev = [c for c in src if c not in set(man.get("test_cases", []))]
    if not dev: raise ValueError("No dev cases in manifest.")

    strata = {}; excluded = []
    for c in dev:
        clp = a.cl_dir / f"{c}.npz"
        if not clp.exists(): excluded.append(c); continue
        cl = np.load(clp)["cl"]; nvalid = int((~np.all(cl.reshape(cl.shape[0], -1) == 0, axis=1)).sum())
        grp = "train_part1" if str(src[c]) == "train_part1" else "validation"
        comp = "complete24" if nvalid == 24 else f"partial{nvalid}"
        strata.setdefault(f"{grp}:{comp}", []).append(c)

    build, val = [], []
    rep = {}
    for s in sorted(strata):
        members = strata[s]; n = len(members)
        hv = sorted(members, key=lambda c: int(hashlib.md5(f"{a.seed_tag}:{c}".encode()).hexdigest(), 16))
        if n == 1:
            build += hv; rep[s] = {"n": 1, "build": 1, "val": 0}; continue
        k = min(max(int(round(n * a.val_pct / 100.0)), 1), n - 1)  # >=1 val AND >=1 build
        val += hv[:k]; build += hv[k:]; rep[s] = {"n": n, "build": n - k, "val": k}

    build = sorted(build); val = sorted(val)
    payload = {"build": build, "val": val, "val_pct": a.val_pct, "seed_tag": a.seed_tag,
               "strata": rep, "excluded_no_cl": excluded, "n_build": len(build), "n_val": len(val),
               "note": "atlas built from BUILD only; projection fitter tuned/compared on VAL only; sealed detector "
                       "test excluded; LOO is a later robustness analysis"}
    payload["split_md5"] = hashlib.md5(json.dumps({"build": build, "val": val}, sort_keys=True).encode()).hexdigest()
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"froze {a.out}: build {len(build)} / val {len(val)}  (excluded no-cl: {len(excluded)}) | md5 {payload['split_md5'][:12]}..")
    print("strata (n/build/val): " + " | ".join(f"{k} {v['n']}/{v['build']}/{v['val']}" for k, v in rep.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
