#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Assemble the SEALED-TEST detector dataset into the det_dev.npz schema, provenance-verified.

The sealed cohort ships split as det_test_inputs.npz (images + geometry) + det_test_gt.npz (footprints),
tied together by det_manifest.json['artifact_sha256']. D0/D1 consume the combined det_dev.npz schema, so this
merges the two — by shared case order — into a single det_test.npz with the same keys D0/D1 read (heatmap
TARGETS and 'source' are training-only and intentionally omitted). It is a DETERMINISTIC assembly: no fitting,
no thresholds, nothing learned. It fails closed if either source file's sha256 disagrees with the manifest.

The printed det_test.npz sha256 is the EXTERNAL ANCHOR passed to D0/D1 --expected-data-sha256 for the sealed run,
so the sealed evaluation is bound to exactly this assembled cohort.

Usage (from RibAssist 3D ROOT):
  python scripts/build_sealed_det_npz.py \
      --inputs outputs/det_out_v2/det_test_inputs.npz \
      --gt     outputs/det_out_v2/det_test_gt.npz \
      --manifest outputs/det_out_v2/det_manifest.json \
      --out    outputs/det_out_v2/det_test.npz
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

try:
    import train_detector as T
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e

# keys taken from each source (det_dev schema minus training-only ap_hm/lat_hm/source)
FROM_INPUTS = ["ap", "lat", "ap_geo", "lat_geo", "ap_sp", "lat_sp", "case"]
FROM_GT = ["nfrac", "ap_fp_pts", "ap_fp_ptr", "lat_fp_pts", "lat_fp_ptr", "ap_ctr", "lat_ctr", "fp_case", "fp_iid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True); ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR})", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path or remove it.")

    man = json.loads(a.manifest.read_text()); art = man.get("artifact_sha256", {})
    # FAIL CLOSED: both sources must match their manifest digests
    for p, key in ((a.inputs, "det_test_inputs.npz"), (a.gt, "det_test_gt.npz")):
        want = art.get(key)
        if not want: raise ValueError(f"manifest has no artifact_sha256[{key!r}]")
        got = T.sha256_file(p)
        if got != want: raise ValueError(f"{p.name} sha256 {got[:12]}.. != manifest {str(want)[:12]}..")
        print(f"  verified {p.name} sha256 {got[:12]}.. == manifest")

    zi = np.load(a.inputs, allow_pickle=False); zg = np.load(a.gt, allow_pickle=False)
    ci = [str(x) for x in zi["case"]]; cg = [str(x) for x in zg["case"]]
    if ci != cg:
        raise ValueError("inputs/gt case order differs; refusing to merge without an explicit remap "
                         f"(inputs[:3]={ci[:3]} gt[:3]={cg[:3]})")
    for k in FROM_INPUTS:
        if k not in zi.files: raise ValueError(f"inputs missing key {k}")
    for k in FROM_GT:
        if k not in zg.files: raise ValueError(f"gt missing key {k}")

    out = {k: zi[k] for k in FROM_INPUTS}
    out.update({k: zg[k] for k in FROM_GT})
    # sanity: fp_case indexes into the case array
    if int(np.asarray(out["fp_case"]).max()) >= len(out["case"]):
        raise ValueError("fp_case index exceeds number of cases")
    np.savez_compressed(a.out, **out)

    out_sha = T.sha256_file(a.out)
    side = a.out.with_suffix(".provenance.json")
    side.write_text(json.dumps({
        "assembled_from": {"inputs": str(a.inputs), "gt": str(a.gt), "manifest": str(a.manifest)},
        "source_sha256": {"det_test_inputs.npz": art["det_test_inputs.npz"], "det_test_gt.npz": art["det_test_gt.npz"]},
        "manifest_split_md5": man.get("split_md5"), "manifest_dataset_version": man.get("dataset_version"),
        "n_cases": len(out["case"]), "n_fractures": int(np.asarray(out["fp_case"]).shape[0]),
        "det_test_npz_sha256": out_sha, "omitted_keys": ["ap_hm", "lat_hm", "source"],
        "note": "deterministic merge of the split sealed cohort into the det_dev schema; heatmap targets omitted "
                "(training-only). Use det_test_npz_sha256 as --expected-data-sha256 for the sealed D0/D1 run."}, indent=2))

    print(f"\nwrote {a.out}  ({len(out['case'])} cases, {int(np.asarray(out['fp_case']).shape[0])} GT fractures)")
    print(f"  det_test.npz sha256 = {out_sha}")
    print(f"  provenance sidecar   = {side}")
    print(f"  >>> pass this as --expected-data-sha256 {out_sha[:16]}.. to the sealed D0/D1 run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
