#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""STAGE D2a — APPEARANCE PAIR-DATASET extractor for the learned correspondence scorer (D2b).

Builds the reproducible appearance + label dataset the D2 two-tower scorer trains on, WITHOUT touching the
frozen detector or the frozen broad graph. It reads the D0 broad pair graph (edges + coordinates + labels)
and the projection images in det_dev.npz, extracts an AP crop at each edge's AP peak and a lateral crop at
its lateral peak, and writes a compact feature NPZ ALIGNED ROW-FOR-ROW with the D0 edge NPZ (so D2b joins by
row order and reuses the D0 coordinates/geometry for the operational scoreboard).

Design choices (per the D2 review):
  * Crops are deduplicated into per-view BANKS keyed by (case, peak) — a peak appears in many edges but is
    cropped once; each edge stores its two bank indices. Far smaller than one crop per edge.
  * Labels come from D0's many-to-one classes: 0 positive (positive_capable + single shared iid), 1 cross-
    instance (both real, different fractures — the HARD negative SI/confidence cannot resolve), 2 one-sided,
    3 fully-false. IDENTITY-AMBIGUOUS edges get label -1 (EXCLUDED — never given an arbitrary positive target).
  * Scalar covariates carried per edge: |dSI| (vox), AP/lateral confidence, confidence min/product/asymmetry,
    duplicate indicators. Appearance is the PRINCIPAL new signal; these are auxiliary so D2b can test whether
    appearance adds value beyond D1's deterministic SI+confidence.
  * Case-level 5-fold assignment (by sorted case id) is stored so D2b splits/CVs by CASE, never by edge.
  * Provenance: carries the D0 data hash + detector run (+ checkpoint SHAs when present) and records the crop
    spec + source-image hash, so the appearance features are reproducible from the recorded checkpoint+coords.

Crops are IMAGE-space patches (row=SI etc.); frozen detector FEATURE patches are an alternative D2b may add.

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory).
Provenance fails closed: D0-NPZ data hash == sha256(--data) == detector det_dev_sha256; NPZ detector run matches.

Usage:
  python eval_correspondence_D2a_crops.py \
      --pairs-npz outputs/correspondence_D0_broad_pairs.npz \
      --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --half 20 \
      --out outputs/correspondence_D2a_crops.npz
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

CLASS_TO_LABEL = {"positive_capable": 0, "cross_instance_only": 1, "one_sided": 2, "fully_false": 3}
LABEL_NAMES = ["positive", "cross_instance", "one_sided", "fully_false", "ambiguous_excluded"]
K_FOLDS = 5


def extract_crop(img, r, c, half):
    """Fixed (2*half)x(2*half) patch centered at (row=r, col=c) with zero padding at the borders."""
    H, W = img.shape; r = int(round(r)); c = int(round(c))
    out = np.zeros((2 * half, 2 * half), np.float32)
    r0, r1, c0, c1 = r - half, r + half, c - half, c + half
    sr0, sr1, sc0, sc1 = max(0, r0), min(H, r1), max(0, c0), min(W, c1)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = img[sr0:sr1, sc0:sc1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-npz", type=Path, required=True); ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--half", type=int, default=20, help="crop half-window (crop is 2*half x 2*half)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch numpy", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; use a new path.")
    if a.half <= 0: raise ValueError("--half must be positive")

    # ---- FAIL-CLOSED provenance ----
    rec = json.loads((a.detector_run / "detector_dev_run.json").read_text())
    det_sha = rec.get("det_dev_sha256")
    if not det_sha: raise ValueError("detector_dev_run.json missing det_dev_sha256")
    data_sha = T.sha256_file(a.data)
    if data_sha != det_sha: raise ValueError("--data hash != detector det_dev_sha256")
    z = np.load(a.pairs_npz, allow_pickle=False)
    if str(z["data_sha256"]) != data_sha: raise ValueError("pairs NPZ data hash != --data")
    if Path(str(z["detector_run"])).resolve() != a.detector_run.resolve():
        raise ValueError("pairs NPZ detector run != --detector-run")
    ck = {}
    if "detector_ap_sha256" in z.files:
        for v in ("ap", "lat"):
            got = T.sha256_file(a.detector_run / f"detector_{v}.pt")
            if got != str(z[f"detector_{v}_sha256"]): raise ValueError(f"detector_{v}.pt hash != NPZ record")
            ck[v] = got

    d = np.load(a.data, allow_pickle=False)
    ap_imgs = d["ap"]; lat_imgs = d["lat"]                     # (Ncase, H, W) projection images
    img_sha = hashlib.sha256(np.ascontiguousarray(ap_imgs).tobytes() + np.ascontiguousarray(lat_imgs).tobytes()).hexdigest()

    cgi = z["case_global_idx"]; cid_arr = np.array([str(x) for x in z["case_id"]]); apx = z["ap_idx"]; ltx = z["lat_idx"]
    ap_row = z["ap_row"]; ap_col = z["ap_col"]; lat_row = z["lat_row"]; lat_col = z["lat_col"]
    ap_s = z["ap_score"].astype(np.float32); lt_s = z["lat_score"].astype(np.float32)
    dsi = z["dsi_vox"].astype(np.float32); cls = z["cls"]; shared_iid = z["shared_iid"]; ident_amb = z["identity_ambiguous"]
    ap_dup = z["ap_is_dup"]; lat_dup = z["lat_is_dup"]
    class_names = [str(x) for x in z["class_names"]]
    cls_to_label = {i: CLASS_TO_LABEL[nm] for i, nm in enumerate(class_names)}
    n_edge = len(cgi); half = a.half
    val_ids = sorted({str(x) for x in cid_arr})
    fold_of = {c: i % K_FOLDS for i, c in enumerate(val_ids)}
    print(f"D2a: {n_edge} edges over {len(val_ids)} cases. crop {2*half}x{2*half}. provenance "
          f"{'checkpoint' if ck else 'run-path'}. building dedup crop banks ...", flush=True)

    # ---- deduplicated crop banks keyed by (case_global_idx, peak_idx) per view ----
    ap_key = {}; lat_key = {}; ap_bank = []; lat_bank = []
    edge_ap_bank = np.empty(n_edge, np.int32); edge_lat_bank = np.empty(n_edge, np.int32)
    for r in range(n_edge):
        if r % 5000 == 0: print(f"  crop {r}/{n_edge} (ap-bank {len(ap_bank)}, lat-bank {len(lat_bank)}) ...", flush=True)
        gi = int(cgi[r])
        ka = (gi, int(apx[r]))
        if ka not in ap_key:
            ap_key[ka] = len(ap_bank); ap_bank.append(extract_crop(ap_imgs[gi], ap_row[r], ap_col[r], half))
        edge_ap_bank[r] = ap_key[ka]
        kl = (gi, int(ltx[r]))
        if kl not in lat_key:
            lat_key[kl] = len(lat_bank); lat_bank.append(extract_crop(lat_imgs[gi], lat_row[r], lat_col[r], half))
        edge_lat_bank[r] = lat_key[kl]

    # ---- labels + scalar covariates ----
    label = np.full(n_edge, -1, np.int8)
    for r in range(n_edge):
        if bool(ident_amb[r]): continue                        # ambiguous -> EXCLUDED (-1)
        lab = cls_to_label[int(cls[r])]
        if lab == 0 and int(shared_iid[r]) < 0: continue        # positive-capable but not uniquely identified -> exclude
        label[r] = lab
    conf_min = np.minimum(ap_s, lt_s); conf_prod = ap_s * lt_s; conf_asym = np.abs(ap_s - lt_s)
    fold = np.array([fold_of[c] for c in cid_arr], np.int8)

    counts = {LABEL_NAMES[i]: int((label == i).sum()) for i in range(4)}
    counts["ambiguous_excluded"] = int((label == -1).sum())
    print(f"  labels: {counts}")

    np.savez_compressed(a.out,
        ap_crops=np.asarray(ap_bank, np.float32), lat_crops=np.asarray(lat_bank, np.float32),
        edge_ap_bank_idx=edge_ap_bank, edge_lat_bank_idx=edge_lat_bank,
        label=label, case_global_idx=cgi.astype(np.int32), case_id=cid_arr, ap_idx=apx.astype(np.int32),
        lat_idx=ltx.astype(np.int32), shared_iid=shared_iid.astype(np.int32), fold=fold,
        dsi_vox=dsi, ap_score=ap_s, lat_score=lt_s, conf_min=conf_min.astype(np.float32),
        conf_prod=conf_prod.astype(np.float32), conf_asym=conf_asym.astype(np.float32),
        ap_is_dup=ap_dup, lat_is_dup=lat_dup, class_names=np.asarray(class_names), label_names=np.asarray(LABEL_NAMES),
        val_case_ids=np.asarray(val_ids), k_folds=np.int32(K_FOLDS), crop_half=np.int32(half),
        # provenance
        data_sha256=np.asarray(data_sha), source_image_sha256=np.asarray(img_sha),
        detector_run=np.asarray(str(a.detector_run)), pairs_npz=np.asarray(str(a.pairs_npz)),
        appearance_kind=np.asarray("image_crop"),
        **({f"detector_{v}_sha256": np.asarray(ck[v]) for v in ck}))

    print(f"\nD2a APPEARANCE DATASET — {n_edge} edges | crop {2*half}x{2*half} | ap-bank {len(ap_bank)} lat-bank {len(lat_bank)}")
    print(f"  labels (row-aligned to D0): positive {counts['positive']} | cross-instance {counts['cross_instance']} "
          f"| one-sided {counts['one_sided']} | fully-false {counts['fully_false']} | ambiguous-excluded {counts['ambiguous_excluded']}")
    print(f"  case folds: {K_FOLDS} (by sorted case id) | provenance {'checkpoint-SHA' if ck else 'run-path'} | image-sha {img_sha[:12]}..")
    print(f"  D2b: join by row order with {a.pairs_npz.name} (coords/geometry) + train two-tower on these crops+scalars, case-CV.")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
