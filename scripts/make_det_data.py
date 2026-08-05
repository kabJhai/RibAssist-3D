#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Detector dataset for RibAssist 3D (full-image fracture detection stage).

Renders each case's full AP and lateral CT-DERIVED ORTHOGRAPHIC LINE-INTEGRAL ATTENUATION
PROJECTION (a simplified simulated radiograph — NOT a calibrated X-ray forward model) and
builds per-view heatmaps centered on projected fracture-instance footprints (Gaussian at
each footprint's distance-transform interior point), plus the projected CT-ANNOTATION
FOOTPRINTS (compact numeric CSR — no pickle) for tolerant lesion-aware one-to-one FROC
matching. (A footprint is an annotation projection, not proof of visible fracture evidence.) Images and
labels are canonicalized (nib.as_closest_canonical) so patients are not flipped relative to
each other.

TEST SEALING is structural. Three artifacts:
  det_dev.npz          — development images + heatmaps + points + footprints + ids (training)
  det_test_inputs.npz  — sealed-test images + geometry ONLY (no labels/points/ids/counts)
  det_test_gt.npz      — sealed-test points + footprints + ids + counts (ONLY the final eval)
plus det_manifest.json (config, software versions, split hash, artifact sha256). Training
loads det_dev.npz only. Footprints are stored as CSR: *_fp_pts [P,2] int16 + *_fp_ptr [I+1]
int32, with fp_case [I] and fp_iid [I]; instance i footprint = pts[ptr[i]:ptr[i+1]].

Usage:
  python make_det_data.py --ribfrac-dir data/ribfrac_train --split-json frozen_split.json \
      --out det_out --size 256 --sigma 4 --overlays 12
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    import scipy
    from scipy.ndimage import zoom, distance_transform_edt
except Exception:  # noqa: BLE001
    nib = None

MAX_FRAC = 256   # sanity guard only (RibFrac cases reach dozens); per-instance data is CSR, not fixed-width
GEN_VERSION = "ribassist-detdata-7"
PROTOCOL_VERSION = "detector-protocol-v1"   # DATA + EVALUATION protocol ONLY (not the learning procedure)
PROTOCOL_SIZE = 256
PROTOCOL_SIGMA_PX = 4.0
NMS_RADIUS_PX = 5
MATCH_RADIUS_PX = 8
ORIENT = {"ap_array": "rows=inferior->superior; cols=patient left->right in canonical RAS array coords",
          "lat_array": "rows=inferior->superior; cols=posterior->anterior",
          "display_note": "AP arrays are NOT horizontally flipped into conventional radiographic viewing "
                          "(anatomical/array convention). Orientation is verified by affine axis codes (R,A,S), "
                          "not by visual landmarks.",
          "projection_note": "projection is along canonical voxel axes after as_closest_canonical; residual "
                             "obliquity is not resampled away (acceptable for this study, acknowledged)."}


def _num(n):
    m = re.search(r"RibFrac0*(\d+)", n, re.I); return int(m.group(1)) if m else None


def find_cases(rfd):
    rf = {}
    for p in Path(rfd).rglob("*"):
        if not p.is_file(): continue
        if re.fullmatch(r"RibFrac\d+-image\.nii(\.gz)?", p.name, re.I): rf.setdefault(_num(p.name), {})["image"] = p
        elif re.fullmatch(r"RibFrac\d+-label\.nii(\.gz)?", p.name, re.I): rf.setdefault(_num(p.name), {})["label"] = p
    img_ids = {k for k, v in rf.items() if "image" in v}; lab_ids = {k for k, v in rf.items() if "label" in v}
    if img_ids != lab_ids:
        raise ValueError(f"Unpaired files: images-only={sorted(img_ids - lab_ids)[:5]}, labels-only={sorted(lab_ids - img_ids)[:5]}")
    return [{"cid": f"RibFrac{k}", "image": rf[k]["image"], "label": rf[k]["label"]} for k in sorted(img_ids)]


def project(mu, pa, ra, ca, sp):
    line = (mu * sp[pa]).sum(axis=pa); rem = [a for a in (0, 1, 2) if a != pa]
    img = np.transpose(line, (rem.index(ra), rem.index(ca)))
    img = img - img.min(); return (img / (img.max() + 1e-8)).astype(np.float32)


def resize_pad(img, S):
    H0, W0 = img.shape; scale = min(S / H0, S / W0); r = zoom(img, (scale, scale), order=1)
    nh, nw = min(r.shape[0], S), min(r.shape[1], S); r = r[:nh, :nw]
    pt, pl = (S - nh) // 2, (S - nw) // 2; out = np.zeros((S, S), np.float32); out[pt:pt + nh, pl:pl + nw] = r
    return out, scale, pt, pl


def footprint_padded(vox, ra, ca, scale, pt, pl, S):
    """Projected-instance footprint in padded pixel coords (unique int16 [k,2])."""
    r = np.clip((vox[ra] * scale + pt).round(), 0, S - 1).astype(np.int16)
    c = np.clip((vox[ca] * scale + pl).round(), 0, S - 1).astype(np.int16)
    return np.unique(np.stack([r, c], 1), axis=0)


def center_from_footprint(fp):
    """Distance-transform interior point of the (padded) footprint raster — the Gaussian
    center, derived from exactly the same geometry as the stored footprint."""
    r, c = fp[:, 0], fp[:, 1]; r0, c0 = int(r.min()), int(c.min())
    m = np.zeros((int(r.max()) - r0 + 1, int(c.max()) - c0 + 1), bool); m[r - r0, c - c0] = True
    dt = distance_transform_edt(m); rr, cc = np.unravel_index(int(dt.argmax()), dt.shape)
    return float(rr + r0), float(cc + c0)


def heatmap(points, S, sigma):
    hm = np.zeros((S, S), np.float32)
    if not points: return hm
    yy, xx = np.mgrid[0:S, 0:S]
    for (r, c) in points:
        hm = np.maximum(hm, np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2)))
    return hm


def csr(fp_list):
    if not fp_list: return np.zeros((0, 2), np.int16), np.array([0], np.int32)
    ptr = np.zeros(len(fp_list) + 1, np.int32)
    for i, fp in enumerate(fp_list): ptr[i + 1] = ptr[i] + len(fp)
    return np.concatenate(fp_list, 0).astype(np.int16), ptr


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ribfrac-dir", type=Path, required=True); ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("det_out")); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=4); ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--expect", type=int, default=300); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--extra-dirs", type=Path, nargs="*", default=[], help="dev-only cases (e.g. val set, for negatives)")
    ap.add_argument("--dataset-tag", default="v1", help="dataset version tag recorded in manifest (bump when composition changes)")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    if nib is None: print("pip install nibabel scipy matplotlib", file=sys.stderr); return 1
    if a.n and not a.allow_partial:
        raise ValueError("--n requires --allow-partial and a separate output directory (debug only).")
    if a.n and a.out == Path("outputs/det_out"):
        raise ValueError("Partial runs may not write to the production dataset directory outputs/det_out.")
    if a.size != PROTOCOL_SIZE or a.sigma != PROTOCOL_SIGMA_PX:
        raise ValueError(f"{PROTOCOL_VERSION} requires --size {PROTOCOL_SIZE} and --sigma {PROTOCOL_SIGMA_PX}.")
    if not a.split_json.exists(): raise FileNotFoundError("A valid frozen split JSON is required.")
    sj_bytes = a.split_json.read_bytes(); sj = json.loads(sj_bytes)
    if not sj.get("test"): raise ValueError("Split JSON has no non-empty 'test' set.")
    test = set(sj["test"])
    cases = find_cases(a.ribfrac_dir)
    for c in cases: c["source"] = "train_part1"
    if a.n: cases = cases[: a.n]
    elif len(cases) != a.expect: raise ValueError(f"Expected {a.expect} paired cases, found {len(cases)}")
    if not a.n and (test - {c["cid"] for c in cases}): raise ValueError("Frozen test cases missing from cohort.")
    for d in a.extra_dirs:  # dev-only cohorts (e.g. validation set); may NOT contain any frozen-test case
        ec = find_cases(d); bad = test & {c["cid"] for c in ec}
        if bad: raise ValueError(f"Extra dir {d} contains frozen-test cases: {sorted(bad)[:5]}")
        for c in ec: c["source"] = f"extra:{d.resolve()}"  # resolved path avoids basename collisions
        cases += ec
    seen = set()
    for c in cases:
        if c["cid"] in seen: raise ValueError(f"Duplicate case {c['cid']} across dirs")
        seen.add(c["cid"])
    # Immutability guard BEFORE any output mutation: refuse to touch an existing production dir
    # (including its overlays). Production output is built in a staging dir and atomically renamed
    # into place only after every assertion, hash, and the manifest have succeeded.
    prod = [a.out / f for f in ("det_dev.npz", "det_test_inputs.npz", "det_test_gt.npz", "det_manifest.json")]
    if not a.n and (a.out.exists() or any(p.exists() for p in prod)):
        raise FileExistsError(f"Output directory already exists: {a.out}. Use a new versioned directory "
                              "(never overwrite a frozen dataset).")
    work = a.out if a.n else a.out.parent / f".{a.out.name}.tmp"
    if not a.n and work.exists():
        shutil.rmtree(work)  # clear a stale staging dir from an interrupted run
    # staging dir MUST share a.out's parent so the final work.rename(a.out) is an atomic
    # same-filesystem rename (it will fail loudly, not silently copy, across mounts)
    (work / "overlay").mkdir(parents=True, exist_ok=True)
    dev_cids = [c["cid"] for c in cases if c["cid"] not in test]
    # Overlay selection is DETERMINISTIC: the fixed seed means regenerating this dataset (same
    # cohort + split) audits the identical set of cases. Overlays are a QA aid, not part of the data.
    rs = np.random.RandomState(0)
    overlay_set = set(rs.choice(dev_cids, size=min(len(dev_cids), a.overlays + 6), replace=False).tolist()) if dev_cids else set()

    DEV = {k: [] for k in ("ap", "lat", "ap_hm", "lat_hm", "nfrac", "ap_geo", "lat_geo", "ap_sp", "lat_sp", "case", "source")}
    TIN = {k: [] for k in ("ap", "lat", "ap_geo", "lat_geo", "ap_sp", "lat_sp", "case")}
    TGT = {k: [] for k in ("nfrac", "case")}
    d_apfp, d_latfp, d_apctr, d_latctr, d_fpcase, d_fpiid = [], [], [], [], [], []  # dev instance-level (CSR)
    t_apfp, t_latfp, t_apctr, t_latctr, t_fpcase, t_fpiid = [], [], [], [], [], []  # test GT instance-level
    ns = dev_frac = 0
    print(f"rendering detector data for {len(cases)} cases (size {a.size}) ...", file=sys.stderr, flush=True)
    for i, cs in enumerate(cases, 1):
        ct = nib.as_closest_canonical(nib.load(str(cs["image"])))
        lab_raw = nib.load(str(cs["label"])); lab = nib.as_closest_canonical(lab_raw)
        if ct.shape != lab.shape:
            raise ValueError(f"{cs['cid']}: canonical image/label shape mismatch {ct.shape} vs {lab.shape}")
        if not np.allclose(ct.affine, lab.affine, atol=1e-3):
            raise ValueError(f"{cs['cid']}: canonical image/label affine mismatch")
        A = ct.affine
        axc = nib.aff2axcodes(A)  # authoritative orientation check (tolerates small obliquity)
        if axc != ("R", "A", "S"):
            raise ValueError(f"{cs['cid']}: expected canonical RAS+ orientation, got {axc}")
        lr, apx, si = 0, 1, 2  # canonical RAS+ voxel axes, guaranteed by the assertion above
        sp = nib.affines.voxel_sizes(A)  # mm/voxel along those canonical axes (no argmax inference)
        fl = np.asarray(lab.get_fdata()).astype(int)
        rawl, rawc = np.unique(np.asarray(lab_raw.dataobj).astype(np.int32), return_counts=True)
        canl, canc = np.unique(fl, return_counts=True)
        if not np.array_equal(rawl, canl): raise ValueError(f"{cs['cid']}: canonicalization changed label IDs")
        if not np.array_equal(rawc, canc): raise ValueError(f"{cs['cid']}: canonicalization changed label voxel counts")
        mu = np.clip(1 + ct.get_fdata() / 1000, 0, None).astype(np.float32)
        im_ap, im_lat = project(mu, apx, si, lr, sp), project(mu, lr, si, apx, sp)
        img_ap, sa, pta, pla = resize_pad(im_ap, a.size); img_lat, sl, ptl, pll = resize_pad(im_lat, a.size)
        labs = [int(v) for v in np.unique(fl) if v != 0]
        if len(labs) > MAX_FRAC: raise ValueError(f"{cs['cid']}: {len(labs)} fractures exceeds MAX_FRAC={MAX_FRAC}")
        apts, lpts, ap_fps, lat_fps = [], [], [], []
        for lb in labs:
            vox = np.array(np.nonzero(fl == lb))
            afp = footprint_padded(vox, si, lr, sa, pta, pla, a.size)
            lfp = footprint_padded(vox, si, apx, sl, ptl, pll, a.size)
            apts.append(center_from_footprint(afp)); lpts.append(center_from_footprint(lfp))
            ap_fps.append(afp); lat_fps.append(lfp)
        ap_sp = [sp[si] / sa, sp[lr] / sa]; lat_sp = [sp[si] / sl, sp[apx] / sl]
        img_ap16, img_lat16 = img_ap.astype(np.float16), img_lat.astype(np.float16)
        if cs["cid"] in test:
            ci = len(TIN["case"])
            for k, v in (("ap", img_ap16), ("lat", img_lat16), ("ap_geo", [sa, pta, pla]), ("lat_geo", [sl, ptl, pll]), ("ap_sp", ap_sp), ("lat_sp", lat_sp), ("case", cs["cid"])): TIN[k].append(v)
            for k, v in (("nfrac", len(labs)), ("case", cs["cid"])): TGT[k].append(v)
            for k in range(len(labs)):
                t_apfp.append(ap_fps[k]); t_latfp.append(lat_fps[k]); t_apctr.append(apts[k]); t_latctr.append(lpts[k]); t_fpcase.append(ci); t_fpiid.append(labs[k])
            print(f"[{i}/{len(cases)}] sealed test case processed", file=sys.stderr, flush=True)
        else:
            ci = len(DEV["case"])
            hm_ap, hm_lat = heatmap(apts, a.size, a.sigma), heatmap(lpts, a.size, a.sigma)
            for k, v in (("ap", img_ap16), ("lat", img_lat16), ("ap_hm", hm_ap.astype(np.float16)), ("lat_hm", hm_lat.astype(np.float16)),
                         ("nfrac", len(labs)), ("ap_geo", [sa, pta, pla]), ("lat_geo", [sl, ptl, pll]), ("ap_sp", ap_sp), ("lat_sp", lat_sp), ("case", cs["cid"]), ("source", cs["source"])): DEV[k].append(v)
            for k in range(len(labs)):
                d_apfp.append(ap_fps[k]); d_latfp.append(lat_fps[k]); d_apctr.append(apts[k]); d_latctr.append(lpts[k]); d_fpcase.append(ci); d_fpiid.append(labs[k])
            dev_frac += len(labs)
            print(f"[{i}/{len(cases)}] {cs['cid']}: {len(labs)} fx split=dev", file=sys.stderr, flush=True)
            if cs["cid"] in overlay_set and ns < a.overlays and apts:
                try:
                    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 2, figsize=(9, 5))
                    for x, im, hm, t in ((ax[0], img_ap, hm_ap, "AP"), (ax[1], img_lat, hm_lat, "lateral")):
                        x.imshow(im, cmap="gray", origin="lower"); x.imshow(hm, cmap="hot", alpha=0.45, origin="lower"); x.set_title(f"{cs['cid']} {t}"); x.axis("off")
                    fig.tight_layout(); fig.savefig(work / "overlay" / f"{cs['cid']}.png", dpi=100); plt.close(fig); ns += 1
                except Exception as e:  # noqa: BLE001
                    print(f"[overlay skip] {e}", file=sys.stderr)

    if not a.n:
        exp_test = len(test); exp_dev = len(cases) - exp_test
        if len(DEV["case"]) != exp_dev: raise RuntimeError(f"Expected {exp_dev} dev cases, generated {len(DEV['case'])}")
        if len(TIN["case"]) != exp_test: raise RuntimeError(f"Expected {exp_test} test cases, generated {len(TIN['case'])}")
        if set(TIN["case"]) != test: raise RuntimeError("Generated sealed-test case set differs from frozen split")

    d_ap_pts, d_ap_ptr = csr(d_apfp); d_lat_pts, d_lat_ptr = csr(d_latfp)
    t_ap_pts, t_ap_ptr = csr(t_apfp); t_lat_pts, t_lat_ptr = csr(t_latfp)
    dev_np = {k: np.array(v) for k, v in DEV.items()}
    dev_np.update(ap_fp_pts=d_ap_pts, ap_fp_ptr=d_ap_ptr, lat_fp_pts=d_lat_pts, lat_fp_ptr=d_lat_ptr,
                  ap_ctr=np.array(d_apctr, np.float32).reshape(-1, 2), lat_ctr=np.array(d_latctr, np.float32).reshape(-1, 2),
                  fp_case=np.array(d_fpcase, np.int32), fp_iid=np.array(d_fpiid, np.int16))
    tgt_np = {k: np.array(v) for k, v in TGT.items()}
    tgt_np.update(ap_fp_pts=t_ap_pts, ap_fp_ptr=t_ap_ptr, lat_fp_pts=t_lat_pts, lat_fp_ptr=t_lat_ptr,
                  ap_ctr=np.array(t_apctr, np.float32).reshape(-1, 2), lat_ctr=np.array(t_latctr, np.float32).reshape(-1, 2),
                  fp_case=np.array(t_fpcase, np.int32), fp_iid=np.array(t_fpiid, np.int16))
    np.savez_compressed(work / "det_dev.npz", **dev_np)
    np.savez_compressed(work / "det_test_inputs.npz", **{k: np.array(v) for k, v in TIN.items()})
    np.savez_compressed(work / "det_test_gt.npz", **tgt_np)
    hashes = {f: sha256(work / f) for f in ("det_dev.npz", "det_test_inputs.npz", "det_test_gt.npz")}
    (work / "det_manifest.json").write_text(json.dumps({
        "generator": GEN_VERSION, "dataset_version": a.dataset_tag, "created": datetime.now().isoformat(timespec="seconds"),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "nibabel": nib.__version__},
        "protocol": {"version": PROTOCOL_VERSION, "scope": "data + evaluation only (resolution, sigma, NMS radius, "
                     "matching radius); the LEARNING procedure is NOT frozen yet — freeze after dev experimentation, before test",
                     "resolution": a.size, "sigma_px": a.sigma, "nms_radius_px": NMS_RADIUS_PX, "matching_radius_px": MATCH_RADIUS_PX},
        "target_point": "distance-transform interior of projected CT-annotation footprint (training Gaussian center; not 3D centroid)",
        "eval_reference": "projected CT-ANNOTATION footprints (CSR: *_fp_pts/_ptr, fp_case, fp_iid) for tolerant lesion-aware matching; a footprint is an annotation projection, NOT proof every pixel shows visible radiographic fracture evidence",
        "partial_debug_run": bool(a.n), "max_frac": MAX_FRAC,
        "projection": "CT-derived orthographic line-integral attenuation projection, per-image min-max",
        "orientation": ORIENT, "resize": "isotropic min-ratio + center pad", "spacing": "post-resize mm/px per view",
        "split_json": str(a.split_json), "split_md5": hashlib.md5(sj_bytes).hexdigest(), "artifact_sha256": hashes,
        "primary_dir": str(a.ribfrac_dir), "extra_dev_dirs": [str(d) for d in a.extra_dirs],
        "n_dev": len(DEV["case"]), "n_test": len(TIN["case"]),
        "dev_source_counts": {s: int(DEV["source"].count(s)) for s in sorted(set(DEV["source"]))},
        "dev_case_sources": {cid: src for cid, src in zip(DEV["case"], DEV["source"])},  # supersedes a parallel dev_cases list
        "test_cases": TIN["case"]}, indent=2))
    if not a.n:
        work.rename(a.out)  # atomic promotion: staging dir becomes the frozen dataset only now that all writes succeeded
    print(f"\nwrote det_dev.npz ({len(DEV['case'])} dev) / det_test_inputs.npz + det_test_gt.npz ({len(TIN['case'])} sealed test)")
    print(f"development fractures: {dev_frac}   (test fracture count withheld until final evaluation)")
    dev_nf = np.asarray(DEV["nfrac"]) if DEV["nfrac"] else np.array([])
    pos = int((dev_nf > 0).sum()); neg = int((dev_nf == 0).sum())
    print(f"development composition: {pos} positive / {neg} negative (0-fracture) cases")
    if pos:
        fpc = dev_nf[dev_nf > 0]
        print(f"fractures per positive case: median {np.median(fpc):.0f}, IQR [{np.percentile(fpc,25):.0f}, {np.percentile(fpc,75):.0f}], max {int(fpc.max())}")
    if neg == 0:
        print("WARNING: no development NEGATIVE (0-fracture) cases — full-image false-positive evaluation "
              "is poorly supported. Add Part 2 / negative cases before trusting FROC.")
    print(f"artifact sha256 recorded in manifest. train on det_dev.npz ONLY; overlays (dev) in {a.out}/overlay/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
