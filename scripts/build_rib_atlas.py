#!/usr/bin/env python3
"""Build and FREEZE a patient-adapted rib-cage ATLAS from DEVELOPMENT rib centerlines (RibSeg v2).

The atlas is a single MEAN rib-cage template in the RAS world-anatomical frame, normalized per cage
by translation + per-axis (anisotropic) scale ONLY — no rotation, so the template axes stay LR/AP/SI
for the reconstruction's independent per-axis scaling. It stores mean + dispersion so downstream
reconstruction can report geometry error and coverage:
  template            [R,K,3]  mean normalized centerline per rib slot
  pointwise_std       [R,K,3]  per-point std across cases (dispersion)
  case_count_per_rib  [R]
  rib_side / rib_num           anatomy label per slot, by MAJORITY VOTE across cases (+ agreement)

Honesty discipline: built OFFLINE from DEVELOPMENT CT centerlines only (sealed test excluded);
CT/RibSeg geometry is used at inference NEVER — only to score reconstruction fidelity on held-out
data. Anchoring MATCHES extract_crops.py so normalized position s aligns with the addressing model.

Output (immutable, staged then renamed): rib_atlas.npz + rib_atlas_manifest.json. Refuses overwrite.

Usage:
  python build_rib_atlas.py --cl-dir data/ribseg/ribseg_v2/cl \
      --image-dirs data/ribfrac_train/Part1 data/ribfrac \
      --det-manifest outputs/det_out_v2/det_manifest.json --out outputs/rib_atlas_v1 --k 60
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except Exception:  # noqa: BLE001
    nib = None

ATLAS_VERSION = "ribassist-atlas-2"


def _num(n):
    m = re.search(r"RibFrac0*(\d+)", n, re.I); return int(m.group(1)) if m else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def find_affine(cid, image_dirs):
    for d in image_dirs:
        for pat in (f"{cid}-image.nii.gz", f"{cid}-image.nii", f"{cid}-label.nii.gz", f"{cid}-label.nii"):
            p = Path(d) / pat
            if p.exists(): return nib.load(str(p)).affine
    for d in image_dirs:
        for p in Path(d).rglob(f"{cid}-*.nii*"):
            return nib.load(str(p)).affine
    return None


def anchor_rib(world):
    """extract_crops.py anchoring: first points have smaller world-y than last (s=0 at spine end)."""
    if world[:50, 1].mean() > world[-50:, 1].mean(): world = world[::-1]
    return world


def resample(world, K):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(world, axis=0), axis=1))]
    if d[-1] < 1e-6: return np.repeat(world[:1], K, axis=0)
    u = np.linspace(0, d[-1], K)
    return np.stack([np.interp(u, d, world[:, k]) for k in range(3)], axis=1)


# RibSeg v2 fixed slot partition: slots 0-11 are one anatomical side, 12-23 the other. The
# assignment of WHICH group is L vs R is NOT guessed from geometry — it is anchored to the NIfTI
# physical convention (affines map voxel->RAS world, so +world_x = patient Right, +world_z =
# Superior). This anchor cannot be systematically reversed the way an arbitrary geometric rule could.
RIBSEG_SIDE_GROUPS = [set(range(0, 12)), set(range(12, 24))]


def derive_side_num(cages_norm, valid):
    """Per-case (side, num) per rib slot, RAS-ANCHORED via the SHARED convention primitive
    (rib_labeling.assign_side_num: side by normalized-x +x=Right vs per-case spine midline; num by
    superior->inferior RAS +z), then MAJORITY VOTE across cases. Sharing the primitive with the
    addressing dataset guarantees 'R7' means the same rib on both sides of the pipeline."""
    from rib_labeling import assign_side_num
    R = cages_norm.shape[1]; votes_side = [{} for _ in range(R)]; votes_num = [{} for _ in range(R)]
    for ci in range(len(cages_norm)):
        cg = cages_norm[ci]; vv = valid[ci]
        cx = cg[:, :, 0].mean(1); cz = cg[:, :, 2].mean(1)
        validr = [r for r in range(R) if vv[r]]
        if not validr: continue
        med = float(np.median(cx[validr]))
        sn = assign_side_num([cx[r] for r in validr], [cz[r] for r in validr], med, keys=validr)
        for r in validr:
            sd, num = sn[r]
            votes_side[r][sd] = votes_side[r].get(sd, 0) + 1
            votes_num[r][num] = votes_num[r].get(num, 0) + 1
    rib_side = np.array(["?"] * R, dtype="U1"); rib_num = np.zeros(R, int); per_slot = np.zeros(R)
    for r in range(R):
        if votes_side[r]:
            sd = max(votes_side[r], key=votes_side[r].get); rib_side[r] = sd
            a_s = votes_side[r][sd] / sum(votes_side[r].values())
            nn = max(votes_num[r], key=votes_num[r].get); rib_num[r] = nn
            a_n = votes_num[r][nn] / sum(votes_num[r].values())
            per_slot[r] = min(a_s, a_n)   # per-slot agreement = weaker of side / num consensus
    return rib_side, rib_num, per_slot


def assert_valid_mapping(rib_side, rib_num, keep, per_slot, min_agree=0.75):
    """Fail LOUDLY unless the mapping is (a) a clean 12L+12R bijection, (b) high per-slot agreement,
    and (c) consistent with the RibSeg fixed slot PARTITION (slots 0-11 all one side, 12-23 all the
    other). (c) is the authoritative cross-check the bijection alone cannot provide — it catches a
    stable left/right swap, which a bijection + high agreement would otherwise pass."""
    L = [int(rib_num[r]) for r in range(len(keep)) if keep[r] and rib_side[r] == "L"]
    Rr = [int(rib_num[r]) for r in range(len(keep)) if keep[r] and rib_side[r] == "R"]
    if sorted(L) != list(range(1, 13)) or sorted(Rr) != list(range(1, 13)):
        raise ValueError(f"Mapping is not 12L+12R with ribs 1-12 once each: L={sorted(L)} R={sorted(Rr)}")
    for g in RIBSEG_SIDE_GROUPS:  # every slot in a RibSeg group must share one side
        sides = {rib_side[r] for r in g if keep[r]}
        if len(sides) > 1:
            raise ValueError(f"RibSeg slot group {sorted(g)} spans both sides {sides}; slot partition violated.")
    low = [r for r in range(len(keep)) if keep[r] and per_slot[r] < min_agree]
    if low:
        raise ValueError(f"Per-slot label agreement < {min_agree} for slots {low} "
                         f"(agreement {[round(float(per_slot[r]),2) for r in low]}); geometry mapping is unstable.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cl-dir", type=Path, required=True)
    ap.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--det-manifest", type=Path, required=True)
    ap.add_argument("--build-cases", type=Path, default=None,
                    help="optional JSON with a 'build' list (from freeze_geometry_split.py): restrict the atlas to "
                         "those cases so geometry-validation cases are held out (leakage control)")
    ap.add_argument("--out", type=Path, default=Path("outputs/rib_atlas_v1"))
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    if nib is None: print("pip install nibabel", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; the frozen atlas is immutable — use a new dir.")

    man = json.loads(a.det_manifest.read_text())
    dev_cases = list(man.get("dev_case_sources", {}).keys()) or man.get("dev_cases", [])
    if not dev_cases: raise ValueError("Manifest has no dev case list (dev_case_sources).")
    test_cases = set(man.get("test_cases", []))
    dev_cases = [c for c in dev_cases if c not in test_cases]
    build_note = "all dev cases"
    if a.build_cases:
        bset = set(json.loads(a.build_cases.read_text())["build"])
        stray = bset - set(dev_cases)
        if stray: raise ValueError(f"--build-cases contains non-dev/sealed-test ids: {sorted(stray)[:5]}")
        dev_cases = [c for c in dev_cases if c in bset]; build_note = f"atlas-build subset ({len(dev_cases)}) from {a.build_cases.name}"
    if a.n: dev_cases = dev_cases[: a.n]
    print(f"building atlas from {len(dev_cases)} DEV cases (sealed test excluded) ...", file=sys.stderr, flush=True)

    R = 24; cages = []; valids = []; used = []; skipped = {}
    for i, cid in enumerate(dev_cases, 1):
        clp = a.cl_dir / f"{cid}.npz"
        if not clp.exists(): skipped[cid] = "no centerline"; continue
        A = find_affine(cid, a.image_dirs)
        if A is None: skipped[cid] = "no affine"; continue
        cl = np.load(clp)["cl"]; Rm = A[:3, :3]; t = A[:3, 3]
        valid = ~np.all(cl.reshape(R, -1) == 0, axis=1)
        if valid.sum() < 6: skipped[cid] = "too few valid ribs"; continue
        vpts = np.concatenate([cl[r] @ Rm.T + t for r in range(R) if valid[r]], 0)  # VALID ribs only
        center = vpts.mean(0); scale = (np.percentile(vpts, 97.5, 0) - np.percentile(vpts, 2.5, 0)) / 2.0
        scale[scale < 1e-3] = 1.0
        cage = np.zeros((R, a.k, 3), np.float64)
        for r in range(R):
            if not valid[r]: continue
            cage[r] = (resample(anchor_rib(cl[r] @ Rm.T + t), a.k) - center) / scale
        cages.append(cage); valids.append(valid); used.append(cid)
        if i % 25 == 0 or i == len(dev_cases): print(f"[{i}/{len(dev_cases)}] processed", file=sys.stderr, flush=True)
    if not cages: raise RuntimeError("No usable dev cases.")
    cages = np.array(cages); valids = np.array(valids)  # [C,R,K,3], [C,R]

    # NO rotational Procrustes: centerlines are already in a common RAS world-anatomical frame, and
    # a general rotation would mix the LR/AP/SI axes that reconstruction scales independently. Mean
    # over translation + per-axis-scale-normalized cages only (keeps the anatomical coordinate basis).
    def masked_mean(cg, vd):
        m = np.zeros((R, a.k, 3)); cnt = np.zeros(R)
        for ci in range(len(cg)):
            for r in range(R):
                if vd[ci, r]: m[r] += cg[ci, r]; cnt[r] += 1
        keep = cnt > 0; m[keep] /= cnt[keep, None, None]; return m, cnt
    mean, cnt = masked_mean(cages, valids)
    # dispersion
    std = np.zeros((R, a.k, 3))
    for r in range(R):
        pts = np.array([cages[ci, r] for ci in range(len(cages)) if valids[ci, r]])
        if len(pts) > 1: std[r] = pts.std(0)
    rib_side, rib_num, per_slot = derive_side_num(cages, valids)
    keep = cnt > 0
    if not a.n:
        assert_valid_mapping(rib_side, rib_num, keep, per_slot)   # bijection + per-slot agreement or fail
    mapping_agree = float(per_slot[keep].mean()) if keep.any() else 0.0
    min_slot_agree = float(per_slot[keep].min()) if keep.any() else 0.0

    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / "rib_atlas.npz", template=mean.astype(np.float32), pointwise_std=std.astype(np.float32),
                        case_count_per_rib=cnt.astype(np.int32), rib_side=rib_side.astype("U1"),
                        rib_num=rib_num.astype(np.int16), valid=keep, k=np.int32(a.k))
    atlas_sha = sha256_file(work / "rib_atlas.npz")
    (work / "rib_atlas_manifest.json").write_text(json.dumps({
        "atlas_version": ATLAS_VERSION, "atlas_sha256": atlas_sha, "k": a.k, "n_rib_slots": R,
        "n_dev_cases_used": len(used), "n_skipped": len(skipped), "skipped": skipped,
        "alignment": "per-cage translation + per-axis (anisotropic) scale ONLY; NO rotation (keeps LR/AP/SI axes "
                     "for independent per-axis reconstruction scaling); centerlines already in common RAS world frame",
        "frame": "normalized: (world_mm - VALID-rib centroid) / per-axis robust half-extent (2.5-97.5 pct of valid-rib points)",
        "anchoring": "apex-anchored by world-y, identical to extract_crops.py (s aligns with addressing model)",
        "side_num_derivation": "RAS-ANCHORED (side by +world_x=Right, num by +world_z=Superior) + MAJORITY VOTE, "
                               "ASSERTED to be a 12L+12R bijection consistent with the RibSeg slot partition "
                               "(0-11 / 12-23) with per-slot agreement >= 0.75",
        "side_num_agreement_mean": round(mapping_agree, 4), "side_num_agreement_min_slot": round(min_slot_agree, 4),
        "dispersion_stored": "pointwise_std [R,K,3], case_count_per_rib [R] (enables geometry error + coverage eval)",
        "honesty": "DEVELOPMENT CT centerlines ONLY; sealed test excluded; CT used at inference NEVER, only to score fidelity",
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__},
        "det_manifest": str(a.det_manifest), "build_scope": build_note, "build_cases_json": (str(a.build_cases) if a.build_cases else None),
        "dev_cases_used": used, "partial_debug": bool(a.n),
        "source_dataset_version": man.get("dataset_version")}, indent=2))
    if a.n:
        print(f"\n[debug] partial atlas in {work} (NOT promoted; capped run).", file=sys.stderr)
    else:
        work.rename(a.out); print(f"\nwrote rib_atlas.npz + manifest to {a.out}/  (sha256 {atlas_sha[:12]}..)")
    print(f"atlas: {int(keep.sum())}/{R} rib slots from {len(used)} dev cases; label majority agreement "
          f"{mapping_agree:.3f}; sides L/R = {int((rib_side[keep]=='L').sum())}/{int((rib_side[keep]=='R').sum())}")
    if skipped: print(f"skipped {len(skipped)}: e.g. {list(skipped.items())[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
