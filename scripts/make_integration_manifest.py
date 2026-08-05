#!/usr/bin/env python3
"""Build the VERSIONED RibAssist 3D integration manifest: one immutable JSON that pins every artifact and
config value the end-to-end pipeline depends on, with cross-checked hashes, so a reviewer can prove the
demo / sealed-test pipeline is exactly this set of frozen components and nothing drifted between them.

It is FAIL-CLOSED and READ-ONLY (writes only the manifest): every required hash must be present AND
consistent, or it aborts. It pins:
  * frozen detector: protocol-lock sha (detector_protocol_frozen.json) OR the dev record, weight hashes,
    architecture, si_tol, the FUSION operating threshold, the unmatched-lateral gate, primary condition;
  * addressing checkpoint: state_dict hash, views, use_pos (asserted False for the deployed model), crop,
    deployment epochs, and the dataset lineage it was aligned to;
  * canonical atlas: version + sha (verified against rib_atlas.npz);
  * dataset lineage: det_dev.npz sha, cross-checked to equal BOTH the detector's and the addressing
    model's recorded det_dev hash (single source of truth);
  * pipeline code: git commit (if available) + per-file sha of every script in the inference path.

Run it AFTER freeze_detector.py, pointed at the frozen detector dir, so the manifest pins the deployed
artifacts. (Detector weight hashes survive the freeze copy bit-identically, so the pins are stable.)

Usage:
  python make_integration_manifest.py \
      --detector-frozen outputs/detector_frozen_v1 \
      --address-model outputs/addressing_model_nopos \
      --atlas outputs/rib_atlas_build \
      --data outputs/det_out_v2/det_dev.npz \
      --code-dir scripts \
      --out outputs/integration_manifest_v1.json
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

# every script that participates in the deployed inference / reconstruction / summary path
PIPELINE_CODE = ["run_ribassist.py", "train_detector.py", "train_address.py", "rib_labeling.py",
                 "make_address_data_detframe.py", "make_rib_targets.py", "reconstruct_3d.py",
                 "build_trauma_summary.py", "build_rib_atlas.py", "evaluate_detector.py",
                 "eval_sealed_test.py", "verify_candidate_identity.py"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def require(d, key, source):
    v = d.get(key)
    if v is None or v == "" or v == {}:
        raise ValueError(f"{source} is missing required field {key!r}")
    return v


def load_detector(det_dir):
    """Prefer the frozen protocol record; fall back to the dev record. Returns (record, record_path,
    record_sha, is_frozen)."""
    frozen = det_dir / "detector_protocol_frozen.json"; dev = det_dir / "detector_dev_run.json"
    if frozen.exists():
        return json.loads(frozen.read_text()), frozen, sha256_file(frozen), True
    if dev.exists():
        return json.loads(dev.read_text()), dev, sha256_file(dev), False
    raise FileNotFoundError(f"No detector_protocol_frozen.json or detector_dev_run.json in {det_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-frozen", "--detector-run", dest="detector", type=Path, required=True,
                    help="frozen detector dir (preferred) or the scored dev-run dir")
    ap.add_argument("--address-model", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="det_dev.npz the pipeline is aligned to")
    ap.add_argument("--code-dir", type=Path, default=Path("scripts"), help="dir holding the pipeline scripts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stamp", type=str, default=None, help="optional ISO timestamp to record (this tool does not read the clock)")
    a = ap.parse_args()
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; the integration manifest is versioned — use a new path.")

    # ---- dataset lineage: the single source of truth every artifact must agree with ----
    data_sha = sha256_file(a.data)

    # ---- detector ----
    rec, rec_path, rec_sha, is_frozen = load_detector(a.detector)
    lp = require(rec, "learning_procedure", str(rec_path)); ep = require(rec, "eval_params", str(rec_path))
    det_hashes = require(rec, "detector_sha256", str(rec_path))
    for v in ("ap", "lat"):
        want = require(det_hashes, v, "detector record detector_sha256")
        got = sha256_file(a.detector / f"detector_{v}.pt")
        if got != want: raise ValueError(f"detector_{v}.pt hash {got[:12]}.. != record {want[:12]}..")
    det_data_sha = require(rec, "det_dev_sha256", str(rec_path))
    if det_data_sha != data_sha:
        raise ValueError(f"det_dev mismatch: --data {data_sha[:12]}.. != detector record {det_data_sha[:12]}..")
    fusion_op = require(lp.get("operating_threshold_per_condition", {}), "fusion", "detector learning_procedure")
    lat_gate = require(lp.get("biplanar_fusion", {}), "unmatched_lateral_score_gate", "detector learning_procedure")
    arch = rec.get("model_arch") or {"kind": "scratch_unet", "base_ch": ep.get("base_ch")}

    # ---- addressing ----
    acfg = json.loads((a.address_model / "addressing_model.json").read_text())
    if bool(acfg.get("use_pos", True)) is not False:
        raise ValueError("deployed addressing model must have use_pos=false (the selected no-position config).")
    a_state_want = require(acfg, "state_dict_sha256", "addressing_model.json")
    a_state_got = sha256_file(a.address_model / "addressing_model.pt")
    if a_state_got != a_state_want:
        raise ValueError(f"addressing checkpoint hash {a_state_got[:12]}.. != config {a_state_want[:12]}..")
    a_aligned = require(acfg, "aligned_det_dev_sha256", "addressing_model.json")
    if a_aligned != data_sha:
        raise ValueError(f"det_dev mismatch: --data {data_sha[:12]}.. != addressing aligned {a_aligned[:12]}..")

    # ---- atlas ----
    amon = json.loads((a.atlas / "rib_atlas_manifest.json").read_text())
    atlas_sha_want = require(amon, "atlas_sha256", "rib_atlas_manifest.json")
    atlas_sha_got = sha256_file(a.atlas / "rib_atlas.npz")
    if atlas_sha_got != atlas_sha_want:
        raise ValueError(f"rib_atlas.npz hash {atlas_sha_got[:12]}.. != manifest {atlas_sha_want[:12]}..")

    # ---- pipeline code lineage ----
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=a.code_dir).stdout.strip() or None
    except Exception:  # noqa: BLE001
        git = None
    code_sha = {}
    for name in PIPELINE_CODE:
        p = a.code_dir / name
        code_sha[name] = sha256_file(p) if p.exists() else None
    missing_code = [n for n, s in code_sha.items() if s is None]

    manifest = {
        "schema": "ribassist-integration-manifest-1",
        "stamp": a.stamp,
        "dataset_lineage": {"det_dev": str(a.data), "det_dev_sha256": data_sha,
                            "note": "single source of truth; detector and addressing both pinned to this hash"},
        "detector": {"dir": str(a.detector), "is_frozen": is_frozen, "record": rec_path.name,
                     "record_sha256": rec_sha, "architecture": arch, "si_tol": ep.get("si_tol"),
                     "weights_sha256": {v: det_hashes[v] for v in ("ap", "lat")},
                     "fusion_operating_threshold": fusion_op, "unmatched_lateral_score_gate": lat_gate,
                     "primary_condition": rec.get("primary_condition"),
                     "best_epoch_per_view": lp.get("best_epoch_per_view"), "epochs": lp.get("epochs"),
                     "boundary_epoch_allowed": bool(rec.get("overlay_qa")) and any(
                         e == lp.get("epochs") for e in (lp.get("best_epoch_per_view") or {}).values()),
                     "overlay_qa": rec.get("overlay_qa")},
        "addressing": {"dir": str(a.address_model), "views": acfg.get("views"), "use_pos": bool(acfg.get("use_pos")),
                       "crop": acfg.get("crop"), "deployment_epochs": acfg.get("deployment_epochs"),
                       "state_dict_sha256": a_state_want, "aligned_det_dev_sha256": a_aligned,
                       "address_dataset": acfg.get("address_dataset"),
                       "address_dataset_sha256": acfg.get("address_dataset_sha256")},
        "atlas": {"dir": str(a.atlas), "atlas_version": amon.get("atlas_version"), "atlas_sha256": atlas_sha_want,
                  "frame": amon.get("frame"), "anchoring": amon.get("anchoring")},
        "pipeline_code": {"git_commit": git, "code_dir": str(a.code_dir), "file_sha256": code_sha,
                          "missing": missing_code},
        "consistency": {"det_dev_single_source": True, "detector_weights_verified": True,
                        "addressing_checkpoint_verified": True, "atlas_verified": True,
                        "addressing_use_pos_false": True},
    }
    # self-hash: sha256 of the manifest with the self-hash field absent, so the file certifies its own content
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=2))

    print(f"wrote integration manifest -> {a.out}")
    print(f"  dataset det_dev_sha256      {data_sha[:16]}..  (detector & addressing both agree)")
    print(f"  detector record_sha256      {rec_sha[:16]}..  ({'FROZEN' if is_frozen else 'DEV — freeze first'})")
    print(f"  fusion op threshold / gate  {fusion_op} / {lat_gate}")
    print(f"  addressing state_dict       {a_state_want[:16]}..  use_pos={bool(acfg.get('use_pos'))} views={acfg.get('views')}")
    print(f"  atlas {amon.get('atlas_version')}         {atlas_sha_want[:16]}..")
    print(f"  code git_commit             {git or '(none)'}" + (f"   MISSING: {missing_code}" if missing_code else ""))
    print(f"\n>>> INTEGRATION LOCK: manifest_sha256 = {manifest['manifest_sha256']}")
    if not is_frozen:
        print("    NOTE: detector is a DEV run, not frozen. Freeze first (freeze_detector.py), then regenerate this "
              "manifest against the frozen dir so it pins the deployed protocol lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
