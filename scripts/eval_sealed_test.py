#!/usr/bin/env python3
"""RibAssist 3D SEALED-TEST evaluator — the ONE confirmatory pass. Runs the FROZEN detector on the
sealed test cohort exactly once, using the EXACT frozen learning procedure, frozen FROC grids,
and frozen per-condition operating thresholds.

Refuses to run unless a valid detector_protocol_frozen.json exists AND its sha256 matches an
externally supplied --expected-protocol-sha256 (the digest you committed to git before opening
the test). This gives an EXTERNAL anchor: modifying the JSON and its internal weight hashes
together no longer passes, because the JSON's own digest would change. Every artifact hash is
verified before any metric is produced:
  * protocol JSON sha256 == --expected-protocol-sha256 (external anchor);
  * peak-extraction params in the JSON == this evaluator's module constants;
  * manifest's det_dev.npz sha == frozen record's (same dataset lineage);
  * det_test_inputs.npz / det_test_gt.npz hash to their manifest-recorded sha256;
  * detector weight files hash to the frozen record's detector_sha256.
Output (sealed_test_results.json) is built in a staging dir and atomically renamed; the target
dir must not already exist, so the confirmatory number is produced once and cannot be re-rolled.

Usage:
  python eval_sealed_test.py --frozen-dir outputs/detector_frozen_v1 \
      --test-dir outputs/det_out_v2 --out outputs/sealed_test_v1 \
      --expected-protocol-sha256 <digest printed by the freeze step>
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import train_detector as T   # reuse the SAME model, peaks, matching, FROC, CIs, and metrics

EVALUATOR_VERSION = "eval-sealed-test-1"


def _verify_sha(path, want, label):
    got = T.sha256_file(path)
    if got != want: raise ValueError(f"{label} sha256 mismatch: file {got[:12]}.. != expected {str(want)[:12]}..")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-dir", type=Path, required=True, help="dir with detector_protocol_frozen.json + detector_*.pt")
    ap.add_argument("--test-dir", type=Path, required=True, help="dir with det_test_inputs.npz, det_test_gt.npz, det_manifest.json")
    ap.add_argument("--out", type=Path, required=True, help="output dir (must NOT already exist)")
    ap.add_argument("--expected-protocol-sha256", required=True, help="externally committed sha256 of detector_protocol_frozen.json")
    a = ap.parse_args()
    if torch is None: print("pip install torch scipy", file=sys.stderr); return 1

    frozen_path = a.frozen_dir / "detector_protocol_frozen.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"No detector_protocol_frozen.json in {a.frozen_dir}; the learning procedure must be "
                                "FROZEN (train_detector.py --freeze-protocol) before the sealed test may be scored.")
    # EXTERNAL anchor: the JSON's own digest must match what was committed before opening the test.
    proto_sha = _verify_sha(frozen_path, a.expected_protocol_sha256, "detector_protocol_frozen.json")
    fr = json.loads(frozen_path.read_text())
    if not fr.get("frozen"): raise ValueError("Protocol JSON is not marked frozen; refusing to score the sealed test.")
    if fr.get("protocol") != T.PROTOCOL_VERSION: raise ValueError(f"Protocol mismatch: {fr.get('protocol')} != {T.PROTOCOL_VERSION}")
    pe = fr.get("learning_procedure", {}).get("peak_extraction")
    if pe != T.PEAK_EXTRACTION:
        raise ValueError(f"Peak-extraction params in protocol {pe} != this evaluator's constants {T.PEAK_EXTRACTION}.")
    if a.out.exists(): raise FileExistsError(f"Output dir {a.out} exists; the sealed-test result is immutable — use a new dir.")

    # ---- provenance: manifest + all four artifacts + detector weights ----
    man = json.loads((a.test_dir / "det_manifest.json").read_text())
    msha = man.get("artifact_sha256", {})
    if msha.get("det_dev.npz") != fr.get("det_dev_sha256"):
        raise ValueError("Dataset lineage mismatch: manifest det_dev.npz sha != frozen record; frozen model was not "
                         "trained on this dataset version.")
    ti_sha = _verify_sha(a.test_dir / "det_test_inputs.npz", msha.get("det_test_inputs.npz"), "det_test_inputs.npz")
    tg_sha = _verify_sha(a.test_dir / "det_test_gt.npz", msha.get("det_test_gt.npz"), "det_test_gt.npz")
    for v, want in fr.get("detector_sha256", {}).items():
        _verify_sha(a.frozen_dir / f"detector_{v}.pt", want, f"detector_{v}.pt")
    print(f"provenance verified: protocol digest {proto_sha[:12]}.. matches external anchor; peak-extraction, dataset "
          "lineage, test artifacts, and detector weights all match.", flush=True)

    # ---- load frozen nets + merge sealed inputs (images/geometry) with sealed GT (footprints) ----
    ep = fr["eval_params"]; base_ch, si_tol, boot, seed = ep["base_ch"], ep["si_tol"], ep["bootstrap"], ep["seed"]
    op_thr = fr["learning_procedure"]["operating_threshold_per_condition"]
    dev = T.device()
    tin = np.load(a.test_dir / "det_test_inputs.npz", allow_pickle=False)
    tgt = np.load(a.test_dir / "det_test_gt.npz", allow_pickle=False)
    if not np.array_equal(tin["case"], tgt["case"]):
        raise ValueError("Sealed test inputs/gt case order differs; cannot align footprints to images.")
    merged = {"ap": tin["ap"], "lat": tin["lat"], "ap_geo": tin["ap_geo"], "lat_geo": tin["lat_geo"], "case": tin["case"],
              "nfrac": tgt["nfrac"], "fp_case": tgt["fp_case"],
              "ap_fp_pts": tgt["ap_fp_pts"], "ap_fp_ptr": tgt["ap_fp_ptr"],
              "lat_fp_pts": tgt["lat_fp_pts"], "lat_fp_ptr": tgt["lat_fp_ptr"]}

    arch = T.arch_from_record(fr)   # rebuild the frozen architecture (from-scratch or pretrained-encoder U-Net)
    nets = {}
    for v in ("ap", "lat"):
        net = T.build_detector(arch, pretrained=False).to(dev)
        net.load_state_dict(torch.load(a.frozen_dir / f"detector_{v}.pt", map_location=dev)); net.eval()
        nets[v] = net
    ap_g, lat_g = T.group_instances(merged, "ap"), T.group_instances(merged, "lat")
    idx = np.arange(len(merged["case"]))
    cache = T.peak_cache(nets, merged, idx, dev)

    grids = fr["frozen_froc_grids"]   # reuse VERBATIM — the test does not choose its own thresholds
    lat_gate = float(fr["learning_procedure"].get("biplanar_fusion", {}).get("unmatched_lateral_score_gate", 0.0))
    if lat_gate > 0: print(f"fusion unmatched-lateral score gate (frozen) = {lat_gate}", flush=True)
    conds = ["ap", "lat", "fusion", "paired"]
    results = {c: T.eval_condition(c, cache, ap_g, lat_g, idx, si_tol, boot, seed,
                                   fixed_grid=grids[c], op_threshold=op_thr[c], lat_gate=lat_gate) for c in conds}

    name = {"ap": "AP-only", "lat": "lateral-only", "fusion": "biplanar-FUSION", "paired": "paired-confirmed"}
    print("\n====== SEALED-TEST FROC (confirmatory, scored ONCE) ======")
    print(f"{'condition':18}{'unit':16}" + "".join(f"{'sens@'+str(t)+'FP':>18}" for t in T.FP_TARGETS))
    for c in conds:
        r = results[c]; row = f"{name[c]:18}{T.UNIT[c]:16}"
        for t in T.FP_TARGETS:
            lo, hi = r["ci"][t]; row += f"{(format(r['sens'][t],'.3f')+' ['+format(lo,'.2f')+','+format(hi,'.2f')+']'):>18}"
        print(row)
    print(f"\n{'condition':18}{'op_thr':>8}{'case-recall':>13}{'count-MAE':>11}{'unmatched-FP':>14}{'AUPRC':>8}")
    for c in conds:
        r = results[c]
        print(f"{name[c]:18}{r['op_threshold']:>8.3f}{r['case_recall']:>13.3f}{r['count_mae']:>11.3f}"
              f"{r['unmatched_fp_at_op']:>14d}{('%.3f'%r['auprc'] if r['auprc'] is not None else '   n/a'):>8}")

    out = {"confirmatory": True, "scored_once": True, "protocol": T.PROTOCOL_VERSION,
           "evaluator_version": EVALUATOR_VERSION, "torch": torch.__version__, "device": str(dev),
           "frozen_dir": str(a.frozen_dir), "test_dir": str(a.test_dir),
           "protocol_sha256": proto_sha, "det_test_inputs_sha256": ti_sha, "det_test_gt_sha256": tg_sha,
           "test_manifest_sha256": T.sha256_file(a.test_dir / "det_manifest.json"),
           "dataset_version": man.get("dataset_version"), "det_dev_sha256": fr["det_dev_sha256"],
           "detector_sha256": fr["detector_sha256"], "n_test_cases": int(len(idx)),
           "results": {c: {"sens_at_targets": {str(k): v for k, v in results[c]["sens"].items()},
                           "ci": {str(k): v for k, v in results[c]["ci"].items()},
                           "froc_curve": results[c]["froc"], "unit": T.UNIT[c],
                           "op_threshold": results[c]["op_threshold"], "case_recall": results[c]["case_recall"],
                           "count_mae": results[c]["count_mae"], "unmatched_fp_at_op": results[c]["unmatched_fp_at_op"],
                           "auprc": results[c]["auprc"]} for c in conds},
           "note": "Single confirmatory pass: frozen learning procedure, frozen FROC grids, frozen operating "
                   "thresholds. Sealed cohort is all-positive, so no clean-patient specificity is computed here. "
                   "By-fracture-class sensitivity was deferred at freeze time and is intentionally not reported."}
    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "sealed_test_results.json").write_text(json.dumps(out, indent=2))
    work.rename(a.out)   # atomic: a crash mid-write never leaves an empty dir blocking the run
    print(f"\nwrote sealed_test_results.json to {a.out}/ . This is the confirmatory result — do not re-run to a new tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
