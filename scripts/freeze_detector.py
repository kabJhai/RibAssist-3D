#!/usr/bin/env python3
"""Freeze the detector by PROMOTING the exact selected development artifacts — no retraining.

An MPS rerun would not reproduce bit-identical weights, so freezing copies and VERIFIES the
selected dev-run artifacts (weights + their recorded hashes + the full development record: frozen
FROC grids, operating thresholds, split ids, selected epochs, eval params, peak-extraction params)
into a new immutable protocol record. It also records the PRE-DECLARED primary condition and the
overlay-QA audit trail. The sealed evaluator then consumes exactly this record.

Guards: the dev run must be non-smoke, views=both, selection informative for every view; overlay QA
must be asserted with audited case ids; the output dir must not exist; and a best-epoch that lands
on the final epoch (possible non-convergence) requires --allow-boundary-epoch.

Usage:
  python freeze_detector.py --dev-run outputs/detector_dev_e80_cos --primary fusion \
      --overlay-qa-passed --overlay-qa-cases "RibFrac19,RibFrac81,RibFrac257,RibFrac266" \
      --out outputs/detector_frozen_v1
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path

PROTOCOL_VERSION = "detector-protocol-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-run", type=Path, required=True, help="dir with detector_dev_run.json + detector_*.pt")
    ap.add_argument("--primary", choices=["ap", "lat", "fusion", "paired"], required=True,
                    help="the PRE-DECLARED primary deployed condition (per the written selection rule)")
    ap.add_argument("--out", type=Path, required=True, help="new immutable frozen dir (must NOT exist)")
    ap.add_argument("--overlay-qa-passed", action="store_true", required=False)
    ap.add_argument("--overlay-qa-cases", default="")
    ap.add_argument("--allow-boundary-epoch", action="store_true",
                    help="permit a best-epoch == final epoch (otherwise blocked as possible non-convergence)")
    a = ap.parse_args()

    rec_path = a.dev_run / "detector_dev_run.json"
    if not rec_path.exists(): raise FileNotFoundError(f"No detector_dev_run.json in {a.dev_run}.")
    rec = json.loads(rec_path.read_text())
    qa_cases = [c.strip() for c in a.overlay_qa_cases.split(",") if c.strip()]

    # ---- guards ----
    if rec.get("smoke"): raise ValueError("Refusing to freeze a --smoke dev run.")
    lp = rec["learning_procedure"]
    if lp.get("views") != "both": raise ValueError("Dev run must be --views both.")
    if not all(rec["learning_procedure"]["selection_informative_per_view"].values()):
        raise ValueError("Refusing to freeze: checkpoint selection never became informative for some view.")
    if not a.overlay_qa_passed: raise ValueError("--overlay-qa-passed is required to freeze (record overlay audit).")
    if not qa_cases: raise ValueError("--overlay-qa-cases is required (audit trail of visually checked case ids).")
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; frozen outputs are immutable — use a new dir.")
    bd = {v: e for v, e in lp["best_epoch_per_view"].items() if e == lp["epochs"]}
    if bd and not a.allow_boundary_epoch:
        raise ValueError(f"best_epoch == final epoch for {bd} (possible non-convergence). Extend/anneal, or pass "
                         "--allow-boundary-epoch if convergence is otherwise established (e.g. cosine tail flat).")
    if a.primary not in rec["dev_internal_froc"]:
        raise ValueError(f"--primary {a.primary} not among evaluated conditions {list(rec['dev_internal_froc'])}.")

    # ---- verify the selected weights match the dev-run hashes (bit-identical promotion) ----
    for v, want in rec["detector_sha256"].items():
        got = sha256_file(a.dev_run / f"detector_{v}.pt")
        if got != want: raise ValueError(f"detector_{v}.pt hash {got[:12]}.. != dev-run record {want[:12]}..")

    # ---- promote: copy weights + write the frozen record (dev record + freeze fields) ----
    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    for v in rec["detector_sha256"]:
        shutil.copy2(a.dev_run / f"detector_{v}.pt", work / f"detector_{v}.pt")
        if sha256_file(work / f"detector_{v}.pt") != rec["detector_sha256"][v]:
            raise ValueError(f"copy of detector_{v}.pt did not preserve hash.")
    frozen = dict(rec)
    frozen["frozen"] = True
    frozen["primary_condition"] = a.primary
    frozen["primary_condition_note"] = ("chosen by the PRE-DECLARED rule in PROJECT_PLAN (not by whichever scalar is "
                                        "largest); FP units differ by condition (AP: /image, fusion: /image-pair).")
    frozen["overlay_qa"] = {"passed": True, "audited_case_ids": qa_cases}
    frozen["promoted_from_dev_run"] = str(a.dev_run)
    (work / "detector_protocol_frozen.json").write_text(json.dumps(frozen, indent=2))
    work.rename(a.out)

    proto_sha = sha256_file(a.out / "detector_protocol_frozen.json")
    print(f"promoted selected artifacts to {a.out}/ (no retraining). primary_condition = {a.primary}.")
    print(f"best_epoch_per_view = {lp['best_epoch_per_view']} | lr_schedule = {lp.get('lr_schedule')}")
    print(f"\n>>> PROTOCOL LOCK: sha256(detector_protocol_frozen.json) = {proto_sha}")
    print("    Commit this digest (git or a protocol-lock record) BEFORE opening the sealed test, then run:")
    print(f"    python eval_sealed_test.py --frozen-dir {a.out} --test-dir <det_out_v2> "
          f"--out <sealed dir> --expected-protocol-sha256 {proto_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
