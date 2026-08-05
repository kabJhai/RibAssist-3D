#!/usr/bin/env bash
# ============================================================================
# SEALED CONFIRMATION — STAGE 1 (freeze + assemble; NO sealed metrics produced)
#
# Does everything that must happen BEFORE the test cohort is scored:
#   1. assemble the sealed cohort into the det_dev schema (det_test.npz), with
#      provenance verified against det_manifest.json, and print its sha256;
#   2. freeze the deployment D1 config for BOTH detectors by all-dev selection
#      (no folds) at the ≤1-false target, on the EXISTING dev pair graphs;
#   3. print the frozen configs + the per-fold dev configs for cross-check.
#
# This produces NO sealed reconstruction numbers. Review the frozen policies and
# the sha256, THEN run run_sealed_stage2_apply.sh.
#
# Run from RibAssist 3D ROOT:  bash scripts/run_sealed_stage1_freeze.sh
# ============================================================================
set -euo pipefail

FROZEN=outputs/detector_dev_scratch_c32_both_gated
L2=outputs/detector_L2_lateral_hnm
DATA=outputs/det_out_v2/det_dev.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/sealed
mkdir -p "$OUT"

FROZEN_DEVGRAPH=outputs/L1_sweep/latN3_f060_D0_pairs.npz     # frozen head, nms3/floor0.06 (dev)
L2_DEVGRAPH=outputs/L2_sweep/L2_latN3_f010_D0_pairs.npz       # L2 head,    nms3/floor0.10 (dev)

echo "############################################################"
echo "# STAGE 1a — assemble sealed det_test.npz (provenance-verified)"
echo "############################################################"
python scripts/build_sealed_det_npz.py \
    --inputs outputs/det_out_v2/det_test_inputs.npz \
    --gt     outputs/det_out_v2/det_test_gt.npz \
    --manifest outputs/det_out_v2/det_manifest.json \
    --out    outputs/det_out_v2/det_test.npz

echo ""
echo "############################################################"
echo "# STAGE 1b — freeze the D1 deployment config on DEV (both detectors)"
echo "############################################################"
echo "--- frozen detector (nms3/floor0.06 dev graph) ---"
python scripts/eval_correspondence_D1_assign.py \
    --pairs-npz "$FROZEN_DEVGRAPH" --detector-run "$FROZEN" --data "$DATA" \
    --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
    --false-3d-cap-at10 1.0 \
    --freeze-policy-out "$OUT/frozen_policy.json" --out "$OUT/_unused_frozen.json"

echo "--- L2 detector (nms3/floor0.10 dev graph) ---"
python scripts/eval_correspondence_D1_assign.py \
    --pairs-npz "$L2_DEVGRAPH" --detector-run "$L2" --data "$DATA" \
    --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
    --false-3d-cap-at10 1.0 \
    --freeze-policy-out "$OUT/L2_policy.json" --out "$OUT/_unused_L2.json"

echo ""
echo "############################################################"
echo "# STAGE 1 SUMMARY — review before Stage 2"
echo "############################################################"
python3 - "$OUT" <<'PY'
import json, sys
d = sys.argv[1]
for tag, pf, devjson in [("frozen", f"{d}/frozen_policy.json", "outputs/L1_sweep/latN3_f060_D1.json"),
                         ("L2",     f"{d}/L2_policy.json",     "outputs/L2_sweep/L2_latN3_f010_D1.json")]:
    p = json.load(open(pf))
    print(f"[{tag}] FROZEN policy (all-dev): gate {int(p['gate'])} {p['cost']}{'+mb' if p['mutual_best'] else ''} "
          f"u={p['u']:.6f} | feasible {p['feasible_within_cap']} | dev recall10 {p['dev_recall10']} @ {p['dev_false_3d_per_case_at_10mm']}/case")
    try:
        fc = json.load(open(devjson)).get("out_of_fold_fold_configs", [])
        uniq = sorted({(int(x['gate']), x['cost'], bool(x['mutual_best']), round(x['u'],4)) for x in fc})
        print(f"        per-fold dev configs (cross-check): {uniq}")
    except Exception as e:
        print(f"        (per-fold cross-check unavailable: {e})")
prov = json.load(open("outputs/det_out_v2/det_test.provenance.json"))
print(f"\nsealed det_test.npz sha256 = {prov['det_test_npz_sha256']}")
print(f"  ({prov['n_cases']} cases, {prov['n_fractures']} GT fractures)")
print("\n>>> Confirm the frozen policies + sha, then run: bash scripts/run_sealed_stage2_apply.sh")
PY

rm -f "$OUT/_unused_frozen.json" "$OUT/_unused_L2.json" 2>/dev/null || true
