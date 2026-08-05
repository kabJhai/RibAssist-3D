#!/usr/bin/env bash
# ============================================================================
# SEALED CONFIRMATION — STAGE 2 (the ONE confirmatory pass on untouched data)
#
# Applies the FROZEN extraction policy + FROZEN D1 config (from Stage 1) to the
# sealed cohort, for BOTH detectors, with NO reselection or calibration on test.
#   frozen detector: lateral nms3/floor0.06  +  frozen_policy.json
#   L2 detector:     lateral nms3/floor0.10  +  L2_policy.json
#   AP: nms5/floor0.05 (both)
# Then produces the paired frozen-vs-L2 comparison.
#
# Provenance is bound to the assembled det_test.npz sha256 (external anchor from
# Stage 1). Scripts refuse to overwrite, so this pass cannot be silently re-rolled.
#
# Run from RibAssist 3D ROOT (AFTER Stage 1 + confirming the frozen policies):
#   bash scripts/run_sealed_stage2_apply.sh
# ============================================================================
set -euo pipefail

FROZEN=outputs/detector_dev_scratch_c32_both_gated
L2=outputs/detector_L2_lateral_hnm
TEST=outputs/det_out_v2/det_test.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/sealed

for f in "$TEST" "$OUT/frozen_policy.json" "$OUT/L2_policy.json" outputs/det_out_v2/det_test.provenance.json; do
  [ -f "$f" ] || { echo "missing $f — run run_sealed_stage1_freeze.sh first"; exit 1; }
done
SHA=$(python3 -c "import json;print(json.load(open('outputs/det_out_v2/det_test.provenance.json'))['det_test_npz_sha256'])")
echo "sealed det_test.npz sha256 = $SHA"

run_one () {  # name detector lat_floor policy_json
  local NAME="$1" DET="$2" LATF="$3" POL="$4"
  echo ""
  echo "############################################################"
  echo "# SEALED $NAME  (lat nms3/floor$LATF ; ap nms5/0.05)  policy=$POL"
  echo "############################################################"
  echo "--- sealed D0 ---"
  python scripts/eval_correspondence_D0_pairaudit.py \
      --detector-run "$DET" --data "$TEST" --image-dirs "${IMG[@]}" \
      --audit-gate-max 30 --ap-nms 5 --ap-floor 0.05 --lat-nms 3 --lat-floor "$LATF" \
      --expected-data-sha256 "$SHA" \
      --out "$OUT/${NAME}_sealed_D0.json"
  echo "--- sealed D1 (apply frozen policy) ---"
  local SIDE="$OUT/${NAME}_sealed_D0_ambiguous_pairs.json"
  python scripts/eval_correspondence_D1_assign.py \
      --pairs-npz "$OUT/${NAME}_sealed_D0_pairs.npz" --detector-run "$DET" --data "$TEST" \
      --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
      --expected-data-sha256 "$SHA" --apply-policy "$POL" \
      $( [ -f "$SIDE" ] && echo --ambiguous-sidecar "$SIDE" ) \
      --out "$OUT/${NAME}_sealed_D1.json"
}

run_one frozen "$FROZEN" 0.06 "$OUT/frozen_policy.json"
run_one L2     "$L2"     0.10 "$OUT/L2_policy.json"

echo ""
echo "############################################################"
echo "# SEALED CONFIRMATION — paired comparison"
echo "############################################################"
python scripts/collate_sealed.py --sealed-dir "$OUT"
