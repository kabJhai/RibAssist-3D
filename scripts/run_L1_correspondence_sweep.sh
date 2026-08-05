#!/usr/bin/env bash
# ============================================================================
# L1 EXTRACTION-POLICY CORRESPONDENCE SWEEP
#
# Tests whether recalibrating the LATERAL peak-extraction policy (the D0/D1/D2
# candidate flood was measured at the miscalibrated floor 0.05 / NMS 5) makes
# cross-view correspondence tractable, BEFORE any lateral retraining (L2).
#
# For each policy it regenerates the D0 broad pair graph under that policy and
# reruns the D1 deterministic assignment-with-abstention frontier. AP is held
# at the deployed extraction (0.05 / NMS 5) so the lateral change is isolated;
# layer AP recalibration afterward on the winning lateral policy.
#
# D2 (learned frontier) is intentionally NOT in this pass — run D2a+D2b only on
# the winning policy once the D0/D1 pass shows which policies are worth it.
#
# Run from the RibAssist 3D ROOT:  bash scripts/run_L1_correspondence_sweep.sh
# Outputs land in outputs/L1_sweep/ (scripts refuse to overwrite; remove that
# directory to re-run cleanly).
# ============================================================================
set -euo pipefail

DET=outputs/detector_dev_scratch_c32_both_gated
DATA=outputs/det_out_v2/det_dev.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/L1_sweep
mkdir -p "$OUT"

# policy tag | ap_nms ap_floor lat_nms lat_floor
POLICIES=(
  "deployed   5 0.05 5 0.05"
  "latN5_f055 5 0.05 5 0.055"
  "latN5_f060 5 0.05 5 0.060"
  "latN5_f065 5 0.05 5 0.065"
  "latN3_f060 5 0.05 3 0.060"
)

for P in "${POLICIES[@]}"; do
  read -r TAG APN APF LATN LATF <<< "$P"
  echo ""
  echo "############################################################"
  echo "# POLICY $TAG  |  ap nms$APN/floor$APF  lat nms$LATN/floor$LATF"
  echo "############################################################"
  D0J="$OUT/${TAG}_D0.json"
  NPZ="$OUT/${TAG}_D0_pairs.npz"
  SIDE="$OUT/${TAG}_D0_ambiguous_pairs.json"
  D1J="$OUT/${TAG}_D1.json"

  echo "--- D0 (pair graph) ---"
  python scripts/eval_correspondence_D0_pairaudit.py \
      --detector-run "$DET" --data "$DATA" \
      --image-dirs "${IMG[@]}" \
      --audit-gate-max 30 \
      --ap-nms "$APN" --ap-floor "$APF" --lat-nms "$LATN" --lat-floor "$LATF" \
      --out "$D0J"

  echo "--- D1 (deterministic frontier) ---"
  python scripts/eval_correspondence_D1_assign.py \
      --pairs-npz "$NPZ" --detector-run "$DET" --data "$DATA" \
      --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
      $( [ -f "$SIDE" ] && echo --ambiguous-sidecar "$SIDE" ) \
      --out "$D1J"
done

echo ""
echo "############################################################"
echo "# SWEEP COMPLETE — collating headline frontier per policy"
echo "############################################################"
python scripts/collate_L1_sweep.py --sweep-dir "$OUT"
