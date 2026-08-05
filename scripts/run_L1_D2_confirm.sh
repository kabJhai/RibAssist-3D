#!/usr/bin/env bash
# ============================================================================
# L1 D2 CONFIRMATION — learned appearance frontier on the winning L1 policy.
#
# The D0/D1 sweep showed the deterministic frontier stays at 0 recall@10 at the
# operational <=1 false-3D/case budget across all extraction policies, even as
# the candidate flood shrank ~4x. This runs the LEARNED two-tower scorer (D2a
# crops -> D2b) on the winning policy to confirm appearance also cannot commit
# real-vs-real at the operational budget on the cleaner field — completing the
# reviewer's stated gate before routing to L2 (lateral retraining) / global
# anatomical correspondence.
#
# Run from the RibAssist 3D ROOT:  bash scripts/run_L1_D2_confirm.sh [POLICY_TAG]
# Default policy tag: latN3_f060 (best pairability ceiling + availability).
# ============================================================================
set -euo pipefail

TAG="${1:-latN3_f060}"
DET=outputs/detector_dev_scratch_c32_both_gated
DATA=outputs/det_out_v2/det_dev.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/L1_sweep
NPZ="$OUT/${TAG}_D0_pairs.npz"

if [ ! -f "$NPZ" ]; then echo "missing $NPZ — run run_L1_correspondence_sweep.sh first"; exit 1; fi

echo "############################################################"
echo "# D2 CONFIRMATION on policy $TAG"
echo "############################################################"

echo "--- D2a (appearance crops) ---"
python scripts/eval_correspondence_D2a_crops.py \
    --pairs-npz "$NPZ" --detector-run "$DET" --data "$DATA" \
    --half 20 \
    --out "$OUT/${TAG}_D2a_crops.npz"

echo "--- D2b (learned two-tower frontier) ---"
python scripts/eval_correspondence_D2b_learned.py \
    --crops-npz "$OUT/${TAG}_D2a_crops.npz" --pairs-npz "$NPZ" \
    --detector-run "$DET" --data "$DATA" \
    --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
    --out "$OUT/${TAG}_D2b_learned.json"

echo ""
echo "D2 confirmation complete for $TAG:"
echo "  crops   -> $OUT/${TAG}_D2a_crops.npz"
echo "  learned -> $OUT/${TAG}_D2b_learned.json"
