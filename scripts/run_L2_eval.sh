#!/usr/bin/env bash
# ============================================================================
# L2 OPERATIONAL EVALUATION — the retrained lateral head through the unchanged
# extraction-policy correspondence scoreboard, under the STANDING policy
# (AP nms5/floor0.05, LAT nms3/floor0.06). Produces the D0 graph + D1
# deterministic frontier for the L2 detector so its out-of-fold recall@10 at
# <=1 false-3D/case is directly comparable to the frozen-detector latN3_f060
# baseline from run_L1_correspondence_sweep.sh.
#
# Success is decided HERE, not by lateral FROC/AUROC: a material rise in
# D1 recall@10 at <=1 false-3D/case over the frozen baseline (0.0%).
#
# Run from the RibAssist 3D ROOT:  bash scripts/run_L2_eval.sh [DET_RUN_DIR]
# Default DET: outputs/detector_L2_lateral_hnm
# ============================================================================
set -euo pipefail

DET="${1:-outputs/detector_L2_lateral_hnm}"
DATA=outputs/det_out_v2/det_dev.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/L2_eval
TAG=L2_latN3_f060
mkdir -p "$OUT"

if [ ! -f "$DET/detector_dev_run.json" ]; then echo "missing $DET — train first with train_lateral_L2.py"; exit 1; fi

echo "############################################################"
echo "# L2 EVAL  DET=$DET  policy ap nms5/0.05  lat nms3/0.06"
echo "############################################################"

echo "--- D0 (pair graph, L2 detector) ---"
python scripts/eval_correspondence_D0_pairaudit.py \
    --detector-run "$DET" --data "$DATA" --image-dirs "${IMG[@]}" \
    --audit-gate-max 30 --ap-nms 5 --ap-floor 0.05 --lat-nms 3 --lat-floor 0.06 \
    --out "$OUT/${TAG}_D0.json"

echo "--- D1 (deterministic frontier, L2 detector) ---"
SIDE="$OUT/${TAG}_D0_ambiguous_pairs.json"
python scripts/eval_correspondence_D1_assign.py \
    --pairs-npz "$OUT/${TAG}_D0_pairs.npz" --detector-run "$DET" --data "$DATA" \
    --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
    $( [ -f "$SIDE" ] && echo --ambiguous-sidecar "$SIDE" ) \
    --out "$OUT/${TAG}_D1.json"

echo ""
echo "L2 eval complete. Compare D1 recall@10 @<=1 false-3D/case vs the frozen-detector baseline:"
echo "  frozen latN3_f060 (L1 sweep): 0.0% @<=1 false/case | 12.2% uncapped"
echo "  L2 result:                    see $OUT/${TAG}_D1.json (operational_headline_out_of_fold + cap sweep)"
echo "To also run the learned frontier on the L2 detector:"
echo "  bash scripts/run_L1_D2_confirm.sh  # after copying the L2 NPZ tag, or adapt --pairs-npz to $OUT/${TAG}_D0_pairs.npz"
