#!/usr/bin/env bash
# ============================================================================
# L2 FLOOR SWEEP — find the retrained lateral head's OWN best operating point.
#
# The L2 head's heatmap amplitude rose from ~0.091 to ~0.155, so the frozen
# head's standing floor (0.06) over-extracts it. This sweeps the L2 head's
# lateral floor through D0->D1 (nms3 fixed; AP nms5/0.05 fixed) to locate the
# floor that maximizes out-of-fold recall@10 at <=1 false-3D/case.
#
# Run from the RibAssist 3D ROOT:  bash scripts/run_L2_floor_sweep.sh
# Outputs -> outputs/L2_sweep/ (remove that dir to re-run cleanly).
# ============================================================================
set -euo pipefail

DET=outputs/detector_L2_lateral_hnm
DATA=outputs/det_out_v2/det_dev.npz
IMG=(data/ribfrac_train data/ribfrac)
SEG=data/ribseg/ribseg_v2/seg
CL=data/ribseg/ribseg_v2/cl
OUT=outputs/L2_sweep
mkdir -p "$OUT"

FLOORS=(0.06 0.08 0.10 0.12)

for F in "${FLOORS[@]}"; do
  TAG="L2_latN3_f${F/./}"
  echo ""
  echo "############################################################"
  echo "# L2 FLOOR $F  (lat nms3 / floor $F ; ap nms5/0.05)"
  echo "############################################################"
  echo "--- D0 ---"
  python scripts/eval_correspondence_D0_pairaudit.py \
      --detector-run "$DET" --data "$DATA" --image-dirs "${IMG[@]}" \
      --audit-gate-max 30 --ap-nms 5 --ap-floor 0.05 --lat-nms 3 --lat-floor "$F" \
      --out "$OUT/${TAG}_D0.json"
  echo "--- D1 ---"
  SIDE="$OUT/${TAG}_D0_ambiguous_pairs.json"
  python scripts/eval_correspondence_D1_assign.py \
      --pairs-npz "$OUT/${TAG}_D0_pairs.npz" --detector-run "$DET" --data "$DATA" \
      --image-dirs "${IMG[@]}" --seg-dir "$SEG" --cl-dir "$CL" \
      $( [ -f "$SIDE" ] && echo --ambiguous-sidecar "$SIDE" ) \
      --out "$OUT/${TAG}_D1.json"
done

echo ""
echo "############################################################"
echo "# L2 FLOOR SWEEP — collation"
echo "############################################################"
python3 - "$OUT" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
print(f"{'floor':>6} {'lat_pk':>7} {'lat_rec':>8} {'dual%':>6} {'broad':>7} {'comp/dv':>8} {'R@10|c1':>8} {'fp@10':>6} {'match':>6} {'R@10|none':>10}")
print("-"*82)
for D1 in sorted(glob.glob(os.path.join(d, "*_D1.json"))):
    D0 = D1.replace("_D1.json", "_D0.json")
    j1 = json.load(open(D1)); j0 = json.load(open(D0)) if os.path.exists(D0) else {}
    pol = j0.get("extraction_policy", {}); floor = pol.get("lat_floor", "?")
    pg = j0.get("pair_graph", {}); av = j0.get("correct_pair_availability_existential", {})
    latpk = pg.get("mean_lat_peaks_per_case", "-"); latrec = av.get("compatible_lat_recall", "-")
    dual = av.get("existential_dual_view_frac", "-"); broad = pg.get("pairs_in_broad_graph_npz", "-")
    comp = "-"
    for r in j0.get("gate_sensitivity_sweep", {}).get("rows", []):
        if abs(float(r.get("gate_vox", -1)) - float(j0.get("si_tol_voxels", 6.0))) < 1e-6:
            comp = r.get("mean_competitors_per_dualview_fracture", "-")
    h = j1.get("operational_headline_out_of_fold", {})
    r10 = h.get("recall10", "-"); fp = h.get("false_3d_per_case_at_10mm", "-"); m = h.get("n_matched_within10", "-")
    none = next((c for c in j1.get("operational_cap_sweep_out_of_fold", []) if c.get("false_3d_cap_per_case_at_10mm") is None), {})
    rnone = none.get("recall10", "-")
    print(f"{str(floor):>6} {str(latpk):>7} {str(latrec):>8} {str(dual):>6} {str(broad):>7} {str(comp):>8} {str(r10):>8} {str(fp):>6} {str(m):>6} {str(rnone):>10}")
print("\nR@10|c1 = out-of-fold operational recall@10 at <=1 false-3D/case (frozen-head baseline was 0.0%).")
PY
