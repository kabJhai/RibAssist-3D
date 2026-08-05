"""Frozen paths and policies for the RibAssist 3D clinician demo."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_NAME = "RibAssist 3D"

# Primary sealed evaluation cohort (image-only keys used at inference).
DATA_NPZ = ROOT / "outputs" / "det_out_v2" / "det_test.npz"
SEALED_DATA_SHA256 = "75c62ab5286dedbfd6e6d994f8b1ede6e969a9e505aaded43379f672fdb0514d"

# Champion stack: detection + rib addressing (original findings).
CHAMPION_DETECTOR = ROOT / "outputs" / "detector_dev_scratch_c32_both_gated"
ADDRESS_MODEL = ROOT / "outputs" / "addressing_model_ap_nopos"

# L2 stack: biplanar correspondence + 3D triangulation.
L2_DETECTOR = ROOT / "outputs" / "detector_L2_lateral_hnm"
L2_POLICY = ROOT / "outputs" / "sealed" / "L2_policy.json"
L2_PAIRS_NPZ = ROOT / "outputs" / "sealed" / "L2_sealed_D0_pairs.npz"
L2_SEALED_D1 = ROOT / "outputs" / "sealed" / "L2_sealed_D1.json"

# Anatomy (patient-specific rib seg + centerlines).
IMAGE_DIRS = [ROOT / "data" / "ribfrac_train", ROOT / "data" / "ribfrac"]
SEG_DIR = ROOT / "data" / "ribseg" / "ribseg_v2" / "seg"
CL_DIR = ROOT / "data" / "ribseg" / "ribseg_v2" / "cl"

# L2 extraction policy (sealed).
AP_NMS, AP_FLOOR = 5, 0.05
LAT_NMS, LAT_FLOOR = 3, 0.10
AUDIT_GATE_MAX = 30.0

# Finding ↔ L2 AP linkage tolerance (pixels, frozen).
LINK_TOLERANCE_PX = 2.0

# Presentation catalog (also any case in det_test.npz).
FEATURED_CASES = ["RibFrac119", "RibFrac142", "RibFrac176", "RibFrac3"]

EXPECTED_MATCHED_AT10 = 15
