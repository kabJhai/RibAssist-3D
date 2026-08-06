# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Load projection NPZ cases and patient rib anatomy."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CL_DIR, DATA_NPZ, IMAGE_DIRS, SEG_DIR, SEALED_DATA_SHA256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CaseImages:
    case_id: str
    case_index: int
    ap: np.ndarray
    lat: np.ndarray
    ap_geo: np.ndarray
    lat_geo: np.ndarray


class ProjectionStore:
    def __init__(self, npz_path: Path = DATA_NPZ):
        self.path = Path(npz_path)
        self.sha256 = sha256_file(self.path)
        if self.sha256 != SEALED_DATA_SHA256:
            raise ValueError(
                f"det_test.npz sha256 mismatch: got {self.sha256[:12]}.. expected {SEALED_DATA_SHA256[:12]}.."
            )
        self._data = np.load(self.path, allow_pickle=False)
        self.case_ids = [str(c) for c in self._data["case"]]
        self._index = {c: i for i, c in enumerate(self.case_ids)}

    def get(self, case_id: str) -> CaseImages:
        if case_id not in self._index:
            raise KeyError(f"{case_id} not in {self.path}")
        i = self._index[case_id]
        return CaseImages(
            case_id=case_id,
            case_index=i,
            ap=self._data["ap"][i].astype(np.float32),
            lat=self._data["lat"][i].astype(np.float32),
            ap_geo=self._data["ap_geo"][i].astype(np.float64),
            lat_geo=self._data["lat_geo"][i].astype(np.float64),
        )

    def gt_footprints(self, case_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """AP and lateral GT footprint point clouds for evaluation overlay."""
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from eval_address_e2e import build_instance_records

        i = self._index[case_id]
        recs = build_instance_records(self._data)
        ap_fps, lat_fps = [], []
        for rec in recs.get(i, []):
            if rec["ap_foot"].size:
                ap_fps.append(rec["ap_foot"])
            if rec["lat_foot"].size:
                lat_fps.append(rec["lat_foot"])
        return ap_fps, lat_fps

    def gt_footprint_for_iid(self, case_id: str, iid: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from eval_address_e2e import build_instance_records

        i = self._index[case_id]
        for rec in build_instance_records(self._data).get(i, []):
            if int(rec["iid"]) == int(iid):
                ap = rec["ap_foot"] if rec["ap_foot"].size else None
                lat = rec["lat_foot"] if rec["lat_foot"].size else None
                return ap, lat
        return None, None


def _find_ribfrac_nifti(case_id: str) -> Path | None:
    num = case_id.replace("RibFrac", "")
    for base in IMAGE_DIRS:
        for sub in ("ribfrac-val-labels", "ribfrac-train-labels", "labels"):
            p = base / sub / f"RibFrac{num}-label.nii.gz"
            if p.exists():
                return p
        p = base / f"RibFrac{num}-label.nii.gz"
        if p.exists():
            return p
    return None


def load_case_anatomy(case_id: str):
    """Seg array, affine, centerlines, and rib info, or None if missing."""
    try:
        import nibabel as nib
        from eval_biplanar_geometry import case_gt
    except ImportError:
        return None
    return case_gt(case_id, IMAGE_DIRS, SEG_DIR, CL_DIR)
