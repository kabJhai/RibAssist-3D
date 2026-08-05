"""Case catalog for the clinician demo."""
from __future__ import annotations

from .config import FEATURED_CASES


def list_cases(all_case_ids: list[str]) -> list[str]:
    """Featured cases first (if present), then remaining sealed cases sorted."""
    featured = [c for c in FEATURED_CASES if c in all_case_ids]
    rest = sorted(c for c in all_case_ids if c not in featured)
    return featured + rest


def case_blurb(case_id: str) -> str:
    blurbs = {
        "RibFrac119": "Best successful 3D reconstruction",
        "RibFrac142": "Typical multi-finding case (select findings individually)",
        "RibFrac176": "Successful multi-finding case",
        "RibFrac3": "Pure abstention: detections kept, no 3D point emitted",
    }
    return blurbs.get(case_id, "")
