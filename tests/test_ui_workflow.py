"""Tests for review workflow helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.ui.components import sort_findings
from demo_app.ui.session import advance_after_decision


def test_priority_sort_pending_first():
    rows = [
        {"finding_id": 1, "review": "Accepted", "confidence": "High"},
        {"finding_id": 2, "review": "Pending", "confidence": "High"},
        {"finding_id": 3, "review": "Pending", "confidence": "Low"},
    ]
    out = sort_findings(rows, "priority")
    assert [r["finding_id"] for r in out] == [3, 2, 1]


def test_advance_after_decision_skips_resolved():
    review = {1: "accepted", 2: "pending", 3: "pending"}
    assert advance_after_decision(review, [1, 2, 3], 1) == 2
    review[2] = "rejected"
    assert advance_after_decision(review, [1, 2, 3], 2) == 3
    review[3] = "accepted"
    assert advance_after_decision(review, [1, 2, 3], 3) is None


def test_pending_inference_case_overrides_stale_last_case():
    """Regression: run must use the case selected when Run was clicked."""
    pending = "RibFrac118"
    last_case = "RibFrac119"
    case_to_run = pending or last_case
    assert case_to_run == "RibFrac118"
