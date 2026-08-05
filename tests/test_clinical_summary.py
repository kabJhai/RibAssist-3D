# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Tests for clinical summary generation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.clinical_summary import build_case_summary, build_decision_card, build_finding_interpretation
from demo_app.model_runtime import OriginalFinding
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding


def _mk(ef_list):
    from demo_app.data_loader import CaseImages
    import numpy as np

    case = CaseImages("T", 0, np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32), np.eye(4), np.eye(4))
    return CaseInferenceResult(case, ef_list, None, None, None, np.zeros((0, 3)), np.zeros((0, 3)))


def test_case_summary_counts():
    findings = [
        EnrichedFinding(OriginalFinding(1, 0.3, "addressed", "paired", "R", 5, 0.4, [1, 2]), localization_status="Localized"),
        EnrichedFinding(OriginalFinding(2, 0.2, "addressed", "paired", "R", 5, 0.2, [3, 4]), localization_status="Abstained"),
    ]
    s = build_case_summary(_mk(findings), {2: "needs_review"})
    assert s.detected == 2
    assert s.localized == 1
    assert s.abstained == 1
    assert "R5" in s.duplicate_rib_levels


def test_finding_interpretation_abstained():
    ef = EnrichedFinding(
        OriginalFinding(1, 0.2, "addressed", "paired", "L", 7, 0.3, [1, 2]),
        localization_status="Abstained",
    )
    interp = build_finding_interpretation(ef)
    assert "Candidate" in " ".join(interp["lines"])
    assert "did not emit" in " ".join(interp["lines"]).lower()


def test_decision_card_localized():
    ef = EnrichedFinding(
        OriginalFinding(1, 0.2, "addressed", "paired", "R", 4, 0.4, [1, 2]),
        localization_status="Localized",
    )
    card = build_decision_card(ef)
    assert "Localized on R4" in card["headline"]
    assert "Suggested review" in card["action"]
