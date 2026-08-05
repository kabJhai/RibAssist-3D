# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Tests for 3D finding map helpers."""
from __future__ import annotations

import numpy as np

from demo_app.case_map_3d import build_case_map_figure
from demo_app.finding_map import MapFinding, build_map_findings, classify_display_kind, display_status_label
from demo_app.model_runtime import OriginalFinding
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding


def _result(findings):
    from demo_app.data_loader import CaseImages

    case = CaseImages("T", 0, np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32), np.eye(4), np.eye(4))
    return CaseInferenceResult(case, findings, None, None, None, np.zeros((0, 3)), np.zeros((0, 3)))


def test_classify_kinds():
    loc = EnrichedFinding(OriginalFinding(1, 0.3, "addressed", "paired", "L", 6, 0.4, [1, 2]), localization_status="Localized")
    cand = EnrichedFinding(OriginalFinding(2, 0.2, "addressed", "paired", "L", 7, 0.3, [3, 4]), localization_status="Abstained")
    assert classify_display_kind(loc) == "localized"
    assert classify_display_kind(cand) == "candidate"
    assert display_status_label("candidate") == "Candidate"


def test_map_figure_multiple_findings():
    cl = np.column_stack([np.arange(0, 80, 2.0), np.zeros(40), np.zeros(40)])
    anatomy = {
        "aff": np.eye(4),
        "rs": np.zeros((4, 4, 4), dtype=np.int32),
        "info": {1: {"side": "L", "num": 6}},
        "cl_world": [cl],
        "fl_groups": {},
    }
    from demo_app.anatomy_scene import AnatomyBounds, AnatomyBundle

    bounds = AnatomyBounds.from_points(cl)
    bundle = AnatomyBundle(
        case_id="T",
        anatomy=anatomy,
        meshes=None,
        bounds=bounds,
        centerline_world=[cl],
        mesh_labels=set(),
        centerline_fallback_labels={1},
        seg_path="",
        seg_sha256="test",
        affine=np.eye(4),
    )
    findings = [
        MapFinding(1, "localized", "L6", 1, point_world=np.array([40.0, 0.0, 0.0])),
        MapFinding(2, "candidate", "L7", 1, candidate_world=np.array([50.0, 1.0, 0.0])),
    ]
    fig = build_case_map_figure(bundle, findings=findings, selected_id=1, height=400)
    names = [t.name for t in fig.data]
    assert any("Confirmed" in (n or "") for n in names)
    assert any("Candidate" in (n or "") for n in names)
    assert any(n and n.startswith("CL ") for n in names)


def test_build_map_findings_skips_rejected():
    ef = EnrichedFinding(
        OriginalFinding(1, 0.2, "addressed", "paired", "R", 3, 0.3, [1, 2]),
        localization_status="Localized",
        point_world=np.array([1.0, 2.0, 3.0]),
    )
    r = _result([ef])
    m = build_map_findings(r, {1: "rejected"}, show_rejected=False)
    assert len(m) == 0
    m2 = build_map_findings(r, {1: "rejected"}, show_rejected=True)
    assert len(m2) == 1
