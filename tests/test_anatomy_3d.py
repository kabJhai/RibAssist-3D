"""Tests for focal 3D fracture localization helpers."""
from __future__ import annotations

import numpy as np

from demo_app.anatomy_3d import FocalLocalization, build_anatomy_figure, local_centerline_segment


def test_local_centerline_segment_selects_neighborhood():
    # Straight line along x, 1 mm spacing
    cl = np.column_stack([np.arange(0, 200, 1.0), np.zeros(200), np.zeros(200)])
    point = np.array([100.0, 0.0, 0.0])
    seg = local_centerline_segment(cl, point, half_mm=20.0)
    assert seg is not None
    assert len(seg) >= 2
    span = seg[-1, 0] - seg[0, 0]
    assert 35 <= span <= 45  # ~40 mm total


def test_localized_figure_includes_segment_and_point():
    cl = np.column_stack([np.arange(0, 120, 2.0), np.zeros(60), np.zeros(60)])
    anatomy = {
        "aff": np.eye(4),
        "rs": np.zeros((4, 4, 4), dtype=np.int32),
        "info": {1: {"side": "R", "num": 7}},
        "cl_world": [cl],
        "fl_groups": {},
    }
    point = np.array([60.0, 0.0, 0.0])
    focal = FocalLocalization(
        status="localized",
        finding_id=1,
        predicted_rib="R7",
        highlight_rib_label=1,
        point_world=point,
        ap_score=0.8,
        lat_score=0.7,
        si_diff_vox=5.0,
    )
    fig = build_anatomy_figure(None, anatomy_fallback=anatomy, focal=focal, height=400)
    names = [t.name for t in fig.data]
    assert "Fracture segment" in names
    assert "Predicted fracture" in names


def test_abstained_figure_has_no_operational_point():
    anatomy = {
        "aff": np.eye(4),
        "rs": np.zeros((4, 4, 4), dtype=np.int32),
        "info": {1: {"side": "R", "num": 2}},
        "cl_world": [np.zeros((10, 3))],
        "fl_groups": {},
    }
    focal = FocalLocalization(
        status="abstained",
        predicted_rib="R2",
        highlight_rib_label=1,
    )
    fig = build_anatomy_figure(None, anatomy_fallback=anatomy, focal=focal, height=400)
    assert not any(t.name == "Predicted fracture" for t in fig.data)
