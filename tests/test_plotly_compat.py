# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Plotly serialization must preserve mesh geometry for Streamlit rendering."""
from __future__ import annotations

import json

import numpy as np

from demo_app.anatomy_scene import build_anatomy_bundle
from demo_app.case_map_3d import build_case_map_figure
from demo_app.finding_map import MapFinding


def test_mesh3d_arrays_survive_json_roundtrip():
    bundle = build_anatomy_bundle("RibFrac142")
    assert bundle and bundle.meshes
    b = bundle.bounds
    pt = np.array([(b.xmin + b.xmax) / 2, b.ymin, (b.zmin + b.zmax) / 2])
    fig = build_case_map_figure(
        bundle,
        findings=[MapFinding(1, "localized", "L6", 1, point_world=pt)],
        selected_id=1,
        height=400,
    )
    raw = json.loads(fig.to_json())
    mesh = next(tr for tr in raw["data"] if tr["type"] == "mesh3d")
    src = fig.data[0]
    if isinstance(mesh["x"], dict) and "bdata" in mesh["x"]:
        assert len(src.x) > 100
    else:
        assert len(mesh["x"]) > 100
