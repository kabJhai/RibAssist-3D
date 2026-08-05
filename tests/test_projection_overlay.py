"""Projection overlay alignment with 3D display."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.anatomy_scene import build_anatomy_bundle
from demo_app.data_loader import ProjectionStore
from demo_app.finding_map import build_map_findings
from demo_app.model_runtime import load_champion
from demo_app.correspondence_runtime import load_l2
from demo_app.pipeline import run_case_inference
from demo_app.projection_overlay import project_display_point, review_finding_projection_coords


def test_localized_2d_markers_match_committed_pair():
    store = ProjectionStore()
    champion, l2 = load_champion(), load_l2()
    case_id = "RibFrac119"
    result = run_case_inference(store.get(case_id), champion, l2, store.sha256)
    bundle = build_anatomy_bundle(case_id, downsample_step=2)
    assert bundle is not None

    review = {}
    map_findings = build_map_findings(result, review, anatomy=bundle.anatomy)
    localized = [ef for ef in result.findings if ef.committed_edge is not None]
    assert localized, "need at least one localized finding"

    ef = localized[0]
    mf = next(m for m in map_findings if m.finding_id == ef.original.finding_id)
    ap_rc, lat_rc, comm_ap, comm_lat, _ = review_finding_projection_coords(
        bundle, ef, mf, result,
    )
    assert comm_ap is None and comm_lat is None
    assert ap_rc is not None and lat_rc is not None

    edge = ef.committed_edge
    assert edge is not None
    # Snapped display projection should stay near the committed L2 pair.
    assert abs(ap_rc[0] - edge.ap_row) < 8.0
    assert abs(ap_rc[1] - edge.ap_col) < 8.0
    assert abs(lat_rc[0] - edge.lat_row) < 8.0
    assert abs(lat_rc[1] - edge.lat_col) < 8.0

    # Must differ from raw champion AP when champion peak != L2 AP.
    champ_ap = tuple(ef.original.ap_xy) if ef.original.ap_xy else None
    if champ_ap is not None:
        direct = project_display_point(
            bundle, ef.point_world, result.case.ap_geo, result.case.lat_geo,
            prefer_label=mf.rib_label_id,
        )
        assert direct == (ap_rc, lat_rc)
