"""Case overview stage."""
from __future__ import annotations

import streamlit as st

from demo_app.anatomy_3d import CAMERA_PRESETS
from demo_app.anatomy_scene import AnatomyBundle
from demo_app.case_map_3d import build_case_map_figure
from demo_app.clinical_summary import CaseSummary
from demo_app.finding_map import build_map_findings, build_rib_groups
from demo_app.pipeline import CaseInferenceResult
from demo_app.ui.components import (
    render_case_summary_card,
    render_findings_panel,
    render_map_with_clicks,
    render_rib_navigator,
    rib_ids_for_filter,
)
from demo_app.ui.session import STAGE_REVIEW, go_summary, open_finding_review


def _open_review(fid: int) -> None:
    open_finding_review(fid)
    st.rerun()


def render_overview(
    *,
    result: CaseInferenceResult,
    bundle: AnatomyBundle,
    summary: CaseSummary,
) -> None:
    review = st.session_state.review
    sel_id = st.session_state.selected_finding
    map_findings = build_map_findings(result, review, show_rejected=st.session_state.show_rejected)
    rib_groups = build_rib_groups(bundle, map_findings)

    map_col, side_col = st.columns([0.64, 0.36], gap="medium")

    with side_col:
        render_case_summary_card(summary, review)
        render_rib_navigator(rib_groups, bundle=bundle, map_findings=map_findings, result=result)

    sel_rib = st.session_state.selected_rib
    highlight_ids = rib_ids_for_filter(bundle, sel_rib)

    with map_col:
        cam_row = st.columns(6)
        for i, name in enumerate(CAMERA_PRESETS):
            if cam_row[i].button(name, key=f"ov_cam_{name}"):
                st.session_state.camera = name
                st.rerun()
        if bundle.warnings:
            for w in bundle.warnings[:1]:
                st.caption(w)
        fig = build_case_map_figure(
            bundle,
            findings=map_findings,
            selected_id=None,
            height=680,
            camera=CAMERA_PRESETS.get(st.session_state.camera, CAMERA_PRESETS["Posterior"]),
            mode="overview",
            highlight_rib_ids=highlight_ids,
        )
        clicked = render_map_with_clicks(fig, key="overview_map")
        if clicked:
            open_finding_review(clicked)
            st.rerun()

    st.divider()
    render_findings_panel(result, review, sel_rib, sel_id, _open_review, bundle=bundle)

    nav_cols = st.columns([0.25, 0.75])
    if nav_cols[0].button("Go to case summary", width="stretch"):
        go_summary()
        st.rerun()
    if sel_id is not None:
        nav_cols[1].caption(f"Selected finding #{sel_id}. Click a row above or a 3D marker to review.")
