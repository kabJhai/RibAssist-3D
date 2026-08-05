#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Streamlit entry point for the RibAssist 3D review workflow."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from demo_app.config import APP_NAME  # noqa: E402
from demo_app.anatomy_scene import build_anatomy_bundle, validate_coordinates  # noqa: E402
from demo_app.case_catalog import list_cases  # noqa: E402
from demo_app.clinical_summary import build_case_summary  # noqa: E402
from demo_app.data_loader import ProjectionStore  # noqa: E402
from demo_app.finding_map import build_map_findings  # noqa: E402
from demo_app.image_viewer import build_projection_figure  # noqa: E402
from demo_app.model_runtime import load_champion  # noqa: E402
from demo_app.correspondence_runtime import load_l2  # noqa: E402
from demo_app.pipeline import CaseInferenceResult, run_case_inference  # noqa: E402
from demo_app.ui.components import inject_styles, render_header, render_stage_nav  # noqa: E402
from demo_app.ui.finding_review import render_finding_review  # noqa: E402
from demo_app.ui.overview import render_overview  # noqa: E402
from demo_app.ui.session import (  # noqa: E402
    STAGE_OVERVIEW,
    STAGE_REVIEW,
    STAGE_SUMMARY,
    init_session,
    set_inference,
)
from demo_app.ui.summary_view import render_case_summary  # noqa: E402

try:
    from eval_biplanar_geometry import fracture_metrics
except ImportError:
    fracture_metrics = None


st.set_page_config(page_title=f"{APP_NAME} Clinical Review", layout="wide", initial_sidebar_state="collapsed")

MESH_CACHE_REV = "smooth-bone-v3"


@st.cache_resource
def _load_models():
    return load_champion(), load_l2()


@st.cache_resource
def _load_store():
    return ProjectionStore()


@st.cache_data(show_spinner=False)
def _cached_inference(case_id: str, data_sha: str):
    store = _load_store()
    champion, l2 = _load_models()
    return run_case_inference(store.get(case_id), champion, l2, data_sha)


@st.cache_data(show_spinner="Building rib anatomy…")
def _cached_anatomy_bundle(case_id: str, step: int, _mesh_rev: str = MESH_CACHE_REV):
    return build_anatomy_bundle(case_id, downsample_step=step)


def _pre_run_state(case_id: str, store: ProjectionStore) -> None:
    st.info(
        "Select a case, review the raw AP and lateral projections, "
        f"then run **{APP_NAME}** to generate findings."
    )
    case = store.get(case_id)
    st.plotly_chart(
        build_projection_figure(case.ap, case.lat, show_candidates=False, height=420, minimal_chrome=True),
        width="stretch",
    )


def main() -> None:
    inject_styles()
    store = _load_store()
    cases = list_cases(store.case_ids)
    if not cases:
        st.error("No cases available.")
        return

    init_session()
    if st.session_state.last_case is None or st.session_state.last_case not in cases:
        st.session_state.last_case = cases[0]

    # Run inference before header widgets (Streamlit forbids mutating widget keys after render).
    if st.session_state.run_pending:
        st.session_state.run_pending = False
        case_id = st.session_state.pending_inference_case or st.session_state.last_case
        st.session_state.pending_inference_case = None
        st.session_state.last_case = case_id
        with st.spinner(f"Running {APP_NAME}…"):
            result = _cached_inference(case_id, store.sha256)
            _cached_anatomy_bundle(case_id, 2)
        set_inference(result)
        st.rerun()

    case_id = st.session_state.last_case
    inference_ready = st.session_state.inference_key == case_id
    header_result: CaseInferenceResult | None = (
        st.session_state.get("inference") if inference_ready else None
    )
    header_finding_ids = (
        [f.original.finding_id for f in header_result.findings] if header_result else []
    )
    header_runtime = header_result.audit.inference_runtime_sec if header_result else None

    _selected_case, run_clicked, _eval = render_header(
        case_id=case_id,
        cases=cases,
        review=st.session_state.review,
        finding_ids=header_finding_ids,
        inference_ready=inference_ready,
        audit_runtime=header_runtime,
    )
    case_id = st.session_state.last_case

    if run_clicked:
        st.session_state.pending_inference_case = case_id
        st.session_state.run_pending = True
        st.rerun()

    inference_ready = st.session_state.inference_key == case_id
    result: CaseInferenceResult | None = st.session_state.get("inference") if inference_ready else None
    finding_ids = [f.original.finding_id for f in result.findings] if result else []

    if not inference_ready or result is None:
        _pre_run_state(case_id, store)
        return

    st.caption("Decision support for rib-fracture review. Not for diagnostic use.")

    can_review = bool(finding_ids)
    stage = render_stage_nav(st.session_state.stage, can_review=can_review)
    if stage != st.session_state.stage:
        st.session_state.stage = stage
        st.rerun()

    bundle = _cached_anatomy_bundle(case_id, 2)
    if bundle is None:
        st.error("Anatomy could not be loaded for this case.")
        return

    map_findings = build_map_findings(result, st.session_state.review, show_rejected=True)
    if st.session_state.get("coords_validated_for") != case_id:
        try:
            validate_coordinates(
                bundle,
                [p for mf in map_findings for p in (mf.point_world, mf.candidate_world) if p is not None],
            )
            st.session_state.coords_validated_for = case_id
        except RuntimeError as exc:
            st.error(str(exc))

    summary = build_case_summary(result, st.session_state.review)

    if st.session_state.stage == STAGE_OVERVIEW:
        render_overview(result=result, bundle=bundle, summary=summary)
    elif st.session_state.stage == STAGE_REVIEW:
        if not can_review:
            st.session_state.stage = STAGE_OVERVIEW
            st.rerun()
        render_finding_review(
            result=result,
            bundle=bundle,
            store=store,
            fracture_metrics=fracture_metrics,
        )
    elif st.session_state.stage == STAGE_SUMMARY:
        render_case_summary(result=result)


if __name__ == "__main__":
    main()
