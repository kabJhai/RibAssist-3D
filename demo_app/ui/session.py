"""Session state for the three-stage clinical workflow."""
from __future__ import annotations

import streamlit as st

from demo_app.pipeline import CaseInferenceResult
from demo_app.review_state import default_review_map

STAGE_OVERVIEW = "overview"
STAGE_REVIEW = "review"
STAGE_SUMMARY = "summary"

STAGE_LABELS = {
    STAGE_OVERVIEW: "Case overview",
    STAGE_REVIEW: "Finding review",
    STAGE_SUMMARY: "Case summary",
}


def init_session() -> None:
    defaults: dict = {
        "stage": STAGE_OVERVIEW,
        "selected_finding": None,
        "selected_rib": None,
        "filter_side": "all",
        "filter_status": "all",
        "filter_review": "all",
        "show_rejected": False,
        "show_other_findings": True,
        "show_case_context": False,
        "rev_zoom_to_finding": True,
        "camera": "Posterior",
        "eval_overlay": False,
        "show_candidates": False,
        "show_heatmap": False,
        "heatmap_opacity": 0.2,
        "inference_key": None,
        "review": {},
        "last_case": None,
        "sort_by": "priority",
        "run_pending": False,
        "pending_inference_case": None,
        "coords_validated_for": None,
        "rib_filter": "All ribs",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_inference_state() -> None:
    """Clear review/inference without touching widget-bound case selection."""
    st.session_state.inference_key = None
    st.session_state.selected_finding = None
    st.session_state.selected_rib = None
    st.session_state.review = {}
    st.session_state.stage = STAGE_OVERVIEW
    st.session_state.coords_validated_for = None
    st.session_state.rib_filter = "All ribs"
    st.session_state.camera = "Posterior"


def on_case_changed() -> None:
    """Selectbox callback. Case id is already on session key `last_case`."""
    clear_inference_state()


def set_inference(result: CaseInferenceResult) -> None:
    st.session_state.inference_key = result.case.case_id
    st.session_state.inference = result
    st.session_state.last_case = result.case.case_id
    ids = [f.original.finding_id for f in result.findings]
    st.session_state.review = default_review_map(ids)
    st.session_state.selected_finding = None
    st.session_state.selected_rib = None
    st.session_state.rib_filter = "All ribs"
    st.session_state.camera = "Posterior"
    st.session_state.stage = STAGE_OVERVIEW


def review_progress(review: dict[int, str], finding_ids: list[int]) -> str:
    if not finding_ids:
        return "No findings"
    accepted = sum(1 for fid in finding_ids if review.get(fid) == "accepted")
    needs = sum(1 for fid in finding_ids if review.get(fid) == "needs_review")
    rejected = sum(1 for fid in finding_ids if review.get(fid) == "rejected")
    pending = len(finding_ids) - accepted - needs - rejected
    return f"{accepted} accepted · {needs} needs review · {pending} pending · {rejected} rejected"


def open_finding_review(finding_id: int) -> None:
    st.session_state.selected_finding = finding_id
    st.session_state.stage = STAGE_REVIEW


def select_rib(rib: str | None) -> None:
    st.session_state.selected_rib = rib


def go_overview() -> None:
    st.session_state.stage = STAGE_OVERVIEW


def go_summary() -> None:
    st.session_state.stage = STAGE_SUMMARY


def advance_after_decision(
    review: dict[int, str],
    finding_ids: list[int],
    decided_id: int,
) -> int | None:
    """Return the next unresolved finding after a review decision, or None if done."""
    unresolved = {
        fid for fid in finding_ids
        if review.get(fid, "pending") in ("pending", "needs_review")
    }
    if not unresolved:
        return None
    try:
        start = finding_ids.index(decided_id) + 1
        scan = finding_ids[start:] + finding_ids[:start]
    except ValueError:
        scan = finding_ids
    for fid in scan:
        if fid in unresolved:
            return fid
    return None


def apply_review_decision(
    review: dict[int, str],
    finding_id: int,
    status: str,
    finding_ids: list[int],
) -> None:
    from demo_app.review_state import set_review

    set_review(review, finding_id, status)
    nxt = advance_after_decision(review, finding_ids, finding_id)
    if nxt is not None:
        open_finding_review(nxt)
    else:
        go_summary()
