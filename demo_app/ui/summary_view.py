"""Case summary / completion stage."""
from __future__ import annotations

import json

import streamlit as st

from demo_app.clinical_summary import build_case_summary, build_completion_impression, format_rib_groups
from demo_app.finding_map import classify_display_kind, review_state_label
from demo_app.pipeline import CaseInferenceResult
from demo_app.ui.session import STAGE_OVERVIEW, STAGE_REVIEW, go_overview, open_finding_review


def render_case_summary(
    *,
    result: CaseInferenceResult,
) -> None:
    review = st.session_state.review
    summary = build_case_summary(result, review)
    impression = build_completion_impression(summary, review)

    st.markdown("### Case summary")
    st.write(impression)

    c1, c2, c3, c4 = st.columns(4)
    accepted = sum(1 for v in review.values() if v == "accepted")
    needs = sum(1 for v in review.values() if v == "needs_review")
    rejected = sum(1 for v in review.values() if v == "rejected")
    pending = summary.detected - accepted - needs - rejected
    c1.metric("Reviewer accepted", accepted)
    c2.metric("Needs review", needs)
    c3.metric("Reviewer rejected", rejected)
    c4.metric("Pending", pending)

    st.markdown("**Model localization breakdown**")
    st.write(
        f"{summary.localized} localized · {summary.candidates} candidate · "
        f"{summary.rib_only} rib-level only"
    )

    st.markdown("**Rib-level grouping**")
    st.text(format_rib_groups(summary.rib_groups))

    groups = {"Reviewer accepted": [], "Needs review": [], "Reviewer rejected": [], "Pending": []}
    for ef in result.findings:
        fid = ef.original.finding_id
        rv = review.get(fid, "pending")
        rib = f"{ef.original.side}{int(ef.original.rib)}" if ef.original.side and ef.original.rib else "-"
        label = f"#{fid} {rib} · {classify_display_kind(ef).replace('_', ' ')} · {review_state_label(rv)}"
        if rv == "accepted":
            groups["Reviewer accepted"].append((fid, label))
        elif rv == "needs_review":
            groups["Needs review"].append((fid, label))
        elif rv == "rejected":
            groups["Reviewer rejected"].append((fid, label))
        else:
            groups["Pending"].append((fid, label))

    for title, items in groups.items():
        if not items:
            continue
        with st.expander(f"{title} ({len(items)})", expanded=title in ("Needs review", "Pending")):
            for fid, label in items:
                if st.button(label, key=f"sum_{title}_{fid}"):
                    open_finding_review(fid)
                    st.session_state.stage = STAGE_REVIEW
                    st.rerun()

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("Return to overview", width="stretch"):
        go_overview()
        st.rerun()
    if b2.button("Review unresolved findings", width="stretch", disabled=not groups["Pending"] and not groups["Needs review"]):
        unresolved = groups["Pending"] or groups["Needs review"]
        if unresolved:
            open_finding_review(unresolved[0][0])
            st.session_state.stage = STAGE_REVIEW
            st.rerun()
    st.download_button(
        "Download audit JSON",
        json.dumps(result.audit.to_dict(), indent=2),
        file_name=f"ribassist_audit_{result.case.case_id}.json",
        mime="application/json",
        width="stretch",
    )

    with st.expander("Technical audit", expanded=False):
        st.caption("Checkpoint hashes, paths, and policy details for evaluators.")
        st.json(result.audit.to_dict())
