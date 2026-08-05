"""Shared UI helpers for the clinical workflow."""
from __future__ import annotations

import streamlit as st

from demo_app.config import APP_NAME
from demo_app.anatomy_3d import CAMERA_PRESETS, camera_focus_on_rib
from demo_app.anatomy_scene import AnatomyBundle
from demo_app.clinical_summary import CaseSummary, short_impression
from demo_app.finding_map import (
    MapFinding,
    build_map_findings,
    build_rib_focus_narrative,
    classify_display_kind,
    confidence_category,
    display_status_label,
    map_finding_display_rib,
    review_state_label,
    rib_label_id,
    status_indicator,
)
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding
from demo_app.plotly_compat import figure_for_streamlit
from demo_app.ui.session import STAGE_LABELS, on_case_changed


CLINICAL_CSS = """
<style>
/* Clear Streamlit's fixed top toolbar (Deploy / menu) */
.block-container {
    padding-top: 3.25rem;
    padding-bottom: 4.5rem;
    max-width: 100%;
}
section[data-testid="stMain"] > div.block-container {
    padding-top: 3.25rem;
}
.card {
    background: #fff; border: 1px solid #e2e5ea; border-radius: 8px;
    padding: 0.75rem 0.9rem; margin-bottom: 0.65rem;
}
.card-title { font-size: 0.78rem; font-weight: 600; color: #5c6578;
              text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem; }
.decision-headline { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.35rem; }
.decision-line { font-size: 0.9rem; color: #374151; margin: 0.15rem 0; }
.decision-action { font-size: 0.88rem; color: #4b5563; margin-top: 0.45rem; font-style: italic; }
div[data-testid="column"] { min-width: 0; }
.finding-table-header {
    font-size: 0.72rem; font-weight: 600; color: #8b93a7;
    text-transform: uppercase; letter-spacing: 0.04em;
    padding: 0.15rem 0.35rem 0.35rem; border-bottom: 1px solid #2a3140;
}
div:has(> .findings-compact-row-marker) button {
    padding: 0.18rem 0.45rem !important; min-height: 1.65rem !important;
    font-size: 0.82rem !important; line-height: 1.2 !important;
    border-radius: 4px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.review-footer-marker) {
    position: sticky;
    bottom: 0;
    z-index: 100;
    background: rgba(14, 17, 23, 0.96);
    backdrop-filter: blur(6px);
    border-top: 1px solid #3a4254 !important;
    padding-top: 0.65rem !important;
    margin-top: 0.5rem;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.review-footer-marker) button {
    min-height: 2.75rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}
.finding-review-compact-head {
    font-size: 0.95rem; font-weight: 600; margin: 0 0 0.35rem 0;
    line-height: 1.35;
}
.finding-review-compact-head span.meta {
    font-weight: 500; color: #8b93a7; font-size: 0.88rem;
}
.interpret-panel {
    background: #fff; border: 1px solid #e2e5ea; border-radius: 8px;
    padding: 0.65rem 0.75rem; margin-top: 0.35rem;
}
.interpret-kv { font-size: 0.82rem; margin: 0.1rem 0 0.25rem; color: #374151; }
.interpret-kv strong { color: #5c6578; font-weight: 600; }
.proj-panel-label {
    font-size: 0.72rem; font-weight: 600; color: #5c6578;
    text-transform: uppercase; letter-spacing: 0.03em;
    margin: 0.15rem 0 0.1rem;
}
div:has(> .finding-review-workspace-marker) div[data-testid="stPlotlyChart"] {
    min-height: 0;
}
div:has(> .finding-review-3d-marker) div[data-testid="stPlotlyChart"] {
    min-height: calc(100vh - 13.5rem);
}
div:has(> .finding-review-right-marker) div[data-testid="stPlotlyChart"] {
    max-height: 32vh;
}
@media (max-width: 1050px) {
    div:has(> .finding-review-workspace-marker) > div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: wrap !important;
    }
    div:has(> .finding-review-workspace-marker) div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    div:has(> .finding-review-3d-marker) div[data-testid="stPlotlyChart"] {
        min-height: 52vh;
    }
    div:has(> .finding-review-right-marker) div[data-testid="stPlotlyChart"] {
        max-height: none;
    }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)


def _rib_label(ef: EnrichedFinding) -> str:
    o = ef.original
    return f"{o.side}{int(o.rib)}" if o.side and o.rib is not None else "-"


def _finding_rows(
    result: CaseInferenceResult,
    review: dict[int, str],
    *,
    bundle: AnatomyBundle | None = None,
    show_rejected: bool = False,
) -> list[dict]:
    map_by_id: dict[int, MapFinding] = {}
    if bundle is not None:
        for mf in build_map_findings(
            result, review, show_rejected=show_rejected, anatomy=bundle.anatomy,
        ):
            map_by_id[mf.finding_id] = mf

    rows = []
    for ef in result.findings:
        fid = ef.original.finding_id
        kind = classify_display_kind(ef)
        mf = map_by_id.get(fid)
        if mf is not None and bundle is not None:
            rib = map_finding_display_rib(bundle, mf) or _rib_label(ef)
        else:
            rib = _rib_label(ef)
        rows.append(
            {
                "finding_id": fid,
                "rib": rib,
                "side": ef.original.side or (rib[:1] if rib and rib != "-" else ""),
                "status": display_status_label(kind),
                "status_key": kind,
                "indicator": status_indicator(kind),
                "confidence": confidence_category(ef),
                "review": review_state_label(review.get(fid, "pending")),
                "review_key": review.get(fid, "pending"),
                "rib_label_id": rib_label_id(result.anatomy, ef),
            }
        )
    return rows


def filter_findings(
    rows: list[dict],
    *,
    side: str,
    rib: str | None,
    status: str,
    review_filter: str,
    show_rejected: bool,
) -> list[dict]:
    out = []
    for row in rows:
        if not show_rejected and row["review_key"] == "rejected":
            continue
        if side != "all" and row["side"] != side:
            continue
        if rib and row["rib"] != rib:
            continue
        if status != "all" and row["status_key"] != status:
            continue
        if review_filter != "all" and row["review_key"] != review_filter:
            continue
        out.append(row)
    return out


def sort_findings(rows: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "rib":
        return sorted(rows, key=lambda r: (r["rib"], r["finding_id"]))
    if sort_by == "confidence":
        order = {"Low": 0, "Moderate": 1, "High": 2}
        return sorted(rows, key=lambda r: (order.get(r["confidence"], 9), r["finding_id"]))
    if sort_by == "review":
        order = {"Pending": 0, "Needs review": 1, "Accepted": 2, "Rejected": 3}
        return sorted(rows, key=lambda r: (order.get(r["review"], 9), r["finding_id"]))
    if sort_by == "priority":
        conf = {"Low": 0, "Moderate": 1, "High": 2}
        rev = {"Pending": 0, "Needs review": 1, "Accepted": 2, "Rejected": 3}
        return sorted(rows, key=lambda r: (rev.get(r["review"], 9), conf.get(r["confidence"], 9), r["finding_id"]))
    return sorted(rows, key=lambda r: r["finding_id"])


def render_stage_nav(current: str, *, can_review: bool) -> str:
    cols = st.columns(len(STAGE_LABELS))
    for col, (key, label) in zip(cols, STAGE_LABELS.items()):
        disabled = key == "review" and not can_review
        if col.button(
            label,
            key=f"stage_btn_{key}",
            width="stretch",
            type="primary" if key == current else "secondary",
            disabled=disabled,
        ):
            return key
    return current


def render_header(
    *,
    case_id: str,
    cases: list[str],
    review: dict[int, str],
    finding_ids: list[int],
    inference_ready: bool,
    audit_runtime: float | None,
) -> tuple[str, bool, bool]:
    accepted = sum(1 for fid in finding_ids if review.get(fid) == "accepted")
    needs = sum(1 for fid in finding_ids if review.get(fid) == "needs_review")
    rejected = sum(1 for fid in finding_ids if review.get(fid) == "rejected")
    pending = len(finding_ids) - accepted - needs - rejected
    decided = accepted + needs + rejected
    total = len(finding_ids)

    top = st.container(border=True)
    with top:
        c1, c2, c3, c4, c5 = st.columns([0.14, 0.28, 0.18, 0.18, 0.22])
        with c1:
            st.markdown(f"**{APP_NAME}**")
        with c2:
            selected_case = st.selectbox(
                "Case",
                cases,
                label_visibility="collapsed",
                key="last_case",
                on_change=on_case_changed,
            )
        with c3:
            if inference_ready and total:
                st.caption(f"Review progress: {decided} of {total}")
                st.progress(decided / total if total else 0.0)
                st.caption(
                    f"Accepted {accepted} · Needs review {needs} · "
                    f"Rejected {rejected} · Pending {pending}"
                )
            elif inference_ready:
                st.caption("No findings")
            else:
                st.caption("No inference run")
        with c4:
            with st.popover("Developer", width="stretch"):
                eval_overlay = st.toggle(
                    "Evaluation overlay",
                    value=st.session_state.get("eval_overlay", False),
                    key="hdr_eval",
                )
                st.session_state.eval_overlay = eval_overlay
        with c5:
            run = st.button(
                f"Run again" if inference_ready else f"Run {APP_NAME}",
                type="primary",
                width="stretch",
                key="hdr_run",
            )
        if inference_ready and audit_runtime is not None:
            st.caption(f"Live inference · {audit_runtime:.1f}s · frozen policy")
    eval_overlay = st.session_state.get("eval_overlay", False)
    return selected_case, run, eval_overlay


def render_case_summary_card(summary: CaseSummary, review: dict[int, str] | None = None) -> None:
    with st.container(border=True):
        st.markdown("**Case summary**")
        st.caption(
            f"{summary.detected} findings · {summary.localized} localized · "
            f"{summary.candidates} candidates · {summary.rib_only} rib-level only"
        )
        if review is not None:
            decided = sum(1 for v in review.values() if v != "pending")
            st.caption(f"{decided} of {summary.detected} findings reviewed by clinician")
        st.write(short_impression(summary))


def render_rib_navigator(
    rib_groups: dict[str, int],
    *,
    bundle: AnatomyBundle | None = None,
    map_findings: list[MapFinding] | None = None,
    result: CaseInferenceResult | None = None,
) -> None:
    with st.container(border=True):
        st.markdown("**Rib navigator**")
        options = ["All ribs"]
        labels: dict[str, str | None] = {"All ribs": None}
        for rib in sorted(rib_groups.keys(), key=lambda r: (r[0], int(r[1:]))):
            label = f"{rib} ({rib_groups[rib]})"
            options.append(label)
            labels[label] = rib
        if "rib_filter" not in st.session_state:
            st.session_state.rib_filter = "All ribs"

        if st.session_state.rib_filter not in labels:
            st.session_state.rib_filter = "All ribs"

        # Keep selected_rib in sync with the dropdown every rerun (not only on_change).
        st.session_state.selected_rib = labels[st.session_state.rib_filter]

        def _sync_rib_filter() -> None:
            st.session_state.selected_rib = labels[st.session_state.rib_filter]

        st.selectbox(
            "Filter by rib",
            options,
            key="rib_filter",
            on_change=_sync_rib_filter,
            label_visibility="collapsed",
        )

        sel_rib = st.session_state.selected_rib
        if sel_rib and bundle is not None and map_findings is not None and result is not None:
            st.markdown(build_rib_focus_narrative(bundle, map_findings, sel_rib, result))
        else:
            st.caption(
                "Showing the full rib cage. Select a rib to highlight it and see "
                "a summary of model findings on that level."
            )


def render_review_footer_nav(
    *,
    idx: int,
    total: int,
    finding_id: int,
    rib_label: str,
    ids: list[int],
) -> None:
    """Sticky bottom navigation for finding review."""
    from demo_app.ui.session import go_overview, open_finding_review

    with st.container(border=True):
        st.markdown('<div class="review-footer-marker"></div>', unsafe_allow_html=True)
        nav = st.columns([0.22, 0.36, 0.21, 0.21])
        if nav[0].button("← Back to case overview", width="stretch", key="rev_footer_back"):
            go_overview()
            st.rerun()
        nav[1].markdown(
            f"<div style='padding-top:0.55rem;text-align:center;font-weight:600;'>"
            f"Finding {idx + 1} of {total} · #{finding_id} · {rib_label}</div>",
            unsafe_allow_html=True,
        )
        if nav[2].button("← Previous", disabled=idx <= 0, width="stretch", key="rev_footer_prev"):
            open_finding_review(ids[idx - 1])
            st.rerun()
        if nav[3].button("Next →", disabled=idx >= total - 1, width="stretch", key="rev_footer_next"):
            open_finding_review(ids[idx + 1])
            st.rerun()


def render_filter_bar() -> tuple[str, str, str, str, bool]:
    c1, c2, c3, c4, c5 = st.columns(5)
    side = c1.selectbox("Side", ["all", "L", "R"], index=["all", "L", "R"].index(st.session_state.filter_side), key="flt_side")
    status = c2.selectbox(
        "Status", ["all", "localized", "candidate", "rib_only"],
        index=["all", "localized", "candidate", "rib_only"].index(st.session_state.filter_status),
        key="flt_status",
    )
    review_f = c3.selectbox(
        "Review", ["all", "pending", "accepted", "needs_review", "rejected"],
        index=["all", "pending", "accepted", "needs_review", "rejected"].index(st.session_state.filter_review),
        key="flt_review",
    )
    sort_options = ["priority", "finding", "rib", "confidence", "review"]
    sort_by = c4.selectbox(
        "Sort",
        sort_options,
        index=sort_options.index(st.session_state.sort_by) if st.session_state.sort_by in sort_options else 0,
        key="flt_sort",
    )
    show_rejected = c5.checkbox("Show rejected", value=st.session_state.show_rejected, key="flt_rej")
    st.session_state.filter_side = side
    st.session_state.filter_status = status
    st.session_state.filter_review = review_f
    st.session_state.sort_by = sort_by
    st.session_state.show_rejected = show_rejected
    return side, status, review_f, sort_by, show_rejected


def render_findings_table(rows: list[dict], selected_id: int | None, on_select) -> None:
    st.markdown("**Findings** (click a row to open review)")
    if not rows:
        st.caption("No findings match the current filters.")
        if st.button("Reset filters", key="reset_flt"):
            st.session_state.filter_side = "all"
            st.session_state.filter_status = "all"
            st.session_state.filter_review = "all"
            st.session_state.selected_rib = None
            st.session_state.rib_filter = "All ribs"
            st.session_state.show_rejected = False
            st.rerun()
        return
    hdr = st.columns([0.10, 0.14, 0.22, 0.22, 0.32])
    hdr[0].markdown('<div class="finding-table-header">#</div>', unsafe_allow_html=True)
    hdr[1].markdown('<div class="finding-table-header">Rib</div>', unsafe_allow_html=True)
    hdr[2].markdown('<div class="finding-table-header">Model status</div>', unsafe_allow_html=True)
    hdr[3].markdown('<div class="finding-table-header">Confidence</div>', unsafe_allow_html=True)
    hdr[4].markdown('<div class="finding-table-header">Review</div>', unsafe_allow_html=True)

    for row in rows:
        fid = row["finding_id"]
        selected = fid == selected_id
        label = (
            f"#{fid}    {row['rib']:<4}    {row['status']:<16}    "
            f"{row['confidence']:<10}    {row['review']}"
        )
        bar_col, btn_col = st.columns([0.006, 0.994], gap="small")
        with bar_col:
            if selected:
                st.markdown(
                    '<div style="width:3px;height:1.6rem;background:#e53935;'
                    'border-radius:2px;margin:0.12rem 0;"></div>',
                    unsafe_allow_html=True,
                )
        with btn_col:
            st.markdown('<div class="findings-compact-row-marker"></div>', unsafe_allow_html=True)
            if st.button(label, key=f"row_{fid}", width="stretch", type="secondary"):
                on_select(fid)


@st.fragment
def render_findings_panel(
    result: CaseInferenceResult,
    review: dict[int, str],
    sel_rib: str | None,
    sel_id: int | None,
    on_select,
    *,
    bundle: AnatomyBundle | None = None,
) -> None:
    """Filters and findings table without rebuilding the 3D view."""
    side, status, review_f, sort_by, show_rejected = render_filter_bar()
    rows = _finding_rows(result, review, bundle=bundle, show_rejected=show_rejected)
    filtered = filter_findings(
        rows, side=side, rib=sel_rib, status=status,
        review_filter=review_f, show_rejected=show_rejected,
    )
    filtered = sort_findings(filtered, sort_by)
    render_findings_table(filtered, sel_id, on_select)


def finding_id_from_plotly(selection) -> int | None:
    if selection is None:
        return None
    points = getattr(selection, "points", None) or selection.get("points", [])
    for pt in points:
        cd = pt.get("customdata") if isinstance(pt, dict) else getattr(pt, "customdata", None)
        if cd is None:
            continue
        try:
            return int(cd[0] if isinstance(cd, (list, tuple)) else cd)
        except (TypeError, ValueError, IndexError):
            continue
    return None


def render_map_with_clicks(fig, key: str) -> int | None:
    chart_state = st.plotly_chart(
        figure_for_streamlit(fig),
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode="points",
        theme=None,
        config={"displayModeBar": True},
    )
    if chart_state is None:
        return None
    selection = getattr(chart_state, "selection", None)
    if selection is None and isinstance(chart_state, dict):
        selection = chart_state.get("selection")
    return finding_id_from_plotly(selection)


def rib_ids_for_filter(bundle: AnatomyBundle, rib: str | None) -> set[int] | None:
    if not rib or bundle.anatomy is None:
        return None
    side, num = rib[0], int(rib[1:])
    ids = set()
    for lb, meta in bundle.anatomy.get("info", {}).items():
        if meta.get("side") == side and int(meta.get("num", -1)) == num:
            ids.add(int(lb))
    return ids or None


def camera_for_context(bundle: AnatomyBundle, rib_ids: set[int] | None, preset: str) -> dict:
    if rib_ids and len(rib_ids) == 1:
        return camera_focus_on_rib(bundle, next(iter(rib_ids)), preset)
    return CAMERA_PRESETS.get(preset, CAMERA_PRESETS["Posterior"])
