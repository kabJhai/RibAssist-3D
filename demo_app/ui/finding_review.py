"""Finding review stage."""
from __future__ import annotations

import streamlit as st

from demo_app.anatomy_3d import CAMERA_PRESETS
from demo_app.anatomy_scene import AnatomyBundle
from demo_app.case_map_3d import build_case_map_figure
from demo_app.clinical_summary import build_decision_card
from demo_app.finding_map import (
    build_map_findings,
    classify_display_kind,
    display_status_label,
    map_finding_display_rib,
)
from demo_app.projection_overlay import (
    centerline_projection_paths,
    context_finding_markers,
    review_finding_projection_coords,
)
from demo_app.image_viewer import build_single_projection_figure
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding
from demo_app.rib_meshes import gt_fracture_points, nearest_gt_iid
from demo_app.ui.components import (
    camera_for_context,
    render_map_with_clicks,
    render_review_footer_nav,
    rib_ids_for_filter,
)
from demo_app.ui.session import apply_review_decision, go_overview

# Right column stacks AP, lateral, and interpretation; left 3D matches total height.
_REVIEW_PANEL_HEIGHT = 740
_PROJ_HEIGHT = int(_REVIEW_PANEL_HEIGHT * 0.285)
# Right column (~35% width) vs projection panel height → typical wide-panel aspect.
_PROJ_PANEL_ASPECT = 2.2


def _rib_label(ef: EnrichedFinding) -> str:
    o = ef.original
    return f"{o.side}{int(o.rib)}" if o.side and o.rib is not None else "-"


def _projection_coords(sel_ef, result, bundle, sel_mf):
    return review_finding_projection_coords(bundle, sel_ef, sel_mf, result)


def _eval_for_finding(ef, anatomy, fracture_metrics):
    if anatomy is None or ef.point_world is None or fracture_metrics is None:
        return None, None, None, None
    iid, dist, nearest = nearest_gt_iid(anatomy, ef.point_world)
    if iid is None:
        return None, None, None, None
    m, _ = fracture_metrics(ef.point_rec, 0.0, anatomy["fl_groups"][iid], anatomy, "", iid)
    return dist, bool(m["rib_exact"]) if m else None, iid, nearest


def _projection_kwargs(
    *,
    result,
    show_hm,
    show_cand,
    sel_ap,
    sel_lat,
    sel_comm_ap,
    sel_comm_lat,
    sel_abst_ap,
    gt_fp_ap,
    gt_fp_lat,
    rib_ap_path,
    rib_lat_path,
    ctx_ap,
    ctx_lat,
    lateral_context,
):
    return dict(
        ap_img=result.case.ap,
        ap_heatmap=result.l2.ap_heatmap if show_hm else None,
        lat_heatmap=result.l2.lat_heatmap if show_hm else None,
        heatmap_opacity=st.session_state.heatmap_opacity,
        show_candidates=show_cand,
        l2_ap_peaks=result.l2.ap_peaks,
        l2_lat_peaks=result.l2.lat_peaks,
        selected_finding_ap=sel_ap,
        selected_finding_lat=sel_lat if not lateral_context else None,
        selected_committed_ap=sel_comm_ap,
        selected_committed_lat=sel_comm_lat if not lateral_context else None,
        selected_abstained_ap=sel_abst_ap,
        gt_footprint_ap=gt_fp_ap,
        gt_footprint_lat=gt_fp_lat,
        rib_highlight_ap=rib_ap_path,
        rib_highlight_lat=rib_lat_path,
        context_findings_ap=ctx_ap,
        context_findings_lat=ctx_lat if not lateral_context else None,
    )


def render_finding_review(
    *,
    result: CaseInferenceResult,
    bundle: AnatomyBundle,
    store,
    fracture_metrics,
) -> None:
    review = st.session_state.review
    sel_id = st.session_state.selected_finding
    findings = result.findings
    if sel_id is None and findings:
        sel_id = findings[0].original.finding_id
        st.session_state.selected_finding = sel_id

    sel_ef = next((f for f in findings if f.original.finding_id == sel_id), None)
    if sel_ef is None:
        st.warning("No finding selected.")
        if st.button("Back to case overview"):
            go_overview()
            st.rerun()
        return

    ids = [f.original.finding_id for f in findings]
    idx = ids.index(sel_id) if sel_id in ids else 0
    kind = classify_display_kind(sel_ef)
    rib_txt = _rib_label(sel_ef)

    st.markdown(
        f'<p class="finding-review-compact-head">Finding {idx + 1} of {len(ids)}'
        f' · #{sel_id} · {rib_txt}'
        f' <span class="meta">· {display_status_label(kind)}</span></p>',
        unsafe_allow_html=True,
    )

    eval_overlay = st.session_state.eval_overlay
    dist_mm = rib_exact = eval_iid = nearest_w = gt_pts = None
    if eval_overlay and sel_ef.point_world is not None:
        dist_mm, rib_exact, eval_iid, nearest_w = _eval_for_finding(
            sel_ef, result.anatomy, fracture_metrics,
        )
        gt_pts = gt_fracture_points(result.anatomy, eval_iid) if eval_iid else None

    map_findings = build_map_findings(result, review, show_rejected=True, anatomy=bundle.anatomy)
    sel_mf = next((m for m in map_findings if m.finding_id == sel_id), None)
    display_rib = map_finding_display_rib(bundle, sel_mf) if sel_mf else None

    sel_ap, sel_lat, sel_comm_ap, sel_comm_lat, sel_abst_ap = _projection_coords(
        sel_ef, result, bundle, sel_mf,
    )
    lateral_context = kind == "rib_only"
    gt_fp_ap = gt_fp_lat = None
    if eval_overlay and eval_iid is not None:
        gt_fp_ap, gt_fp_lat = store.gt_footprint_for_iid(result.case.case_id, eval_iid)

    assigned_rib = sel_mf.rib if sel_mf and sel_mf.rib != "-" else None
    highlight = rib_ids_for_filter(bundle, display_rib)
    if not highlight and assigned_rib:
        highlight = rib_ids_for_filter(bundle, assigned_rib)
    rib_lb = next(iter(highlight), None) if highlight else None
    rib_ap_path, rib_lat_path = centerline_projection_paths(
        bundle.anatomy, rib_lb, result.case.ap_geo, result.case.lat_geo,
    )

    st.markdown('<div class="finding-review-workspace-marker"></div>', unsafe_allow_html=True)
    left, right = st.columns([0.65, 0.35], gap="small")

    with left:
        st.markdown('<div class="finding-review-3d-marker"></div>', unsafe_allow_html=True)
        ctrl = st.columns([0.42, 0.58])
        show_ctx = ctrl[0].toggle(
            "Show case context",
            value=st.session_state.show_case_context,
            key="rev_show_ctx",
        )
        st.session_state.show_case_context = show_ctx

        cam_btns = st.columns(len(CAMERA_PRESETS))
        for i, name in enumerate(CAMERA_PRESETS):
            if cam_btns[i].button(name, key=f"rev_cam_{name}"):
                st.session_state.camera = name
                st.rerun()

        cam = camera_for_context(bundle, highlight, st.session_state.camera)
        fig3d = build_case_map_figure(
            bundle,
            findings=map_findings,
            selected_id=sel_id,
            eval_gt_points=gt_pts if eval_overlay else None,
            eval_nearest=nearest_w if eval_overlay else None,
            height=_REVIEW_PANEL_HEIGHT,
            camera=cam,
            mode="review",
            show_other_findings=show_ctx,
            highlight_rib_ids=highlight,
        )
        render_map_with_clicks(fig3d, key="review_map")

    with right:
        st.markdown('<div class="finding-review-right-marker"></div>', unsafe_allow_html=True)

        with st.expander("Projection overlays", expanded=False):
            show_cand = st.toggle(
                "Candidate peaks",
                value=st.session_state.show_candidates,
                key="rev_cand",
            )
            show_hm = st.toggle(
                "Heatmap",
                value=st.session_state.show_heatmap,
                key="rev_hm",
            )
            st.session_state.show_candidates = show_cand
            st.session_state.show_heatmap = show_hm
            st.session_state.heatmap_opacity = st.slider(
                "Heatmap opacity",
                0.05, 0.5,
                float(st.session_state.heatmap_opacity),
                0.05,
                key="rev_hm_op",
            )

        show_ctx = st.session_state.show_case_context
        ctx_ap, ctx_lat = (
            context_finding_markers(result, map_findings, sel_id) if show_ctx else (None, None)
        )
        proj_kw = _projection_kwargs(
            result=result,
            show_hm=show_hm,
            show_cand=show_cand,
            sel_ap=sel_ap,
            sel_lat=sel_lat,
            sel_comm_ap=sel_comm_ap,
            sel_comm_lat=sel_comm_lat,
            sel_abst_ap=sel_abst_ap,
            gt_fp_ap=gt_fp_ap,
            gt_fp_lat=gt_fp_lat,
            rib_ap_path=rib_ap_path,
            rib_lat_path=rib_lat_path,
            ctx_ap=ctx_ap,
            ctx_lat=ctx_lat,
            lateral_context=lateral_context,
        )
        st.markdown('<p class="proj-panel-label">AP projection</p>', unsafe_allow_html=True)
        fig_ap = build_single_projection_figure(
            result.case.ap,
            view="ap",
            height=_PROJ_HEIGHT,
            zoom_to_finding=True,
            auto_scale=True,
            panel_aspect=_PROJ_PANEL_ASPECT,
            **proj_kw,
        )
        st.plotly_chart(fig_ap, width="stretch", key="review_ap")

        st.markdown(
            '<p class="proj-panel-label">Lateral projection</p>',
            unsafe_allow_html=True,
        )
        fig_lat = build_single_projection_figure(
            result.case.lat,
            view="lat",
            height=_PROJ_HEIGHT,
            lateral_context_only=lateral_context,
            zoom_to_finding=not lateral_context,
            auto_scale=True,
            panel_aspect=_PROJ_PANEL_ASPECT,
            **proj_kw,
        )
        st.plotly_chart(fig_lat, width="stretch", key="review_lat")
        if lateral_context:
            st.caption("No linked lateral candidate.")

        st.markdown('<div class="finding-review-interpret-marker"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            card = build_decision_card(sel_ef, eval_dist_mm=dist_mm, eval_rib_exact=rib_exact)
            st.markdown(
                f'<p class="interpret-kv"><strong>Predicted rib:</strong> {rib_txt}</p>'
                f'<p class="interpret-kv"><strong>Localization:</strong> '
                f'{display_status_label(kind)}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p class="decision-headline">{card["headline"]}</p>', unsafe_allow_html=True)
            for line in card["lines"][:2]:
                st.markdown(f'<p class="decision-line">{line}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="decision-action">{card["action"]}</p>', unsafe_allow_html=True)

            fid = sel_ef.original.finding_id
            a1, a2, a3 = st.columns(3)
            if a1.button("Accept", type="primary", width="stretch", key="rev_acc"):
                apply_review_decision(review, fid, "accepted", ids)
                st.rerun()
            if a2.button("Needs review", width="stretch", key="rev_needs"):
                apply_review_decision(review, fid, "needs_review", ids)
                st.rerun()
            if a3.button("Reject", width="stretch", key="rev_rej"):
                apply_review_decision(review, fid, "rejected", ids)
                st.rerun()

            with st.expander("Technical details", expanded=False):
                o = sel_ef.original
                st.write(f"Detection confidence: {o.detection_confidence:.3f}")
                st.write(f"Addressing score: {(o.address_score or 0):.3f}")
                st.write(f"Source: {o.source} · Correspondence: {sel_ef.correspondence_status}")
                if display_rib:
                    st.write(f"3D display rib: {display_rib}")
                if o.ap_xy:
                    st.write(f"AP ({o.ap_xy[0]:.1f}, {o.ap_xy[1]:.1f})")
                if sel_ef.committed_edge:
                    e = sel_ef.committed_edge
                    st.write(
                        f"Pair scores AP {e.ap_score:.3f} · Lat {e.lat_score:.3f} · "
                        f"ΔSI {e.dsi_vox:.1f} vox"
                    )
                for line in card["lines"][2:]:
                    st.write(line)

    render_review_footer_nav(
        idx=idx,
        total=len(ids),
        finding_id=sel_id,
        rib_label=rib_txt,
        ids=ids,
    )
