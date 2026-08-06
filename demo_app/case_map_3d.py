# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Case-level 3D rib map with all findings."""
from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go

from demo_app.anatomy_3d import (
    COLOR_ERROR,
    COLOR_FRACTURE,
    COLOR_GT,
    CAMERA_PRESETS,
    _centerline_for_label,
    local_centerline_segment,
)
from demo_app.anatomy_scene import AnatomyBounds, AnatomyBundle, log_figure_diagnostics, nearest_rib_label, snap_to_rib_surface
from demo_app.finding_map import MapFinding
from demo_app.rib_meshes import RibMesh

# Opaque bone surfaces; per-rib tone helps at overlaps
SCENE_BG = "#1E2430"
COLOR_RIB_L = ("#D8D2C8", "#CFC9BF", "#C6C0B6", "#BDB7AD")
COLOR_RIB_R = ("#EDE5DA", "#E4DCD1", "#DBD3C8", "#D2CABF")
COLOR_RIB_SELECTED = "#8FA8DC"
COLOR_RIB_CUE = "#6B8FD4"
COLOR_CANDIDATE = "#FFB300"
COLOR_FRACTURE_DIM = "#FF7043"

RIB_MESH_OPACITY = 1.0
MESH_LIGHTING = dict(ambient=0.90, diffuse=0.42, specular=0.18, roughness=0.32, fresnel=0.03)
SELECTED_MESH_LIGHTING = dict(ambient=0.78, diffuse=0.62, specular=0.42, roughness=0.18, fresnel=0.08)
DIMMED_MESH_LIGHTING = dict(ambient=0.94, diffuse=0.28, specular=0.08, roughness=0.42, fresnel=0.01)
CL_FALLBACK_WIDTH = 4
CL_FALLBACK_COLOR = "#C5BDB0"

UNSELECTED_DIM = 0.45
CONTEXT_LOCALIZED_DIM = 0.28
CONTEXT_CANDIDATE_DIM = 0.18
CONTEXT_OTHER_DIM = 0.25
MapMode = Literal["overview", "review"]
RIB_CONTEXT_OPACITY = 0.42


def _label_id_for_rib_str(anatomy, rib: str) -> int | None:
    if anatomy is None or not rib or rib == "-" or len(rib) < 2:
        return None
    try:
        side, num = rib[0], int(rib[1:])
    except ValueError:
        return None
    for lb, meta in anatomy.get("info", {}).items():
        if meta.get("side") == side and int(meta.get("num", -1)) == num:
            return int(lb)
    return None


def _finding_point(mf: MapFinding) -> np.ndarray | None:
    return mf.point_world if mf.point_world is not None else mf.candidate_world


def _rib_bone_color(rib: RibMesh, *, selected: bool, rib_cue: bool) -> str:
    if selected:
        return COLOR_RIB_SELECTED
    if rib_cue:
        return COLOR_RIB_CUE
    palette = COLOR_RIB_L if rib.side == "L" else COLOR_RIB_R
    return palette[(max(1, rib.num) - 1) % len(palette)]


def _add_rib_mesh(
    fig: go.Figure,
    mesh: RibMesh,
    *,
    color: str,
    opacity: float,
    name: str,
    lighting: dict | None = None,
) -> None:
    v, f = mesh.vertices, mesh.faces
    fig.add_trace(
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color,
            opacity=opacity,
            flatshading=False,
            lighting=lighting or MESH_LIGHTING,
            contour=dict(show=False),
            name=name,
            showlegend=False,
            visible=True,
            hovertemplate=f"Rib {mesh.side}{mesh.num}<extra></extra>",
        )
    )


def _add_centerline(
    fig: go.Figure,
    cl: np.ndarray,
    *,
    color: str,
    width: int,
    opacity: float,
    name: str,
    dash: bool = False,
) -> None:
    if cl is None or len(cl) < 2:
        return
    fig.add_trace(
        go.Scatter3d(
            x=cl[:, 0], y=cl[:, 1], z=cl[:, 2],
            mode="lines",
            line=dict(color=color, width=width, dash="dash" if dash else "solid"),
            opacity=opacity,
            name=name,
            showlegend=False,
            visible=True,
        )
    )


def _add_segment(
    fig, anatomy, label_id, point, *, color, width, opacity, dashed, name, customdata, showlegend=False,
):
    if anatomy is None or label_id is None or point is None:
        return
    cl = _centerline_for_label(anatomy, label_id)
    seg = local_centerline_segment(cl, point) if cl is not None else None
    if seg is None or len(seg) < 2:
        return
    if dashed:
        step = max(1, len(seg) // 10)
        seg = seg[::step]
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=max(2, width - 2), dash="dash"),
                marker=dict(size=width, color=color, opacity=opacity, symbol="circle-open"),
                name=name, customdata=customdata * len(seg), showlegend=showlegend, visible=True,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines", line=dict(color=color, width=width), opacity=opacity,
                name=name, customdata=customdata, showlegend=showlegend, visible=True,
            )
        )


def _debug_overlay(fig, bundle: AnatomyBundle, selected_pt, selected_id):
    b = bundle.bounds.padded(0.05)
    # bounding box edges
    corners = [
        (b.xmin, b.ymin, b.zmin), (b.xmax, b.ymin, b.zmin),
        (b.xmax, b.ymax, b.zmin), (b.xmin, b.ymax, b.zmin),
        (b.xmin, b.ymin, b.zmax), (b.xmax, b.ymin, b.zmax),
        (b.xmax, b.ymax, b.zmax), (b.xmin, b.ymax, b.zmax),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, c in edges:
        p, q = corners[a], corners[c]
        fig.add_trace(
            go.Scatter3d(
                x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]],
                mode="lines", line=dict(color="cyan", width=2), showlegend=False, name="Debug bbox",
            )
        )
    if bundle.meshes and bundle.meshes.ribs:
        cen = np.mean(np.vstack([m.vertices for m in bundle.meshes.ribs]), axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[cen[0]], y=[cen[1]], z=[cen[2]], mode="markers+text",
                text=["mesh centroid"], marker=dict(size=4, color="cyan"), showlegend=False,
            )
        )
    if selected_pt is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[selected_pt[0]], y=[selected_pt[1]], z=[selected_pt[2]],
                mode="markers", marker=dict(size=6, color="white", symbol="diamond"), showlegend=False,
                name="Debug selected pt",
            )
        )
    fig.add_annotation(
        text=(
            f"meshes={len(bundle.meshes.ribs) if bundle.meshes else 0} "
            f"cl_fallback={len(bundle.centerline_fallback_labels)} "
            f"sel=#{selected_id}<br>"
            f"x=[{b.xmin:.0f},{b.xmax:.0f}] y=[{b.ymin:.0f},{b.ymax:.0f}] z=[{b.zmin:.0f},{b.zmax:.0f}]"
        ),
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        font=dict(size=10, color="cyan"), bgcolor="rgba(0,0,0,0.5)",
    )


def build_case_map_figure(
    bundle: AnatomyBundle | None,
    *,
    findings: list[MapFinding],
    selected_id: int | None = None,
    eval_gt_points: np.ndarray | None = None,
    eval_nearest: np.ndarray | None = None,
    debug: bool = False,
    height: int = 820,
    camera: dict | None = None,
    mode: MapMode = "overview",
    show_other_findings: bool = False,
    highlight_rib_ids: set[int] | None = None,
) -> go.Figure:
    fig = go.Figure()

    if bundle is None or bundle.anatomy is None:
        fig.add_annotation(text="Anatomy unavailable for this case", showarrow=False, y=0.5)
        fig.update_layout(height=height)
        return fig

    anatomy = bundle.anatomy
    info = anatomy.get("info", {})
    rib_cue_ids = {mf.rib_label_id for mf in findings if mf.kind == "rib_only" and mf.rib_label_id}
    selected_rib_ids: set[int] = set(highlight_rib_ids or [])
    if highlight_rib_ids:
        selected_rib_ids = set(highlight_rib_ids)
    elif mode == "review" and selected_id is not None:
        sel_m = next((m for m in findings if m.finding_id == selected_id), None)
        if sel_m is not None:
            p = _finding_point(sel_m)
            if p is not None:
                lb, _ = nearest_rib_label(bundle, p)
                if lb is not None:
                    selected_rib_ids.add(lb)
            if sel_m.rib_label_id is not None:
                selected_rib_ids.add(sel_m.rib_label_id)
            elif sel_m.rib:
                lb = _label_id_for_rib_str(anatomy, sel_m.rib)
                if lb is not None:
                    selected_rib_ids.add(lb)

    review_focus = mode == "review" and selected_id is not None
    focus_ribs = review_focus and bool(selected_rib_ids)
    use_rib_cue_mesh = mode == "review"
    if bundle.meshes and bundle.meshes.ribs:
        for rib in bundle.meshes.ribs:
            selected = rib.label in selected_rib_ids
            cue = use_rib_cue_mesh and rib.label in rib_cue_ids and not focus_ribs
            color = _rib_bone_color(rib, selected=selected, rib_cue=cue and not selected)
            if selected:
                lighting = SELECTED_MESH_LIGHTING
                opacity = RIB_MESH_OPACITY
            elif focus_ribs:
                lighting = MESH_LIGHTING
                opacity = RIB_CONTEXT_OPACITY
            else:
                lighting = MESH_LIGHTING
                opacity = RIB_MESH_OPACITY
            _add_rib_mesh(
                fig, rib,
                color=color,
                opacity=opacity,
                name=f"Rib {rib.side}{rib.num}",
                lighting=lighting,
            )

    # --- 2. Centerline fallback for ribs without mesh ---
    if not bundle.mesh_labels:
        for idx, cl in enumerate(bundle.centerline_world):
            if cl is None or len(cl) < 2:
                continue
            lb = idx + 1
            meta = info.get(lb, {})
            _add_centerline(
                fig, np.asarray(cl, dtype=np.float64),
                color=COLOR_RIB_CUE if lb in rib_cue_ids else CL_FALLBACK_COLOR,
                width=5, opacity=0.9,
                name=f"CL {meta.get('side', '?')}{meta.get('num', lb)}",
            )
    else:
        for lb in sorted(bundle.centerline_fallback_labels):
            if lb - 1 < len(bundle.centerline_world):
                cl = np.asarray(bundle.centerline_world[lb - 1], dtype=np.float64)
                meta = info.get(lb, {})
                _add_centerline(
                    fig, cl, color=COLOR_RIB_CUE if lb in rib_cue_ids else CL_FALLBACK_COLOR,
                    width=CL_FALLBACK_WIDTH, opacity=0.85,
                    name=f"CL {meta.get('side', '?')}{meta.get('num', lb)}",
                )

    # --- 4–7. Finding overlays (rib cues, candidates, localized, emphasis) ---
    legend_flags = {"localized": False, "candidate": False, "rib_only": False}

    for mf in findings:
        fid = mf.finding_id
        selected = fid == selected_id
        if review_focus and not selected and not show_other_findings:
            continue
        if selected:
            dim = 1.0
        elif review_focus and show_other_findings:
            if mf.kind == "localized":
                dim = CONTEXT_LOCALIZED_DIM
            elif mf.kind == "candidate":
                dim = CONTEXT_CANDIDATE_DIM
            else:
                dim = CONTEXT_OTHER_DIM
        elif review_focus:
            dim = 0.25
        else:
            dim = UNSELECTED_DIM
        cd = [[fid]]
        show_label = selected and mode == "overview"
        hover_extra = (
            f"#{fid} · {mf.rib} · {mf.kind.replace('_', ' ')} · "
            f"{mf.review_status.replace('_', ' ')}<extra></extra>"
        )

        if mf.kind == "rib_only" and mf.rib_label_id is not None:
            cl = _centerline_for_label(anatomy, mf.rib_label_id)
            if cl is not None and (selected or mode == "overview"):
                _add_centerline(
                    fig, cl, color=COLOR_RIB_CUE, width=14 if selected and mode == "review" else (10 if selected else 4),
                    opacity=0.98 if selected and mode == "review" else (0.95 if selected else 0.35),
                    name="Rib-level finding only", dash=False,
                )
            if selected and cl is not None and len(cl):
                cen = cl.mean(axis=0)
                fig.add_trace(
                    go.Scatter3d(
                        x=[cen[0]], y=[cen[1]], z=[cen[2]], mode="markers",
                        marker=dict(
                            size=14 if mode == "review" else 8,
                            color=COLOR_RIB_CUE, symbol="diamond",
                        ),
                        customdata=cd, name="Rib-level finding only",
                        hovertemplate=hover_extra, showlegend=not legend_flags["rib_only"], visible=True,
                    )
                )
                legend_flags["rib_only"] = True
        elif mf.kind == "rib_only" and selected and mf.rib:
            lb = _label_id_for_rib_str(anatomy, mf.rib)
            if lb is not None:
                cl = _centerline_for_label(anatomy, lb)
                if cl is not None:
                    _add_centerline(
                        fig, cl, color=COLOR_RIB_CUE, width=14 if mode == "review" else 10,
                        opacity=0.98 if mode == "review" else 0.95,
                        name="Rib-level finding only", dash=False,
                    )
                    cen = cl.mean(axis=0)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cen[0]], y=[cen[1]], z=[cen[2]], mode="markers",
                            marker=dict(size=14 if mode == "review" else 8, color=COLOR_RIB_CUE, symbol="diamond"),
                            customdata=cd, name="Rib-level finding only",
                            hovertemplate=hover_extra, showlegend=not legend_flags["rib_only"], visible=True,
                        )
                    )
                    legend_flags["rib_only"] = True

        elif mf.kind == "candidate" and mf.candidate_world is not None:
            raw_p = mf.candidate_world
            disp_p, anatomy_lb, _ = snap_to_rib_surface(bundle, raw_p, prefer_label=mf.rib_label_id)
            p = disp_p
            if selected or mode == "overview":
                _add_segment(
                    fig, anatomy, anatomy_lb or mf.rib_label_id, raw_p,
                    color=COLOR_CANDIDATE, width=12 if selected else 5,
                    opacity=0.85 * dim if selected else 0.45 * dim, dashed=True,
                    name="Candidate location", customdata=cd,
                    showlegend=not legend_flags["candidate"],
                )
                legend_flags["candidate"] = True
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2]],
                    mode="markers" + ("+text" if show_label else ""),
                    text=[f"#{fid}"] if show_label else None,
                    textfont=dict(size=12, color=COLOR_CANDIDATE) if show_label else None,
                    marker=dict(
                        size=24 if selected and review_focus else (20 if selected else 9),
                        color="rgba(0,0,0,0)", symbol="circle-open",
                        line=dict(width=5 if selected and review_focus else (4 if selected else 2), color=COLOR_CANDIDATE),
                    ),
                    customdata=cd, visible=True,
                    hovertemplate=hover_extra,
                    name="Candidate location", showlegend=False,
                )
            )

        elif mf.kind == "localized" and mf.point_world is not None:
            raw_p = mf.point_world
            disp_p, anatomy_lb, _ = snap_to_rib_surface(bundle, raw_p, prefer_label=mf.rib_label_id)
            p = disp_p
            if selected or mode == "overview":
                _add_segment(
                    fig, anatomy, anatomy_lb or mf.rib_label_id, raw_p,
                    color=COLOR_FRACTURE, width=16 if selected else 6,
                    opacity=dim, dashed=False,
                    name="Confirmed 3D localization", customdata=cd,
                    showlegend=not legend_flags["localized"],
                )
                legend_flags["localized"] = True
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2]],
                    mode="markers" + ("+text" if show_label else ""),
                    text=[f"#{fid}"] if show_label else None,
                    textfont=dict(size=12, color="white") if show_label else None,
                    marker=dict(
                        size=24 if selected and review_focus else (20 if selected else 9),
                        color=COLOR_FRACTURE, symbol="circle",
                        line=dict(width=4 if selected and review_focus else (3 if selected else 1), color="white"),
                    ),
                    customdata=cd, visible=True,
                    hovertemplate=hover_extra,
                    name="Confirmed 3D localization", showlegend=False,
                )
            )

    # --- 7. GT eval (selected only) ---
    if eval_gt_points is not None and len(eval_gt_points) and selected_id is not None:
        sel = next((m for m in findings if m.finding_id == selected_id), None)
        p = sel.point_world if sel and sel.kind == "localized" else None
        fig.add_trace(
            go.Scatter3d(
                x=eval_gt_points[:, 0], y=eval_gt_points[:, 1], z=eval_gt_points[:, 2],
                mode="markers", name="Selected GT", visible=True,
                marker=dict(size=2.5, color=COLOR_GT, opacity=0.35),
            )
        )
        if p is not None and eval_nearest is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0], eval_nearest[0]], y=[p[1], eval_nearest[1]], z=[p[2], eval_nearest[2]],
                    mode="lines", line=dict(color=COLOR_ERROR, width=4), name="Error", visible=True,
                )
            )

    if debug:
        sel_m = next((m for m in findings if m.finding_id == selected_id), None)
        _debug_overlay(fig, bundle, _finding_point(sel_m) if sel_m else None, selected_id)

    # Scene ranges from full anatomy, not fracture points
    ranges = bundle.bounds.as_plotly_ranges()
    cam = camera or CAMERA_PRESETS["Posterior"]

    title = "Focused finding, 3D anatomy" if mode == "review" else "Case overview, 3D rib map"
    title_extra = ""
    if bundle.warnings:
        title_extra = f"<br><sup>{bundle.warnings[0]}</sup>"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>{title_extra}", x=0.02, font=dict(size=14, color="white")),
        height=height,
        scene=dict(
            xaxis=dict(title="L ← LR → R", range=ranges["xaxis"], autorange=False, showbackground=True, backgroundcolor=SCENE_BG),
            yaxis=dict(title="Posterior ← AP → Anterior", range=ranges["yaxis"], autorange=False, showbackground=True, backgroundcolor=SCENE_BG),
            zaxis=dict(title="Inferior ← SI → Superior", range=ranges["zaxis"], autorange=False, showbackground=True, backgroundcolor=SCENE_BG),
            aspectmode="data",
            dragmode="turntable",
            camera=cam,
            bgcolor=SCENE_BG,
        ),
        paper_bgcolor=SCENE_BG,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=10, color="white")),
    )
    if mode == "overview":
        fig.update_layout(
            annotations=[
                dict(
                    text=(
                        "<span style='color:#FF5722'>●</span> Localized &nbsp; "
                        "<span style='color:#FFB300'>○</span> Candidate &nbsp; "
                        "<span style='color:#6B8FD4'>▬</span> Rib-level only"
                    ),
                    xref="paper", yref="paper", x=0.02, y=1.01, showarrow=False,
                    font=dict(size=10, color="#ccc"), align="left",
                )
            ],
        )

    fracture_pts = [_finding_point(m) for m in findings if _finding_point(m) is not None]
    if debug:
        log_figure_diagnostics(fig, case_id=bundle.case_id, bundle=bundle, fracture_points=fracture_pts, camera=cam)

    return fig
