# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""3D rib anatomy for the case map and finding review."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go

from demo_app.rib_meshes import CaseMeshes, RibMesh

LocalizationStatus = Literal["localized", "abstained", "unlinked", "ap_only", "unavailable"]

CAMERA_PRESETS = {
    "Anterior": dict(eye=dict(x=0.0, y=-2.2, z=0.2)),
    "Posterior": dict(eye=dict(x=0.0, y=2.2, z=0.2)),
    "Left lateral": dict(eye=dict(x=-2.2, y=0.0, z=0.2)),
    "Right lateral": dict(eye=dict(x=2.2, y=0.0, z=0.2)),
    "Reset": dict(eye=dict(x=0.0, y=-2.2, z=0.2)),
}


def camera_focus_on_rib(
    bundle,
    rib_label_id: int | None,
    preset: str = "Posterior",
) -> dict:
    """Shift the Plotly camera center toward a rib (preset eye/up stay in scene-normalized units)."""
    base = CAMERA_PRESETS.get(preset, CAMERA_PRESETS["Posterior"])
    if bundle is None or rib_label_id is None or not bundle.meshes:
        return base
    mesh = next((m for m in bundle.meshes.ribs if m.label == rib_label_id), None)
    if mesh is None or len(mesh.vertices) == 0:
        return base
    center = mesh.vertices.mean(axis=0)
    b = bundle.bounds.padded()

    def _norm(v: float, lo: float, hi: float) -> float:
        half = max(0.5 * (hi - lo), 1.0)
        return float((v - 0.5 * (lo + hi)) / half)

    cam = dict(base)
    cam["center"] = dict(
        x=_norm(center[0], b.xmin, b.xmax),
        y=_norm(center[1], b.ymin, b.ymax),
        z=_norm(center[2], b.zmin, b.zmax),
    )
    cam.setdefault("up", dict(x=0.0, y=0.0, z=1.0))
    return cam

COLOR_RIB_BG = "#BDBDBD"
COLOR_RIB_SELECTED = "#7E8CCF"
COLOR_FRACTURE = "#FF5722"
COLOR_GT = "#E040FB"
COLOR_ERROR = "#FFD54F"

RIB_BG_OPACITY = 0.24
RIB_SELECTED_OPACITY = 0.52
SEGMENT_WIDTH = 10
POINT_SIZE = 14
HALO_SIZE = 22
HALO_OPACITY = 0.18
LOCAL_HALF_MM = 20.0
SURFACE_PATCH_MM = 18.0


@dataclass
class FocalLocalization:
    status: LocalizationStatus
    finding_id: int | None = None
    predicted_rib: str | None = None
    highlight_rib_label: int | None = None
    point_world: np.ndarray | None = None
    ap_score: float | None = None
    lat_score: float | None = None
    si_diff_vox: float | None = None
    counterfactual_world: np.ndarray | None = None
    show_counterfactual: bool = False
    show_gt: bool = False
    gt_points: np.ndarray | None = None
    nearest_gt_world: np.ndarray | None = None
    error_mm: float | None = None
    rib_exact: bool | None = None
    rib_within1: bool | None = None


def _arc_lengths(cl: np.ndarray) -> np.ndarray:
    if len(cl) < 2:
        return np.zeros(len(cl))
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def local_centerline_segment(
    cl: np.ndarray,
    point_world: np.ndarray,
    half_mm: float = LOCAL_HALF_MM,
) -> np.ndarray | None:
    """Return centerline points within ±half_mm arc length of nearest point to fracture."""
    if cl is None or len(cl) < 2:
        return None
    cl = np.asarray(cl, dtype=np.float64)
    p = np.asarray(point_world, dtype=np.float64)
    idx = int(np.argmin(np.linalg.norm(cl - p[None], axis=1)))
    s = _arc_lengths(cl)
    s0 = float(s[idx])
    mask = (s >= s0 - half_mm) & (s <= s0 + half_mm)
    seg = cl[mask]
    return seg if len(seg) >= 2 else cl[max(0, idx - 1) : idx + 2]


def _centerline_for_label(anatomy, label: int) -> np.ndarray | None:
    if anatomy is None or label is None:
        return None
    cl_world = anatomy.get("cl_world", [])
    if label - 1 < 0 or label - 1 >= len(cl_world):
        return None
    cl = cl_world[label - 1]
    if cl is None or len(cl) < 2:
        return None
    return np.asarray(cl, dtype=np.float64)


def _focal_surface_patch(mesh: RibMesh, point_world: np.ndarray, radius_mm: float = SURFACE_PATCH_MM) -> np.ndarray | None:
    v = mesh.vertices
    d = np.linalg.norm(v - point_world[None], axis=1)
    patch = v[d <= radius_mm]
    return patch if len(patch) >= 3 else None


def _add_mesh(fig: go.Figure, mesh: RibMesh, *, color: str, opacity: float, name: str, showlegend: bool) -> None:
    v, f = mesh.vertices, mesh.faces
    fig.add_trace(
        go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=opacity, flatshading=False,
            name=name, showlegend=showlegend,
            hovertemplate=f"{mesh.side}{mesh.num}<extra></extra>",
        )
    )


def _add_point_cloud_ribs(fig, anatomy, highlight_lb: int | None) -> None:
    from nibabel.affines import apply_affine

    aff = anatomy["aff"]
    rs = anatomy["rs"]
    info = anatomy.get("info", {})
    for lb in sorted(info.keys()):
        if lb <= 0:
            continue
        vox = np.array(np.nonzero(rs == lb))
        if vox.size == 0:
            continue
        step = max(1, vox.shape[1] // 400)
        pts = apply_affine(aff, vox[:, ::step].T.astype(np.float64))
        selected = highlight_lb == lb
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name=f"{info[lb].get('side', '?')}{info[lb].get('num', lb)}",
                marker=dict(
                    size=1.8,
                    color=COLOR_RIB_SELECTED if selected else COLOR_RIB_BG,
                    opacity=RIB_SELECTED_OPACITY if selected else RIB_BG_OPACITY,
                ),
                showlegend=selected,
            )
        )


def _title_for(focal: FocalLocalization) -> list[str]:
    lines = ["<b>3D fracture localization</b>"]
    if focal.predicted_rib and focal.predicted_rib != "-":
        lines.append(f"Predicted rib: {focal.predicted_rib}")
    if focal.status == "localized":
        lines.append("Status: Localized")
        lines.append("Approximate fracture location shown")
        if focal.show_gt and focal.error_mm is not None:
            lines.append(f"Fracture-volume distance: {focal.error_mm:.1f} mm")
            if focal.rib_exact is not None:
                lines.append(f"Rib-exact: {'yes' if focal.rib_exact else 'no'}")
            if focal.rib_within1 is not None:
                lines.append(f"Rib±1: {'yes' if focal.rib_within1 else 'no'}")
    elif focal.status == "abstained":
        lines.append("Status: Abstained")
        lines.append("Rib level available; exact 3D fracture location not emitted")
    elif focal.status in ("unlinked", "ap_only"):
        lines.append("Status: 2D finding only")
        lines.append("No reliable biplanar 3D localization available")
    return lines


def _hover_point(focal: FocalLocalization) -> str:
    parts = [f"Finding #{focal.finding_id}" if focal.finding_id else "Finding"]
    if focal.predicted_rib:
        parts.append(f"Rib {focal.predicted_rib}")
    if focal.ap_score is not None:
        parts.append(f"AP {focal.ap_score:.3f}")
    if focal.lat_score is not None:
        parts.append(f"Lat {focal.lat_score:.3f}")
    if focal.si_diff_vox is not None:
        parts.append(f"ΔSI {focal.si_diff_vox:.1f} vox")
    parts.append("3D: Localized")
    return "<br>".join(parts) + "<extra></extra>"


def build_anatomy_figure(
    meshes: CaseMeshes | None,
    *,
    anatomy_fallback=None,
    focal: FocalLocalization | None = None,
    height: int = 720,
    camera: dict | None = None,
) -> go.Figure:
    focal = focal or FocalLocalization(status="unavailable")
    fig = go.Figure()

    if (meshes is None or not meshes.ribs) and anatomy_fallback is None:
        fig.add_annotation(text="Anatomy unavailable for this case", showarrow=False, y=0.5)
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=80, b=0))
        return fig

    hl = focal.highlight_rib_label
    show_focal = focal.status == "localized" and focal.point_world is not None

    if meshes and meshes.ribs:
        for rib in meshes.ribs:
            selected = hl == rib.label
            _add_mesh(
                fig, rib,
                color=COLOR_RIB_SELECTED if selected else COLOR_RIB_BG,
                opacity=RIB_SELECTED_OPACITY if selected else RIB_BG_OPACITY,
                name=f"{rib.side}{rib.num}",
                showlegend=selected and not show_focal,
            )
            if show_focal and selected:
                patch = _focal_surface_patch(rib, focal.point_world)
                if patch is not None:
                    fig.add_trace(
                        go.Scatter3d(
                            x=patch[:, 0], y=patch[:, 1], z=patch[:, 2],
                            mode="markers", name="Fracture vicinity",
                            marker=dict(size=3, color=COLOR_FRACTURE, opacity=0.55),
                            showlegend=True,
                        )
                    )
    elif anatomy_fallback is not None:
        _add_point_cloud_ribs(fig, anatomy_fallback, hl)

    # Local centerline segment (main fracture cue)
    if show_focal and anatomy_fallback is not None and hl is not None:
        cl = _centerline_for_label(anatomy_fallback, hl)
        seg = local_centerline_segment(cl, focal.point_world) if cl is not None else None
        if seg is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                    mode="lines", name="Fracture segment",
                    line=dict(color=COLOR_FRACTURE, width=SEGMENT_WIDTH),
                    showlegend=True,
                )
            )

    if show_focal:
        p = focal.point_world
        # Uncertainty halo
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="markers", name="Localization halo",
                marker=dict(size=HALO_SIZE, color=COLOR_FRACTURE, opacity=HALO_OPACITY),
                showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="markers", name="Predicted fracture",
                marker=dict(
                    size=POINT_SIZE, color=COLOR_FRACTURE, symbol="circle",
                    line=dict(width=3, color="white"),
                ),
                hovertemplate=_hover_point(focal),
                showlegend=True,
            )
        )

    if focal.status == "abstained" and focal.show_counterfactual and focal.counterfactual_world is not None:
        c = focal.counterfactual_world
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]], y=[c[1]], z=[c[2]],
                mode="markers", name="Counterfactual (eval only)",
                marker=dict(size=8, color=COLOR_FRACTURE, symbol="diamond-open",
                            line=dict(width=2, color=COLOR_FRACTURE, dash="dash")),
                hovertemplate="Uncommitted candidate (evaluation only)<extra></extra>",
            )
        )

    if focal.show_gt and focal.gt_points is not None and len(focal.gt_points):
        fig.add_trace(
            go.Scatter3d(
                x=focal.gt_points[:, 0], y=focal.gt_points[:, 1], z=focal.gt_points[:, 2],
                mode="markers", name="Selected GT fracture",
                marker=dict(size=2.5, color=COLOR_GT, opacity=0.32),
            )
        )

    if focal.show_gt and show_focal and focal.nearest_gt_world is not None:
        p = focal.point_world
        g = focal.nearest_gt_world
        fig.add_trace(
            go.Scatter3d(
                x=[g[0]], y=[g[1]], z=[g[2]],
                mode="markers", name="Nearest GT voxel",
                marker=dict(size=6, color=COLOR_GT, symbol="diamond",
                            line=dict(width=1, color="white")),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[p[0], g[0]], y=[p[1], g[1]], z=[p[2], g[2]],
                mode="lines", name="Error segment",
                line=dict(color=COLOR_ERROR, width=4),
            )
        )
        if focal.error_mm is not None:
            mid = (p + g) / 2.0
            fig.add_trace(
                go.Scatter3d(
                    x=[mid[0]], y=[mid[1]], z=[mid[2]],
                    mode="text", text=[f"{focal.error_mm:.1f} mm"],
                    textfont=dict(size=11, color=COLOR_ERROR),
                    hoverinfo="skip", showlegend=False,
                )
            )

    # Status annotation for non-localized
    if focal.status == "abstained":
        fig.add_annotation(
            text=(
                "No 3D location emitted.<br>"
                "Rib level predicted, but along-rib location is unavailable<br>"
                "because correspondence abstained."
            ),
            xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False,
            align="left", font=dict(size=11, color="#555"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
        )
    elif focal.status in ("unlinked", "ap_only"):
        fig.add_annotation(
            text=(
                "2D fracture finding retained.<br>"
                "No reliable biplanar 3D localization available."
            ),
            xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False,
            align="left", font=dict(size=11, color="#555"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
        )

    cam = camera or CAMERA_PRESETS["Reset"]
    fig.update_layout(
        title=dict(text="<br>".join(_title_for(focal)), x=0.02, xanchor="left", font=dict(size=13)),
        height=height,
        scene=dict(
            xaxis_title="L ← LR → R",
            yaxis_title="Posterior ← AP → Anterior",
            zaxis_title="Inferior ← SI → Superior",
            aspectmode="data",
            dragmode="turntable",
            camera=cam,
            bgcolor="#FAFAFA",
        ),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=88, b=0),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
    )
    return fig
