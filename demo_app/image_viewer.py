"""Plotly AP and lateral viewers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class CropBox:
    r0: int
    r1: int
    c0: int
    c1: int


def _normalize01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, np.float32)
    lo, hi = float(np.percentile(img, 0.5)), float(np.percentile(img, 99.5))
    if hi <= lo:
        hi = float(img.max()) or 1.0
        lo = float(img.min())
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _crop_box(img: np.ndarray, margin: int = 4) -> CropBox:
    norm = _normalize01(img)
    mask = norm > 0.04
    if not mask.any():
        h, w = img.shape
        return CropBox(0, h, 0, w)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    h, w = img.shape
    r0 = max(0, int(rows[0]) - margin)
    r1 = min(h, int(rows[-1]) + margin + 1)
    c0 = max(0, int(cols[0]) - margin)
    c1 = min(w, int(cols[-1]) + margin + 1)
    return CropBox(r0, r1, c0, c1)


def _apply_crop(img: np.ndarray, box: CropBox) -> np.ndarray:
    return img[box.r0 : box.r1, box.c0 : box.c1]


def _display_ap(img: np.ndarray) -> np.ndarray:
    return np.fliplr(_normalize01(img))


def _display_lat(img: np.ndarray) -> np.ndarray:
    return _normalize01(img)


def _rc_to_display(
    row: float,
    col: float,
    view: str,
    size: int,
    ap_box: CropBox | None = None,
    lat_box: CropBox | None = None,
) -> tuple[float, float]:
    if view == "ap":
        col = (size - 1) - col
        box = ap_box
    else:
        box = lat_box
    if box:
        row -= box.r0
        col -= box.c0
    return float(row), float(col)


def build_projection_figure(
    ap_img: np.ndarray,
    lat_img: np.ndarray,
    *,
    ap_heatmap: np.ndarray | None = None,
    lat_heatmap: np.ndarray | None = None,
    heatmap_opacity: float = 0.25,
    show_candidates: bool = True,
    l2_ap_peaks: np.ndarray | None = None,
    l2_lat_peaks: np.ndarray | None = None,
    selected_finding_ap: tuple[float, float] | None = None,
    selected_finding_lat: tuple[float, float] | None = None,
    selected_committed_ap: tuple[float, float] | None = None,
    selected_committed_lat: tuple[float, float] | None = None,
    selected_abstained_ap: tuple[float, float] | None = None,
    gt_footprint_ap: np.ndarray | None = None,
    gt_footprint_lat: np.ndarray | None = None,
    rib_highlight_ap: np.ndarray | None = None,
    rib_highlight_lat: np.ndarray | None = None,
    context_findings_ap: list[dict] | None = None,
    context_findings_lat: list[dict] | None = None,
    height: int = 780,
    lateral_context_only: bool = False,
    minimal_chrome: bool = False,
) -> go.Figure:
    ap_box = _crop_box(ap_img)
    lat_box = _crop_box(lat_img)
    ap_show = _apply_crop(_display_ap(ap_img), ap_box)
    lat_show = _apply_crop(_display_lat(lat_img), lat_box)
    S = ap_img.shape[-1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "AP projection (R | L)",
            "Lateral (context only)" if lateral_context_only else "Lateral projection",
        ),
        horizontal_spacing=0.02,
    )

    for col, z, name in ((1, ap_show, "AP"), (2, lat_show, "Lateral")):
        h, w = z.shape
        opacity = 0.45 if (lateral_context_only and col == 2) else 1.0
        fig.add_trace(
            go.Heatmap(
                z=z,
                colorscale=[[0, "rgb(8,8,12)"], [0.5, "rgb(120,120,130)"], [1, "rgb(245,245,250)"]],
                showscale=False,
                hoverinfo="skip",
                name=name,
                zsmooth=False,
                opacity=opacity,
            ),
            row=1,
            col=col,
        )
        # Row index increases toward superior (head); y grows bottom→top for upright anatomy.
        fig.update_xaxes(range=[-0.5, w - 0.5], row=1, col=col)
        fig.update_yaxes(range=[-0.5, h - 0.5], row=1, col=col)

    fig.add_annotation(
        x=0.04, y=0.96, xref="x domain", yref="y domain", text="R",
        showarrow=False, font=dict(color="#00BCD4", size=20, family="Arial Black"),
    )
    fig.add_annotation(
        x=0.96, y=0.96, xref="x domain", yref="y domain", text="L",
        showarrow=False, font=dict(color="#00BCD4", size=20, family="Arial Black"),
    )

    def _add_hm(raw_hm: np.ndarray, box: CropBox, col: int, flip: bool) -> None:
        hm = _normalize01(raw_hm)
        if flip:
            hm = np.fliplr(hm)
        hm = _apply_crop(hm, box)
        fig.add_trace(
            go.Heatmap(
                z=hm,
                colorscale="Hot",
                opacity=heatmap_opacity,
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    if ap_heatmap is not None:
        _add_hm(ap_heatmap, ap_box, 1, flip=True)
    if lat_heatmap is not None:
        _add_hm(lat_heatmap, lat_box, 2, flip=False)

    def _marker(
        x: float,
        y: float,
        *,
        col: int,
        color: str,
        size: int,
        symbol: str,
        opacity: float = 1.0,
        name: str,
        showlegend: bool = False,
        line_width: int = 2,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers", name=name, showlegend=showlegend,
                marker=dict(
                    color=color, size=size, symbol=symbol, opacity=opacity,
                    line=dict(width=line_width, color=color if symbol == "circle-open" else "white"),
                ),
            ),
            row=1,
            col=col,
        )

    def _path(
        pts: np.ndarray,
        *,
        view: str,
        col: int,
        color: str,
        width: int,
        opacity: float,
        name: str,
        showlegend: bool = False,
    ) -> None:
        xs, ys = [], []
        for r, c in pts:
            y, x = _rc_to_display(float(r), float(c), view, S, ap_box, lat_box)
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=name, showlegend=showlegend,
                line=dict(color=color, width=width),
                opacity=opacity,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    COLOR_RIB_HIGHLIGHT = "#8FA8DC"
    if rib_highlight_ap is not None and len(rib_highlight_ap) >= 2:
        _path(
            rib_highlight_ap, view="ap", col=1,
            color=COLOR_RIB_HIGHLIGHT, width=5, opacity=0.95,
            name="Selected rib", showlegend=not minimal_chrome,
        )
    if rib_highlight_lat is not None and len(rib_highlight_lat) >= 2 and not lateral_context_only:
        _path(
            rib_highlight_lat, view="lat", col=2,
            color=COLOR_RIB_HIGHLIGHT, width=5, opacity=0.95,
            name="Selected rib",
        )

    CONTEXT_LOCALIZED = "#FF7043"
    CONTEXT_CANDIDATE = "#FFB300"
    CONTEXT_RIB_ONLY = "#6B8FD4"

    if context_findings_ap:
        for i, entry in enumerate(context_findings_ap):
            kind = entry.get("kind", "localized")
            if kind == "candidate":
                color, opacity, size, symbol = CONTEXT_CANDIDATE, 0.22, 8, "circle-open"
            elif kind == "rib_only":
                color, opacity, size, symbol = CONTEXT_RIB_ONLY, 0.28, 9, "diamond"
            else:
                color, opacity, size, symbol = CONTEXT_LOCALIZED, 0.30, 9, "circle"
            r, c = entry["row"], entry["col"]
            y, x = _rc_to_display(r, c, "ap", S, ap_box, lat_box)
            _marker(
                x, y, col=1, color=color, size=size, symbol=symbol,
                opacity=opacity, name="Other findings",
                showlegend=(i == 0 and not minimal_chrome), line_width=2,
            )
    if context_findings_lat and not lateral_context_only:
        for entry in context_findings_lat:
            kind = entry.get("kind", "localized")
            if kind == "candidate":
                color, opacity, size, symbol = CONTEXT_CANDIDATE, 0.22, 8, "circle-open"
            elif kind == "rib_only":
                color, opacity, size, symbol = CONTEXT_RIB_ONLY, 0.28, 9, "diamond"
            else:
                color, opacity, size, symbol = CONTEXT_LOCALIZED, 0.30, 9, "circle"
            r, c = entry["row"], entry["col"]
            y, x = _rc_to_display(r, c, "lat", S, ap_box, lat_box)
            _marker(
                x, y, col=2, color=color, size=size, symbol=symbol,
                opacity=opacity, name="Other findings", line_width=2,
            )

    if show_candidates and l2_ap_peaks is not None:
        sel_ap_rc = selected_finding_ap
        for p in l2_ap_peaks:
            r, c = _rc_to_display(p[0], p[1], "ap", S, ap_box, lat_box)
            is_sel = sel_ap_rc and abs(p[0] - sel_ap_rc[0]) < 0.5 and abs(p[1] - sel_ap_rc[1]) < 0.5
            if is_sel:
                continue
            _marker(c, r, col=1, color="#FFB300", size=7, symbol="circle-open",
                    opacity=0.35, name="Candidates")
    if show_candidates and l2_lat_peaks is not None:
        sel_lat_rc = selected_finding_lat
        for p in l2_lat_peaks:
            r, c = _rc_to_display(p[0], p[1], "lat", S, ap_box, lat_box)
            is_sel = sel_lat_rc and abs(p[0] - sel_lat_rc[0]) < 0.5 and abs(p[1] - sel_lat_rc[1]) < 0.5
            if is_sel:
                continue
            _marker(c, r, col=2, color="#FFB300", size=7, symbol="circle-open", opacity=0.35, name="Candidates")

    if selected_committed_ap:
        ar, ac = _rc_to_display(selected_committed_ap[0], selected_committed_ap[1], "ap", S, ap_box, lat_box)
        _marker(ac, ar, col=1, color="#00C853", size=14, symbol="circle-open", name="Accepted pair", showlegend=True)
    if selected_committed_lat:
        lr, lc = _rc_to_display(selected_committed_lat[0], selected_committed_lat[1], "lat", S, ap_box, lat_box)
        _marker(lc, lr, col=2, color="#00C853", size=14, symbol="circle-open", name="Accepted pair")

    if selected_abstained_ap:
        ar, ac = _rc_to_display(selected_abstained_ap[0], selected_abstained_ap[1], "ap", S, ap_box, lat_box)
        _marker(ac, ar, col=1, color="#FF1744", size=16, symbol="x", name="Abstained", showlegend=True)

    if selected_finding_ap:
        ar, ac = _rc_to_display(selected_finding_ap[0], selected_finding_ap[1], "ap", S, ap_box, lat_box)
        _marker(ac, ar, col=1, color="#FF5722", size=20, symbol="circle", name="Selected finding", showlegend=not minimal_chrome, line_width=3)
    if selected_finding_lat and not lateral_context_only:
        lr, lc = _rc_to_display(selected_finding_lat[0], selected_finding_lat[1], "lat", S, ap_box, lat_box)
        _marker(lc, lr, col=2, color="#FF5722", size=20, symbol="circle", name="Selected finding")

    if gt_footprint_ap is not None and len(gt_footprint_ap):
        xs, ys = [], []
        for r, c in zip(gt_footprint_ap[:, 0], gt_footprint_ap[:, 1]):
            y, x = _rc_to_display(r, c, "ap", S, ap_box, lat_box)
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", name="GT (eval)",
                marker=dict(color="#E040FB", size=3, opacity=0.18, symbol="circle-open",
                            line=dict(width=1, color="#E040FB")),
            ),
            row=1, col=1,
        )
    if gt_footprint_lat is not None and len(gt_footprint_lat):
        xs, ys = [], []
        for r, c in gt_footprint_lat:
            y, x = _rc_to_display(r, c, "lat", S, ap_box, lat_box)
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", name="GT (eval)",
                marker=dict(color="#E040FB", size=3, opacity=0.18, symbol="circle-open",
                            line=dict(width=1, color="#E040FB")),
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(showticklabels=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(showticklabels=False, constrain="domain", row=1, col=2)
    fig.update_yaxes(showticklabels=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        height=height,
        margin=dict(l=2, r=2, t=24 if minimal_chrome else 36, b=2),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10),
        ) if not minimal_chrome else dict(orientation="h", y=-0.15, x=0, font=dict(size=9)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        dragmode="pan",
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def _auto_frame_ranges(
    h: int,
    w: int,
    points: list[tuple[float, float]],
    *,
    panel_aspect: float = 2.1,
    padding_frac: float = 0.20,
    min_span_frac: float = 0.10,
) -> tuple[list[float], list[float]]:
    """Frame all anchor points and expand to the panel aspect ratio."""
    if not points:
        return [-0.5, w - 0.5], [-0.5, h - 0.5]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, w * min_span_frac)
    span_y = max(max_y - min_y, h * min_span_frac)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    half_x = 0.5 * span_x * (1.0 + 2.0 * padding_frac)
    half_y = 0.5 * span_y * (1.0 + 2.0 * padding_frac)

    content_aspect = (2.0 * half_x) / max(2.0 * half_y, 1e-6)
    if content_aspect < panel_aspect:
        half_x = half_y * panel_aspect
    else:
        half_y = half_x / panel_aspect

    x0, x1 = cx - half_x, cx + half_x
    y0, y1 = cy - half_y, cy + half_y

    if x0 < -0.5:
        x1 -= x0 + 0.5
        x0 = -0.5
    if x1 > w - 0.5:
        x0 -= x1 - (w - 0.5)
        x1 = w - 0.5
    if y0 < -0.5:
        y1 -= y0 + 0.5
        y0 = -0.5
    if y1 > h - 0.5:
        y0 -= y1 - (h - 0.5)
        y1 = h - 0.5

    x0 = max(-0.5, x0)
    x1 = min(w - 0.5, x1)
    y0 = max(-0.5, y0)
    y1 = min(h - 0.5, y1)
    return [x0, x1], [y0, y1]


def _axis_ranges(
    h: int,
    w: int,
    focus_xy: tuple[float, float] | None,
    *,
    zoom_to_finding: bool,
    zoom_margin_frac: float = 0.16,
) -> tuple[list[float], list[float]]:
    if not zoom_to_finding or focus_xy is None:
        return [-0.5, w - 0.5], [-0.5, h - 0.5]
    return _auto_frame_ranges(h, w, [focus_xy], padding_frac=zoom_margin_frac, min_span_frac=zoom_margin_frac)


def build_single_projection_figure(
    img: np.ndarray,
    *,
    view: str,
    ap_img: np.ndarray | None = None,
    ap_heatmap: np.ndarray | None = None,
    lat_heatmap: np.ndarray | None = None,
    heatmap_opacity: float = 0.25,
    show_candidates: bool = True,
    l2_ap_peaks: np.ndarray | None = None,
    l2_lat_peaks: np.ndarray | None = None,
    selected_finding_ap: tuple[float, float] | None = None,
    selected_finding_lat: tuple[float, float] | None = None,
    selected_committed_ap: tuple[float, float] | None = None,
    selected_committed_lat: tuple[float, float] | None = None,
    selected_abstained_ap: tuple[float, float] | None = None,
    gt_footprint_ap: np.ndarray | None = None,
    gt_footprint_lat: np.ndarray | None = None,
    rib_highlight_ap: np.ndarray | None = None,
    rib_highlight_lat: np.ndarray | None = None,
    context_findings_ap: list[dict] | None = None,
    context_findings_lat: list[dict] | None = None,
    height: int = 220,
    lateral_context_only: bool = False,
    zoom_to_finding: bool = False,
    auto_scale: bool = True,
    panel_aspect: float = 2.1,
    title: str | None = None,
) -> go.Figure:
    """Single AP or lateral projection with pan/zoom and optional auto-framed finding view."""
    if view not in ("ap", "lat"):
        raise ValueError(f"view must be 'ap' or 'lat', got {view!r}")

    ap_ref = ap_img if ap_img is not None else img
    ap_box = _crop_box(ap_ref)
    lat_box = _crop_box(img if view == "lat" else ap_ref)
    S = ap_ref.shape[-1]

    if view == "ap":
        z = _apply_crop(_display_ap(img), ap_box)
        box = ap_box
        heatmap = ap_heatmap
        flip_hm = True
        peaks = l2_ap_peaks
        sel_finding = selected_finding_ap
        sel_committed = selected_committed_ap
        sel_abstained = selected_abstained_ap
        rib_path = rib_highlight_ap
        ctx_markers = context_findings_ap
        gt_fp = gt_footprint_ap
        default_title = "AP projection (R | L)"
    else:
        z = _apply_crop(_display_lat(img), lat_box)
        box = lat_box
        heatmap = lat_heatmap
        flip_hm = False
        peaks = l2_lat_peaks
        sel_finding = selected_finding_lat if not lateral_context_only else None
        sel_committed = selected_committed_lat if not lateral_context_only else None
        sel_abstained = None
        rib_path = rib_highlight_lat if not lateral_context_only else None
        ctx_markers = context_findings_lat if not lateral_context_only else None
        gt_fp = gt_footprint_lat if not lateral_context_only else None
        default_title = (
            "Lateral (context only)" if lateral_context_only else "Lateral projection"
        )

    h, w = z.shape
    opacity = 0.45 if lateral_context_only else 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=[[0, "rgb(8,8,12)"], [0.5, "rgb(120,120,130)"], [1, "rgb(245,245,250)"]],
            showscale=False,
            hoverinfo="skip",
            name="AP" if view == "ap" else "Lateral",
            zsmooth=False,
            opacity=opacity,
        )
    )

    if view == "ap":
        fig.add_annotation(
            x=0.04, y=0.96, xref="paper", yref="paper", text="R",
            showarrow=False, font=dict(color="#00BCD4", size=16, family="Arial Black"),
        )
        fig.add_annotation(
            x=0.96, y=0.96, xref="paper", yref="paper", text="L",
            showarrow=False, font=dict(color="#00BCD4", size=16, family="Arial Black"),
        )

    if heatmap is not None:
        hm = _normalize01(heatmap)
        if flip_hm:
            hm = np.fliplr(hm)
        hm = _apply_crop(hm, box)
        fig.add_trace(
            go.Heatmap(
                z=hm, colorscale="Hot", opacity=heatmap_opacity,
                showscale=False, hoverinfo="skip",
            )
        )

    def _marker(x, y, *, color, size, symbol, opacity_m=1.0, name, line_width=2):
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers", name=name, showlegend=False,
                marker=dict(
                    color=color, size=size, symbol=symbol, opacity=opacity_m,
                    line=dict(width=line_width, color=color if symbol == "circle-open" else "white"),
                ),
            )
        )

    def _path(pts, *, color, width, opacity_p, name):
        xs, ys = [], []
        for r, c in pts:
            y, x = _rc_to_display(float(r), float(c), view, S, ap_box, lat_box)
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", name=name, showlegend=False,
                line=dict(color=color, width=width), opacity=opacity_p, hoverinfo="skip",
            )
        )
        return list(zip(xs, ys))

    frame_points: list[tuple[float, float]] = []
    rib_path_xy: list[tuple[float, float]] = []

    COLOR_RIB_HIGHLIGHT = "#8FA8DC"
    if rib_path is not None and len(rib_path) >= 2:
        rib_path_xy = _path(
            rib_path, color=COLOR_RIB_HIGHLIGHT, width=5, opacity_p=0.95, name="Selected rib",
        )
        frame_points.extend(rib_path_xy)

    CONTEXT_LOCALIZED = "#FF7043"
    CONTEXT_CANDIDATE = "#FFB300"
    CONTEXT_RIB_ONLY = "#6B8FD4"
    if ctx_markers:
        for entry in ctx_markers:
            kind = entry.get("kind", "localized")
            if kind == "candidate":
                color, opacity_m, size, symbol = CONTEXT_CANDIDATE, 0.22, 8, "circle-open"
            elif kind == "rib_only":
                color, opacity_m, size, symbol = CONTEXT_RIB_ONLY, 0.28, 9, "diamond"
            else:
                color, opacity_m, size, symbol = CONTEXT_LOCALIZED, 0.30, 9, "circle"
            r, c = entry["row"], entry["col"]
            y, x = _rc_to_display(r, c, view, S, ap_box, lat_box)
            _marker(x, y, color=color, size=size, symbol=symbol, opacity_m=opacity_m, name="Other")

    if show_candidates and peaks is not None:
        sel_rc = sel_finding
        for p in peaks:
            r, c = _rc_to_display(p[0], p[1], view, S, ap_box, lat_box)
            is_sel = sel_rc and abs(p[0] - sel_rc[0]) < 0.5 and abs(p[1] - sel_rc[1]) < 0.5
            if is_sel:
                continue
            _marker(c, r, color="#FFB300", size=7, symbol="circle-open", opacity_m=0.35, name="Candidate")

    if sel_committed:
        ar, ac = _rc_to_display(sel_committed[0], sel_committed[1], view, S, ap_box, lat_box)
        _marker(ac, ar, color="#00C853", size=14, symbol="circle-open", name="Accepted pair")
        frame_points.append((ac, ar))
    if sel_abstained and view == "ap":
        ar, ac = _rc_to_display(sel_abstained[0], sel_abstained[1], view, S, ap_box, lat_box)
        _marker(ac, ar, color="#FF1744", size=16, symbol="x", name="Abstained")
        frame_points.append((ac, ar))
    if sel_finding:
        ar, ac = _rc_to_display(sel_finding[0], sel_finding[1], view, S, ap_box, lat_box)
        _marker(ac, ar, color="#FF5722", size=20, symbol="circle", name="Selected finding", line_width=3)
        frame_points.append((ac, ar))

    if gt_fp is not None and len(gt_fp):
        xs, ys = [], []
        rows = gt_fp[:, 0] if hasattr(gt_fp, "shape") and len(gt_fp.shape) > 1 else [p[0] for p in gt_fp]
        cols = gt_fp[:, 1] if hasattr(gt_fp, "shape") and len(gt_fp.shape) > 1 else [p[1] for p in gt_fp]
        for r, c in zip(rows, cols):
            y, x = _rc_to_display(float(r), float(c), view, S, ap_box, lat_box)
            xs.append(x)
            ys.append(y)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", name="GT (eval)", showlegend=False,
                marker=dict(color="#E040FB", size=3, opacity=0.18, symbol="circle-open",
                            line=dict(width=1, color="#E040FB")),
            )
        )

    use_auto = auto_scale and (zoom_to_finding or lateral_context_only or bool(frame_points))
    if use_auto and frame_points:
        x_range, y_range = _auto_frame_ranges(
            h, w, frame_points, panel_aspect=panel_aspect,
        )
    elif zoom_to_finding and frame_points:
        x_range, y_range = _axis_ranges(h, w, frame_points[0], zoom_to_finding=True)
    else:
        x_range, y_range = [-0.5, w - 0.5], [-0.5, h - 0.5]

    fig.update_xaxes(range=x_range, showticklabels=False, constrain="domain", fixedrange=False)
    fig.update_yaxes(
        range=y_range, showticklabels=False, scaleanchor="x", scaleratio=1, fixedrange=False,
    )
    fig.update_layout(
        height=height,
        title=dict(text=title or default_title, font=dict(size=11), x=0, xanchor="left"),
        margin=dict(l=2, r=2, t=22, b=2),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        dragmode="pan",
        showlegend=False,
    )
    return fig
