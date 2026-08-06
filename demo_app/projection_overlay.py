# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Project rib anatomy and finding markers onto AP/lateral views."""
from __future__ import annotations

import numpy as np
from nibabel.affines import apply_affine

from demo_app.anatomy_3d import _centerline_for_label
from demo_app.anatomy_scene import AnatomyBundle, snap_to_rib_surface
from demo_app.finding_map import MapFinding, classify_display_kind
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding


def _forward_project_canonical(fc: np.ndarray, ap_geo: np.ndarray, lat_geo: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    sa, pta, pla = [float(x) for x in ap_geo]
    sl, ptl, pll = [float(x) for x in lat_geo]
    lr, apx, si = [float(x) for x in fc]
    ap = (si * sa + pta, lr * sa + pla)
    lat = (si * sl + ptl, apx * sl + pll)
    return ap, lat


def projection_rc_from_world(
    point_world: np.ndarray,
    anatomy,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Project a world-space point to AP/lateral (row, col) using case geometry."""
    if anatomy is None:
        raise ValueError("anatomy is required")
    aff = anatomy.get("aff")
    if aff is None:
        raise ValueError("anatomy affine is required")
    inv = np.linalg.inv(np.asarray(aff, dtype=np.float64))
    fc = apply_affine(inv, np.asarray(point_world, dtype=np.float64))
    return _forward_project_canonical(fc, ap_geo, lat_geo)


def project_display_point(
    bundle: AnatomyBundle,
    point_world: np.ndarray,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
    *,
    prefer_label: int | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Match 3D review display: snap to rib surface, then forward-project."""
    disp_p, _, _ = snap_to_rib_surface(bundle, point_world, prefer_label=prefer_label)
    return projection_rc_from_world(disp_p, bundle.anatomy, ap_geo, lat_geo)


def review_finding_projection_coords(
    bundle: AnatomyBundle,
    ef: EnrichedFinding,
    mf: MapFinding | None,
    result: CaseInferenceResult,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
    tuple[float, float] | None,
]:
    """
    2D marker coords aligned with the 3D review display point.

    Returns (sel_ap, sel_lat, sel_comm_ap, sel_comm_lat, sel_abst_ap).
    """
    kind = classify_display_kind(ef)
    ap_geo = result.case.ap_geo
    lat_geo = result.case.lat_geo
    prefer = mf.rib_label_id if mf else None

    if kind == "localized" and ef.point_world is not None and bundle.anatomy is not None:
        ap_rc, lat_rc = project_display_point(
            bundle, ef.point_world, ap_geo, lat_geo, prefer_label=prefer,
        )
        return ap_rc, lat_rc, None, None, None

    if kind == "candidate" and mf is not None and mf.candidate_world is not None and bundle.anatomy is not None:
        ap_rc, lat_rc = project_display_point(
            bundle, mf.candidate_world, ap_geo, lat_geo, prefer_label=prefer,
        )
        sel_abst_ap = None
        if ef.l2_ap_index is not None:
            for edge in result.l2.edges:
                if edge.ap_idx == ef.l2_ap_index:
                    sel_abst_ap = (edge.ap_row, edge.ap_col)
                    break
        return ap_rc, lat_rc, None, None, sel_abst_ap

    o = ef.original
    sel_ap = tuple(o.ap_xy) if o.ap_xy else None
    sel_lat = None
    sel_comm_ap = sel_comm_lat = None
    sel_abst_ap = None
    if ef.committed_edge:
        e = ef.committed_edge
        sel_comm_ap = (e.ap_row, e.ap_col)
        sel_comm_lat = (e.lat_row, e.lat_col)
        sel_lat = sel_comm_lat
    elif ef.lat_xy_committed:
        sel_lat = tuple(ef.lat_xy_committed)
    elif o.lat_xy:
        sel_lat = tuple(o.lat_xy)
    if kind == "candidate" and ef.l2_ap_index is not None:
        for edge in result.l2.edges:
            if edge.ap_idx == ef.l2_ap_index:
                sel_abst_ap = (edge.ap_row, edge.ap_col)
                if sel_lat is None:
                    sel_lat = (edge.lat_row, edge.lat_col)
                break
    return sel_ap, sel_lat, sel_comm_ap, sel_comm_lat, sel_abst_ap


def centerline_projection_paths(
    anatomy,
    rib_label_id: int | None,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return Nx2 (row, col) arrays for AP and lateral rib centerline paths."""
    if anatomy is None or rib_label_id is None:
        return None, None
    cl = _centerline_for_label(anatomy, rib_label_id)
    if cl is None or len(cl) < 2:
        return None, None
    aff = anatomy.get("aff")
    if aff is None:
        return None, None
    inv = np.linalg.inv(np.asarray(aff, dtype=np.float64))
    from nibabel.affines import apply_affine

    ap_pts: list[tuple[float, float]] = []
    lat_pts: list[tuple[float, float]] = []
    for p in cl:
        fc = apply_affine(inv, np.asarray(p, dtype=np.float64))
        ap_rc, lat_rc = _forward_project_canonical(fc, ap_geo, lat_geo)
        ap_pts.append(ap_rc)
        lat_pts.append(lat_rc)
    return np.asarray(ap_pts, dtype=np.float64), np.asarray(lat_pts, dtype=np.float64)


def _finding_projection_coords(ef: EnrichedFinding, result: CaseInferenceResult) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    o = ef.original
    ap = tuple(o.ap_xy) if o.ap_xy else None
    lat = None
    if ef.committed_edge:
        e = ef.committed_edge
        lat = (e.lat_row, e.lat_col)
    elif ef.lat_xy_committed:
        lat = tuple(ef.lat_xy_committed)
    elif o.lat_xy:
        lat = tuple(o.lat_xy)
    if classify_display_kind(ef) == "candidate" and ef.l2_ap_index is not None:
        for edge in result.l2.edges:
            if edge.ap_idx == ef.l2_ap_index:
                lat = (edge.lat_row, edge.lat_col)
                break
    return ap, lat


def context_finding_markers(
    result: CaseInferenceResult,
    map_findings: list[MapFinding],
    selected_id: int,
) -> tuple[list[dict], list[dict]]:
    """Other findings for projection context overlays."""
    ef_by_id = {f.original.finding_id: f for f in result.findings}
    ap_markers: list[dict] = []
    lat_markers: list[dict] = []
    for mf in map_findings:
        if mf.finding_id == selected_id:
            continue
        ef = ef_by_id.get(mf.finding_id)
        if ef is None:
            continue
        ap, lat = _finding_projection_coords(ef, result)
        if ap is None:
            continue
        entry = {"row": ap[0], "col": ap[1], "kind": mf.kind}
        ap_markers.append(entry)
        if lat is not None:
            lat_markers.append({"row": lat[0], "col": lat[1], "kind": mf.kind})
    return ap_markers, lat_markers
