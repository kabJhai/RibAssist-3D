# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Anatomy bounds, coordinate validation, and scene setup for 3D panel."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from demo_app.config import CL_DIR, SEG_DIR
from demo_app.data_loader import load_case_anatomy, sha256_file
from demo_app.rib_meshes import CaseMeshes, RibMesh, build_case_meshes

logger = logging.getLogger(__name__)

PADDING_FRAC = 0.08


@dataclass
class AnatomyBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_points(cls, pts: np.ndarray) -> AnatomyBounds:
        if pts is None or len(pts) == 0:
            raise ValueError("cannot compute bounds from empty points")
        pts = np.asarray(pts, dtype=np.float64)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        return cls(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))

    def padded(self, frac: float = PADDING_FRAC) -> AnatomyBounds:
        dx = (self.xmax - self.xmin) * frac
        dy = (self.ymax - self.ymin) * frac
        dz = (self.zmax - self.zmin) * frac
        return AnatomyBounds(
            self.xmin - dx, self.xmax + dx,
            self.ymin - dy, self.ymax + dy,
            self.zmin - dz, self.zmax + dz,
        )

    def contains(self, pt: np.ndarray, margin_mm: float = 40.0) -> bool:
        p = np.asarray(pt, dtype=np.float64)
        return (
            self.xmin - margin_mm <= p[0] <= self.xmax + margin_mm
            and self.ymin - margin_mm <= p[1] <= self.ymax + margin_mm
            and self.zmin - margin_mm <= p[2] <= self.zmax + margin_mm
        )

    def overlaps(self, other: AnatomyBounds) -> bool:
        return not (
            self.xmax < other.xmin or self.xmin > other.xmax
            or self.ymax < other.ymin or self.ymin > other.ymax
            or self.zmax < other.zmin or self.zmin > other.zmax
        )

    def as_plotly_ranges(self) -> dict[str, list[float]]:
        p = self.padded()
        return {"xaxis": [p.xmin, p.xmax], "yaxis": [p.ymin, p.ymax], "zaxis": [p.zmin, p.zmax]}


@dataclass
class AnatomyBundle:
    case_id: str
    anatomy: dict
    meshes: CaseMeshes | None
    bounds: AnatomyBounds
    centerline_world: list[np.ndarray]
    mesh_labels: set[int]
    centerline_fallback_labels: set[int]
    seg_path: str
    seg_sha256: str
    affine: np.ndarray
    warnings: list[str] = field(default_factory=list)


def _seg_path_for_case(case_id: str) -> Path | None:
    for pat in (f"{case_id}-rib-seg.nii.gz", f"{case_id}.nii.gz"):
        p = SEG_DIR / pat
        if p.exists():
            return p
    hits = list(SEG_DIR.glob(f"{case_id}*rib-seg*.nii.gz"))
    return hits[0] if hits else None


def _centerline_points(anatomy) -> np.ndarray:
    cls = [np.asarray(c, dtype=np.float64) for c in anatomy.get("cl_world", []) if c is not None and len(c)]
    if not cls:
        raise ValueError("no centerlines")
    return np.vstack(cls)


def _mesh_points(meshes: CaseMeshes) -> np.ndarray:
    return np.vstack([m.vertices for m in meshes.ribs])


def nearest_rib_label(bundle: AnatomyBundle, point: np.ndarray) -> tuple[int | None, float]:
    """Return rib label with closest mesh surface to `point` (mm)."""
    if not bundle.meshes or not bundle.meshes.ribs:
        return None, float("inf")
    p = np.asarray(point, dtype=np.float64)
    best_lb, best_d = None, float("inf")
    for rib in bundle.meshes.ribs:
        d = float(np.linalg.norm(rib.vertices - p[None], axis=1).min())
        if d < best_d:
            best_lb, best_d = rib.label, d
    return best_lb, best_d


def snap_to_rib_surface(
    bundle: AnatomyBundle,
    point: np.ndarray,
    *,
    max_mm: float = 35.0,
    prefer_label: int | None = None,
) -> tuple[np.ndarray, int | None, float]:
    """Return display point on rib mesh (nearest vertex), optionally preferring one label."""
    p = np.asarray(point, dtype=np.float64)
    if not bundle.meshes or not bundle.meshes.ribs:
        return p, None, float("inf")

    def _nearest_on_rib(rib) -> tuple[np.ndarray, float]:
        d = np.linalg.norm(rib.vertices - p[None], axis=1)
        i = int(d.argmin())
        return rib.vertices[i], float(d[i])

    if prefer_label is not None:
        rib = next((r for r in bundle.meshes.ribs if r.label == prefer_label), None)
        if rib is not None:
            pt, dist = _nearest_on_rib(rib)
            if dist <= max_mm:
                return pt, prefer_label, dist

    lb, dist = nearest_rib_label(bundle, p)
    if lb is None:
        return p, None, float("inf")
    rib = next(r for r in bundle.meshes.ribs if r.label == lb)
    pt, _ = _nearest_on_rib(rib)
    if dist <= max_mm:
        return pt, lb, dist
    return p, lb, dist


def validate_coordinates(
    bundle: AnatomyBundle,
    fracture_points: list[np.ndarray],
    *,
    margin_mm: float = 40.0,
) -> None:
    """Fail closed if fracture points fall outside anatomy or mesh misses centerlines."""
    cl_bounds = AnatomyBounds.from_points(_centerline_points(bundle.anatomy))
    if bundle.meshes and bundle.meshes.ribs:
        mesh_bounds = AnatomyBounds.from_points(_mesh_points(bundle.meshes))
        if not mesh_bounds.overlaps(cl_bounds):
            raise RuntimeError(
                f"{bundle.case_id}: transformed rib-mesh bounds do not overlap centerline bounds "
                f"(mesh x=[{mesh_bounds.xmin:.1f},{mesh_bounds.xmax:.1f}] "
                f"cl x=[{cl_bounds.xmin:.1f},{cl_bounds.xmax:.1f}])."
            )
    for i, pt in enumerate(fracture_points):
        if pt is None:
            continue
        if not bundle.bounds.contains(pt, margin_mm=margin_mm):
            raise RuntimeError(
                f"{bundle.case_id}: fracture point {i} {np.asarray(pt).round(1).tolist()} "
                f"outside anatomy bounds (margin {margin_mm} mm)."
            )


def build_anatomy_bundle(case_id: str, *, downsample_step: int = 2) -> AnatomyBundle | None:
    anatomy = load_case_anatomy(case_id)
    if anatomy is None:
        return None

    segp = _seg_path_for_case(case_id)
    seg_sha = sha256_file(segp) if segp else "missing"
    warnings: list[str] = []

    meshes = build_case_meshes(case_id, downsample_step=downsample_step, anatomy=anatomy)
    mesh_labels = {m.label for m in meshes.ribs} if meshes else set()
    all_labels = {int(lb) for lb in anatomy.get("info", {}) if int(lb) > 0}
    cl_fallback = all_labels - mesh_labels

    if not mesh_labels:
        warnings.append("Rib surface generation failed; showing centerline fallback.")

    cl_pts = _centerline_points(anatomy)
    bounds = AnatomyBounds.from_points(cl_pts)
    if meshes and meshes.ribs:
        mesh_bounds = AnatomyBounds.from_points(_mesh_points(meshes))
        bounds = AnatomyBounds(
            min(bounds.xmin, mesh_bounds.xmin), max(bounds.xmax, mesh_bounds.xmax),
            min(bounds.ymin, mesh_bounds.ymin), max(bounds.ymax, mesh_bounds.ymax),
            min(bounds.zmin, mesh_bounds.zmin), max(bounds.zmax, mesh_bounds.zmax),
        )

    return AnatomyBundle(
        case_id=case_id,
        anatomy=anatomy,
        meshes=meshes,
        bounds=bounds,
        centerline_world=list(anatomy.get("cl_world", [])),
        mesh_labels=mesh_labels,
        centerline_fallback_labels=cl_fallback,
        seg_path=str(segp) if segp else "",
        seg_sha256=seg_sha,
        affine=np.asarray(anatomy["aff"], dtype=np.float64),
        warnings=warnings,
    )


def log_figure_diagnostics(
    fig,
    *,
    case_id: str,
    bundle: AnatomyBundle | None,
    fracture_points: list[np.ndarray] | None = None,
    camera: dict | None = None,
) -> None:
    """Log trace and scene diagnostics (call when debug enabled)."""
    logger.info("=== 3D figure diagnostics case=%s ===", case_id)
    if bundle:
        b = bundle.bounds
        logger.info(
            "seg=%s sha=%s.. labels=%s mesh_count=%d cl_fallback=%s",
            bundle.seg_path, bundle.seg_sha256[:12], sorted(bundle.anatomy.get("info", {}).keys()),
            len(bundle.meshes.ribs) if bundle.meshes else 0,
            sorted(bundle.centerline_fallback_labels),
        )
        logger.info(
            "anatomy bounds x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]",
            b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax,
        )
        logger.info("affine diag=%s", np.diag(bundle.affine[:3, :3]).round(3).tolist())
    if fracture_points:
        pts = [p for p in fracture_points if p is not None]
        if pts:
            P = np.vstack(pts)
            logger.info(
                "fracture bounds x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]",
                P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), P[:, 2].min(), P[:, 2].max(),
            )
    scene = fig.layout.scene
    logger.info("camera=%s", camera or (scene.camera.to_plotly_json() if scene and scene.camera else None))
    if scene:
        for ax in ("xaxis", "yaxis", "zaxis"):
            a = getattr(scene, ax, None)
            if a and a.range:
                logger.info("scene.%s.range=%s", ax, a.range)

    for i, t in enumerate(fig.data):
        name = t.name
        ttype = t.type
        vis = getattr(t, "visible", True)
        op = getattr(t, "opacity", None)
        col = getattr(t, "color", None)
        if col is None:
            marker = getattr(t, "marker", None)
            col = getattr(marker, "color", None) if marker is not None else None
        leg = getattr(t, "showlegend", None)
        if ttype == "mesh3d" and t.x is not None:
            vx = np.column_stack([t.x, t.y, t.z])
            logger.info(
                "trace[%d] name=%r type=%s visible=%s verts=%d faces=%d "
                "opacity=%s color=%s legend=%s x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]",
                i, name, ttype, vis, len(vx), len(t.i) if t.i is not None else 0,
                op, col, leg,
                vx[:, 0].min(), vx[:, 0].max(), vx[:, 1].min(), vx[:, 1].max(), vx[:, 2].min(), vx[:, 2].max(),
            )
        elif t.x is not None:
            xs, ys, zs = np.asarray(t.x), np.asarray(t.y), np.asarray(t.z)
            logger.info(
                "trace[%d] name=%r type=%s visible=%s n=%d opacity=%s color=%s legend=%s "
                "x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]",
                i, name, ttype, vis, len(xs), op, col, leg,
                xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max(),
            )


def cache_key(case_id: str, bundle: AnatomyBundle, step: int) -> str:
    return f"{case_id}|{bundle.seg_sha256}|{step}|{hashlib.sha256(bundle.affine.tobytes()).hexdigest()[:16]}"
