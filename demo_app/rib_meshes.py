# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Per-rib anatomical meshes from segmentation (marching cubes)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nibabel.affines import apply_affine

from demo_app.data_loader import load_case_anatomy


@dataclass(frozen=True)
class RibMesh:
    label: int
    side: str
    num: int
    vertices: np.ndarray  # (N, 3) world mm
    faces: np.ndarray  # (M, 3) int


@dataclass(frozen=True)
class CaseMeshes:
    case_id: str
    ribs: tuple[RibMesh, ...]


def _downsample_mask(mask: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return mask
    return mask[::step, ::step, ::step]


def voxel_indices_to_world(verts: np.ndarray, aff: np.ndarray, downsample_step: int) -> np.ndarray:
    """Convert marching-cubes vertex indices to world mm.

    skimage.measure.marching_cubes returns vertices in array-index order
    (axis0, axis1, axis2) matching rs[i,j,k] voxel indices. nibabel.apply_affine
    expects the same (i,j,k) ordering for the canonical NIfTI affine (no axis swap).
    """
    # Continuous index coords -> original voxel grid indices before downsampling
    vox = verts * float(downsample_step)
    return apply_affine(aff, vox.astype(np.float64))


def _mesh_from_label(
    rs: np.ndarray, aff: np.ndarray, label: int, step: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None
    mask = rs == label
    if mask.sum() < 24:
        return None
    try:
        # step_size subsamples the volume; vertices stay in full voxel index space
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.float32), level=0.5, step_size=max(1, step),
        )
    except (ValueError, RuntimeError):
        return None
    if len(verts) < 3 or len(faces) < 1:
        return None
    world = apply_affine(aff, verts.astype(np.float64))
    faces = faces.astype(np.int32)
    faces = faces[:, [0, 2, 1]]
    return world, faces


def build_case_meshes(
    case_id: str, *, downsample_step: int = 2, anatomy=None,
) -> CaseMeshes | None:
    anatomy = anatomy or load_case_anatomy(case_id)
    if anatomy is None:
        return None
    rs = anatomy["rs"]
    aff = anatomy["aff"]
    info = anatomy.get("info", {})
    ribs: list[RibMesh] = []
    for lb in sorted(info.keys()):
        if lb <= 0:
            continue
        out = _mesh_from_label(rs, aff, int(lb), downsample_step)
        if out is None and downsample_step > 1:
            out = _mesh_from_label(rs, aff, int(lb), 1)
        if out is None:
            continue
        verts, faces = out
        meta = info[lb]
        ribs.append(
            RibMesh(
                label=int(lb),
                side=str(meta.get("side", "?")),
                num=int(meta.get("num", lb)),
                vertices=verts,
                faces=faces,
            )
        )
    return CaseMeshes(case_id=case_id, ribs=tuple(ribs)) if ribs else None


def gt_fracture_points(anatomy, iid: int, *, max_pts: int = 400) -> np.ndarray | None:
    if anatomy is None:
        return None
    vx = anatomy.get("fl_groups", {}).get(int(iid))
    if vx is None or vx.shape[1] == 0:
        return None
    pts = apply_affine(anatomy["aff"], vx.T.astype(np.float64))
    if len(pts) > max_pts:
        rng = np.random.RandomState(int(iid))
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts


def nearest_gt_iid(anatomy, point_world: np.ndarray) -> tuple[int | None, float | None, np.ndarray | None]:
    if anatomy is None or point_world is None:
        return None, None, None
    best_d, best_iid, best_pt = 1e9, None, None
    for iid, vx in anatomy.get("fl_groups", {}).items():
        if vx.shape[1] == 0:
            continue
        w = apply_affine(anatomy["aff"], vx.T.astype(np.float64))
        idx = int(np.argmin(np.linalg.norm(w - point_world[None], axis=1)))
        d = float(np.linalg.norm(w[idx] - point_world))
        if d < best_d:
            best_d, best_iid, best_pt = d, int(iid), w[idx]
    return best_iid, best_d, best_pt
