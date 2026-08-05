#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Shared per-case rib-labeling convention, so the addressing dataset and the reconstruction atlas
agree on what "R7" means. RAS-anchored, identical convention to build_rib_atlas.py:
  * side: +LR axis = patient Right; a rib is Right if its centroid LR >= the rib-cage LR midline (spine);
  * rib number 1..12: superior -> inferior (descending +SI axis) within each side, capped at 12.

Callers pass the axis indices for their frame:
  * canonical RAS+ volume (as_closest_canonical): lr_axis=0, si_axis=2  (det-frame addressing builder);
  * argmax world axes (extract_crops.py legacy frame): its own lr/si axis indices.
This is the single source of truth for the per-case (side, number) mapping."""
from __future__ import annotations
import numpy as np


def assign_side_num(lr, si, spine_lr, keys=None):
    """THE convention primitive (single source of truth). Given per-item LR and SI centroid
    coordinates and the spine LR midline: side = 'R' if lr >= spine_lr else 'L'; num = 1..12
    superior->inferior (descending SI) within each side, capped at 12. Both the addressing dataset
    (per-case rib-seg) and the rib atlas (per-case normalized cages) call this so 'R7' means the same
    rib in both. Returns a list of (side, num) in input order, or a dict keyed by `keys` if given."""
    lr = np.asarray(lr, float); si = np.asarray(si, float); n = len(lr)
    side = ["R" if lr[i] >= spine_lr else "L" for i in range(n)]
    num = [0] * n
    for sd in ("L", "R"):
        ids = [i for i in range(n) if side[i] == sd]
        for rank, i in enumerate(sorted(ids, key=lambda i: -si[i]), 1):
            num[i] = min(12, rank)
    if keys is not None: return {keys[i]: (side[i], num[i]) for i in range(n)}
    return list(zip(side, num))


def side_num_from_seg(rs, lr_axis=0, si_axis=2):
    """From a rib-segmentation label volume, return (info, spine_lr):
    info[label] = {"c": centroid(voxel,3), "side": "L"/"R", "num": 1..12}; spine_lr = LR midline.
    Uses assign_side_num for the (side, num) convention."""
    nz = np.array(np.nonzero(rs))
    spine_lr = float(nz[lr_axis].mean()) if nz.size else rs.shape[lr_axis] / 2.0
    labels = [int(v) for v in np.unique(rs) if v != 0]
    cent = {lb: np.array(np.nonzero(rs == lb)).mean(1) for lb in labels}
    sn = assign_side_num([cent[lb][lr_axis] for lb in labels], [cent[lb][si_axis] for lb in labels], spine_lr, keys=labels)
    return {lb: {"c": cent[lb], "side": sn[lb][0], "num": sn[lb][1]} for lb in labels}, spine_lr
