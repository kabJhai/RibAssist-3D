# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Link original addressing findings to L2 AP correspondence candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class L2ApCandidate:
    index: int
    row: float
    col: float
    score: float


@dataclass(frozen=True)
class LinkResult:
    finding_index: int
    ap_xy: tuple[float, float] | None
    l2_ap_index: int | None
    link_distance_px: float | None
    status: str  # linked | unlinked | no_ap_peak


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def link_findings_to_l2_ap(
    findings_ap_xy: list[tuple[float, float] | None],
    l2_candidates: Iterable[L2ApCandidate],
    tolerance_px: float,
) -> list[LinkResult]:
    """Deterministic greedy assignment: sort by distance, one L2 AP per finding."""
    cands = list(l2_candidates)
    used_l2: set[int] = set()
    results: list[LinkResult] = []

    # Precompute all feasible pairs (finding_idx, l2_idx, dist)
    pairs: list[tuple[float, int, int]] = []
    for fi, ap_xy in enumerate(findings_ap_xy):
        if ap_xy is None:
            continue
        for c in cands:
            d = _dist(ap_xy, (c.row, c.col))
            if d <= tolerance_px:
                pairs.append((d, fi, c.index))

    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    assigned_finding: set[int] = set()

    for d, fi, l2_idx in pairs:
        if fi in assigned_finding or l2_idx in used_l2:
            continue
        assigned_finding.add(fi)
        used_l2.add(l2_idx)
        ap_xy = findings_ap_xy[fi]
        results.append(
            LinkResult(fi, ap_xy, l2_idx, d, "linked")
        )

    linked_by_fi = {r.finding_index: r for r in results}
    out: list[LinkResult] = []
    for fi, ap_xy in enumerate(findings_ap_xy):
        if ap_xy is None:
            out.append(LinkResult(fi, None, None, None, "no_ap_peak"))
        elif fi in linked_by_fi:
            out.append(linked_by_fi[fi])
        else:
            out.append(LinkResult(fi, ap_xy, None, None, "unlinked"))
    return out
