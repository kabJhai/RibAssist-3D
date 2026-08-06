# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Build 3D case-map entries from inference results."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from demo_app.correspondence_runtime import triangulate_edge, world_point
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding

MapKind = Literal["localized", "candidate", "rib_only"]


@dataclass
class MapFinding:
    finding_id: int
    kind: MapKind
    rib: str
    rib_label_id: int | None
    point_world: np.ndarray | None = None
    candidate_world: np.ndarray | None = None
    review_status: str = "pending"
    rejected: bool = False


def _rib_label(ef: EnrichedFinding) -> str:
    o = ef.original
    if o.side and o.rib is not None:
        return f"{o.side}{int(o.rib)}"
    return "-"


def _rib_label_id(anatomy, ef: EnrichedFinding) -> int | None:
    if anatomy is None:
        return None
    o = ef.original
    if not (o.side and o.rib is not None):
        return None
    for lb, meta in anatomy.get("info", {}).items():
        if meta.get("side") == o.side and int(meta.get("num", -1)) == int(o.rib):
            return int(lb)
    return None


def _candidate_world(ef: EnrichedFinding, result: CaseInferenceResult) -> np.ndarray | None:
    if ef.l2_ap_index is None or result.anatomy is None:
        return None
    committed = set(result.l2.committed_rows)
    for edge in result.l2.edges:
        if edge.ap_idx == ef.l2_ap_index and edge.global_row not in committed:
            p_rec, _ = triangulate_edge(edge, result.case.ap_geo, result.case.lat_geo)
            return world_point(p_rec, result.anatomy)
    return None


def classify_display_kind(ef: EnrichedFinding) -> MapKind:
    if ef.localization_status == "Localized":
        return "localized"
    if ef.localization_status == "Abstained":
        return "candidate"
    return "rib_only"


def display_status_label(kind: MapKind) -> str:
    if kind == "localized":
        return "Localized"
    if kind == "candidate":
        return "Candidate"
    return "Rib-level only"


def build_map_findings(
    result: CaseInferenceResult,
    review: dict[int, str],
    *,
    show_rejected: bool = False,
    anatomy=None,
) -> list[MapFinding]:
    anatomy = anatomy or result.anatomy
    out: list[MapFinding] = []
    for ef in result.findings:
        fid = ef.original.finding_id
        rv = review.get(fid, "pending")
        rejected = rv == "rejected"
        if rejected and not show_rejected:
            continue
        kind = classify_display_kind(ef)
        entry = MapFinding(
            finding_id=fid,
            kind=kind,
            rib=_rib_label(ef),
            rib_label_id=_rib_label_id(result.anatomy, ef),
            review_status=rv,
            rejected=rejected,
        )
        if kind == "localized":
            entry.point_world = ef.point_world
        elif kind == "candidate":
            entry.candidate_world = _candidate_world(ef, result)
            if entry.candidate_world is None:
                entry.kind = "rib_only"
        out.append(entry)
    return out


def confidence_dots(ef: EnrichedFinding) -> str:
    score = ef.original.address_score or ef.original.detection_confidence
    if score >= 0.55:
        return "●●●"
    if score >= 0.35:
        return "●●○"
    return "●○○"


def confidence_category(ef: EnrichedFinding) -> str:
    score = ef.original.address_score or ef.original.detection_confidence
    if score >= 0.55:
        return "High"
    if score >= 0.35:
        return "Moderate"
    return "Low"


def status_indicator(kind: MapKind) -> str:
    if kind == "localized":
        return "● Localized"
    if kind == "candidate":
        return "○ Candidate"
    return "▬ Rib-level only"


def review_state_label(status: str) -> str:
    return {
        "pending": "Pending",
        "accepted": "Accepted",
        "needs_review": "Needs review",
        "rejected": "Rejected",
    }.get(status, status.replace("_", " ").title())


def rib_label_id(anatomy, ef: EnrichedFinding) -> int | None:
    return _rib_label_id(anatomy, ef)


def _rib_str_from_label(anatomy, label_id: int | None) -> str | None:
    if anatomy is None or label_id is None:
        return None
    meta = anatomy.get("info", {}).get(label_id, {})
    side, num = meta.get("side"), meta.get("num")
    if side and num is not None:
        return f"{side}{int(num)}"
    return None


def map_finding_display_rib(bundle, mf: MapFinding) -> str | None:
    """Rib label for filters; uses the rib where the 3D point lands."""
    from demo_app.anatomy_scene import snap_to_rib_surface

    anatomy = bundle.anatomy if bundle is not None else None
    point = mf.point_world if mf.point_world is not None else mf.candidate_world
    if point is not None and bundle is not None:
        _, label_id, _ = snap_to_rib_surface(bundle, point, prefer_label=mf.rib_label_id)
        rib = _rib_str_from_label(anatomy, label_id)
        if rib:
            return rib
    if mf.rib and mf.rib != "-":
        return mf.rib
    return _rib_str_from_label(anatomy, mf.rib_label_id)


def build_rib_groups(bundle, map_findings: list[MapFinding]) -> dict[str, int]:
    """One navigator entry per rib that has map content."""
    ctr: Counter[str] = Counter()
    for mf in map_findings:
        rib = map_finding_display_rib(bundle, mf)
        if rib:
            ctr[rib] += 1
    return dict(sorted(ctr.items(), key=lambda x: (-x[1], x[0])))


def findings_on_display_rib(bundle, map_findings: list[MapFinding], rib: str) -> list[MapFinding]:
    return [mf for mf in map_findings if map_finding_display_rib(bundle, mf) == rib]


def build_rib_focus_narrative(
    bundle,
    map_findings: list[MapFinding],
    rib: str,
    result: CaseInferenceResult,
) -> str:
    """Plain-language summary of model findings visible on a selected rib."""
    on_rib = findings_on_display_rib(bundle, map_findings, rib)
    if not on_rib:
        return f"No findings are mapped to **{rib}** on the 3D view."

    ef_by_id = {f.original.finding_id: f for f in result.findings}
    localized = [mf for mf in on_rib if mf.kind == "localized"]
    candidates = [mf for mf in on_rib if mf.kind == "candidate"]
    rib_only = [mf for mf in on_rib if mf.kind == "rib_only"]
    n = len(on_rib)

    lead = f"**{rib}** shows {n} suspected fracture finding{'s' if n != 1 else ''} on the 3D map."
    detail: list[str] = []

    if localized:
        conf_parts: list[str] = []
        for mf in localized:
            ef = ef_by_id.get(mf.finding_id)
            conf = confidence_category(ef).lower() if ef else "unknown"
            conf_parts.append(f"#{mf.finding_id} ({conf} confidence)")
        detail.append(
            f"The model emitted committed 3D localizations for "
            f"{len(localized)} finding{'s' if len(localized) != 1 else ''}: "
            + ", ".join(conf_parts) + "."
        )

    if candidates:
        detail.append(
            f"{len(candidates)} finding{'s' if len(candidates) != 1 else ''} "
            f"{'have' if len(candidates) != 1 else 'has'} compatible AP–lateral pairs, "
            "but correspondence did not clear the frozen threshold. "
            "Candidate locations are shown in amber for manual review."
        )

    if rib_only:
        detail.append(
            f"{len(rib_only)} rib-level finding{'s' if len(rib_only) != 1 else ''} "
            "retained rib assignment without a reliable 3D point. Inspect AP and lateral projections."
        )

    pending = sum(1 for mf in on_rib if mf.review_status == "pending")
    if pending == n:
        detail.append("All findings on this rib are pending clinician review.")
    elif pending:
        detail.append(f"{pending} of {n} still pending clinician review.")

    ids = ", ".join(f"#{mf.finding_id}" for mf in on_rib)
    detail.append(f"Finding IDs: {ids}.")

    return lead + " " + " ".join(detail)
