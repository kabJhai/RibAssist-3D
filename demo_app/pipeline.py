# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Orchestrate live per-case RibAssist 3D inference for the demo UI."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from demo_app.config import LINK_TOLERANCE_PX  # noqa: E402
from demo_app.correspondence_runtime import (  # noqa: E402
    L2CaseResult,
    L2Models,
    PairEdge,
    run_l2_case,
    triangulate_edge,
    world_point,
)
from demo_app.data_loader import CaseImages, load_case_anatomy  # noqa: E402
from demo_app.finding_linker import link_findings_to_l2_ap  # noqa: E402
from demo_app.model_runtime import ChampionModels, OriginalFinding, run_champion  # noqa: E402
from demo_app.provenance import InferenceAudit, build_audit  # noqa: E402


@dataclass
class EnrichedFinding:
    original: OriginalFinding
    link_status: str = "unlinked"
    l2_ap_index: int | None = None
    link_distance_px: float | None = None
    correspondence_status: str = "none"  # localized | abstained | unlinked | ap_only
    localization_status: str = "Unlinked"
    committed_edge: PairEdge | None = None
    point_rec: np.ndarray | None = None
    point_world: np.ndarray | None = None
    lat_xy_committed: list[float] | None = None
    eval_dist_mm: float | None = None
    eval_rib_exact: bool | None = None


@dataclass
class CaseInferenceResult:
    case: CaseImages
    findings: list[EnrichedFinding]
    l2: L2CaseResult
    anatomy: object
    audit: InferenceAudit
    champion_ap_peaks: np.ndarray
    champion_lat_peaks: np.ndarray


def _edges_by_ap(edges: list[PairEdge]) -> dict[int, list[PairEdge]]:
    out: dict[int, list[PairEdge]] = {}
    for e in edges:
        out.setdefault(e.ap_idx, []).append(e)
    return out


def run_case_inference(
    case: CaseImages,
    champion: ChampionModels,
    l2: L2Models,
    data_sha: str,
) -> CaseInferenceResult:
    t0 = time.perf_counter()
    findings_raw, cap, clp = run_champion(
        champion, case.ap, case.lat, case.ap_geo, case.lat_geo,
    )
    l2_res = run_l2_case(l2, case.ap, case.lat, case.ap_geo, case.lat_geo)
    anatomy = load_case_anatomy(case.case_id)

    committed_set = set(l2_res.committed_rows)
    committed_edges = [e for e in l2_res.edges if e.global_row in committed_set]
    edges_ap = _edges_by_ap(l2_res.edges)

    ap_xys = [
        (f.ap_xy[0], f.ap_xy[1]) if f.ap_xy else None for f in findings_raw
    ]
    links = link_findings_to_l2_ap(ap_xys, l2_res.l2_ap_candidates, LINK_TOLERANCE_PX)

    enriched: list[EnrichedFinding] = []
    emitted_3d = 0
    for f, link in zip(findings_raw, links):
        ef = EnrichedFinding(original=f)
        ef.link_status = link.status
        ef.l2_ap_index = link.l2_ap_index
        ef.link_distance_px = link.link_distance_px

        if f.source == "ap_only" or f.ap_xy is None:
            ef.correspondence_status = "ap_only"
            ef.localization_status = "Unlinked"
        elif link.l2_ap_index is None:
            ef.correspondence_status = "unlinked"
            ef.localization_status = "Unlinked"
        else:
            # edges touching this L2 AP index
            cand_edges = [e for e in l2_res.edges if e.ap_idx == link.l2_ap_index]
            com = [e for e in cand_edges if e.global_row in committed_set]
            if com:
                edge = com[0]
                ef.committed_edge = edge
                ef.correspondence_status = "localized"
                ef.localization_status = "Localized"
                p_rec, _ = triangulate_edge(edge, case.ap_geo, case.lat_geo)
                ef.point_rec = p_rec
                ef.point_world = world_point(p_rec, anatomy)
                ef.lat_xy_committed = [round(edge.lat_row, 2), round(edge.lat_col, 2)]
                emitted_3d += 1
            elif cand_edges:
                ef.correspondence_status = "abstained"
                ef.localization_status = "Abstained"
            else:
                ef.correspondence_status = "unlinked"
                ef.localization_status = "Unlinked"
        enriched.append(ef)

    runtime = time.perf_counter() - t0
    gate = float(l2_res.policy["gate"])
    eligible = sum(1 for e in l2_res.edges if e.dsi_vox <= gate)
    audit = build_audit(
        case.case_id,
        data_sha,
        champion.detector_run,
        champion.address_model_dir,
        l2.detector_run,
        {**l2_res.policy, "u": l2_res.u},
        len(l2_res.ap_peaks),
        len(l2_res.lat_peaks),
        eligible,
        len(committed_edges),
        emitted_3d,
        runtime,
    )
    return CaseInferenceResult(case, enriched, l2_res, anatomy, audit, cap, clp)
