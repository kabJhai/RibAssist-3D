# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Champion detector + addressing inference (original findings)."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ribassist as RR  # noqa: E402
import train_detector as T  # noqa: E402
import train_address as TA  # noqa: E402

from demo_app.config import ADDRESS_MODEL, CHAMPION_DETECTOR  # noqa: E402


@dataclass
class OriginalFinding:
    finding_id: int
    detection_confidence: float
    address_status: str
    source: str
    side: str | None = None
    rib: int | None = None
    address_score: float | None = None
    ap_xy: list[float] | None = None
    lat_xy: list[float] | None = None
    not_addressed_reason: str | None = None


@dataclass
class ChampionModels:
    device: Any
    nets: dict
    si_tol: float
    op_thr: float
    lat_gate: float
    address_net: Any
    views: str
    use_pos: bool
    crop: int
    half: int
    detector_run: Path
    address_model_dir: Path


def load_champion(device=None) -> ChampionModels:
    dev = device or T.device()
    nets, _arch, si_tol, op_thr, lat_gate, _rec = RR.load_detector(CHAMPION_DETECTOR, dev)
    cfg, views, use_pos, crop, half_ds, _man = RR.load_addressing(ADDRESS_MODEL)
    half = RR.resolve_half(cfg, half_ds, None, False)
    net = TA.Net(views, use_pos).to(dev)
    net.load_state_dict(
        __import__("torch").load(ADDRESS_MODEL / "addressing_model.pt", map_location=dev)
    )
    net.eval()
    return ChampionModels(
        dev, nets, si_tol, op_thr, lat_gate, net, views, use_pos, crop, half,
        CHAMPION_DETECTOR, ADDRESS_MODEL,
    )


def run_champion(
    models: ChampionModels,
    ap_img: np.ndarray,
    lat_img: np.ndarray,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
) -> tuple[list[OriginalFinding], np.ndarray, np.ndarray]:
    active, ap_peaks, lat_peaks = RR.fused_candidates_at_op(
        models.nets, ap_img, lat_img, ap_geo, lat_geo,
        models.si_tol, models.lat_gate, models.op_thr, models.device,
    )
    enriched = [RR.enrich_candidate(c, ap_peaks, lat_peaks) for c in active]
    addressed = RR.address_candidates(
        enriched, ap_img, lat_img, models.address_net, models.views,
        models.use_pos, models.crop, models.half, models.device,
    ) if enriched else []

    findings: list[OriginalFinding] = []
    for i, a in enumerate(addressed, 1):
        findings.append(
            OriginalFinding(
                finding_id=i,
                detection_confidence=float(a["detection_confidence"]),
                address_status=str(a["address_status"]),
                source=str(a["source"]),
                side=a.get("side"),
                rib=a.get("rib"),
                address_score=a.get("address_score"),
                ap_xy=a.get("ap_xy"),
                lat_xy=a.get("lat_xy"),
                not_addressed_reason=a.get("not_addressed_reason"),
            )
        )
    return findings, ap_peaks, lat_peaks
