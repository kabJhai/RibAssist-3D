# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Inference audit trail and provenance hashes."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_detector as T  # noqa: E402

from demo_app.config import AP_FLOOR, AP_NMS, LAT_FLOOR, LAT_NMS, SEALED_DATA_SHA256  # noqa: E402
from demo_app.data_loader import sha256_file  # noqa: E402


@dataclass
class InferenceAudit:
    result_source: str = "Live model inference"
    case_id: str = ""
    input_data_sha256: str = ""
    champion_detector_run: str = ""
    champion_detector_ap_sha256: str = ""
    champion_detector_lat_sha256: str = ""
    addressing_model_dir: str = ""
    addressing_checkpoint_sha256: str = ""
    l2_detector_run: str = ""
    l2_detector_ap_sha256: str = ""
    l2_detector_lat_sha256: str = ""
    ap_extraction: str = f"nms{AP_NMS}/floor{AP_FLOOR}"
    lateral_extraction: str = f"nms{LAT_NMS}/floor{LAT_FLOOR}"
    correspondence_policy: dict = field(default_factory=dict)
    ap_candidate_count: int = 0
    lateral_candidate_count: int = 0
    eligible_pair_count: int = 0
    accepted_pair_count: int = 0
    emitted_3d_point_count: int = 0
    inference_runtime_sec: float = 0.0
    timestamp_utc: str = ""
    cached_inference_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_hashes(run_dir: Path) -> tuple[str, str]:
    ap = sha256_file(run_dir / "detector_ap.pt")
    lat = sha256_file(run_dir / "detector_lat.pt")
    return ap, lat


def build_audit(
    case_id: str,
    data_sha: str,
    champion_run: Path,
    address_dir: Path,
    l2_run: Path,
    policy: dict,
    ap_cands: int,
    lat_cands: int,
    eligible: int,
    accepted: int,
    emitted_3d: int,
    runtime_sec: float,
) -> InferenceAudit:
    cap, cal = checkpoint_hashes(champion_run)
    lap, lal = checkpoint_hashes(l2_run)
    return InferenceAudit(
        case_id=case_id,
        input_data_sha256=data_sha,
        champion_detector_run=str(champion_run),
        champion_detector_ap_sha256=cap,
        champion_detector_lat_sha256=cal,
        addressing_model_dir=str(address_dir),
        addressing_checkpoint_sha256=sha256_file(address_dir / "addressing_model.pt"),
        l2_detector_run=str(l2_run),
        l2_detector_ap_sha256=lap,
        l2_detector_lat_sha256=lal,
        correspondence_policy={
            "gate": policy.get("gate"),
            "cost": policy.get("cost"),
            "mutual_best": policy.get("mutual_best"),
            "u": policy.get("u"),
        },
        ap_candidate_count=ap_cands,
        lateral_candidate_count=lat_cands,
        eligible_pair_count=eligible,
        accepted_pair_count=accepted,
        emitted_3d_point_count=emitted_3d,
        inference_runtime_sec=round(runtime_sec, 3),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def assert_sealed_data_sha(data_sha: str) -> None:
    if data_sha != SEALED_DATA_SHA256:
        raise ValueError(f"Expected sealed data sha {SEALED_DATA_SHA256}, got {data_sha}")
