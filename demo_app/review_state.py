# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Clinician review actions (Streamlit session persistence helpers)."""
from __future__ import annotations

REVIEW_CHOICES = ("pending", "accepted", "rejected", "needs_review")


def default_review_map(finding_ids: list[int]) -> dict[int, str]:
    return {fid: "pending" for fid in finding_ids}


def set_review(state: dict, finding_id: int, status: str) -> None:
    if status not in REVIEW_CHOICES:
        raise ValueError(status)
    state[finding_id] = status
