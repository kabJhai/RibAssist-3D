# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Tests for finding ↔ L2 AP linkage."""
from demo_app.finding_linker import L2ApCandidate, link_findings_to_l2_ap


def test_exact_link():
    cands = [L2ApCandidate(0, 10.0, 20.0, 0.9), L2ApCandidate(1, 50.0, 50.0, 0.8)]
    findings = [(10.0, 20.0), (99.0, 99.0)]
    links = link_findings_to_l2_ap(findings, cands, tolerance_px=2.0)
    assert links[0].status == "linked"
    assert links[0].l2_ap_index == 0
    assert links[0].link_distance_px == 0.0
    assert links[1].status == "unlinked"


def test_competing_findings_deterministic():
    cands = [L2ApCandidate(0, 10.0, 10.0, 0.5)]
    findings = [(10.5, 10.5), (10.2, 10.2)]
    links = link_findings_to_l2_ap(findings, cands, tolerance_px=2.0)
    linked = [l for l in links if l.status == "linked"]
    assert len(linked) == 1
    assert linked[0].finding_index == 1  # closer wins first in sorted greedy


def test_no_ap_peak():
    links = link_findings_to_l2_ap([None], [], tolerance_px=2.0)
    assert links[0].status == "no_ap_peak"
