# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Tests for split manifest validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from validate_local_artifacts import case_list_sha256, check_split


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "split_manifest.json"


def test_published_splits_match_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, spec in manifest["published_splits"].items():
        ok, errors = check_split(ROOT / name, spec)
        assert ok, errors


def test_case_list_hashes_stable():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "frozen_split.json").read_text(encoding="utf-8"))
    frozen_spec = manifest["published_splits"]["frozen_split.json"]
    assert case_list_sha256(frozen, frozen_spec["lists"]) == frozen_spec["case_list_sha256"]


def test_geometry_split_matches_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    geom = json.loads((ROOT / "geometry_split.json").read_text(encoding="utf-8"))
    geom_spec = manifest["published_splits"]["geometry_split.json"]
    assert case_list_sha256(geom, geom_spec["lists"]) == geom_spec["case_list_sha256"]


def test_validate_script_splits_only():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_local_artifacts.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
