"""Sealed L2 policy replay must reproduce exactly 15 matched@10mm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("scipy")
pytest.importorskip("nibabel")

from demo_app.config import DATA_NPZ, EXPECTED_MATCHED_AT10, L2_SEALED_D1, SEALED_DATA_SHA256  # noqa: E402
from demo_app.data_loader import sha256_file  # noqa: E402
from demo_app.replay_validation import assert_sealed_replay, count_matched_at10  # noqa: E402


def test_sealed_data_sha():
    assert sha256_file(DATA_NPZ) == SEALED_DATA_SHA256


def test_sealed_replay_matched_at10():
    expected = int(json.loads(L2_SEALED_D1.read_text())["operational_headline"]["n_matched_within10"])
    assert expected == EXPECTED_MATCHED_AT10
    n = count_matched_at10()
    assert n == EXPECTED_MATCHED_AT10, f"replay matched@10 = {n}, expected {EXPECTED_MATCHED_AT10}"


def test_assert_sealed_replay_helper():
    assert assert_sealed_replay() == EXPECTED_MATCHED_AT10
