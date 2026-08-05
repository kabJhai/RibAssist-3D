"""Smoke test: one-case live inference without Streamlit."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")


def test_live_inference_ribfrac119():
    from demo_app.data_loader import ProjectionStore
    from demo_app.model_runtime import load_champion
    from demo_app.correspondence_runtime import load_l2
    from demo_app.pipeline import run_case_inference

    store = ProjectionStore()
    case = store.get("RibFrac119")
    champion = load_champion()
    l2 = load_l2()
    result = run_case_inference(case, champion, l2, store.sha256)
    assert len(result.findings) > 0
    assert result.audit.result_source == "Live model inference"
    assert result.audit.input_data_sha256 == store.sha256
