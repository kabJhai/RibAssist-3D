#!/usr/bin/env bash
# Smoke-test the RibAssist 3D clinician demo (no Streamlit UI).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PY:-}" ]]; then
  if [[ -x "../.venv/bin/python" ]]; then
    PY="../.venv/bin/python"
  else
    PY="python3"
  fi
fi
export PYTHONPATH=.
$PY -m pytest tests/test_finding_linker.py tests/test_correspondence_replay.py tests/test_demo_inference.py tests/test_anatomy_scene.py tests/test_finding_map.py -q
echo "Demo smoke tests passed."
