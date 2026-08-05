# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Helpers so Streamlit/Plotly render mesh + scatter in the same coordinate frame."""
from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import plotly.graph_objects as go

_DTYPE = {
    "f8": np.float64,
    "f4": np.float32,
    "i4": np.int32,
    "u4": np.uint32,
    "i2": np.int16,
    "u2": np.uint16,
    "i1": np.int8,
    "u1": np.uint8,
}


def _expand_bdata(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=_DTYPE[obj["dtype"]])
            if "shape" in obj:
                shape = tuple(int(s) for s in obj["shape"].split(","))
                arr = arr.reshape(shape)
            return arr.tolist()
        return {k: _expand_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_bdata(v) for v in obj]
    return obj


def figure_for_streamlit(fig: go.Figure) -> go.Figure:
    """Expand Plotly 6 binary arrays so 3D meshes and markers share one JSON frame."""
    payload = _expand_bdata(json.loads(fig.to_json()))
    return go.Figure(data=payload["data"], layout=payload["layout"])
