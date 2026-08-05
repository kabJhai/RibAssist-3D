"""L2 detector peaks, pair graph, frozen D1 policy, and 3D back-projection."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_detector as T  # noqa: E402
import run_ribassist as RR  # noqa: E402
from eval_correspondence_D1_assign import assign_abstain, edge_cost, u_grid  # noqa: E402
from eval_biplanar_geometry import back_project  # noqa: E402
from nibabel.affines import apply_affine  # noqa: E402

from demo_app.config import (  # noqa: E402
    AP_FLOOR,
    AP_NMS,
    AUDIT_GATE_MAX,
    LAT_FLOOR,
    LAT_NMS,
    L2_DETECTOR,
    L2_POLICY,
)
from demo_app.finding_linker import L2ApCandidate  # noqa: E402


@dataclass
class PairEdge:
    global_row: int
    ap_idx: int
    lat_idx: int
    ap_row: float
    ap_col: float
    lat_row: float
    lat_col: float
    ap_score: float
    lat_score: float
    dsi_vox: float


@dataclass
class L2CaseResult:
    ap_peaks: np.ndarray
    lat_peaks: np.ndarray
    ap_heatmap: np.ndarray
    lat_heatmap: np.ndarray
    edges: list[PairEdge]
    committed_rows: list[int]
    l2_ap_candidates: list[L2ApCandidate]
    policy: dict
    u: float
    commit_threshold: float


@dataclass
class L2Models:
    device: Any
    nets: dict
    detector_run: Path


def load_l2(device=None) -> L2Models:
    dev = device or T.device()
    nets, _a, _si, _op, _gate, _rec = RR.load_detector(L2_DETECTOR, dev)
    return L2Models(dev, nets, L2_DETECTOR)


def _extract_peaks(
    nets: dict,
    ap_img: np.ndarray,
    lat_img: np.ndarray,
    device,
    ap_nms: int,
    ap_floor: float,
    lat_nms: int,
    lat_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        ap_t = torch.from_numpy(ap_img.astype(np.float32))[None, None].to(device)
        lat_t = torch.from_numpy(lat_img.astype(np.float32))[None, None].to(device)
        ap_hm_t = nets["ap"](ap_t)[0, 0]
        lat_hm_t = nets["lat"](lat_t)[0, 0]
        ap_hm = ap_hm_t.detach().cpu().numpy()
        lat_hm = lat_hm_t.detach().cpu().numpy()
    ap_pk = T.peaks_from_hm(ap_hm_t, radius=ap_nms, thresh=ap_floor)
    lat_pk = T.peaks_from_hm(lat_hm_t, radius=lat_nms, thresh=lat_floor)
    return ap_pk, lat_pk, ap_hm, lat_hm


def build_case_edges(
    ap_peaks: np.ndarray,
    lat_peaks: np.ndarray,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
    gate_max: float = AUDIT_GATE_MAX,
) -> list[PairEdge]:
    if len(ap_peaks) == 0 or len(lat_peaks) == 0:
        return []
    si_ap = T.si_voxel(ap_peaks[:, 0], ap_geo)
    si_lat = T.si_voxel(lat_peaks[:, 0], lat_geo)
    dsi = np.abs(si_ap[:, None] - si_lat[None, :])
    ii, jj = np.nonzero(dsi <= gate_max)
    edges: list[PairEdge] = []
    for k, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
        edges.append(
            PairEdge(
                global_row=k,
                ap_idx=int(i),
                lat_idx=int(j),
                ap_row=float(ap_peaks[i, 0]),
                ap_col=float(ap_peaks[i, 1]),
                lat_row=float(lat_peaks[j, 0]),
                lat_col=float(lat_peaks[j, 1]),
                ap_score=float(ap_peaks[i, 2]),
                lat_score=float(lat_peaks[j, 2]),
                dsi_vox=float(dsi[i, j]),
            )
        )
    return edges


def assign_case(
    edges: list[PairEdge],
    gate: float,
    cost_name: str,
    mutual_best: bool,
    u: float,
) -> list[int]:
    """Return global_row indices of committed pair edges."""
    if not edges:
        return []
    rr = [e for e in edges if e.dsi_vox <= gate]
    if not rr:
        return []
    ap_u = sorted({e.ap_idx for e in rr})
    lat_u = sorted({e.lat_idx for e in rr})
    ai = {v: i for i, v in enumerate(ap_u)}
    lj = {v: i for i, v in enumerate(lat_u)}
    na, nl = len(ap_u), len(lat_u)
    BIG = 1e9
    M = np.full((na, nl), BIG)
    cost_of: dict[tuple[int, int], tuple[int, float]] = {}
    for e in rr:
        i, j = ai[e.ap_idx], lj[e.lat_idx]
        cval = float(
            edge_cost(
                cost_name,
                np.array([e.ap_score]),
                np.array([e.lat_score]),
                np.array([e.dsi_vox]),
                gate,
            )[0]
        )
        if cval < M[i, j]:
            M[i, j] = cval
            cost_of[(i, j)] = (e.global_row, cval)
    if mutual_best:
        rm, cm = M.min(1), M.min(0)
        cost_of = {
            k: v for k, v in cost_of.items()
            if M[k] <= rm[k[0]] + 1e-12 and M[k] <= cm[k[1]] + 1e-12
        }
    return list(assign_abstain(na, nl, cost_of, u))


def load_frozen_policy() -> tuple[dict, float, float, bool, float]:
    pol = json.loads(L2_POLICY.read_text())
    gate = float(pol["gate"])
    cost = str(pol["cost"])
    mb = bool(pol["mutual_best"])
    ug = list(u_grid(cost, gate))
    u = min(ug, key=lambda x: abs(x - float(pol["u"])))
    return pol, gate, cost, mb, u


def run_l2_case(
    models: L2Models,
    ap_img: np.ndarray,
    lat_img: np.ndarray,
    ap_geo: np.ndarray,
    lat_geo: np.ndarray,
) -> L2CaseResult:
    ap_pk, lat_pk, ap_hm, lat_hm = _extract_peaks(
        models.nets, ap_img, lat_img, models.device, AP_NMS, AP_FLOOR, LAT_NMS, LAT_FLOOR,
    )
    edges = build_case_edges(ap_pk, lat_pk, ap_geo, lat_geo)
    pol, gate, cost, mb, u = load_frozen_policy()
    committed = assign_case(edges, gate, cost, mb, u)
    commit_thr = max(0.0, 1.0 - 2.0 * u)
    l2_cands = [
        L2ApCandidate(i, float(ap_pk[i, 0]), float(ap_pk[i, 1]), float(ap_pk[i, 2]))
        for i in range(len(ap_pk))
    ]
    return L2CaseResult(
        ap_peaks=ap_pk,
        lat_peaks=lat_pk,
        ap_heatmap=ap_hm,
        lat_heatmap=lat_hm,
        edges=edges,
        committed_rows=committed,
        l2_ap_candidates=l2_cands,
        policy=pol,
        u=u,
        commit_threshold=commit_thr,
    )


def triangulate_edge(edge: PairEdge, ap_geo: np.ndarray, lat_geo: np.ndarray) -> tuple[np.ndarray, float]:
    p_rec, si_dis = back_project(
        (edge.ap_row, edge.ap_col),
        (edge.lat_row, edge.lat_col),
        ap_geo,
        lat_geo,
    )
    return np.asarray(p_rec, dtype=np.float64), float(si_dis)


def world_point(p_rec: np.ndarray, anatomy) -> np.ndarray | None:
    if anatomy is None:
        return None
    return apply_affine(anatomy["aff"], p_rec)


def replay_accepted_from_npz(z, gate, cost, mb, u) -> dict[str, list[int]]:
    """Same replay logic as make_demo_figures.replay_accepted (for tests)."""
    cid_arr = np.array([str(x) for x in z["case_id"]])
    apx, ltx = z["ap_idx"], z["lat_idx"]
    ap_s = z["ap_score"].astype(np.float64)
    lt_s = z["lat_score"].astype(np.float64)
    dsi = z["dsi_vox"].astype(np.float64)
    gmask = dsi <= gate
    costs_all = edge_cost(cost, ap_s, lt_s, dsi, gate)
    rows_by: dict[str, list[int]] = {}
    for r in range(len(cid_arr)):
        rows_by.setdefault(cid_arr[r], []).append(r)
    out: dict[str, list[int]] = {}
    for cid, rows in rows_by.items():
        rr = [r for r in rows if gmask[r]]
        if not rr:
            out[cid] = []
            continue
        aps_u = sorted({int(apx[r]) for r in rr})
        lts_u = sorted({int(ltx[r]) for r in rr})
        ai = {v: i for i, v in enumerate(aps_u)}
        lj = {v: i for i, v in enumerate(lts_u)}
        na, nl = len(aps_u), len(lts_u)
        BIG = 1e9
        Mreal = np.full((na, nl), BIG)
        cost_of: dict[tuple[int, int], tuple[int, float]] = {}
        for r in rr:
            i, j = ai[int(apx[r])], lj[int(ltx[r])]
            if costs_all[r] < Mreal[i, j]:
                Mreal[i, j] = costs_all[r]
                cost_of[(i, j)] = (r, float(costs_all[r]))
        if mb:
            rowmin, colmin = Mreal.min(1), Mreal.min(0)
            cost_of = {
                k: v for k, v in cost_of.items()
                if Mreal[k] <= rowmin[k[0]] + 1e-12 and Mreal[k] <= colmin[k[1]] + 1e-12
            }
        out[cid] = list(assign_abstain(na, nl, cost_of, u))
    return out
