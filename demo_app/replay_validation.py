"""Sealed L2 policy replay validation (independent of UI)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from nibabel.affines import apply_affine  # noqa: E402

from eval_biplanar_geometry import back_project, case_gt, fracture_metrics  # noqa: E402

from demo_app.config import (  # noqa: E402
    CL_DIR,
    DATA_NPZ,
    IMAGE_DIRS,
    L2_PAIRS_NPZ,
    L2_POLICY,
    L2_SEALED_D1,
    SEG_DIR,
)
from demo_app.correspondence_runtime import load_frozen_policy, replay_accepted_from_npz  # noqa: E402
from demo_app.data_loader import sha256_file  # noqa: E402

CORRECT_MM = 10.0


def count_matched_at10(*, data_npz=DATA_NPZ, pairs_npz=L2_PAIRS_NPZ, policy_path=L2_POLICY) -> int:
    """Reproduce sealed matched@10mm using the frozen pair graph + D1 policy."""
    pol, gate, cost, mb, u = load_frozen_policy()
    if policy_path != L2_POLICY:
        pol = json.loads(Path(policy_path).read_text())
        gate = float(pol["gate"])
        cost = str(pol["cost"])
        mb = bool(pol["mutual_best"])
        from eval_correspondence_D1_assign import u_grid

        u = min(u_grid(cost, gate), key=lambda x: abs(x - float(pol["u"])))

    data_sha = sha256_file(data_npz)
    z = np.load(pairs_npz, allow_pickle=False)
    assert str(z["data_sha256"]) == data_sha, "pairs NPZ data hash mismatch"

    accepted = replay_accepted_from_npz(z, gate, cost, mb, u)
    d = np.load(data_npz, allow_pickle=False)
    cases = [str(c) for c in d["case"]]
    cid_arr = np.array([str(x) for x in z["case_id"]])
    ap_row, ap_col = z["ap_row"], z["ap_col"]
    lat_row, lat_col = z["lat_row"], z["lat_col"]
    cgi = z["case_global_idx"]
    all_ap_geo, all_lat_geo = z["all_ap_geo"], z["all_lat_geo"]

    rows_by_case: dict[str, list[int]] = {}
    for r in range(len(cid_arr)):
        rows_by_case.setdefault(cid_arr[r], []).append(r)

    matched = []
    for cid in cases:
        g = case_gt(cid, IMAGE_DIRS, SEG_DIR, CL_DIR)
        if g is None:
            continue
        rows = accepted.get(cid, [])
        iids = [int(k) for k in g["fl_groups"].keys()]
        trees = {
            iid: cKDTree(apply_affine(g["aff"], g["fl_groups"][iid].T.astype(np.float64)))
            for iid in iids
            if g["fl_groups"][iid].shape[1] > 0
        }
        if not rows:
            continue
        pts = {}
        for r in rows:
            p_rec, si_dis = back_project(
                (float(ap_row[r]), float(ap_col[r])),
                (float(lat_row[r]), float(lat_col[r])),
                all_ap_geo[cgi[r]],
                all_lat_geo[cgi[r]],
            )
            pts[r] = (p_rec, si_dis, apply_affine(g["aff"], p_rec))
        big = 1e9
        m = np.full((len(rows), len(iids)), big)
        for pi, r in enumerate(rows):
            for gi, iid in enumerate(iids):
                if iid in trees:
                    dd, _ = trees[iid].query(pts[r][2][None], k=1)
                    dd = float(dd[0])
                    if dd <= CORRECT_MM:
                        m[pi, gi] = dd
        ri, ci = linear_sum_assignment(m)
        for pi, gi in zip(ri, ci):
            if m[pi, gi] <= CORRECT_MM:
                r = rows[pi]
                iid = iids[gi]
                p_rec, si_dis, _ = pts[r]
                metrics, _ = fracture_metrics(p_rec, si_dis, g["fl_groups"][iid], g, cid, iid)
                if metrics:
                    matched.append((cid, r, iid, float(m[pi, gi]), metrics))
    return len(matched)


def assert_sealed_replay(expected: int | None = None) -> int:
    expected = expected or int(
        json.loads(L2_SEALED_D1.read_text())["operational_headline"]["n_matched_within10"]
    )
    n = count_matched_at10()
    if n != expected:
        raise AssertionError(
            f"Policy replay produced {n} matched fractures at 10 mm; expected {expected}."
        )
    return n
