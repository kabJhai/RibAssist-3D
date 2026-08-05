"""Acceptance tests for 3D anatomy scene correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_app.anatomy_scene import AnatomyBounds, build_anatomy_bundle, validate_coordinates
from demo_app.case_map_3d import build_case_map_figure
from demo_app.finding_map import MapFinding


CASES = ["RibFrac119", "RibFrac142", "RibFrac176", "RibFrac3"]


def _rib_mesh_traces(fig):
    return [t for t in fig.data if t.type == "mesh3d" and t.name and t.name.startswith("Rib ")]


def _centerline_traces(fig):
    return [t for t in fig.data if t.type == "scatter3d" and t.name and t.name.startswith("CL ")]


@pytest.fixture(scope="module")
def bundles():
    out = {}
    for cid in CASES:
        b = build_anatomy_bundle(cid)
        if b is not None:
            out[cid] = b
    return out


@pytest.mark.parametrize("case_id", CASES)
def test_case_produces_mesh_or_centerline(case_id, bundles):
    bundle = bundles.get(case_id)
    if bundle is None:
        pytest.skip(f"anatomy unavailable for {case_id}")
    assert bundle.mesh_labels or len(bundle.centerline_world) >= 1


@pytest.mark.parametrize("case_id", CASES)
def test_anatomy_bounds_contain_localized_fracture(case_id, bundles):
    from demo_app.data_loader import ProjectionStore
    from demo_app.model_runtime import load_champion
    from demo_app.correspondence_runtime import load_l2
    from demo_app.pipeline import run_case_inference

    bundle = bundles.get(case_id)
    if bundle is None:
        pytest.skip(f"anatomy unavailable for {case_id}")

    store = ProjectionStore()
    result = run_case_inference(store.get(case_id), load_champion(), load_l2(), store.sha256)
    pts = [f.point_world for f in result.findings if f.point_world is not None]
    if not pts:
        pytest.skip(f"no localized findings for {case_id}")
    validate_coordinates(bundle, pts)


@pytest.mark.parametrize("case_id", CASES)
def test_scene_ranges_from_anatomy_not_fracture_only(case_id, bundles):
    bundle = bundles.get(case_id)
    if bundle is None:
        pytest.skip(f"anatomy unavailable for {case_id}")

    # Synthetic fracture point far from anatomy center; ranges must still match anatomy
    far_pt = np.array([bundle.bounds.xmax + 500, bundle.bounds.ymax, bundle.bounds.zmax])
    findings = [MapFinding(99, "localized", "L1", 1, point_world=far_pt)]
    fig = build_case_map_figure(bundle, findings=findings, selected_id=99, height=400)
    expected = bundle.bounds.as_plotly_ranges()
    scene = fig.layout.scene
    assert list(scene.xaxis.range) == pytest.approx(expected["xaxis"], rel=1e-4)
    assert list(scene.yaxis.range) == pytest.approx(expected["yaxis"], rel=1e-4)
    assert list(scene.zaxis.range) == pytest.approx(expected["zaxis"], rel=1e-4)
    assert scene.aspectmode == "data"
    assert scene.xaxis.autorange is False


@pytest.mark.parametrize("case_id", CASES)
def test_figure_has_anatomy_after_finding_switch(case_id, bundles):
    bundle = bundles.get(case_id)
    if bundle is None:
        pytest.skip(f"anatomy unavailable for {case_id}")

    b = bundle.bounds
    pt1 = np.array([(b.xmin + b.xmax) / 2, b.ymin, b.zmin])
    pt2 = np.array([(b.xmin + b.xmax) / 2 + 5, b.ymin + 5, b.zmin])
    findings = [
        MapFinding(1, "localized", "L6", 1, point_world=pt1),
        MapFinding(2, "candidate", "L7", 2, candidate_world=pt2),
    ]
    fig_a = build_case_map_figure(bundle, findings=findings, selected_id=1, height=400)
    fig_b = build_case_map_figure(bundle, findings=findings, selected_id=2, height=400)

    anatomy_a = len(_rib_mesh_traces(fig_a)) + len(_centerline_traces(fig_a))
    anatomy_b = len(_rib_mesh_traces(fig_b)) + len(_centerline_traces(fig_b))
    assert anatomy_a >= 1
    assert anatomy_b >= 1
    assert anatomy_a == anatomy_b


def test_mesh_faces_point_outward():
    bundle = build_anatomy_bundle("RibFrac119")
    assert bundle and bundle.meshes and bundle.meshes.ribs
    rib = bundle.meshes.ribs[0]
    v, f = rib.vertices, rib.faces
    cen = v.mean(axis=0)
    sample = f[: min(400, len(f))]
    outward = 0
    for tri in sample:
        p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
        n = np.cross(p1 - p0, p2 - p0)
        fc = (p0 + p1 + p2) / 3
        if np.dot(n, fc - cen) > 0:
            outward += 1
    assert outward / len(sample) > 0.65, "rib mesh faces should predominantly point outward"


def test_mesh_bounds_overlap_centerlines(bundles):
    for case_id, bundle in bundles.items():
        if not bundle.meshes or not bundle.meshes.ribs:
            continue
        cl_bounds = AnatomyBounds.from_points(np.vstack(bundle.centerline_world))
        mesh_pts = np.vstack([m.vertices for m in bundle.meshes.ribs])
        mesh_bounds = AnatomyBounds.from_points(mesh_pts)
        assert mesh_bounds.overlaps(cl_bounds), f"{case_id}: mesh vs CL bounds mismatch"
