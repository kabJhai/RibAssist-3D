#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""RibAssist 3D CANONICAL RIB-LEVEL CONTEXT (schema ribassist-rib-level-2).

Takes addressed detections (side, predicted rib level; along-rib s is recorded but EXPLORATORY and
NOT used) and renders them on a FIXED mean rib atlas at the granularity the addressing model actually
supports: side + APPROXIMATE rib level. For each predicted (side, rib) it highlights that rib's
centerline in bold and its +/-1 same-side neighbours as an ADJACENT-RIB CONTEXT band (exact-rib is
limited); it never places a point at an s along the rib, so nothing implies a precise lesion location.
This is anatomical CONTEXT, NOT a patient-specific reconstruction and NOT distinct fracture sites.

Discrete pattern only: unique predicted rib levels per side, consecutive predicted levels, both-sided
occurrence. There is NO region / clustering / inter-detection distance (those needed along-rib s) and
NO multisite / flail claim. Multiple detections at one predicted rib level are reported as a QUALITY
flag (possible duplicate detector responses), not as separate fractures. Measured performance lives in
the evaluation manifest (eval_address_e2e.py), never hard-coded here.

Emits per case:
  * reconstruction.json  — schema ribassist-rib-level-2: predicted_rib_level_detections (side, rib,
    adjacent_rib_context_levels, s_predicted_exploratory, confidence), discrete pattern, cues, quality_flags;
  * reconstruction.html  — rotatable Plotly canonical rib-level context (predicted ribs bold + context);
  * reconstruction.png   — static render if kaleido is available (json records png_written).

The 'projection' silhouette scale fit was evaluated on 65 held-out cases and REJECTED (worse than the
fixed reference); it is retained only as a documented negative baseline. Default is the fixed atlas.

Usage:
  python reconstruct_3d.py --atlas outputs/rib_atlas_build --candidates addressed_candidates.npz \
      --out outputs/reconstructions
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

REF_DIMS = {"lr_width_mm": 300.0, "ap_depth_mm": 200.0, "si_height_mm": 250.0}
# NOTE: no global along-rib error scale. The deployed addressing model does not demonstrate reliable
# along-rib s, so this renderer highlights predicted rib LEVEL (+/-1 uncertainty) rather than placing a
# point at an s along the rib; there is no position-error radius (it would imply an s accuracy we lack).


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_atlas(atlas_dir):
    atlas_dir = Path(atlas_dir)
    man = json.loads((atlas_dir / "rib_atlas_manifest.json").read_text())
    got = sha256_file(atlas_dir / "rib_atlas.npz")
    if got != man.get("atlas_sha256"):
        raise ValueError(f"atlas sha mismatch: {got[:12]}.. != manifest {str(man.get('atlas_sha256'))[:12]}..")
    z = np.load(atlas_dir / "rib_atlas.npz", allow_pickle=False)
    side, num, valid = z["rib_side"].astype(str), z["rib_num"], z["valid"]
    lut = {(str(side[r]), int(num[r])): r for r in range(len(side)) if valid[r]}
    expected = {(sd, n) for sd in ("L", "R") for n in range(1, 13)}
    missing = expected - set(lut)
    if missing:
        raise ValueError(f"Atlas is incomplete: missing rib slots {sorted(missing)}. A valid atlas must have all "
                         "24 (side, rib) mappings; refusing to reconstruct with silent substitutions.")
    return {"template": z["template"], "std": z["pointwise_std"], "lut": lut, "K": int(z["k"]), "manifest": man}


# ---------------------------------------------------------------------------------------------
# Projection-only patient-frame fitter (NO CT): estimate thoracic LR/AP/SI extent from the images.
# ---------------------------------------------------------------------------------------------
def fit_patient_frame(ap_img, lat_img, ap_sp, lat_sp, thresh=0.12):
    """Estimate patient thoracic dimensions (mm) from the two projections using the stored per-view
    post-resize spacing (mm/px). AP -> LR width + SI height; lateral -> AP depth. Silhouette by a
    simple attenuation threshold. Bounded and honest (includes soft tissue; error is quantified by
    the geometry evaluator). Returns dims{lr,ap,si mm}, center=[0,0,0] (patient-local frame), meta."""
    def extent(img, thr):
        m = img > (thr * float(img.max()) if img.max() > 0 else thr)
        rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
        if len(rows) == 0 or len(cols) == 0: return 0.0, 0.0
        return float(rows[-1] - rows[0] + 1), float(cols[-1] - cols[0] + 1)   # (row_span_px, col_span_px)
    ap_rows, ap_cols = extent(np.asarray(ap_img, float), thresh)
    lat_rows, lat_cols = extent(np.asarray(lat_img, float), thresh)
    ap_sp = np.asarray(ap_sp, float); lat_sp = np.asarray(lat_sp, float)   # [si_mm/px, lr_mm/px], [si_mm/px, ap_mm/px]
    dims = {"lr_width_mm": round(ap_cols * ap_sp[1], 1),
            "ap_depth_mm": round(lat_cols * lat_sp[1], 1),
            "si_height_mm": round(0.5 * (ap_rows * ap_sp[0] + lat_rows * lat_sp[0]), 1)}
    meta = {"method": "projection silhouette threshold (attenuation), anisotropic scale only, NO pose", "threshold": thresh,
            "ap_span_px": [ap_rows, ap_cols], "lat_span_px": [lat_rows, lat_cols],
            "caveat": "silhouette includes soft tissue/arms; magnitude error quantified by the geometry evaluator"}
    if min(dims.values()) <= 0:  # NO silent reference fallback inside the projection method
        raise ValueError("Projection silhouette fit failed: empty or invalid extent. Use --scale-mode reference "
                         "explicitly if a fixed frame is intended.")
    return dims, [0.0, 0.0, 0.0], meta


def denorm(pt_norm, dims, center):
    s = np.array([dims["lr_width_mm"] / 2, dims["ap_depth_mm"] / 2, dims["si_height_mm"] / 2])
    return np.asarray(center) + np.asarray(pt_norm) * s


def rib_centerline_mm(atlas, side, rib, dims, center):
    r = atlas["lut"].get((side, int(rib)))
    if r is None:  # atlas completeness is asserted at load, so this means a bad candidate label
        raise KeyError(f"candidate rib ({side}{rib}) has no atlas slot; refusing to substitute another rib.")
    return denorm(atlas["template"][r], dims, center), r


def adjacent_rib_levels(side, rib, atlas):
    """The predicted rib level plus its +/-1 same-side neighbours that exist in the atlas — shown as
    CONTEXT because exact-rib discrimination is limited (measured values live in the evaluation manifest,
    not hard-coded here). e.g. L2 -> [L1,L2,L3]. This is an adjacent-rib context band, NOT a calibrated
    uncertainty interval."""
    lv = [int(rib) + d for d in (-1, 0, 1)]
    return [f"{side}{n}" for n in lv if 1 <= n <= 12 and (side, n) in atlas["lut"]]


def _longest_run_set(pairs):
    """Return (length, rib_pairs) of the longest run of consecutive same-side ribs."""
    best_len, best_set = 0, []
    cur = []
    for sd, n in sorted(pairs, key=lambda t: (t[0], t[1])):
        if cur and cur[-1][0] == sd and n == cur[-1][1] + 1: cur.append((sd, n))
        else: cur = [(sd, n)]
        if len(cur) > best_len: best_len, best_set = len(cur), list(cur)
    return best_len, best_set


def extract_pattern(detections):
    """DISCRETE-ONLY pattern, using ONLY what the addressing model demonstrably provides: side and
    (approximate) rib level. Along-rib s is NOT used (exploratory), so there is no region / clustering /
    inter-detection distance. The pipeline has NOT established distinct fracture SITES, so nothing here
    claims 'multisite' or flail: multiple detections at one predicted rib level are reported as a QUALITY
    flag (possible duplicate detector responses), not as separate fractures."""
    from collections import Counter
    ribset = sorted({(x["side"], x["rib"]) for x in detections}, key=lambda t: (t[0], t[1]))
    unique = [f"{sd}{n}" for sd, n in ribset]
    counts = Counter((x["side"], x["rib"]) for x in detections)
    run_len, run_set = _longest_run_set(ribset)
    per_side = {}
    for sd in ("L", "R"):
        lv = sorted(n for s2, n in ribset if s2 == sd)
        if lv: per_side[sd] = [f"{sd}{n}" for n in lv]
    return {"n_addressed_detections": len(detections), "unique_rib_levels": unique,
            "n_unique_rib_levels": len(unique), "predicted_rib_levels_by_side": per_side,
            "longest_consecutive_run": run_len, "longest_run_ribs": [f"{sd}{n}" for sd, n in run_set],
            "bilateral": len({sd for sd, _ in ribset}) == 2,
            "multiple_detections_same_rib_level": {f"{sd}{n}": counts[(sd, n)] for (sd, n), c in counts.items() if c >= 2},
            "granularity": "side + approximate rib level; adjacent ribs shown as context; distinct fracture "
                           "sites NOT established"}


def decision_cues(pattern, detections):
    """Review cues from discrete addresses ONLY (side, approximate rib level, consecutive levels, both-sided).
    Returns (cues, quality_flags). NO definite/possible qualifier — the detector score is not a calibrated
    probability and there is no clinical-involvement claim; the system predicts addresses. Multiple detections
    at one rib level is a QUALITY flag (possible duplicate responses), NOT a trauma cue. No multisite/flail."""
    cues, flags = [], []
    if pattern["longest_consecutive_run"] >= 3:
        cues.append(f">=3 consecutive predicted rib levels: {'-'.join(pattern['longest_run_ribs'])}; "
                    "rib levels are approximate.")
    if pattern["bilateral"]:
        cues.append("predicted rib levels occur on both sides")
    if pattern["multiple_detections_same_rib_level"]:
        flags.append(f"Multiple detections resolved to the same predicted rib level "
                     f"{pattern['multiple_detections_same_rib_level']}; may represent duplicate detector responses, "
                     "not distinct fractures.")
    if not cues:
        cues.append("no high-order pattern cue triggered")
    return cues, flags


def plotly_html(case_id, atlas, detections, dims, center, out_html, out_png, want_png):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False, False
    fig = go.Figure()
    # RIB-LEVEL rendering (NOT point placement). The addressing model gives side + APPROXIMATE rib level and
    # no reliable along-rib s, so we HIGHLIGHT whole ribs:
    #   - each predicted rib level: bold red centerline (the model's best guess);
    #   - its +/-1 same-side neighbours: lighter red ADJACENT-RIB CONTEXT band (exact-rib is limited);
    #   - all other ribs: faint gray context.
    # No marker is placed at an s along the rib, so the picture never implies a precise lesion location.
    from collections import Counter
    predicted = Counter((x["side"], int(x["rib"])) for x in detections)     # detections per (side, rib level)
    band = set()                                                            # +/-1 same-side neighbour ribs (context)
    for (sd, rb) in predicted:
        for lab in adjacent_rib_levels(sd, rb, atlas):
            band.add((lab[0], int(lab[1:])))
    band -= set(predicted)
    for (sd, n), r in atlas["lut"].items():
        cl = denorm(atlas["template"][r], dims, center)
        if (sd, n) in predicted:
            col, w, name = "#e4572e", 7, f"predicted {sd}{n} (x{predicted[(sd, n)]})"
        elif (sd, n) in band:
            col, w, name = "#f3a08c", 4, f"adjacent-rib context ({sd}{n})"
        else:
            col, w, name = "#c9d2dd", 2, f"{sd} rib {n}"
        fig.add_trace(go.Scatter3d(x=cl[:, 0], y=cl[:, 1], z=cl[:, 2], mode="lines",
                                   line=dict(color=col, width=w), showlegend=False,
                                   hovertext=name, hoverinfo="text"))
        if (sd, n) in predicted:  # label the predicted rib once, at its superior-most point
            top = cl[int(np.argmax(cl[:, 2]))]
            fig.add_trace(go.Scatter3d(x=[top[0]], y=[top[1]], z=[top[2]], mode="text",
                                       text=[f"{sd}{n} ×{predicted[(sd, n)]}"], textposition="top center",
                                       textfont=dict(size=12, color="#7a1a00"), showlegend=False, hoverinfo="skip"))
    dup = {f"{sd}{rb}": c for (sd, rb), c in predicted.items() if c > 1}
    coincident_note = (" | multiple detections at: " + ", ".join(f"{k} x{v}" for k, v in dup.items())) if dup else ""
    # RADIOLOGICAL initial view to match the AP overlay: look from the anterior (+y) with superior (+z) up,
    # so patient-Right (+x) renders on the viewer's LEFT — same handedness as the radiological AP panel.
    fig.update_layout(title=f"RibAssist 3D PREDICTED RIB LEVELS (bold) + adjacent-rib context — side + approximate "
                            f"rib, NOT a reconstruction — {case_id}" + coincident_note,
                      scene=dict(xaxis_title="L–R (mm): +x = patient Right (shown left)",
                                 yaxis_title="P–A (mm)", zaxis_title="I–S (mm)", aspectmode="data",
                                 camera=dict(eye=dict(x=0.0, y=2.4, z=0.2), up=dict(x=0, y=0, z=1))),
                      template="plotly_white")
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    png_ok = False
    if want_png:  # opt-in: kaleido can hang in headless sandboxes, so never attempt unless asked
        try:
            fig.write_image(str(out_png)); png_ok = True
        except Exception:
            pass
    return True, png_ok


def build_candidates(path, limit):
    path = Path(path); by_case = {}
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)
        side, rib, s, case = d["side"], d["rib"], d["s"], d["case"].astype(str)
        conf = d["conf"].astype(float) if "conf" in d else np.ones(len(s))
        for i in range(len(s)):
            by_case.setdefault(str(case[i]), []).append(
                {"side": "L" if int(side[i]) == 0 else "R", "rib": int(rib[i]), "s": float(s[i]), "confidence": float(conf[i])})
    else:
        for row in json.loads(path.read_text()):
            by_case.setdefault(str(row["case_id"]), []).append(
                {"side": row["side"], "rib": int(row["rib"]), "s": float(row["s"]), "confidence": float(row.get("confidence", 1.0))})
    cases = list(by_case.keys());  cases = cases[:limit] if limit else cases
    return {c: by_case[c] for c in cases}


def load_projections(path):
    if path is None: return {}
    d = np.load(path, allow_pickle=False); case = d["case"].astype(str)
    return {str(case[i]): {"ap": d["ap"][i].astype(np.float32), "lat": d["lat"][i].astype(np.float32),
                           "ap_sp": d["ap_sp"][i], "lat_sp": d["lat_sp"][i]} for i in range(len(case))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--projections", type=Path, default=None, help="npz with ap/lat/ap_sp/lat_sp/case for projection fit")
    ap.add_argument("--out", type=Path, default=Path("outputs/reconstructions"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale-mode", choices=["reference", "projection"], default="reference")
    ap.add_argument("--png", action="store_true", help="also write static PNG via kaleido (may be slow/absent; success recorded)")
    a = ap.parse_args()
    if a.scale_mode == "projection" and a.projections is None:
        raise ValueError("--scale-mode projection requires --projections (AP+lat images to fit from).")
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; reconstruction outputs are versioned/immutable — use a new dir.")
    atlas = load_atlas(a.atlas); cands = build_candidates(a.candidates, a.limit); proj = load_projections(a.projections)
    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"reconstructing {len(cands)} case(s) | atlas {atlas['manifest']['atlas_version']} | scale-mode {a.scale_mode}", flush=True)
    index = []
    for cid, raw in cands.items():
        if a.scale_mode == "projection":
            if cid not in proj:  # NO silent fallback — projection mode must actually adapt from images
                raise KeyError(f"{cid}: missing projection inputs; projection mode cannot adapt this case.")
            p = proj[cid]; dims, center, fitmeta = fit_patient_frame(p["ap"], p["lat"], p["ap_sp"], p["lat_sp"])
            scale_note = "projection-fitted anisotropic scale (WEAK BASELINE: body silhouette, no pose)"
        else:
            dims, center, fitmeta = dict(REF_DIMS), [0.0, 0.0, 0.0], {"method": "fixed canonical atlas frame (deployed substrate)"}
            scale_note = "fixed canonical atlas frame (deployed substrate; geometry is CONTEXT, not patient-specific)"
        detections = []
        for k, cd in enumerate(raw, 1):
            rib_centerline_mm(atlas, cd["side"], cd["rib"], dims, center)  # validates the (side,rib) atlas slot exists
            detections.append({"detection_id": k, "side": cd["side"], "rib": int(cd["rib"]),
                               "adjacent_rib_context_levels": adjacent_rib_levels(cd["side"], int(cd["rib"]), atlas),
                               "s_predicted_exploratory": round(float(cd["s"]), 3),   # recorded, NOT used for placement
                               "confidence_score": round(float(cd.get("confidence", 1.0)), 3)})
        pattern = extract_pattern(detections); cues, quality_flags = decision_cues(pattern, detections)
        cdir = work / cid; cdir.mkdir(parents=True, exist_ok=True)
        html_ok, png_ok = plotly_html(cid, atlas, detections, dims, center, cdir / "reconstruction.html", cdir / "reconstruction.png", a.png)
        rec = {"case_id": cid, "schema": "ribassist-rib-level-2", "geometry_type": "canonical_rib_level_context",
               "scale": scale_note, "atlas_version": atlas["manifest"]["atlas_version"],
               "atlas_sha256": atlas["manifest"]["atlas_sha256"],
               "predicted_rib_level_detections": detections, "pattern": pattern,
               "decision_support_cues": cues, "quality_flags": quality_flags,
               "png_written": png_ok, "html_written": html_ok,
               "claim_tier": "PREDICTED RIB-LEVEL CONTEXT on a fixed canonical atlas. The addressing model provides "
                             "side and an APPROXIMATE rib level; exact rib and along-rib position are NOT established "
                             "(along-rib s is exploratory). The figure shows the predicted rib in bold and adjacent "
                             "ribs as context; no lesion point is placed. Measured performance lives in the evaluation "
                             "manifest, not here. NOT a patient-specific reconstruction and NOT distinct fracture sites.",
               "along_rib_s": "recorded per detection as s_predicted_exploratory; not visualized, not a supported capability",
               "pattern_derivation": "DISCRETE addresses only (side, approximate rib level, consecutive levels, "
                                     "bilateral). No region/clustering/multisite/flail claims."}
        (cdir / "reconstruction.json").write_text(json.dumps(rec, indent=2))
        index.append({"case_id": cid, "n_detections": len(detections), "scale_mode": scale_note,
                      "cues": cues, "quality_flags": quality_flags, "png": png_ok})
        print(f"  {cid}: {len(detections)} detections | rib levels {pattern['predicted_rib_levels_by_side']} | "
              f"run={pattern['longest_consecutive_run']} bilat={pattern['bilateral']}", flush=True)
    # input manifest + hashes + full provenance, then atomic promotion of the immutable output dir
    try:
        import subprocess
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        git = None
    run_manifest = {"schema_version": "ribassist-rib-level-context-run-2", "git_commit": git,
                    "atlas_dir": str(a.atlas), "atlas_sha256": atlas["manifest"]["atlas_sha256"],
                    "candidates": str(a.candidates), "candidates_sha256": sha256_file(a.candidates),
                    "projections": (str(a.projections) if a.projections else None),
                    "projections_sha256": (sha256_file(a.projections) if a.projections else None),
                    "scale_mode": a.scale_mode, "reference_dims_mm": REF_DIMS,
                    "render": "predicted rib level in bold with adjacent-rib context; no along-rib point placement",
                    "silhouette_threshold": 0.12, "cli": {k: str(v) for k, v in vars(a).items()},
                    "n_cases": len(index)}
    (work / "reconstruction_index.json").write_text(json.dumps(index, indent=2))
    (work / "reconstruction_run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    work.rename(a.out)
    print(f"\nwrote {len(index)} reconstruction(s) to {a.out}/ (immutable; input manifest + hashes recorded).")
    if index and not index[0].get("png"):
        print("NOTE: PNG not written (use --png with kaleido for static export); HTML + JSON are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
