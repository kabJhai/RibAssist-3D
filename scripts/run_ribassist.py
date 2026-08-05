#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""RibAssist 3D end-to-end INFERENCE glue: one case, image-only, through the frozen champion path.

    AP/lat projections (det_dev images ONLY — never CT/RibSeg/GT/labels)
        v  frozen detector weights (rebuilt via train_detector.arch_from_record/build_detector)
    per-view heatmaps -> peaks               (train_detector.peaks_from_hm — REUSED verbatim)
        v  gated biplanar fusion             (train_detector.build_case_candidates — REUSED verbatim)
    fused candidates filtered at the FROZEN fusion operating threshold
        v  PAIRED candidates use both real peak centers; single-view (ap_only/lat_only) are PRESERVED
           but NOT addressed (no fabricated address from a center-imputed crop)
    96x96 padded crops at the two real coords  (make_address_data_detframe.crop_at — REUSED verbatim)
        v  addressing model (no-position deployment checkpoint; crops only, zero position input)
    (side, rib 1..12, s) + detection confidence plus decomposed addressing score/probabilities  [addressed sites only]
        v
    addressed_candidates.{json,npz}  ->  reconstruct_3d.py  ->  build_trauma_summary.py  + integration figure

CRITICAL REUSE RULE: this script NEVER re-implements peak extraction, scoring, pairing, the unmatched
rule, or the lateral gate. It imports the SAME train_detector functions evaluate_detector.py calls and
rebuilds the detector from the SAME dev-run record, so the demo's candidate set is identical to the
evaluation path's op-point set by construction (verify_candidate_identity.py asserts this).

LEAKAGE RULE: the inference path reads ONLY d["ap"], d["lat"], d["ap_geo"], d["lat_geo"], d["case"] from
det_dev.npz — the projection images and their SI geometry, which ARE the detector's legitimate input.
It never reads GT centers (ap_ctr/lat_ctr/fp_*), rib-seg, or fracture labels, and loads no CT.

Usage (detector-identity smoke test, stops at addressed candidates — no atlas needed):
  python run_ribassist.py --detector-run outputs/detector_dev_scratch_c32_both_gated \
      --address-model outputs/addressing_model_nopos --data outputs/det_out_v2/det_dev.npz \
      --case RibFrac225 --out outputs/demo/RibFrac225

Full demo (adds 3D reconstruction + trauma summary + integration figure):
  python run_ribassist.py ... --atlas outputs/rib_atlas_v1
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

try:
    import torch
    import train_detector as T
    import train_address as TA
    from make_address_data_detframe import crop_at, image_spine_col
except Exception:  # noqa: BLE001
    torch = None

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent   # recorded dataset paths are relative to the repo root, not scripts/


# ----------------------------------------------------------------------------------------------------
# Frozen-config loading (mirrors evaluate_detector.py's reconstruction, so candidates cannot diverge)
# ----------------------------------------------------------------------------------------------------
def load_detector(run_dir, dev, lat_gate_override=None):
    """Rebuild the frozen detector EXACTLY as evaluate_detector.py does, and resolve the frozen fusion
    operating threshold + unmatched-lateral gate + si_tol from the SAME dev-run record. Fails loudly if
    the record lacks a scored fusion operating point (i.e. was not produced by evaluate_detector)."""
    rec = json.loads((run_dir / "detector_dev_run.json").read_text())
    arch = T.arch_from_record(rec)
    views = [v for v in ("ap", "lat") if (run_dir / f"detector_{v}.pt").exists()]
    if set(views) != {"ap", "lat"}:
        raise ValueError(f"fusion needs BOTH detector_ap.pt and detector_lat.pt; found {views} in {run_dir}")
    # provenance FAILS CLOSED: a frozen run rejects a missing detector_sha256 mapping or per-view hash
    hashes = rec.get("detector_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("detector_dev_run.json is missing required 'detector_sha256' mapping")
    for v in views:
        want = hashes.get(v)
        if not want:
            raise ValueError(f"detector_dev_run.json is missing required detector_sha256[{v!r}]")
        got = T.sha256_file(run_dir / f"detector_{v}.pt")
        if got != want:
            raise ValueError(f"detector_{v}.pt hash mismatch: {got[:12]}.. != {want[:12]}..")
    nets = {}
    for v in views:
        net = T.build_detector(arch, pretrained=False).to(dev)
        net.load_state_dict(torch.load(run_dir / f"detector_{v}.pt", map_location=dev)); net.eval(); nets[v] = net

    lp = rec.get("learning_procedure", {}); ep = rec.get("eval_params", {})
    si_tol = ep.get("si_tol")
    if si_tol is None: raise ValueError("dev-run eval_params has no si_tol; cannot reproduce fusion pairing.")
    op = lp.get("operating_threshold_per_condition", {}).get("fusion")
    if op is None:
        raise ValueError("dev-run has no learning_procedure.operating_threshold_per_condition['fusion']. "
                         "Score the run with evaluate_detector.py (which freezes the fusion operating point) first. "
                         f"present keys: {list(lp.get('operating_threshold_per_condition', {}).keys())}")
    gate_rec = lp.get("biplanar_fusion", {}).get("unmatched_lateral_score_gate")
    if lat_gate_override is not None:
        lat_gate = float(lat_gate_override)
    elif gate_rec is not None:
        lat_gate = float(gate_rec)
    else:
        raise ValueError("dev-run has no biplanar_fusion.unmatched_lateral_score_gate and no --lat-gate override; "
                         "refusing to guess the fusion gate (it changes the candidate set).")
    return nets, arch, float(si_tol), float(op), lat_gate, rec


def resolve_half(cfg, half_ds, research_half=None, allow_override=False):
    """Frozen crop geometry: the manifest half is authoritative. FAIL CLOSED if it is unavailable and no
    explicitly-flagged research override is given (identical policy in run_ribassist and the evaluator)."""
    if research_half is not None:
        if not allow_override:
            raise ValueError("Crop geometry is frozen. A half-window override requires an explicit research flag.")
        return int(research_half)
    if half_ds is None:
        raise ValueError("Addressing dataset manifest is unavailable "
                         f"(from recorded address_dataset={cfg.get('address_dataset')!r}); refusing to guess the "
                         "trained crop half-window.")
    return int(half_ds)


def verify_provenance(data_path, address_model_dir, cfg, det_rec):
    """FAIL-CLOSED cross-artifact provenance, shared by run_ribassist and eval_address_e2e so an evaluator
    can never score mixed artifacts while the inference path is locked. Requires (present AND matching):
    det_dev hash == addressing aligned_det_dev_sha256 == detector det_dev_sha256; addressing checkpoint hash
    == state_dict_sha256. (Detector weight hashes are already checked in load_detector.) Returns data_sha."""
    def require_hash(record, key, source):
        v = record.get(key)
        if not v: raise ValueError(f"{source} is missing required provenance field {key!r}")
        return v
    data_sha = T.sha256_file(data_path)
    exp_addr_data = require_hash(cfg, "aligned_det_dev_sha256", "addressing_model.json")
    if data_sha != exp_addr_data:
        raise ValueError(f"det_dev hash mismatch: current {data_sha[:12]}.. != addressing model {exp_addr_data[:12]}..")
    exp_addr_w = require_hash(cfg, "state_dict_sha256", "addressing_model.json")
    got_addr_w = T.sha256_file(Path(address_model_dir) / "addressing_model.pt")
    if got_addr_w != exp_addr_w:
        raise ValueError(f"addressing checkpoint hash mismatch: {got_addr_w[:12]}.. != {exp_addr_w[:12]}..")
    exp_det_data = require_hash(det_rec, "det_dev_sha256", "detector_dev_run.json")   # top-level in evaluate_detector output
    if data_sha != exp_det_data:
        raise ValueError(f"det_dev hash mismatch: current {data_sha[:12]}.. != detector run {exp_det_data[:12]}..")
    return data_sha


def load_addressing(model_dir):
    """Load the deployable addressing checkpoint + its config (views/use_pos/crop). Recover the crop
    HALF-window from the training dataset's manifest (crop geometry is part of the trained-model spec),
    resolving the recorded relative dataset path against cwd / repo-root / model dir. Returns half=None
    only when the manifest genuinely cannot be located; main() then errors unless --research-half is set."""
    cfg = json.loads((model_dir / "addressing_model.json").read_text())
    views, use_pos, crop = cfg["views"], bool(cfg["use_pos"]), int(cfg["crop"])
    half = None; man_found = None
    ds = cfg.get("address_dataset")
    if ds:
        man = Path(ds); man = man.with_name(man.stem + "_manifest.json")
        if man.is_absolute():
            cand = [man]
        else:
            cand = [Path.cwd() / man, PROJECT_ROOT / man, model_dir / man]
        man_found = next((p for p in cand if p.exists()), None)
        if man_found is not None:
            dm = json.loads(man_found.read_text())
            half = int(dm["half"]) if dm.get("half") is not None else None
            if dm.get("crop") is not None and int(dm["crop"]) != crop:
                raise ValueError(f"addressing crop {crop} != dataset manifest crop {dm['crop']}")
    return cfg, views, use_pos, crop, half, man_found


# ----------------------------------------------------------------------------------------------------
# Candidate generation (the identity-critical core; importable so verify_candidate_identity.py checks it)
# ----------------------------------------------------------------------------------------------------
def fused_candidates_at_op(nets, ap_img, lat_img, ap_geo, lat_geo, si_tol, lat_gate, op_threshold, dev):
    """Run the frozen detector on ONE case's AP/lat images and return the fusion candidate set FILTERED
    at the frozen operating threshold — using train_detector.peaks_from_hm + build_case_candidates
    verbatim. Returns (active_candidates, ap_peaks, lat_peaks). `active_candidates` are the SAME dicts
    build_case_candidates emits ({score, ap: idx|None, lat: idx|None}), score>=op_threshold."""
    with torch.no_grad():
        ap_hm = nets["ap"](torch.from_numpy(ap_img.astype(np.float32))[None, None].to(dev))[0, 0]
        lat_hm = nets["lat"](torch.from_numpy(lat_img.astype(np.float32))[None, None].to(dev))[0, 0]
        ap_peaks = T.peaks_from_hm(ap_hm); lat_peaks = T.peaks_from_hm(lat_hm)
    cands = T.build_case_candidates("fusion", ap_peaks, lat_peaks, ap_geo, lat_geo, si_tol, lat_gate)
    active = [c for c in cands if c["score"] >= op_threshold]
    active.sort(key=lambda c: -c["score"])   # rank for stable candidate numbering (does not change the set)
    return active, ap_peaks, lat_peaks


def enrich_candidate(cd, ap_peaks, lat_peaks):
    """Attach the REAL per-view peak center for every present view; the missing view stays None. No
    cross-view coordinate is fabricated: the missing-view column has no single-view correspondence
    (AP col = L/R, lat col = A/P are independent), so a center-imputed crop would be unlike any paired
    crop the addressing model trained on. Single-view detections are therefore preserved but NOT
    addressed downstream (address_status set in address_candidates). ap_rc/lat_rc are (row,col) or None."""
    ai, li = cd["ap"], cd["lat"]
    ap_rc = (float(ap_peaks[ai, 0]), float(ap_peaks[ai, 1])) if ai is not None else None
    lat_rc = (float(lat_peaks[li, 0]), float(lat_peaks[li, 1])) if li is not None else None
    source = "paired" if (ai is not None and li is not None) else ("ap_only" if li is None else "lat_only")
    return {"score": float(cd["score"]), "ap_rc": ap_rc, "lat_rc": lat_rc, "source": source}


# ----------------------------------------------------------------------------------------------------
# Addressing
# ----------------------------------------------------------------------------------------------------
def needed_views(model_views):
    """The view crops the addressing model actually CONSUMES. 'ap'/'lat' -> that one view; 'both' -> both."""
    return ("ap", "lat") if model_views == "both" else (model_views,)


def address_candidates(cands, ap_img, lat_img, net, views, use_pos, crop, half, dev):
    """VIEW-AWARE addressing. A candidate is addressed ONLY if it has a REAL detector peak for every view
    the model consumes (needed_views); its crops come from those real peaks alone. No view the model does
    not consume is cropped or fed, and NO missing view is ever fabricated. This closes the SI-paired-lateral
    train/inference mismatch: detector fusion pairs an AP peak to whatever lateral peak is nearest in SI,
    which is frequently an off-anatomy edge artifact, so feeding that lateral crop to a biplanar model
    addresses a real AP fracture against an unrelated lateral region. With the deployed AP-only model,
    paired AND ap_only detections (both carry a real AP peak) are addressed from the AP crop; lateral-only
    detections are retained but NOT addressed; the lateral image is CONTEXT, never an addressing input.

    Candidates lacking a required-view peak are preserved with address_status='not_addressed' + a reason.
    Position inputs are zero for the no-position model; the unused view stream (Net.<view> is None) gets a
    zero tensor it ignores."""
    need = needed_views(views); S = ap_img.shape[-1]
    spine_col = image_spine_col(ap_img) if use_pos else None   # midline proxy ONLY needed by a position model
    out = []
    for cd in cands:
        ap_rc, lat_rc = cd["ap_rc"], cd["lat_rc"]; rc = {"ap": ap_rc, "lat": lat_rc}
        base = {"source": cd["source"], "detection_confidence": round(float(cd["score"]), 4),
                "conf": round(float(cd["score"]), 4),
                "ap_xy": ([round(ap_rc[0], 2), round(ap_rc[1], 2)] if ap_rc else None),
                "lat_xy": ([round(lat_rc[0], 2), round(lat_rc[1], 2)] if lat_rc else None)}
        missing = [v for v in need if rc[v] is None]
        if missing:   # cannot address without a real peak in every consumed view; NEVER fabricate one
            out.append({**base, "address_status": "not_addressed",
                        "not_addressed_reason": f"no {'/'.join(missing)} peak for the {views} addressing model",
                        "side": None, "rib": None, "s": None, "address_score": None,
                        "side_probability": None, "predicted_side_probability": None, "rib_probability": None})
            continue
        if "ap" in need:
            ar, ac = ap_rc
            apc = torch.from_numpy(crop_at(ap_img, ar, ac, half, crop)[0].astype(np.float32))[None, None].to(dev)
            pap = (torch.tensor([[ar / S, (ac - spine_col) / S]], dtype=torch.float32, device=dev) if use_pos
                   else torch.zeros((1, 2), dtype=torch.float32, device=dev))
        else:
            apc = torch.zeros((1, 1, crop, crop), device=dev); pap = torch.zeros((1, 2), device=dev)
        if "lat" in need:
            lr, lc = lat_rc
            latc = torch.from_numpy(crop_at(lat_img, lr, lc, half, crop)[0].astype(np.float32))[None, None].to(dev)
            plat = (torch.tensor([[lr / S, lc / S]], dtype=torch.float32, device=dev) if use_pos
                    else torch.zeros((1, 2), dtype=torch.float32, device=dev))
        else:
            latc = torch.zeros((1, 1, crop, crop), device=dev); plat = torch.zeros((1, 2), device=dev)
        with torch.no_grad():
            side_logit, rib_logits, s_pred = net(apc, latc, pap, plat)
            side_prob = float(torch.sigmoid(side_logit)[0])
            rib_probs = torch.softmax(rib_logits, 1)[0].cpu().numpy()
            s_val = float(s_pred[0])   # Net.forward already sigmoids s
        rib = int(rib_probs.argmax()) + 1; rib_prob = float(rib_probs[rib - 1])
        side = "R" if side_prob >= 0.5 else "L"; side_conf = max(side_prob, 1.0 - side_prob)
        s = float(np.clip(s_val, 0.0, 1.0))
        # RANKING SCORE, not a calibrated P(address correct): product of the chosen-side prob and the
        # top rib softmax prob. Kept separate from the detector score, and named 'score' not 'confidence'.
        address_score = float(side_conf * rib_prob)
        out.append({**base, "address_status": "addressed", "addressed_from_views": list(need),
                    "side": side, "rib": rib, "s": round(s, 4),
                    "address_score": round(address_score, 4), "side_probability": round(side_prob, 4),
                    "predicted_side_probability": round(side_conf, 4), "rib_probability": round(rib_prob, 4)})
    return out


# ----------------------------------------------------------------------------------------------------
# Integration figure
# ----------------------------------------------------------------------------------------------------
def _place_label(marker_rc, placed, r0=11.0, min_sep=15.0):
    """Deterministic collision-aware label offset: try 8 directions at increasing radius until the label
    clears every already-placed label (>= min_sep apart), so overlapping detections get legible numbers.
    Purely cosmetic — never moves a marker, only its number. Returns (lx, ly) and appends to `placed`."""
    import math
    r, c = marker_rc
    for radius in (r0, r0 * 1.9, r0 * 2.9, r0 * 4.0):
        for ang in (-45, 0, 45, 90, -90, 135, 180, -135):   # lower-right first, then around the clock
            a = math.radians(ang); lx = c + radius * math.cos(a); ly = r + radius * math.sin(a)
            if all((lx - px) ** 2 + (ly - py) ** 2 >= min_sep ** 2 for px, py in placed):
                placed.append((lx, ly)); return lx, ly
    lx, ly = c + r0, r + r0 + len(placed) * min_sep   # fallback: stack
    placed.append((lx, ly)); return lx, ly


def integration_figure(case, ap_img, lat_img, addressed, need, out_png, caveats=None):
    """View-aware overlay. A panel whose view the addressing model CONSUMES (in `need`) marks addressed
    detections (solid red) and detections not addressable in that view (dashed yellow), each at its REAL
    peak. A panel whose view is NOT consumed is CONTEXT: its detector peaks are drawn faint gray with no
    candidate number and the title says '(context - not used for addressing)', so the lateral image is
    never presented as a confirmed correspondence."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as e:  # noqa: BLE001
        print(f"[figure skip] {e}", file=sys.stderr); return False
    fig = plt.figure(figsize=(13, 6))
    axP = fig.add_subplot(1, 3, 1); axL = fig.add_subplot(1, 3, 2); axT = fig.add_subplot(1, 3, 3)
    S = ap_img.shape[-1]
    for ax, img, view, xy_key in ((axP, ap_img, "ap", "ap_xy"), (axL, lat_img, "lat", "lat_xy")):
        used = view in need
        # AP is displayed in RADIOLOGICAL convention (patient RIGHT on the viewer's left) to MATCH the 3D
        # atlas (which renders +x = patient-Right on the left); the AP projection column is +x = patient-Right,
        # so we mirror the DISPLAY only (np.fliplr) and mirror marker columns to c' = (S-1)-c. The addressing
        # model reads crops from the UNFLIPPED array at unflipped coords, so this is display-only.
        radiological = (view == "ap")
        shown = np.fliplr(np.asarray(img, np.float32)) if radiological else np.asarray(img, np.float32)
        ax.imshow(shown, cmap="gray", origin="lower")
        title = ("AP  (radiological: R | L)" if view == "ap" else "lateral  (context - not used for addressing)")
        ax.set_title(title, fontsize=9); ax.axis("off")
        if view == "ap":   # explicit patient-side labels so the film and the 3D can be checked against each other
            ax.text(0.03, 0.97, "R", color="cyan", fontsize=13, weight="bold", ha="left", va="top", transform=ax.transAxes)
            ax.text(0.97, 0.97, "L", color="cyan", fontsize=13, weight="bold", ha="right", va="top", transform=ax.transAxes)
        if not used:
            # CONTEXT view: show the clean image only. The lateral detector peaks are unreliable
            # (SI-paired edge artifacts in the anterior air) and are NOT used for addressing, so drawing
            # them would imply a correspondence that does not exist. Intentionally no markers here.
            continue
        placed = []
        for k, a in enumerate(addressed, 1):
            rc = a.get(xy_key)
            if rc is None: continue   # no real peak in this view for this detection
            r, c = rc; cx = (S - 1) - c if radiological else c
            addressed_here = a["address_status"] == "addressed"
            ax.add_patch(Circle((cx, r), radius=7, fill=False, linewidth=2,
                                linestyle="-" if addressed_here else "--",
                                edgecolor="red" if addressed_here else "yellow"))
            lx, ly = _place_label((r, cx), placed)
            if (lx - cx) ** 2 + (ly - r) ** 2 > 16 ** 2:
                ax.plot([cx, lx], [r, ly], color="yellow", linewidth=0.6, alpha=0.7)
            ax.text(lx, ly, str(k), color="yellow", fontsize=9, weight="bold", ha="center", va="center")
    axT.axis("off")
    n_det = len(addressed); n_addr = sum(1 for a in addressed if a["address_status"] == "addressed")
    # group by SIDE + PREDICTED RIB LEVEL only — along-rib s is exploratory and is NOT shown in the primary
    # visual (it stays in the machine-readable audit). Multiple detections at one rib level are named, not hidden.
    from collections import defaultdict
    grp = defaultdict(list)
    for k, a in enumerate(addressed, 1):
        if a["address_status"] == "addressed": grp[(a["side"], a["rib"])].append(k)
    dup = {kk: v for kk, v in grp.items() if len(v) > 1}
    lines = [f"{case}", f"Detections: {n_det}", f"Addressed: {n_addr}", f"Not addressed: {n_det - n_addr}",
             f"Addressing consumes: {'+'.join(need)}", ""]
    CAP = 24
    for k, a in enumerate(addressed[:CAP], 1):
        if a["address_status"] == "addressed":
            lines.append(f"#{k}  predicted {a['side']}{a['rib']}")
            lines.append(f"     det {a['detection_confidence']:.2f} | adr-score {a['address_score']:.2f} | {a['source']}")
        else:
            lines.append(f"#{k}  (not addressed - {a['source']})")
            lines.append(f"     det {a['detection_confidence']:.2f} | {a.get('not_addressed_reason','')}")
    if len(addressed) > CAP: lines.append(f"... and {len(addressed) - CAP} more (see addressed_candidates.json)")
    if dup:
        lines.append(""); lines.append("Multiple detections at the same predicted rib level:")
        for (sd, rb), v in dup.items():
            lines.append(f"  {sd}{rb}: detections #{','.join(map(str, v))}")
    if caveats and caveats.get("s_note"):
        lines.append(""); lines.append(caveats["s_note"])
    axT.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, transform=axT.transAxes)
    fig.suptitle(f"RibAssist 3D integration — {case}   (AP radiological R|L; red solid = addressed; "
                 "yellow dashed = detected, not addressed; lateral = context, unmarked)", fontsize=9)
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector-run", type=Path, required=True)
    ap.add_argument("--address-model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="det_dev.npz (images used; GT never read)")
    ap.add_argument("--case", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, default=None, help="if given, also run reconstruct_3d + build_trauma_summary")
    ap.add_argument("--research-half", type=int, default=None,
                    help="RESEARCH ONLY: override the crop half-window (frozen crop geometry otherwise comes from "
                         "the addressing dataset manifest). Requires --research-allow-half-override.")
    ap.add_argument("--research-allow-half-override", action="store_true",
                    help="required to honor --research-half; crop geometry is part of the trained-model spec")
    ap.add_argument("--lat-gate", type=float, default=None, help="RESEARCH ONLY: deviate from the frozen gate")
    ap.add_argument("--research-allow-gate-override", action="store_true",
                    help="required to honor --lat-gate; without it the frozen recorded gate is enforced")
    a = ap.parse_args()
    if torch is None:
        print("pip install torch nibabel scipy matplotlib scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; demo outputs are versioned — use a new dir.")
    # ---- fix 5: the integration path is the FROZEN evaluated algorithm; a gate override forfeits that ----
    if a.lat_gate is not None and not a.research_allow_gate_override:
        raise ValueError("The integration path must use the recorded frozen lateral gate. Gate overrides are not "
                         "permitted (pass --research-allow-gate-override to intentionally leave the frozen algorithm).")
    dev = T.device()

    # ---- frozen configs ----
    gate_override = a.lat_gate if a.research_allow_gate_override else None
    nets, arch, si_tol, op_thr, lat_gate, det_rec = load_detector(a.detector_run, dev, gate_override)
    cfg, views, use_pos, crop, half_ds, man_found = load_addressing(a.address_model)
    half = resolve_half(cfg, half_ds, a.research_half, a.research_allow_half_override)
    if a.research_half is not None: man_found = None
    net = TA.Net(views, use_pos).to(dev)
    net.load_state_dict(torch.load(a.address_model / "addressing_model.pt", map_location=dev)); net.eval()

    # provenance FAILS CLOSED (shared with the evaluator so mixed artifacts cannot be scored)
    data_sha = verify_provenance(a.data, a.address_model, cfg, det_rec)

    print(f"device {dev} | detector arch={arch.get('kind')} base_ch={arch.get('base_ch')} | "
          f"fusion op_thr={op_thr:.4f} lat_gate={lat_gate:.4f} si_tol={si_tol} | "
          f"addressing views={views} use_pos={use_pos} crop={crop} half={half}"
          + (f" (manifest {man_found})" if man_found else " (RESEARCH --research-half override)"), flush=True)

    # ---- load ONLY the inference-legal arrays from det_dev.npz (no GT/labels) ----
    d = np.load(a.data, allow_pickle=False)
    case_to_idx = {str(c): i for i, c in enumerate(d["case"])}
    if a.case not in case_to_idx:
        raise KeyError(f"{a.case} not in {a.data}. Available e.g.: {list(case_to_idx)[:5]} ...")
    ci = case_to_idx[a.case]
    ap_img = d["ap"][ci].astype(np.float32); lat_img = d["lat"][ci].astype(np.float32)
    ap_geo = d["ap_geo"][ci]; lat_geo = d["lat_geo"][ci]
    if ap_img.shape[-1] != T.PROTOCOL_SIZE: raise ValueError(f"image {ap_img.shape} != protocol {T.PROTOCOL_SIZE}")

    # ---- detector -> fusion candidates at the frozen op point (REUSED functions) ----
    active, ap_peaks, lat_peaks = fused_candidates_at_op(
        nets, ap_img, lat_img, ap_geo, lat_geo, si_tol, lat_gate, op_thr, dev)
    enriched = [enrich_candidate(cd, ap_peaks, lat_peaks) for cd in active]
    src_counts = {s: sum(1 for e in enriched if e["source"] == s) for s in ("paired", "ap_only", "lat_only")}
    print(f"{a.case}: AP peaks {len(ap_peaks)} | lat peaks {len(lat_peaks)} | "
          f"fused candidates @op {len(active)}  (paired {src_counts['paired']}, "
          f"ap_only {src_counts['ap_only']}, lat_only {src_counts['lat_only']})", flush=True)

    # ---- addressing: view-aware. a candidate is addressed iff it has a real peak for EVERY view the
    #      model consumes (needed_views). AP-only model -> paired + ap_only addressed; lat_only retained. ----
    addressed = address_candidates(enriched, ap_img, lat_img, net, views, use_pos, crop, half, dev) if enriched else []
    sites = [x for x in addressed if x["address_status"] == "addressed"]   # ONLY these enter reconstruction
    need = needed_views(views)
    n_addressable = sum(1 for e in enriched if all(e[f"{v}_rc"] is not None for v in need))
    # ---- hard runtime accounting invariants (not just a manual eyeball) ----
    assert len(addressed) == len(active), f"detection accounting mismatch: {len(addressed)} != {len(active)}"
    assert len(sites) == n_addressable, f"addressed {len(sites)} != candidates with all {need} peaks {n_addressable}"
    for x in addressed:
        if x["address_status"] == "addressed":
            assert all((x["ap_xy"] if v == "ap" else x["lat_xy"]) is not None for v in need), x
            assert x["side"] in ("L", "R") and 1 <= x["rib"] <= 12 and 0.0 <= x["s"] <= 1.0
        else:
            assert x["side"] is None and x["rib"] is None and x["s"] is None and x["address_score"] is None

    # ---- honesty caveat on along-rib s: report the measured values, no arbitrary reliability threshold ----
    ism = cfg.get("inner_select_metrics", {}); base = cfg.get("trivial_baseline", {})
    s_model = ism.get("s_mae"); s_base = base.get("s_mae"); s_note = None
    if s_model is not None and s_base is not None:
        s_note = (f"Along-rib s is EXPLORATORY: dev inner-select s-MAE {s_model:.3f} vs median baseline "
                  f"{s_base:.3f}. Precise along-rib placement is not supported.")

    a.out.mkdir(parents=True, exist_ok=True)
    audit = {"case": a.case, "n_detections": len(addressed), "n_addressed": len(sites),
             "n_not_addressed": len(addressed) - len(sites),
             "source_counts": src_counts, "addressing_consumes_views": list(need),
             "along_rib_s": {"model_inner_select_s_mae": s_model, "median_baseline_s_mae": s_base,
                             "status": "exploratory — precise along-rib placement not supported", "note": s_note},
             "frozen_config": {"detector_run": str(a.detector_run), "fusion_op_threshold": round(op_thr, 4),
                               "unmatched_lateral_gate": round(lat_gate, 4), "si_tol": si_tol,
                               "address_model": str(a.address_model), "address_views": views, "use_pos": use_pos,
                               "crop": crop, "half": half, "det_dev_sha256": data_sha},
             "policy": "VIEW-AWARE: a detection is addressed only if it has a real detector peak in every view "
                       "the addressing model consumes; crops come from real peaks alone, no view is fabricated. "
                       "For the AP-only model, paired + ap_only detections are addressed from the AP crop and the "
                       "lateral image is CONTEXT (never an addressing input); lateral-only detections are retained "
                       "but not addressed. Coincident addresses are reported, never merged.",
             "leakage_note": "inference read only det_dev AP/lat images + SI geometry; no CT/RibSeg/GT/labels.",
             "candidates": addressed}
    (a.out / "addressed_candidates.json").write_text(json.dumps(audit, indent=2))
    # reconstruct_3d input (npz path: side 0/1, rib, s, case, conf; extra arrays ignored) — ONLY addressed sites,
    # 'conf' = detection score (kept distinct from the uncalibrated address_score).
    n = len(sites)
    np.savez_compressed(a.out / "addressed_candidates.npz",
                        side=np.array([0 if x["side"] == "L" else 1 for x in sites], np.int64),
                        rib=np.array([x["rib"] for x in sites], np.int64),
                        s=np.array([x["s"] for x in sites], np.float32),
                        case=np.array([a.case] * n if n else [], dtype="<U32"),
                        conf=np.array([x["conf"] for x in sites], np.float32),
                        address_score=np.array([x["address_score"] for x in sites], np.float32))
    fig_ok = integration_figure(a.case, ap_img, lat_img, addressed, need, a.out / "integration_overview.png",
                                caveats={"s_note": s_note})
    print(f"{a.case}: addressed {len(sites)} / detections {len(addressed)} "
          f"(not addressed {len(addressed) - len(sites)}; addressing consumes {list(need)}); "
          f"wrote addressed_candidates.json + .npz" + ("  + integration_overview.png" if fig_ok else ""))

    # ---- optional: reconstruction + trauma summary (unchanged downstream scripts) ----
    recon_html = None
    if a.atlas is not None and not sites:
        # No ADDRESSED sites to place (either no detections at all, or only single-view detections that
        # policy does not address). reconstruct_3d + build_trauma_summary would error on an empty set, so
        # skip them — the correct end-to-end behavior rather than a crash.
        why = ("no detections above the frozen fusion operating threshold" if not addressed
               else f"{len(addressed)} detection(s) retained but none addressable by the {views} model "
                    f"(need a real peak in every view: {list(need)})")
        (a.out / "index.html").write_text(
            f"<!doctype html><meta charset=utf-8><title>RibAssist 3D — {a.case}</title>"
            f"<body style='font:14px system-ui;margin:24px'><h1>RibAssist 3D — {a.case}</h1>"
            f"<p>No addressed sites (op={op_thr:.3f}, gate={lat_gate:.3f}): {why}. No 3D reconstruction.</p>")
        print(f"{a.case}: 0 addressed sites -> reconstruction skipped ({why}).")
    elif a.atlas is not None:
        recon_dir = a.out / "recon"; summ_dir = a.out / "summary"
        # pass ABSOLUTE paths and DO NOT override cwd: the child scripts must resolve --atlas / --candidates /
        # --out against the directory the user invoked run_ribassist from (repo root), NOT against scripts/.
        # (subprocess adds each script's own dir to sys.path, so `import train_detector` still works.)
        subprocess.run([sys.executable, str(HERE / "reconstruct_3d.py"), "--atlas", str(a.atlas.resolve()),
                        "--candidates", str((a.out / "addressed_candidates.npz").resolve()),
                        "--out", str(recon_dir.resolve()), "--scale-mode", "reference"], check=True)
        subprocess.run([sys.executable, str(HERE / "build_trauma_summary.py"),
                        "--recon-dir", str(recon_dir.resolve()), "--out", str(summ_dir.resolve())], check=True)
        rh = recon_dir / a.case / "reconstruction.html"
        recon_rel = f"recon/{a.case}/reconstruction.html" if rh.exists() else None
        if recon_rel is None:   # do NOT silently drop the interactive 3D artifact
            print(f"WARNING: {rh} was not written — the interactive 3D HTML is missing. reconstruct_3d needs plotly "
                  "(pip install plotly). reconstruction.json is present; rerun after installing to get the 3D view.",
                  file=sys.stderr)
        _write_index(a.out, a.case, fig_ok, recon_rel, summ_dir / "summaries.txt")
        print(f"wrote reconstruction + trauma summary; open {a.out/'index.html'}")
    return 0


def _write_index(out, case, fig_ok, recon_rel, summ_txt):
    """Single portfolio page: integration figure + link to the 3D reconstruction + the trauma summary text.
    recon_rel is a path RELATIVE to `out` (or None) so the link works regardless of absolute location. When
    reconstruction ran but the 3D HTML is absent, the page says so explicitly rather than omitting it silently."""
    summ = summ_txt.read_text() if summ_txt.exists() else "(summary unavailable)"
    rel = recon_rel
    parts = [f"<!doctype html><meta charset=utf-8><title>RibAssist 3D — {case}</title>",
             "<style>body{font:14px system-ui;margin:24px;max-width:1100px}pre{background:#f6f8fa;padding:14px;"
             "border-radius:8px;overflow:auto}img{max-width:100%;border:1px solid #ddd;border-radius:8px}"
             ".warn{color:#b23}a.btn{display:inline-block;padding:8px 14px;background:#0b62d6;color:#fff;"
             "border-radius:6px;text-decoration:none}</style>",
             f"<h1>RibAssist 3D — {case}</h1>"]
    if fig_ok: parts.append('<h2>Detections &amp; predicted addresses</h2><img src="integration_overview.png">')
    if rel:
        parts.append(f'<h2>Interactive canonical rib-level context</h2><p><a class=btn href="{rel}">Open</a></p>')
    else:
        parts.append('<h2>Interactive canonical rib-level context</h2><p class=warn>Interactive HTML not generated '
                     '(reconstruct_3d needs <code>plotly</code>). reconstruction.json is present under recon/.</p>')
    parts.append(f"<h2>Structured trauma summary</h2><pre>{summ}</pre>")
    (out / "index.html").write_text("\n".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
