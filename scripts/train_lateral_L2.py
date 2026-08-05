#!/usr/bin/env python3
"""STAGE L2 — LATERAL-SPECIFIC RETRAINING (hard-negative mining + positive-branch strengthening).

The L0 -> L1 -> D2 program closed the pair-scoring correspondence family on the current detections: even on the
best-recalibrated field (lateral nms3/floor0.06) both deterministic (D1) and learned (D2) correspondence stay at
0% operational recall@10 at <=1 false-3D/case. Two lateral-detector-quality constraints bind: (1) AVAILABILITY —
the lateral head's heatmap maxes at ~0.091 (AP ~0.176), so ~27% of fractures never produce a compatible lateral
peak; (2) REAL-VS-REAL SEPARABILITY — same- and cross-fracture pairs are geometrically identical (|dSI| medians
equal) and only weakly separable by appearance (pos-vs-cross AUROC 0.66). Both are lateral-representation
problems. L2 retrains ONLY the lateral head to attack both at once.

Two coupled interventions (per the review):
  * HARD-NEGATIVE MINING (periodic re-mining). Every --remine-every epochs, the CURRENT lateral head is run on
    the training cases, peaks are extracted with the standing policy (nms3/floor0.06), the peaks NOT compatible
    with any GT footprint (the retained false peaks — in-support spurious maxima, boundary/background having
    already been calibrated away) are located, and a per-pixel NEGATIVE-emphasis weight map is splatted at those
    sites. The focal negative branch is up-weighted there, so training pushes those specific false responses
    down. Re-mining adapts the emphasis as the head improves (curriculum-like). The FIRST mine (epoch 0, from
    the warm-started champion head) is exactly "the false peaks latN3_f060 retains".
  * POSITIVE-BRANCH STRENGTHENING. Hard-negative mining alone risks compressing amplitude further and LOWERING
    availability. To lift the 0.091 ceiling we raise the focal positive weight (--pos-weight), train longer, and
    optionally add the soft-Dice region term (--dice-weight). L3's factorial separates the two effects.

The AP head, protocol, splits, and target heatmaps are FROZEN. The lateral checkpoint is selected on a
TRAINING-only criterion (final epoch by default; NO validation-based checkpoint pick) so the detector-validation
cohort stays untouched for the downstream D0->D1->D2 operational read. Per-epoch validation AMPLITUDE / peak /
compatible-recall numbers are printed as DIAGNOSTICS ONLY (never used to select).

The retrained head is frozen into a new detector-run directory that mirrors the champion (AP checkpoint + run
record copied; lateral checkpoint + its sha256 replaced), so the existing extraction-policy D0/D1/D2 pipeline
consumes it UNCHANGED. NOTE: the inherited fusion operating threshold + unmatched-lateral gate in the copied
record are STALE for the new lateral head (they were calibrated on the old one); they are required by the record
schema but are NOT exercised by the extraction-policy correspondence path. Re-score with evaluate_detector.py
before ever deploying the fusion path.

DIAGNOSTIC STATUS: development on the detector-validation split (biased; sealed test first confirmatory).
SUCCESS CRITERION (operational): a MATERIAL increase in out-of-fold recall@10 at <=1 false-3D/case through the
unchanged D0->D1 (and D2) scoreboard under the nms3/floor0.06 policy — NOT a better lateral FROC/AUROC or fewer
peaks. Evaluate with run_L1_correspondence_sweep.sh pointed at --detector-run <this L2 run>.

Usage (run from the RibAssist 3D ROOT):
  python scripts/train_lateral_L2.py \
      --champion-run outputs/detector_dev_scratch_c32_both_gated \
      --data outputs/det_out_v2/det_dev.npz \
      --epochs 60 --pos-weight 3.0 --dice-weight 0.0 \
      --remine-every 10 --hnm-weight 3.0 --hnm-sigma 4.0 \
      --lat-nms 3 --lat-floor 0.06 --init champion \
      --out-run outputs/detector_L2_lateral_hnm
"""
from __future__ import annotations
import argparse, json, shutil, sys
from copy import deepcopy
from pathlib import Path
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import train_detector as T
    _OK = True
except Exception as _e:  # noqa: BLE001
    _OK = False; _IMPORT_ERR = _e


class L2Views(Dataset):
    """Lateral training views + a MUTABLE per-case negative-emphasis weight map. Peak-preserving integer
    shift is applied identically to image / heatmap / weight so the emphasis stays aligned to the target;
    intensity jitter hits the image only. NO horizontal flip (the lateral horizontal axis is anterior-
    posterior; a flip is not an anatomically valid augmentation). The weight array is held by REFERENCE so
    periodic re-mining between epochs is picked up (DataLoader num_workers=0, as in train_detector.Views)."""
    def __init__(self, img, hm, negw, shift_px=8, jitter=True, seed=0):
        self.img = img; self.hm = hm; self.negw = negw            # negw: (N,H,W) mutable, updated by re-mining
        self.shift_px = int(shift_px); self.jitter = bool(jitter)
        self.rng = np.random.RandomState(seed + 777)
    def __len__(self): return len(self.img)
    def __getitem__(self, i):
        img, hm, w = self.img[i], self.hm[i], self.negw[i]
        if self.shift_px > 0:
            dy = int(self.rng.randint(-self.shift_px, self.shift_px + 1))
            dx = int(self.rng.randint(-self.shift_px, self.shift_px + 1))
            if dy or dx:
                img = T._shift(img, dy, dx, 0.0); hm = T._shift(hm, dy, dx, 0.0); w = T._shift(w, dy, dx, 1.0)
        if self.jitter:
            img = np.clip(img * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.02, 0.02), 0.0, 1.0)
        return (torch.from_numpy(np.ascontiguousarray(img.astype(np.float32)))[None],
                torch.from_numpy(np.ascontiguousarray(hm.astype(np.float32)))[None],
                torch.from_numpy(np.ascontiguousarray(w.astype(np.float32)))[None])


def focal_hardneg(pred, gt, negw, pos_weight=1.0, a=2.0, b=4.0):
    """Penalty-reduced focal (train_detector.focal) with a per-pixel NEGATIVE weight map. negw==1 everywhere
    reproduces T.focal exactly; mined false-peak sites carry negw>1 to up-weight their negative gradient."""
    pred = pred.clamp(1e-6, 1 - 1e-6); pos = gt.eq(1.0).float()
    pos_loss = pos_weight * (torch.log(pred) * (1 - pred) ** a * pos).sum()
    neg_loss = (torch.log(1 - pred) * pred ** a * (1 - gt) ** b * (1 - pos) * negw).sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


def mine_false_peaks(net, imgs, foots, dev, nms, floor, radius, hnm_weight, sigma):
    """Return a fresh (N,H,W) negative-emphasis map: 1.0 everywhere + a Gaussian bump of height hnm_weight at
    every extracted peak that is NOT within `radius` of any GT footprint (a retained false peak)."""
    N, H, W = imgs.shape
    negw = np.ones((N, H, W), np.float32)
    half = int(np.ceil(3 * sigma)); yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    bump = hnm_weight * np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma)).astype(np.float32)
    net.eval(); n_fp = 0
    with torch.no_grad():
        for i in range(N):
            hm = net(torch.from_numpy(imgs[i][None, None].astype(np.float32)).to(dev))[0, 0]
            pk = T.peaks_from_hm(hm, radius=nms, thresh=floor)
            for r in range(len(pk)):
                if foots[i] and min(T._min_dist(pk[r, :2], f) for f in foots[i]) <= radius:
                    continue                                      # compatible -> not a false peak
                n_fp += 1
                cy, cx = int(round(pk[r, 0])), int(round(pk[r, 1]))
                y0, y1 = max(0, cy - half), min(H, cy + half + 1); x0, x1 = max(0, cx - half), min(W, cx + half + 1)
                by0, bx0 = y0 - (cy - half), x0 - (cx - half)
                negw[i, y0:y1, x0:x1] = np.maximum(negw[i, y0:y1, x0:x1],
                                                   1.0 + bump[by0:by0 + (y1 - y0), bx0:bx0 + (x1 - x0)])
    return negw, n_fp


def val_diagnostics(net, imgs, foots, dev, nms, floor, radius):
    """Diagnostic ONLY (never used for selection): amplitude ceiling + availability under the standing policy."""
    net.eval(); maxes = []; npk = []; hit = 0; ngt = 0
    with torch.no_grad():
        for i in range(len(imgs)):
            hm = net(torch.from_numpy(imgs[i][None, None].astype(np.float32)).to(dev))[0, 0]
            maxes.append(float(hm.max().cpu())); pk = T.peaks_from_hm(hm, radius=nms, thresh=floor); npk.append(len(pk))
            for f in foots[i]:
                ngt += 1
                if len(pk) and min(T._min_dist(pk[r, :2], f) for r in range(len(pk))) <= radius: hit += 1
    return {"max_hm_median": round(float(np.median(maxes)), 4), "peaks_per_case": round(float(np.mean(npk)), 1),
            "compat_recall": round(hit / ngt, 4) if ngt else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion-run", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-run", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=3.0, help="focal positive-branch weight (>1 lifts amplitude)")
    ap.add_argument("--dice-weight", type=float, default=0.0, help="optional soft-Dice term added to focal")
    ap.add_argument("--remine-every", type=int, default=10, help="re-mine false peaks every K epochs (0 = mine once)")
    ap.add_argument("--hnm-weight", type=float, default=3.0, help="height of the negative-emphasis bump at false peaks")
    ap.add_argument("--hnm-sigma", type=float, default=4.0, help="Gaussian sigma (px) of the emphasis bump")
    ap.add_argument("--lat-nms", type=int, default=3); ap.add_argument("--lat-floor", type=float, default=0.06)
    ap.add_argument("--init", choices=["champion", "scratch"], default="champion")
    ap.add_argument("--shift-px", type=int, default=8); ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not _OK:
        print(f"missing deps ({_IMPORT_ERR}); pip install torch numpy", file=sys.stderr); return 1
    if a.out_run.exists(): raise FileExistsError(f"{a.out_run} exists; use a new --out-run.")
    if a.pos_weight <= 0 or a.lat_nms < 1 or not (0 < a.lat_floor < 1): raise ValueError("bad hyperparameters")
    dev = T.device(); torch.manual_seed(a.seed); np.random.seed(a.seed)

    # ---- provenance: champion record + data hash (fail closed) ----
    rec = json.loads((a.champion_run / "detector_dev_run.json").read_text())
    data_sha = T.sha256_file(a.data)
    if data_sha != rec.get("det_dev_sha256"): raise ValueError("--data hash != champion det_dev_sha256")
    arch = T.arch_from_record(rec)
    for v in ("ap", "lat"):
        if not (a.champion_run / f"detector_{v}.pt").exists(): raise FileNotFoundError(f"missing champion detector_{v}.pt")

    d = np.load(a.data, allow_pickle=False)
    cases = [str(c) for c in d["case"]]; cidx = {c: i for i, c in enumerate(cases)}
    split = rec["split"]; val_ids = [str(c) for c in split["val_case_ids"] if str(c) in cidx]
    tr_ids = [str(c) for c in split.get("train_case_ids", [])] or [c for c in cases if c not in set(val_ids)]
    tr_ids = [c for c in tr_ids if c in cidx]
    tr_idx = np.array([cidx[c] for c in tr_ids], int); va_idx = np.array([cidx[c] for c in val_ids], int)
    lat_foot = T.group_instances(d, "lat")
    tr_foots = [lat_foot[i] for i in tr_idx]; va_foots = [lat_foot[i] for i in va_idx]
    tr_imgs = d["lat"][tr_idx].astype(np.float32); tr_hm = d["lat_hm"][tr_idx].astype(np.float32)
    va_imgs = d["lat"][va_idx].astype(np.float32)
    RADIUS = T.MATCH_RADIUS_PX

    net = T.build_detector(arch, pretrained=False).to(dev)
    if a.init == "champion":
        net.load_state_dict(torch.load(a.champion_run / "detector_lat.pt", map_location=dev))
    opt = torch.optim.Adam(net.parameters(), a.lr, weight_decay=a.weight_decay)

    print(f"L2 lateral retrain: {len(tr_idx)} train / {len(va_idx)} val cases | init {a.init} | arch {arch.get('kind')} "
          f"base_ch {arch.get('base_ch')} | pos_weight {a.pos_weight} dice {a.dice_weight} | mine nms{a.lat_nms}/"
          f"floor{a.lat_floor} every {a.remine_every} (bump {a.hnm_weight}@sigma{a.hnm_sigma}) | {a.epochs} epochs on {dev}",
          flush=True)

    negw = np.ones((len(tr_idx), T.PROTOCOL_SIZE, T.PROTOCOL_SIZE), np.float32)
    ds = L2Views(tr_imgs, tr_hm, negw, shift_px=a.shift_px, jitter=not a.no_jitter, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0)
    history = []

    for ep in range(a.epochs):
        if a.remine_every == 0:
            if ep == 0:
                nw, n_fp = mine_false_peaks(net, tr_imgs, tr_foots, dev, a.lat_nms, a.lat_floor, RADIUS, a.hnm_weight, a.hnm_sigma)
                negw[:] = nw; print(f"  [mine ep0] {n_fp} false peaks emphasized (once)", flush=True)
        elif ep % a.remine_every == 0:
            nw, n_fp = mine_false_peaks(net, tr_imgs, tr_foots, dev, a.lat_nms, a.lat_floor, RADIUS, a.hnm_weight, a.hnm_sigma)
            negw[:] = nw; print(f"  [mine ep{ep}] {n_fp} retained false peaks emphasized ({n_fp/len(tr_idx):.1f}/case)", flush=True)

        net.train(); ep_loss = 0.0; nb = 0
        for img, hm, w in dl:
            opt.zero_grad()
            pred = net(img.to(dev))
            loss = focal_hardneg(pred, hm.to(dev), w.to(dev), pos_weight=a.pos_weight)
            if a.dice_weight > 0.0: loss = loss + a.dice_weight * T.soft_dice_loss(pred, hm.to(dev))
            loss.backward(); opt.step(); ep_loss += float(loss.detach().cpu()); nb += 1

        diag = val_diagnostics(net, va_imgs, va_foots, dev, a.lat_nms, a.lat_floor, RADIUS)   # DIAGNOSTIC ONLY
        rec_ep = {"epoch": ep + 1, "mean_loss": round(ep_loss / max(nb, 1), 4), **{f"val_{k}": v for k, v in diag.items()}}
        history.append(rec_ep)
        print(f"  [lat] ep {ep+1}/{a.epochs} loss {ep_loss/max(nb,1):.4f} | val max_hm(med) {diag['max_hm_median']} "
              f"peaks/case {diag['peaks_per_case']} compat-recall {diag['compat_recall']}  (diagnostic)", flush=True)

    # ---- FREEZE: mirror champion run dir; replace lateral checkpoint + hash; inherit AP + record ----
    a.out_run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.champion_run / "detector_ap.pt", a.out_run / "detector_ap.pt")
    torch.save(net.state_dict(), a.out_run / "detector_lat.pt")
    lat_sha = T.sha256_file(a.out_run / "detector_lat.pt"); ap_sha = T.sha256_file(a.out_run / "detector_ap.pt")
    out_rec = deepcopy(rec)
    out_rec.setdefault("detector_sha256", {})["lat"] = lat_sha; out_rec["detector_sha256"]["ap"] = ap_sha
    out_rec["l2_provenance"] = {
        "stage": "L2 lateral-specific retraining (hard-negative mining + positive-branch strengthening)",
        "champion_run": str(a.champion_run), "init": a.init, "epochs": a.epochs, "pos_weight": a.pos_weight,
        "dice_weight": a.dice_weight, "remine_every": a.remine_every, "hnm_weight": a.hnm_weight,
        "hnm_sigma": a.hnm_sigma, "mine_policy": {"lat_nms": a.lat_nms, "lat_floor": a.lat_floor},
        "lr": a.lr, "weight_decay": a.weight_decay, "shift_px": a.shift_px, "jitter": not a.no_jitter, "seed": a.seed,
        "selection": "final epoch (training-only; NO val-based checkpoint selection)",
        "val_history_diagnostic_only": history,
        "WARNING_stale_fusion_operating_point": "AP head + fusion operating_threshold + unmatched_lateral_gate are "
            "INHERITED from the champion and are STALE for this retrained lateral head. They satisfy the record "
            "schema but are NOT used by the extraction-policy D0->D1->D2 correspondence path. Re-score with "
            "evaluate_detector.py before using the deployed fusion path."}
    (a.out_run / "detector_dev_run.json").write_text(json.dumps(out_rec, indent=2))

    print(f"\nL2 COMPLETE — froze retrained lateral head into {a.out_run}")
    print(f"  AP inherited (sha {ap_sha[:12]}..) | lateral retrained (sha {lat_sha[:12]}..) | data {data_sha[:12]}..")
    d0 = history[0]; dN = history[-1]
    print(f"  val amplitude(med) {d0['val_max_hm_median']} -> {dN['val_max_hm_median']} | compat-recall "
          f"{d0['val_compat_recall']} -> {dN['val_compat_recall']} | peaks/case {d0['val_peaks_per_case']} -> {dN['val_peaks_per_case']} (diagnostic)")
    print(f"  NEXT: evaluate operationally under the standing policy —")
    print(f"    edit run_L1_correspondence_sweep.sh DET={a.out_run}, or run D0/D1 with --lat-nms {a.lat_nms} --lat-floor {a.lat_floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
