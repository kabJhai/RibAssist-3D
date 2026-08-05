#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""RibAssist 3D detector (DEVELOPMENT harness) — per-view heatmap U-Nets over the CT-derived
orthographic AP + lateral projections, trained on det_dev.npz ONLY.

DEVELOPMENT ONLY, by construction:
  * loads det_dev.npz + det_manifest.json and NOTHING else (never det_test_*.npz);
  * VERIFIES the dev artifact's sha256 against the manifest before use;
  * splits the dev pool by a deterministic, label-blind hash WITHIN strata
    (train_part1-positive / validation-positive / validation-negative) so the 20 negatives are
    represented in both slices; the exact assigned case ids + composition are recorded;
  * checkpoint selection uses ONLY the dev-internal VAL slice; the sealed test is untouched.

FROZEN DATA+EVALUATION protocol (Detector Protocol v1, read from manifest & asserted):
resolution 256, Gaussian sigma 4 px, NMS radius 5 px, matching radius 8 px. This script fixes
the LEARNING procedure by dev experimentation and ALWAYS writes a non-authoritative
detector_dev_run.json (weights + eval). It NEVER freezes: the protocol pipeline is
train_detector.py -> evaluate_detector.py (re-score saved weights with current code) ->
freeze_detector.py (promote exact artifacts) -> eval_sealed_test.py (score once).

Evaluation conditions, reported with EXPLICIT (non-identical) denominators:
  * AP-only        — FP per AP image
  * lateral-only   — FP per lateral image
  * biplanar FUSION (CANDIDATE deployed condition) — FP per image PAIR.
  * paired-confirmed (secondary) — FP per image pair; an instance counts only if BOTH views hit.
The primary deployed condition is NOT assumed to be biplanar; it is chosen by a PRE-DECLARED rule
(see PROJECT_PLAN) after convergence, comparing fusion vs AP on the stated FROC/AUPRC/recall metrics.

DEPLOYED FUSION ALGORITHM (declared, and reproduced verbatim in eval_sealed_test.py): build the
candidate set ONCE from all peaks above the extraction floor — pair AP<->lat one-to-one by SI
geometry, assign each fused candidate a FIXED score max(view scores), retain unmatched single-view
peaks — then threshold that FIXED candidate list. This is a geometry-based UNION + de-duplication of
per-view candidates (a candidate matches if EITHER present view is within radius); it does NOT
require two-view corroboration. It gives a conventional monotonic ranked FROC. It is NOT
"threshold each view then pair". Cross-view geometry uses ONLY the stored SI coordinate — never CT.

Freezing is DECOUPLED from training: this script writes a dev run (weights + detector_dev_run.json);
promote the SELECTED artifacts with freeze_detector.py (copy + verify, no retraining).

Usage (pipeline):
  python train_detector.py    --data outputs/det_out_v2/det_dev.npz --views both --epochs 80 \
      --batch 8 --base-ch 16 --lr 0.001 --bootstrap 1000 --out outputs/detector_dev_e80
  python evaluate_detector.py --dev-run outputs/detector_dev_e80 --data outputs/det_out_v2/det_dev.npz \
      --out outputs/detector_dev_e80_scored
  python freeze_detector.py   --dev-run outputs/detector_dev_e80_scored --primary <ap|fusion> \
      --overlay-qa-passed --overlay-qa-cases "RibFrac19,..." --out outputs/detector_frozen_v1
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from scipy.optimize import linear_sum_assignment
except Exception:  # noqa: BLE001
    torch = None

PROTOCOL_VERSION = "detector-protocol-v1"
PROTOCOL_SIZE = 256
PROTOCOL_SIGMA_PX = 4.0
NMS_RADIUS_PX = 5
MATCH_RADIUS_PX = 8
MIN_PEAK_SCORE = 0.05    # extraction floor: peaks below this never enter the FROC curve
MAX_PEAKS_PER_VIEW = 256  # per-view peak cap (can suppress FPs and, in dense cases, TPs)
FP_TARGETS = (0.5, 1.0, 2.0, 4.0)
PEAK_EXTRACTION = {"nms_radius_px": NMS_RADIUS_PX, "minimum_score": MIN_PEAK_SCORE,
                   "maximum_peaks_per_view": MAX_PEAKS_PER_VIEW}
UNIT = {"ap": "FP/AP-image", "lat": "FP/lat-image", "fusion": "FP/image-pair", "paired": "FP/image-pair"}


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# Data + provenance + stratified split
# --------------------------------------------------------------------------------------------
def load_dev(data_path):
    data_path = Path(data_path)
    if data_path.name != "det_dev.npz":
        raise ValueError(f"DEVELOPMENT harness: point --data at det_dev.npz, not {data_path.name}.")
    man_path = data_path.with_name("det_manifest.json")
    if not man_path.exists(): raise FileNotFoundError(f"Missing manifest next to data: {man_path}")
    man = json.loads(man_path.read_text())
    p = man.get("protocol", {})
    checks = {"version": (p.get("version"), PROTOCOL_VERSION), "resolution": (p.get("resolution"), PROTOCOL_SIZE),
              "sigma_px": (p.get("sigma_px"), PROTOCOL_SIGMA_PX), "nms_radius_px": (p.get("nms_radius_px"), NMS_RADIUS_PX),
              "matching_radius_px": (p.get("matching_radius_px"), MATCH_RADIUS_PX)}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad: raise ValueError(f"Manifest protocol mismatch (got vs expected): {bad}")
    want = man.get("artifact_sha256", {}).get("det_dev.npz")
    if not want: raise ValueError("Manifest has no artifact_sha256 for det_dev.npz; cannot verify provenance.")
    got = sha256_file(data_path)
    if got != want: raise ValueError(f"det_dev.npz sha256 mismatch: file {got[:12]}.. != manifest {want[:12]}..")
    d = np.load(data_path, allow_pickle=False)  # v2 arrays are numeric/string; no pickle needed
    if d["ap"].shape[-1] != PROTOCOL_SIZE or d["ap"].shape[-2] != PROTOCOL_SIZE:
        raise ValueError(f"Image resolution {d['ap'].shape[-2:]} != protocol {PROTOCOL_SIZE}")
    return d, man, got


def stratum_of(source, nfrac):
    grp = "train_part1" if str(source) == "train_part1" else "validation"
    return f"{grp}-{'positive' if int(nfrac) > 0 else 'negative'}"


def stratified_val_split(cases, sources, nfrac, val_pct, seed_tag="detector-dev-val"):
    """Deterministic, label-blind split WITHIN strata that GUARANTEES representation: every
    stratum of size>1 gets at least one train and at least one val case (singletons -> train).
    Members are ordered by md5(case) (label-blind); the first k = clip(round(n*val_pct/100),1,n-1)
    go to val. Returns (train_mask, val_mask, strata_report)."""
    strata = np.array([stratum_of(sources[i], nfrac[i]) for i in range(len(cases))])
    val = np.zeros(len(cases), bool)
    for s in sorted(set(strata)):
        members = np.where(strata == s)[0]; n = len(members)
        if n == 1: continue  # singleton stratum -> train (cannot be in both slices)
        hvals = [int(hashlib.md5(f"{seed_tag}:{cases[i]}".encode()).hexdigest(), 16) for i in members]
        order = members[np.argsort(hvals, kind="stable")]
        k = min(max(int(round(n * val_pct / 100.0)), 1), n - 1)  # >=1 val AND >=1 train
        val[order[:k]] = True
    rep = {}
    for s in sorted(set(strata)):
        m = strata == s
        rep[s] = {"n": int(m.sum()), "train": int((m & ~val).sum()), "val": int((m & val).sum())}
    return ~val, val, rep


def group_instances(d, view):
    pts, ptr, fpc = d[f"{view}_fp_pts"], d[f"{view}_fp_ptr"], d["fp_case"]
    per = [[] for _ in range(len(d["case"]))]
    for i in range(len(fpc)):
        per[int(fpc[i])].append(pts[ptr[i]:ptr[i + 1]].astype(np.int32))
    return per


def _shift(a, dy, dx, fill=0.0):
    """Integer-pixel translation with edge fill (NO interpolation). Unlike rotation/scaling this
    preserves the target heatmap's exact 1.0 peaks, so the penalty-reduced focal loss's positive
    branch (gt.eq(1.0)) stays intact. Applied identically to image and heatmap so they stay aligned."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys0, ys1 = max(0, dy), min(h, h + dy); xs0, xs1 = max(0, dx), min(w, w + dx)
    yt0, yt1 = max(0, -dy), min(h, h - dy); xt0, xt1 = max(0, -dx), min(w, w - dx)
    out[ys0:ys1, xs0:xs1] = a[yt0:yt1, xt0:xt1]
    return out


class Views(Dataset):
    def __init__(self, d, idx, view, augment=False, seed=0, strong=False, shift_px=8, rib=None, rib_has=None):
        self.img = d[view][idx].astype(np.float32); self.hm = d[f"{view}_hm"][idx].astype(np.float32)
        self.augment = augment; self.strong = strong; self.shift_px = int(shift_px)
        self.rib = rib[idx].astype(np.float32) if rib is not None else None   # rib-region target for THIS view
        self.rib_has = rib_has[idx].astype(bool) if rib_has is not None else None
        self.rng = np.random.RandomState(seed + 12345)
    def __len__(self): return len(self.img)
    def __getitem__(self, i):
        img, hm = self.img[i], self.hm[i]
        rib = self.rib[i] if self.rib is not None else np.zeros_like(hm)
        has = bool(self.rib_has[i]) if self.rib_has is not None else False
        if self.augment:  # hflip (image + heatmap + RIB together) + mild intensity jitter
            if self.rng.rand() < 0.5:
                img = img[:, ::-1].copy(); hm = hm[:, ::-1].copy(); rib = rib[:, ::-1].copy()
            img = np.clip(img * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.02, 0.02), 0.0, 1.0)
        if self.strong:  # PEAK-PRESERVING only: integer shift (image + heatmap + RIB) + gamma + noise
            if self.shift_px > 0:
                dy = int(self.rng.randint(-self.shift_px, self.shift_px + 1))
                dx = int(self.rng.randint(-self.shift_px, self.shift_px + 1))
                if dy or dx: img = _shift(img, dy, dx, 0.0); hm = _shift(hm, dy, dx, 0.0); rib = _shift(rib, dy, dx, 0.0)
            g = float(self.rng.uniform(0.8, 1.25)); img = np.clip(img, 0.0, 1.0) ** g
            img = np.clip(img + self.rng.normal(0.0, 0.02, img.shape).astype(np.float32), 0.0, 1.0)
        return (torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(np.ascontiguousarray(hm))[None],
                torch.from_numpy(np.ascontiguousarray(rib))[None], torch.tensor(has))


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def _norm(ch, mode="batch"):
    """BatchNorm (default) or GroupNorm. At batch size 8 the per-channel BN statistics are noisy,
    especially in the deep/low-resolution stages; GroupNorm is batch-independent. num_groups=8 where
    it divides the channel count, else the largest power-of-two divisor (never fails)."""
    if mode == "group":
        g = 8
        while ch % g != 0 and g > 1: g //= 2
        return nn.GroupNorm(g, ch)
    return nn.BatchNorm2d(ch)


def dconv(a, b, norm="batch"):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), _norm(b, norm), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), _norm(b, norm), nn.ReLU(inplace=True))


class ResDConv(nn.Module):
    """Residual double-conv block: conv-norm-relu-conv-norm + a 1x1-projected identity shortcut, then
    relu. Same in/out channels and receptive field as dconv(), so it is a drop-in for the scratch
    U-Net; the skip connection eases optimization (a controlled step up from the plain c32 U-Net)."""
    def __init__(self, a, b, norm="batch"):
        super().__init__()
        self.c1 = nn.Conv2d(a, b, 3, padding=1); self.b1 = _norm(b, norm)
        self.c2 = nn.Conv2d(b, b, 3, padding=1); self.b2 = _norm(b, norm)
        self.proj = nn.Conv2d(a, b, 1) if a != b else nn.Identity()
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True); y = self.b2(self.c2(y))
        return F.relu(y + self.proj(x), inplace=True)


class UNet(nn.Module):
    """Scratch per-view U-Net. rib_head=True adds a SECOND 1x1 output conv (outc_rib) on the SHARED
    decoder features — a multi-task auxiliary rib-region head. forward() ALWAYS returns just the
    fracture heatmap (sigmoid), so every eval/peak/sealed call is unchanged and the rib head is unused
    at inference; forward_multitask() additionally returns the rib LOGITS for the training loss. The
    rib head lives in the state_dict, so reconstruction must pass rib_head=True to load those weights."""
    def __init__(self, c=16, residual=False, norm="batch", rib_head=False):
        super().__init__()
        blk = (lambda a, b: ResDConv(a, b, norm)) if residual else (lambda a, b: dconv(a, b, norm))
        self.inc = blk(1, c); self.d1 = blk(c, 2 * c); self.d2 = blk(2 * c, 4 * c); self.d3 = blk(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.u3 = blk(8 * c + 4 * c, 4 * c); self.u2 = blk(4 * c + 2 * c, 2 * c); self.u1 = blk(2 * c + c, c)
        self.outc = nn.Conv2d(c, 1, 1)
        self.rib_head = bool(rib_head)
        if self.rib_head: self.outc_rib = nn.Conv2d(c, 1, 1)
    def _up(self, x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1)
    def _features(self, x):
        x0 = self.inc(x); x1 = self.d1(self.pool(x0)); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2))
        y = self.u3(self._up(x3, x2)); y = self.u2(self._up(y, x1)); y = self.u1(self._up(y, x0))
        return y
    def forward(self, x):
        return torch.sigmoid(self.outc(self._features(x)))   # fracture heatmap ONLY (inference/eval path unchanged)
    def forward_multitask(self, x):
        y = self._features(x); return torch.sigmoid(self.outc(y)), self.outc_rib(y)   # (fracture_hm, rib_LOGITS)


class SigmoidHead(nn.Module):
    """Wrap a logits-producing segmentation network (e.g. segmentation_models_pytorch Unet) so it
    emits a heatmap in [0,1], matching the rest of the pipeline (peaks_from_hm floor, focal target
    semantics). State-dict keys are prefixed 'net.'; save and reconstruct with this same wrapper.

    standardize=True z-scores each input image (per-sample, over spatial dims) BEFORE the encoder.
    ImageNet-pretrained BatchNorms expect standardized inputs; feeding raw [0,1] attenuation drives
    the pretrained encoder to a degenerate near-constant regime whose cheapest loss minimizer is to
    suppress every prediction -> heatmap collapses, sensitivity pins at 0. It is PARAMETER-FREE, so
    state_dict keys are unchanged, and — because it lives inside forward() — it is applied identically
    at training, checkpoint-selection, peak extraction, and the sealed test (recorded in model_arch)."""
    def __init__(self, net, standardize=False, eps=1e-5):
        super().__init__(); self.net = net; self.standardize = bool(standardize); self.eps = float(eps)
    def forward(self, x):
        if self.standardize:
            m = x.mean(dim=(-2, -1), keepdim=True); s = x.std(dim=(-2, -1), keepdim=True)
            x = (x - m) / (s + self.eps)
        return torch.sigmoid(self.net(x))


DECODER_CHANNELS = (256, 128, 64, 32, 16)   # smp Unet default; recorded for exact reconstruction


def build_detector(arch, pretrained=False):
    """Single source of truth for detector construction across train/evaluate/freeze/sealed.
    arch is a dict recorded in the dev run:
      {"kind":"scratch_unet","base_ch":16}                                  -> the v1 from-scratch U-Net
      {"kind":"smp","encoder":"resnet34","encoder_weights":"imagenet", ...} -> pretrained-encoder U-Net
    pretrained=True loads the encoder's ImageNet weights (TRAINING only; needs network on first use).
    pretrained=False builds an identically-shaped skeleton with random init (RECONSTRUCTION path:
    the trained state_dict is loaded on top, so encoder init is irrelevant and NO download happens)."""
    kind = arch.get("kind", "scratch_unet")
    if kind == "scratch_unet":
        return UNet(int(arch.get("base_ch", 16)), residual=bool(arch.get("residual", False)),
                    norm=arch.get("norm", "batch"), rib_head=bool(arch.get("rib_head", False)))
    if kind == "smp":
        import segmentation_models_pytorch as smp
        w = arch.get("encoder_weights") if pretrained else None
        m = smp.Unet(encoder_name=arch["encoder"], encoder_weights=w,
                     in_channels=int(arch.get("in_channels", 1)), classes=1,
                     decoder_channels=tuple(arch.get("decoder_channels", DECODER_CHANNELS)), activation=None)
        return SigmoidHead(m, standardize=bool(arch.get("standardize", False)))
    raise ValueError(f"unknown model kind {kind!r}")


def arch_from_record(rec):
    """Recover the model architecture from a dev/frozen record. Runs written before the pretrained
    backbone existed have no 'model_arch' -> they are the from-scratch U-Net with base_ch in eval_params."""
    a = rec.get("model_arch")
    if a: return a
    return {"kind": "scratch_unet", "base_ch": int(rec.get("eval_params", {}).get("base_ch", 16))}


def focal(pred, gt, a=2.0, b=4.0, pos_weight=1.0):
    pred = pred.clamp(1e-6, 1 - 1e-6); pos = gt.eq(1.0).float()
    pos_loss = pos_weight * (torch.log(pred) * (1 - pred) ** a * pos).sum()
    neg_loss = (torch.log(1 - pred) * pred ** a * (1 - gt) ** b * (1 - pos)).sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


def soft_dice_loss(pred, gt, eps=1.0):
    """Soft Dice on the continuous [0,1] prediction vs the Gaussian target — a region-overlap term
    that complements the point-focal loss (adds gradient where focal is near-saturated)."""
    num = 2.0 * (pred * gt).sum() + eps
    den = (pred * pred).sum() + (gt * gt).sum() + eps
    return 1.0 - num / den


def det_loss(pred, gt, dice_weight=0.0, pos_weight=1.0):
    L = focal(pred, gt, pos_weight=pos_weight)
    if dice_weight > 0.0: L = L + dice_weight * soft_dice_loss(pred, gt)
    return L


def rib_aux_loss(rib_logits, rib_target):
    """Auxiliary rib-region segmentation loss = BCE-with-logits + soft Dice. Takes LOGITS (numerically
    stable BCE); Dice is on the sigmoid. Caller passes only the has_rib subset of the batch."""
    bce = F.binary_cross_entropy_with_logits(rib_logits, rib_target)
    dice = soft_dice_loss(torch.sigmoid(rib_logits), rib_target)
    return bce + dice


# --------------------------------------------------------------------------------------------
# Peaks + matching
# --------------------------------------------------------------------------------------------
def peaks_from_hm(hm, radius=NMS_RADIUS_PX, thresh=MIN_PEAK_SCORE, cap=MAX_PEAKS_PER_VIEW):
    x = hm[None, None]; k = 2 * radius + 1
    mx = F.max_pool2d(x, k, stride=1, padding=radius)
    keep = (x == mx) & (x > thresh)
    ys, xs = torch.nonzero(keep[0, 0], as_tuple=True)
    sc = hm[ys, xs]
    if len(sc) > cap:
        top = torch.topk(sc, cap).indices; ys, xs, sc = ys[top], xs[top], sc[top]
    return np.stack([ys.cpu().numpy(), xs.cpu().numpy(), sc.cpu().numpy()], 1) if len(sc) else np.zeros((0, 3))


def _min_dist(peak_rc, fp):
    dd = fp.astype(np.float32) - peak_rc[None, :]
    return float(np.sqrt((dd ** 2).sum(1)).min())


def _assign(cost, BIG=1e6):
    if cost.size == 0: return 0
    ri, ci = linear_sum_assignment(cost)
    return int(sum(cost[r, c] < BIG for r, c in zip(ri, ci)))


def match_one_view(peaks, foots, radius=MATCH_RADIUS_PX):
    n_p, n_f = len(peaks), len(foots)
    if n_f == 0: return 0, n_p, 0
    if n_p == 0: return 0, 0, n_f
    BIG = 1e6; cost = np.full((n_p, n_f), BIG, np.float32)
    for i in range(n_p):
        for j in range(n_f):
            dm = _min_dist(peaks[i, :2], foots[j])
            if dm <= radius: cost[i, j] = dm
    tp = _assign(cost); return tp, n_p - tp, n_f - tp


def si_voxel(rows, geo):
    scale, pt = float(geo[0]), float(geo[1]); return (np.asarray(rows, np.float32) - pt) / max(scale, 1e-8)


def form_pairs(ap_peaks, lat_peaks, ap_geo, lat_geo, si_tol):
    """One-to-one AP<->lat pairing by |dSI| (geometry only). Returns pairs, unpaired_ap, unpaired_lat."""
    na, nl = len(ap_peaks), len(lat_peaks)
    if na == 0 or nl == 0: return [], list(range(na)), list(range(nl))
    si_a, si_l = si_voxel(ap_peaks[:, 0], ap_geo), si_voxel(lat_peaks[:, 0], lat_geo)
    BIG = 1e6; c = np.abs(si_a[:, None] - si_l[None, :]); c = np.where(c <= si_tol, c, BIG)
    ri, ci = linear_sum_assignment(c)
    pairs = [(int(r), int(cc)) for r, cc in zip(ri, ci) if c[r, cc] < BIG]
    pa, pl = {r for r, _ in pairs}, {cc for _, cc in pairs}
    return pairs, [i for i in range(na) if i not in pa], [j for j in range(nl) if j not in pl]


# NOTE: the deployed fusion candidate set is built by build_case_candidates() below (pair ONCE at
# the extraction floor, assign fixed candidate scores, then threshold the fixed list). The earlier
# threshold-BEFORE-pair helpers were removed; do not reintroduce "threshold then pair" semantics.


# --------------------------------------------------------------------------------------------
# FROC: dense grid from candidate scores; per-case precompute for cheap case-clustered bootstrap
# --------------------------------------------------------------------------------------------
def build_grid(all_scores, cap=300):
    """Dense FROC grid = sorted-descending unique candidate scores (exact operating vertices),
    capped, anchored at 1.0 (top) and MIN_PEAK_SCORE (bottom = 'all peaks surviving the
    extraction floor'; there is no meaningful threshold below the floor). Method is frozen;
    thresholds are data-derived on development, then reused verbatim on the sealed test."""
    u = np.unique(np.asarray(all_scores, np.float32)) if len(all_scores) else np.array([], np.float32)
    if len(u) > cap: u = u[np.linspace(0, len(u) - 1, cap).round().astype(int)]
    g = np.unique(np.concatenate([[1.0], u, [MIN_PEAK_SCORE]]))[::-1]
    return g.astype(np.float32)


def sens_curve(TP, FP, npos, n_img):
    """Summed-over-cases -> (fp_per_img[G], sensitivity[G])."""
    return FP / max(n_img, 1), TP / max(float(np.sum(npos)), 1.0)


def sens_at_targets(TP, FP, npos, n_img, targets=FP_TARGETS):
    fp_img, sens = sens_curve(TP, FP, npos, n_img); out = {}
    for t in targets:
        ok = fp_img <= t; out[t] = float(sens[ok].max()) if ok.any() else 0.0
    return out


def bootstrap_ci(TP, FP, npos, n_iter=1000, targets=FP_TARGETS, seed=0):
    rng = np.random.RandomState(seed); C = TP.shape[0]; acc = {t: [] for t in targets}
    for _ in range(n_iter):
        s = rng.randint(0, C, C)
        r = sens_at_targets(TP[s].sum(0), FP[s].sum(0), npos[s], C, targets)
        for t in targets: acc[t].append(r[t])
    return {t: (float(np.percentile(acc[t], 2.5)), float(np.percentile(acc[t], 97.5))) for t in targets}


# --------------------------------------------------------------------------------------------
# Peak cache + per-condition evaluation
# --------------------------------------------------------------------------------------------
def peak_cache(nets, d, idx, dev):
    cache = []
    for n in nets.values(): n.eval()
    with torch.no_grad():
        for i in idx:
            entry = {"ap_geo": d["ap_geo"][i], "lat_geo": d["lat_geo"][i]}
            for v in ("ap", "lat"):
                if v in nets:
                    hm = nets[v](torch.from_numpy(d[v][i].astype(np.float32))[None, None].to(dev))[0, 0]
                    entry[v] = peaks_from_hm(hm)
            cache.append(entry)
    return cache


def _peaks(entry, v): return entry.get(v, np.zeros((0, 3)))


def build_case_candidates(cond, ap, lat, ga, gl, si_tol, lat_gate=0.0):
    """Build the ranked candidate set ONCE (at the extraction floor), with FIXED scores, so the FROC
    is a conventional ranked curve: lowering the threshold only ADDS candidates, it never re-pairs and
    changes candidate identity. Fusion pair score = max(view scores); paired pair score = min (both
    views must be confident). Each candidate keeps refs to its AP/lat peak indices for matching.

    lat_gate (>=0, default 0.0 = current behavior, retain ALL unmatched lateral peaks): AP-anchored
    gated union. Paired candidates and unmatched-AP candidates are ALWAYS retained; an UNMATCHED
    lateral (single-view) candidate is retained only if its score >= lat_gate. Rationale: the lateral
    stream is weakly discriminative, so its unmatched single-view peaks add recall at high FP budgets
    but pollute the ranking near the clinically important low-FP region. Gating them is a development-
    only calibration of the (unfrozen) biplanar association layer; it NEVER touches AP/lat/paired."""
    if cond == "ap": return [{"score": float(p[2]), "ap": i, "lat": None} for i, p in enumerate(ap)]
    if cond == "lat": return [{"score": float(p[2]), "ap": None, "lat": i} for i, p in enumerate(lat)]
    pairs, ua, ul = form_pairs(ap, lat, ga, gl, si_tol)
    if cond == "fusion":
        c = [{"score": float(max(ap[a, 2], lat[l, 2])), "ap": int(a), "lat": int(l)} for a, l in pairs]
        c += [{"score": float(ap[a, 2]), "ap": int(a), "lat": None} for a in ua]
        c += [{"score": float(lat[l, 2]), "ap": None, "lat": int(l)} for l in ul if lat[l, 2] >= lat_gate]
        return c
    return [{"score": float(min(ap[a, 2], lat[l, 2])), "ap": int(a), "lat": int(l)} for a, l in pairs]  # paired


def match_candidates(cond, cands, ap, lat, afoot, lfoot, radius=MATCH_RADIUS_PX):
    """One-to-one Hungarian assignment of an ALREADY-threshold-filtered candidate list to instances.
    ap/lat/fusion: a candidate hits an instance if ANY of its present view-peaks is within radius;
    paired: BOTH view-peaks must be within radius. Adding candidates can only keep/raise the bipartite
    matching, so TP is monotonic as the threshold falls -> a conventional ranked FROC."""
    n_f = len(afoot)
    if not cands: return 0, 0, n_f
    if n_f == 0: return 0, len(cands), 0
    BIG = 1e6; cost = np.full((len(cands), n_f), BIG, np.float32)
    for i, cd in enumerate(cands):
        for j in range(n_f):
            if cond == "paired":
                if cd["ap"] is not None and cd["lat"] is not None:
                    da = _min_dist(ap[cd["ap"], :2], afoot[j]); dl = _min_dist(lat[cd["lat"], :2], lfoot[j])
                    if da <= radius and dl <= radius: cost[i, j] = da + dl
            else:
                ds = []
                if cd["ap"] is not None: ds.append(_min_dist(ap[cd["ap"], :2], afoot[j]))
                if cd["lat"] is not None: ds.append(_min_dist(lat[cd["lat"], :2], lfoot[j]))
                dm = min(ds) if ds else BIG
                if dm <= radius: cost[i, j] = dm
    tp = _assign(cost); return tp, len(cands) - tp, n_f - tp


def greedy_match(cands, ap, lat, afoot, lfoot, radius=MATCH_RADIUS_PX):
    """Threshold-free ranked greedy assignment for AUPRC: candidates by score desc, each to its
    nearest unmatched instance within radius. Returns [(score, is_tp), ...]."""
    used = set(); out = []
    for i in sorted(range(len(cands)), key=lambda k: -cands[k]["score"]):
        cd = cands[i]; best, bestd = None, 1e9
        for j in range(len(afoot)):
            if j in used: continue
            ds = []
            if cd["ap"] is not None: ds.append(_min_dist(ap[cd["ap"], :2], afoot[j]))
            if cd["lat"] is not None: ds.append(_min_dist(lat[cd["lat"], :2], lfoot[j]))
            dm = min(ds) if ds else 1e9
            if dm <= radius and dm < bestd: bestd, best = dm, j
        out.append((float(cd["score"]), best is not None))
        if best is not None: used.add(best)
    return out


def average_precision(pairs, npos_total):
    if not pairs or npos_total == 0: return 0.0
    pairs = sorted(pairs, key=lambda x: -x[0]); tp = fp = 0; prev = 0.0; ap = 0.0
    for _, istp in pairs:
        tp, fp = (tp + 1, fp) if istp else (tp, fp + 1)
        prec = tp / (tp + fp); rec = tp / npos_total; ap += prec * (rec - prev); prev = rec
    return float(ap)


def eval_condition(cond, cache, ap_g, lat_g, idx, si_tol, boot, seed, fixed_grid=None, op_threshold=None, lat_gate=0.0):
    """Conventional ranked FROC: each case's candidate set is built ONCE (fixed scores), and the
    threshold only FILTERS that fixed set, so the curve is monotonic (no threshold-dependent
    re-pairing). Plus operating-point metrics and threshold-free AUPRC. fixed_grid/op_threshold
    (frozen on dev) are reused verbatim on the sealed test. lat_gate calibrates the fusion association
    layer (unmatched-lateral score gate); default 0.0 reproduces the ungated union exactly."""
    C = len(idx); afoot = [ap_g[i] for i in idx]; lfoot = [lat_g[i] for i in idx]
    npos = np.array([len(afoot[c]) for c in range(C)], np.float32)
    cands = [build_case_candidates(cond, _peaks(cache[c], "ap"), _peaks(cache[c], "lat"),
                                   cache[c]["ap_geo"], cache[c]["lat_geo"], si_tol, lat_gate) for c in range(C)]
    sc = [cd["score"] for cl in cands for cd in cl]   # grid from FIXED candidate scores
    grid = np.asarray(fixed_grid, np.float32) if fixed_grid is not None else build_grid(sc)
    G = len(grid); TP = np.zeros((C, G)); FP = np.zeros((C, G))
    for c in range(C):
        ap, lat = _peaks(cache[c], "ap"), _peaks(cache[c], "lat")
        for j, t in enumerate(grid):
            active = [cd for cd in cands[c] if cd["score"] >= t]
            TP[c, j], FP[c, j], _ = match_candidates(cond, active, ap, lat, afoot[c], lfoot[c])
    pt = sens_at_targets(TP.sum(0), FP.sum(0), npos, C); ci = bootstrap_ci(TP, FP, npos, boot, seed=seed)
    fp_img, sens = sens_curve(TP.sum(0), FP.sum(0), npos, C)
    if op_threshold is None:  # frozen operating point = grid threshold nearest 1 FP/image (dev-derived)
        op_threshold = float(grid[int(np.argmin(np.abs(fp_img - 1.0)))])
    op_tp = np.zeros(C); op_fp = np.zeros(C)
    for c in range(C):
        ap, lat = _peaks(cache[c], "ap"), _peaks(cache[c], "lat")
        active = [cd for cd in cands[c] if cd["score"] >= op_threshold]
        op_tp[c], op_fp[c], _ = match_candidates(cond, active, ap, lat, afoot[c], lfoot[c])
    posmask = npos > 0
    case_recall = float((op_tp[posmask] >= 1).mean()) if posmask.any() else 0.0
    count_mae = float(np.abs((op_tp + op_fp) - npos).mean())
    auprc = None
    if cond in ("ap", "lat", "fusion"):   # threshold-free AUPRC over the SAME fixed candidate set
        allp = []
        for c in range(C):
            allp += greedy_match(cands[c], _peaks(cache[c], "ap"), _peaks(cache[c], "lat"), afoot[c], lfoot[c])
        auprc = average_precision(allp, float(npos.sum()))
    neg = np.where(npos == 0)[0]; j1 = int(np.argmin(np.abs(fp_img - 1.0)))
    return {"sens": pt, "ci": ci, "grid": grid.tolist(),
            "froc": list(zip([round(float(x), 4) for x in fp_img], [round(float(y), 4) for y in sens])),
            "op_threshold": round(op_threshold, 4), "case_recall": round(case_recall, 4),
            "count_mae": round(count_mae, 4), "unmatched_fp_at_op": int(op_fp.sum()),
            "auprc": (round(auprc, 4) if auprc is not None else None),
            "clean_case_fp_at~1fp": [int(FP[c, j1]) for c in neg], "n_neg": int(len(neg))}


def train_view(d, tr_idx, va_idx, view, epochs, batch, arch, lr, dev, seed, augment=False, cosine=False,
               dice_weight=0.0, pos_weight=1.0, strong_aug=False, weight_decay=0.0, verbose=True,
               rib=None, rib_has=None, rib_weight=0.0):
    torch.manual_seed(seed)
    dl_tr = DataLoader(Views(d, tr_idx, view, augment=augment, seed=seed, strong=strong_aug, rib=rib, rib_has=rib_has),
                       batch_size=batch, shuffle=True,
                       num_workers=0)  # forced 0: the dataset holds a mutable RNG (would need worker seeding otherwise)
    net = build_detector(arch, pretrained=True).to(dev); opt = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    foots = [group_instances(d, view)[i] for i in va_idx]; npos = np.array([len(f) for f in foots], np.float32)
    va_imgs = d[view][va_idx].astype(np.float32)
    multitask = rib_weight > 0.0
    best_key, best_state, best_ep = -1.0, None, -1; history = []; last_rib = 0.0
    for ep in range(epochs):
        net.train(); ep_loss = 0.0; ep_rib = 0.0; nb = 0
        for img, hm, rib_t, has in dl_tr:
            opt.zero_grad()
            if multitask:  # shared decoder -> fracture head (selection/eval) + rib head (LOGITS, aux only)
                pred, rib_logits = net.forward_multitask(img.to(dev))
                floss = det_loss(pred, hm.to(dev), dice_weight, pos_weight)
                hb = has.to(dev).bool()
                # rib loss ONLY on cases that HAVE a rib target (never supervise where there is no label)
                rl = rib_aux_loss(rib_logits[hb], rib_t.to(dev)[hb]) if hb.any() else floss.new_zeros(())
                loss = floss + rib_weight * rl; ep_rib += float(rl.detach().cpu())
            else:
                loss = det_loss(net(img.to(dev)), hm.to(dev), dice_weight, pos_weight)
            loss.backward(); opt.step()
            ep_loss += float(loss.detach().cpu()); nb += 1
        last_rib = ep_rib / max(nb, 1)
        net.eval(); C = len(va_idx); grid = np.linspace(1.0, 0.0, 101); TP = np.zeros((C, len(grid))); FP = np.zeros((C, len(grid)))
        max_hms = np.zeros(C); n_peaks = np.zeros(C)   # score-distribution diagnostics (MEASURE collapse, not infer it)
        with torch.no_grad():
            for c in range(C):
                hm = net(torch.from_numpy(va_imgs[c])[None, None].to(dev))[0, 0]
                max_hms[c] = float(hm.max().cpu())      # raw heatmap max, BEFORE the 0.05 extraction floor
                pk = peaks_from_hm(hm); n_peaks[c] = len(pk)   # local maxima surviving the 0.05 floor
                for j, t in enumerate(grid):
                    kp = pk[pk[:, 2] >= t] if len(pk) else pk; tp, fp, _ = match_one_view(kp, foots[c]); TP[c, j] = tp; FP[c, j] = fp
        s = sens_at_targets(TP.sum(0), FP.sum(0), npos, C); key = s[1.0] + 0.25 * s[0.5]
        cur_lr = opt.param_groups[0]["lr"]
        # directly-measured failure signals: is the model even producing peaks above the extraction floor?
        diag = {"max_hm_mean": round(float(max_hms.mean()), 4), "max_hm_median": round(float(np.median(max_hms)), 4),
                "frac_cases_peak_above_0.05": round(float((n_peaks > 0).mean()), 4),
                "mean_n_peaks_above_0.05": round(float(n_peaks.mean()), 2)}
        rec_ep = {"epoch": ep + 1, "mean_focal_loss": round(ep_loss / max(nb, 1), 4),
                  "val_sens@0.5FP": round(s[0.5], 4), "val_sens@1FP": round(s[1.0], 4), "lr": round(cur_lr, 6), **diag}
        if multitask: rec_ep["mean_rib_loss"] = round(last_rib, 4)
        history.append(rec_ep)
        if sched is not None: sched.step()
        if key > best_key: best_key, best_ep = key, ep + 1; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if verbose: print(f"  [{view}] epoch {ep+1}/{epochs}  loss {ep_loss/max(nb,1):.4f}"
                          + (f" rib {last_rib:.3f}" if multitask else "") + f"  val sens@1FP {s[1.0]:.3f} "
                          f"@0.5FP {s[0.5]:.3f}  | max_hm {diag['max_hm_mean']:.3f}/{diag['max_hm_median']:.3f} "
                          f"(mean/med)  peaks>0.05: {diag['frac_cases_peak_above_0.05']*100:.0f}% cases, "
                          f"{diag['mean_n_peaks_above_0.05']:.1f}/case", file=sys.stderr, flush=True)
    # selection is INFORMATIVE only if the criterion ever exceeded 0 (some val sensitivity was achieved);
    # otherwise every epoch tied at 0 and best_ep is just the first epoch — not a real selection.
    informative = best_key > 0.0
    net.load_state_dict(best_state); return net, best_ep, history, informative


def diagnose_view(net, d, va_idx, view, dev, n=6):
    """When val sensitivity stays 0, print signals that separate the failure modes: (1) scores too
    low (max heatmap / #peaks), (2) peaks in wrong places (min dist strongest peak->footprint)."""
    foots = [group_instances(d, view)[i] for i in va_idx]; imgs = d[view][va_idx].astype(np.float32)
    print(f"  [diagnose {view}] max_hm | n_peaks>0.05 | strongest_peak->nearest_footprint(px)", file=sys.stderr)
    net.eval()
    with torch.no_grad():
        for c in range(min(n, len(va_idx))):
            hm = net(torch.from_numpy(imgs[c])[None, None].to(dev))[0, 0]
            mx = float(hm.max().cpu()); pk = peaks_from_hm(hm)
            if len(pk) and foots[c]:
                strongest = pk[int(pk[:, 2].argmax()), :2]
                dmin = min(_min_dist(strongest, fp) for fp in foots[c])
            else:
                dmin = float("nan")
            print(f"    case{c}: max_hm {mx:.3f} | peaks {len(pk)} | dist {dmin:.1f}  (match radius {MATCH_RADIUS_PX})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--views", choices=["ap", "lat", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base-ch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-pct", type=int, default=20); ap.add_argument("--si-tol", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--backbone", choices=["scratch", "residual", "smp"], default="scratch",
                    help="scratch = the v1 from-scratch U-Net (baseline); residual = scratch U-Net with residual "
                         "double-conv blocks; smp = pretrained-encoder U-Net (segmentation_models_pytorch).")
    ap.add_argument("--residual", action="store_true",
                    help="scratch backbone only: residual double-conv blocks (equivalent to --backbone residual).")
    ap.add_argument("--norm", choices=["batch", "group"], default="batch",
                    help="normalization for scratch/residual blocks: batch (default) or group (batch-independent; "
                         "better at small batch sizes). Ignored for --backbone smp.")
    ap.add_argument("--encoder", default="resnet34", help="smp encoder name, e.g. resnet34 or efficientnet-b0 (only with --backbone smp)")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="smp encoder pretraining ('imagenet' or 'none'); recorded so eval/sealed rebuild the same shape")
    ap.add_argument("--dice-weight", type=float, default=0.0, help="weight of the soft-Dice term added to focal (0 = focal only)")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="explicit positive-branch weight in the focal loss")
    ap.add_argument("--strong-aug", action="store_true",
                    help="peak-preserving spatial+photometric augmentation (integer shift + gamma + gaussian noise); "
                         "requires --augment. NO rotation/scale (those would destroy the target's exact 1.0 peaks).")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (mild regularization for the pretrained encoder)")
    ap.add_argument("--no-standardize", action="store_true",
                    help="disable per-image input z-scoring for --backbone smp (ON by default; pretrained "
                         "encoders expect standardized inputs — raw [0,1] input collapses them to zero sensitivity)")
    ap.add_argument("--augment", action="store_true", help="training augmentation (hflip + intensity jitter) to combat overfitting")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay to 0 over the run (stabilizes the noisy tail, sharpens convergence)")
    ap.add_argument("--rib-targets", type=Path, default=None,
                    help="rib-region auxiliary side-car (det_dev_rib*.npz from make_rib_targets.py). Enables the "
                         "multi-task rib head (scratch/residual backbone only). TRAINING supervision only.")
    ap.add_argument("--rib-aux-weight", type=float, default=0.0, help="weight of the rib auxiliary loss (BCE-with-logits + Dice)")
    ap.add_argument("--out", type=Path, default=Path("outputs/detector_dev"))
    ap.add_argument("--smoke", action="store_true", help="mark run as debug; caps epochs<=2, bootstrap<=20; blocks freezing")
    ap.add_argument("--freeze-protocol", action="store_true",
                    help="DEPRECATED: freezing is done by freeze_detector.py (promote), not by retraining")
    a = ap.parse_args()
    if torch is None: print("pip install torch scipy", file=sys.stderr); return 1
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass
    if a.smoke:  # a smoke run can never be a long run masquerading as a quick check
        a.epochs = min(a.epochs, 2); a.bootstrap = min(a.bootstrap, 20)

    # ---- resolve the model architecture (recorded so evaluate/freeze/sealed rebuild it exactly) ----
    if a.strong_aug and not a.augment:
        raise SystemExit("--strong-aug requires --augment (it extends the base hflip+intensity augmentation).")
    enc_w = None if str(a.encoder_weights).lower() in ("none", "", "null") else a.encoder_weights
    if a.backbone == "smp":
        try:
            import segmentation_models_pytorch as _smp  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"--backbone smp needs segmentation_models_pytorch: pip install segmentation-models-pytorch "
                             f"({e})")
        arch = {"kind": "smp", "encoder": a.encoder, "encoder_weights": enc_w, "in_channels": 1,
                "decoder_channels": list(DECODER_CHANNELS), "standardize": (not a.no_standardize)}
    else:
        residual = (a.backbone == "residual") or a.residual
        arch = {"kind": "scratch_unet", "base_ch": a.base_ch, "residual": residual, "norm": a.norm}

    # Freezing is DECOUPLED from training: an MPS rerun would not reproduce bit-identical weights, so
    # you must PROMOTE the exact selected dev-run artifacts, not retrain. This script only writes a dev run.
    if a.freeze_protocol:
        raise SystemExit("--freeze-protocol is removed from the trainer (retraining to freeze is not reproducible). "
                         "Run a dev run here, then promote the exact selected artifacts:\n"
                         "  python freeze_detector.py --dev-run <this --out dir> --primary <ap|fusion> "
                         "--overlay-qa-passed --overlay-qa-cases \"...\" --out <new frozen dir>")

    d, man, dev_sha = load_dev(a.data); dev = device()
    cases, sources, nfrac = d["case"], d["source"], d["nfrac"]
    tr_mask, va_mask, strata = stratified_val_split(cases, sources, nfrac, a.val_pct)
    tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
    if len(va_idx) == 0 or len(tr_idx) == 0: raise ValueError("Empty dev-internal train or val slice; adjust --val-pct.")
    print(f"device {dev} | dataset {man.get('dataset_version')} | {len(cases)} dev -> train {len(tr_idx)} / val {len(va_idx)}", flush=True)
    print(f"strata (n/train/val): " + " | ".join(f"{k} {v['n']}/{v['train']}/{v['val']}" for k, v in strata.items()), flush=True)

    # ---- rib auxiliary side-car (multi-task rib head): load + HARD-verify alignment to this det_dev ----
    rib_data = None; rib_cfg = None
    if (a.rib_targets is not None) or (a.rib_aux_weight > 0):
        if a.rib_targets is None or a.rib_aux_weight <= 0:
            raise SystemExit("--rib-targets and --rib-aux-weight must be given together (weight > 0).")
        if arch["kind"] != "scratch_unet":
            raise SystemExit("the rib auxiliary head is implemented for --backbone scratch/residual only.")
        rd = np.load(a.rib_targets, allow_pickle=False)
        if [str(c) for c in rd["case"]] != [str(c) for c in d["case"]]:
            raise SystemExit("rib side-car case order != det_dev case order — refusing to misalign supervision.")
        rman_path = a.rib_targets.with_name(a.rib_targets.stem + "_manifest.json")
        rman = json.loads(rman_path.read_text()) if rman_path.exists() else {}
        want = rman.get("aligned_to", {}).get("det_dev_sha256")
        if want is None:
            print("WARNING: rib side-car manifest missing det_dev_sha256; cannot verify alignment.", file=sys.stderr)
        elif want != dev_sha:
            raise SystemExit(f"rib side-car was aligned to det_dev sha {want[:12]}.. but this det_dev is {dev_sha[:12]}..")
        rib_data = rd; arch["rib_head"] = True
        rib_cfg = {"weight": a.rib_aux_weight, "targets": str(a.rib_targets),
                   "rib_target_sha256": rman.get("rib_target_sha256"),
                   "source": rman.get("rib_auxiliary_target", {}).get("source"),
                   "loss": "BCE-with-logits + soft Dice", "masked_to_has_rib": True,
                   "checkpoint_selection": "fracture val sens ONLY (rib head never selects)",
                   "inference": "fracture head only; rib head unused at eval/sealed"}
        print(f"rib auxiliary: weight {a.rib_aux_weight} | {int(rd['has_rib'].sum())}/{len(rd['case'])} cases have a rib target "
              f"| det_dev alignment {'VERIFIED' if want == dev_sha else 'UNVERIFIED'}", flush=True)

    print(f"model: {arch['kind']}" + (f" encoder={arch['encoder']} weights={arch['encoder_weights']}"
          f" standardize={arch['standardize']}"
          if arch["kind"] == "smp" else f" base_ch={arch['base_ch']}{' residual' if arch.get('residual') else ''}"
          f" norm={arch.get('norm', 'batch')}")
          + f" | loss=focal" + (f"+{a.dice_weight}*dice" if a.dice_weight > 0 else "")
          + (f" pos_weight={a.pos_weight}" if a.pos_weight != 1.0 else "")
          + (f" | rib-aux w={a.rib_aux_weight}" if rib_data is not None else "")
          + (f" | strong-aug" if a.strong_aug else "") + (f" wd={a.weight_decay}" if a.weight_decay else ""), flush=True)
    nets = {}; best_ep = {}; history = {}; informative = {}
    for view in (("ap", "lat") if a.views == "both" else (a.views,)):
        rv = rib_data[f"{view}_rib_mask"] if rib_data is not None else None
        rh = rib_data["has_rib"] if rib_data is not None else None
        nets[view], best_ep[view], history[view], informative[view] = train_view(
            d, tr_idx, va_idx, view, a.epochs, a.batch, arch, a.lr, dev, a.seed, augment=a.augment, cosine=a.cosine,
            dice_weight=a.dice_weight, pos_weight=a.pos_weight, strong_aug=a.strong_aug, weight_decay=a.weight_decay,
            rib=rv, rib_has=rh, rib_weight=a.rib_aux_weight)
    # If a view's checkpoint-selection criterion never became informative (val sensitivity stayed 0),
    # its best_epoch is a tied-zero default, and freezing it would be meaningless.
    non_informative = [v for v, ok in informative.items() if not ok]
    if non_informative:
        print(f"\nWARNING: selection criterion NEVER became informative for {non_informative} (val sensitivity "
              f"stayed 0; best_epoch is a tied-zero default). Running diagnostics:", file=sys.stderr)
        for v in non_informative:
            diagnose_view(nets[v], d, va_idx, v, dev)

    ap_g, lat_g = group_instances(d, "ap"), group_instances(d, "lat")
    cache = peak_cache(nets, d, va_idx, dev)
    conds = [c for c in ("ap", "lat") if c in nets]
    if a.views == "both": conds += ["fusion", "paired"]
    results = {c: eval_condition(c, cache, ap_g, lat_g, va_idx, a.si_tol, a.bootstrap, a.seed) for c in conds}

    print("\n====== DETECTOR DEV-INTERNAL FROC (hash-stratified VAL slice; NOT the sealed test) ======")
    name = {"ap": "AP-only", "lat": "lateral-only", "fusion": "biplanar-FUSION", "paired": "paired-confirmed"}
    # LEADING condition marked by evidence (highest AUPRC among ap/lat/fusion), not by assumption
    rankable = [c for c in conds if results[c]["auprc"] is not None]
    leader = max(rankable, key=lambda c: results[c]["auprc"]) if rankable else None
    print(f"{'condition':18}{'unit':16}" + "".join(f"{'sens@'+str(t)+'FP':>18}" for t in FP_TARGETS))
    for c in conds:
        r = results[c]; row = f"{(name[c]+' *' if c == leader else name[c]):18}{UNIT[c]:16}"
        for t in FP_TARGETS:
            lo, hi = r["ci"][t]; row += f"{(format(r['sens'][t],'.3f')+' ['+format(lo,'.2f')+','+format(hi,'.2f')+']'):>18}"
        print(row)
    print(f"* LEADING condition on this run (highest AUPRC) = {name.get(leader, 'n/a')}. This is DESCRIPTIVE, not a")
    print("  frozen choice: the deployed primary condition is settled only after convergence + the bounded fusion")
    print("  comparison. fusion/paired are candidate biplanar conditions. FROC is a fixed-candidate ranked curve.")
    print(f"\n--- operating-point metrics (at each condition's frozen threshold ~1 FP/img) + AUPRC ---")
    print(f"{'condition':18}{'op_thr':>8}{'case-recall':>13}{'count-MAE':>11}{'unmatched-FP':>14}{'AUPRC':>8}")
    for c in conds:
        r = results[c]
        print(f"{name[c]:18}{r['op_threshold']:>8.3f}{r['case_recall']:>13.3f}{r['count_mae']:>11.3f}"
              f"{r['unmatched_fp_at_op']:>14d}{('%.3f'%r['auprc'] if r['auprc'] is not None else '   n/a'):>8}")
    print("  DEVELOPMENT ONLY — these informed the procedure and are NOT a confirmatory test. Clean-case")
    print("  specificity rests on only 20 negatives (exploratory, wide intervals). No CT anatomy at inference.")

    # ---- persist a DEV RUN (never frozen here; freeze_detector.py promotes the selected artifacts) ----
    work = a.out
    work.mkdir(parents=True, exist_ok=True)
    det_sha = {}
    for v, net in nets.items():
        wpath = work / f"detector_{v}.pt"
        torch.save(net.state_dict(), wpath); det_sha[v] = sha256_file(wpath)
    record = {
        "frozen": False, "smoke": bool(a.smoke),
        "protocol": PROTOCOL_VERSION, "dataset_version": man.get("dataset_version"), "det_dev_sha256": dev_sha,
        "detector_sha256": det_sha,   # sealed evaluator verifies weights match these
        "eval_params": {"si_tol": a.si_tol, "bootstrap": a.bootstrap, "seed": a.seed, "base_ch": a.base_ch},
        "model_arch": arch,   # single source of truth: evaluate/freeze/sealed rebuild the net from this
        "software": {"torch": torch.__version__, "numpy": np.__version__, "device": str(dev),
                     "deterministic_requested": True, "seed": a.seed},
        "model": {"architecture": (f"per-view UNet(base_ch={a.base_ch}{', residual' if arch.get('residual') else ''}"
                                   f", norm={arch.get('norm', 'batch')})"
                                   if arch["kind"] == "scratch_unet"
                                   else f"per-view smp.Unet(encoder={arch['encoder']}, weights={arch['encoder_weights']}, "
                                        f"in_ch=1) + sigmoid head"),
                  "param_count_per_view": {v: sum(p.numel() for p in net.parameters()) for v, net in nets.items()}},
        "learning_procedure": {
            "views": a.views,
            "loss": ("penalty-reduced focal (a=2,b=4)" + (f" + {a.dice_weight}*soft-Dice" if a.dice_weight > 0 else "")
                     + (f", pos_weight={a.pos_weight}" if a.pos_weight != 1.0 else "")),
            "optimizer": f"Adam(lr={a.lr}" + (f", weight_decay={a.weight_decay}" if a.weight_decay else "") + ")",
            "epochs": a.epochs, "batch": a.batch, "base_ch": a.base_ch,
            "lr_schedule": ("cosine annealing to 0" if a.cosine else "constant"),
            "augmentation": (("hflip(image+heatmap) + intensity jitter [x0.9-1.1, +/-0.02]"
                              + (" + strong: integer-shift(+/-8px) + gamma[0.8-1.25] + gaussian-noise(0.02), "
                                 "peak-preserving (no rotate/scale)" if a.strong_aug else "")) if a.augment else "none"),
            "final_model": "checkpoint trained on the fixed TRAIN slice ONLY (no retrain-on-all-325)",
            "dev_internal_val": f"hash-stratified {a.val_pct}% by md5(case) within "
                                "(train_part1-pos / validation-pos / validation-neg); >=1 train & >=1 val per stratum",
            "checkpoint_criterion": "max val sens@1FP + 0.25*sens@0.5FP (per view)",
            "best_epoch_per_view": best_ep,
            "selection_informative_per_view": informative,   # False => criterion never left 0; best_epoch is a tied-zero default
            "peak_extraction": PEAK_EXTRACTION,   # nms radius, minimum score, max peaks/view — sealed evaluator asserts these
            "operating_threshold_per_condition": {c: results[c]["op_threshold"] for c in conds},   # frozen ~1 FP/img; reused on test
            "froc_grid": "dense = sorted-unique candidate scores (per condition), anchored 1.0 (top) and "
                         "minimum_score (bottom); the per-condition grids below are FROZEN and reused verbatim on the sealed test",
            "froc_interpolation": "sensitivity = maximum achieved sensitivity subject to FP/image <= target "
                                  "(step, no linear interpolation between operating points)",
            "confirmatory_endpoints": "lesion-level FROC sensitivity at 0.5/1/2/4 FP with case-clustered 95% CIs, full "
                                      "FROC curves, and (at the frozen ~1 FP/img operating point) case-level recall, "
                                      "fracture-count MAE, and unmatched-peak (FP) count; plus threshold-free AUPRC. "
                                      "By-fracture-class sensitivity is DEFERRED (not a v1 endpoint): det GT stores "
                                      "instance ids (fp_iid) but not fracture-class codes, which would need a separate "
                                      "frozen instance-class artifact; it will NOT be added after seeing test results.",
            "biplanar_fusion": {"algorithm": "build candidate set ONCE at the extraction floor (pair AP<->lat by SI, "
                                "assign FIXED score, retain unmatched single-view), THEN threshold the fixed candidate "
                                "list. NOT 'threshold each view then pair'. Gives a monotonic ranked FROC.",
                                "pairing": f"one-to-one SI-voxel Hungarian, tol {a.si_tol} voxels (geometry only)",
                                "pair_score": "max(view scores)",
                                "matching": "a candidate matches an instance if EITHER present view-peak is within the "
                                "radius (min-distance). This is geometry-based UNION + de-duplication, NOT two-view "
                                "corroboration; any AUPRC gain reflects complementary recall/dedup, not stronger per-lesion evidence.",
                                "retain_single_view": True,
                                "unmatched_lateral_score_gate": 0.0},   # 0 at training (ungated); calibrated post-hoc by
                                # calibrate_fusion.py, then baked in via evaluate_detector.py --lat-gate and reused on the sealed test
            "paired_confirmed": "secondary: both view-peaks must hit (corroboration); NOT the deployed fusion condition",
            "rib_auxiliary": rib_cfg},   # None unless the multi-task rib head was trained
        "frozen_froc_grids": {c: results[c]["grid"] for c in conds},   # reused verbatim on the sealed test
        "split": {"strata": strata,
                  "train_case_ids": [str(cases[i]) for i in tr_idx], "val_case_ids": [str(cases[i]) for i in va_idx],
                  "train_pos": int((nfrac[tr_idx] > 0).sum()), "train_neg": int((nfrac[tr_idx] == 0).sum()),
                  "val_pos": int((nfrac[va_idx] > 0).sum()), "val_neg": int((nfrac[va_idx] == 0).sum())},
        "training_history_per_view": history,
        "dev_internal_froc": {c: {"sens_at_targets": {str(k): v for k, v in results[c]["sens"].items()},
                                  "ci": {str(k): v for k, v in results[c]["ci"].items()},
                                  "froc_curve": results[c]["froc"], "unit": UNIT[c],
                                  "op_threshold": results[c]["op_threshold"], "case_recall": results[c]["case_recall"],
                                  "count_mae": results[c]["count_mae"], "unmatched_fp_at_op": results[c]["unmatched_fp_at_op"],
                                  "auprc": results[c]["auprc"],
                                  "clean_case_fp_at~1fp": results[c]["clean_case_fp_at~1fp"],
                                  "n_negatives": results[c]["n_neg"]} for c in conds},
        "note": "Clean-case specificity is exploratory (20 dev negatives). Sealed test is scored ONCE by "
                "eval_sealed_test.py using EXACTLY this frozen learning_procedure and these frozen_froc_grids."}
    (work / "detector_dev_run.json").write_text(json.dumps(record, indent=2))
    print(f"\nwrote detector_dev_run.json (dev run, NOT frozen) + weights to {a.out}/ . Sealed test UNTOUCHED.")
    print("To lock after selection: promote the EXACT artifacts with")
    print(f"  python freeze_detector.py --dev-run {a.out} --primary <ap|fusion> --overlay-qa-passed --overlay-qa-cases \"...\" --out <new dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
