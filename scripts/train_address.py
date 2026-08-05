#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""RibAssist 3D addressing model (localization engine): from a CT-derived, fracture-centered
crop plus the candidate's 2D coordinates (detector-provided; spine-relative horizontally),
predict side (L/R), rib number (1-12), and normalized along-rib position s.

AP-vs-biplanar ablation by GROUP K-FOLD cross-validation (default), so every case is tested
once and the case-clustered bootstrap runs over ALL cases — stable on small data. Within
each fold: an inner train/select split picks the best checkpoint (best select exact-rib,
tie-break lower s-MAE), applied identically to AP and biplanar. A trivial prevalence
baseline (per fold) shows the models read the image, not fracture-location priors.

Usage:
  python train_address.py --data address_dataset.npz --folds 5 --epochs 40
  python train_address.py --data address_dataset.npz --folds 5 --epochs 40 --no-pos
"""
from __future__ import annotations
import argparse, sys, copy
from pathlib import Path
import numpy as np

try:
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
except Exception:  # noqa: BLE001
    torch = None


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


class Crops(Dataset):
    def __init__(self, d, idx):
        self.ap = d["ap"][idx].astype(np.float32); self.lat = d["lat"][idx].astype(np.float32)
        self.pap = d["ap_xy"][idx].astype(np.float32); self.plat = d["lat_xy"][idx].astype(np.float32)
        self.side = d["side"][idx].astype(np.float32); self.rib = (d["rib"][idx] - 1).astype(np.int64)
        self.s = d["s"][idx].astype(np.float32)
    def __len__(self): return len(self.side)
    def __getitem__(self, i):
        return (torch.from_numpy(self.ap[i])[None], torch.from_numpy(self.lat[i])[None],
                torch.from_numpy(self.pap[i]), torch.from_numpy(self.plat[i]),
                self.side[i], self.rib[i], self.s[i])


def stream():
    def blk(a, b): return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())


class Net(nn.Module):
    def __init__(self, views, use_pos):
        super().__init__(); self.views, self.use_pos = views, use_pos
        self.ap = stream() if views in ("ap", "both") else None
        self.lat = stream() if views in ("lat", "both") else None
        ns = 2 if views == "both" else 1
        f = 64 * ns + (2 * ns if use_pos else 0)
        self.hs = nn.Linear(f, 1); self.hr = nn.Linear(f, 12); self.hp = nn.Linear(f, 1)
    def forward(self, ap, lat, pap, plat):
        z = []
        if self.ap is not None: z.append(self.ap(ap))
        if self.lat is not None: z.append(self.lat(lat))
        if self.use_pos:
            if self.views in ("ap", "both"): z.append(pap)
            if self.views in ("lat", "both"): z.append(plat)
        f = torch.cat(z, 1)
        return self.hs(f).squeeze(1), self.hr(f), torch.sigmoid(self.hp(f)).squeeze(1)


def predict(net, dl, dev):
    net.eval(); R, S, P = [], [], []
    with torch.no_grad():
        for ap, lat, pap, plat, *_ in dl:
            sl, rl, sp = net(ap.to(dev), lat.to(dev), pap.to(dev), plat.to(dev))
            R.append(torch.softmax(rl, 1).cpu().numpy()); S.append(torch.sigmoid(sl).cpu().numpy()); P.append(sp.cpu().numpy())
    return np.concatenate(R), np.concatenate(S), np.concatenate(P)


def metrics(rP, sP, pP, rib_t, side_t, s_t):
    rp = rP.argmax(1) + 1
    return np.array([((sP > .5) == side_t).mean(), (rp == rib_t).mean(),
                     (np.abs(rp - rib_t) <= 1).mean(), np.abs(pP - s_t).mean()])


def run_once(d, tr, sel, te, views, use_pos, epochs, dev, seed=0):
    torch.manual_seed(seed)
    dl_tr = DataLoader(Crops(d, tr), batch_size=32, shuffle=True)
    dl_sel = DataLoader(Crops(d, sel), batch_size=128); dl_te = DataLoader(Crops(d, te), batch_size=128)
    net = Net(views, use_pos).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce, ce, mse = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss()
    st = (d["rib"][sel], d["side"][sel], d["s"][sel]); bk, bs = (-1, 1e9), None
    for _ in range(epochs):
        net.train()
        for ap, lat, pap, plat, side, rib, s in dl_tr:
            opt.zero_grad()
            sl, rl, sp = net(ap.to(dev), lat.to(dev), pap.to(dev), plat.to(dev))
            (bce(sl, side.to(dev)) + ce(rl, rib.to(dev)) + mse(sp, s.to(dev))).backward(); opt.step()
        m = metrics(*predict(net, dl_sel, dev), *st); key = (m[1], -m[3])
        if key > bk: bk, bs = key, copy.deepcopy(net.state_dict())
    net.load_state_dict(bs); return predict(net, dl_te, dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--no-pos", action="store_true")
    a = ap.parse_args()
    if torch is None: print("pip install torch scikit-learn", file=sys.stderr); return 1
    d = np.load(a.data, allow_pickle=True); g = d["case"]; use_pos = not a.no_pos; dev = device()
    print(f"device {dev} | use_pos={use_pos} | {len(g)} crops / {len(set(g))} cases | {a.folds}-fold CV", flush=True)
    order, PA, PB, TR, TS, TP, BASE = [], [], [], [], [], [], []
    gkf = GroupKFold(n_splits=a.folds)
    for k, (tv, te) in enumerate(gkf.split(g, groups=g)):
        g2 = g[tv]; tr_r, sel_r = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(g2, groups=g2))
        tr, sel = tv[tr_r], tv[sel_r]
        pa = run_once(d, tr, sel, te, "ap", use_pos, a.epochs, dev)
        pb = run_once(d, tr, sel, te, "both", use_pos, a.epochs, dev)
        PA.append(pa); PB.append(pb)
        TR.append(d["rib"][te]); TS.append(d["side"][te]); TP.append(d["s"][te]); order.append(g[te])
        sm = int(round(d["side"][tr].mean())); rm = int(np.bincount(d["rib"][tr]).argmax()); pmed = float(np.median(d["s"][tr]))
        BASE.append(np.array([(d["side"][te] == sm).mean(), (d["rib"][te] == rm).mean(),
                              (np.abs(d["rib"][te] - rm) <= 1).mean(), np.abs(d["s"][te] - pmed).mean()]))
        print(f"  fold {k}: test {len(set(g[te]))}c | AP exact {metrics(*pa, d['rib'][te], d['side'][te], d['s'][te])[1]:.2f} "
              f"| BOTH exact {metrics(*pb, d['rib'][te], d['side'][te], d['s'][te])[1]:.2f}", flush=True)
    cat = lambda L: (np.concatenate([x[0] for x in L]), np.concatenate([x[1] for x in L]), np.concatenate([x[2] for x in L]))
    PA, PB = cat(PA), cat(PB); rib = np.concatenate(TR); side = np.concatenate(TS); s = np.concatenate(TP); cases = np.concatenate(order)
    mA, mB = metrics(*PA, rib, side, s), metrics(*PB, rib, side, s)
    base = np.mean([b for b in BASE], 0)
    # case-clustered bootstrap of the paired gap over ALL cases
    uc = np.unique(cases); G = {i: [] for i in range(4)}
    for _ in range(1000):
        samp = uc[np.random.randint(0, len(uc), len(uc))]; idx = np.concatenate([np.where(cases == c)[0] for c in samp])
        ga = metrics(PA[0][idx], PA[1][idx], PA[2][idx], rib[idx], side[idx], s[idx])
        gb = metrics(PB[0][idx], PB[1][idx], PB[2][idx], rib[idx], side[idx], s[idx])
        for i in range(4): G[i].append(gb[i] - ga[i])
    lab = ["side-acc", "rib-exact", "rib±1", "s-MAE"]
    print("\n====== ADDRESSING ABLATION ({}-fold CV, POST-HOC on development set) ======".format(a.folds))
    print(f"{'metric':10}{'baseline':>11}{'AP-only':>11}{'AP+lateral':>13}{'BOTH-AP 95% CI':>22}")
    for i, L in enumerate(lab):
        lo, hi = np.percentile(G[i], 2.5), np.percentile(G[i], 97.5)
        print(f"{L:10}{base[i]:>11.3f}{mA[i]:>11.3f}{mB[i]:>13.3f}"
              f"{('['+format(lo,'+.3f')+', '+format(hi,'+.3f')+']'):>22}")
    print("\nPOST-HOC stability analysis on the DEVELOPMENT set: these 60 cases already informed")
    print("design choices (coordinate encoding, side handling), so this is NOT a fresh")
    print("confirmatory test. Each baseline column is that endpoint's own training prior")
    print("(majority side / modal rib / median s). Checkpoint selection uses only an inner")
    print("split; the outer fold is evaluated once. Biplanar benefit is supported where the")
    print("CI excludes 0 (below 0 for s-MAE); it reduces split luck but does not add test data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
