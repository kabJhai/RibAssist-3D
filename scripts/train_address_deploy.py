#!/usr/bin/env python3
"""Train and SAVE a DEPLOYABLE RibAssist 3D addressing model on the DET-FRAME dataset.

train_address.py only cross-validates (AP-vs-biplanar ablation) and saves nothing. For the end-to-end
pipeline we need ONE frozen addressing checkpoint that maps a detector-frame crop + 2D coords ->
(side, rib 1..12, s). An inner case-grouped train/select split is used ONLY to choose the training
DURATION (best_epoch by inner-select rib-exact, tie-break lower s-MAE — identical selection rule to
train_address); the deployment checkpoint is then a freshly-initialized model REFIT FROM SCRATCH on
the FULL development dataset for that many epochs (reusing train_address's Net / Crops / metrics
verbatim, so the deployed model is exactly the ablation's architecture). Writes:
  addressing_model.pt    — state_dict refit from scratch on the full development dataset for the
                           internally selected epoch count
  addressing_model.json  — config (views, use_pos, crop) + provenance (dataset + det_dev sha) +
                           the SELECTION model's inner-select metrics (exploratory) + a trivial baseline

Biplanar ('both') + position coords are the default because the addressing ablation showed the lateral
view and spine-relative coords help exact-rib and s-MAE.

Usage:
  python train_address_deploy.py --data outputs/det_out_v2/address_dataset_detframe.npz \
      --views both --epochs 60 --out outputs/addressing_model_v1
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np

try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import GroupShuffleSplit
    import train_address as TA
except Exception:  # noqa: BLE001
    torch = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="det-frame address dataset (make_address_data_detframe.py)")
    ap.add_argument("--views", choices=["ap", "lat", "both"], default="both")
    ap.add_argument("--no-pos", action="store_true", help="drop the 2D coordinate inputs (spine-relative position)")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--select-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if torch is None: print("pip install torch scikit-learn", file=sys.stderr); return 1
    if a.out.exists(): raise FileExistsError(f"{a.out} exists; deployment checkpoints are versioned — use a new dir.")
    torch.manual_seed(a.seed)
    data_sha = sha256_file(a.data); d = np.load(a.data, allow_pickle=False)
    use_pos = not a.no_pos; dev = TA.device(); g = d["case"]
    # det_dev alignment (for provenance) from the sibling manifest, if present
    man_path = a.data.with_name(a.data.stem + "_manifest.json")
    det_dev_sha = None
    if man_path.exists():
        det_dev_sha = json.loads(man_path.read_text()).get("aligned_to", {}).get("det_dev_sha256")

    tr, sel = next(GroupShuffleSplit(1, test_size=a.select_frac, random_state=a.seed).split(g, groups=g))
    print(f"device {dev} | views={a.views} use_pos={use_pos} | {len(g)} crops / {len(set(g))} cases "
          f"-> train {len(tr)} / select {len(sel)} ({len(set(g[sel]))} cases)", flush=True)
    dl_tr = DataLoader(TA.Crops(d, tr), batch_size=32, shuffle=True)
    dl_sel = DataLoader(TA.Crops(d, sel), batch_size=128)
    net = TA.Net(a.views, use_pos).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    bce, ce, mse = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss()
    st = (d["rib"][sel], d["side"][sel], d["s"][sel]); best_key, best_state, best_ep = (-1, 1e9), None, -1
    for ep in range(a.epochs):
        net.train()
        for apc, latc, pap, plat, side, rib, s in dl_tr:
            opt.zero_grad()
            sl, rl, sp = net(apc.to(dev), latc.to(dev), pap.to(dev), plat.to(dev))
            (bce(sl, side.to(dev)) + ce(rl, rib.to(dev)) + mse(sp, s.to(dev))).backward(); opt.step()
        m = TA.metrics(*TA.predict(net, dl_sel, dev), *st); key = (m[1], -m[3])   # (rib-exact, -s-MAE)
        if key > best_key: best_key, best_state, best_ep = key, copy.deepcopy(net.state_dict()), ep + 1
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep+1}/{a.epochs}  select rib-exact {m[1]:.3f}  side {m[0]:.3f}  rib±1 {m[2]:.3f}  s-MAE {m[3]:.3f}",
                  file=sys.stderr, flush=True)
    net.load_state_dict(best_state)
    ms = TA.metrics(*TA.predict(net, dl_sel, dev), *st)   # inner-select metrics = the SELECTION model (85% train)
    sm = int(round(d["side"][tr].mean())); rm = int(np.bincount(d["rib"][tr]).argmax()); pmed = float(np.median(d["s"][tr]))
    base = [float((d["side"][sel] == sm).mean()), float((d["rib"][sel] == rm).mean()),
            float((np.abs(d["rib"][sel] - rm) <= 1).mean()), float(np.abs(d["s"][sel] - pmed).mean())]

    # ---- REFIT the DEPLOYMENT checkpoint from scratch on ALL cases for the selected best_epoch ----
    # (the inner split only picks best_epoch; the deployed model must not discard the select cases on a
    #  dataset this small). The inner-select metrics above describe the SELECTION model, NOT this refit.
    print(f"refitting deployment model from scratch on ALL {len(g)} crops / {len(set(g))} cases for {best_ep} epochs ...",
          flush=True)
    torch.manual_seed(a.seed)
    full_net = TA.Net(a.views, use_pos).to(dev); full_opt = torch.optim.Adam(full_net.parameters(), 1e-3)
    dl_full = DataLoader(TA.Crops(d, np.arange(len(g))), batch_size=32, shuffle=True)
    for ep in range(best_ep):
        full_net.train()
        for apc, latc, pap, plat, side, rib, s in dl_full:
            full_opt.zero_grad()
            sl, rl, sp = full_net(apc.to(dev), latc.to(dev), pap.to(dev), plat.to(dev))
            (bce(sl, side.to(dev)) + ce(rl, rib.to(dev)) + mse(sp, s.to(dev))).backward(); full_opt.step()
    deployment_state = copy.deepcopy(full_net.state_dict())

    work = a.out.parent / f".{a.out.name}.tmp"
    if work.exists():
        import shutil; shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    torch.save(deployment_state, work / "addressing_model.pt")
    cfg = {"model": "RibAssist 3D addressing (train_address.Net)", "views": a.views, "use_pos": use_pos,
           "crop": int(d["ap"].shape[-1]), "seed": a.seed, "max_epochs_searched": a.epochs, "select_frac": a.select_frac,
           "selection_rule": "max inner-select rib-exact, tie-break lower s-MAE",
           "deployment_epochs": best_ep, "deployment_model_training_cases": int(len(set(g))),
           "deployment_model_training_crops": int(len(g)), "selection_model_training_cases": int(len(set(g[tr]))),
           "address_dataset": str(a.data), "address_dataset_sha256": data_sha, "aligned_det_dev_sha256": det_dev_sha,
           "inner_select_metrics": {"side_acc": round(ms[0], 4), "rib_exact": round(ms[1], 4),
                                    "rib_within1": round(ms[2], 4), "s_mae": round(ms[3], 4)},
           "trivial_baseline": {"side_acc": round(base[0], 4), "rib_exact": round(base[1], 4),
                                "rib_within1": round(base[2], 4), "s_mae": round(base[3], 4)},
           "state_dict_sha256": sha256_file(work / "addressing_model.pt"),
           "note": "epoch count selected using a case-disjoint inner split; deployment checkpoint subsequently refit "
                   "from scratch on ALL development cases for the selected number of epochs. inner_select_metrics are "
                   "the SELECTION model's exploratory estimate, NOT the deployment checkpoint's (which then also "
                   "trained on the selection cases). The sealed test is the confirmatory read.",
           "software": {"torch": torch.__version__, "numpy": np.__version__, "device": str(dev)}}
    (work / "addressing_model.json").write_text(json.dumps(cfg, indent=2))
    work.rename(a.out)
    print(f"\nsaved deployable addressing model to {a.out}/ (deployment refit on ALL {len(set(g))} cases, {best_ep} epochs)")
    print(f"  inner-select (SELECTION model, {len(set(g[tr]))} train cases): side {ms[0]:.3f} | rib-exact {ms[1]:.3f} "
          f"| rib±1 {ms[2]:.3f} | s-MAE {ms[3]:.3f}")
    print(f"  baseline: side {base[0]:.3f} | rib-exact {base[1]:.3f} | rib±1 {base[2]:.3f} | s-MAE {base[3]:.3f}")
    print("  (inner-select is optimistic AND describes the selection model, not the refit checkpoint; the SEALED "
          "test is the confirmatory read.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
