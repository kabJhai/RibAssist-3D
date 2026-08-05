#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Freeze a fresh, UNTOUCHED internal test slice from the labeled TRAINING cohort, before
any results are examined. Assignment is deterministic (hash of case id) — it never looks at
labels, images, or performance, so the held-out set cannot be contaminated by peeking.

Writes frozen_split.json: {"test": [...], "dev_pool": [...], "test_pct", "note"}. The
existing RibFrac validation set stays in development; this frozen slice is scored exactly
once, at the very end.

Usage: python freeze_split.py --ribfrac-train-dir data/ribfrac_train --test-pct 18
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def _num(n):
    m = re.search(r"RibFrac[_-]?0*(\d+)", n, re.I)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ribfrac-train-dir", type=Path, required=True)
    ap.add_argument("--test-pct", type=int, default=18)
    ap.add_argument("--out", type=Path, default=Path("frozen_split.json"))
    a = ap.parse_args()
    cids = sorted({f"RibFrac{_num(p.name)}" for p in Path(a.ribfrac_train_dir).rglob("*image*.nii*")
                   if _num(p.name) is not None})
    if not cids:
        print("No RibFrac train image files found under", a.ribfrac_train_dir); return 1
    test, dev = [], []
    for c in cids:
        h = int(hashlib.md5(c.encode()).hexdigest(), 16) % 100   # deterministic, label-blind
        (test if h < a.test_pct else dev).append(c)
    a.out.write_text(json.dumps({"test": test, "dev_pool": dev, "test_pct": a.test_pct,
                                 "note": "frozen by hash(case_id); scored once at the end"}, indent=2))
    print(f"{len(cids)} train cases -> dev_pool {len(dev)} | UNTOUCHED test {len(test)}  (~{a.test_pct}%)")
    print(f"wrote {a.out}. Do NOT evaluate on the test cases until the final run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
