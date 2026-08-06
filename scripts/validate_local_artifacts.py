#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Validate published split JSONs and optional local artifact hashes.

The repository ships case-id splits only (no ground-truth rows). After you
fetch or regenerate local tensors, checkpoints, and labels, run this script
to check that your copies match the expected fingerprints in split_manifest.json.

Examples:
  python scripts/validate_local_artifacts.py
  python scripts/validate_local_artifacts.py --demo --strict
  python scripts/validate_local_artifacts.py --all --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "split_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_list_sha256(data: dict, list_keys: list[str]) -> str:
    payload = {key: sorted(data[key]) for key in list_keys}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def check_split(path: Path, spec: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return False, [f"missing {path.relative_to(ROOT)}"]

    got_file = sha256_file(path)
    want_file = spec["sha256"]
    if got_file != want_file:
        errors.append(
            f"{path.name} file sha256 mismatch: got {got_file[:12]}.. want {want_file[:12]}.."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    list_keys = spec["lists"]
    for key in list_keys:
        if key not in data or not isinstance(data[key], list):
            errors.append(f"{path.name} missing list key {key!r}")

    if not errors:
        got_lists = case_list_sha256(data, list_keys)
        want_lists = spec["case_list_sha256"]
        if got_lists != want_lists:
            errors.append(
                f"{path.name} case-list sha256 mismatch: got {got_lists[:12]}.. want {want_lists[:12]}.."
            )

    split_md5 = spec.get("split_md5")
    if split_md5 and data.get("split_md5") != split_md5:
        errors.append(
            f"{path.name} split_md5 mismatch: got {data.get('split_md5')!r} want {split_md5!r}"
        )

    return not errors, errors


def check_artifacts(group: dict[str, str], *, strict: bool) -> tuple[int, int, list[str]]:
    ok = 0
    missing = 0
    errors: list[str] = []

    for rel, want in group.items():
        path = ROOT / rel
        if not path.is_file():
            missing += 1
            msg = f"missing {rel}"
            if strict:
                errors.append(msg)
            else:
                print(f"  skip: {msg}")
            continue

        got = sha256_file(path)
        if got != want:
            errors.append(
                f"{rel} sha256 mismatch: got {got[:12]}.. want {want[:12]}.."
            )
        else:
            ok += 1
            print(f"  ok:   {rel}")

    return ok, missing, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to split manifest (default: {DEFAULT_MANIFEST.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="also verify demo-stack local artifacts",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="verify demo + sealed-eval + optional research artifacts",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when optional local artifacts are missing",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = manifest.get("published_splits", {})
    local = manifest.get("local_artifacts", {})

    print("Checking published splits (case IDs only)...")
    split_errors: list[str] = []
    for name, spec in published.items():
        ok, errs = check_split(ROOT / name, spec)
        if ok:
            print(f"  ok:   {name}")
        else:
            for err in errs:
                split_errors.append(err)
                print(f"  FAIL: {err}")

    artifact_errors: list[str] = []
    if args.demo or args.all:
        print("Checking demo local artifacts...")
        _, _, errs = check_artifacts(local.get("demo", {}), strict=args.strict)
        artifact_errors.extend(errs)

    if args.all:
        print("Checking sealed-eval local artifacts...")
        _, _, errs = check_artifacts(local.get("sealed_eval", {}), strict=args.strict)
        artifact_errors.extend(errs)
        print("Checking optional research artifacts...")
        _, _, errs = check_artifacts(local.get("research_optional", {}), strict=args.strict)
        artifact_errors.extend(errs)

    if split_errors:
        print("\nSplit validation failed.", file=sys.stderr)
        return 1

    if artifact_errors:
        print("\nLocal artifact validation failed.", file=sys.stderr)
        for err in artifact_errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    if args.demo or args.all:
        print("\nValidation passed.")
    else:
        print("\nSplit validation passed. Run with --demo after fetching local artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
