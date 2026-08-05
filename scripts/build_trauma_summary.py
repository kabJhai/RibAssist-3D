#!/usr/bin/env python3
"""RibAssist 3D structured rib-trauma SUMMARY from the canonical rib-level context (schema
ribassist-rib-level-2). Deterministic rendering of reconstruct_3d.py's per-case reconstruction.json
(discrete pattern + review cues + quality flags) into a human-readable summary + a machine-readable
descriptors table.

Findings + review cues ONLY — no management directives. Granularity is side + APPROXIMATE rib level;
adjacent ribs are context; along-rib position and exact rib are NOT established, and distinct fracture
SITES are NOT claimed (no multisite / flail language). Multiple detections at one predicted rib level
are surfaced as a QUALITY/AUDIT flag (possible duplicate detector responses), not as a trauma cue. The
frame is a fixed canonical atlas, not patient-specific geometry.

Input:  a reconstructions dir (each <case>/reconstruction.json from reconstruct_3d.py).
Output: <out>/summaries.txt, <out>/descriptors.csv, <out>/summaries.json.

Usage:
  python build_trauma_summary.py --recon-dir outputs/reconstructions --out outputs/trauma_summary
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def render(rec):
    p = rec["pattern"]
    right = [u for u in p["unique_rib_levels"] if u.startswith("R")]
    left = [u for u in p["unique_rib_levels"] if u.startswith("L")]
    lines = [f"=== RibAssist 3D Rib-Trauma Summary — {rec['case_id']} ===",
             f"Addressed detections: {p['n_addressed_detections']}   |   distinct predicted rib levels: {p['n_unique_rib_levels']}",
             f"  Right rib levels: {', '.join(right) if right else 'none'} ({len(right)})",
             f"  Left rib levels:  {', '.join(left) if left else 'none'} ({len(left)})",
             "Granularity: side + APPROXIMATE rib level; adjacent ribs are context; along-rib position not reported; "
             "distinct fracture sites NOT established.",
             "Pattern:"]
    if p["longest_consecutive_run"] >= 2:
        lines.append(f"  - longest consecutive run: {p['longest_consecutive_run']} predicted rib levels "
                     f"({'-'.join(p['longest_run_ribs'])}, each approximate)")
    lines.append(f"  - bilateral: {'yes' if p['bilateral'] else 'no'}")
    lines.append("Review cues:")
    for c in rec["decision_support_cues"]:
        lines.append(f"  - {c}")
    if rec.get("quality_flags"):
        lines.append("Quality/audit flags:")
        for q in rec["quality_flags"]:
            lines.append(f"  - {q}")
    lines.append("Note: canonical rib-LEVEL context (fixed atlas; not patient-specific). Side and approximate rib "
                 "level only; exact rib and along-rib position are NOT established. Review cues, not directives.")
    return "\n".join(lines)


def descriptor_row(rec):
    p = rec["pattern"]
    return {"case": rec["case_id"], "n_detections": p["n_addressed_detections"],
            "n_unique_rib_levels": p["n_unique_rib_levels"], "unique_rib_levels": ";".join(p["unique_rib_levels"]),
            "right_rib_levels": ";".join(u for u in p["unique_rib_levels"] if u.startswith("R")),
            "left_rib_levels": ";".join(u for u in p["unique_rib_levels"] if u.startswith("L")),
            "bilateral": p["bilateral"], "longest_consecutive_run": p["longest_consecutive_run"],
            "multiple_detections_same_rib_level": ";".join(p["multiple_detections_same_rib_level"].keys()),
            "n_cues": len(rec["decision_support_cues"]), "n_quality_flags": len(rec.get("quality_flags", []))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/trauma_summary"))
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    recs = []
    for jp in sorted(a.recon_dir.glob("*/reconstruction.json")):
        recs.append(json.loads(jp.read_text()))
    if not recs:
        print(f"No reconstruction.json under {a.recon_dir}/*/ — run reconstruct_3d.py first."); return 1
    a.out.mkdir(parents=True, exist_ok=True)
    texts = [render(r) for r in recs]; rows = [descriptor_row(r) for r in recs]
    (a.out / "summaries.txt").write_text("\n\n".join(texts))
    (a.out / "summaries.json").write_text(json.dumps({r["case_id"]: {"pattern": r["pattern"],
                                          "cues": r["decision_support_cues"],
                                          "quality_flags": r.get("quality_flags", [])} for r in recs}, indent=2))
    with open(a.out / "descriptors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    bil = sum(1 for r in rows if r["bilateral"]); runs3 = sum(1 for r in rows if r["longest_consecutive_run"] >= 3)
    print(f"cases: {len(rows)} | mean detections/case: {sum(r['n_detections'] for r in rows)/len(rows):.1f} | "
          f"bilateral: {bil} | >=3-consecutive: {runs3}")
    print(f"wrote {a.out}/summaries.txt, descriptors.csv, summaries.json")
    if a.show:
        print("\n" + "\n\n".join(texts[: a.show]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
