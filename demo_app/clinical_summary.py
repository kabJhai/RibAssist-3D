# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kabila Haile Soboka
"""Case-level and finding-level clinical decision-support summaries."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from demo_app.finding_map import classify_display_kind
from demo_app.pipeline import CaseInferenceResult, EnrichedFinding

LOW_ADDRESS_THRESHOLD = 0.35
NEAR_BOUNDARY_MM = 10.0


@dataclass
class CaseSummary:
    case_id: str
    detected: int = 0
    addressed: int = 0
    paired: int = 0
    localized: int = 0
    candidates: int = 0
    rib_only: int = 0
    abstained: int = 0
    unlinked: int = 0
    need_review: int = 0
    review_priority: int = 0
    rib_groups: dict[str, int] = field(default_factory=dict)
    predominant_levels: str = ""
    duplicate_rib_levels: list[str] = field(default_factory=list)
    low_confidence_ids: list[int] = field(default_factory=list)
    automated_impression: str = ""
    review_suggestions: list[str] = field(default_factory=list)
    along_rib_warning: str = (
        "Along-rib position is exploratory and should not be used as precise anatomical localization."
    )


def _rib_label(f: EnrichedFinding) -> str | None:
    o = f.original
    if o.side and o.rib is not None:
        return f"{o.side}{int(o.rib)}"
    return None


def _confidence_word(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.55:
        return "high"
    if score >= LOW_ADDRESS_THRESHOLD:
        return "moderate"
    return "low"


def build_case_summary(result: CaseInferenceResult, review: dict[int, str]) -> CaseSummary:
    findings = result.findings
    s = CaseSummary(case_id=result.case.case_id, detected=len(findings))

    rib_ctr: Counter[str] = Counter()
    for ef in findings:
        o = ef.original
        if o.address_status == "addressed":
            s.addressed += 1
        if o.source == "paired":
            s.paired += 1
        kind = classify_display_kind(ef)
        if kind == "localized":
            s.localized += 1
        elif kind == "candidate":
            s.candidates += 1
            s.abstained += 1
        else:
            s.rib_only += 1
            if ef.localization_status == "Unlinked":
                s.unlinked += 1
        if review.get(fid := ef.original.finding_id) == "needs_review":
            s.need_review += 1
        rl = _rib_label(ef)
        if rl:
            rib_ctr[rl] += 1

    s.rib_groups = dict(sorted(rib_ctr.items(), key=lambda x: (-x[1], x[0])))
    s.duplicate_rib_levels = [k for k, v in rib_ctr.items() if v > 1]

    nums = []
    for k in rib_ctr:
        try:
            nums.append(int(k[1:]))
        except ValueError:
            pass
    if nums:
        lo, hi = min(nums), max(nums)
        sides = sorted({k[0] for k in rib_ctr})
        s.predominant_levels = f"{''.join(sides)}{lo}–{hi}" if lo != hi else f"{''.join(sides)}{lo}"

    priority_ids: list[int] = []
    for ef in findings:
        fid = ef.original.finding_id
        reasons = []
        if review.get(fid) == "needs_review":
            reasons.append("flagged")
        if (ef.original.address_score or 0) < LOW_ADDRESS_THRESHOLD:
            reasons.append("low_address")
            s.low_confidence_ids.append(fid)
        if ef.localization_status == "Abstained":
            reasons.append("abstained")
        if _rib_label(ef) in s.duplicate_rib_levels:
            reasons.append("duplicate_rib")
        if reasons:
            priority_ids.append(fid)
    s.review_priority = len(set(priority_ids))

    # Full automated impression for the collapsed panel
    parts = [
        f"{s.detected} suspected rib-fracture finding{'s' if s.detected != 1 else ''} were identified."
    ]
    if s.localized:
        parts.append(
            f"{s.localized} finding{'s' if s.localized != 1 else ''} "
            f"with model-emitted 3D localization."
        )
    if s.candidates:
        parts.append(
            f"{s.candidates} candidate location{'s' if s.candidates != 1 else ''} "
            "did not clear the frozen correspondence threshold."
        )
    if s.rib_only:
        parts.append(f"{s.rib_only} rib-level finding{'s' if s.rib_only != 1 else ''} without a reliable 3D point.")
    if s.duplicate_rib_levels:
        dup = ", ".join(f"{k} ×{rib_ctr[k]}" for k in s.duplicate_rib_levels[:3])
        parts.append(f"Multiple findings share rib levels ({dup}).")
    s.automated_impression = " ".join(parts)

    suggestions: list[str] = []
    if s.duplicate_rib_levels:
        suggestions.append("Review duplicated findings at the same rib level.")
    if s.low_confidence_ids:
        suggestions.append("Prioritize findings with low addressing confidence.")
    if s.abstained:
        suggestions.append(
            "Manually review abstained findings. Detection stays positive but no 3D point was emitted."
        )
    suggestions.append(s.along_rib_warning)
    s.review_suggestions = suggestions
    return s


def compact_case_line(summary: CaseSummary) -> str:
    return (
        f"{summary.detected} finding{'s' if summary.detected != 1 else ''} · "
        f"{summary.localized} localized · "
        f"{summary.candidates} candidate{'s' if summary.candidates != 1 else ''} · "
        f"{summary.rib_only} AP-only/rib-level · "
        f"{summary.need_review} need review"
    )


def short_impression(summary: CaseSummary) -> str:
    parts: list[str] = []
    if summary.localized:
        parts.append(
            f"{summary.localized} finding{'s' if summary.localized != 1 else ''} "
            f"with model-emitted 3D localization."
        )
    if summary.candidates:
        parts.append(
            f"{summary.candidates} compatible pair{'s' if summary.candidates != 1 else ''} "
            "shown as uncommitted candidate locations."
        )
    if summary.duplicate_rib_levels:
        parts.append("Review duplicated findings at the same rib level.")
    if not parts:
        parts.append("Review rib-level findings on projections.")
    return " ".join(parts[:3])


def build_finding_interpretation(
    ef: EnrichedFinding,
    *,
    eval_dist_mm: float | None = None,
    eval_rib_exact: bool | None = None,
) -> dict[str, str | list[str]]:
    o = ef.original
    rib_txt = f"{o.side}{int(o.rib)}" if o.side and o.rib is not None else "unspecified rib"
    kind = classify_display_kind(ef)
    lines: list[str] = []

    if kind == "localized":
        lines.append("Model committed biplanar correspondence.")
        lines.append(f"Approximate 3D fracture location emitted on {rib_txt}.")
        conf = _confidence_word(o.address_score)
        lines.append(f"Addressing confidence is {conf}.")
        if eval_dist_mm is not None:
            lines.append(f"Fracture-volume distance: {eval_dist_mm:.1f} mm.")
            if eval_rib_exact is not None:
                lines.append(f"Rib-exact: {'yes' if eval_rib_exact else 'no'}.")
        action = "Review both projections before acceptance."
    elif kind == "candidate":
        lines.append("Compatible AP and lateral observations were detected.")
        lines.append("Correspondence confidence did not clear the frozen threshold.")
        lines.append("Candidate location is shown in amber for manual review.")
        lines.append("Frozen policy did not emit this point.")
        action = "Confirm whether the candidate location is clinically plausible."
    else:
        lines.append("Fracture finding retained on AP.")
        lines.append("Rib level was addressed, but no reliable biplanar 3D location is available.")
        action = "Review AP projection and rib assignment."

    return {
        "title": "Suggested interpretation",
        "lines": lines,
        "action": f"Suggested review: {action}",
    }


def format_rib_groups(groups: dict[str, int]) -> str:
    if not groups:
        return "No rib-level assignments"
    return "\n".join(f"{k}: {v} finding{'s' if v != 1 else ''}" for k, v in groups.items())


def build_decision_card(
    ef: EnrichedFinding,
    *,
    eval_dist_mm: float | None = None,
    eval_rib_exact: bool | None = None,
) -> dict[str, str | list[str]]:
    """Compact decision-support card for finding review."""
    o = ef.original
    rib_txt = f"{o.side}{int(o.rib)}" if o.side and o.rib is not None else "unspecified rib"
    kind = classify_display_kind(ef)
    conf = _confidence_word(o.address_score)

    if kind == "localized":
        headline = f"Localized on {rib_txt}"
        lines = [
            "Model-emitted AP–lateral correspondence.",
            "Approximate 3D fracture location available.",
            f"Addressing confidence: {conf.capitalize()}.",
        ]
        if eval_dist_mm is not None:
            lines.append(f"Evaluation distance: {eval_dist_mm:.1f} mm.")
        if eval_rib_exact is not None:
            lines.append(f"Rib-exact (eval): {'yes' if eval_rib_exact else 'no'}.")
        action = "Confirm the finding on both projections before accepting."
    elif kind == "candidate":
        headline = f"Candidate location on {rib_txt}"
        lines = [
            "A compatible AP–lateral pair was found, but correspondence confidence did not meet the frozen threshold.",
            "Candidate location is shown for manual review only.",
        ]
        action = "Manually compare the proposed location across both views."
    else:
        headline = f"Rib-level finding on {rib_txt}"
        lines = [
            "The AP detector retained a fracture finding and assigned a rib level.",
            "No reliable lateral correspondence or 3D point is available.",
        ]
        action = f"Inspect {rib_txt} manually on AP and lateral views."

    return {"headline": headline, "lines": lines, "action": f"Suggested review: {action}"}


def build_completion_impression(summary: CaseSummary, review: dict[int, str]) -> str:
    accepted = sum(1 for v in review.values() if v == "accepted")
    needs = sum(1 for v in review.values() if v == "needs_review")
    rejected = sum(1 for v in review.values() if v == "rejected")
    pending = summary.detected - accepted - needs - rejected

    parts = [
        f"{accepted} finding{'s' if accepted != 1 else ''} accepted by reviewer",
        f"{needs} marked for additional review" if needs else None,
        f"{rejected} rejected by reviewer" if rejected else None,
    ]
    lead = ", ".join(p for p in parts if p) + "."
    detail = []
    if summary.localized:
        detail.append(
            f"{summary.localized} finding{'s' if summary.localized != 1 else ''} "
            f"{'have' if summary.localized != 1 else 'has'} model-emitted biplanar 3D localization."
        )
    if summary.rib_only:
        detail.append(
            f"{summary.rib_only} remain rib-level finding{'s' if summary.rib_only != 1 else ''} "
            "without reliable 3D localization."
        )
    if summary.duplicate_rib_levels:
        dup = ", ".join(summary.duplicate_rib_levels[:4])
        detail.append(f"Multiple findings occur at {dup}.")
    if pending:
        detail.append(f"{pending} finding{'s' if pending != 1 else ''} still pending review.")
    return lead + (" " + " ".join(detail) if detail else "")
