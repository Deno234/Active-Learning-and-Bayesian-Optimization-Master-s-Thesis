from __future__ import annotations

from typing import Mapping

LABEL_REVIEW_FIELDS = [
    "label_rubric",
    "label_confidence",
    "label_evidence_tags",
    "label_evidence_summary",
    "reviewer",
    "reviewed_at",
]

LABEL_RUBRIC_OPTIONS = ("", "self_assembling", "not_self_assembling", "uncertain_rerun")
LABEL_CONFIDENCE_OPTIONS = ("", "high", "medium", "low")
LABEL_EVIDENCE_TAG_OPTIONS = (
    "ap_supports_label",
    "sasa_supports_label",
    "trajectory_visual_check",
    "aggregation_contact_pattern",
    "borderline",
    "artifact_issue",
    "rerun_recommended",
)

RUBRIC_TO_CGMD_LABEL = {
    "self_assembling": "1",
    "not_self_assembling": "0",
    "uncertain_rerun": "",
}
CGMD_LABEL_TO_DEFAULT_RUBRIC = {
    "1": "self_assembling",
    "0": "not_self_assembling",
}


def normalize_evidence_tags(value: object) -> str:
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in str(value or "").replace(";", ",").split(","):
        tag = raw_tag.strip().lower().replace(" ", "_").replace("-", "_")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return ", ".join(tags)


def label_for_rubric(rubric: object) -> str:
    return RUBRIC_TO_CGMD_LABEL.get(str(rubric or "").strip(), "")


def default_rubric_for_label(label: object) -> str:
    return CGMD_LABEL_TO_DEFAULT_RUBRIC.get(str(label or "").strip(), "")


def review_schema_for_row(row: Mapping[str, object]) -> str:
    return "structured" if any(field in row for field in LABEL_REVIEW_FIELDS) else "legacy"


def review_evidence_status(row: Mapping[str, object]) -> dict[str, object]:
    label = str(row.get("cgmd_label", "") or "").strip()
    schema = str(row.get("label_review_schema", "") or "").strip() or review_schema_for_row(row)
    structured_required = schema != "legacy"
    rubric = str(row.get("label_rubric", "") or "").strip()
    confidence = str(row.get("label_confidence", "") or "").strip().lower()
    tags = normalize_evidence_tags(row.get("label_evidence_tags", ""))
    summary = str(row.get("label_evidence_summary", "") or "").strip()
    notes = str(row.get("review_notes", "") or "").strip()
    reviewer = str(row.get("reviewer", "") or "").strip()
    reviewed_at = str(row.get("reviewed_at", "") or "").strip()
    missing: list[str] = []
    blockers: list[str] = []

    if label not in {"0", "1"}:
        missing.append("cgmd_label")

    if structured_required:
        if rubric not in LABEL_RUBRIC_OPTIONS or not rubric:
            missing.append("label_rubric")
        elif rubric == "uncertain_rerun":
            blockers.append("rubric is uncertain/rerun")
        expected_label = label_for_rubric(rubric)
        if expected_label and label in {"0", "1"} and expected_label != label:
            blockers.append("cgmd_label/rubric mismatch")
        if confidence not in LABEL_CONFIDENCE_OPTIONS or not confidence:
            missing.append("label_confidence")
        if not summary:
            missing.append("label_evidence_summary")
        if not notes:
            missing.append("review_notes")
    elif label in {"0", "1"} and not notes:
        missing.append("review_notes")

    ingest_ready = label in {"0", "1"} and not missing and not blockers
    if ingest_ready and structured_required:
        state = "Evidence-backed label"
    elif ingest_ready:
        state = "Legacy label (notes-only)"
    elif rubric == "uncertain_rerun":
        state = "Uncertain / rerun"
    elif label in {"0", "1"}:
        state = "Label saved, evidence incomplete"
    else:
        state = "Needs review / label"

    return {
        "state": state,
        "ingest_ready": ingest_ready,
        "schema": schema,
        "label": label,
        "rubric": rubric,
        "confidence": confidence,
        "evidence_tags": tags,
        "evidence_summary": summary,
        "review_notes": notes,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "missing": missing,
        "blockers": blockers,
        "missing_text": ", ".join(missing) if missing else "-",
        "blocker_text": ", ".join(blockers) if blockers else "-",
    }
