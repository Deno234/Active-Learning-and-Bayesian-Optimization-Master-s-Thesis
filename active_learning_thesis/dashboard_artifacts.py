from __future__ import annotations

import csv
from pathlib import Path


RUNTIME_SUFFIXES = (".gro", ".cpt", ".xtc", ".tpr")
DYNAMICS_COMPLETE_STATUSES = {"dynamics_complete", "sasa_complete", "analysis_complete"}
FULL_ANALYSIS_STATUSES = {"sasa_complete", "analysis_complete"}
ACTIVE_REMOTE_SYNC_STATES = {"staged_remote", "submitted", "running"}
RETURNED_REMOTE_SYNC_STATES = {"outputs_returned", "finalized_local"}


def _append_unique(items: list[str], value: str) -> None:
    text = str(value).strip()
    if text and text not in items:
        items.append(text)


def _artifact_groups() -> dict[str, dict[str, list[str]]]:
    return {}


def _ensure_group(groups: dict[str, dict[str, list[str]]], group: str) -> dict[str, list[str]]:
    return groups.setdefault(group, {"expected": [], "found": [], "missing": []})


def _expect_artifact(
    groups: dict[str, dict[str, list[str]]],
    expected: list[str],
    group: str,
    artifact: str,
) -> None:
    _append_unique(expected, artifact)
    _append_unique(_ensure_group(groups, group)["expected"], artifact)


def _mark_found(
    groups: dict[str, dict[str, list[str]]],
    found: list[str],
    group: str,
    artifact: str,
) -> None:
    _append_unique(found, artifact)
    _append_unique(_ensure_group(groups, group)["found"], artifact)


def _mark_missing(
    groups: dict[str, dict[str, list[str]]],
    group: str,
    artifact: str,
    message: str,
    *,
    errors: list[str],
    warnings: list[str],
    level: str = "error",
) -> None:
    _append_unique(_ensure_group(groups, group)["missing"], artifact)
    if level == "warning":
        warnings.append(message)
    else:
        errors.append(message)


def _group_summary(groups: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    expected_groups: list[str] = []
    complete_groups: list[str] = []
    missing_groups: list[str] = []
    found_groups: list[str] = []
    group_states: list[dict[str, str]] = []

    for group, state in groups.items():
        expected = list(state.get("expected", []))
        found = list(state.get("found", []))
        missing = list(state.get("missing", []))
        if expected:
            expected_groups.append(group)
        if found:
            found_groups.append(group)
        if missing:
            missing_groups.append(group)
            group_state = "missing"
        elif expected and all(item in found for item in expected):
            complete_groups.append(group)
            group_state = "complete"
        elif expected:
            group_state = "waiting"
        else:
            group_state = "not expected"
        group_states.append(
            {
                "group": group,
                "state": group_state,
                "expected": ", ".join(expected) or "-",
                "found": ", ".join(found) or "-",
                "missing": ", ".join(missing) or "-",
            }
        )

    expected_count = len(expected_groups)
    complete_count = len(complete_groups)
    if expected_count:
        stage_completeness = f"{complete_count}/{expected_count} expected artifact groups complete"
    else:
        stage_completeness = "No artifact groups expected yet"

    return {
        "expected_groups": expected_groups,
        "complete_groups": complete_groups,
        "missing_groups": missing_groups,
        "found_groups": found_groups,
        "group_states": group_states,
        "stage_completeness": stage_completeness,
    }


def _join_or_dash(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def _run_retry_action(*, severity: str, verification_state: str, sync_status: str) -> str:
    if severity == "error":
        if sync_status == "outputs_staged":
            return "Re-run the SUPEK pull into the staging area, then finalize only after the staging folder contains files."
        return "Restore the live run folder/config first, then re-run the SUPEK pull or local finalize step that produced this state."
    if severity == "warning":
        return "Inspect the staged/live SUPEK folder before launching follow-up model work."
    if verification_state == "Waiting on remote outputs":
        return "Poll SUPEK, fetch logs if the job leaves the queue, then pull artifacts back when the remote job is done."
    return "No retry needed; continue with the next model-workflow step."


def _campaign_retry_action(
    *,
    missing_groups: list[str],
    severity: str,
    verification_state: str,
    sync_status: str,
    profile: str,
) -> str:
    missing = set(missing_groups)
    if "Campaign structure" in missing:
        return "Restore the campaign package files or re-run prepare-md-stage for this peptide/stage before touching BURA again."
    if "Candidate source batch" in missing:
        return "Restore/create the source batch CSV or promote the reporting-only peptide into a real batch before ingest or retry."
    if "BURA staged download" in missing:
        return "Re-run the BURA pull/copy-back into the local staging path, then inspect the staged files before finalizing."
    if "Runtime MD outputs" in missing:
        return "Re-pull the BURA package or rerun local finalize so the .gro/.cpt/.xtc/.tpr outputs land in the live package directory."
    if "Full analysis evidence" in missing:
        if profile == "full":
            return "Re-run or finalize the full-analysis post-processing until SASA/AP evidence files are present, then review manually."
        return "Inspect the review row: non-full stages should not normally require SASA/AP evidence."
    if "Ingest handoff" in missing:
        return "Regenerate cgmd_ingest.csv only after the human cgmd_label, round_id, and ingest provenance are correct."
    if severity == "warning":
        return "Inspect the warning groups before trusting this campaign for review, ingest, or ladder advancement."
    if verification_state == "Waiting on remote outputs":
        return "Keep polling/fetching BURA logs, then pull and finalize artifacts when the tracked job completes."
    if sync_status in RETURNED_REMOTE_SYNC_STATES:
        return "No retry needed; the local artifact set matches the current reviewed state."
    return "No retry needed yet; continue with the next safe MD ladder step."


def _canonical_path(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return str(Path(text))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _matching_outputs(package_dir: Path, sequence: str, suffixes: tuple[str, ...]) -> list[Path]:
    hits: list[Path] = []
    if not package_dir.exists():
        return hits
    sequence_prefix = f"{sequence}_"
    for path in package_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(sequence_prefix):
            continue
        if any(path.name.endswith(suffix) for suffix in suffixes):
            hits.append(path)
    return hits


def _stage_file(campaign_dir: Path, raw_value: str) -> Path:
    text = str(raw_value).strip()
    if not text:
        return Path("")
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return campaign_dir / candidate


def _severity_rank(severity: str) -> int:
    return {"error": 0, "warning": 1, "info": 2, "ok": 3}.get(severity, 4)


def _verification_state(
    *,
    errors: list[str],
    warnings: list[str],
    waiting: str,
) -> tuple[str, str, str]:
    if errors:
        return "Attention needed", "error", errors[0]
    if warnings:
        return "Attention needed", "warning", warnings[0]
    if waiting:
        return "Waiting on remote outputs", "info", waiting
    return "Verified", "ok", "Required artifacts for the current state are present."


def _run_artifact_row(run: dict[str, object]) -> dict[str, object] | None:
    sync_status = str(run.get("remote_sync_status", "")).strip()
    local_stage_text = str(run.get("local_stage_path", "")).strip()
    local_stage_path = Path(local_stage_text) if local_stage_text else None
    remote_path = str(run.get("remote_path", "")).strip()
    if sync_status in {"", "not_synced"} and not local_stage_path and not remote_path:
        return None

    run_dir = Path(str(run.get("run_dir", "")))
    found: list[str] = []
    expected: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    groups = _artifact_groups()

    config_path = run_dir / "config.json"
    _expect_artifact(groups, expected, "Run structure", "config.json")
    if config_path.exists():
        _mark_found(groups, found, "Run structure", "config.json")
    else:
        _mark_missing(
            groups,
            "Run structure",
            "config.json",
            "Missing run config.json in the live run directory.",
            errors=errors,
            warnings=warnings,
        )

    waiting = ""
    if sync_status in {"staged_remote", "submitted", "running"}:
        waiting = "Remote SUPEK execution is still active or staged; output verification will tighten after pull-back."
        if not remote_path:
            warnings.append("Tracked SUPEK state is missing the remote run path.")
    if sync_status == "outputs_staged":
        _expect_artifact(groups, expected, "SUPEK staged download", "staged download")
        if local_stage_path is not None and local_stage_path.exists():
            _mark_found(groups, found, "SUPEK staged download", "staged download")
            if local_stage_path.is_dir() and any(local_stage_path.iterdir()):
                _mark_found(groups, found, "SUPEK staged download", "staged files")
            else:
                _mark_missing(
                    groups,
                    "SUPEK staged download",
                    "staged files",
                    "The staged SUPEK download path exists but is currently empty.",
                    errors=errors,
                    warnings=warnings,
                    level="warning",
                )
        else:
            _mark_missing(
                groups,
                "SUPEK staged download",
                "staged download",
                "SUPEK artifacts were marked as staged, but the local staging path is missing.",
                errors=errors,
                warnings=warnings,
            )
    if sync_status in {"outputs_returned", "finalized_local"} and not run_dir.exists():
        _mark_missing(
            groups,
            "Run structure",
            "live run directory",
            "The live run directory is missing even though SUPEK outputs were returned.",
            errors=errors,
            warnings=warnings,
        )

    verification_state, severity, summary = _verification_state(errors=errors, warnings=warnings, waiting=waiting)
    group_summary = _group_summary(groups)
    safest_retry_action = _run_retry_action(
        severity=severity,
        verification_state=verification_state,
        sync_status=sync_status,
    )
    if severity == "error":
        next_move = f"Open Operations -> Transfers and fix the missing run artifacts. Safest retry: {safest_retry_action}"
    elif severity == "warning":
        next_move = f"Inspect the live or staged SUPEK files before trusting the returned outputs. Safest retry: {safest_retry_action}"
    elif verification_state == "Waiting on remote outputs":
        next_move = safest_retry_action
    else:
        next_move = "The SUPEK handoff looks healthy. Use Model Workflow for the next local thesis step."

    return {
        "scope": "run",
        "cluster": "supek",
        "run": str(run.get("run_display_name", run.get("run_name", ""))),
        "sequence": "-",
        "campaign": "-",
        "md_profile": "-",
        "target_kind": "Model run",
        "target": str(run.get("run_display_name", run.get("run_name", ""))),
        "sync_state": sync_status or "not_synced",
        "verification_state": verification_state,
        "severity": severity,
        "summary": summary,
        "missing_artifacts": ", ".join(errors + warnings) or "-",
        "expected_artifacts": _join_or_dash(expected),
        "found_artifacts": _join_or_dash(found),
        "expected_groups": _join_or_dash(list(group_summary["expected_groups"])),
        "complete_groups": _join_or_dash(list(group_summary["complete_groups"])),
        "missing_groups": _join_or_dash(list(group_summary["missing_groups"])),
        "found_groups": _join_or_dash(list(group_summary["found_groups"])),
        "group_states": list(group_summary["group_states"]),
        "stage_completeness": str(group_summary["stage_completeness"]),
        "safest_retry_action": safest_retry_action,
        "next_move": next_move,
        "run_dir": str(run_dir),
        "campaign_dir": "",
        "source_path": str(run_dir),
        "remote_path": remote_path or "-",
        "staging_path": str(local_stage_path) if local_stage_path is not None else "-",
        "live_path": str(run_dir),
        "remote_job_id": str(run.get("remote_job_id", "")).strip() or "-",
        "sequence_key": "",
    }


def _campaign_artifact_row(run: dict[str, object], campaign: dict[str, object]) -> dict[str, object]:
    campaign_dir = Path(str(campaign.get("campaign_dir", "")))
    package_dir = Path(str(campaign.get("package_path", "")))
    review_path = Path(str(campaign.get("review_path", "")))
    manifest_path = Path(str(campaign.get("manifest_path", "")))
    source_batch_text = str(campaign.get("source_batch_csv", "")).strip()
    source_batch_path = Path(source_batch_text) if source_batch_text else None
    local_stage_text = str(campaign.get("local_stage_path", "")).strip()
    local_stage_path = Path(local_stage_text) if local_stage_text else None
    sequence = str(campaign.get("sequence", "")).strip()
    sync_status = str(campaign.get("sync_status", "")).strip() or "not_synced"
    profile = str(campaign.get("md_profile", "")).strip()
    label_value = str(campaign.get("cgmd_label", "")).strip()
    round_id = str(campaign.get("round_id", "")).strip()

    expected: list[str] = []
    found: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    groups = _artifact_groups()

    _expect_artifact(groups, expected, "Campaign structure", "manifest.csv")
    if manifest_path.exists():
        _mark_found(groups, found, "Campaign structure", "manifest.csv")
    else:
        _mark_missing(
            groups,
            "Campaign structure",
            "manifest.csv",
            "Missing manifest.csv for the MD campaign.",
            errors=errors,
            warnings=warnings,
        )

    _expect_artifact(groups, expected, "Campaign structure", "md_review.csv")
    if review_path.exists():
        _mark_found(groups, found, "Campaign structure", "md_review.csv")
    else:
        _mark_missing(
            groups,
            "Campaign structure",
            "md_review.csv",
            "Missing md_review.csv for the MD campaign.",
            errors=errors,
            warnings=warnings,
        )

    _expect_artifact(groups, expected, "Campaign structure", "package dir")
    if package_dir.exists():
        _mark_found(groups, found, "Campaign structure", "package dir")
    else:
        _mark_missing(
            groups,
            "Campaign structure",
            "package dir",
            "Missing package directory for the peptide.",
            errors=errors,
            warnings=warnings,
        )

    package_pdb = package_dir / f"{sequence}.pdb"
    _expect_artifact(groups, expected, "Campaign structure", "package PDB")
    if package_pdb.exists():
        _mark_found(groups, found, "Campaign structure", "package PDB")
    else:
        _mark_missing(
            groups,
            "Campaign structure",
            "package PDB",
            "Missing packaged peptide PDB in the campaign package directory.",
            errors=errors,
            warnings=warnings,
        )

    _expect_artifact(groups, expected, "Candidate source batch", "source batch CSV")
    if source_batch_path is not None and source_batch_path.exists():
        _mark_found(groups, found, "Candidate source batch", "source batch CSV")
    else:
        _mark_missing(
            groups,
            "Candidate source batch",
            "source batch CSV",
            "The source batch CSV recorded in stage metadata is missing.",
            errors=errors,
            warnings=warnings,
            level="warning",
        )

    runtime_outputs = _matching_outputs(package_dir, sequence, RUNTIME_SUFFIXES)
    if runtime_outputs:
        _mark_found(groups, found, "Runtime MD outputs", "runtime outputs")

    waiting = ""
    remote_path = str(campaign.get("remote_path", "")).strip()
    if sync_status in {"staged_remote", "submitted", "running"}:
        waiting = "Remote BURA execution is still staged or active; local output verification will tighten after copy-back."
        if not remote_path:
            warnings.append("Tracked BURA state is missing the remote campaign path.")
    if sync_status == "outputs_staged":
        _expect_artifact(groups, expected, "BURA staged download", "staged download")
        if local_stage_path is not None and local_stage_path.exists():
            _mark_found(groups, found, "BURA staged download", "staged download")
            if local_stage_path.is_dir() and any(local_stage_path.iterdir()):
                _mark_found(groups, found, "BURA staged download", "staged files")
            else:
                _mark_missing(
                    groups,
                    "BURA staged download",
                    "staged files",
                    "The staged BURA download path exists but is currently empty.",
                    errors=errors,
                    warnings=warnings,
                    level="warning",
                )
        else:
            _mark_missing(
                groups,
                "BURA staged download",
                "staged download",
                "BURA outputs were marked as staged, but the local staging path is missing.",
                errors=errors,
                warnings=warnings,
            )
    else:
        expects_runtime_outputs = (
            str(campaign.get("job_root_status", "")) in DYNAMICS_COMPLETE_STATUSES
            or sync_status in RETURNED_REMOTE_SYNC_STATES
        )
        if expects_runtime_outputs:
            _expect_artifact(groups, expected, "Runtime MD outputs", "runtime outputs")
            if not runtime_outputs:
                _mark_missing(
                    groups,
                    "Runtime MD outputs",
                    "runtime outputs",
                    "The review or sync status says MD outputs should be local, but no .gro/.cpt/.xtc/.tpr files were found in the live package directory.",
                    errors=errors,
                    warnings=warnings,
                )

        sasa_path = _stage_file(campaign_dir, str(campaign.get("sasa_file", "")))
        if str(campaign.get("sasa_file", "")).strip():
            _expect_artifact(groups, expected, "Full analysis evidence", "SASA summary")
            if sasa_path.exists():
                _mark_found(groups, found, "Full analysis evidence", "SASA summary")
            else:
                _mark_missing(
                    groups,
                    "Full analysis evidence",
                    "SASA summary",
                    "The review row points to a missing SASA summary file.",
                    errors=errors,
                    warnings=warnings,
                )
        elif profile == "full" and str(campaign.get("job_root_status", "")) in FULL_ANALYSIS_STATUSES:
            _expect_artifact(groups, expected, "Full analysis evidence", "SASA summary")
            fallback_sasa = next(package_dir.glob("*_sasa.xvg"), None) if package_dir.exists() else None
            if fallback_sasa is not None:
                _mark_found(groups, found, "Full analysis evidence", "SASA summary")
            else:
                _mark_missing(
                    groups,
                    "Full analysis evidence",
                    "SASA summary",
                    "Full-analysis outputs were reported past dynamics, but no SASA summary is present.",
                    errors=errors,
                    warnings=warnings,
                )

        ap_path = _stage_file(campaign_dir, str(campaign.get("ap_file", "")))
        if str(campaign.get("ap_file", "")).strip():
            _expect_artifact(groups, expected, "Full analysis evidence", "AP summary")
            if ap_path.exists():
                _mark_found(groups, found, "Full analysis evidence", "AP summary")
            else:
                _mark_missing(
                    groups,
                    "Full analysis evidence",
                    "AP summary",
                    "The review row points to a missing AP summary file.",
                    errors=errors,
                    warnings=warnings,
                )
        elif profile == "full" and str(campaign.get("job_root_status", "")) == "analysis_complete":
            _expect_artifact(groups, expected, "Full analysis evidence", "AP summary")
            fallback_ap = next(package_dir.glob("*_AP_SASA.txt"), None) if package_dir.exists() else None
            if fallback_ap is not None:
                _mark_found(groups, found, "Full analysis evidence", "AP summary")
            else:
                _mark_missing(
                    groups,
                    "Full analysis evidence",
                    "AP summary",
                    "Full analysis was marked complete, but the AP summary file is missing.",
                    errors=errors,
                    warnings=warnings,
                )

    ingest_path = campaign_dir / "cgmd_ingest.csv"
    if ingest_path.exists():
        _expect_artifact(groups, expected, "Ingest handoff", "cgmd_ingest.csv")
        _mark_found(groups, found, "Ingest handoff", "cgmd_ingest.csv")
        ingest_rows = _read_csv_rows(ingest_path)
        matching_ingest = next((row for row in ingest_rows if str(row.get("sequence", "")).strip() == sequence), None)
        if matching_ingest is None:
            _mark_missing(
                groups,
                "Ingest handoff",
                "matching peptide row",
                "cgmd_ingest.csv exists, but it does not contain this peptide.",
                errors=errors,
                warnings=warnings,
            )
        else:
            ingest_label = str(matching_ingest.get("cgmd_label", "")).strip()
            ingest_round = str(matching_ingest.get("round_id", "")).strip()
            if label_value in {"0", "1"} and ingest_label != label_value:
                _mark_missing(
                    groups,
                    "Ingest handoff",
                    "matching cgmd_label",
                    "cgmd_ingest.csv label does not match the reviewed cgmd_label.",
                    errors=errors,
                    warnings=warnings,
                )
            if round_id and ingest_round != round_id:
                _mark_missing(
                    groups,
                    "Ingest handoff",
                    "matching round_id",
                    "cgmd_ingest.csv round_id does not match the campaign review row.",
                    errors=errors,
                    warnings=warnings,
                )
    elif label_value in {"0", "1"} and bool(campaign.get("promoted_to_real_batch_at", "")):
        _expect_artifact(groups, expected, "Ingest handoff", "cgmd_ingest.csv")
        _mark_missing(
            groups,
            "Ingest handoff",
            "cgmd_ingest.csv",
            "This peptide is promoted and labeled, but cgmd_ingest.csv has not been created yet.",
            errors=errors,
            warnings=warnings,
            level="warning",
        )

    verification_state, severity, summary = _verification_state(errors=errors, warnings=warnings, waiting=waiting)
    group_summary = _group_summary(groups)
    missing_groups = list(group_summary["missing_groups"])
    safest_retry_action = _campaign_retry_action(
        missing_groups=missing_groups,
        severity=severity,
        verification_state=verification_state,
        sync_status=sync_status,
        profile=profile,
    )
    if severity == "error":
        next_move = f"Open MD Validation -> Artifact verification and fix the missing artifact group before advancing. Safest retry: {safest_retry_action}"
    elif severity == "warning":
        next_move = f"Inspect the campaign files before trusting the returned MD result or creating ingest inputs. Safest retry: {safest_retry_action}"
    elif verification_state == "Waiting on remote outputs":
        next_move = safest_retry_action
    else:
        next_move = "Artifact integrity looks healthy for this campaign. Continue with review, ingest, or the next ladder stage."

    return {
        "scope": "campaign",
        "cluster": str(campaign.get("cluster", "")).strip() or "bura",
        "run": str(run.get("run_display_name", run.get("run_name", ""))),
        "sequence": sequence,
        "campaign": str(campaign.get("campaign", "")),
        "md_profile": profile,
        "target_kind": "MD campaign",
        "target": f"{sequence} ({campaign_dir.name})" if sequence else campaign_dir.name,
        "sync_state": sync_status,
        "verification_state": verification_state,
        "severity": severity,
        "summary": summary,
        "missing_artifacts": ", ".join(errors + warnings) or "-",
        "expected_artifacts": _join_or_dash(expected),
        "found_artifacts": _join_or_dash(found),
        "expected_groups": _join_or_dash(list(group_summary["expected_groups"])),
        "complete_groups": _join_or_dash(list(group_summary["complete_groups"])),
        "missing_groups": _join_or_dash(missing_groups),
        "found_groups": _join_or_dash(list(group_summary["found_groups"])),
        "group_states": list(group_summary["group_states"]),
        "stage_completeness": str(group_summary["stage_completeness"]),
        "safest_retry_action": safest_retry_action,
        "next_move": next_move,
        "run_dir": str(run.get("run_dir", "")),
        "campaign_dir": str(campaign_dir),
        "source_path": str(campaign_dir),
        "remote_path": remote_path or "-",
        "staging_path": str(local_stage_path) if local_stage_path is not None else "-",
        "live_path": str(package_dir) if package_dir else str(campaign_dir),
        "remote_job_id": str(campaign.get("remote_job_id", "")).strip() or "-",
        "sequence_key": sequence,
    }


def build_artifact_verification_rows(
    run_summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in run_summaries:
        run_row = _run_artifact_row(run)
        if run_row is not None:
            rows.append(run_row)
        for campaign in list(run.get("md_campaigns", [])):
            rows.append(_campaign_artifact_row(run, campaign))
    return sorted(
        rows,
        key=lambda row: (
            _severity_rank(str(row.get("severity", ""))),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
            str(row.get("campaign", "")),
        ),
    )


def build_artifact_verification_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "verified": sum(1 for row in rows if str(row.get("verification_state", "")) == "Verified"),
        "waiting": sum(1 for row in rows if str(row.get("verification_state", "")) == "Waiting on remote outputs"),
        "attention": sum(1 for row in rows if str(row.get("verification_state", "")) == "Attention needed"),
        "errors": sum(1 for row in rows if str(row.get("severity", "")) == "error"),
        "warnings": sum(1 for row in rows if str(row.get("severity", "")) == "warning"),
    }
