from __future__ import annotations

import csv
import json
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from active_learning_thesis.dashboard_actions import (
    ACTIVE_ACTION_STATUSES,
    FINAL_ACTION_STATUSES,
    REPO_ROOT,
    create_dashboard_action,
    draft_dashboard_action,
    list_dashboard_actions,
    load_dashboard_action,
    resume_dashboard_action,
    submit_finalize_md_stage_action,
    submit_prepare_md_stage_action,
    update_dashboard_action,
)
from active_learning_thesis.dashboard_md_batches import (
    find_dashboard_md_source_batch,
    find_run_md_source_batch,
)
from active_learning_thesis.dashboard_md_recovery import find_md_slate_rebind_candidate
from active_learning_thesis.dashboard_md_slate_state import (
    dashboard_md_slates_path,
    load_dashboard_md_slate,
    list_dashboard_md_slates,
    save_dashboard_md_slate,
)
from active_learning_thesis.dashboard_profiles import get_cluster_profile, load_cluster_profiles
from active_learning_thesis.dashboard_remote import (
    draft_bura_normalize_action,
    draft_bura_preflight_action,
    draft_bura_pull_package_action,
    draft_bura_submit_action,
    draft_bura_upload_campaign_action,
    parse_squeue_output,
    queue_bura_poll_action,
    queue_bura_readiness_action,
)
from active_learning_thesis.dashboard_remote_state import list_cluster_snapshots, list_sync_records, update_sync_status
from active_learning_thesis.md_orchestrator import STAGE_META_FILENAME

SLATE_STAGE_ORDER = ["line_smoke", "production_smoke", "full"]
SLATE_STAGE_CAPS = {
    "line_smoke": 2,
    "production_smoke": 1,
    "full": 5,
}
SLATE_FINAL_STATUSES = {"completed", "completed_with_failures", "cancelled"}
PEPTIDE_FINAL_STATUSES = {"completed", "blocked", "skipped"}
POLL_INTERVAL_SECONDS = 10
REHEARSAL_STEPS = (
    "prepare",
    "upload",
    "readiness",
    "normalize",
    "preflight",
    "submit",
    "poll_running",
    "poll_finished",
    "pull",
    "finalize",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _canonical_path(value: str | Path | None) -> str:
    if not value:
        return ""
    return str(Path(value).resolve())


def _path_name(value: str | Path | None) -> str:
    if not value:
        return ""
    return Path(value).name


def _safe_read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                {str(key): str(value or "") for key, value in row.items()}
                for row in reader
            ]
    except Exception:
        return []


def _expected_terminal_status(md_profile: str) -> str:
    return "analysis_complete" if md_profile == "full" else "dynamics_complete"


def _record_finished_local_action(
    *,
    run_root: Path,
    title: str,
    kind: str,
    related_run: str = "",
    related_sequence: str = "",
    related_campaign: str = "",
    output_path: str | Path | None = None,
    display_command: str = "",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    started_at = _now_iso()
    action = create_dashboard_action(
        run_root=run_root,
        title=title,
        kind=kind,
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-md-slate-local"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run,
        related_sequence=related_sequence,
        related_campaign=related_campaign,
        output_path=output_path,
        display_command=display_command or title,
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata=metadata or {},
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    from active_learning_thesis.dashboard_actions import _write_action  # local import to avoid wider surface

    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_rehearsal_event(
    slate: dict[str, object],
    *,
    sequence: str,
    stage: str,
    step: str,
    state: str,
    detail: str,
    remote_job_id: str = "",
) -> None:
    events = slate.setdefault("rehearsal_events", [])
    if not isinstance(events, list):
        events = []
        slate["rehearsal_events"] = events
    events.append(
        {
            "time": _now_iso(),
            "sequence": sequence,
            "stage": stage,
            "step": step,
            "state": state,
            "remote_job_id": remote_job_id,
            "detail": detail,
        }
    )


def _rehearsal_campaign_name(slate_id: str, sequence: str, md_profile: str) -> str:
    safe_sequence = "".join(ch for ch in sequence if ch.isalnum())[:12] or "peptide"
    return f"rehearsal_{slate_id}_{md_profile}_{safe_sequence}".lower()


def _write_rehearsal_campaign(
    *,
    run_root: Path,
    slate: dict[str, object],
    peptide: dict[str, object],
    md_profile: str,
    remote_job_id: str,
) -> Path:
    sequence = str(peptide.get("sequence", "")).strip()
    slate_id = str(slate.get("slate_id", "")).strip()
    campaign = _rehearsal_campaign_name(slate_id, sequence, md_profile)
    rehearsal_root = dashboard_md_slates_path(run_root).parent / "md_slate_rehearsals" / slate_id
    campaign_dir = rehearsal_root / campaign
    package_dir = campaign_dir / "packages" / sequence
    package_dir.mkdir(parents=True, exist_ok=True)
    expected_status = _expected_terminal_status(md_profile)
    meta = {
        "sequence": sequence,
        "md_profile": md_profile,
        "cluster": "bura",
        "campaign": campaign,
        "campaign_dir": str(campaign_dir),
        "expected_terminal_status": expected_status,
        "rehearsal": True,
        "rehearsal_slate_id": slate_id,
        "rehearsal_remote_job_id": remote_job_id,
        "created_at": _now_iso(),
    }
    (campaign_dir / STAGE_META_FILENAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _write_csv_rows(
        campaign_dir / "manifest.csv",
        [
            {
                "sequence": sequence,
                "round_id": "rehearsal",
                "acquisition_strategy": str(peptide.get("strategy", "rehearsal")),
                "campaign": campaign,
                "cluster": "bura",
                "md_profile": md_profile,
                "package_dir": f"packages/{sequence}",
                "pdb_status": "rehearsed",
            }
        ],
    )
    (package_dir / "rehearsal.log").write_text(
        "\n".join(
            [
                f"Rehearsal slate: {slate_id}",
                f"Sequence: {sequence}",
                f"Stage: {md_profile}",
                f"Fake BURA job id: {remote_job_id}",
                "This file is dashboard-generated rehearsal evidence only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sasa_file = ""
    ap_file = ""
    if md_profile == "full":
        sasa_path = package_dir / f"{sequence}_sasa_rehearsal.xvg"
        ap_path = package_dir / f"{sequence}_AP_SASA_rehearsal.txt"
        sasa_path.write_text("# rehearsal SASA evidence only\n0 0.0\n", encoding="utf-8")
        ap_path.write_text("# rehearsal AP evidence only\nAP_200ns 0.0\n", encoding="utf-8")
        sasa_file = f"packages/{sequence}/{sasa_path.name}"
        ap_file = f"packages/{sequence}/{ap_path.name}"
    _write_csv_rows(
        campaign_dir / "md_review.csv",
        [
            {
                "sequence": sequence,
                "round_id": "rehearsal",
                "campaign": campaign,
                "cluster": "bura",
                "md_profile": md_profile,
                "package_dir": f"packages/{sequence}",
                "job_root_status": expected_status,
                "ap_5ns": "",
                "ap_12ns": "",
                "ap_25ns": "",
                "ap_50ns": "",
                "ap_100ns": "",
                "ap_200ns": "",
                "sasa_file": sasa_file,
                "ap_file": ap_file,
                "review_notes": "Rehearsal only: no physical MD result was produced.",
                "cgmd_label": "",
            }
        ],
    )
    update_sync_status(
        run_root,
        cluster="bura",
        target_key=str(campaign_dir),
        status="finalized_local",
        related_run=str(slate.get("run_dir", "")),
        related_campaign=str(campaign_dir),
        related_sequence=sequence,
        remote_path=f"rehearsal://{slate_id}/{md_profile}/{sequence}",
        remote_job_id=remote_job_id,
        metadata={
            "rehearsal": True,
            "rehearsal_slate_id": slate_id,
            "expected_terminal_status": expected_status,
        },
    )
    return campaign_dir


def _default_campaign_name(run_dir: Path, sequence: str, md_profile: str) -> str:
    run_name = run_dir.name or "run"
    return f"{run_name}_{md_profile}_{sequence[:12]}".lower()


def find_md_source_batch_csv(run_root: Path, run_dir: Path, sequence: str) -> str:
    dashboard_batch = find_dashboard_md_source_batch(run_root, run_dir, sequence)
    if dashboard_batch:
        return dashboard_batch
    return find_run_md_source_batch(run_dir, sequence)


def _stage_history_entry(peptide: dict[str, object], md_profile: str) -> dict[str, object]:
    history = peptide.setdefault("stage_history", [])
    if not isinstance(history, list):
        history = []
        peptide["stage_history"] = history
    for entry in history:
        if isinstance(entry, dict) and str(entry.get("md_profile", "")) == md_profile:
            return entry
    entry = {
        "md_profile": md_profile,
        "campaign": "",
        "campaign_dir": "",
        "step": "prepare",
        "status": "pending",
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "completed_at": "",
        "last_action_id": "",
        "last_action_kind": "",
        "last_action_title": "",
        "last_action_status": "",
        "remote_job_id": "",
        "sync_status": "",
        "failure_reason": "",
        "waiting_reason": "",
        "review_ready": False,
    }
    history.append(entry)
    return entry


def _next_stage(md_profile: str) -> str:
    try:
        index = SLATE_STAGE_ORDER.index(md_profile)
    except ValueError:
        return ""
    if index + 1 >= len(SLATE_STAGE_ORDER):
        return ""
    return SLATE_STAGE_ORDER[index + 1]


def _campaign_for_stage(run_dir: Path, sequence: str, md_profile: str) -> dict[str, str]:
    campaigns_root = run_dir / "md_campaigns"
    best: tuple[int, dict[str, str]] | None = None
    if not campaigns_root.exists():
        return {}
    for campaign_dir in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
        meta = _safe_read_json(campaign_dir / STAGE_META_FILENAME)
        manifest_rows = _safe_read_csv(campaign_dir / "manifest.csv")
        review_rows = _safe_read_csv(campaign_dir / "md_review.csv")
        manifest_row = next(
            (row for row in manifest_rows if str(row.get("sequence", "")).strip() == sequence),
            None,
        )
        if manifest_row is None:
            continue
        profile = str(meta.get("md_profile", "") or manifest_row.get("md_profile", "")).strip()
        if profile != md_profile:
            continue
        review_row = next(
            (row for row in review_rows if str(row.get("sequence", "")).strip() == sequence),
            {},
        )
        status = str(review_row.get("job_root_status", "package_prepared")).strip()
        sync_record = next(
            (
                record for record in list_sync_records(run_dir.parent)
                if str(record.get("cluster", "")) == "bura"
                and _canonical_path(record.get("related_campaign")) == _canonical_path(campaign_dir)
            ),
            {},
        )
        sync_status = str(sync_record.get("status", "not_synced")).strip() or "not_synced"
        status_rank = {
            "analysis_complete": 5,
            "dynamics_complete": 4,
            "sasa_complete": 3,
            "package_prepared": 2,
            "pdb_missing": 1,
        }.get(status, 0)
        sync_rank = {
            "finalized_local": 7,
            "outputs_returned": 6,
            "outputs_staged": 5,
            "running": 4,
            "submitted": 3,
            "staged_remote": 2,
            "not_synced": 1,
        }.get(sync_status, 0)
        rank = status_rank * 10 + sync_rank
        candidate = {
            "campaign": campaign_dir.name,
            "campaign_dir": str(campaign_dir),
            "job_root_status": status,
            "review_path": str(campaign_dir / "md_review.csv"),
            "sync_status": sync_status,
            "remote_job_id": str(sync_record.get("remote_job_id", "")).strip(),
            "expected_terminal_status": str(meta.get("expected_terminal_status", "")).strip() or _expected_terminal_status(md_profile),
        }
        if best is None or rank >= best[0]:
            best = (rank, candidate)
    return best[1] if best is not None else {}


def _readiness_missing_tokens(action: dict[str, object]) -> list[str]:
    stdout_log = Path(str(action.get("stdout_log", "")))
    if not stdout_log.exists():
        return []
    text = stdout_log.read_text(encoding="utf-8", errors="replace")
    return sorted({token.strip() for token in text.split() if token.strip().endswith("_missing")})


def _failure_summary(action: dict[str, object]) -> str:
    title = str(action.get("title", "")).strip() or str(action.get("kind", "Dashboard action"))
    status = str(action.get("status", "")).strip() or "failed"
    stderr_log = Path(str(action.get("stderr_log", "")))
    excerpt = ""
    if stderr_log.exists():
        lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
        excerpt = "\n".join(lines[-3:]).strip()
    if excerpt:
        return f"{title} ended with {status}: {excerpt}"
    return f"{title} ended with {status}."


def _queue_summary_rows(queue_rows: list[dict[str, str]]) -> dict[str, int]:
    summary = {"pending": 0, "running": 0, "held": 0}
    for job in queue_rows:
        state = str(job.get("state", "")).upper()
        reason = str(job.get("reason", "")).lower()
        if state == "R":
            summary["running"] += 1
        else:
            summary["pending"] += 1
            if "held" in reason or "dependencynever" in reason:
                summary["held"] += 1
    return summary


def _active_slate_stage_counts(slates: list[dict[str, object]]) -> dict[str, int]:
    counts = {key: 0 for key in SLATE_STAGE_CAPS}
    for slate in slates:
        if not isinstance(slate, dict):
            continue
        if bool(slate.get("paused", False)) or str(slate.get("status", "")) in SLATE_FINAL_STATUSES:
            continue
        for peptide in list(slate.get("peptides", [])):
            if not isinstance(peptide, dict):
                continue
            if str(peptide.get("status", "")) != "active":
                continue
            current_stage = str(peptide.get("current_stage", "")).strip()
            current_step = str(peptide.get("current_step", "")).strip()
            if current_stage in counts and current_step in {"submit", "poll"}:
                counts[current_stage] += 1
    return counts


def _current_bura_utilization(run_root: Path, slates: list[dict[str, object]]) -> dict[str, object]:
    sync_records = list_sync_records(run_root)
    active_slate_campaigns = {
        _canonical_path(peptide.get("current_campaign_dir"))
        for slate in slates
        for peptide in slate.get("peptides", [])
        if isinstance(peptide, dict)
    }
    tracked_counts = {key: 0 for key in SLATE_STAGE_CAPS}
    tracked_rows: list[dict[str, str]] = []
    for record in sync_records:
        if str(record.get("cluster", "")) != "bura":
            continue
        if str(record.get("status", "")) not in {"submitted", "running"}:
            continue
        campaign_dir = str(record.get("related_campaign", "")).strip()
        if not campaign_dir:
            continue
        stage_state = _safe_read_json(Path(campaign_dir) / STAGE_META_FILENAME)
        md_profile = str(stage_state.get("md_profile", "")).strip()
        if md_profile in tracked_counts and _canonical_path(campaign_dir) not in active_slate_campaigns:
            tracked_counts[md_profile] += 1
            tracked_rows.append(
                {
                    "campaign": _path_name(campaign_dir),
                    "md_profile": md_profile,
                    "remote_job_id": str(record.get("remote_job_id", "")),
                    "sync_status": str(record.get("status", "")),
                }
            )
    snapshots = list_cluster_snapshots(run_root)
    latest_bura_snapshot = next((item for item in snapshots if str(item.get("cluster", "")) == "bura"), {})
    queue_jobs = latest_bura_snapshot.get("jobs", []) if isinstance(latest_bura_snapshot, dict) else []
    return {
        "caps": dict(SLATE_STAGE_CAPS),
        "tracked_external_counts": tracked_counts,
        "tracked_external_rows": tracked_rows,
        "snapshot_summary": _queue_summary_rows(queue_jobs if isinstance(queue_jobs, list) else []),
    }


def _resolve_launch_source_batch(run_root: Path, run_dir: Path, item: dict[str, str]) -> str:
    source_batch_csv = str(item.get("source_batch_csv", "")).strip()
    if source_batch_csv == "-":
        source_batch_csv = ""
    if source_batch_csv:
        return source_batch_csv
    return find_md_source_batch_csv(run_root, run_dir, str(item.get("sequence", "")))


def _source_batch_state(source_batch_csv: str, sequence: str) -> tuple[str, str]:
    if not source_batch_csv:
        return "missing", "No source batch CSV currently contains this peptide."
    source_batch_path = Path(source_batch_csv)
    if not source_batch_path.exists():
        return "missing", f"Saved source batch CSV no longer exists: {source_batch_path}"
    rows = _safe_read_csv(source_batch_path)
    if not any(str(row.get("sequence", "")).strip() == sequence for row in rows):
        return "missing", f"Source batch CSV exists but does not contain {sequence}: {source_batch_path}"
    return "ready", str(source_batch_path)


def _launch_action_sequence() -> str:
    return (
        "prepare-md-stage -> BURA upload -> readiness -> normalize -> preflight -> "
        "submit -> poll -> pull -> finalize -> auto-advance"
    )


def build_md_slate_launch_readiness(
    *,
    run_root: Path,
    run_dir: Path,
    run_name: str,
    peptides: list[dict[str, str]],
    profiles_payload: dict[str, object] | None = None,
    md_slates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    normalized_run_dir = _canonical_path(run_dir)
    run_path = Path(normalized_run_dir)
    slates = list(md_slates) if md_slates is not None else list_dashboard_md_slates(run_root)
    utilization = _current_bura_utilization(run_root, slates)
    active_counts = _active_slate_stage_counts(slates)
    tracked_counts = utilization.get("tracked_external_counts", {}) if isinstance(utilization, dict) else {}
    profiles = profiles_payload if profiles_payload is not None else load_cluster_profiles()
    bura_profile = get_cluster_profile(profiles, "bura") if isinstance(profiles, dict) else None
    bura_ready = bura_profile is not None
    run_config_path = run_path / "config.json"
    run_ready = run_path.exists() and run_config_path.exists()
    run_dir_mismatch = len(
        {
            _canonical_path(item.get("run_dir") or normalized_run_dir)
            for item in peptides
            if str(item.get("run_dir", "") or normalized_run_dir).strip()
        }
        - {normalized_run_dir}
    ) > 0

    cap_rows: list[dict[str, str]] = []
    available_by_stage: dict[str, int] = {}
    for stage, cap in SLATE_STAGE_CAPS.items():
        active = active_counts.get(stage, 0)
        tracked = int(tracked_counts.get(stage, 0) or 0)
        occupancy = active + tracked
        available = max(cap - occupancy, 0)
        planned = len(peptides) if stage == "line_smoke" else 0
        first_wave = min(planned, available) if stage == "line_smoke" else 0
        queued = max(planned - first_wave, 0) if stage == "line_smoke" else 0
        available_by_stage[stage] = available
        cap_rows.append(
            {
                "stage": stage,
                "cap": str(cap),
                "active_slate_jobs": str(active),
                "other_tracked_jobs": str(tracked),
                "occupancy_now": f"{occupancy} / {cap}",
                "available_now": str(available),
                "planned_new_jobs": str(planned),
                "can_start_in_first_wave": str(first_wave),
                "queued_behind_cap": str(queued),
            }
        )

    peptide_rows: list[dict[str, str]] = []
    launchable_peptides: list[dict[str, str]] = []
    seen_sequences: set[str] = set()
    for index, item in enumerate(peptides, start=1):
        sequence = str(item.get("sequence", "")).strip()
        source_batch_csv = _resolve_launch_source_batch(run_root, run_path, item)
        batch_state, batch_detail = _source_batch_state(source_batch_csv, sequence)
        default_campaign = _default_campaign_name(run_path, sequence or "peptide", "line_smoke")
        campaign_dir = run_path / "md_campaigns" / default_campaign
        blockers: list[str] = []
        warnings: list[str] = []
        if not sequence:
            blockers.append("Missing peptide sequence.")
        if sequence in seen_sequences:
            blockers.append("Duplicate sequence in this launch slate.")
        seen_sequences.add(sequence)
        if run_dir_mismatch:
            blockers.append("All peptides in one slate must belong to the same parent run.")
        if not run_ready:
            blockers.append(f"Run directory or config.json is missing: {run_config_path}")
        if not bura_ready:
            blockers.append("BURA cluster profile is not configured or enabled.")
        if batch_state != "ready":
            blockers.append(batch_detail)
        if campaign_dir.exists():
            blockers.append(f"Default line_smoke campaign already exists: {campaign_dir}")
        if available_by_stage.get("line_smoke", 0) <= 0:
            warnings.append("No BURA line_smoke submit capacity is free right now; launch can prepare locally but remote submit will wait.")
        elif index > available_by_stage.get("line_smoke", 0):
            warnings.append("This peptide will wait behind the line_smoke cap after local prepare/upload.")

        launch_state = "Launch-ready" if not blockers else "Blocked before launch"
        cap_state = "first wave" if index <= available_by_stage.get("line_smoke", 0) and not blockers else "queued/waiting"
        if blockers:
            next_move = blockers[0]
        elif warnings:
            next_move = warnings[0]
        else:
            next_move = "Ready for the first supervised slate child action after approval."
        row = {
            "sequence": sequence or "-",
            "run": run_name or run_path.name,
            "launch_state": launch_state,
            "current_stage": "line_smoke",
            "source_batch_state": batch_state,
            "source_batch_csv": source_batch_csv or "-",
            "campaign": default_campaign,
            "campaign_state": "will be created by prepare-md-stage" if not campaign_dir.exists() else "already exists",
            "package_state": "will be created by prepare-md-stage",
            "pdb_state": "will be built or validated during prepare-md-stage",
            "metadata_state": "will be written as md_stage_meta.json",
            "resource_request": "1 node | 1 task/node | 2 CPUs/task",
            "cap_state": cap_state,
            "first_child_action": "prepare-md-stage",
            "planned_child_actions": _launch_action_sequence(),
            "blocker": "; ".join(blockers) or "-",
            "warning": "; ".join(warnings) or "-",
            "next_move": next_move,
        }
        peptide_rows.append(row)
        if not blockers:
            launchable_peptides.append(
                {
                    "sequence": sequence,
                    "run_dir": normalized_run_dir,
                    "source_batch_csv": source_batch_csv,
                    "source": str(item.get("source", "")).strip(),
                    "strategy": str(item.get("strategy", "")).strip(),
                    "priority_band": str(item.get("priority_band", "")).strip(),
                }
            )

    blocked_rows = [row for row in peptide_rows if row["launch_state"] != "Launch-ready"]
    launchable_count = len(launchable_peptides)
    line_available = available_by_stage.get("line_smoke", 0)
    starts_now = min(launchable_count, line_available)
    queued_by_caps = max(launchable_count - starts_now, 0)
    if not peptides:
        verdict = "Do not launch yet"
        summary = "No peptides are selected for this MD slate."
    elif launchable_count == 0:
        verdict = "Do not launch yet"
        summary = "No selected peptide has all launch prerequisites."
    elif blocked_rows or queued_by_caps:
        verdict = "Partially ready"
        summary = (
            f"{launchable_count}/{len(peptides)} peptide(s) are launch-ready; "
            f"{len(blocked_rows)} blocked before launch; {queued_by_caps} will wait behind caps."
        )
    else:
        verdict = "Ready to launch"
        summary = f"All {launchable_count} peptide(s) pass the launch gate and fit the current line_smoke cap."

    return {
        "verdict": verdict,
        "summary": summary,
        "peptide_count": len(peptides),
        "launchable_count": launchable_count,
        "blocked_count": len(blocked_rows),
        "starts_now": starts_now,
        "queued_by_caps": queued_by_caps,
        "bura_profile_ready": "yes" if bura_ready else "no",
        "run_ready": "yes" if run_ready else "no",
        "cap_rows": cap_rows,
        "peptide_rows": peptide_rows,
        "launchable_peptides": launchable_peptides,
        "blocked_sequences": [str(row.get("sequence", "")) for row in blocked_rows],
        "child_actions_preview": [
            {
                "order": "1",
                "action": "prepare-md-stage",
                "scope": "local",
                "approval": "covered by slate approval",
                "notes": "One prepare action is scheduled per launch-ready peptide.",
            },
            {
                "order": "2+",
                "action": "BURA upload/readiness/normalize/preflight/submit/poll/pull/finalize",
                "scope": "bura/local",
                "approval": "covered by slate approval",
                "notes": "The supervisor advances each peptide independently while enforcing the 2/1/1 caps.",
            },
        ],
    }


def build_md_slate_rows(run_root: Path, actions: list[dict[str, object]]) -> list[dict[str, object]]:
    action_by_id = {str(action.get("id", "")): action for action in actions}
    rows: list[dict[str, object]] = []
    for slate in list_dashboard_md_slates(run_root):
        supervisor_action = action_by_id.get(str(slate.get("supervisor_action_id", "")), {})
        status = str(slate.get("status", "awaiting_approval"))
        if bool(slate.get("paused", False)):
            status = "paused"
        elif str(supervisor_action.get("status", "")) in {"awaiting_approval", "queued", "running"}:
            status = str(supervisor_action.get("status", status))
        elif str(supervisor_action.get("status", "")) in {"cancelled", "manual_override"} and status not in SLATE_FINAL_STATUSES:
            status = "cancelled"
        elif str(supervisor_action.get("status", "")) == "failed" and status not in SLATE_FINAL_STATUSES:
            status = "failed"
        peptides = list(slate.get("peptides", [])) if isinstance(slate.get("peptides", []), list) else []
        rows.append(
            {
                **slate,
                "effective_status": status,
                "peptide_count": len(peptides),
                "completed_count": sum(1 for item in peptides if str(item.get("status", "")) == "completed"),
                "blocked_count": sum(1 for item in peptides if str(item.get("status", "")) == "blocked"),
                "active_count": sum(1 for item in peptides if str(item.get("status", "")) == "active"),
                "review_ready_count": sum(1 for item in peptides if bool(item.get("review_ready", False))),
            }
        )
    return rows


def build_md_slate_monitor_rows(slate: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for peptide in slate.get("peptides", []):
        if not isinstance(peptide, dict):
            continue
        rows.append(
            {
                "sequence": str(peptide.get("sequence", "")),
                "mode": str(slate.get("execution_mode", "live")) or "live",
                "stage": str(peptide.get("current_stage", "")) or "-",
                "step": str(peptide.get("current_step", "")) or "-",
                "state": str(peptide.get("status", "")) or "-",
                "campaign": str(peptide.get("current_campaign", "")) or "-",
                "remote_job_id": str(peptide.get("remote_job_id", "")) or "-",
                "last_action": str(peptide.get("last_action_title", "")) or "-",
                "blocked_reason": str(peptide.get("failure_reason", "")) or "-",
                "waiting_reason": str(peptide.get("waiting_reason", "")) or "-",
                "review_ready": "yes" if bool(peptide.get("review_ready", False)) else "no",
            }
        )
    return rows


def latest_run_md_slate(run_root: Path, actions: list[dict[str, object]], run_dir: str) -> dict[str, object] | None:
    run_key = _canonical_path(run_dir)
    for slate in build_md_slate_rows(run_root, actions):
        if _canonical_path(slate.get("run_dir")) == run_key:
            return slate
    return None


def build_bura_utilization_summary(run_root: Path, actions: list[dict[str, object]], *, run_dir: str = "") -> dict[str, object]:
    slate_rows = build_md_slate_rows(run_root, actions)
    if run_dir:
        run_key = _canonical_path(run_dir)
        slate_rows = [row for row in slate_rows if _canonical_path(row.get("run_dir")) == run_key]
    return _current_bura_utilization(run_root, slate_rows)


def draft_md_slate_run_action(
    *,
    run_root: Path,
    run_dir: Path,
    run_name: str,
    peptides: list[dict[str, str]],
    planner_id: str = "",
    planner_name: str = "",
    operator_note: str = "",
) -> dict[str, object]:
    if not peptides:
        raise ValueError("Select at least one peptide for the MD slate first.")
    normalized_run_dir = _canonical_path(run_dir)
    if len({_canonical_path(item.get("run_dir")) for item in peptides}) != 1:
        raise ValueError("An MD slate can only contain peptides from one parent run at a time.")
    readiness = build_md_slate_launch_readiness(
        run_root=run_root,
        run_dir=Path(normalized_run_dir),
        run_name=run_name,
        peptides=peptides,
    )
    blocked_rows = [
        row
        for row in list(readiness.get("peptide_rows", []))
        if isinstance(row, dict) and str(row.get("launch_state", "")) != "Launch-ready"
    ]
    if blocked_rows:
        missing_batch_sequences = [
            str(row.get("sequence", "")).strip()
            for row in blocked_rows
            if "source batch CSV" in str(row.get("blocker", ""))
        ]
        if missing_batch_sequences:
            preview = ", ".join(missing_batch_sequences[:6])
            if len(missing_batch_sequences) > 6:
                preview += f", +{len(missing_batch_sequences) - 6} more"
            raise ValueError(
                "Cannot launch the MD slate yet because no source batch CSV contains: "
                f"{preview}. Export or create a batch CSV row for those peptides before starting MD."
            )
        preview = "; ".join(
            f"{row.get('sequence', '-')}: {row.get('blocker', 'not launch-ready')}"
            for row in blocked_rows[:4]
        )
        if len(blocked_rows) > 4:
            preview += f"; +{len(blocked_rows) - 4} more"
        raise ValueError(
            "Cannot launch the MD slate yet because the dry-run gate found blockers: "
            f"{preview}"
        )
    launch_validation = [
        (item, str(item.get("source_batch_csv", "")).strip())
        for item in list(readiness.get("launchable_peptides", []))
        if isinstance(item, dict)
    ]
    slate_id = uuid.uuid4().hex[:12]
    slate = {
        "slate_id": slate_id,
        "run_dir": normalized_run_dir,
        "run_name": run_name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "awaiting_approval",
        "paused": False,
        "supervisor_action_id": "",
        "planner_id": str(planner_id).strip(),
        "planner_name": str(planner_name).strip(),
        "operator_note": str(operator_note).strip(),
        "caps": dict(SLATE_STAGE_CAPS),
        "launch_readiness": {
            "verdict": str(readiness.get("verdict", "")),
            "summary": str(readiness.get("summary", "")),
            "starts_now": int(readiness.get("starts_now", 0) or 0),
            "queued_by_caps": int(readiness.get("queued_by_caps", 0) or 0),
        },
        "peptides": [
            {
                "sequence": str(item.get("sequence", "")).strip(),
                "run_dir": normalized_run_dir,
                "run_name": run_name,
                "source_batch_csv": source_batch_csv,
                "source": str(item.get("source", "")).strip(),
                "strategy": str(item.get("strategy", "")).strip(),
                "priority_band": str(item.get("priority_band", "")).strip(),
                "status": "pending",
                "current_stage": "line_smoke",
                "current_step": "prepare",
                "current_campaign": "",
                "current_campaign_dir": "",
                "last_action_id": "",
                "last_action_kind": "",
                "last_action_title": "",
                "last_action_status": "",
                "last_update_at": _now_iso(),
                "remote_job_id": "",
                "failure_reason": "",
                "waiting_reason": "",
                "blocked_stage": "",
                "review_ready": False,
                "poll_not_before": "",
                "stage_history": [],
            }
            for item, source_batch_csv in launch_validation
        ],
    }
    save_dashboard_md_slate(run_root, slate)
    preview = ", ".join(str(item.get("sequence", "")) for item in peptides[:6])
    if len(peptides) > 6:
        preview += f", +{len(peptides) - 6} more"
    action = draft_dashboard_action(
        run_root=run_root,
        title=f"Launch MD slate for {Path(normalized_run_dir).name} ({len(peptides)} peptides)",
        kind="md-slate-run",
        command=sys.executable,
        args=[
            "-m",
            "active_learning_thesis",
            "dashboard-run-md-slate",
            "--run-root",
            str(run_root),
            "--slate-id",
            slate_id,
        ],
        cwd=REPO_ROOT,
        scope="bura",
        cluster="bura",
        related_run=normalized_run_dir,
        display_command=f"dashboard-managed MD slate ({preview})",
        metadata={
            "slate_id": slate_id,
            "peptide_count": len(peptides),
            "planner_id": str(planner_id).strip(),
            "planner_name": str(planner_name).strip(),
            "operator_note": str(operator_note).strip(),
            "launch_readiness_verdict": str(readiness.get("verdict", "")),
            "launch_readiness_summary": str(readiness.get("summary", "")),
        },
        exclusive=False,
    )
    slate["supervisor_action_id"] = str(action.get("id", ""))
    save_dashboard_md_slate(run_root, slate)
    return action


def _normalize_rehearsal_failure_plan(failure_plan: dict[str, object] | None) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    if not isinstance(failure_plan, dict):
        return normalized
    for sequence, raw_rule in failure_plan.items():
        sequence_text = str(sequence).strip()
        if not sequence_text:
            continue
        if isinstance(raw_rule, dict):
            stage = str(raw_rule.get("stage", "")).strip() or "line_smoke"
            step = str(raw_rule.get("step", "")).strip() or "submit"
            reason = str(raw_rule.get("reason", "")).strip()
        else:
            stage = str(raw_rule).strip() or "line_smoke"
            step = "submit"
            reason = ""
        normalized[sequence_text] = {
            "stage": stage,
            "step": step,
            "reason": reason or f"Rehearsal injected failure at {stage}/{step}.",
        }
    return normalized


def _rehearsal_failure_matches(rule: dict[str, str] | None, *, md_profile: str, step: str) -> bool:
    if not rule:
        return False
    expected_stage = str(rule.get("stage", "")).strip()
    expected_step = str(rule.get("step", "")).strip()
    if expected_stage and expected_stage != md_profile:
        return False
    if expected_step == step:
        return True
    return expected_step == "poll" and step.startswith("poll_")


def _run_rehearsal_stage(
    *,
    run_root: Path,
    slate: dict[str, object],
    peptide: dict[str, object],
    md_profile: str,
    peptide_index: int,
    failure_rule: dict[str, str] | None,
    log: Callable[[str], None],
) -> bool:
    sequence = str(peptide.get("sequence", "")).strip()
    slate_id = str(slate.get("slate_id", "")).strip()
    remote_job_id = f"rehearsal-{slate_id[:6]}-{SLATE_STAGE_ORDER.index(md_profile) + 1}-{peptide_index + 1}"
    campaign_dir: Path | None = None
    for step in REHEARSAL_STEPS:
        if step == "prepare":
            campaign_dir = _write_rehearsal_campaign(
                run_root=run_root,
                slate=slate,
                peptide=peptide,
                md_profile=md_profile,
                remote_job_id=remote_job_id,
            )
            peptide["current_campaign"] = campaign_dir.name
            peptide["current_campaign_dir"] = str(campaign_dir)
        if step == "submit":
            peptide["remote_job_id"] = remote_job_id
        state = "running" if step == "poll_running" else "ok"
        detail = f"Rehearsed {step.replace('_', ' ')} for {sequence} without touching BURA."
        _append_rehearsal_event(
            slate,
            sequence=sequence,
            stage=md_profile,
            step=step,
            state=state,
            detail=detail,
            remote_job_id=remote_job_id if step in {"submit", "poll_running", "poll_finished"} else "",
        )
        if _rehearsal_failure_matches(failure_rule, md_profile=md_profile, step=step):
            action = _record_finished_local_action(
                run_root=run_root,
                title=f"Rehearsal blocked {md_profile} for {sequence}",
                kind="md-slate-rehearsal-stage",
                related_run=str(slate.get("run_dir", "")),
                related_sequence=sequence,
                related_campaign=str(campaign_dir or ""),
                output_path=campaign_dir or None,
                display_command=f"rehearsal failure: {sequence} {md_profile}/{step}",
                metadata={
                    "rehearsal": True,
                    "slate_id": slate_id,
                    "sequence": sequence,
                    "md_profile": md_profile,
                    "step": step,
                    "remote_job_id": remote_job_id,
                },
            )
            failure_reason = str((failure_rule or {}).get("reason", "")).strip() or f"Rehearsal injected failure at {md_profile}/{step}."
            _set_stage_step(
                peptide,
                md_profile=md_profile,
                step=step,
                status="blocked",
                action={**action, "remote_job_id": remote_job_id, "sync_status": "rehearsal_failed"},
                failure_reason=failure_reason,
            )
            _append_rehearsal_event(
                slate,
                sequence=sequence,
                stage=md_profile,
                step=step,
                state="blocked",
                detail=failure_reason,
                remote_job_id=remote_job_id,
            )
            log(f"{sequence}: rehearsal blocked at {md_profile}/{step}")
            return False

    action = _record_finished_local_action(
        run_root=run_root,
        title=f"Rehearsed {md_profile} for {sequence}",
        kind="md-slate-rehearsal-stage",
        related_run=str(slate.get("run_dir", "")),
        related_sequence=sequence,
        related_campaign=str(campaign_dir or ""),
        output_path=campaign_dir or None,
        display_command=f"rehearsal: {sequence} {md_profile}",
        metadata={
            "rehearsal": True,
            "slate_id": slate_id,
            "sequence": sequence,
            "md_profile": md_profile,
            "remote_job_id": remote_job_id,
        },
    )
    stage_entry = _stage_history_entry(peptide, md_profile)
    stage_entry["campaign"] = (campaign_dir.name if campaign_dir else "") or str(peptide.get("current_campaign", ""))
    stage_entry["campaign_dir"] = str(campaign_dir or peptide.get("current_campaign_dir", ""))
    stage_entry["step"] = "finalize"
    stage_entry["status"] = "completed"
    stage_entry["updated_at"] = _now_iso()
    stage_entry["completed_at"] = _now_iso()
    stage_entry["last_action_id"] = str(action.get("id", ""))
    stage_entry["last_action_kind"] = str(action.get("kind", ""))
    stage_entry["last_action_title"] = str(action.get("title", ""))
    stage_entry["last_action_status"] = str(action.get("status", ""))
    stage_entry["remote_job_id"] = remote_job_id
    stage_entry["sync_status"] = "finalized_local"
    stage_entry["review_ready"] = bool(md_profile == "full")
    peptide["last_action_id"] = str(action.get("id", ""))
    peptide["last_action_kind"] = str(action.get("kind", ""))
    peptide["last_action_title"] = str(action.get("title", ""))
    peptide["last_action_status"] = str(action.get("status", ""))
    peptide["last_update_at"] = _now_iso()
    peptide["waiting_reason"] = ""
    peptide["failure_reason"] = ""
    peptide["blocked_stage"] = ""
    if md_profile == "full":
        peptide["status"] = "completed"
        peptide["current_step"] = ""
        peptide["review_ready"] = True
        peptide["remote_job_id"] = remote_job_id
    else:
        peptide["status"] = "pending"
        peptide["current_stage"] = _next_stage(md_profile)
        peptide["current_step"] = "prepare"
        peptide["current_campaign"] = ""
        peptide["current_campaign_dir"] = ""
        peptide["remote_job_id"] = ""
    log(f"{sequence}: rehearsal completed {md_profile}")
    return True


def run_md_slate_rehearsal(
    run_root: Path,
    slate_id: str,
    *,
    failure_plan: dict[str, object] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, object]:
    logger = log or (lambda message: None)
    slate = load_dashboard_md_slate(run_root, slate_id)
    if str(slate.get("execution_mode", "")) != "rehearsal":
        raise ValueError(f"Slate is not a rehearsal slate: {slate_id}")
    if bool(slate.get("paused", False)):
        slate["status"] = "paused"
        slate["updated_at"] = _now_iso()
        return save_dashboard_md_slate(run_root, slate)
    failure_rules = _normalize_rehearsal_failure_plan(failure_plan)
    peptides = [peptide for peptide in list(slate.get("peptides", [])) if isinstance(peptide, dict)]
    max_active = {key: 0 for key in SLATE_STAGE_CAPS}
    slate["status"] = "running"
    slate["rehearsal_events"] = []
    for md_profile in SLATE_STAGE_ORDER:
        stage_peptides = [
            peptide
            for peptide in peptides
            if str(peptide.get("status", "")) not in PEPTIDE_FINAL_STATUSES
            and str(peptide.get("current_stage", "")).strip() == md_profile
        ]
        cap = int(SLATE_STAGE_CAPS.get(md_profile, 1))
        for wave_index in range(0, len(stage_peptides), cap):
            wave = stage_peptides[wave_index:wave_index + cap]
            if not wave:
                continue
            max_active[md_profile] = max(max_active.get(md_profile, 0), len(wave))
            _append_rehearsal_event(
                slate,
                sequence="slate",
                stage=md_profile,
                step="capacity_wave",
                state="ok",
                detail=f"Rehearsed capacity wave {wave_index // cap + 1}: {len(wave)} active of cap {cap}.",
            )
            for peptide in wave:
                sequence = str(peptide.get("sequence", "")).strip()
                peptide_index = peptides.index(peptide)
                _run_rehearsal_stage(
                    run_root=run_root,
                    slate=slate,
                    peptide=peptide,
                    md_profile=md_profile,
                    peptide_index=peptide_index,
                    failure_rule=failure_rules.get(sequence),
                    log=logger,
                )
    statuses = [str(peptide.get("status", "")) for peptide in peptides]
    if statuses and all(status in {"completed", "skipped"} for status in statuses):
        slate["status"] = "completed"
    elif any(status == "blocked" for status in statuses):
        slate["status"] = "completed_with_failures"
    else:
        slate["status"] = "running"
    slate["rehearsal_summary"] = {
        "mode": "rehearsal",
        "peptides": len(peptides),
        "completed": sum(1 for status in statuses if status == "completed"),
        "blocked": sum(1 for status in statuses if status == "blocked"),
        "review_ready": sum(1 for peptide in peptides if bool(peptide.get("review_ready", False))),
        "max_active_line_smoke": max_active.get("line_smoke", 0),
        "max_active_production_smoke": max_active.get("production_smoke", 0),
        "max_active_full": max_active.get("full", 0),
        "touched_remote_clusters": "no",
        "cgmd_label_assigned": "no",
    }
    slate["updated_at"] = _now_iso()
    return save_dashboard_md_slate(run_root, slate)


def launch_md_slate_rehearsal_action(
    *,
    run_root: Path,
    run_dir: Path,
    run_name: str,
    peptides: list[dict[str, str]],
    planner_id: str = "",
    planner_name: str = "",
    operator_note: str = "",
    failure_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    if not peptides:
        raise ValueError("Select at least one peptide for the MD rehearsal first.")
    normalized_run_dir = _canonical_path(run_dir)
    if len({_canonical_path(item.get("run_dir")) for item in peptides}) != 1:
        raise ValueError("An MD rehearsal slate can only contain peptides from one parent run at a time.")
    launch_validation: list[tuple[dict[str, str], str]] = []
    missing_sequences: list[str] = []
    for item in peptides:
        sequence = str(item.get("sequence", "")).strip()
        source_batch_csv = _resolve_launch_source_batch(run_root, Path(normalized_run_dir), item)
        batch_state, _batch_detail = _source_batch_state(source_batch_csv, sequence)
        if batch_state != "ready":
            missing_sequences.append(sequence or "-")
            continue
        launch_validation.append((item, source_batch_csv))
    if missing_sequences:
        preview = ", ".join(missing_sequences[:6])
        if len(missing_sequences) > 6:
            preview += f", +{len(missing_sequences) - 6} more"
        raise ValueError(f"Cannot rehearse the MD slate because no source batch CSV contains: {preview}.")
    slate_id = uuid.uuid4().hex[:12]
    slate = {
        "slate_id": slate_id,
        "run_dir": normalized_run_dir,
        "run_name": run_name,
        "execution_mode": "rehearsal",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "running",
        "paused": False,
        "supervisor_action_id": "",
        "planner_id": str(planner_id).strip(),
        "planner_name": str(planner_name).strip(),
        "operator_note": str(operator_note).strip(),
        "caps": dict(SLATE_STAGE_CAPS),
        "rehearsal_summary": {},
        "rehearsal_events": [],
        "peptides": [
            {
                "sequence": str(item.get("sequence", "")).strip(),
                "run_dir": normalized_run_dir,
                "run_name": run_name,
                "source_batch_csv": source_batch_csv,
                "source": str(item.get("source", "")).strip(),
                "strategy": str(item.get("strategy", "")).strip(),
                "priority_band": str(item.get("priority_band", "")).strip(),
                "status": "pending",
                "current_stage": "line_smoke",
                "current_step": "prepare",
                "current_campaign": "",
                "current_campaign_dir": "",
                "last_action_id": "",
                "last_action_kind": "",
                "last_action_title": "",
                "last_action_status": "",
                "last_update_at": _now_iso(),
                "remote_job_id": "",
                "failure_reason": "",
                "waiting_reason": "",
                "blocked_stage": "",
                "review_ready": False,
                "poll_not_before": "",
                "stage_history": [],
            }
            for item, source_batch_csv in launch_validation
        ],
    }
    save_dashboard_md_slate(run_root, slate)
    preview = ", ".join(str(item.get("sequence", "")) for item in peptides[:6])
    if len(peptides) > 6:
        preview += f", +{len(peptides) - 6} more"
    action = _record_finished_local_action(
        run_root=run_root,
        title=f"Run MD slate rehearsal for {Path(normalized_run_dir).name} ({len(peptides)} peptides)",
        kind="md-slate-rehearsal",
        related_run=normalized_run_dir,
        display_command=f"dashboard rehearsal MD slate ({preview})",
        metadata={
            "slate_id": slate_id,
            "peptide_count": len(peptides),
            "planner_id": str(planner_id).strip(),
            "planner_name": str(planner_name).strip(),
            "operator_note": str(operator_note).strip(),
            "rehearsal": True,
            "touches_remote_clusters": False,
            "assigns_cgmd_label": False,
        },
    )
    slate["supervisor_action_id"] = str(action.get("id", ""))
    save_dashboard_md_slate(run_root, slate)
    saved = run_md_slate_rehearsal(run_root, slate_id, failure_plan=failure_plan)
    updated_metadata = dict(action.get("metadata", {})) if isinstance(action.get("metadata", {}), dict) else {}
    updated_metadata["slate_status"] = str(saved.get("status", ""))
    updated_metadata["review_ready_count"] = int(saved.get("rehearsal_summary", {}).get("review_ready", 0)) if isinstance(saved.get("rehearsal_summary", {}), dict) else 0
    action = update_dashboard_action(run_root, str(action.get("id", "")), metadata=updated_metadata)
    return action


def pause_md_slate(run_root: Path, slate_id: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    slate["paused"] = True
    slate["status"] = "paused"
    slate["updated_at"] = _now_iso()
    supervisor_action_id = str(slate.get("supervisor_action_id", "")).strip()
    if supervisor_action_id:
        try:
            supervisor_action = load_dashboard_action(run_root, supervisor_action_id)
        except Exception:
            supervisor_action = {}
        supervisor_status = str(supervisor_action.get("status", "")).strip()
        if supervisor_status and supervisor_status != "paused":
            if supervisor_status == "running":
                update_dashboard_action(run_root, supervisor_action_id, status="paused")
            elif supervisor_status not in FINAL_ACTION_STATUSES:
                from active_learning_thesis.dashboard_actions import pause_dashboard_action  # local import keeps the module surface modest

                pause_dashboard_action(run_root, supervisor_action_id)
    saved = save_dashboard_md_slate(run_root, slate)
    _record_finished_local_action(
        run_root=run_root,
        title=f"Pause MD slate {slate_id}",
        kind="md-slate-pause",
        related_run=str(saved.get("run_dir", "")),
        display_command=f"pause md slate {slate_id}",
        metadata={"slate_id": slate_id},
    )
    return saved


def resume_md_slate(run_root: Path, slate_id: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    slate["paused"] = False
    if str(slate.get("status", "")) in {"paused", "cancelled"}:
        slate["status"] = "running"
    slate["updated_at"] = _now_iso()
    supervisor_action_id = str(slate.get("supervisor_action_id", "")).strip()
    if supervisor_action_id:
        try:
            supervisor_action = load_dashboard_action(run_root, supervisor_action_id)
        except Exception:
            supervisor_action = {}
        if str(supervisor_action.get("status", "")).strip() == "paused":
            resume_dashboard_action(run_root, supervisor_action_id)
    saved = save_dashboard_md_slate(run_root, slate)
    _record_finished_local_action(
        run_root=run_root,
        title=f"Resume MD slate {slate_id}",
        kind="md-slate-resume",
        related_run=str(saved.get("run_dir", "")),
        display_command=f"resume md slate {slate_id}",
        metadata={"slate_id": slate_id},
    )
    return saved


def stop_blocked_md_slate_peptide(run_root: Path, slate_id: str, sequence: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    for peptide in slate.get("peptides", []):
        if str(peptide.get("sequence", "")) != sequence:
            continue
        peptide["status"] = "skipped"
        peptide["last_update_at"] = _now_iso()
        peptide["waiting_reason"] = ""
        peptide["failure_reason"] = peptide.get("failure_reason", "") or "Skipped after blocking."
        slate["updated_at"] = _now_iso()
        saved = save_dashboard_md_slate(run_root, slate)
        _record_finished_local_action(
            run_root=run_root,
            title=f"Skip blocked MD peptide {sequence}",
            kind="md-slate-skip-peptide",
            related_run=str(saved.get("run_dir", "")),
            related_sequence=sequence,
            display_command=f"skip blocked peptide {sequence} in slate {slate_id}",
            metadata={"slate_id": slate_id, "sequence": sequence},
        )
        return saved
    raise ValueError(f"Sequence {sequence} is not part of MD slate {slate_id}.")


def retry_blocked_md_slate_peptide(run_root: Path, slate_id: str, sequence: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    for peptide in slate.get("peptides", []):
        if str(peptide.get("sequence", "")) != sequence:
            continue
        current_stage = str(peptide.get("blocked_stage", "")).strip() or str(peptide.get("current_stage", "")).strip() or "line_smoke"
        current_campaign_dir = str(peptide.get("current_campaign_dir", "")).strip()
        restart_step = "prepare"
        if current_campaign_dir and Path(current_campaign_dir).exists():
            restart_step = str(peptide.get("current_step", "")).strip() or "upload"
        peptide["status"] = "pending"
        peptide["current_stage"] = current_stage
        peptide["current_step"] = restart_step
        peptide["last_action_id"] = ""
        peptide["last_action_kind"] = ""
        peptide["last_action_title"] = ""
        peptide["last_action_status"] = ""
        peptide["failure_reason"] = ""
        peptide["waiting_reason"] = ""
        peptide["blocked_stage"] = ""
        peptide["poll_not_before"] = ""
        peptide["last_update_at"] = _now_iso()
        stage_entry = _stage_history_entry(peptide, current_stage)
        stage_entry["status"] = "pending"
        stage_entry["step"] = restart_step
        stage_entry["failure_reason"] = ""
        stage_entry["waiting_reason"] = ""
        stage_entry["updated_at"] = _now_iso()
        slate["updated_at"] = _now_iso()
        if str(slate.get("status", "")) == "completed_with_failures":
            slate["status"] = "running"
        saved = save_dashboard_md_slate(run_root, slate)
        _record_finished_local_action(
            run_root=run_root,
            title=f"Retry blocked MD peptide {sequence}",
            kind="md-slate-retry-peptide",
            related_run=str(saved.get("run_dir", "")),
            related_sequence=sequence,
            related_campaign=current_campaign_dir,
            display_command=f"retry blocked peptide {sequence} in slate {slate_id}",
            metadata={"slate_id": slate_id, "sequence": sequence, "restart_step": restart_step},
        )
        return saved
    raise ValueError(f"Sequence {sequence} is not part of MD slate {slate_id}.")


def recover_md_slate_peptide(run_root: Path, slate_id: str, sequence: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    for peptide in slate.get("peptides", []):
        if str(peptide.get("sequence", "")) != sequence:
            continue
        if bool(peptide.get("review_ready", False)) or str(peptide.get("status", "")) in {"completed", "skipped"}:
            raise ValueError(f"{sequence} is already terminal in MD slate {slate_id}.")
        if str(peptide.get("remote_job_id", "")).strip():
            raise ValueError(
                f"{sequence} still has a tracked remote job id. Rebind the peptide to the latest tracked action instead of resetting the stage."
            )
        if str(peptide.get("status", "")) == "active" and str(peptide.get("last_action_id", "")).strip():
            raise ValueError(
                f"{sequence} still has an active child action. Rebind it first, or wait for that action to finish before recovering the stage."
            )
        current_stage = str(peptide.get("blocked_stage", "")).strip() or str(peptide.get("current_stage", "")).strip() or "line_smoke"
        current_campaign_dir = str(peptide.get("current_campaign_dir", "")).strip()
        restart_step = "prepare"
        if current_campaign_dir and Path(current_campaign_dir).exists():
            restart_step = str(peptide.get("current_step", "")).strip() or "upload"
        peptide["status"] = "pending"
        peptide["current_stage"] = current_stage
        peptide["current_step"] = restart_step
        peptide["last_action_id"] = ""
        peptide["last_action_kind"] = ""
        peptide["last_action_title"] = ""
        peptide["last_action_status"] = ""
        peptide["failure_reason"] = ""
        peptide["waiting_reason"] = ""
        peptide["blocked_stage"] = ""
        peptide["poll_not_before"] = ""
        peptide["last_update_at"] = _now_iso()
        stage_entry = _stage_history_entry(peptide, current_stage)
        stage_entry["status"] = "pending"
        stage_entry["step"] = restart_step
        stage_entry["failure_reason"] = ""
        stage_entry["waiting_reason"] = ""
        stage_entry["updated_at"] = _now_iso()
        slate["updated_at"] = _now_iso()
        if str(slate.get("status", "")) in {"completed_with_failures", "paused"}:
            slate["status"] = "running"
        saved = save_dashboard_md_slate(run_root, slate)
        _record_finished_local_action(
            run_root=run_root,
            title=f"Recover MD peptide {sequence}",
            kind="md-slate-recover-peptide",
            related_run=str(saved.get("run_dir", "")),
            related_sequence=sequence,
            related_campaign=current_campaign_dir,
            display_command=f"recover peptide {sequence} in slate {slate_id}",
            metadata={"slate_id": slate_id, "sequence": sequence, "restart_step": restart_step},
        )
        return saved
    raise ValueError(f"Sequence {sequence} is not part of MD slate {slate_id}.")


def rebind_md_slate_peptide(run_root: Path, slate_id: str, sequence: str) -> dict[str, object]:
    slate = load_dashboard_md_slate(run_root, slate_id)
    actions = list_dashboard_actions(run_root)
    for peptide in slate.get("peptides", []):
        if str(peptide.get("sequence", "")) != sequence:
            continue
        candidate = find_md_slate_rebind_candidate(actions, slate, peptide)
        if not candidate:
            raise ValueError(f"No tracked child action could be found to rebind {sequence} inside MD slate {slate_id}.")
        peptide["last_action_id"] = str(candidate.get("id", ""))
        peptide["last_action_kind"] = str(candidate.get("kind", ""))
        peptide["last_action_title"] = str(candidate.get("title", ""))
        peptide["last_action_status"] = ""
        peptide["waiting_reason"] = ""
        peptide["failure_reason"] = ""
        peptide["last_update_at"] = _now_iso()
        remote_job_id = str(candidate.get("remote_job_id", "")).strip()
        if remote_job_id:
            peptide["remote_job_id"] = remote_job_id
        if str(candidate.get("kind", "")) == "prepare-md-stage":
            campaign_dir = str(candidate.get("output_path", "")).strip()
            if campaign_dir:
                peptide["current_campaign_dir"] = campaign_dir
                peptide["current_campaign"] = _path_name(campaign_dir)
        if str(candidate.get("status", "")) in ACTIVE_ACTION_STATUSES:
            peptide["status"] = "active"
        else:
            peptide["status"] = "pending"
            peptide["waiting_reason"] = "Waiting for the slate supervisor to resync with the rebound action."
        stage_entry = _stage_history_entry(peptide, str(peptide.get("current_stage", "")).strip() or "line_smoke")
        stage_entry["last_action_id"] = str(candidate.get("id", ""))
        stage_entry["last_action_kind"] = str(candidate.get("kind", ""))
        stage_entry["last_action_title"] = str(candidate.get("title", ""))
        stage_entry["last_action_status"] = ""
        stage_entry["remote_job_id"] = remote_job_id or str(stage_entry.get("remote_job_id", ""))
        stage_entry["updated_at"] = _now_iso()
        slate["updated_at"] = _now_iso()
        if str(slate.get("status", "")) == "completed_with_failures":
            slate["status"] = "running"
        saved = save_dashboard_md_slate(run_root, slate)
        _record_finished_local_action(
            run_root=run_root,
            title=f"Rebind MD peptide {sequence}",
            kind="md-slate-rebind-peptide",
            related_run=str(saved.get("run_dir", "")),
            related_sequence=sequence,
            related_campaign=str(peptide.get("current_campaign_dir", "")),
            display_command=f"rebind peptide {sequence} in slate {slate_id}",
            metadata={
                "slate_id": slate_id,
                "sequence": sequence,
                "rebound_action_id": str(candidate.get("id", "")),
            },
        )
        return saved
    raise ValueError(f"Sequence {sequence} is not part of MD slate {slate_id}.")


def _set_stage_step(
    peptide: dict[str, object],
    *,
    md_profile: str,
    step: str,
    status: str,
    action: dict[str, object] | None = None,
    failure_reason: str = "",
) -> None:
    stage_entry = _stage_history_entry(peptide, md_profile)
    stage_entry["step"] = step
    stage_entry["status"] = status
    stage_entry["updated_at"] = _now_iso()
    if action is not None:
        stage_entry["last_action_id"] = str(action.get("id", ""))
        stage_entry["last_action_kind"] = str(action.get("kind", ""))
        stage_entry["last_action_title"] = str(action.get("title", ""))
        stage_entry["last_action_status"] = str(action.get("status", ""))
        stage_entry["remote_job_id"] = str(action.get("remote_job_id", "") or peptide.get("remote_job_id", ""))
        stage_entry["sync_status"] = str(action.get("sync_status", "") or stage_entry.get("sync_status", ""))
    stage_entry["failure_reason"] = failure_reason if status == "blocked" else ""
    stage_entry["waiting_reason"] = ""
    peptide["current_stage"] = md_profile
    peptide["current_step"] = step
    peptide["status"] = status
    peptide["waiting_reason"] = ""
    if status == "blocked":
        peptide["failure_reason"] = failure_reason
        peptide["blocked_stage"] = md_profile
    else:
        peptide["failure_reason"] = ""
        if str(peptide.get("blocked_stage", "")) == md_profile:
            peptide["blocked_stage"] = ""
    peptide["last_update_at"] = _now_iso()
    if action is not None:
        peptide["last_action_id"] = str(action.get("id", ""))
        peptide["last_action_kind"] = str(action.get("kind", ""))
        peptide["last_action_title"] = str(action.get("title", ""))
        peptide["last_action_status"] = str(action.get("status", ""))


def _set_waiting_reason(peptide: dict[str, object], *, md_profile: str, step: str, reason: str) -> None:
    stage_entry = _stage_history_entry(peptide, md_profile)
    stage_entry["step"] = step
    stage_entry["status"] = "pending"
    stage_entry["waiting_reason"] = reason
    stage_entry["updated_at"] = _now_iso()
    peptide["current_stage"] = md_profile
    peptide["current_step"] = step
    peptide["status"] = "pending"
    peptide["waiting_reason"] = reason
    peptide["last_update_at"] = _now_iso()


def _apply_action_success(run_root: Path, slate: dict[str, object], peptide: dict[str, object], action: dict[str, object]) -> None:
    md_profile = str(peptide.get("current_stage", "")).strip() or "line_smoke"
    kind = str(action.get("kind", "")).strip()
    if kind == "prepare-md-stage":
        campaign_dir = str(action.get("output_path", "")).strip()
        peptide["current_campaign_dir"] = campaign_dir
        peptide["current_campaign"] = _path_name(campaign_dir)
        _set_stage_step(peptide, md_profile=md_profile, step="upload", status="pending", action=action)
        stage_entry = _stage_history_entry(peptide, md_profile)
        stage_entry["campaign_dir"] = campaign_dir
        stage_entry["campaign"] = _path_name(campaign_dir)
        stage_entry["started_at"] = stage_entry.get("started_at", "") or _now_iso()
        return
    if kind == "bura-upload-campaign":
        _set_stage_step(peptide, md_profile=md_profile, step="readiness", status="pending", action=action)
        return
    if kind == "bura-submit-readiness":
        missing = _readiness_missing_tokens(action)
        if missing:
            failure_reason = "Readiness check found missing submit dependencies: " + ", ".join(missing)
            _set_stage_step(peptide, md_profile=md_profile, step="readiness", status="blocked", action=action, failure_reason=failure_reason)
            peptide["failure_reason"] = failure_reason
            peptide["blocked_stage"] = md_profile
            return
        _set_stage_step(peptide, md_profile=md_profile, step="normalize", status="pending", action=action)
        return
    if kind == "bura-normalize-scripts":
        _set_stage_step(peptide, md_profile=md_profile, step="preflight", status="pending", action=action)
        return
    if kind == "bura-preflight":
        _set_stage_step(peptide, md_profile=md_profile, step="submit", status="pending", action=action)
        return
    if kind == "bura-submit-chain":
        remote_job_id = str(action.get("remote_job_id", "")).strip()
        if not remote_job_id:
            failure_reason = "The BURA submit step finished without returning a tracked Slurm job id."
            _set_stage_step(peptide, md_profile=md_profile, step="submit", status="blocked", action=action, failure_reason=failure_reason)
            peptide["failure_reason"] = failure_reason
            peptide["blocked_stage"] = md_profile
            return
        peptide["remote_job_id"] = remote_job_id
        peptide["poll_not_before"] = ""
        _set_stage_step(peptide, md_profile=md_profile, step="poll", status="active", action=action)
        return
    if kind == "bura-poll-squeue":
        remote_job_id = str(peptide.get("remote_job_id", "")).strip()
        stdout_log = Path(str(action.get("stdout_log", "")))
        jobs = parse_squeue_output(stdout_log.read_text(encoding="utf-8", errors="replace")) if stdout_log.exists() else []
        matching = [job for job in jobs if str(job.get("job_id", "")).strip() == remote_job_id]
        if matching:
            peptide["poll_not_before"] = (datetime.now() + timedelta(seconds=POLL_INTERVAL_SECONDS)).isoformat(timespec="seconds")
            _set_stage_step(peptide, md_profile=md_profile, step="poll", status="active", action=action)
            return
        peptide["poll_not_before"] = ""
        _set_stage_step(peptide, md_profile=md_profile, step="pull", status="pending", action=action)
        return
    if kind == "bura-pull-package":
        _set_stage_step(peptide, md_profile=md_profile, step="finalize", status="pending", action=action)
        return
    if kind == "finalize-md-stage":
        campaign_dir = Path(str(peptide.get("current_campaign_dir", "")))
        review_row = next(
            (row for row in _safe_read_csv(campaign_dir / "md_review.csv") if str(row.get("sequence", "")).strip() == str(peptide.get("sequence", "")).strip()),
            {},
        )
        job_root_status = str(review_row.get("job_root_status", "")).strip()
        expected_terminal_status = _expected_terminal_status(md_profile)
        if job_root_status != expected_terminal_status:
            failure_reason = f"{md_profile} finalized with {job_root_status or '<unknown>'} instead of {expected_terminal_status}."
            _set_stage_step(peptide, md_profile=md_profile, step="finalize", status="blocked", action=action, failure_reason=failure_reason)
            peptide["failure_reason"] = failure_reason
            peptide["blocked_stage"] = md_profile
            return
        stage_entry = _stage_history_entry(peptide, md_profile)
        stage_entry["completed_at"] = _now_iso()
        stage_entry["status"] = "completed"
        stage_entry["review_ready"] = bool(md_profile == "full")
        stage_entry["sync_status"] = "finalized_local"
        if md_profile == "full":
            peptide["review_ready"] = True
            peptide["status"] = "completed"
            peptide["current_step"] = ""
            peptide["waiting_reason"] = ""
            peptide["failure_reason"] = ""
            peptide["blocked_stage"] = ""
            peptide["last_action_status"] = str(action.get("status", ""))
            peptide["last_update_at"] = _now_iso()
            return
        next_profile = _next_stage(md_profile)
        peptide["current_stage"] = next_profile
        peptide["current_step"] = "prepare"
        peptide["current_campaign"] = ""
        peptide["current_campaign_dir"] = ""
        peptide["remote_job_id"] = ""
        peptide["status"] = "pending"
        peptide["waiting_reason"] = ""
        peptide["failure_reason"] = ""
        peptide["blocked_stage"] = ""
        peptide["last_action_status"] = str(action.get("status", ""))
        peptide["last_update_at"] = _now_iso()


def _apply_action_failure(peptide: dict[str, object], action: dict[str, object]) -> None:
    failure_reason = _failure_summary(action)
    md_profile = str(peptide.get("current_stage", "")).strip() or "line_smoke"
    _set_stage_step(peptide, md_profile=md_profile, step=str(peptide.get("current_step", "prepare")), status="blocked", action=action, failure_reason=failure_reason)
    peptide["failure_reason"] = failure_reason
    peptide["blocked_stage"] = md_profile


def _sync_with_child_action(peptide: dict[str, object], action: dict[str, object], *, run_root: Path, slate: dict[str, object]) -> None:
    status = str(action.get("status", "")).strip()
    if status in ACTIVE_ACTION_STATUSES:
        peptide["status"] = "active"
        peptide["waiting_reason"] = ""
        peptide["last_action_status"] = status
        peptide["last_update_at"] = _now_iso()
        return
    if status in FINAL_ACTION_STATUSES:
        if status == "succeeded":
            _apply_action_success(run_root, slate, peptide, action)
        else:
            _apply_action_failure(peptide, action)


def _stage_active_counts(run_root: Path, slate: dict[str, object]) -> dict[str, int]:
    utilization = _current_bura_utilization(run_root, [slate])
    counts = dict(utilization.get("tracked_external_counts", {}))
    for key in SLATE_STAGE_CAPS:
        counts.setdefault(key, 0)
    for peptide in slate.get("peptides", []):
        if not isinstance(peptide, dict):
            continue
        if str(peptide.get("status", "")) != "active":
            continue
        current_stage = str(peptide.get("current_stage", "")).strip()
        current_step = str(peptide.get("current_step", "")).strip()
        if current_stage in counts and current_step in {"submit", "poll"}:
            counts[current_stage] += 1
    return counts


def _schedule_peptide_step(run_root: Path, slate: dict[str, object], peptide: dict[str, object], *, log: Callable[[str], None]) -> dict[str, object] | None:
    run_dir = Path(str(slate.get("run_dir", "")))
    profile = load_cluster_profiles()
    bura_profile = get_cluster_profile(profile, "bura")
    if bura_profile is None:
        failure_reason = "BURA cluster profile is not configured."
        peptide["status"] = "blocked"
        peptide["failure_reason"] = failure_reason
        peptide["blocked_stage"] = str(peptide.get("current_stage", "line_smoke"))
        return None
    sequence = str(peptide.get("sequence", "")).strip()
    current_stage = str(peptide.get("current_stage", "")).strip() or "line_smoke"
    current_step = str(peptide.get("current_step", "")).strip() or "prepare"
    campaign_dir = Path(str(peptide.get("current_campaign_dir", "")).strip())
    if not campaign_dir.exists():
        existing_campaign = _campaign_for_stage(run_dir, sequence, current_stage)
        if existing_campaign:
            campaign_dir = Path(str(existing_campaign.get("campaign_dir", "")))
            peptide["current_campaign_dir"] = str(campaign_dir)
            peptide["current_campaign"] = str(existing_campaign.get("campaign", ""))
    if current_step == "upload":
        action = draft_bura_upload_campaign_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            requires_approval=False,
        )
    elif current_step == "readiness":
        action = queue_bura_readiness_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
        )
    elif current_step == "normalize":
        action = draft_bura_normalize_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            requires_approval=False,
        )
    elif current_step == "preflight":
        action = draft_bura_preflight_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            requires_approval=False,
        )
    elif current_step == "submit":
        counts = _stage_active_counts(run_root, slate)
        if counts.get(current_stage, 0) >= SLATE_STAGE_CAPS.get(current_stage, 1):
            cap = SLATE_STAGE_CAPS.get(current_stage, 1)
            _set_waiting_reason(
                peptide,
                md_profile=current_stage,
                step=current_step,
                reason=f"Waiting for BURA {current_stage} capacity ({counts.get(current_stage, 0)}/{cap} already active).",
            )
            return None
        action = draft_bura_submit_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            exclude_nodes="",
            requires_approval=False,
        )
    elif current_step == "poll":
        not_before = str(peptide.get("poll_not_before", "")).strip()
        if not_before and not_before > _now_iso():
            _set_waiting_reason(
                peptide,
                md_profile=current_stage,
                step=current_step,
                reason=f"Waiting until {not_before} before the next BURA queue poll.",
            )
            return None
        action = queue_bura_poll_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            remote_job_id=str(peptide.get("remote_job_id", "")),
        )
    elif current_step == "pull":
        action = draft_bura_pull_package_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            profile=bura_profile,
            related_run=str(run_dir),
            requires_approval=False,
        )
    elif current_step == "finalize":
        action = submit_finalize_md_stage_action(
            run_root=run_root,
            campaign_dir=campaign_dir,
            sequence=sequence,
            related_run=str(run_dir),
            launch_worker=True,
        )
    else:
        return None
    peptide["last_action_id"] = str(action.get("id", ""))
    peptide["last_action_kind"] = str(action.get("kind", ""))
    peptide["last_action_title"] = str(action.get("title", ""))
    peptide["last_action_status"] = str(action.get("status", ""))
    peptide["status"] = "active"
    peptide["last_update_at"] = _now_iso()
    stage_entry = _stage_history_entry(peptide, current_stage)
    if current_step == "prepare" and not str(stage_entry.get("campaign_dir", "")).strip():
        stage_entry["campaign"] = _default_campaign_name(run_dir, sequence, current_stage)
        stage_entry["campaign_dir"] = str(run_dir / "md_campaigns" / stage_entry["campaign"])
    stage_entry["step"] = current_step
    stage_entry["status"] = "active"
    stage_entry["last_action_id"] = str(action.get("id", ""))
    stage_entry["last_action_kind"] = str(action.get("kind", ""))
    stage_entry["last_action_title"] = str(action.get("title", ""))
    stage_entry["last_action_status"] = str(action.get("status", ""))
    stage_entry["updated_at"] = _now_iso()
    log(f"{sequence}: queued {current_step} for {current_stage} via action {action.get('id', '')}")
    return action


def _batch_csv_for_sequence(run_root: Path, run_dir: Path, sequence: str, *, preferred_path: str = "") -> Path:
    preferred = str(preferred_path).strip()
    if preferred:
        preferred_candidate = Path(preferred)
        if preferred_candidate.exists():
            return preferred_candidate
        raise FileNotFoundError(f"Saved source batch CSV no longer exists for {sequence}: {preferred_candidate}")
    candidate = find_md_source_batch_csv(run_root, run_dir, sequence)
    if candidate:
        return Path(candidate)
    batches_root = run_dir / "batches"
    raise FileNotFoundError(f"Could not find a source batch CSV for {sequence} in {batches_root}")


def _schedule_prepare_action(run_root: Path, slate: dict[str, object], peptide: dict[str, object], *, log: Callable[[str], None]) -> dict[str, object]:
    run_dir = Path(str(slate.get("run_dir", "")))
    sequence = str(peptide.get("sequence", "")).strip()
    current_stage = str(peptide.get("current_stage", "")).strip() or "line_smoke"
    action = submit_prepare_md_stage_action(
        run_root=run_root,
        run_dir=run_dir,
        batch_csv=_batch_csv_for_sequence(run_root, run_dir, sequence, preferred_path=str(peptide.get("source_batch_csv", ""))),
        sequence=sequence,
        campaign=_default_campaign_name(run_dir, sequence, current_stage),
        md_profile=current_stage,
        cluster="bura",
        launch_worker=True,
    )
    peptide["last_action_id"] = str(action.get("id", ""))
    peptide["last_action_kind"] = str(action.get("kind", ""))
    peptide["last_action_title"] = str(action.get("title", ""))
    peptide["last_action_status"] = str(action.get("status", ""))
    peptide["status"] = "active"
    peptide["last_update_at"] = _now_iso()
    stage_entry = _stage_history_entry(peptide, current_stage)
    campaign_name = _default_campaign_name(run_dir, sequence, current_stage)
    stage_entry["campaign"] = campaign_name
    stage_entry["campaign_dir"] = str(run_dir / "md_campaigns" / campaign_name)
    stage_entry["step"] = "prepare"
    stage_entry["status"] = "active"
    stage_entry["last_action_id"] = str(action.get("id", ""))
    stage_entry["last_action_kind"] = str(action.get("kind", ""))
    stage_entry["last_action_title"] = str(action.get("title", ""))
    stage_entry["last_action_status"] = str(action.get("status", ""))
    stage_entry["updated_at"] = _now_iso()
    log(f"{sequence}: queued prepare for {current_stage} via action {action.get('id', '')}")
    return action


def _schedule_next_action(run_root: Path, slate: dict[str, object], peptide: dict[str, object], *, log: Callable[[str], None]) -> dict[str, object] | None:
    current_step = str(peptide.get("current_step", "")).strip() or "prepare"
    if current_step == "prepare":
        return _schedule_prepare_action(run_root, slate, peptide, log=log)
    return _schedule_peptide_step(run_root, slate, peptide, log=log)


def tick_md_slate(run_root: Path, slate_id: str, *, log: Callable[[str], None] | None = None) -> dict[str, object]:
    logger = log or (lambda message: None)
    slate = load_dashboard_md_slate(run_root, slate_id)
    if bool(slate.get("paused", False)):
        slate["status"] = "paused"
        slate["updated_at"] = _now_iso()
        return save_dashboard_md_slate(run_root, slate)
    if str(slate.get("status", "")) in SLATE_FINAL_STATUSES:
        return slate

    peptides = list(slate.get("peptides", [])) if isinstance(slate.get("peptides", []), list) else []
    for peptide in peptides:
        if not isinstance(peptide, dict):
            continue
        last_action_id = str(peptide.get("last_action_id", "")).strip()
        if not last_action_id:
            continue
        try:
            action = load_dashboard_action(run_root, last_action_id)
        except Exception:
            continue
        if str(action.get("status", "")) in ACTIVE_ACTION_STATUSES:
            peptide["status"] = "active"
            peptide["waiting_reason"] = ""
            peptide["last_action_status"] = str(action.get("status", ""))
            continue
        if str(action.get("status", "")) in FINAL_ACTION_STATUSES:
            if str(peptide.get("last_action_status", "")) != str(action.get("status", "")):
                _sync_with_child_action(peptide, action, run_root=run_root, slate=slate)

    if str(slate.get("status", "")) not in SLATE_FINAL_STATUSES:
        slate["status"] = "running"

    for peptide in peptides:
        if not isinstance(peptide, dict):
            continue
        if str(peptide.get("status", "")) in PEPTIDE_FINAL_STATUSES or str(peptide.get("status", "")) == "active":
            continue
        try:
            _schedule_next_action(run_root, slate, peptide, log=logger)
        except ValueError as exc:
            if "Another dashboard action is already queued" in str(exc):
                _set_waiting_reason(
                    peptide,
                    md_profile=str(peptide.get("current_stage", "")).strip() or "line_smoke",
                    step=str(peptide.get("current_step", "")).strip() or "prepare",
                    reason="Waiting for another dashboard action on this peptide to finish first.",
                )
                logger(f"{peptide.get('sequence', '')}: waiting for conflicting dashboard action to finish")
                continue
            failure_reason = str(exc).strip() or "Slate scheduling failed."
            peptide["status"] = "blocked"
            peptide["failure_reason"] = failure_reason
            peptide["blocked_stage"] = str(peptide.get("current_stage", "line_smoke"))
            peptide["last_update_at"] = _now_iso()
        except Exception as exc:
            failure_reason = str(exc).strip() or "Slate scheduling failed."
            if isinstance(exc, FileNotFoundError):
                failure_reason = (
                    f"{failure_reason} Export or create a source batch CSV for this peptide, "
                    "then retry it from the slate board."
                )
            peptide["status"] = "blocked"
            peptide["failure_reason"] = failure_reason
            peptide["blocked_stage"] = str(peptide.get("current_stage", "line_smoke"))
            peptide["last_update_at"] = _now_iso()

    statuses = [str(item.get("status", "")) for item in peptides if isinstance(item, dict)]
    if statuses and all(status in {"completed", "skipped"} for status in statuses):
        slate["status"] = "completed"
    elif any(status == "blocked" for status in statuses) and all(status in PEPTIDE_FINAL_STATUSES for status in statuses):
        slate["status"] = "completed_with_failures"
    elif bool(slate.get("paused", False)):
        slate["status"] = "paused"
    else:
        slate["status"] = "running"
    slate["updated_at"] = _now_iso()
    return save_dashboard_md_slate(run_root, slate)


def run_md_slate_supervisor(run_root: Path, slate_id: str, *, log: Callable[[str], None] | None = None) -> dict[str, object]:
    logger = log or (lambda message: print(message, flush=True))
    logger(f"Starting MD slate supervisor for {slate_id}")
    while True:
        slate = tick_md_slate(run_root, slate_id, log=logger)
        status = str(slate.get("status", ""))
        if status in {"completed", "completed_with_failures", "cancelled", "paused"}:
            logger(f"Slate {slate_id} finished with status={status}")
            return slate
        time.sleep(0.5)
