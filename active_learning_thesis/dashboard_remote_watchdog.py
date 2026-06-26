from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


ACTIVE_REMOTE_STATES = {"submitted", "running"}
READY_REMOTE_STATES = {"outputs_staged", "outputs_returned"}
FINAL_REMOTE_STATES = {"finalized_local"}
SETUP_REMOTE_STATES = {"staged_remote"}
QUEUE_SNAPSHOT_STALE_HOURS = 4
CLUSTER_HEALTH_STALE_HOURS = 12
QUEUE_RUNNING_STATES = {"R", "RUNNING"}
QUEUE_WAITING_STATES = {"PD", "Q", "PENDING", "CONFIGURING", "CF"}
QUEUE_FAILED_STATES = {"F", "FAILED", "CA", "CANCELLED", "NF", "TO", "TIMEOUT"}
VERDICT_ORDER = {
    "needs_recovery": 0,
    "needs_check": 1,
    "ready": 2,
    "watch": 3,
    "staged": 4,
    "complete": 5,
    "healthy": 6,
}


def _canonical_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def _path_name(value: object) -> str:
    text = str(value or "").strip()
    return Path(text).name if text else ""


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _age_hours(value: object, *, now: datetime) -> int | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return max(int((now - parsed).total_seconds() // 3600), 0)


def _short_job_id(value: object) -> str:
    text = str(value or "").strip()
    return text.split(".", 1)[0] if "." in text else text


def _job_ids_match(left: object, right: object) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    return left_text == right_text or _short_job_id(left_text) == _short_job_id(right_text)


def _find_queue_job(jobs: list[dict[str, object]], remote_job_id: str) -> dict[str, object] | None:
    for job in jobs:
        if _job_ids_match(job.get("job_id", ""), remote_job_id):
            return job
    return None


def _queue_state_group(job: dict[str, object] | None) -> str:
    if not job:
        return "missing"
    state = str(job.get("state", "")).upper().strip()
    reason = str(job.get("reason", "")).lower()
    if state in QUEUE_RUNNING_STATES:
        return "running"
    if state in {"H", "HELD"} or "held" in reason or "dependencynever" in reason:
        return "held"
    if state in QUEUE_WAITING_STATES:
        return "waiting"
    if state in QUEUE_FAILED_STATES:
        return "failed"
    return "visible"


def _queue_state_label(job: dict[str, object] | None) -> str:
    if not job:
        return "not visible"
    state = str(job.get("state", "")).upper().strip()
    reason = str(job.get("reason", "")).strip()
    group = _queue_state_group(job)
    if group == "waiting" and reason:
        return f"waiting: {reason}"
    if group == "held" and reason:
        return f"held: {reason}"
    return group if group != "visible" else (state.lower() or "visible")


def _latest_snapshot_by_cluster(snapshots: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for snapshot in snapshots:
        cluster = str(snapshot.get("cluster", "")).strip().lower()
        if not cluster:
            continue
        existing = latest.get(cluster)
        if existing is None or str(snapshot.get("collected_at", "")) >= str(existing.get("collected_at", "")):
            latest[cluster] = snapshot
    return latest


def _health_by_cluster(cluster_health: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in cluster_health:
        cluster = str(row.get("cluster", "")).strip().lower()
        if not cluster:
            continue
        existing = latest.get(cluster)
        if existing is None or str(row.get("checked_at", "")) >= str(existing.get("checked_at", "")):
            latest[cluster] = row
    return latest


def _snapshot_jobs(snapshot: dict[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(snapshot, dict):
        return []
    jobs = snapshot.get("jobs", [])
    return [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []


def _run_display_map(run_summaries: list[dict[str, object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for run in run_summaries:
        run_dir = _canonical_path(run.get("run_dir", ""))
        if run_dir:
            mapping[run_dir] = str(run.get("run_display_name", run.get("run_name", ""))) or _path_name(run_dir)
    return mapping


def _sync_entries(
    *,
    run_summaries: list[dict[str, object]],
    sync_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    run_map = _run_display_map(run_summaries)
    entries: list[dict[str, object]] = []
    for record in sync_records:
        if not isinstance(record, dict):
            continue
        cluster = str(record.get("cluster", "")).strip().lower()
        status = str(record.get("status", "")).strip()
        if cluster not in {"supek", "bura"} or status not in (ACTIVE_REMOTE_STATES | READY_REMOTE_STATES | FINAL_REMOTE_STATES | SETUP_REMOTE_STATES):
            continue
        run_dir = _canonical_path(record.get("related_run", ""))
        campaign_dir = _canonical_path(record.get("related_campaign", ""))
        target_key = _canonical_path(record.get("target_key", ""))
        sequence = str(record.get("related_sequence", "")).strip()
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        entries.append(
            {
                "source": "sync_record",
                "cluster": cluster,
                "target_kind": "Model run" if cluster == "supek" else "MD campaign",
                "run": run_map.get(run_dir, _path_name(run_dir) or "-"),
                "run_dir": run_dir,
                "campaign": _path_name(campaign_dir),
                "campaign_dir": campaign_dir,
                "sequence": sequence,
                "stage": str(metadata.get("md_profile", "")).strip(),
                "dashboard_state": status,
                "remote_job_id": str(record.get("remote_job_id", "")).strip(),
                "remote_path": str(record.get("remote_path", "")).strip(),
                "target_key": target_key,
                "updated_at": str(record.get("updated_at", "")).strip(),
                "open_view": "Model Workflow" if cluster == "supek" else "MD Validation",
            }
        )
    return entries


def _slate_entries(md_slates: list[dict[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for slate in md_slates:
        if not isinstance(slate, dict):
            continue
        slate_id = str(slate.get("slate_id", "")).strip()
        run_dir = _canonical_path(slate.get("run_dir", ""))
        run = str(slate.get("run_name", "")).strip() or _path_name(run_dir) or "-"
        for peptide in slate.get("peptides", []):
            if not isinstance(peptide, dict):
                continue
            status = str(peptide.get("status", "")).strip()
            remote_job_id = str(peptide.get("remote_job_id", "")).strip()
            review_ready = bool(peptide.get("review_ready", False))
            if not remote_job_id and status not in {"active", "blocked"} and not review_ready:
                continue
            campaign_dir = _canonical_path(peptide.get("current_campaign_dir", ""))
            entries.append(
                {
                    "source": "md_slate",
                    "cluster": "bura",
                    "target_kind": "MD slate peptide",
                    "run": run,
                    "run_dir": run_dir,
                    "campaign": str(peptide.get("current_campaign", "")).strip() or _path_name(campaign_dir),
                    "campaign_dir": campaign_dir,
                    "sequence": str(peptide.get("sequence", "")).strip(),
                    "stage": str(peptide.get("current_stage", "")).strip(),
                    "dashboard_state": status or "pending",
                    "remote_job_id": remote_job_id,
                    "remote_path": "",
                    "target_key": campaign_dir,
                    "updated_at": str(peptide.get("last_update_at", "")).strip(),
                    "open_view": "MD Validation",
                    "slate_id": slate_id,
                    "last_action_id": str(peptide.get("last_action_id", "")).strip(),
                    "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                    "review_ready": review_ready,
                    "failure_reason": str(peptide.get("failure_reason", "")).strip(),
                    "waiting_reason": str(peptide.get("waiting_reason", "")).strip(),
                }
            )
    return entries


def _matches_entry(row: dict[str, object], entry: dict[str, object]) -> bool:
    if str(row.get("cluster", "")).strip().lower() != str(entry.get("cluster", "")).strip().lower():
        return False
    job_id = str(entry.get("remote_job_id", "")).strip()
    row_job_id = str(row.get("remote_job_id", "")).strip()
    if job_id and (job_id in row_job_id or _short_job_id(job_id) in row_job_id):
        return True
    for field in ["run_dir", "campaign_dir"]:
        if _canonical_path(row.get(field, "")) and _canonical_path(row.get(field, "")) == _canonical_path(entry.get(field, "")):
            return True
    return (
        str(row.get("sequence", "")).strip()
        and str(row.get("sequence", "")).strip() == str(entry.get("sequence", "")).strip()
        and str(row.get("stage", "")).strip() == str(entry.get("stage", "")).strip()
        and _canonical_path(row.get("run_dir", "")) == _canonical_path(entry.get("run_dir", ""))
    )


def _matching_reconciliation(
    entry: dict[str, object],
    remote_reconciliation: list[dict[str, object]],
) -> dict[str, object] | None:
    ranked = [
        row
        for row in remote_reconciliation
        if isinstance(row, dict)
        and str(row.get("severity", "")).strip() in {"warning", "error"}
        and _matches_entry(row, entry)
    ]
    if not ranked:
        return None
    severity_rank = {"error": 0, "warning": 1}
    return sorted(ranked, key=lambda row: severity_rank.get(str(row.get("severity", "")), 9))[0]


def _matching_artifact_issue(
    entry: dict[str, object],
    artifact_verification: list[dict[str, object]],
) -> dict[str, object] | None:
    for row in artifact_verification:
        if not isinstance(row, dict):
            continue
        if str(row.get("verification_state", "")) != "Attention needed":
            continue
        if _canonical_path(row.get("campaign_dir", "")) and _canonical_path(row.get("campaign_dir", "")) == _canonical_path(entry.get("campaign_dir", "")):
            return row
        if _canonical_path(row.get("run_dir", "")) == _canonical_path(entry.get("run_dir", "")) and str(row.get("sequence", "")).strip() in {"", "-", str(entry.get("sequence", "")).strip()}:
            return row
    return None


def _read_only_action(cluster: str, action: str) -> tuple[str, str]:
    if cluster == "supek":
        if action == "poll":
            return "supek-poll-qstat", "Poll SUPEK queue"
        if action == "logs":
            return "supek-fetch-logs", "Fetch latest SUPEK logs"
        return "supek-submit-preflight", "Refresh SUPEK preflight"
    if action == "logs":
        return "bura-inspect-logs", "Fetch latest BURA logs"
    if action == "health":
        return "bura-submit-readiness", "Refresh BURA readiness"
    return "bura-poll-squeue", "Poll BURA queue"


def _heartbeat_id(entry: dict[str, object]) -> str:
    parts = [
        str(entry.get("source", "")),
        str(entry.get("cluster", "")),
        _canonical_path(entry.get("target_key", "")),
        str(entry.get("remote_job_id", "")),
        str(entry.get("sequence", "")),
        str(entry.get("stage", "")),
    ]
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _verdict_for_entry(
    entry: dict[str, object],
    *,
    queue_job: dict[str, object] | None,
    snapshot_age_hours: int | None,
    health_status: str,
    health_age_hours: int | None,
    reconciliation: dict[str, object] | None,
    artifact_issue: dict[str, object] | None,
) -> tuple[str, str, str, str, str]:
    cluster = str(entry.get("cluster", "")).strip().lower()
    status = str(entry.get("dashboard_state", "")).strip()
    remote_job_id = str(entry.get("remote_job_id", "")).strip()
    queue_group = _queue_state_group(queue_job)

    if artifact_issue is not None:
        return (
            "needs_recovery",
            "artifact issue",
            str(artifact_issue.get("next_move", "")) or "Inspect the artifact verification row before advancing this remote handoff.",
            *_read_only_action(cluster, "logs"),
        )
    if reconciliation is not None:
        action = "logs" if str(reconciliation.get("issue_type", "")) in {"tracked_missing_from_queue", "tracked_failed_in_queue"} else "poll"
        return (
            "needs_recovery",
            str(reconciliation.get("issue", "")) or "remote reconciliation issue",
            str(reconciliation.get("next_move", "")) or "Open Operations -> Remote jobs and reconcile the remote state.",
            *_read_only_action(cluster, action),
        )
    if status == "blocked":
        return (
            "needs_recovery",
            str(entry.get("failure_reason", "")) or "slate peptide is blocked",
            str(entry.get("failure_reason", "")) or "Open MD Validation -> Recovery center and decide whether to retry, rebind, or skip this peptide.",
            *_read_only_action(cluster, "logs"),
        )
    if bool(entry.get("review_ready", False)) or status in READY_REMOTE_STATES:
        return (
            "ready",
            "outputs are ready for local follow-up",
            "Open the relevant validation page and review, copy back, or finalize the returned outputs.",
            "",
            "No read-only remote follow-up needed",
        )
    if status in FINAL_REMOTE_STATES:
        return (
            "complete",
            "remote handoff finalized locally",
            "No remote watchdog action is needed for this item.",
            "",
            "No action",
        )
    if status in SETUP_REMOTE_STATES:
        return (
            "staged",
            "staged remotely but not submitted",
            "Run the relevant preflight/readiness step before submitting mutating work.",
            *_read_only_action(cluster, "health"),
        )
    if status in ACTIVE_REMOTE_STATES or remote_job_id:
        if not remote_job_id:
            return (
                "needs_check",
                "missing remote job id",
                "Fetch logs or inspect the last submit action so the dashboard can bind the scheduler job id.",
                *_read_only_action(cluster, "logs"),
            )
        if snapshot_age_hours is None:
            return (
                "needs_check",
                "no queue snapshot",
                f"Poll the {cluster.upper()} queue so the dashboard has a current scheduler snapshot.",
                *_read_only_action(cluster, "poll"),
            )
        if snapshot_age_hours > QUEUE_SNAPSHOT_STALE_HOURS:
            return (
                "needs_check",
                f"queue snapshot is {snapshot_age_hours}h old",
                f"Refresh the {cluster.upper()} queue snapshot before trusting this job state.",
                *_read_only_action(cluster, "poll"),
            )
        if health_status not in {"", "ok", "unknown"}:
            return (
                "needs_check",
                f"{cluster.upper()} health is {health_status}",
                f"Refresh {cluster.upper()} health/readiness before launching or recovering additional remote work.",
                *_read_only_action(cluster, "health"),
            )
        if health_age_hours is None or health_age_hours > CLUSTER_HEALTH_STALE_HOURS:
            return (
                "needs_check",
                "cluster health is stale",
                f"Refresh {cluster.upper()} health/readiness so the dashboard can trust remote follow-ups.",
                *_read_only_action(cluster, "health"),
            )
        if queue_group in {"failed", "held", "missing"}:
            return (
                "needs_recovery",
                f"queue state is {queue_group}",
                "Fetch logs first, then recover, retry, or mark stale based on the evidence.",
                *_read_only_action(cluster, "logs"),
            )
        return (
            "watch",
            f"queue state is {queue_group}",
            "Keep monitoring; poll again before making any mutating recovery decision.",
            *_read_only_action(cluster, "poll"),
        )
    return (
        "healthy",
        "no active remote drift detected",
        "No remote watchdog action is needed right now.",
        "",
        "No action",
    )


def _watchdog_row(
    entry: dict[str, object],
    *,
    snapshots_by_cluster: dict[str, dict[str, object]],
    health_by_cluster: dict[str, dict[str, object]],
    remote_reconciliation: list[dict[str, object]],
    artifact_verification: list[dict[str, object]],
    now: datetime,
) -> dict[str, object]:
    cluster = str(entry.get("cluster", "")).strip().lower()
    snapshot = snapshots_by_cluster.get(cluster)
    health = health_by_cluster.get(cluster, {})
    jobs = _snapshot_jobs(snapshot)
    queue_job = _find_queue_job(jobs, str(entry.get("remote_job_id", "")))
    snapshot_age = _age_hours((snapshot or {}).get("collected_at", ""), now=now)
    health_age = _age_hours(health.get("checked_at", ""), now=now)
    health_status = str(health.get("overall_status", "")).strip() or "unknown"
    reconciliation = _matching_reconciliation(entry, remote_reconciliation)
    artifact_issue = _matching_artifact_issue(entry, artifact_verification)
    verdict, reason, next_move, action_kind, action_label = _verdict_for_entry(
        entry,
        queue_job=queue_job,
        snapshot_age_hours=snapshot_age,
        health_status=health_status,
        health_age_hours=health_age,
        reconciliation=reconciliation,
        artifact_issue=artifact_issue,
    )
    severity = "error" if verdict == "needs_recovery" else ("warning" if verdict == "needs_check" else "info")
    return {
        "heartbeat_id": _heartbeat_id(entry),
        "source": entry.get("source", ""),
        "cluster": cluster,
        "target_kind": entry.get("target_kind", ""),
        "run": entry.get("run", ""),
        "run_dir": entry.get("run_dir", ""),
        "campaign": entry.get("campaign", ""),
        "campaign_dir": entry.get("campaign_dir", ""),
        "sequence": entry.get("sequence", ""),
        "stage": entry.get("stage", ""),
        "slate_id": entry.get("slate_id", ""),
        "dashboard_state": entry.get("dashboard_state", ""),
        "remote_job_id": entry.get("remote_job_id", ""),
        "queue_state": _queue_state_label(queue_job),
        "health_status": health_status,
        "snapshot_age_hours": snapshot_age,
        "health_age_hours": health_age,
        "last_local_update": entry.get("updated_at", ""),
        "verdict": verdict,
        "severity": severity,
        "reason": reason,
        "safe_next_move": next_move,
        "recommended_action_kind": action_kind,
        "recommended_action": action_label,
        "open_view": entry.get("open_view", "Operations"),
    }


def build_remote_watchdog_rows(
    *,
    run_summaries: list[dict[str, object]],
    md_slates: list[dict[str, object]],
    sync_records: list[dict[str, object]],
    snapshots: list[dict[str, object]],
    cluster_health: list[dict[str, object]],
    remote_reconciliation: list[dict[str, object]],
    artifact_verification: list[dict[str, object]],
    now: datetime | None = None,
) -> list[dict[str, object]]:
    current_time = now or datetime.now()
    snapshots_by_cluster = _latest_snapshot_by_cluster(snapshots)
    health_rows_by_cluster = _health_by_cluster(cluster_health)
    entries = _sync_entries(run_summaries=run_summaries, sync_records=sync_records)
    entries.extend(_slate_entries(md_slates))
    rows = [
        _watchdog_row(
            entry,
            snapshots_by_cluster=snapshots_by_cluster,
            health_by_cluster=health_rows_by_cluster,
            remote_reconciliation=remote_reconciliation,
            artifact_verification=artifact_verification,
            now=current_time,
        )
        for entry in entries
    ]
    return sorted(
        rows,
        key=lambda row: (
            VERDICT_ORDER.get(str(row.get("verdict", "")), 99),
            str(row.get("cluster", "")),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
            str(row.get("stage", "")),
        ),
    )


def build_remote_watchdog_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "needs_recovery": 0,
        "needs_check": 0,
        "ready": 0,
        "watch": 0,
        "staged": 0,
        "complete": 0,
        "healthy": 0,
        "read_only_followups": 0,
    }
    for row in rows:
        verdict = str(row.get("verdict", "")).strip()
        if verdict in summary:
            summary[verdict] += 1
        if str(row.get("recommended_action_kind", "")).strip():
            summary["read_only_followups"] += 1
    return summary


def filter_remote_watchdog_rows(
    rows: list[dict[str, object]],
    *,
    run_dirs: set[str] | None = None,
    sequence: str = "All",
    md_profile: str = "All",
    status: str = "All",
) -> list[dict[str, object]]:
    normalized_run_dirs = {_canonical_path(item) for item in (run_dirs or set()) if str(item).strip()}
    filtered: list[dict[str, object]] = []
    for row in rows:
        run_dir = _canonical_path(row.get("run_dir", ""))
        if normalized_run_dirs and run_dir and run_dir not in normalized_run_dirs:
            continue
        if sequence != "All" and str(row.get("sequence", "")) not in {"", "-", sequence}:
            continue
        if md_profile != "All" and str(row.get("stage", "")) not in {"", "-", md_profile}:
            continue
        if status != "All":
            candidates = {
                str(row.get("verdict", "")),
                str(row.get("dashboard_state", "")),
                str(row.get("queue_state", "")),
                str(row.get("health_status", "")),
            }
            if status not in candidates:
                continue
        filtered.append(row)
    return filtered
