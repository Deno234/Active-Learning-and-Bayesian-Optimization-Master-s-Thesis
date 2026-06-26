from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from active_learning_thesis.md_orchestrator import STAGE_META_FILENAME

ACTIVE_REMOTE_STATUSES = {"submitted", "running"}
MD_STAGE_ORDER = ("line_smoke", "production_smoke", "full")
QUEUE_RUNNING_STATES = {"R", "RUNNING"}
QUEUE_WAITING_STATES = {"PD", "Q", "PENDING", "CONFIGURING", "CF"}
QUEUE_FAILED_STATES = {"F", "FAILED", "CA", "CANCELLED", "NF", "TO", "TIMEOUT"}


def _canonical_path(value: str | Path | None) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).resolve())
    except Exception:
        return str(value)


def _path_name(value: str | Path | None) -> str:
    if not value:
        return ""
    return Path(str(value)).name


def _safe_read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
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
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _short_job_id(value: str) -> str:
    text = str(value).strip()
    return text.split(".", 1)[0] if "." in text else text


def _job_ids_match(left: str, right: str) -> bool:
    left_text = str(left).strip()
    right_text = str(right).strip()
    if not left_text or not right_text:
        return False
    return left_text == right_text or _short_job_id(left_text) == _short_job_id(right_text)


def _find_queue_job(jobs: list[dict[str, object]], remote_job_id: str) -> dict[str, object] | None:
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if _job_ids_match(str(job.get("job_id", "")), remote_job_id):
            return job
    return None


def _queue_state_label(job: dict[str, object] | None) -> str:
    if not job:
        return "not visible"
    state = str(job.get("state", "")).upper().strip()
    reason = str(job.get("reason", "")).strip()
    if state in QUEUE_RUNNING_STATES:
        return "running"
    if state in QUEUE_WAITING_STATES:
        return f"waiting: {reason}" if reason else "waiting"
    if state in {"H", "HELD"} or "held" in reason.lower() or "dependencynever" in reason.lower():
        return f"held: {reason}" if reason else "held"
    if state in QUEUE_FAILED_STATES:
        return "failed"
    return state.lower() or "visible"


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


def _latest_snapshots_by_cluster(snapshots: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        cluster = str(snapshot.get("cluster", "")).strip().lower()
        if not cluster:
            continue
        existing = latest.get(cluster)
        if existing is None or str(snapshot.get("collected_at", "")) >= str(existing.get("collected_at", "")):
            latest[cluster] = snapshot
    return latest


def _jobs_for_snapshot(snapshot: dict[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(snapshot, dict):
        return []
    jobs = snapshot.get("jobs", [])
    return [job for job in jobs if isinstance(job, dict)] if isinstance(jobs, list) else []


def _stage_for_campaign(campaign_dir: str, fallback: str = "") -> str:
    campaign_path = Path(str(campaign_dir))
    meta = _safe_read_json(campaign_path / STAGE_META_FILENAME)
    stage = str(meta.get("md_profile", "")).strip()
    if stage:
        return stage
    if campaign_path.name in MD_STAGE_ORDER:
        return campaign_path.name
    if fallback in MD_STAGE_ORDER:
        return fallback
    for row in _safe_read_csv(campaign_path / "manifest.csv"):
        stage = str(row.get("md_profile", "")).strip()
        if stage:
            return stage
    return fallback


def _sequence_for_campaign(campaign_dir: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    meta = _safe_read_json(Path(str(campaign_dir)) / STAGE_META_FILENAME)
    sequence = str(meta.get("sequence", "")).strip()
    if sequence:
        return sequence
    for row in _safe_read_csv(Path(str(campaign_dir)) / "manifest.csv"):
        sequence = str(row.get("sequence", "")).strip()
        if sequence:
            return sequence
    return ""


def _run_display_map(run_summaries: list[dict[str, object]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for run in run_summaries:
        run_dir = _canonical_path(str(run.get("run_dir", "")))
        if run_dir:
            mapping[run_dir] = str(run.get("run_display_name", run.get("run_name", ""))) or _path_name(run_dir)
    return mapping


def _display_run(run_dir: str, mapping: dict[str, str]) -> str:
    key = _canonical_path(run_dir)
    return mapping.get(key, _path_name(run_dir) or "-")


def _tracked_sync_entries(
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
        remote_job_id = str(record.get("remote_job_id", "")).strip()
        if cluster not in {"supek", "bura"} or status not in ACTIVE_REMOTE_STATUSES or not remote_job_id:
            continue
        run_dir = str(record.get("related_run", "")).strip()
        campaign_dir = str(record.get("related_campaign", "")).strip()
        sequence = str(record.get("related_sequence", "")).strip()
        stage = ""
        if cluster == "bura":
            stage = _stage_for_campaign(campaign_dir)
            sequence = _sequence_for_campaign(campaign_dir, sequence)
        entries.append(
            {
                "source": "sync_record",
                "cluster": cluster,
                "run": _display_run(run_dir, run_map),
                "run_dir": _canonical_path(run_dir),
                "campaign": _path_name(campaign_dir),
                "campaign_dir": _canonical_path(campaign_dir),
                "sequence": sequence,
                "stage": stage,
                "dashboard_state": status,
                "remote_job_id": remote_job_id,
                "updated_at": str(record.get("updated_at", "")),
                "target_key": _canonical_path(str(record.get("target_key", ""))),
            }
        )
    return entries


def _slate_peptide_entries(md_slates: list[dict[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for slate in md_slates:
        if not isinstance(slate, dict):
            continue
        slate_id = str(slate.get("slate_id", "")).strip()
        run_dir = _canonical_path(str(slate.get("run_dir", "")))
        run_name = str(slate.get("run_name", "")).strip() or _path_name(run_dir)
        for peptide in slate.get("peptides", []):
            if not isinstance(peptide, dict):
                continue
            remote_job_id = str(peptide.get("remote_job_id", "")).strip()
            if not remote_job_id:
                continue
            entries.append(
                {
                    "source": "md_slate",
                    "cluster": "bura",
                    "slate_id": slate_id,
                    "run": run_name,
                    "run_dir": run_dir,
                    "campaign": str(peptide.get("current_campaign", "")) or _path_name(str(peptide.get("current_campaign_dir", ""))),
                    "campaign_dir": _canonical_path(str(peptide.get("current_campaign_dir", ""))),
                    "sequence": str(peptide.get("sequence", "")).strip(),
                    "stage": str(peptide.get("current_stage", "")).strip(),
                    "step": str(peptide.get("current_step", "")).strip(),
                    "dashboard_state": str(peptide.get("status", "")).strip(),
                    "waiting_reason": str(peptide.get("waiting_reason", "")).strip(),
                    "last_action_id": str(peptide.get("last_action_id", "")).strip(),
                    "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                    "remote_job_id": remote_job_id,
                    "updated_at": str(peptide.get("last_update_at", "")).strip(),
                }
            )
    return entries


def _sync_entry_for_campaign(entries: list[dict[str, object]], campaign_dir: str) -> dict[str, object] | None:
    campaign_key = _canonical_path(campaign_dir)
    if not campaign_key:
        return None
    for entry in entries:
        if str(entry.get("source", "")) != "sync_record":
            continue
        if _canonical_path(str(entry.get("campaign_dir", ""))) == campaign_key:
            return entry
    return None


def _row(
    *,
    issue_type: str,
    severity: str,
    issue: str,
    cluster: str,
    target_kind: str,
    run: str = "",
    run_dir: str = "",
    campaign: str = "",
    campaign_dir: str = "",
    sequence: str = "",
    stage: str = "",
    remote_job_id: str = "",
    queue_state: str = "",
    dashboard_state: str = "",
    summary: str = "",
    recommended_recovery: str = "",
    next_move: str = "",
    open_view: str = "",
    related_action_id: str = "",
    slate_id: str = "",
    step: str = "",
    candidate_jobs: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "severity": severity,
        "issue": issue,
        "issue_type": issue_type,
        "cluster": cluster,
        "target_kind": target_kind,
        "run": run or _path_name(run_dir) or "-",
        "run_dir": run_dir,
        "campaign": campaign or "-",
        "campaign_dir": campaign_dir,
        "sequence": sequence or "-",
        "stage": stage or "-",
        "remote_job_id": remote_job_id or "-",
        "queue_state": queue_state or "-",
        "dashboard_state": dashboard_state or "-",
        "summary": summary,
        "recommended_recovery": recommended_recovery,
        "next_move": next_move,
        "open_view": open_view or ("MD Validation" if cluster == "bura" else "Model Workflow"),
        "related_action_id": related_action_id,
        "slate_id": slate_id,
        "step": step or "-",
        "candidate_jobs": candidate_jobs or [],
    }
    identity_parts = [
        issue_type,
        cluster,
        remote_job_id,
        _canonical_path(run_dir),
        _canonical_path(campaign_dir),
        sequence,
        stage,
        slate_id,
    ]
    body["reconciliation_id"] = hashlib.sha1("::".join(identity_parts).encode("utf-8")).hexdigest()[:16]
    return body


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row.get("issue_type", "")),
            str(row.get("cluster", "")),
            str(row.get("remote_job_id", "")),
            _canonical_path(str(row.get("run_dir", ""))),
            str(row.get("sequence", "")),
            str(row.get("stage", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def build_remote_reconciliation_rows(
    *,
    run_summaries: list[dict[str, object]],
    md_slates: list[dict[str, object]],
    sync_records: list[dict[str, object]],
    snapshots: list[dict[str, object]],
    actions: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    del actions
    latest_snapshots = _latest_snapshots_by_cluster(snapshots)
    jobs_by_cluster = {
        cluster: _jobs_for_snapshot(snapshot)
        for cluster, snapshot in latest_snapshots.items()
        if str(snapshot.get("collected_at", "")).strip()
    }
    sync_entries = _tracked_sync_entries(run_summaries=run_summaries, sync_records=sync_records)
    slate_entries = _slate_peptide_entries(md_slates)
    rows: list[dict[str, object]] = []

    tracked_ids_by_cluster: dict[str, set[str]] = {}
    for entry in sync_entries + slate_entries:
        cluster = str(entry.get("cluster", "")).strip()
        job_id = str(entry.get("remote_job_id", "")).strip()
        if not cluster or not job_id:
            continue
        tracked_ids_by_cluster.setdefault(cluster, set()).add(job_id)
        tracked_ids_by_cluster[cluster].add(_short_job_id(job_id))

    for entry in sync_entries:
        cluster = str(entry.get("cluster", "")).strip()
        if cluster not in jobs_by_cluster:
            continue
        job_id = str(entry.get("remote_job_id", "")).strip()
        job = _find_queue_job(jobs_by_cluster.get(cluster, []), job_id)
        queue_state = _queue_state_label(job)
        queue_group = _queue_state_group(job)
        open_view = "MD Validation" if cluster == "bura" else "Model Workflow"
        target_kind = "MD campaign" if cluster == "bura" else "Model run"
        if job is None:
            rows.append(
                _row(
                    issue_type="tracked_missing_from_queue",
                    severity="warning",
                    issue="Tracked job missing from queue",
                    cluster=cluster,
                    target_kind=target_kind,
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=str(entry.get("campaign_dir", "")),
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=job_id,
                    queue_state=queue_state,
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary=f"{cluster.upper()} job {job_id} is tracked locally but is not visible in the latest queue snapshot.",
                    recommended_recovery="Fetch logs, then pull artifacts if outputs exist",
                    next_move="Fetch the latest remote logs first. If the run produced outputs, pull them back and finalize; if logs show failure, recover or retry from the last safe step.",
                    open_view=open_view,
                )
            )
        elif queue_group == "failed":
            rows.append(
                _row(
                    issue_type="tracked_failed_in_queue",
                    severity="error",
                    issue="Tracked job is failed in queue",
                    cluster=cluster,
                    target_kind=target_kind,
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=str(entry.get("campaign_dir", "")),
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=job_id,
                    queue_state=queue_state,
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary=f"{cluster.upper()} job {job_id} is still visible, but the scheduler reports a failed or cancelled state.",
                    recommended_recovery="Fetch logs and mark stale before retrying",
                    next_move="Inspect logs before retrying. Keep the dashboard state as evidence, then recover or relaunch only after the failure cause is clear.",
                    open_view=open_view,
                )
            )
        elif str(entry.get("dashboard_state", "")) == "submitted" and queue_group == "running":
            rows.append(
                _row(
                    issue_type="tracked_running_dashboard_waiting",
                    severity="info",
                    issue="Queue says running while dashboard says submitted",
                    cluster=cluster,
                    target_kind=target_kind,
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=str(entry.get("campaign_dir", "")),
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=job_id,
                    queue_state=queue_state,
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary=f"{cluster.upper()} job {job_id} is running, but the local sync record has not been refreshed past submitted.",
                    recommended_recovery="Poll again",
                    next_move="Poll the queue again from the dashboard so the sync record advances to running.",
                    open_view=open_view,
                )
            )

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for entry in sync_entries:
        if str(entry.get("cluster", "")) != "bura":
            continue
        sequence = str(entry.get("sequence", "")).strip()
        stage = str(entry.get("stage", "")).strip()
        run_dir = _canonical_path(str(entry.get("run_dir", "")))
        if not sequence or not stage or not run_dir:
            continue
        grouped.setdefault(("bura", run_dir, sequence, stage), []).append(entry)
    for (_cluster, run_dir, sequence, stage), entries in grouped.items():
        unique_job_ids = sorted({str(entry.get("remote_job_id", "")).strip() for entry in entries if str(entry.get("remote_job_id", "")).strip()})
        if len(unique_job_ids) <= 1:
            continue
        rows.append(
            _row(
                issue_type="duplicate_tracked_job",
                severity="warning",
                issue="Duplicate active jobs for peptide stage",
                cluster="bura",
                target_kind="MD campaign",
                run=str(entries[0].get("run", "")),
                run_dir=run_dir,
                campaign=", ".join(str(entry.get("campaign", "")) for entry in entries if str(entry.get("campaign", "")).strip()),
                sequence=sequence,
                stage=stage,
                remote_job_id=", ".join(unique_job_ids),
                queue_state="multiple tracked",
                dashboard_state="submitted/running",
                summary=f"{sequence} has {len(unique_job_ids)} active BURA job ids tracked for {stage}.",
                recommended_recovery="Cancel duplicate or rebind",
                next_move="Inspect the campaign directories and latest logs. Keep the intended job, cancel duplicates if they are accidental, and rebind the slate peptide if it points at the wrong job.",
                open_view="MD Validation",
                candidate_jobs=[
                    {
                        "job_id": str(entry.get("remote_job_id", "")).strip(),
                        "campaign": str(entry.get("campaign", "")).strip(),
                        "campaign_dir": str(entry.get("campaign_dir", "")).strip(),
                        "dashboard_state": str(entry.get("dashboard_state", "")).strip(),
                    }
                    for entry in entries
                    if str(entry.get("remote_job_id", "")).strip()
                ],
            )
        )

    for entry in slate_entries:
        job_id = str(entry.get("remote_job_id", "")).strip()
        campaign_dir = str(entry.get("campaign_dir", "")).strip()
        sync_entry = _sync_entry_for_campaign(sync_entries, campaign_dir)
        sync_job_id = str((sync_entry or {}).get("remote_job_id", "")).strip()
        job = _find_queue_job(jobs_by_cluster.get("bura", []), job_id)
        queue_group = _queue_state_group(job)
        if sync_entry and sync_job_id and not _job_ids_match(sync_job_id, job_id):
            rows.append(
                _row(
                    issue_type="slate_job_mismatch",
                    severity="warning",
                    issue="Slate job id differs from campaign sync",
                    cluster="bura",
                    target_kind="MD slate peptide",
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=campaign_dir,
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=f"{job_id} (slate) / {sync_job_id} (sync)",
                    queue_state=_queue_state_label(job),
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary="The slate peptide and campaign sync record disagree about which BURA job is authoritative.",
                    recommended_recovery="Rebind job",
                    next_move="Open the slate monitor and rebind the peptide to the correct latest child action/job before polling or pulling.",
                    open_view="MD Validation",
                    related_action_id=str(entry.get("last_action_id", "")),
                    slate_id=str(entry.get("slate_id", "")),
                    step=str(entry.get("step", "")),
                )
            )
        elif campaign_dir and not sync_entry:
            rows.append(
                _row(
                    issue_type="slate_job_missing_sync",
                    severity="warning",
                    issue="Slate job has no campaign sync record",
                    cluster="bura",
                    target_kind="MD slate peptide",
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=campaign_dir,
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=job_id,
                    queue_state=_queue_state_label(job),
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary="The slate peptide has a BURA job id, but the campaign sync record is missing.",
                    recommended_recovery="Rebind job",
                    next_move="Rebind the peptide to the latest tracked child action, or poll/fetch logs once to recreate the campaign sync record.",
                    open_view="MD Validation",
                    related_action_id=str(entry.get("last_action_id", "")),
                    slate_id=str(entry.get("slate_id", "")),
                    step=str(entry.get("step", "")),
                )
            )
        if job is not None and str(entry.get("dashboard_state", "")) == "pending":
            rows.append(
                _row(
                    issue_type="tracked_running_dashboard_waiting",
                    severity="warning" if queue_group == "running" else "info",
                    issue="Visible job while slate is waiting",
                    cluster="bura",
                    target_kind="MD slate peptide",
                    run=str(entry.get("run", "")),
                    run_dir=str(entry.get("run_dir", "")),
                    campaign=str(entry.get("campaign", "")),
                    campaign_dir=campaign_dir,
                    sequence=str(entry.get("sequence", "")),
                    stage=str(entry.get("stage", "")),
                    remote_job_id=job_id,
                    queue_state=_queue_state_label(job),
                    dashboard_state=str(entry.get("dashboard_state", "")),
                    summary="The BURA job is visible in the queue, but the slate peptide is not marked active.",
                    recommended_recovery="Poll again",
                    next_move="Poll the tracked job again before pulling artifacts or advancing the slate.",
                    open_view="MD Validation",
                    related_action_id=str(entry.get("last_action_id", "")),
                    slate_id=str(entry.get("slate_id", "")),
                    step=str(entry.get("step", "")),
                )
            )

    for cluster, jobs in jobs_by_cluster.items():
        tracked_ids = tracked_ids_by_cluster.get(cluster, set())
        for job in jobs:
            job_id = str(job.get("job_id", "")).strip()
            if not job_id:
                continue
            if job_id in tracked_ids or _short_job_id(job_id) in tracked_ids:
                continue
            rows.append(
                _row(
                    issue_type="external_queue_job",
                    severity="info",
                    issue=f"External {cluster.upper()} job in user queue",
                    cluster=cluster,
                    target_kind="External queue job",
                    remote_job_id=job_id,
                    queue_state=_queue_state_label(job),
                    dashboard_state="not dashboard-tracked",
                    summary=f"{cluster.upper()} job {job_id} is visible in the latest queue snapshot but is not tied to a dashboard run, campaign, or slate.",
                    recommended_recovery="No action unless capacity looks tight",
                    next_move="Use this as capacity context only. It does not hard-block slate scheduling unless you decide it is yours and should be reconciled manually.",
                    open_view="Operations",
                )
            )

    severity_rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(
        _dedupe_rows(rows),
        key=lambda row: (
            severity_rank.get(str(row.get("severity", "")), 9),
            str(row.get("cluster", "")),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
            str(row.get("stage", "")),
        ),
    )


def build_remote_reconciliation_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "total": len(rows),
        "errors": 0,
        "warnings": 0,
        "infos": 0,
        "tracked_missing": 0,
        "duplicates": 0,
        "mismatches": 0,
        "external_jobs": 0,
        "external_bura_jobs": 0,
    }
    for row in rows:
        severity = str(row.get("severity", "")).strip()
        issue_type = str(row.get("issue_type", "")).strip()
        cluster = str(row.get("cluster", "")).strip()
        if severity == "error":
            summary["errors"] += 1
        elif severity == "warning":
            summary["warnings"] += 1
        elif severity == "info":
            summary["infos"] += 1
        if issue_type == "tracked_missing_from_queue":
            summary["tracked_missing"] += 1
        if issue_type == "duplicate_tracked_job":
            summary["duplicates"] += 1
        if issue_type in {"slate_job_mismatch", "slate_job_missing_sync"}:
            summary["mismatches"] += 1
        if issue_type == "external_queue_job":
            summary["external_jobs"] += 1
            if cluster == "bura":
                summary["external_bura_jobs"] += 1
    return summary


def filter_remote_reconciliation_rows(
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
        run_dir = _canonical_path(str(row.get("run_dir", "")))
        if normalized_run_dirs and run_dir and run_dir not in normalized_run_dirs:
            continue
        if sequence != "All" and str(row.get("sequence", "")) not in {"", "-", sequence}:
            continue
        if md_profile != "All" and str(row.get("stage", "")) not in {"", "-", md_profile}:
            continue
        if status != "All":
            candidates = {
                str(row.get("severity", "")),
                str(row.get("issue_type", "")),
                str(row.get("queue_state", "")),
                str(row.get("dashboard_state", "")),
            }
            if status not in candidates:
                continue
        filtered.append(row)
    return filtered
