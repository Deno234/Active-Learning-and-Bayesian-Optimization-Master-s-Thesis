from __future__ import annotations

from datetime import datetime
from pathlib import Path

from active_learning_thesis.dashboard_actions import ACTIVE_ACTION_STATUSES, FINAL_ACTION_STATUSES

RECOVERY_ACTIVE_STALE_MINUTES = {
    "prepare": 20,
    "upload": 20,
    "readiness": 20,
    "normalize": 20,
    "preflight": 20,
    "submit": 20,
    "poll": 45,
    "pull": 20,
    "finalize": 20,
}
RECOVERY_PENDING_STALE_MINUTES = 30
RECOVERY_CAPACITY_WAIT_MINUTES = 60
RECOVERY_POLL_OVERDUE_MINUTES = 5
RELEVANT_MD_ACTION_KINDS = {
    "prepare-md-stage",
    "bura-upload-campaign",
    "bura-submit-readiness",
    "bura-normalize-scripts",
    "bura-preflight",
    "bura-submit-chain",
    "bura-poll-squeue",
    "bura-pull-package",
    "finalize-md-stage",
}
STEP_ACTION_KIND_MAP = {
    "prepare": {"prepare-md-stage"},
    "upload": {"bura-upload-campaign"},
    "readiness": {"bura-submit-readiness"},
    "normalize": {"bura-normalize-scripts"},
    "preflight": {"bura-preflight"},
    "submit": {"bura-submit-chain"},
    "poll": {"bura-poll-squeue", "bura-submit-chain"},
    "pull": {"bura-pull-package", "bura-poll-squeue"},
    "finalize": {"finalize-md-stage", "bura-pull-package"},
}


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
    return Path(value).name


def _parse_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _action_timestamp(action: dict[str, object]) -> str:
    for key in ("updated_at", "finished_at", "started_at", "created_at"):
        value = str(action.get(key, "")).strip()
        if value:
            return value
    return ""


def _minutes_since(timestamp: str, *, now: datetime | None = None) -> int | None:
    parsed = _parse_iso(timestamp)
    if parsed is None:
        return None
    reference = now or datetime.now()
    delta = reference - parsed
    return max(int(delta.total_seconds() // 60), 0)


def _matches_peptide_action(action: dict[str, object], slate: dict[str, object], peptide: dict[str, object]) -> bool:
    if str(action.get("kind", "")).strip() not in RELEVANT_MD_ACTION_KINDS:
        return False
    if str(action.get("related_sequence", "")).strip() != str(peptide.get("sequence", "")).strip():
        return False
    run_key = _canonical_path(slate.get("run_dir"))
    campaign_key = _canonical_path(peptide.get("current_campaign_dir"))
    action_run = _canonical_path(action.get("related_run"))
    action_campaign = _canonical_path(action.get("related_campaign"))
    output_path = _canonical_path(action.get("output_path"))
    if campaign_key and campaign_key in {action_campaign, output_path}:
        return True
    return bool(run_key and action_run == run_key)


def find_md_slate_rebind_candidate(
    actions: list[dict[str, object]],
    slate: dict[str, object],
    peptide: dict[str, object],
) -> dict[str, object] | None:
    step = str(peptide.get("current_step", "")).strip() or "prepare"
    preferred_kinds = STEP_ACTION_KIND_MAP.get(step, set())
    campaign_key = _canonical_path(peptide.get("current_campaign_dir"))
    candidates = [
        action
        for action in actions
        if _matches_peptide_action(action, slate, peptide)
    ]
    if not candidates:
        return None

    def sort_key(action: dict[str, object]) -> tuple[int, int, int, int]:
        status = str(action.get("status", "")).strip()
        action_campaign = _canonical_path(action.get("related_campaign")) or _canonical_path(action.get("output_path"))
        timestamp = _parse_iso(_action_timestamp(action)) or datetime.min
        return (
            0 if str(action.get("kind", "")).strip() in preferred_kinds else 1,
            0 if campaign_key and action_campaign == campaign_key else 1,
            0 if status in ACTIVE_ACTION_STATUSES else 1,
            -int(timestamp.timestamp()) if timestamp != datetime.min else 0,
        )

    return sorted(candidates, key=sort_key)[0]


def build_md_slate_exception_rows(
    md_slates: list[dict[str, object]],
    actions: list[dict[str, object]],
    *,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    action_by_id = {str(action.get("id", "")): action for action in actions}
    rows: list[dict[str, object]] = []
    reference = now or datetime.now()

    for slate in md_slates:
        slate_id = str(slate.get("slate_id", "")).strip()
        run_dir = str(slate.get("run_dir", "")).strip()
        run_name = str(slate.get("run_name", "")).strip() or _path_name(run_dir)
        for peptide in list(slate.get("peptides", [])):
            if not isinstance(peptide, dict):
                continue
            sequence = str(peptide.get("sequence", "")).strip()
            status = str(peptide.get("status", "")).strip()
            current_stage = str(peptide.get("current_stage", "")).strip() or "line_smoke"
            current_step = str(peptide.get("current_step", "")).strip() or "prepare"
            last_action_id = str(peptide.get("last_action_id", "")).strip()
            last_action = action_by_id.get(last_action_id, {})
            waiting_reason = str(peptide.get("waiting_reason", "")).strip()
            failure_reason = str(peptide.get("failure_reason", "")).strip()
            remote_job_id = str(peptide.get("remote_job_id", "")).strip()
            last_update_at = str(peptide.get("last_update_at", "")).strip()
            age_minutes = _minutes_since(last_update_at, now=reference)
            rebind_candidate = find_md_slate_rebind_candidate(actions, slate, peptide)
            row: dict[str, object] | None = None

            if status == "blocked":
                row = {
                    "slate_id": slate_id,
                    "run": run_name,
                    "run_dir": run_dir,
                    "sequence": sequence,
                    "stage": current_stage,
                    "step": current_step,
                    "state": status,
                    "severity": "error",
                    "exception_type": "blocked",
                    "summary": failure_reason or f"{sequence} is blocked at {current_stage} / {current_step}.",
                    "next_move": "Retry the peptide from the last checkpoint, or skip it if you want the rest of the slate to finish without it.",
                    "last_update_at": last_update_at,
                    "age_minutes": age_minutes if age_minutes is not None else "",
                    "waiting_reason": waiting_reason,
                    "failure_reason": failure_reason,
                    "remote_job_id": remote_job_id,
                    "last_action": str(peptide.get("last_action_title", "")).strip(),
                    "last_action_id": last_action_id,
                    "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                    "recover_available": True,
                    "rebind_available": bool(rebind_candidate),
                    "skip_available": True,
                    "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                }
            elif status == "active":
                if last_action_id and not last_action and (age_minutes or 0) >= 5:
                    row = {
                        "slate_id": slate_id,
                        "run": run_name,
                        "run_dir": run_dir,
                        "sequence": sequence,
                        "stage": current_stage,
                        "step": current_step,
                        "state": status,
                        "severity": "warning",
                        "exception_type": "missing_action",
                        "summary": f"{sequence} is marked active, but the slate can no longer find action {last_action_id}.",
                        "next_move": "Rebind the peptide to the latest tracked action if the remote job still exists, or recover it from the last checkpoint if the action is really gone.",
                        "last_update_at": last_update_at,
                        "age_minutes": age_minutes if age_minutes is not None else "",
                        "waiting_reason": waiting_reason,
                        "failure_reason": failure_reason,
                        "remote_job_id": remote_job_id,
                        "last_action": str(peptide.get("last_action_title", "")).strip(),
                        "last_action_id": last_action_id,
                        "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                        "recover_available": not bool(remote_job_id),
                        "rebind_available": bool(rebind_candidate),
                        "skip_available": False,
                        "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                    }
                else:
                    threshold = RECOVERY_ACTIVE_STALE_MINUTES.get(current_step, 30)
                    if age_minutes is not None and age_minutes >= threshold:
                        row = {
                            "slate_id": slate_id,
                            "run": run_name,
                            "run_dir": run_dir,
                            "sequence": sequence,
                            "stage": current_stage,
                            "step": current_step,
                            "state": status,
                            "severity": "error" if age_minutes >= threshold * 3 else "warning",
                            "exception_type": "stale_active",
                            "summary": f"{sequence} has been stuck in {current_stage} / {current_step} for about {age_minutes} minutes without a slate update.",
                            "next_move": "Inspect the latest logs, then rebind the peptide to the latest tracked action or recover the stage from the last checkpoint if the job really disappeared.",
                            "last_update_at": last_update_at,
                            "age_minutes": age_minutes,
                            "waiting_reason": waiting_reason,
                            "failure_reason": failure_reason,
                            "remote_job_id": remote_job_id,
                            "last_action": str(peptide.get("last_action_title", "")).strip(),
                            "last_action_id": last_action_id,
                            "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                            "recover_available": not bool(remote_job_id),
                            "rebind_available": bool(rebind_candidate),
                            "skip_available": False,
                            "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                        }
            elif status == "pending":
                poll_not_before = str(peptide.get("poll_not_before", "")).strip()
                poll_due_minutes = _minutes_since(poll_not_before, now=reference)
                if current_step == "poll" and poll_not_before and poll_due_minutes is not None and poll_due_minutes >= RECOVERY_POLL_OVERDUE_MINUTES:
                    row = {
                        "slate_id": slate_id,
                        "run": run_name,
                        "run_dir": run_dir,
                        "sequence": sequence,
                        "stage": current_stage,
                        "step": current_step,
                        "state": status,
                        "severity": "warning",
                        "exception_type": "poll_overdue",
                        "summary": f"{sequence} should have polled BURA again by now, but the next poll window has already passed.",
                        "next_move": "Rebind the peptide to the latest tracked action if the remote job is still alive, or recover the stage from the last checkpoint if polling fell out of sync.",
                        "last_update_at": last_update_at,
                        "age_minutes": age_minutes if age_minutes is not None else "",
                        "waiting_reason": waiting_reason,
                        "failure_reason": failure_reason,
                        "remote_job_id": remote_job_id,
                        "last_action": str(peptide.get("last_action_title", "")).strip(),
                        "last_action_id": last_action_id,
                        "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                        "recover_available": not bool(remote_job_id),
                        "rebind_available": bool(rebind_candidate),
                        "skip_available": False,
                        "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                    }
                elif waiting_reason and "capacity" in waiting_reason.lower() and age_minutes is not None and age_minutes >= RECOVERY_CAPACITY_WAIT_MINUTES:
                    row = {
                        "slate_id": slate_id,
                        "run": run_name,
                        "run_dir": run_dir,
                        "sequence": sequence,
                        "stage": current_stage,
                        "step": current_step,
                        "state": status,
                        "severity": "warning",
                        "exception_type": "capacity_wait",
                        "summary": f"{sequence} has been waiting on the local {current_stage} BURA cap for about {age_minutes} minutes.",
                        "next_move": "This is usually a throughput issue rather than a failure. Let the active jobs finish, or pause the slate if you want to stop scheduling new work for a while.",
                        "last_update_at": last_update_at,
                        "age_minutes": age_minutes,
                        "waiting_reason": waiting_reason,
                        "failure_reason": failure_reason,
                        "remote_job_id": remote_job_id,
                        "last_action": str(peptide.get("last_action_title", "")).strip(),
                        "last_action_id": last_action_id,
                        "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                        "recover_available": False,
                        "rebind_available": False,
                        "skip_available": False,
                        "rebind_action_id": "",
                    }
                elif last_action_id and not last_action and (age_minutes or 0) >= 5:
                    row = {
                        "slate_id": slate_id,
                        "run": run_name,
                        "run_dir": run_dir,
                        "sequence": sequence,
                        "stage": current_stage,
                        "step": current_step,
                        "state": status,
                        "severity": "warning",
                        "exception_type": "missing_action",
                        "summary": f"{sequence} is pending, but the slate lost the child action record it expected to resume from.",
                        "next_move": "Rebind the peptide to the latest tracked action if one exists, or recover it from the last checkpoint to restart the local scheduling path cleanly.",
                        "last_update_at": last_update_at,
                        "age_minutes": age_minutes if age_minutes is not None else "",
                        "waiting_reason": waiting_reason,
                        "failure_reason": failure_reason,
                        "remote_job_id": remote_job_id,
                        "last_action": str(peptide.get("last_action_title", "")).strip(),
                        "last_action_id": last_action_id,
                        "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                        "recover_available": not bool(remote_job_id),
                        "rebind_available": bool(rebind_candidate),
                        "skip_available": False,
                        "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                    }
                elif age_minutes is not None and age_minutes >= RECOVERY_PENDING_STALE_MINUTES and not waiting_reason:
                    row = {
                        "slate_id": slate_id,
                        "run": run_name,
                        "run_dir": run_dir,
                        "sequence": sequence,
                        "stage": current_stage,
                        "step": current_step,
                        "state": status,
                        "severity": "warning",
                        "exception_type": "pending_stale",
                        "summary": f"{sequence} has been pending at {current_stage} / {current_step} for about {age_minutes} minutes without a clear waiting reason.",
                        "next_move": "Recover the peptide from the last checkpoint so the scheduler can try that stage again, or inspect the recent local actions if you want to understand the drift first.",
                        "last_update_at": last_update_at,
                        "age_minutes": age_minutes,
                        "waiting_reason": waiting_reason,
                        "failure_reason": failure_reason,
                        "remote_job_id": remote_job_id,
                        "last_action": str(peptide.get("last_action_title", "")).strip(),
                        "last_action_id": last_action_id,
                        "last_action_status": str(peptide.get("last_action_status", "")).strip(),
                        "recover_available": not bool(remote_job_id),
                        "rebind_available": bool(rebind_candidate),
                        "skip_available": False,
                        "rebind_action_id": str((rebind_candidate or {}).get("id", "")),
                    }

            if row is not None:
                rows.append(row)

    severity_order = {"error": 0, "warning": 1, "info": 2}
    return sorted(
        rows,
        key=lambda row: (
            severity_order.get(str(row.get("severity", "")), 3),
            -int(row.get("age_minutes", 0) or 0),
            str(row.get("run", "")),
            str(row.get("sequence", "")),
        ),
    )


def build_md_slate_exception_summary(exception_rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "total": len(exception_rows),
        "errors": sum(1 for row in exception_rows if str(row.get("severity", "")) == "error"),
        "warnings": sum(1 for row in exception_rows if str(row.get("severity", "")) == "warning"),
        "blocked": sum(1 for row in exception_rows if str(row.get("exception_type", "")) == "blocked"),
        "stale": sum(
            1
            for row in exception_rows
            if str(row.get("exception_type", "")) in {"stale_active", "pending_stale", "poll_overdue", "missing_action"}
        ),
        "slates": len({str(row.get("slate_id", "")) for row in exception_rows if str(row.get("slate_id", "")).strip()}),
    }
