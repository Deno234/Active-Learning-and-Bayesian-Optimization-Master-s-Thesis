from __future__ import annotations

from datetime import datetime
from pathlib import Path


ACTIVE_ACTION_STATUSES = {"queued", "running"}
APPROVAL_STATUSES = {"draft", "awaiting_approval"}
TERMINAL_ATTENTION_STATUSES = {"failed", "cancelled", "manual_override"}

QUEUED_STALE_HOURS = 2.0
RUNNING_STALE_HOURS = 12.0
APPROVAL_STALE_HOURS = 24.0


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now()


def _age_hours(action: dict[str, object], *, now: datetime) -> float | None:
    for key in ("updated_at", "started_at", "created_at"):
        parsed = _parse_datetime(action.get(key))
        if parsed is not None:
            return max(0.0, (now - parsed).total_seconds() / 3600.0)
    return None


def _read_tail(path_value: object, *, max_lines: int = 18) -> str:
    text = str(path_value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > 65536:
                handle.seek(-65536, 2)
            raw = handle.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:]).strip()


def _first_matching_line(text: str, patterns: tuple[str, ...]) -> str:
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    for line in reversed(text.splitlines()):
        lowered_line = line.lower()
        if any(pattern in lowered_line for pattern in lowered_patterns):
            return line.strip()
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _path_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return Path(text).name or text
    except OSError:
        return text


def _action_target(action: dict[str, object]) -> str:
    if str(action.get("related_sequence", "")).strip():
        return str(action.get("related_sequence", "")).strip()
    if str(action.get("related_campaign", "")).strip():
        return _path_name(action.get("related_campaign"))
    if str(action.get("related_run", "")).strip():
        return _path_name(action.get("related_run"))
    if str(action.get("output_path", "")).strip():
        return _path_name(action.get("output_path"))
    return "-"


def _rule(
    *,
    issue_type: str,
    reason: str,
    safe_next_move: str,
    patterns: tuple[str, ...],
    confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "issue_type": issue_type,
        "reason": reason,
        "safe_next_move": safe_next_move,
        "patterns": patterns,
        "confidence": confidence,
    }


DIAGNOSIS_RULES: tuple[dict[str, object], ...] = (
    _rule(
        issue_type="ssh_auth",
        reason="SSH authentication or host verification failed.",
        safe_next_move="Open Operations -> Cluster health, fix the SSH key/agent/host key issue, then rerun a read-only health or log action before retrying the write action.",
        patterns=(
            "permission denied (publickey)",
            "agent has no identities",
            "could not open a connection to your authentication agent",
            "host key verification failed",
            "ssh authentication",
        ),
        confidence=0.95,
    ),
    _rule(
        issue_type="scheduler_or_queue",
        reason="The remote scheduler or queue state looks like the blocker.",
        safe_next_move="Run the relevant queue poll or fetch logs first. If the job is held or dependency-blocked, inspect scheduler output before cancelling or resubmitting.",
        patterns=(
            "qsub",
            "sbatch",
            "squeue",
            "qstat",
            "scheduler",
            "job held",
            "dependency",
            "too many failed attempts",
            "invalid account",
            "partition",
        ),
        confidence=0.85,
    ),
    _rule(
        issue_type="missing_path",
        reason="A required local or remote file/path appears to be missing.",
        safe_next_move="Use the path shown in the logs to inspect the run/campaign/artifact, then rerun the safest read-only readiness or artifact check before retrying.",
        patterns=(
            "no such file or directory",
            "filenotfounderror",
            "cannot find the path",
            "path does not exist",
            "realpath",
            "missing file",
            "missing path",
        ),
        confidence=0.9,
    ),
    _rule(
        issue_type="md_artifact_or_review",
        reason="The failure mentions MD review, ingest, package, or analysis artifacts.",
        safe_next_move="Open MD Validation -> Artifact verification or Review & ingest for the affected peptide/campaign, fix the missing evidence, then regenerate ingest or retry the MD step.",
        patterns=(
            "manifest.csv",
            "md_review.csv",
            "cgmd_ingest",
            "cgmd_label",
            "package directory",
            "pdb",
            "sasa",
            "ap_sasa",
            "analysis_complete",
        ),
        confidence=0.85,
    ),
    _rule(
        issue_type="config_mismatch",
        reason="An existing run/study configuration does not match the requested action.",
        safe_next_move="Inspect the existing config and the requested plan. Resume only if intentional, or create a new run/study name to avoid mixing evidence.",
        patterns=(
            "does not match this study plan",
            "config mismatch",
            "allow-config-mismatch",
            "existing run config",
        ),
        confidence=0.95,
    ),
    _rule(
        issue_type="environment_import",
        reason="The Python environment or package imports are not available for this action.",
        safe_next_move="Check the selected environment, installed dependencies, and remote activation command, then rerun a read-only import/preflight check.",
        patterns=(
            "modulenotfounderror",
            "importerror",
            "no module named",
            "conda",
            "tensorflow",
            "cuda",
            "failed to import",
        ),
        confidence=0.85,
    ),
    _rule(
        issue_type="permissions",
        reason="The action could not write, read, or execute because of permissions.",
        safe_next_move="Check directory ownership/permissions and whether another process is locking the file, then retry once the path is writable.",
        patterns=("permissionerror", "permission denied", "access is denied", "operation not permitted"),
        confidence=0.85,
    ),
    _rule(
        issue_type="resource_limit",
        reason="The action appears to have hit a memory, disk, or resource limit.",
        safe_next_move="Reduce the workload or request larger resources, then rerun a small smoke/canary before retrying the full action.",
        patterns=("memoryerror", "out of memory", "cuda out of memory", "killed", "disk quota", "no space left"),
        confidence=0.8,
    ),
    _rule(
        issue_type="invalid_input",
        reason="The command rejected an argument or input value.",
        safe_next_move="Correct the GUI field or command argument, then queue the action again. Prefer the GUI readiness message if one is shown.",
        patterns=("valueerror", "invalid", "must be", "cannot convert", "argument", "parse error"),
        confidence=0.75,
    ),
    _rule(
        issue_type="data_parse",
        reason="A CSV, JSON, or report file could not be parsed cleanly.",
        safe_next_move="Open the referenced file and check for truncation, malformed rows, or a stale partial write before regenerating the artifact.",
        patterns=("jsondecodeerror", "csv", "could not parse", "malformed", "empty data", "bad line"),
        confidence=0.75,
    ),
)


def _diagnose_failed_text(text: str) -> dict[str, object]:
    lowered = text.lower()
    for rule in DIAGNOSIS_RULES:
        patterns = tuple(str(item) for item in rule["patterns"])
        if any(pattern.lower() in lowered for pattern in patterns):
            return {
                "issue_type": str(rule["issue_type"]),
                "reason": str(rule["reason"]),
                "safe_next_move": str(rule["safe_next_move"]),
                "confidence": float(rule["confidence"]),
                "evidence": _first_matching_line(text, patterns),
            }
    return {
        "issue_type": "unknown_failure",
        "reason": "The action failed, but the captured logs do not match a known failure pattern yet.",
        "safe_next_move": "Open stderr/stdout, read the last concrete error line, then rerun the safest read-only check for this action area before retrying.",
        "confidence": 0.45,
        "evidence": _first_matching_line(text, ()),
    }


def diagnose_dashboard_action(action: dict[str, object], *, now: datetime | None = None) -> dict[str, object]:
    current_time = now or _now()
    status = str(action.get("status", "")).strip()
    stderr_excerpt = _read_tail(action.get("stderr_log"))
    stdout_excerpt = _read_tail(action.get("stdout_log"))
    combined_text = "\n".join(part for part in (stderr_excerpt, stdout_excerpt) if part).strip()
    age = _age_hours(action, now=current_time)
    age_value = round(age, 2) if age is not None else ""

    attention = "Healthy"
    issue_type = "none"
    reason = "No obvious action issue detected."
    safe_next_move = "No action needed."
    evidence = ""
    confidence = 0.6
    priority = 9

    if status == "failed":
        diagnosis = _diagnose_failed_text(combined_text)
        attention = "Needs fix"
        issue_type = str(diagnosis["issue_type"])
        reason = str(diagnosis["reason"])
        safe_next_move = str(diagnosis["safe_next_move"])
        evidence = str(diagnosis["evidence"])
        confidence = float(diagnosis["confidence"])
        priority = 0
    elif status == "queued" and age is not None and age >= QUEUED_STALE_HOURS:
        attention = "Possibly stuck"
        issue_type = "queued_too_long"
        reason = f"Action has been queued for about {age:.1f} hours without starting."
        safe_next_move = "Open Operations -> Approval queue/action history, confirm no conflicting action is running, then pause/cancel/requeue if it is stale."
        confidence = 0.7
        priority = 1
    elif status == "running" and age is not None and age >= RUNNING_STALE_HOURS:
        attention = "Possibly stuck"
        issue_type = "running_too_long"
        reason = f"Action has been running for about {age:.1f} hours."
        safe_next_move = "Inspect stdout/stderr first. If no progress is visible, use pause/cancel/manual override only after checking whether a remote job is still valid."
        confidence = 0.65
        priority = 1
    elif status in APPROVAL_STATUSES:
        attention = "Waiting approval"
        issue_type = "approval_waiting"
        reason = "Action is waiting for operator approval before it can run."
        safe_next_move = "Open Operations -> Approval queue, read the action contract, then approve, pause, or cancel it."
        confidence = 0.85
        priority = 2 if age is None or age < APPROVAL_STALE_HOURS else 1
    elif status == "paused":
        attention = "Waiting operator"
        issue_type = "paused"
        reason = "Action is paused by the operator."
        safe_next_move = "Resume only when the original blocker is cleared; otherwise cancel or leave it paused with an operator note."
        confidence = 0.85
        priority = 3
    elif status in TERMINAL_ATTENTION_STATUSES:
        attention = "Closed"
        issue_type = status
        reason = f"Action ended with terminal status `{status}`."
        safe_next_move = "No automatic recovery is queued. Use the logs and output path if this was not intentional."
        confidence = 0.65
        priority = 4
    elif status == "succeeded" and stderr_excerpt:
        attention = "Needs review"
        issue_type = "stderr_on_success"
        reason = "Action succeeded but stderr contains output that may be worth checking."
        safe_next_move = "Read the stderr excerpt. If it is only a benign warning, keep the action; otherwise rerun a safer check before trusting the output."
        evidence = _first_matching_line(stderr_excerpt, ())
        confidence = 0.55
        priority = 5

    return {
        "priority": priority,
        "attention": attention,
        "issue_type": issue_type,
        "confidence": round(confidence, 2),
        "reason": reason,
        "safe_next_move": safe_next_move,
        "evidence": evidence,
        "action_id": str(action.get("id", "")),
        "title": str(action.get("title", "")),
        "kind": str(action.get("kind", "")),
        "status": status,
        "scope": str(action.get("scope", "local")),
        "cluster": str(action.get("cluster", "")),
        "target": _action_target(action),
        "exit_code": "" if action.get("exit_code") is None else str(action.get("exit_code")),
        "created_at": str(action.get("created_at", "")),
        "started_at": str(action.get("started_at", "")),
        "finished_at": str(action.get("finished_at", "")),
        "age_hours": age_value,
        "remote_job_id": str(action.get("remote_job_id", "")),
        "sync_status": str(action.get("sync_status", "")),
        "stdout_log": str(action.get("stdout_log", "")),
        "stderr_log": str(action.get("stderr_log", "")),
        "output_path": str(action.get("output_path", "")),
        "display_command": str(action.get("display_command", "")),
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
    }


def build_action_debug_rows(
    actions: list[dict[str, object]],
    *,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    current_time = now or _now()
    rows = [diagnose_dashboard_action(action, now=current_time) for action in actions]
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("priority", 9)),
            str(row.get("created_at", "")),
            str(row.get("action_id", "")),
        ),
    )


def build_action_debug_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "needs_fix": sum(1 for row in rows if row.get("attention") == "Needs fix"),
        "possibly_stuck": sum(1 for row in rows if row.get("attention") == "Possibly stuck"),
        "waiting_operator": sum(1 for row in rows if str(row.get("attention", "")).startswith("Waiting")),
        "needs_review": sum(1 for row in rows if row.get("attention") == "Needs review"),
        "needs_attention": sum(1 for row in rows if int(row.get("priority", 9)) <= 5),
        "healthy": sum(1 for row in rows if row.get("attention") == "Healthy"),
    }


def action_debug_display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    display_keys = [
        "attention",
        "issue_type",
        "title",
        "kind",
        "status",
        "target",
        "exit_code",
        "confidence",
        "reason",
        "safe_next_move",
    ]
    return [{key: row.get(key, "") for key in display_keys} for row in rows]


def build_action_debug_packet_markdown(row: dict[str, object]) -> str:
    lines = [
        "# Dashboard Action Debug Packet",
        "",
        f"- Action: {row.get('title', '')}",
        f"- Action id: {row.get('action_id', '')}",
        f"- Kind/status: {row.get('kind', '')} / {row.get('status', '')}",
        f"- Scope/cluster: {row.get('scope', '')} / {row.get('cluster', '') or '-'}",
        f"- Target: {row.get('target', '')}",
        f"- Exit code: {row.get('exit_code', '') or '-'}",
        f"- Diagnosis: {row.get('attention', '')} ({row.get('issue_type', '')}, confidence {row.get('confidence', '')})",
        f"- Reason: {row.get('reason', '')}",
        f"- Safe next move: {row.get('safe_next_move', '')}",
    ]
    evidence = str(row.get("evidence", "")).strip()
    if evidence:
        lines.append(f"- Evidence: {evidence}")
    lines.extend(
        [
            f"- stdout log: {row.get('stdout_log', '')}",
            f"- stderr log: {row.get('stderr_log', '')}",
            f"- output path: {row.get('output_path', '')}",
            "",
            "## Command",
            "```bash",
            str(row.get("display_command", "")),
            "```",
        ]
    )
    stderr_excerpt = str(row.get("stderr_excerpt", "")).strip()
    stdout_excerpt = str(row.get("stdout_excerpt", "")).strip()
    if stderr_excerpt:
        lines.extend(["", "## stderr excerpt", "```text", stderr_excerpt, "```"])
    if stdout_excerpt:
        lines.extend(["", "## stdout excerpt", "```text", stdout_excerpt, "```"])
    return "\n".join(lines).strip() + "\n"
