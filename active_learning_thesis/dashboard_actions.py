from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
import tempfile
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Sequence

from active_learning_thesis.md_review_evidence import (
    LABEL_CONFIDENCE_OPTIONS,
    LABEL_REVIEW_FIELDS,
    LABEL_RUBRIC_OPTIONS,
    normalize_evidence_tags,
)
from active_learning_thesis.md_workflow import MANIFEST_FIELDS, REVIEW_FIELDS, make_md_ingest_csv
from active_learning_thesis.dashboard_decisions import add_dashboard_decisions
from active_learning_thesis.dashboard_md_batches import (
    export_dashboard_md_source_batch,
    find_run_md_source_batch,
    load_md_source_batch_row,
)
from active_learning_thesis.dashboard_run_setup import normalize_run_name
from active_learning_thesis.dashboard_study_setup import normalize_study_name

ACTIONS_DIRNAME = "_dashboard_actions"
LOGS_DIRNAME = "logs"
APPROVAL_PENDING_STATUSES = {"draft", "awaiting_approval"}
ACTIVE_ACTION_STATUSES = {"queued", "running"}
FINAL_ACTION_STATUSES = {"succeeded", "failed", "cancelled", "manual_override"}
PAUSABLE_ACTION_STATUSES = {"draft", "awaiting_approval", "queued"}
RESUMABLE_ACTION_STATUSES = {"paused"}
CONFLICT_ACTION_STATUSES = {"awaiting_approval", "queued", "running"}
EXTERNAL_TERMINAL_ACTION_STATUSES = {"paused", "cancelled", "manual_override"}
REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKER_PROCESSES: list[subprocess.Popen[object]] = []


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_writable_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    probe_path = path / f".write_probe_{uuid.uuid4().hex}.tmp"
    try:
        probe_path.write_text("", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except PermissionError:
        probe_path.unlink(missing_ok=True)
        raise
    return path



def _fallback_state_root_path(run_root: Path) -> Path:
    user_root = Path.home() / ".active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return user_root / run_key / ACTIONS_DIRNAME


def _temp_fallback_state_root_path(run_root: Path) -> Path:
    temp_root = Path(tempfile.gettempdir()) / "active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return temp_root / run_key / ACTIONS_DIRNAME



def _fallback_state_root(run_root: Path) -> Path:
    for candidate in (_fallback_state_root_path(run_root), _temp_fallback_state_root_path(run_root)):
        try:
            return _ensure_writable_dir(candidate)
        except PermissionError:
            continue
    return _ensure_writable_dir(_temp_fallback_state_root_path(run_root))



def _resolved_actions_root(run_root: Path) -> Path:
    preferred = run_root / ACTIONS_DIRNAME
    try:
        _ensure_writable_dir(preferred)
        _ensure_writable_dir(preferred / LOGS_DIRNAME)
        return preferred
    except PermissionError:
        fallback = _fallback_state_root(run_root)
        _ensure_writable_dir(fallback / LOGS_DIRNAME)
        return fallback



def actions_root(run_root: Path) -> Path:
    return _resolved_actions_root(run_root)



def logs_root(run_root: Path) -> Path:
    preferred = actions_root(run_root) / LOGS_DIRNAME
    try:
        return _ensure_writable_dir(preferred)
    except PermissionError:
        return _ensure_writable_dir(_fallback_state_root(run_root) / LOGS_DIRNAME)



def _action_roots(run_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in [actions_root(run_root), run_root / ACTIONS_DIRNAME, _fallback_state_root_path(run_root), _temp_fallback_state_root_path(run_root)]:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            roots.append(candidate)
    return roots



def _action_path(run_root: Path, action_id: str) -> Path:
    preferred = actions_root(run_root) / f"{action_id}.json"
    if preferred.exists():
        return preferred
    fallback = _fallback_state_root_path(run_root) / f"{action_id}.json"
    if fallback.exists():
        return fallback
    temp_fallback = _temp_fallback_state_root_path(run_root) / f"{action_id}.json"
    if temp_fallback.exists():
        return temp_fallback
    return preferred



def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    last_error: Exception | None = None
    try:
        for _ in range(8):
            try:
                temp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error is not None:
            raise last_error
    finally:
        temp_path.unlink(missing_ok=True)



def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))



def _normalize_text_path(value: str | Path | None) -> str:
    if not value:
        return ""
    return str(Path(value).resolve())


def _normalize_sequence(value: str | None) -> str:
    return str(value or "").strip().upper()



def serialize_action_command(command: str, args: Sequence[str]) -> str:
    return subprocess.list2cmdline([command, *[str(part) for part in args]])



def display_python_command(args: Sequence[str]) -> str:
    return subprocess.list2cmdline(["python", *[str(part) for part in args]])



def list_dashboard_actions(run_root: Path) -> list[dict[str, object]]:
    actions_by_id: dict[str, dict[str, object]] = {}
    for root in _action_roots(run_root):
        for path in root.glob("*.json"):
            try:
                payload = _read_json(path)
            except Exception:
                continue
            payload["action_file"] = str(path)
            action_id = str(payload.get("id", path.stem))
            existing = actions_by_id.get(action_id)
            if existing is None:
                actions_by_id[action_id] = payload
                continue
            existing_updated = str(existing.get("updated_at") or existing.get("finished_at") or existing.get("created_at", ""))
            candidate_updated = str(payload.get("updated_at") or payload.get("finished_at") or payload.get("created_at", ""))
            if candidate_updated >= existing_updated:
                actions_by_id[action_id] = payload

    def sort_key(item: dict[str, object]) -> tuple[str, str]:
        return (str(item.get("created_at", "")), str(item.get("id", "")))

    return sorted(actions_by_id.values(), key=sort_key, reverse=True)



def load_dashboard_action(run_root: Path, action_id: str) -> dict[str, object]:
    path = _action_path(run_root, action_id)
    payload = _read_json(path)
    payload["action_file"] = str(path)
    return payload



def _write_action(run_root: Path, payload: dict[str, object]) -> Path:
    action_id = str(payload["id"])
    preferred_root = actions_root(run_root)
    preferred_path = preferred_root / f"{action_id}.json"
    try:
        _atomic_write_json(preferred_path, payload)
        return preferred_path
    except PermissionError:
        fallback_root = _fallback_state_root(run_root)
        fallback_path = fallback_root / f"{action_id}.json"
        _atomic_write_json(fallback_path, payload)
        return fallback_path



def _append_operator_note(existing_note: str, operator_note: str) -> str:
    if not operator_note.strip():
        return existing_note
    return f"{existing_note}\n{operator_note}".strip()



def update_dashboard_action(run_root: Path, action_id: str, **updates) -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    for key, value in updates.items():
        action[key] = value
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action



def _requested_terminal_status(action: dict[str, object]) -> str:
    status = str(action.get("status", ""))
    return status if status in EXTERNAL_TERMINAL_ACTION_STATUSES else ""



def _action_blocks_conflicts(action: dict[str, object]) -> bool:
    status = str(action.get("status", ""))
    if status in CONFLICT_ACTION_STATUSES:
        return True
    if status in FINAL_ACTION_STATUSES and (
        action.get("worker_pid") is not None or action.get("command_pid") is not None
    ):
        return True
    return False



def find_conflicting_action(
    run_root: Path,
    *,
    related_run: str = "",
    related_campaign: str = "",
    related_sequence: str = "",
    ignore_action_id: str = "",
) -> dict[str, object] | None:
    normalized_run = _normalize_text_path(related_run)
    normalized_campaign = _normalize_text_path(related_campaign)
    normalized_sequence = _normalize_sequence(related_sequence)
    for action in list_dashboard_actions(run_root):
        if ignore_action_id and str(action.get("id", "")) == ignore_action_id:
            continue
        if not bool(action.get("exclusive", True)):
            continue
        if not _action_blocks_conflicts(action):
            continue
        action_run = _normalize_text_path(str(action.get("related_run", "")))
        action_campaign = _normalize_text_path(str(action.get("related_campaign", "")))
        action_sequence = _normalize_sequence(str(action.get("related_sequence", "")))
        if normalized_campaign and action_campaign == normalized_campaign:
            return action
        if normalized_run and action_run == normalized_run:
            if normalized_sequence or action_sequence:
                if normalized_sequence and action_sequence and normalized_sequence != action_sequence:
                    continue
            return action
    return None


def _reap_worker_processes() -> None:
    global _WORKER_PROCESSES
    _WORKER_PROCESSES = [proc for proc in _WORKER_PROCESSES if proc.poll() is None]


def shutdown_dashboard_action_workers() -> None:
    global _WORKER_PROCESSES
    for process in list(_WORKER_PROCESSES):
        try:
            _terminate_running_process(process)
        except Exception:
            pass
        try:
            process.wait(timeout=0)
        except Exception:
            pass
    _WORKER_PROCESSES = []


def _spawn_action_worker(action_path: Path) -> None:
    _reap_worker_processes()
    command = [
        sys.executable,
        "-m",
        "active_learning_thesis.dashboard_action_worker",
        "--action-file",
        str(action_path),
    ]
    kwargs: dict[str, object] = {
        "cwd": str(REPO_ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    _WORKER_PROCESSES.append(process)


def _action_subprocess_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {"close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_process_tree(pid_value: str | int | None) -> None:
    pid_text = str(pid_value or "").strip()
    if not pid_text.isdigit():
        return
    pid = int(pid_text)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return



def _terminate_running_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except Exception:
        _terminate_process_tree(process.pid)
    for _ in range(20):
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        process.kill()
    except Exception:
        _terminate_process_tree(process.pid)
    try:
        process.wait(timeout=5)
    except Exception:
        return



def _finalize_external_terminal_request(
    action_path: Path,
    *,
    terminal_status: str,
) -> dict[str, object]:
    """Request shutdown of a running action and make its terminal state reliable."""
    payload = _read_json(action_path)
    _terminate_process_tree(payload.get("command_pid"))
    _terminate_process_tree(payload.get("worker_pid"))

    deadline = time.time() + 3.0
    while time.time() < deadline:
        latest = _read_json(action_path)
        if (
            str(latest.get("status", "")) == terminal_status
            and bool(latest.get("finished_at"))
            and latest.get("command_pid") is None
            and latest.get("worker_pid") is None
        ):
            latest["action_file"] = str(action_path)
            return latest
        time.sleep(0.1)

    latest = _read_json(action_path)
    latest["status"] = terminal_status
    latest["finished_at"] = latest.get("finished_at") or _now_iso()
    latest["command_pid"] = None
    latest["worker_pid"] = None
    _atomic_write_json(action_path, latest)
    latest["action_file"] = str(action_path)
    return latest


def create_dashboard_action(
    *,
    run_root: Path,
    title: str,
    kind: str,
    command: str,
    args: Sequence[str],
    cwd: Path,
    scope: str = "local",
    cluster: str = "",
    related_run: str = "",
    related_sequence: str = "",
    related_campaign: str = "",
    output_path: str | Path | None = None,
    display_command: str | None = None,
    launch_worker: bool = False,
    requires_approval: bool = False,
    initial_status: str | None = None,
    operator_note: str = "",
    metadata: dict[str, object] | None = None,
    exclusive: bool = True,
) -> dict[str, object]:
    run_root = run_root.resolve()
    cwd = cwd.resolve()
    related_run = _normalize_text_path(related_run)
    related_campaign = _normalize_text_path(related_campaign)
    output_path_text = _normalize_text_path(output_path)
    normalized_args = [str(part) for part in args]
    action_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    status = initial_status or ("awaiting_approval" if requires_approval else "queued")

    if exclusive and status in CONFLICT_ACTION_STATUSES:
        conflict = find_conflicting_action(
            run_root,
            related_run=related_run,
            related_campaign=related_campaign,
            related_sequence=related_sequence,
        )
        if conflict is not None:
            raise ValueError(
                "Another dashboard action is already queued, awaiting approval, or running for this run/campaign/peptide: "
                f"{conflict.get('title', conflict.get('id', 'unknown'))}"
            )

    stdout_log = logs_root(run_root) / f"{action_id}.stdout.log"
    stderr_log = logs_root(run_root) / f"{action_id}.stderr.log"
    payload: dict[str, object] = {
        "id": action_id,
        "title": title,
        "kind": kind,
        "scope": scope,
        "cluster": cluster,
        "command": command,
        "args": normalized_args,
        "cwd": str(cwd),
        "status": status,
        "requires_approval": requires_approval,
        "created_at": _now_iso(),
        "approval_timestamp": "",
        "started_at": "",
        "finished_at": "",
        "exit_code": None,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "related_run": related_run,
        "related_sequence": related_sequence,
        "related_campaign": related_campaign,
        "display_command": display_command or serialize_action_command(command, normalized_args),
        "output_path": output_path_text,
        "remote_job_id": "",
        "worker_pid": None,
        "command_pid": None,
        "sync_status": "",
        "operator_note": operator_note,
        "metadata": metadata or {},
        "exclusive": exclusive,
    }
    action_path = _write_action(run_root, payload)
    if launch_worker and status == "queued":
        _spawn_action_worker(action_path)
    payload["action_file"] = str(action_path)
    return payload



def submit_dashboard_action(**kwargs) -> dict[str, object]:
    kwargs.setdefault("scope", "local")
    kwargs.setdefault("launch_worker", True)
    kwargs.setdefault("requires_approval", False)
    kwargs.setdefault("initial_status", "queued")
    kwargs.setdefault("exclusive", True)
    return create_dashboard_action(**kwargs)



def draft_dashboard_action(**kwargs) -> dict[str, object]:
    kwargs.setdefault("launch_worker", False)
    kwargs.setdefault("requires_approval", True)
    kwargs.setdefault("initial_status", "awaiting_approval")
    kwargs.setdefault("exclusive", True)
    return create_dashboard_action(**kwargs)



def approve_dashboard_action(
    run_root: Path,
    action_id: str,
    *,
    operator_note: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    if action.get("status") not in APPROVAL_PENDING_STATUSES:
        raise ValueError(f"Action is not awaiting approval: {action_id}")

    conflict = find_conflicting_action(
        run_root,
        related_run=str(action.get("related_run", "")),
        related_campaign=str(action.get("related_campaign", "")),
        related_sequence=str(action.get("related_sequence", "")),
        ignore_action_id=action_id,
    )
    if conflict is not None:
        raise ValueError(
            "Another dashboard action is already queued, awaiting approval, or running for this run/campaign/peptide: "
            f"{conflict.get('title', conflict.get('id', 'unknown'))}"
        )

    action["operator_note"] = _append_operator_note(str(action.get("operator_note", "")), operator_note)
    action["status"] = "queued"
    action["approval_timestamp"] = _now_iso()
    action_path = _write_action(run_root, action)
    if launch_worker:
        _spawn_action_worker(action_path)
    action["action_file"] = str(action_path)
    return action



def pause_dashboard_action(run_root: Path, action_id: str, *, operator_note: str = "") -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    if action.get("status") not in PAUSABLE_ACTION_STATUSES:
        raise ValueError(f"Action cannot be paused from state: {action.get('status', '')}")
    action["status"] = "paused"
    action["operator_note"] = _append_operator_note(str(action.get("operator_note", "")), operator_note)
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action



def resume_dashboard_action(
    run_root: Path,
    action_id: str,
    *,
    operator_note: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    if action.get("status") not in RESUMABLE_ACTION_STATUSES:
        raise ValueError(f"Action cannot be resumed from state: {action.get('status', '')}")
    next_status = "queued"
    if bool(action.get("requires_approval", False)) and not str(action.get("approval_timestamp", "")).strip():
        next_status = "awaiting_approval"
    if bool(action.get("exclusive", True)) and next_status in CONFLICT_ACTION_STATUSES:
        conflict = find_conflicting_action(
            run_root,
            related_run=str(action.get("related_run", "")),
            related_campaign=str(action.get("related_campaign", "")),
            related_sequence=str(action.get("related_sequence", "")),
            ignore_action_id=action_id,
        )
        if conflict is not None:
            raise ValueError(
                "Another dashboard action is already queued, awaiting approval, or running for this run/campaign/peptide: "
                f"{conflict.get('title', conflict.get('id', 'unknown'))}"
            )
    action["status"] = next_status
    action["operator_note"] = _append_operator_note(str(action.get("operator_note", "")), operator_note)
    action_path = _write_action(run_root, action)
    if launch_worker and next_status == "queued":
        _spawn_action_worker(action_path)
    action["action_file"] = str(action_path)
    return action



def cancel_dashboard_action(run_root: Path, action_id: str, *, operator_note: str = "") -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    if action.get("status") in FINAL_ACTION_STATUSES:
        action["action_file"] = str(_action_path(run_root, action_id))
        return action
    was_running = action.get("status") == "running"
    action["status"] = "cancelled"
    if not was_running:
        action["finished_at"] = action.get("finished_at") or _now_iso()
    action["operator_note"] = _append_operator_note(str(action.get("operator_note", "")), operator_note)
    action_path = _write_action(run_root, action)
    if was_running:
        return _finalize_external_terminal_request(action_path, terminal_status="cancelled")
    action["action_file"] = str(action_path)
    return action



def mark_manual_override(run_root: Path, action_id: str, *, operator_note: str = "") -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    was_running = action.get("status") == "running"
    action["status"] = "manual_override"
    if not was_running:
        action["finished_at"] = action.get("finished_at") or _now_iso()
    action["operator_note"] = _append_operator_note(str(action.get("operator_note", "")), operator_note)
    action_path = _write_action(run_root, action)
    if was_running:
        return _finalize_external_terminal_request(action_path, terminal_status="manual_override")
    action["action_file"] = str(action_path)
    return action



def rerun_dashboard_action(
    run_root: Path,
    action_id: str,
    *,
    launch_worker: bool = True,
) -> dict[str, object]:
    action = load_dashboard_action(run_root, action_id)
    requires_approval = bool(action.get("requires_approval", False))
    initial_status = "awaiting_approval" if requires_approval else "queued"
    return create_dashboard_action(
        run_root=run_root,
        title=str(action.get("title", "Dashboard action")),
        kind=str(action.get("kind", "generic")),
        command=str(action["command"]),
        args=[str(part) for part in action.get("args", [])],
        cwd=Path(str(action["cwd"])),
        scope=str(action.get("scope", "local")),
        cluster=str(action.get("cluster", "")),
        related_run=str(action.get("related_run", "")),
        related_sequence=str(action.get("related_sequence", "")),
        related_campaign=str(action.get("related_campaign", "")),
        output_path=str(action.get("output_path", "")),
        display_command=str(action.get("display_command", "")) or None,
        launch_worker=launch_worker and not requires_approval,
        requires_approval=requires_approval,
        initial_status=initial_status,
        operator_note=str(action.get("operator_note", "")),
        metadata=dict(action.get("metadata", {})) if isinstance(action.get("metadata"), dict) else {},
        exclusive=bool(action.get("exclusive", True)),
    )



def submit_prepare_md_stage_action(
    *,
    run_root: Path,
    run_dir: Path,
    batch_csv: Path,
    sequence: str,
    campaign: str,
    md_profile: str,
    cluster: str = "bura",
    reuse_pdb_from: str | Path | None = None,
    exclude_nodes: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "prepare-md-stage",
        "--run-dir",
        str(run_dir),
        "--batch-csv",
        str(batch_csv),
        "--sequence",
        sequence,
        "--campaign",
        campaign,
        "--md-profile",
        md_profile,
        "--cluster",
        cluster,
    ]
    if reuse_pdb_from:
        args.extend(["--reuse-pdb-from", str(reuse_pdb_from)])
    if exclude_nodes:
        args.extend(["--exclude-nodes", exclude_nodes])
    campaign_dir = run_dir / "md_campaigns" / campaign
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Prepare {md_profile} for {sequence}",
        kind="prepare-md-stage",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(campaign_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"cluster": cluster},
    )


def submit_prepare_manual_md_stage_action(
    *,
    run_root: Path,
    run_dir: Path,
    sequence: str,
    campaign: str,
    md_profile: str,
    cluster: str = "bura",
    reuse_pdb_from: str | Path | None = None,
    exclude_nodes: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "prepare-manual-md-stage",
        "--run-dir",
        str(run_dir),
        "--sequence",
        sequence,
        "--campaign",
        campaign,
        "--md-profile",
        md_profile,
        "--cluster",
        cluster,
    ]
    if reuse_pdb_from:
        args.extend(["--reuse-pdb-from", str(reuse_pdb_from)])
    if exclude_nodes:
        args.extend(["--exclude-nodes", exclude_nodes])
    campaign_dir = run_dir / "md_campaigns" / campaign
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Prepare manual {md_profile} MD sandbox for {sequence}",
        kind="prepare-manual-md-stage",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(campaign_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"cluster": cluster, "manual_md_sandbox": "true"},
    )



def submit_finalize_md_stage_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    related_run: str,
    staged_package_dir: str | Path | None = None,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "finalize-md-stage",
        "--campaign-dir",
        str(campaign_dir),
    ]
    if staged_package_dir:
        args.extend(["--staged-package-dir", str(staged_package_dir)])
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Finalize MD stage for {sequence}",
        kind="finalize-md-stage",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(campaign_dir / "md_review.csv"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
    )


def submit_prepare_bura_md_benchmark_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    benchmark_name: str,
    nsteps: int,
    layouts: str = "",
    walltime: str = "02:00:00",
    related_run: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "prepare-bura-md-benchmark",
        "--campaign-dir",
        str(campaign_dir),
        "--sequence",
        sequence,
        "--benchmark-name",
        benchmark_name,
        "--nsteps",
        str(nsteps),
        "--walltime",
        walltime,
    ]
    if layouts:
        args.extend(["--layouts", layouts])
    benchmark_dir = campaign_dir / "bura_benchmarks" / benchmark_name
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Prepare BURA MD benchmark for {sequence}",
        kind="prepare-bura-md-benchmark",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(benchmark_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"benchmark_name": benchmark_name, "nsteps": str(nsteps), "layouts": layouts},
    )


def submit_parse_bura_md_benchmark_action(
    *,
    run_root: Path,
    benchmark_dir: Path,
    sequence: str,
    campaign_dir: Path,
    related_run: str = "",
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "parse-bura-md-benchmark",
        "--benchmark-dir",
        str(benchmark_dir),
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Parse BURA MD benchmark for {sequence}",
        kind="parse-bura-md-benchmark",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(benchmark_dir / "benchmark_results.csv"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
    )



def submit_make_md_ingest_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    review_csv: Path,
    sequence: str,
    related_run: str,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "make-md-ingest-csv",
        "--campaign-dir",
        str(campaign_dir),
        "--review-csv",
        str(review_csv),
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Create MD ingest CSV for {sequence}",
        kind="make-md-ingest-csv",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        output_path=str(campaign_dir / "cgmd_ingest.csv"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
    )


def submit_phase3_make_ingest_action(
    *,
    run_root: Path,
    output_root: Path,
    branch: str,
    round_id: int = 1,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "phase3-real-al",
        "make-ingest-csv",
        "--output-root",
        str(output_root),
        "--branch",
        str(branch),
        "--round",
        str(round_id),
    ]
    ingest_path = output_root / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "ingest" / "cgmd_ingest.csv"
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Create Phase 3 ingest CSV for {branch}",
        kind="phase3-make-ingest-csv",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(output_root / "branches" / branch),
        output_path=str(ingest_path),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"output_root": str(output_root), "branch": branch, "round_id": round_id},
    )


def submit_bulk_make_md_ingest_action(
    *,
    run_root: Path,
    items: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not items:
        raise ValueError("No MD ingest CSV items were provided.")

    started_at = _now_iso()
    normalized_items: list[dict[str, str]] = []
    related_runs: list[str] = []
    related_campaigns: list[str] = []
    sequences: list[str] = []
    output_paths: list[str] = []

    for item in items:
        campaign_dir = Path(str(item.get("campaign_dir", "")))
        review_csv = Path(str(item.get("review_csv", "")))
        sequence = str(item.get("sequence", "")).strip()
        if not sequence:
            raise ValueError("Each bulk ingest item must include a peptide sequence.")
        ingest_path = make_md_ingest_csv(campaign_dir, review_csv)
        normalized_items.append(
            {
                "campaign_dir": str(campaign_dir),
                "review_csv": str(review_csv),
                "sequence": sequence,
                "related_run": str(item.get("related_run", "")),
            }
        )
        related_run = str(item.get("related_run", "")).strip()
        if related_run:
            related_runs.append(related_run)
        related_campaigns.append(str(campaign_dir))
        output_paths.append(str(ingest_path))
        sequences.append(sequence)

    unique_runs = {item for item in related_runs if item}
    unique_campaigns = {item for item in related_campaigns if item}
    related_run_value = related_runs[0] if len(unique_runs) == 1 and related_runs else ""
    related_campaign_value = related_campaigns[0] if len(unique_campaigns) == 1 and related_campaigns else ""
    summary_bits = sequences[:5]
    if len(sequences) > 5:
        summary_bits.append(f"+{len(sequences) - 5} more")
    display_command = "bulk create cgmd_ingest.csv: " + ", ".join(summary_bits)
    action = create_dashboard_action(
        run_root=run_root,
        title=f"Create MD ingest CSVs ({len(normalized_items)} peptides)",
        kind="bulk-make-md-ingest",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-make-ingest-bulk"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run_value,
        related_campaign=related_campaign_value,
        display_command=display_command,
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "items": normalized_items,
            "output_paths": output_paths,
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def submit_bulk_candidate_decision_action(
    *,
    run_root: Path,
    items: Sequence[dict[str, object]],
    decision_type: str,
    title: str,
    rationale: str,
    next_step: str,
    evidence_prefix: str = "",
) -> dict[str, object]:
    if not items:
        raise ValueError("No candidate decision items were provided.")
    if not title.strip() or not rationale.strip():
        raise ValueError("Batch candidate decisions require both a title and a rationale.")

    started_at = _now_iso()
    normalized_items: list[dict[str, str]] = []
    decision_entries: list[dict[str, object]] = []
    related_runs: list[str] = []
    sequence_labels: list[str] = []
    for item in items:
        sequence = str(item.get("sequence", "")).strip()
        if not sequence:
            raise ValueError("Each candidate decision item must include a peptide sequence.")
        run_dir = str(item.get("run_dir", "")).strip()
        run_name = str(item.get("run_name", item.get("run", ""))).strip()
        source = str(item.get("source", "")).strip()
        strategy = str(item.get("strategy", "")).strip()
        priority_band = str(item.get("priority_band", "")).strip()
        item_next_step = str(item.get("next_step_override", "")).strip() or next_step
        evidence_parts = [part for part in [evidence_prefix.strip(), f"source={source}" if source else "", f"strategy={strategy}" if strategy else "", f"priority={priority_band}" if priority_band else ""] if part]
        decision_entries.append(
            {
                "scope": "candidate",
                "decision_type": decision_type,
                "title": f"{title}: {sequence}",
                "rationale": rationale,
                "run_dir": run_dir,
                "run_name": run_name,
                "sequence": sequence,
                "evidence": " | ".join(evidence_parts),
                "next_step": item_next_step,
            }
        )
        normalized_items.append(
            {
                "sequence": sequence,
                "run_dir": run_dir,
                "run_name": run_name,
                "source": source,
                "strategy": strategy,
                "priority_band": priority_band,
            }
        )
        if run_dir:
            related_runs.append(run_dir)
        sequence_labels.append(sequence)

    add_dashboard_decisions(run_root, entries=decision_entries)

    unique_runs = {item for item in related_runs if item}
    related_run_value = related_runs[0] if len(unique_runs) == 1 and related_runs else ""
    summary_bits = sequence_labels[:5]
    if len(sequence_labels) > 5:
        summary_bits.append(f"+{len(sequence_labels) - 5} more")
    display_command = f"bulk candidate decision ({decision_type}): " + ", ".join(summary_bits)
    action = create_dashboard_action(
        run_root=run_root,
        title=f"{title} ({len(normalized_items)} peptides)",
        kind="bulk-candidate-decision",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-save-candidate-bulk"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run_value,
        display_command=display_command,
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "decision_type": decision_type,
            "title": title,
            "rationale": rationale,
            "next_step": next_step,
            "items": normalized_items,
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def submit_export_md_source_batch_action(
    *,
    run_root: Path,
    items: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not items:
        raise ValueError("No MD source batch export items were provided.")

    started_at = _now_iso()
    normalized_items: list[dict[str, str]] = []
    related_runs: list[str] = []
    sequence_labels: list[str] = []
    output_paths: list[str] = []

    for item in items:
        sequence = str(item.get("sequence", "")).strip()
        run_dir = Path(str(item.get("run_dir", "")))
        if not sequence:
            raise ValueError("Each MD source batch export item must include a peptide sequence.")
        if not str(run_dir):
            raise ValueError(f"Missing run_dir for MD source batch export item {sequence}.")
        round_id = str(item.get("round_id", "")).strip() or "0"
        acquisition_strategy = str(item.get("strategy", "")).strip() or str(item.get("source", "")).strip() or "dashboard_md_export"
        batch_path = export_dashboard_md_source_batch(
            run_root,
            run_dir=run_dir,
            sequence=sequence,
            round_id=round_id,
            acquisition_strategy=acquisition_strategy,
        )
        normalized_items.append(
            {
                "sequence": sequence,
                "run_dir": str(run_dir),
                "run_name": str(item.get("run_name", item.get("run", ""))).strip(),
                "round_id": round_id,
                "strategy": acquisition_strategy,
                "source": str(item.get("source", "")).strip(),
                "output_path": str(batch_path),
            }
        )
        related_runs.append(str(run_dir))
        sequence_labels.append(sequence)
        output_paths.append(str(batch_path))

    unique_runs = {item for item in related_runs if item}
    related_run_value = related_runs[0] if len(unique_runs) == 1 and related_runs else ""
    summary_bits = sequence_labels[:5]
    if len(sequence_labels) > 5:
        summary_bits.append(f"+{len(sequence_labels) - 5} more")
    display_command = "export MD source batch CSV: " + ", ".join(summary_bits)
    action = create_dashboard_action(
        run_root=run_root,
        title=f"Create MD source batch CSVs ({len(normalized_items)} peptides)",
        kind="export-md-source-batch",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-export-md-source-batch"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run_value,
        display_command=display_command,
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "items": normalized_items,
            "output_paths": output_paths,
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def submit_promote_reporting_md_campaign_action(
    *,
    run_root: Path,
    campaign_dir: Path,
    sequence: str,
    related_run: str,
) -> dict[str, object]:
    if not str(related_run).strip():
        raise ValueError("Promotion requires the parent run directory.")
    run_dir = Path(str(related_run))
    batch_csv = find_run_md_source_batch(run_dir, sequence)
    if not batch_csv:
        raise ValueError(
            "This reporting-only peptide cannot be promoted yet because no real proposed batch currently contains it."
        )
    batch_path = Path(batch_csv)
    batch_row = load_md_source_batch_row(batch_path, sequence)
    manifest_path = campaign_dir / "manifest.csv"
    review_path = campaign_dir / "md_review.csv"
    meta_path = campaign_dir / "md_stage_meta.json"
    ingest_path = campaign_dir / "cgmd_ingest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.csv in campaign: {campaign_dir}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing md_review.csv in campaign: {campaign_dir}")

    manifest_fields, manifest_rows = _read_csv_rows(manifest_path)
    matched_manifest = False
    updated_manifest_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        updated = dict(row)
        if str(updated.get("sequence", "")).strip().upper() == str(sequence).strip().upper():
            for field in [
                "sequence",
                "round_id",
                "acquisition_strategy",
                "pred_mean",
                "pred_std",
                "pred_entropy",
                "pred_mutual_information",
                "acquisition_score",
            ]:
                updated[field] = str(batch_row.get(field, updated.get(field, "")))
            matched_manifest = True
        updated_manifest_rows.append(updated)
    if not matched_manifest:
        raise ValueError(f"Sequence {sequence!r} was not found in {manifest_path.name}.")
    _write_csv_rows(manifest_path, manifest_fields or MANIFEST_FIELDS, updated_manifest_rows)

    review_fields, review_rows = _read_csv_rows(review_path)
    matched_review = False
    updated_review_rows: list[dict[str, str]] = []
    for row in review_rows:
        updated = dict(row)
        if str(updated.get("sequence", "")).strip().upper() == str(sequence).strip().upper():
            updated["round_id"] = str(batch_row.get("round_id", updated.get("round_id", "")))
            matched_review = True
        updated_review_rows.append(updated)
    if not matched_review:
        raise ValueError(f"Sequence {sequence!r} was not found in {review_path.name}.")
    _write_csv_rows(review_path, review_fields or REVIEW_FIELDS, updated_review_rows)

    meta_payload: dict[str, object] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta_payload = loaded
        except Exception:
            meta_payload = {}
    previous_source_batch = str(meta_payload.get("source_batch_csv", "")).strip()
    if previous_source_batch and previous_source_batch != str(batch_path):
        meta_payload["promoted_from_source_batch_csv"] = previous_source_batch
    meta_payload["source_batch_csv"] = str(batch_path)
    meta_payload["promoted_to_real_batch_at"] = _now_iso()
    meta_payload["promoted_round_id"] = str(batch_row.get("round_id", ""))
    meta_payload["promotion_source"] = "real_proposed_batch"
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")

    removed_stale_ingest_csv = ingest_path.exists()
    if ingest_path.exists():
        ingest_path.unlink()

    started_at = _now_iso()
    action = create_dashboard_action(
        run_root=run_root,
        title=f"Promote {sequence} into real AL batch",
        kind="promote-reporting-md-campaign",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-promote-reporting-md-campaign"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=str(run_dir),
        related_sequence=sequence,
        related_campaign=str(campaign_dir),
        display_command=f"promote reporting-only MD campaign for {sequence} to {batch_path.name}",
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "sequence": sequence,
            "campaign_dir": str(campaign_dir),
            "batch_csv": str(batch_path),
            "round_id": str(batch_row.get("round_id", "")),
            "removed_stale_ingest_csv": removed_stale_ingest_csv,
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{str(key): str(value or "") for key, value in row.items()} for row in reader]
    return fieldnames, rows


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _update_md_review_row(
    *,
    review_csv: Path,
    sequence: str,
    cgmd_label: str,
    review_notes: str,
    label_rubric: str | None = None,
    label_confidence: str | None = None,
    label_evidence_tags: str | None = None,
    label_evidence_summary: str | None = None,
    reviewer: str | None = None,
) -> dict[str, dict[str, str]]:
    if not review_csv.exists():
        raise FileNotFoundError(f"Review CSV does not exist: {review_csv}")
    normalized_label = str(cgmd_label).strip()
    if normalized_label not in {"", "0", "1"}:
        raise ValueError("cgmd_label must be one of: empty, 0, or 1.")
    extra_updates: dict[str, str] = {}
    if label_rubric is not None:
        normalized_rubric = str(label_rubric).strip()
        if normalized_rubric not in LABEL_RUBRIC_OPTIONS:
            raise ValueError("label_rubric must be empty, self_assembling, not_self_assembling, or uncertain_rerun.")
        extra_updates["label_rubric"] = normalized_rubric
    if label_confidence is not None:
        normalized_confidence = str(label_confidence).strip().lower()
        if normalized_confidence not in LABEL_CONFIDENCE_OPTIONS:
            raise ValueError("label_confidence must be empty, high, medium, or low.")
        extra_updates["label_confidence"] = normalized_confidence
    if label_evidence_tags is not None:
        extra_updates["label_evidence_tags"] = normalize_evidence_tags(label_evidence_tags)
    if label_evidence_summary is not None:
        extra_updates["label_evidence_summary"] = str(label_evidence_summary).strip()
    if reviewer is not None:
        extra_updates["reviewer"] = str(reviewer).strip()
    if extra_updates and "reviewed_at" not in extra_updates:
        extra_updates["reviewed_at"] = _now_iso()

    fieldnames, rows = _read_csv_rows(review_csv)
    if not rows:
        raise ValueError("Review CSV is empty.")
    if "review_notes" not in fieldnames:
        fieldnames.append("review_notes")
    if "cgmd_label" not in fieldnames:
        fieldnames.append("cgmd_label")
    if extra_updates:
        for field in LABEL_REVIEW_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)
    matched = False
    changed_fields: dict[str, dict[str, str]] = {}
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        if str(updated.get("sequence", "")).strip() == sequence:
            before = dict(updated)
            updated["cgmd_label"] = normalized_label
            updated["review_notes"] = str(review_notes)
            updated.update(extra_updates)
            for field in sorted({"cgmd_label", "review_notes", *extra_updates.keys()}):
                previous = str(before.get(field, ""))
                new = str(updated.get(field, ""))
                if previous != new:
                    changed_fields[field] = {"previous": previous, "new": new}
            matched = True
        updated_rows.append(updated)
    if not matched:
        raise ValueError(f"Sequence {sequence!r} was not found in {review_csv.name}.")
    _write_csv_rows(review_csv, fieldnames, updated_rows)
    return changed_fields


def submit_update_md_review_action(
    *,
    run_root: Path,
    review_csv: Path,
    sequence: str,
    cgmd_label: str,
    review_notes: str,
    related_run: str,
    related_campaign: str,
    label_rubric: str | None = None,
    label_confidence: str | None = None,
    label_evidence_tags: str | None = None,
    label_evidence_summary: str | None = None,
    reviewer: str | None = None,
) -> dict[str, object]:
    started_at = _now_iso()
    changed_fields = _update_md_review_row(
        review_csv=review_csv,
        sequence=sequence,
        cgmd_label=cgmd_label,
        review_notes=review_notes,
        label_rubric=label_rubric,
        label_confidence=label_confidence,
        label_evidence_tags=label_evidence_tags,
        label_evidence_summary=label_evidence_summary,
        reviewer=reviewer,
    )
    label_text = str(cgmd_label).strip() or "<empty>"
    note_text = str(review_notes).strip() or "<empty>"
    rubric_text = str(label_rubric or "").strip()
    confidence_text = str(label_confidence or "").strip()
    evidence_bits = []
    if rubric_text:
        evidence_bits.append(f"rubric={rubric_text}")
    if confidence_text:
        evidence_bits.append(f"confidence={confidence_text}")
    action = create_dashboard_action(
        run_root=run_root,
        title=f"Save MD review decision for {sequence}",
        kind="update-md-review",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-save-review"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run,
        related_sequence=sequence,
        related_campaign=related_campaign,
        output_path=str(review_csv),
        display_command=(
            f"update md_review.csv for {sequence}: cgmd_label={label_text} review_notes={note_text!r}"
            + (f" {' '.join(evidence_bits)}" if evidence_bits else "")
        ),
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "review_csv": str(review_csv),
            "label_rubric": rubric_text,
            "label_confidence": confidence_text,
            "label_evidence_tags": normalize_evidence_tags(label_evidence_tags or ""),
            "label_evidence_summary": str(label_evidence_summary or "").strip(),
            "reviewer": str(reviewer or "").strip(),
            "changed_fields": changed_fields,
            "edited_at": started_at,
            "target_sequence": sequence,
            "related_campaign": related_campaign,
            "source_review_csv": str(review_csv),
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


def submit_bulk_update_md_review_action(
    *,
    run_root: Path,
    edits: Sequence[dict[str, object]],
) -> dict[str, object]:
    if not edits:
        raise ValueError("No MD review edits were provided.")

    started_at = _now_iso()
    normalized_edits: list[dict[str, str]] = []
    related_runs: list[str] = []
    related_campaigns: list[str] = []
    sequence_labels: list[str] = []
    review_csvs: set[str] = set()

    for edit in edits:
        review_csv = Path(str(edit.get("review_csv", "")))
        sequence = str(edit.get("sequence", "")).strip()
        if not sequence:
            raise ValueError("Each MD review edit must include a peptide sequence.")
        cgmd_label = str(edit.get("cgmd_label", ""))
        review_notes = str(edit.get("review_notes", ""))
        changed_fields = _update_md_review_row(
            review_csv=review_csv,
            sequence=sequence,
            cgmd_label=cgmd_label,
            review_notes=review_notes,
            label_rubric=str(edit["label_rubric"]) if "label_rubric" in edit else None,
            label_confidence=str(edit["label_confidence"]) if "label_confidence" in edit else None,
            label_evidence_tags=str(edit["label_evidence_tags"]) if "label_evidence_tags" in edit else None,
            label_evidence_summary=str(edit["label_evidence_summary"]) if "label_evidence_summary" in edit else None,
            reviewer=str(edit["reviewer"]) if "reviewer" in edit else None,
        )
        normalized_edits.append(
            {
                "review_csv": str(review_csv),
                "sequence": sequence,
                "cgmd_label": cgmd_label,
                "review_notes": review_notes,
                "label_rubric": str(edit.get("label_rubric", "")),
                "label_confidence": str(edit.get("label_confidence", "")),
                "label_evidence_tags": str(edit.get("label_evidence_tags", "")),
                "label_evidence_summary": str(edit.get("label_evidence_summary", "")),
                "reviewer": str(edit.get("reviewer", "")),
                "related_run": str(edit.get("related_run", "")),
                "related_campaign": str(edit.get("related_campaign", "")),
                "changed_fields": changed_fields,
                "edited_at": started_at,
            }
        )
        related_run = str(edit.get("related_run", "")).strip()
        related_campaign = str(edit.get("related_campaign", "")).strip()
        if related_run:
            related_runs.append(related_run)
        if related_campaign:
            related_campaigns.append(related_campaign)
        review_csvs.add(str(review_csv))
        label_text = cgmd_label.strip() or "<empty>"
        sequence_labels.append(f"{sequence}={label_text}")

    unique_runs = {item for item in related_runs if item}
    unique_campaigns = {item for item in related_campaigns if item}
    related_run_value = related_runs[0] if len(unique_runs) == 1 and related_runs else ""
    related_campaign_value = related_campaigns[0] if len(unique_campaigns) == 1 and related_campaigns else ""
    summary_bits = sequence_labels[:5]
    if len(sequence_labels) > 5:
        summary_bits.append(f"+{len(sequence_labels) - 5} more")
    display_command = "bulk update md_review.csv: " + ", ".join(summary_bits)
    action = create_dashboard_action(
        run_root=run_root,
        title=f"Save MD review decisions ({len(normalized_edits)} peptides)",
        kind="bulk-update-md-review",
        command=sys.executable,
        args=["-m", "active_learning_thesis", "dashboard-save-review-bulk"],
        cwd=REPO_ROOT,
        scope="local",
        related_run=related_run_value,
        related_campaign=related_campaign_value,
        display_command=display_command,
        launch_worker=False,
        requires_approval=False,
        initial_status="succeeded",
        exclusive=False,
        metadata={
            "review_csvs": sorted(review_csvs),
            "edits": normalized_edits,
        },
    )
    action["status"] = "succeeded"
    action["started_at"] = started_at
    action["finished_at"] = _now_iso()
    action["exit_code"] = 0
    action_path = _write_action(run_root, action)
    action["action_file"] = str(action_path)
    return action


SAFE_RUN_COMMANDS = {
    "propose-round": "Propose next peptide batch",
    "run-discovery": "Run peptide discovery",
    "evaluate-final": "Run frozen holdout evaluation",
    "run-replay": "Run replay benchmark",
}



def submit_run_workflow_action(
    *,
    run_root: Path,
    command_name: str,
    run_dir: Path,
    launch_worker: bool = True,
) -> dict[str, object]:
    if command_name not in SAFE_RUN_COMMANDS:
        raise ValueError(f"Unsupported dashboard run action: {command_name}")
    args = [
        "-m",
        "active_learning_thesis",
        command_name,
        "--run-dir",
        str(run_dir),
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title=SAFE_RUN_COMMANDS[command_name],
        kind=command_name,
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        output_path=str(run_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
    )


def submit_freeze_final_action(
    *,
    run_root: Path,
    run_dir: Path,
    metric: str = "f1",
    run_evaluation: bool = False,
    force: bool = False,
    allow_unresolved: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "freeze-final",
        "--run-dir",
        str(run_dir),
        "--metric",
        str(metric or "f1"),
    ]
    if run_evaluation:
        args.append("--run-evaluation")
    if force:
        args.append("--force")
    if allow_unresolved:
        args.append("--allow-unresolved")
    return submit_dashboard_action(
        run_root=run_root,
        title="Freeze final thesis result",
        kind="freeze-final",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        output_path=str(run_dir / "final_freeze"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={
            "metric": str(metric or "f1"),
            "run_evaluation": bool(run_evaluation),
            "force": bool(force),
            "allow_unresolved": bool(allow_unresolved),
        },
    )


def submit_thesis_packet_action(
    *,
    run_root: Path,
    title: str = "thesis_packet",
    metric: str = "f1",
    output_dir: Path | None = None,
    skip_dashboard: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "export-thesis-packet",
        "--run-root",
        str(run_root),
        "--title",
        str(title or "thesis_packet"),
        "--metric",
        str(metric or "f1"),
    ]
    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])
    if skip_dashboard:
        args.append("--skip-dashboard")
    return submit_dashboard_action(
        run_root=run_root,
        title="Export thesis packet",
        kind="export-thesis-packet",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        output_path=str(output_dir or (run_root / "_thesis_packets")),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        exclusive=False,
        metadata={
            "title": str(title or "thesis_packet"),
            "metric": str(metric or "f1"),
            "skip_dashboard": bool(skip_dashboard),
        },
    )


def submit_thesis_figures_action(
    *,
    run_root: Path,
    packet_dir: Path,
    metric: str = "",
    output_dir: Path | None = None,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "build-thesis-figures",
        "--packet-dir",
        str(packet_dir),
    ]
    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])
    if str(metric or "").strip():
        args.extend(["--metric", str(metric).strip()])
    return submit_dashboard_action(
        run_root=run_root,
        title="Build thesis figures",
        kind="build-thesis-figures",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(packet_dir),
        output_path=str(output_dir or (packet_dir / "thesis_figures")),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={
            "packet_dir": str(packet_dir),
            "metric": str(metric or "").strip(),
        },
    )


def submit_thesis_canary_action(
    *,
    run_root: Path,
    name: str = "seeded_thesis_canary",
    seed: int = 20260425,
    peptides: int = 2,
    force: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "thesis-canary",
        "--run-root",
        str(run_root),
        "--name",
        str(name or "seeded_thesis_canary"),
        "--seed",
        str(int(seed)),
        "--peptides",
        str(int(peptides)),
    ]
    if force:
        args.append("--force")
    return submit_dashboard_action(
        run_root=run_root,
        title="Run seeded thesis canary",
        kind="thesis-canary",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        output_path=str(run_root / "_thesis_canaries"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        exclusive=False,
        metadata={
            "name": str(name or "seeded_thesis_canary"),
            "seed": int(seed),
            "peptides": int(peptides),
            "force": bool(force),
        },
    )


def build_dashboard_init_run_action_payload(
    *,
    run_root: Path,
    run_name: str,
    random_seed: int,
    batch_size: int,
    max_rounds: int,
    epochs: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    replay_strategies: list[str] | None = None,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    pin_run: bool = True,
    run_label: str = "",
    train_baseline_after_init: bool = True,
    run_replay_after_init: bool = False,
) -> dict[str, object]:
    normalized_run_name = normalize_run_name(str(run_name), fallback="")
    if not normalized_run_name:
        raise ValueError("Run name is empty.")
    args = [
        "-m",
        "active_learning_thesis",
        "dashboard-init-run",
        "--run-root",
        str(run_root),
        "--run-name",
        normalized_run_name,
        "--random-seed",
        str(int(random_seed)),
        "--batch-size",
        str(int(batch_size)),
        "--max-rounds",
        str(int(max_rounds)),
        "--epochs",
        str(int(epochs)),
        "--candidate-pool-min",
        str(int(candidate_pool_min)),
        "--replay-seed-size",
        str(int(replay_seed_size)),
        "--real-strategy",
        str(real_strategy or "ensemble_mi"),
        "--replay-strategies",
        ",".join(replay_strategies or []),
        "--generator-objective-mode",
        str(generator_objective_mode),
        "--binary-threshold-strategy",
        str(binary_threshold_strategy),
    ]
    if train_family_for_init:
        args.append("--train-family-for-init")
    if not use_calibrated_acquisition:
        args.append("--raw-acquisition")
    if use_similarity_penalty:
        args.append("--use-similarity-penalty")
    if not use_length_penalty:
        args.append("--no-length-penalty")
    if pin_run:
        args.append("--pin-run")
    if str(run_label).strip():
        args.extend(["--run-label", str(run_label).strip()])
    if not train_baseline_after_init:
        args.append("--skip-baseline-init")
    if run_replay_after_init:
        args.append("--run-replay-after-init")
    target_run_dir = run_root / normalized_run_name
    metadata = {
        "run_name": normalized_run_name,
        "random_seed": int(random_seed),
        "batch_size": int(batch_size),
        "max_rounds": int(max_rounds),
        "epochs": int(epochs),
        "candidate_pool_min": int(candidate_pool_min),
        "replay_seed_size": int(replay_seed_size),
        "real_strategy": str(real_strategy or "ensemble_mi"),
        "replay_strategies": list(replay_strategies or []),
        "train_family_for_init": bool(train_family_for_init),
        "use_calibrated_acquisition": bool(use_calibrated_acquisition),
        "generator_objective_mode": str(generator_objective_mode),
        "use_similarity_penalty": bool(use_similarity_penalty),
        "use_length_penalty": bool(use_length_penalty),
        "binary_threshold_strategy": str(binary_threshold_strategy),
        "pin_run": bool(pin_run),
        "run_label": str(run_label).strip(),
        "train_baseline_after_init": bool(train_baseline_after_init),
        "run_replay_after_init": bool(run_replay_after_init),
    }
    return {
        "run_name": normalized_run_name,
        "target_run_dir": str(target_run_dir),
        "command": sys.executable,
        "args": args,
        "cwd": str(REPO_ROOT),
        "display_command": display_python_command(args),
        "metadata": metadata,
    }


def submit_dashboard_init_run_action(
    *,
    run_root: Path,
    run_name: str,
    random_seed: int,
    batch_size: int,
    max_rounds: int,
    epochs: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    replay_strategies: list[str] | None = None,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    pin_run: bool = True,
    run_label: str = "",
    train_baseline_after_init: bool = True,
    run_replay_after_init: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    payload = build_dashboard_init_run_action_payload(
        run_root=run_root,
        run_name=run_name,
        random_seed=random_seed,
        batch_size=batch_size,
        max_rounds=max_rounds,
        epochs=epochs,
        candidate_pool_min=candidate_pool_min,
        replay_seed_size=replay_seed_size,
        real_strategy=real_strategy,
        replay_strategies=replay_strategies,
        train_family_for_init=train_family_for_init,
        use_calibrated_acquisition=use_calibrated_acquisition,
        generator_objective_mode=generator_objective_mode,
        use_similarity_penalty=use_similarity_penalty,
        use_length_penalty=use_length_penalty,
        binary_threshold_strategy=binary_threshold_strategy,
        pin_run=pin_run,
        run_label=run_label,
        train_baseline_after_init=train_baseline_after_init,
        run_replay_after_init=run_replay_after_init,
    )
    return submit_dashboard_action(
        run_root=run_root,
        title="Create new thesis run",
        kind="init-run",
        command=str(payload["command"]),
        args=list(payload["args"]),
        cwd=Path(str(payload["cwd"])),
        related_run=str(payload["target_run_dir"]),
        output_path=str(payload["target_run_dir"]),
        display_command=str(payload["display_command"]),
        launch_worker=launch_worker,
        metadata=dict(payload["metadata"]),
    )


def submit_run_study_action(
    *,
    run_root: Path,
    study_name: str,
    seeds: int,
    seed_start: int,
    seed_step: int,
    epochs: int,
    max_rounds: int,
    batch_size: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    strategies: list[str] | None = None,
    metric: str = "f1",
    target: float | None = None,
    ensemble_size: int = 5,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    dry_run: bool = False,
    force_replay: bool = False,
    summarize: bool = True,
    allow_config_mismatch: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    normalized_study_name = normalize_study_name(str(study_name), fallback="")
    if not normalized_study_name:
        raise ValueError("Study name is empty.")
    args = [
        "-m",
        "active_learning_thesis",
        "run-study",
        "--study-name",
        normalized_study_name,
        "--run-root",
        str(run_root),
        "--seeds",
        str(int(seeds)),
        "--seed-start",
        str(int(seed_start)),
        "--seed-step",
        str(int(seed_step)),
        "--epochs",
        str(int(epochs)),
        "--max-rounds",
        str(int(max_rounds)),
        "--batch-size",
        str(int(batch_size)),
        "--candidate-pool-min",
        str(int(candidate_pool_min)),
        "--replay-seed-size",
        str(int(replay_seed_size)),
        "--real-strategy",
        str(real_strategy or "ensemble_mi"),
        "--ensemble-size",
        str(int(ensemble_size)),
        "--metric",
        str(metric or "f1"),
        "--generator-objective-mode",
        str(generator_objective_mode),
        "--binary-threshold-strategy",
        str(binary_threshold_strategy),
    ]
    if strategies:
        args.extend(["--strategies", *[str(item) for item in strategies]])
    if target is not None:
        args.extend(["--target", str(target)])
    if train_family_for_init:
        args.append("--train-family-for-init")
    if not use_calibrated_acquisition:
        args.append("--raw-acquisition")
    if use_similarity_penalty:
        args.append("--use-similarity-penalty")
    if not use_length_penalty:
        args.append("--no-length-penalty")
    if dry_run:
        args.append("--dry-run")
    if force_replay:
        args.append("--force-replay")
    if not summarize:
        args.append("--no-summarize")
    if allow_config_mismatch:
        args.append("--allow-config-mismatch")
    study_dir = run_root / "_studies" / normalized_study_name
    return submit_dashboard_action(
        run_root=run_root,
        title="Run study / ablation",
        kind="run-study",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(study_dir),
        output_path=str(study_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={
            "study_name": normalized_study_name,
            "seeds": int(seeds),
            "seed_start": int(seed_start),
            "seed_step": int(seed_step),
            "epochs": int(epochs),
            "max_rounds": int(max_rounds),
            "batch_size": int(batch_size),
            "candidate_pool_min": int(candidate_pool_min),
            "replay_seed_size": int(replay_seed_size),
            "real_strategy": str(real_strategy or "ensemble_mi"),
            "strategies": list(str(item) for item in (strategies or [])),
            "metric": str(metric or "f1"),
            "target": target,
            "ensemble_size": int(ensemble_size),
            "train_family_for_init": bool(train_family_for_init),
            "use_calibrated_acquisition": bool(use_calibrated_acquisition),
            "generator_objective_mode": str(generator_objective_mode),
            "use_similarity_penalty": bool(use_similarity_penalty),
            "use_length_penalty": bool(use_length_penalty),
            "binary_threshold_strategy": str(binary_threshold_strategy),
            "dry_run": bool(dry_run),
            "force_replay": bool(force_replay),
            "summarize": bool(summarize),
            "allow_config_mismatch": bool(allow_config_mismatch),
        },
    )


def submit_summarize_study_action(
    *,
    run_root: Path,
    metric: str = "f1",
    target: float | None = None,
    output_dir: Path | None = None,
    run_names: list[str] | None = None,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "summarize-study",
        "--run-root",
        str(run_root),
        "--metric",
        str(metric or "f1"),
    ]
    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])
    if target is not None:
        args.extend(["--target", str(target)])
    if run_names:
        args.extend(["--runs", *[str(item) for item in run_names]])
    resolved_output = output_dir or (run_root / "_study_evidence")
    return submit_dashboard_action(
        run_root=run_root,
        title="Summarize study evidence",
        kind="summarize-study",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(resolved_output),
        output_path=str(resolved_output),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        exclusive=False,
        metadata={
            "metric": str(metric or "f1"),
            "target": target,
            "output_dir": str(resolved_output),
            "run_names": list(run_names or []),
        },
    )


def submit_compare_studies_action(
    *,
    run_root: Path,
    baseline_study: str,
    candidate_study: str,
    metric: str = "f1",
    target: float | None = None,
    output_dir: Path | None = None,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "compare-studies",
        "--run-root",
        str(run_root),
        "--baseline-study",
        str(baseline_study),
        "--candidate-study",
        str(candidate_study),
        "--metric",
        str(metric or "f1"),
    ]
    if target is not None:
        args.extend(["--target", str(target)])
    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])
    return submit_dashboard_action(
        run_root=run_root,
        title="Compare studies",
        kind="compare-studies",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(output_dir or (run_root / "_studies" / "_comparisons")),
        output_path=str(output_dir or (run_root / "_studies" / "_comparisons")),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        exclusive=False,
        metadata={
            "baseline_study": str(baseline_study),
            "candidate_study": str(candidate_study),
            "metric": str(metric or "f1"),
            "target": target,
            "output_dir": str(output_dir) if output_dir is not None else "",
        },
    )


def submit_ingest_round_action(
    *,
    run_root: Path,
    run_dir: Path,
    import_csv: Path,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "ingest-round",
        "--run-dir",
        str(run_dir),
        "--import-csv",
        str(import_csv),
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title="Ingest returned labels",
        kind="ingest-round",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        output_path=str(run_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"run_dir": str(run_dir), "import_csv": str(import_csv)},
    )


def submit_phase3_ingest_action(
    *,
    run_root: Path,
    output_root: Path,
    branch: str,
    round_id: int = 1,
    import_csv: Path | None = None,
    launch_worker: bool = True,
) -> dict[str, object]:
    ingest_csv = import_csv or (
        output_root
        / "branches"
        / branch
        / "rounds"
        / f"round_{round_id:03d}"
        / "ingest"
        / "cgmd_ingest.csv"
    )
    args = [
        "-m",
        "active_learning_thesis",
        "phase3-real-al",
        "ingest",
        "--output-root",
        str(output_root),
        "--branch",
        str(branch),
        "--round",
        str(round_id),
        "--import-csv",
        str(ingest_csv),
    ]
    return submit_dashboard_action(
        run_root=run_root,
        title=f"Ingest Phase 3 labels into {branch}",
        kind="phase3-ingest",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(output_root / "branches" / branch),
        output_path=str(output_root / "branches" / branch / "current_labeled_ledger.csv"),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={
            "output_root": str(output_root),
            "branch": branch,
            "round_id": round_id,
            "import_csv": str(ingest_csv),
        },
    )



def submit_continue_feedback_action(
    *,
    run_root: Path,
    run_dir: Path,
    propose_next_batch: bool = False,
    launch_worker: bool = True,
) -> dict[str, object]:
    args = [
        "-m",
        "active_learning_thesis",
        "dashboard-continue-feedback",
        "--run-dir",
        str(run_dir),
    ]
    if propose_next_batch:
        args.append("--propose-next-batch")
    title = "Continue AL from reviewed peptides"
    if propose_next_batch:
        title += " and propose next batch"
    return submit_dashboard_action(
        run_root=run_root,
        title=title,
        kind="continue-al-feedback",
        command=sys.executable,
        args=args,
        cwd=REPO_ROOT,
        related_run=str(run_dir),
        output_path=str(run_dir),
        display_command=display_python_command(args),
        launch_worker=launch_worker,
        metadata={"propose_next_batch": propose_next_batch},
    )


def execute_action_file(action_file: Path) -> dict[str, object]:
    action_path = Path(action_file)
    payload = _read_json(action_path)
    requested_terminal_status = _requested_terminal_status(payload)
    if requested_terminal_status:
        payload["status"] = requested_terminal_status
        payload["finished_at"] = payload.get("finished_at") or _now_iso()
        payload["worker_pid"] = None
        payload["command_pid"] = None
        _atomic_write_json(action_path, payload)
        payload["action_file"] = str(action_path)
        return payload
    if payload.get("status") != "queued":
        raise ValueError(f"Only queued actions can be executed, got: {payload.get('status', '')}")

    payload["status"] = "running"
    payload["started_at"] = _now_iso()
    payload["finished_at"] = ""
    payload["exit_code"] = None
    payload["worker_pid"] = os.getpid()
    payload["command_pid"] = None
    _atomic_write_json(action_path, payload)

    stdout_log = Path(str(payload["stdout_log"]))
    stderr_log = Path(str(payload["stderr_log"]))
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    process: subprocess.Popen[str] | None = None
    try:
        with stdout_log.open("w", encoding="utf-8", newline="") as stdout_handle, stderr_log.open(
            "w", encoding="utf-8", newline=""
        ) as stderr_handle:
            process = subprocess.Popen(
                [str(payload["command"]), *[str(part) for part in payload.get("args", [])]],
                cwd=str(payload["cwd"]),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_action_subprocess_kwargs(),
            )
            payload["command_pid"] = process.pid
            _atomic_write_json(action_path, payload)

            termination_requested = ""
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                latest = _read_json(action_path)
                latest_terminal_status = _requested_terminal_status(latest)
                if latest_terminal_status:
                    termination_requested = latest_terminal_status
                    _terminate_running_process(process)
                    break
                time.sleep(0.2)

            if process.poll() is None:
                process.wait(timeout=10)

            latest = _read_json(action_path)
            payload = latest
            latest_terminal_status = _requested_terminal_status(latest)
            if latest_terminal_status or termination_requested:
                payload["status"] = latest_terminal_status or termination_requested
                payload["exit_code"] = process.returncode
            else:
                payload["status"] = "succeeded" if process.returncode == 0 else "failed"
                payload["exit_code"] = process.returncode
    except KeyboardInterrupt:
        if process is not None:
            _terminate_running_process(process)
        payload["status"] = "cancelled"
        payload["exit_code"] = None
    except Exception as exc:
        latest = _read_json(action_path) if action_path.exists() else payload
        latest_terminal_status = _requested_terminal_status(latest)
        if latest_terminal_status:
            payload = latest
            payload["status"] = latest_terminal_status
            payload["exit_code"] = process.returncode if process is not None else None
        else:
            stderr_log.write_text(f"{exc}\n", encoding="utf-8")
            payload = latest
            payload["status"] = "failed"
            payload["exit_code"] = None
    payload["worker_pid"] = None
    payload["command_pid"] = None
    payload["finished_at"] = _now_iso()
    try:
        from active_learning_thesis.dashboard_remote import post_process_action

        post_updates = post_process_action(action_path.parent.parent, payload)
        if isinstance(post_updates, dict):
            payload.update(post_updates)
    except Exception as exc:
        if payload["status"] == "succeeded":
            payload["status"] = "failed"
        stderr_log.write_text(
            stderr_log.read_text(encoding="utf-8", errors="replace") + f"\npost_process_error: {exc}\n",
            encoding="utf-8",
        )
    _atomic_write_json(action_path, payload)
    payload["action_file"] = str(action_path)
    return payload



def read_log_excerpt(path: str | Path, *, max_lines: int = 20) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    text = target.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])



def open_local_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen(
        [opener, str(target)],
        cwd=str(target.parent if target.is_file() else target),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )




