from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

MD_SLATES_FILENAME = "dashboard_md_slates.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _canonical_local_path(value: str | Path | None, *, base: Path | None = None) -> str:
    if not value:
        return ""
    try:
        raw_path = Path(value)
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path.resolve())
        else:
            if base is not None:
                candidates.append((base / raw_path).resolve())
            candidates.append(raw_path.resolve())
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0]) if candidates else str(raw_path)
    except Exception:
        return str(value)


def _preferred_md_slates_path(run_root: Path) -> Path:
    user_root = Path.home() / ".active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return user_root / run_key / MD_SLATES_FILENAME


def _fallback_md_slates_path(run_root: Path) -> Path:
    temp_root = Path(tempfile.gettempdir()) / "active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return temp_root / run_key / MD_SLATES_FILENAME


def dashboard_md_slates_path(run_root: Path) -> Path:
    preferred = _preferred_md_slates_path(run_root)
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = _fallback_md_slates_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def default_dashboard_md_slates() -> dict[str, object]:
    return {"slates": []}


def _normalize_stage_entry(run_root: Path, entry: dict[str, object]) -> dict[str, object]:
    return {
        "md_profile": str(entry.get("md_profile", "")).strip(),
        "campaign": str(entry.get("campaign", "")).strip(),
        "campaign_dir": _canonical_local_path(entry.get("campaign_dir"), base=run_root),
        "step": str(entry.get("step", "")).strip() or "prepare",
        "status": str(entry.get("status", "")).strip() or "pending",
        "started_at": str(entry.get("started_at", "")).strip(),
        "updated_at": str(entry.get("updated_at", "")).strip() or _now_iso(),
        "completed_at": str(entry.get("completed_at", "")).strip(),
        "last_action_id": str(entry.get("last_action_id", "")).strip(),
        "last_action_kind": str(entry.get("last_action_kind", "")).strip(),
        "last_action_title": str(entry.get("last_action_title", "")).strip(),
        "last_action_status": str(entry.get("last_action_status", "")).strip(),
        "remote_job_id": str(entry.get("remote_job_id", "")).strip(),
        "sync_status": str(entry.get("sync_status", "")).strip(),
        "failure_reason": str(entry.get("failure_reason", "")).strip(),
        "waiting_reason": str(entry.get("waiting_reason", "")).strip(),
        "review_ready": bool(entry.get("review_ready", False)),
    }


def _normalize_peptide_entry(run_root: Path, entry: dict[str, object]) -> dict[str, object]:
    stage_history = entry.get("stage_history", [])
    return {
        "sequence": str(entry.get("sequence", "")).strip(),
        "run_dir": _canonical_local_path(entry.get("run_dir"), base=run_root),
        "run_name": str(entry.get("run_name", "")).strip(),
        "source_batch_csv": _canonical_local_path(entry.get("source_batch_csv"), base=run_root),
        "source": str(entry.get("source", "")).strip(),
        "strategy": str(entry.get("strategy", "")).strip(),
        "priority_band": str(entry.get("priority_band", "")).strip(),
        "status": str(entry.get("status", "")).strip() or "pending",
        "current_stage": str(entry.get("current_stage", "")).strip() or "line_smoke",
        "current_step": str(entry.get("current_step", "")).strip() or "prepare",
        "current_campaign": str(entry.get("current_campaign", "")).strip(),
        "current_campaign_dir": _canonical_local_path(entry.get("current_campaign_dir"), base=run_root),
        "last_action_id": str(entry.get("last_action_id", "")).strip(),
        "last_action_kind": str(entry.get("last_action_kind", "")).strip(),
        "last_action_title": str(entry.get("last_action_title", "")).strip(),
        "last_action_status": str(entry.get("last_action_status", "")).strip(),
        "last_update_at": str(entry.get("last_update_at", "")).strip() or _now_iso(),
        "remote_job_id": str(entry.get("remote_job_id", "")).strip(),
        "failure_reason": str(entry.get("failure_reason", "")).strip(),
        "waiting_reason": str(entry.get("waiting_reason", "")).strip(),
        "blocked_stage": str(entry.get("blocked_stage", "")).strip(),
        "review_ready": bool(entry.get("review_ready", False)),
        "poll_not_before": str(entry.get("poll_not_before", "")).strip(),
        "stage_history": [
            _normalize_stage_entry(run_root, stage_entry)
            for stage_entry in stage_history
            if isinstance(stage_entry, dict)
        ] if isinstance(stage_history, list) else [],
    }


def _normalize_slate(run_root: Path, slate: dict[str, object]) -> dict[str, object]:
    peptides = slate.get("peptides", [])
    caps = slate.get("caps", {})
    rehearsal_summary = slate.get("rehearsal_summary", {})
    rehearsal_events = slate.get("rehearsal_events", [])
    return {
        "slate_id": str(slate.get("slate_id", "")).strip(),
        "run_dir": _canonical_local_path(slate.get("run_dir"), base=run_root),
        "run_name": str(slate.get("run_name", "")).strip(),
        "execution_mode": str(slate.get("execution_mode", "live")).strip() or "live",
        "created_at": str(slate.get("created_at", "")).strip() or _now_iso(),
        "updated_at": str(slate.get("updated_at", "")).strip() or _now_iso(),
        "status": str(slate.get("status", "")).strip() or "awaiting_approval",
        "paused": bool(slate.get("paused", False)),
        "supervisor_action_id": str(slate.get("supervisor_action_id", "")).strip(),
        "planner_id": str(slate.get("planner_id", "")).strip(),
        "planner_name": str(slate.get("planner_name", "")).strip(),
        "operator_note": str(slate.get("operator_note", "")).strip(),
        "caps": {
            "line_smoke": int(caps.get("line_smoke", 2)) if isinstance(caps, dict) else 2,
            "production_smoke": int(caps.get("production_smoke", 1)) if isinstance(caps, dict) else 1,
            "full": int(caps.get("full", 1)) if isinstance(caps, dict) else 1,
        },
        "rehearsal_summary": dict(rehearsal_summary) if isinstance(rehearsal_summary, dict) else {},
        "rehearsal_events": [
            {str(key): str(value) for key, value in event.items()}
            for event in rehearsal_events
            if isinstance(event, dict)
        ] if isinstance(rehearsal_events, list) else [],
        "peptides": [
            _normalize_peptide_entry(run_root, peptide)
            for peptide in peptides
            if isinstance(peptide, dict)
        ] if isinstance(peptides, list) else [],
    }


def _normalize_payload(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_slates = payload.get("slates", [])
    slates_by_id: dict[str, dict[str, object]] = {}
    if isinstance(raw_slates, list):
        for raw_slate in raw_slates:
            if not isinstance(raw_slate, dict):
                continue
            slate = _normalize_slate(run_root, raw_slate)
            slate_id = str(slate.get("slate_id", "")).strip()
            if slate_id:
                slates_by_id[slate_id] = slate
    return {
        "slates": sorted(
            slates_by_id.values(),
            key=lambda item: (str(item.get("updated_at", "")), str(item.get("slate_id", ""))),
            reverse=True,
        )
    }


def load_dashboard_md_slates(run_root: Path) -> dict[str, object]:
    path = dashboard_md_slates_path(run_root)
    slates_by_id: dict[str, dict[str, object]] = {}
    exists = False
    for candidate in [path, _preferred_md_slates_path(run_root), _fallback_md_slates_path(run_root)]:
        if not candidate.exists():
            continue
        exists = True
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(loaded, dict):
            continue
        normalized = _normalize_payload(run_root, loaded)
        for slate in normalized["slates"]:
            if isinstance(slate, dict):
                slates_by_id[str(slate.get("slate_id", ""))] = slate
    payload = {
        "slates": sorted(
            slates_by_id.values(),
            key=lambda item: (str(item.get("updated_at", "")), str(item.get("slate_id", ""))),
            reverse=True,
        )
    }
    payload["path"] = str(path)
    payload["exists"] = exists
    return payload


def save_dashboard_md_slates(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_md_slates_path(run_root)
    normalized = _normalize_payload(run_root, payload)
    body = {"slates": normalized["slates"]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_md_slates_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def list_dashboard_md_slates(run_root: Path) -> list[dict[str, object]]:
    loaded = load_dashboard_md_slates(run_root)
    slates = loaded.get("slates", [])
    return list(slates) if isinstance(slates, list) else []


def load_dashboard_md_slate(run_root: Path, slate_id: str) -> dict[str, object]:
    for slate in list_dashboard_md_slates(run_root):
        if str(slate.get("slate_id", "")) == slate_id:
            return slate
    raise FileNotFoundError(f"MD slate does not exist: {slate_id}")


def save_dashboard_md_slate(run_root: Path, slate: dict[str, object]) -> dict[str, object]:
    payload = load_dashboard_md_slates(run_root)
    slates = list(payload.get("slates", [])) if isinstance(payload.get("slates", []), list) else []
    normalized = _normalize_slate(run_root, slate)
    slate_id = str(normalized.get("slate_id", "")).strip()
    if not slate_id:
        raise ValueError("MD slate is missing slate_id.")
    existing = [item for item in slates if str(item.get("slate_id", "")) != slate_id]
    existing.append(normalized)
    save_dashboard_md_slates(run_root, {"slates": existing})
    return load_dashboard_md_slate(run_root, slate_id)
