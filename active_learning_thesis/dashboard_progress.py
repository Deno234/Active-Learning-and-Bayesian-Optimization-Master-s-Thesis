from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from active_learning_thesis.dashboard_state_paths import (
    dashboard_state_path,
    fallback_dashboard_state_path,
    preferred_dashboard_state_path,
)

PROGRESS_FILENAME = "dashboard_progress.json"


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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _preferred_progress_path(run_root: Path) -> Path:
    return preferred_dashboard_state_path(run_root, PROGRESS_FILENAME)


def _fallback_progress_path(run_root: Path) -> Path:
    return fallback_dashboard_state_path(run_root, PROGRESS_FILENAME)


def dashboard_progress_path(run_root: Path) -> Path:
    return dashboard_state_path(run_root, PROGRESS_FILENAME)


def default_dashboard_progress() -> dict[str, object]:
    return {"entries": []}


def _normalize_entry(run_root: Path, entry: dict[str, object]) -> dict[str, object]:
    created_at = str(entry.get("created_at", "")).strip() or _now_iso()
    return {
        "id": str(entry.get("id", "")).strip() or uuid.uuid4().hex[:12],
        "created_at": created_at,
        "scope": str(entry.get("scope", "")).strip() or "global",
        "plan_kind": str(entry.get("plan_kind", "")).strip() or "unknown",
        "run_dir": _canonical_local_path(entry.get("run_dir"), base=run_root),
        "sequence": str(entry.get("sequence", "")).strip(),
        "campaign_dir": _canonical_local_path(entry.get("campaign_dir"), base=run_root),
        "checkpoint": str(entry.get("checkpoint", "")).strip(),
        "action_label": str(entry.get("action_label", "")).strip(),
        "action_status": str(entry.get("action_status", "")).strip(),
        "action_id": str(entry.get("action_id", "")).strip(),
        "note": str(entry.get("note", "")).strip(),
    }


def _normalize_payload(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_entries = payload.get("entries", [])
    entries_by_id: dict[str, dict[str, object]] = {}
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = _normalize_entry(run_root, raw_entry)
            entries_by_id[str(entry["id"])] = entry
    entries = sorted(
        entries_by_id.values(),
        key=lambda item: (str(item.get("created_at", "")), str(item.get("id", ""))),
        reverse=True,
    )
    return {"entries": entries}


def load_dashboard_progress(run_root: Path) -> dict[str, object]:
    path = dashboard_progress_path(run_root)
    entries_by_id: dict[str, dict[str, object]] = {}
    exists = False
    for candidate in [path, _preferred_progress_path(run_root), _fallback_progress_path(run_root)]:
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
        for entry in normalized["entries"]:
            if isinstance(entry, dict):
                entries_by_id[str(entry.get("id", ""))] = entry
    merged = {
        "entries": sorted(
            entries_by_id.values(),
            key=lambda item: (str(item.get("created_at", "")), str(item.get("id", ""))),
            reverse=True,
        )
    }
    merged["path"] = str(path)
    merged["exists"] = exists
    return merged


def save_dashboard_progress(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_progress_path(run_root)
    normalized = _normalize_payload(run_root, payload)
    body = {"entries": normalized["entries"]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_progress_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def list_dashboard_progress(run_root: Path) -> list[dict[str, object]]:
    loaded = load_dashboard_progress(run_root)
    entries = loaded.get("entries", [])
    return list(entries) if isinstance(entries, list) else []


def record_dashboard_progress(
    run_root: Path,
    *,
    scope: str,
    plan_kind: str,
    checkpoint: str,
    action_label: str,
    run_dir: str | Path | None = None,
    sequence: str = "",
    campaign_dir: str | Path | None = None,
    action_status: str = "",
    action_id: str = "",
    note: str = "",
) -> dict[str, object]:
    loaded = load_dashboard_progress(run_root)
    entries = list(loaded.get("entries", [])) if isinstance(loaded.get("entries", []), list) else []
    entries.insert(
        0,
        _normalize_entry(
            run_root,
            {
                "id": uuid.uuid4().hex[:12],
                "created_at": _now_iso(),
                "scope": scope,
                "plan_kind": plan_kind,
                "run_dir": run_dir,
                "sequence": sequence,
                "campaign_dir": campaign_dir,
                "checkpoint": checkpoint,
                "action_label": action_label,
                "action_status": action_status,
                "action_id": action_id,
                "note": note,
            },
        ),
    )
    save_dashboard_progress(run_root, {"entries": entries})
    return load_dashboard_progress(run_root)
