from __future__ import annotations

import json
from pathlib import Path

from active_learning_thesis.dashboard_state_paths import (
    dashboard_state_path,
    dashboard_state_run_key,
    fallback_dashboard_state_path,
    preferred_dashboard_state_path,
)

DASHBOARD_PREFERENCES_FILENAME = "dashboard_preferences.json"
DASHBOARD_UI_MODES = ["Rich mode", "Stable mode"]
DEFAULT_DASHBOARD_UI_MODE = "Stable mode"
DASHBOARD_WORKFLOW_MODES = ["Guided thesis mode", "Expert mode"]
DEFAULT_DASHBOARD_WORKFLOW_MODE = "Guided thesis mode"
DASHBOARD_APPROVAL_MODES = ["Trusted actions", "Strict approvals"]
DEFAULT_DASHBOARD_APPROVAL_MODE = "Trusted actions"
DASHBOARD_REFRESH_MODES = ["Smart polling", "Manual refresh"]
DEFAULT_DASHBOARD_REFRESH_MODE = "Smart polling"


def _run_key(run_root: Path) -> str:
    return dashboard_state_run_key(run_root)


def _preferred_preferences_path(run_root: Path) -> Path:
    return preferred_dashboard_state_path(run_root, DASHBOARD_PREFERENCES_FILENAME)


def _fallback_preferences_path(run_root: Path) -> Path:
    return fallback_dashboard_state_path(run_root, DASHBOARD_PREFERENCES_FILENAME)


def dashboard_preferences_path(run_root: Path) -> Path:
    return dashboard_state_path(run_root, DASHBOARD_PREFERENCES_FILENAME)


def default_dashboard_preferences() -> dict[str, object]:
    return {
        "ui_mode": DEFAULT_DASHBOARD_UI_MODE,
        "workflow_mode": DEFAULT_DASHBOARD_WORKFLOW_MODE,
        "approval_mode": DEFAULT_DASHBOARD_APPROVAL_MODE,
        "refresh_mode": DEFAULT_DASHBOARD_REFRESH_MODE,
    }


def _normalize_preferences(payload: dict[str, object]) -> dict[str, object]:
    ui_mode = str(payload.get("ui_mode", DEFAULT_DASHBOARD_UI_MODE))
    if ui_mode not in DASHBOARD_UI_MODES:
        ui_mode = DEFAULT_DASHBOARD_UI_MODE
    workflow_mode = str(payload.get("workflow_mode", DEFAULT_DASHBOARD_WORKFLOW_MODE))
    if workflow_mode not in DASHBOARD_WORKFLOW_MODES:
        workflow_mode = DEFAULT_DASHBOARD_WORKFLOW_MODE
    approval_mode = str(payload.get("approval_mode", DEFAULT_DASHBOARD_APPROVAL_MODE))
    if approval_mode not in DASHBOARD_APPROVAL_MODES:
        approval_mode = DEFAULT_DASHBOARD_APPROVAL_MODE
    refresh_mode = str(payload.get("refresh_mode", DEFAULT_DASHBOARD_REFRESH_MODE))
    if refresh_mode not in DASHBOARD_REFRESH_MODES:
        refresh_mode = DEFAULT_DASHBOARD_REFRESH_MODE
    return {
        "ui_mode": ui_mode,
        "workflow_mode": workflow_mode,
        "approval_mode": approval_mode,
        "refresh_mode": refresh_mode,
    }


def load_dashboard_preferences(run_root: Path) -> dict[str, object]:
    path = dashboard_preferences_path(run_root)
    payload = default_dashboard_preferences()
    for candidate in [path, _preferred_preferences_path(run_root), _fallback_preferences_path(run_root)]:
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
                path = candidate
                break
        except Exception:
            continue
    normalized = _normalize_preferences(payload)
    normalized["path"] = str(path)
    normalized["exists"] = path.exists()
    return normalized


def save_dashboard_preferences(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_preferences_path(run_root)
    normalized = _normalize_preferences(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_preferences_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
        return fallback


def set_dashboard_ui_mode(run_root: Path, ui_mode: str) -> dict[str, object]:
    preferences = load_dashboard_preferences(run_root)
    preferences["ui_mode"] = ui_mode
    save_dashboard_preferences(run_root, preferences)
    return load_dashboard_preferences(run_root)
