from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path


DASHBOARD_STATE_DIRNAME = "dashboard_state"


def dashboard_state_run_key(run_root: Path) -> str:
    """Stable per-workspace key used by existing dashboard state files."""

    return hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]


def preferred_dashboard_state_dir(run_root: Path) -> Path:
    return Path.home() / ".active_learning_thesis" / DASHBOARD_STATE_DIRNAME / dashboard_state_run_key(run_root)


def fallback_dashboard_state_dir(run_root: Path) -> Path:
    return (
        Path(tempfile.gettempdir())
        / "active_learning_thesis"
        / DASHBOARD_STATE_DIRNAME
        / dashboard_state_run_key(run_root)
    )


def preferred_dashboard_state_path(run_root: Path, filename: str) -> Path:
    return preferred_dashboard_state_dir(run_root) / filename


def fallback_dashboard_state_path(run_root: Path, filename: str) -> Path:
    return fallback_dashboard_state_dir(run_root) / filename


def dashboard_state_path(run_root: Path, filename: str) -> Path:
    preferred = preferred_dashboard_state_path(run_root, filename)
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = fallback_dashboard_state_path(run_root, filename)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback

