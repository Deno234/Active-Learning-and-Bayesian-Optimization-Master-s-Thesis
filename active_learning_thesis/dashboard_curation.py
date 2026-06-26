from __future__ import annotations

import json
from pathlib import Path

from active_learning_thesis.dashboard_state_paths import (
    dashboard_state_path,
    fallback_dashboard_state_path,
    preferred_dashboard_state_path,
)

CURATION_FILENAME = "dashboard_curation.json"


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


def _preferred_curation_path(run_root: Path) -> Path:
    return preferred_dashboard_state_path(run_root, CURATION_FILENAME)


def _fallback_curation_path(run_root: Path) -> Path:
    return fallback_dashboard_state_path(run_root, CURATION_FILENAME)


def dashboard_curation_path(run_root: Path) -> Path:
    return dashboard_state_path(run_root, CURATION_FILENAME)


def default_dashboard_curation() -> dict[str, object]:
    return {
        "pinned_runs": [],
        "hidden_runs": [],
        "labels": {},
        "ignored_reconciliation_ids": [],
    }


def _normalize_payload(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    pinned_runs = [
        _canonical_local_path(item, base=run_root)
        for item in payload.get("pinned_runs", [])
        if str(item).strip()
    ] if isinstance(payload.get("pinned_runs"), list) else []
    hidden_runs = [
        _canonical_local_path(item, base=run_root)
        for item in payload.get("hidden_runs", [])
        if str(item).strip()
    ] if isinstance(payload.get("hidden_runs"), list) else []
    labels_payload = payload.get("labels", {})
    labels: dict[str, str] = {}
    if isinstance(labels_payload, dict):
        for raw_path, label in labels_payload.items():
            normalized_path = _canonical_local_path(raw_path, base=run_root)
            if normalized_path and str(label).strip():
                labels[normalized_path] = str(label).strip()
    ignored_reconciliation_ids = [
        str(item).strip()
        for item in payload.get("ignored_reconciliation_ids", [])
        if str(item).strip()
    ] if isinstance(payload.get("ignored_reconciliation_ids"), list) else []
    return {
        "pinned_runs": sorted(dict.fromkeys(pinned_runs)),
        "hidden_runs": sorted(dict.fromkeys(hidden_runs)),
        "labels": labels,
        "ignored_reconciliation_ids": sorted(dict.fromkeys(ignored_reconciliation_ids)),
    }


def load_dashboard_curation(run_root: Path) -> dict[str, object]:
    path = dashboard_curation_path(run_root)
    payload = default_dashboard_curation()
    for candidate in [path, _preferred_curation_path(run_root), _fallback_curation_path(run_root)]:
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
    normalized = _normalize_payload(run_root, payload)
    normalized["path"] = str(path)
    normalized["exists"] = path.exists()
    return normalized


def save_dashboard_curation(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_curation_path(run_root)
    normalized = _normalize_payload(run_root, payload)
    body = {
        "pinned_runs": normalized["pinned_runs"],
        "hidden_runs": normalized["hidden_runs"],
        "labels": normalized["labels"],
        "ignored_reconciliation_ids": normalized["ignored_reconciliation_ids"],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_curation_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def pin_dashboard_run(run_root: Path, run_dir: str | Path) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    run_key = _canonical_local_path(run_dir, base=run_root)
    pinned = set(str(item) for item in curation.get("pinned_runs", []))
    pinned.add(run_key)
    curation["pinned_runs"] = sorted(pinned)
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)


def unpin_dashboard_run(run_root: Path, run_dir: str | Path) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    run_key = _canonical_local_path(run_dir, base=run_root)
    curation["pinned_runs"] = [item for item in curation.get("pinned_runs", []) if str(item) != run_key]
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)


def hide_dashboard_run(run_root: Path, run_dir: str | Path) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    run_key = _canonical_local_path(run_dir, base=run_root)
    hidden = set(str(item) for item in curation.get("hidden_runs", []))
    hidden.add(run_key)
    curation["hidden_runs"] = sorted(hidden)
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)


def show_dashboard_run(run_root: Path, run_dir: str | Path) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    run_key = _canonical_local_path(run_dir, base=run_root)
    curation["hidden_runs"] = [item for item in curation.get("hidden_runs", []) if str(item) != run_key]
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)


def set_dashboard_run_label(run_root: Path, run_dir: str | Path, label: str) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    run_key = _canonical_local_path(run_dir, base=run_root)
    labels = dict(curation.get("labels", {})) if isinstance(curation.get("labels"), dict) else {}
    if label.strip():
        labels[run_key] = label.strip()
    else:
        labels.pop(run_key, None)
    curation["labels"] = labels
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)


def ignore_dashboard_reconciliation_item(run_root: Path, reconciliation_id: str) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    ignored = set(str(item) for item in curation.get("ignored_reconciliation_ids", []) if str(item).strip())
    if str(reconciliation_id).strip():
        ignored.add(str(reconciliation_id).strip())
    curation["ignored_reconciliation_ids"] = sorted(ignored)
    save_dashboard_curation(run_root, curation)
    return load_dashboard_curation(run_root)
