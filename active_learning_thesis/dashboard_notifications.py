from __future__ import annotations

import json
from pathlib import Path

from active_learning_thesis.dashboard_state_paths import (
    dashboard_state_path,
    fallback_dashboard_state_path,
    preferred_dashboard_state_path,
)

NOTIFICATIONS_FILENAME = "dashboard_notifications.json"


def _preferred_notifications_path(run_root: Path) -> Path:
    return preferred_dashboard_state_path(run_root, NOTIFICATIONS_FILENAME)


def _fallback_notifications_path(run_root: Path) -> Path:
    return fallback_dashboard_state_path(run_root, NOTIFICATIONS_FILENAME)


def dashboard_notifications_path(run_root: Path) -> Path:
    return dashboard_state_path(run_root, NOTIFICATIONS_FILENAME)


def default_dashboard_notifications() -> dict[str, object]:
    return {
        "delivered": {},
        "acknowledged_ids": [],
        "cluster_status": {},
    }


def _normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    delivered_payload = payload.get("delivered", {})
    acknowledged_payload = payload.get("acknowledged_ids", [])
    cluster_status_payload = payload.get("cluster_status", {})
    delivered = {
        str(key): str(value).strip()
        for key, value in delivered_payload.items()
        if str(key).strip() and str(value).strip()
    } if isinstance(delivered_payload, dict) else {}
    acknowledged_ids = sorted(
        dict.fromkeys(str(item).strip() for item in acknowledged_payload if str(item).strip())
    ) if isinstance(acknowledged_payload, list) else []
    cluster_status = {
        str(key): str(value).strip()
        for key, value in cluster_status_payload.items()
        if str(key).strip()
    } if isinstance(cluster_status_payload, dict) else {}
    return {
        "delivered": delivered,
        "acknowledged_ids": acknowledged_ids,
        "cluster_status": cluster_status,
    }


def load_dashboard_notifications(run_root: Path) -> dict[str, object]:
    path = dashboard_notifications_path(run_root)
    payload = default_dashboard_notifications()
    exists = False
    for candidate in [path, _preferred_notifications_path(run_root), _fallback_notifications_path(run_root)]:
        if not candidate.exists():
            continue
        exists = True
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict):
            payload = loaded
            path = candidate
            break
    normalized = _normalize_payload(payload)
    normalized["path"] = str(path)
    normalized["exists"] = exists
    return normalized


def save_dashboard_notifications(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_notifications_path(run_root)
    normalized = _normalize_payload(payload)
    body = {
        "delivered": normalized["delivered"],
        "acknowledged_ids": normalized["acknowledged_ids"],
        "cluster_status": normalized["cluster_status"],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_notifications_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def acknowledge_dashboard_notifications(run_root: Path, notification_ids: list[str]) -> dict[str, object]:
    payload = load_dashboard_notifications(run_root)
    acknowledged = set(str(item) for item in payload.get("acknowledged_ids", []) if str(item).strip())
    delivered = dict(payload.get("delivered", {})) if isinstance(payload.get("delivered", {}), dict) else {}
    for notification_id in notification_ids:
        notification_key = str(notification_id).strip()
        if notification_key and notification_key in delivered:
            acknowledged.add(notification_key)
    payload["acknowledged_ids"] = sorted(acknowledged)
    save_dashboard_notifications(run_root, payload)
    return load_dashboard_notifications(run_root)
