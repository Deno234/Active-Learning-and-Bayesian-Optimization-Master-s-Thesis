from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

REMOTE_STATE_DIRNAME = "_dashboard_remote_state"
SNAPSHOTS_DIRNAME = "snapshots"
DOWNLOADS_DIRNAME = "downloads"
UPLOADS_DIRNAME = "uploads"
HEALTH_DIRNAME = "health"
SYNC_STATUS_FILENAME = "sync_status.json"


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


def _preferred_remote_state_root_path(run_root: Path) -> Path:
    return run_root / REMOTE_STATE_DIRNAME


def _fallback_remote_state_root_path(run_root: Path) -> Path:
    user_root = Path.home() / ".active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return user_root / run_key / REMOTE_STATE_DIRNAME


def _fallback_remote_state_root(run_root: Path) -> Path:
    path = _fallback_remote_state_root_path(run_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_roots(run_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in [remote_state_root(run_root), _preferred_remote_state_root_path(run_root), _fallback_remote_state_root_path(run_root)]:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            roots.append(candidate)
    return roots


def _resolved_remote_state_root(run_root: Path) -> Path:
    preferred = _preferred_remote_state_root_path(run_root)
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        (preferred / SNAPSHOTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (preferred / DOWNLOADS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (preferred / UPLOADS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (preferred / HEALTH_DIRNAME).mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        return _fallback_remote_state_root(run_root)



def remote_state_root(run_root: Path) -> Path:
    return _resolved_remote_state_root(run_root)



def _state_subdir(run_root: Path, dirname: str, cluster: str = "") -> Path:
    preferred_root = remote_state_root(run_root)
    preferred = preferred_root / dirname if dirname else preferred_root
    if cluster:
        preferred = preferred / cluster
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback_root = _fallback_remote_state_root(run_root)
        fallback = fallback_root / dirname if dirname else fallback_root
        if cluster:
            fallback = fallback / cluster
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback



def snapshots_root(run_root: Path) -> Path:
    return _state_subdir(run_root, SNAPSHOTS_DIRNAME)



def downloads_root(run_root: Path, cluster: str = "") -> Path:
    return _state_subdir(run_root, DOWNLOADS_DIRNAME, cluster)



def uploads_root(run_root: Path, cluster: str = "") -> Path:
    return _state_subdir(run_root, UPLOADS_DIRNAME, cluster)



def health_root(run_root: Path) -> Path:
    return _state_subdir(run_root, HEALTH_DIRNAME)



def sync_status_path(run_root: Path) -> Path:
    return _state_subdir(run_root, "") / SYNC_STATUS_FILENAME



def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    last_error: Exception | None = None
    for _ in range(8):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error



def _fallback_equivalent_path(run_root: Path, path: Path) -> Path:
    preferred_root = _preferred_remote_state_root_path(run_root).resolve()
    fallback_root = _fallback_remote_state_root(run_root).resolve()
    try:
        relative = path.resolve().relative_to(preferred_root)
    except Exception:
        return path
    return fallback_root / relative



def _write_json_with_fallback(run_root: Path, path: Path, payload: dict[str, object]) -> Path:
    try:
        _atomic_write_json(path, payload)
        return path
    except PermissionError:
        fallback_path = _fallback_equivalent_path(run_root, path)
        if fallback_path == path:
            raise
        _atomic_write_json(fallback_path, payload)
        return fallback_path



def _normalize_sync_record(
    cluster: str,
    record: dict[str, object],
    *,
    run_root: Path,
    fallback_target_key: str = "",
) -> dict[str, object]:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
    normalized = {
        **record,
        "cluster": cluster or str(record.get("cluster", "")),
        "target_key": _canonical_local_path(str(record.get("target_key", "") or fallback_target_key), base=run_root),
        "related_run": _canonical_local_path(str(record.get("related_run", "")), base=run_root),
        "related_campaign": _canonical_local_path(str(record.get("related_campaign", "")), base=run_root),
        "related_sequence": str(record.get("related_sequence", "")),
        "remote_path": str(record.get("remote_path", "")),
        "remote_job_id": str(record.get("remote_job_id", "")),
        "metadata": metadata,
        "updated_at": str(record.get("updated_at", "")),
    }
    return normalized



def _sync_record_identity(record_key: str, record: dict[str, object]) -> str:
    cluster = str(record.get("cluster", ""))
    target_key = str(record.get("target_key", "")) or str(record.get("related_campaign", "")) or str(record.get("related_run", ""))
    if target_key:
        return sync_record_key(cluster, target_key)
    return record_key


def _merge_sync_records(existing: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    existing_updated = str(existing.get("updated_at", ""))
    candidate_updated = str(candidate.get("updated_at", ""))
    newer, older = (candidate, existing) if candidate_updated >= existing_updated else (existing, candidate)
    newer_metadata = newer.get("metadata", {}) if isinstance(newer.get("metadata"), dict) else {}
    older_metadata = older.get("metadata", {}) if isinstance(older.get("metadata"), dict) else {}
    merged = dict(newer)
    merged["metadata"] = {**older_metadata, **newer_metadata}
    for field in ["target_key", "related_run", "related_campaign", "related_sequence", "remote_path", "remote_job_id"]:
        if not str(merged.get(field, "")):
            merged[field] = older.get(field, "")
    return merged



def _load_sync_status_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"updated_at": "", "records": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"updated_at": "", "records": {}}
    records = payload.get("records", {})
    if not isinstance(records, dict):
        records = {}
    normalized_records: dict[str, dict[str, object]] = {}
    for record_key, record in records.items():
        if not isinstance(record, dict):
            continue
        cluster = str(record.get("cluster", "") or str(record_key).split("::", 1)[0])
        fallback_target_key = str(record_key).split("::", 1)[1] if "::" in str(record_key) else ""
        normalized = _normalize_sync_record(cluster, record, run_root=path.parent.parent, fallback_target_key=fallback_target_key)
        identity = _sync_record_identity(str(record_key), normalized)
        existing = normalized_records.get(identity)
        if existing is None:
            normalized_records[identity] = normalized
        else:
            normalized_records[identity] = _merge_sync_records(existing, normalized)
    return {
        "updated_at": str(payload.get("updated_at", "")),
        "records": normalized_records,
    }



def load_sync_status(run_root: Path) -> dict[str, object]:
    merged = {"updated_at": "", "records": {}}
    merged_records: dict[str, dict[str, object]] = {}
    for root in _state_roots(run_root):
        payload = _load_sync_status_file(root / SYNC_STATUS_FILENAME)
        merged["updated_at"] = max(str(merged.get("updated_at", "")), str(payload.get("updated_at", "")))
        records = payload.get("records", {}) if isinstance(payload, dict) else {}
        if not isinstance(records, dict):
            continue
        for record_key, record in records.items():
            if not isinstance(record, dict):
                continue
            identity = _sync_record_identity(record_key, record)
            existing = merged_records.get(identity)
            if existing is None:
                merged_records[identity] = record
            else:
                merged_records[identity] = _merge_sync_records(existing, record)
    merged["records"] = merged_records
    return merged



def save_sync_status(run_root: Path, payload: dict[str, object]) -> Path:
    path = sync_status_path(run_root)
    return _write_json_with_fallback(run_root, path, payload)



def sync_record_key(cluster: str, target_key: str) -> str:
    return f"{cluster}::{target_key}"



def update_sync_status(
    run_root: Path,
    *,
    cluster: str,
    target_key: str,
    status: str,
    related_run: str = "",
    related_campaign: str = "",
    related_sequence: str = "",
    remote_path: str = "",
    remote_job_id: str | None = "",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    normalized_target_key = _canonical_local_path(target_key, base=run_root)
    normalized_related_run = _canonical_local_path(related_run, base=run_root)
    normalized_related_campaign = _canonical_local_path(related_campaign, base=run_root)
    payload = load_sync_status(run_root)
    records = payload.setdefault("records", {})
    record_key = sync_record_key(cluster, normalized_target_key)
    existing = records.get(record_key, {}) if isinstance(records, dict) else {}
    if not isinstance(existing, dict):
        existing = {}
    existing_metadata = existing.get("metadata", {}) if isinstance(existing.get("metadata"), dict) else {}
    merged_metadata = dict(existing_metadata)
    if isinstance(metadata, dict):
        merged_metadata.update(metadata)
    updated = {
        **existing,
        "cluster": cluster,
        "target_key": normalized_target_key,
        "status": status,
        "related_run": normalized_related_run or str(existing.get("related_run", "")),
        "related_campaign": normalized_related_campaign or str(existing.get("related_campaign", "")),
        "related_sequence": related_sequence or str(existing.get("related_sequence", "")),
        "remote_path": remote_path or str(existing.get("remote_path", "")),
        "remote_job_id": (
            ""
            if remote_job_id is None
            else remote_job_id or str(existing.get("remote_job_id", ""))
        ),
        "metadata": merged_metadata,
        "updated_at": _now_iso(),
    }
    records[record_key] = updated
    payload["updated_at"] = _now_iso()
    save_sync_status(run_root, payload)
    return updated



def list_sync_records(run_root: Path) -> list[dict[str, object]]:
    payload = load_sync_status(run_root)
    records = payload.get("records", {}) if isinstance(payload, dict) else {}
    if not isinstance(records, dict):
        return []
    return sorted(records.values(), key=lambda item: str(item.get("updated_at", "")), reverse=True)



def snapshot_path(run_root: Path, cluster: str) -> Path:
    return snapshots_root(run_root) / f"{cluster}_jobs.json"



def save_cluster_snapshot(run_root: Path, cluster: str, payload: dict[str, object]) -> Path:
    body = {
        "cluster": cluster,
        "collected_at": payload.get("collected_at", _now_iso()),
        "jobs": payload.get("jobs", []),
        "summary": payload.get("summary", {}),
        "raw_excerpt": payload.get("raw_excerpt", ""),
    }
    path = snapshot_path(run_root, cluster)
    return _write_json_with_fallback(run_root, path, body)



def _load_cluster_snapshot_file(path: Path, cluster: str) -> dict[str, object]:
    if not path.exists():
        return {"cluster": cluster, "collected_at": "", "jobs": [], "summary": {}, "raw_excerpt": ""}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"cluster": cluster, "collected_at": "", "jobs": [], "summary": {}, "raw_excerpt": ""}
    return {
        "cluster": cluster,
        "collected_at": str(payload.get("collected_at", "")),
        "jobs": payload.get("jobs", []) if isinstance(payload.get("jobs", []), list) else [],
        "summary": payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {},
        "raw_excerpt": str(payload.get("raw_excerpt", "")),
    }



def load_cluster_snapshot(run_root: Path, cluster: str) -> dict[str, object]:
    chosen = {"cluster": cluster, "collected_at": "", "jobs": [], "summary": {}, "raw_excerpt": ""}
    for root in _state_roots(run_root):
        candidate = _load_cluster_snapshot_file(root / SNAPSHOTS_DIRNAME / f"{cluster}_jobs.json", cluster)
        if str(candidate.get("collected_at", "")) >= str(chosen.get("collected_at", "")):
            chosen = candidate
    return chosen



def list_cluster_snapshots(run_root: Path) -> list[dict[str, object]]:
    clusters: set[str] = set()
    for root in _state_roots(run_root):
        snapshot_dir = root / SNAPSHOTS_DIRNAME
        if not snapshot_dir.exists():
            continue
        for path in snapshot_dir.glob("*_jobs.json"):
            clusters.add(path.stem.replace("_jobs", ""))
    return [load_cluster_snapshot(run_root, cluster) for cluster in sorted(clusters)]




def cluster_health_path(run_root: Path, cluster: str) -> Path:
    return health_root(run_root) / f"{cluster}_health.json"



def save_cluster_health(run_root: Path, cluster: str, payload: dict[str, object]) -> Path:
    body = {
        "cluster": cluster,
        "checked_at": str(payload.get("checked_at", _now_iso())),
        "overall_status": str(payload.get("overall_status", "unknown")),
        "local_auth_status": payload.get("local_auth_status", {}),
        "remote_status": payload.get("remote_status", {}),
        "summary": str(payload.get("summary", "")),
        "hint": str(payload.get("hint", "")),
        "details": payload.get("details", []) if isinstance(payload.get("details", []), list) else [],
    }
    path = cluster_health_path(run_root, cluster)
    return _write_json_with_fallback(run_root, path, body)



def _load_cluster_health_file(path: Path, cluster: str) -> dict[str, object]:
    if not path.exists():
        return {
            "cluster": cluster,
            "checked_at": "",
            "overall_status": "unknown",
            "local_auth_status": {},
            "remote_status": {},
            "summary": "",
            "hint": "",
            "details": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {
            "cluster": cluster,
            "checked_at": "",
            "overall_status": "unknown",
            "local_auth_status": {},
            "remote_status": {},
            "summary": "",
            "hint": "",
            "details": [],
        }
    return {
        "cluster": cluster,
        "checked_at": str(payload.get("checked_at", "")),
        "overall_status": str(payload.get("overall_status", "unknown")),
        "local_auth_status": payload.get("local_auth_status", {}) if isinstance(payload.get("local_auth_status", {}), dict) else {},
        "remote_status": payload.get("remote_status", {}) if isinstance(payload.get("remote_status", {}), dict) else {},
        "summary": str(payload.get("summary", "")),
        "hint": str(payload.get("hint", "")),
        "details": payload.get("details", []) if isinstance(payload.get("details", []), list) else [],
    }



def load_cluster_health(run_root: Path, cluster: str) -> dict[str, object]:
    chosen = {
        "cluster": cluster,
        "checked_at": "",
        "overall_status": "unknown",
        "local_auth_status": {},
        "remote_status": {},
        "summary": "",
        "hint": "",
        "details": [],
    }
    for root in _state_roots(run_root):
        candidate = _load_cluster_health_file(root / HEALTH_DIRNAME / f"{cluster}_health.json", cluster)
        if str(candidate.get("checked_at", "")) >= str(chosen.get("checked_at", "")):
            chosen = candidate
    return chosen



def list_cluster_health_checks(run_root: Path) -> list[dict[str, object]]:
    clusters: set[str] = set()
    for root in _state_roots(run_root):
        health_dir = root / HEALTH_DIRNAME
        if not health_dir.exists():
            continue
        for file_path in health_dir.glob("*_health.json"):
            clusters.add(file_path.stem.replace("_health", ""))
    return [load_cluster_health(run_root, cluster) for cluster in sorted(clusters)]
