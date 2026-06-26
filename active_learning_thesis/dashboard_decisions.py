from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence

from active_learning_thesis.dashboard_state_paths import (
    dashboard_state_path,
    fallback_dashboard_state_path,
    preferred_dashboard_state_path,
)

DECISIONS_FILENAME = "dashboard_decisions.json"


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


def _preferred_decisions_path(run_root: Path) -> Path:
    return preferred_dashboard_state_path(run_root, DECISIONS_FILENAME)


def _fallback_decisions_path(run_root: Path) -> Path:
    return fallback_dashboard_state_path(run_root, DECISIONS_FILENAME)


def dashboard_decisions_path(run_root: Path) -> Path:
    return dashboard_state_path(run_root, DECISIONS_FILENAME)


def default_dashboard_decisions() -> dict[str, object]:
    return {
        "entries": [],
    }


def _normalize_entry(run_root: Path, entry: dict[str, object]) -> dict[str, object]:
    created_at = str(entry.get("created_at", "")).strip() or _now_iso()
    updated_at = str(entry.get("updated_at", "")).strip() or created_at
    decision_id = str(entry.get("id", "")).strip() or uuid.uuid4().hex[:12]
    scope = str(entry.get("scope", "")).strip() or "global"
    return {
        "id": decision_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "scope": scope,
        "run_dir": _canonical_local_path(entry.get("run_dir"), base=run_root),
        "run_name": str(entry.get("run_name", "")).strip(),
        "sequence": str(entry.get("sequence", "")).strip(),
        "campaign_dir": _canonical_local_path(entry.get("campaign_dir"), base=run_root),
        "decision_type": str(entry.get("decision_type", "")).strip(),
        "title": str(entry.get("title", "")).strip(),
        "rationale": str(entry.get("rationale", "")).strip(),
        "evidence": str(entry.get("evidence", "")).strip(),
        "next_step": str(entry.get("next_step", "")).strip(),
    }


def _normalize_payload(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_entries = payload.get("entries", [])
    entries_by_id: dict[str, dict[str, object]] = {}
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = _normalize_entry(run_root, raw_entry)
            existing = entries_by_id.get(str(entry["id"]))
            if existing is None or str(entry.get("updated_at", "")) >= str(existing.get("updated_at", "")):
                entries_by_id[str(entry["id"])] = entry
    entries = sorted(
        entries_by_id.values(),
        key=lambda item: (str(item.get("created_at", "")), str(item.get("id", ""))),
        reverse=True,
    )
    return {"entries": entries}


def load_dashboard_decisions(run_root: Path) -> dict[str, object]:
    path = dashboard_decisions_path(run_root)
    entries_by_id: dict[str, dict[str, object]] = {}
    exists = False
    for candidate in [path, _preferred_decisions_path(run_root), _fallback_decisions_path(run_root)]:
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
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", ""))
            existing = entries_by_id.get(entry_id)
            if existing is None or str(entry.get("updated_at", "")) >= str(existing.get("updated_at", "")):
                entries_by_id[entry_id] = entry
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


def save_dashboard_decisions(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_decisions_path(run_root)
    normalized = _normalize_payload(run_root, payload)
    body = {"entries": normalized["entries"]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_decisions_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def list_dashboard_decisions(run_root: Path) -> list[dict[str, object]]:
    loaded = load_dashboard_decisions(run_root)
    entries = loaded.get("entries", [])
    return list(entries) if isinstance(entries, list) else []


def add_dashboard_decision(
    run_root: Path,
    *,
    scope: str,
    decision_type: str,
    title: str,
    rationale: str,
    run_dir: str | Path | None = None,
    run_name: str = "",
    sequence: str = "",
    campaign_dir: str | Path | None = None,
    evidence: str = "",
    next_step: str = "",
) -> dict[str, object]:
    loaded = load_dashboard_decisions(run_root)
    entries = list(loaded.get("entries", [])) if isinstance(loaded.get("entries", []), list) else []
    now = _now_iso()
    entries.insert(
        0,
        _normalize_entry(
            run_root,
            {
                "id": uuid.uuid4().hex[:12],
                "created_at": now,
                "updated_at": now,
                "scope": scope,
                "run_dir": run_dir,
                "run_name": run_name,
                "sequence": sequence,
                "campaign_dir": campaign_dir,
                "decision_type": decision_type,
                "title": title,
                "rationale": rationale,
                "evidence": evidence,
                "next_step": next_step,
            },
        ),
    )
    save_dashboard_decisions(run_root, {"entries": entries})
    return load_dashboard_decisions(run_root)


def add_dashboard_decisions(
    run_root: Path,
    *,
    entries: Sequence[dict[str, object]],
) -> dict[str, object]:
    loaded = load_dashboard_decisions(run_root)
    existing_entries = list(loaded.get("entries", [])) if isinstance(loaded.get("entries", []), list) else []
    now = _now_iso()
    inserted: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        inserted.append(
            _normalize_entry(
                run_root,
                {
                    "id": entry.get("id", "") or uuid.uuid4().hex[:12],
                    "created_at": str(entry.get("created_at", "")).strip() or now,
                    "updated_at": str(entry.get("updated_at", "")).strip() or now,
                    "scope": entry.get("scope", "global"),
                    "run_dir": entry.get("run_dir"),
                    "run_name": entry.get("run_name", ""),
                    "sequence": entry.get("sequence", ""),
                    "campaign_dir": entry.get("campaign_dir"),
                    "decision_type": entry.get("decision_type", ""),
                    "title": entry.get("title", ""),
                    "rationale": entry.get("rationale", ""),
                    "evidence": entry.get("evidence", ""),
                    "next_step": entry.get("next_step", ""),
                },
            )
        )
    if not inserted:
        return loaded
    save_dashboard_decisions(run_root, {"entries": [*inserted, *existing_entries]})
    return load_dashboard_decisions(run_root)
