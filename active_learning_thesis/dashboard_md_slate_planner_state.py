from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

MD_SLATE_PLANNERS_FILENAME = "dashboard_md_slate_planners.json"


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


def _preferred_md_slate_planners_path(run_root: Path) -> Path:
    user_root = Path.home() / ".active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return user_root / run_key / MD_SLATE_PLANNERS_FILENAME


def _fallback_md_slate_planners_path(run_root: Path) -> Path:
    temp_root = Path(tempfile.gettempdir()) / "active_learning_thesis" / "dashboard_state"
    run_key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return temp_root / run_key / MD_SLATE_PLANNERS_FILENAME


def dashboard_md_slate_planners_path(run_root: Path) -> Path:
    preferred = _preferred_md_slate_planners_path(run_root)
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = _fallback_md_slate_planners_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def default_dashboard_md_slate_planners() -> dict[str, object]:
    return {"planners": []}


def _normalize_planner_candidate(run_root: Path, entry: dict[str, object]) -> dict[str, object]:
    return {
        "sequence": str(entry.get("sequence", "")).strip(),
        "run_dir": _canonical_local_path(entry.get("run_dir"), base=run_root),
        "run_name": str(entry.get("run_name", "")).strip(),
        "source": str(entry.get("source", "")).strip(),
        "strategy": str(entry.get("strategy", "")).strip(),
        "priority_band": str(entry.get("priority_band", "")).strip(),
        "proposal_round": str(entry.get("proposal_round", "")).strip(),
        "source_batch_csv": _canonical_local_path(entry.get("source_batch_csv"), base=run_root),
        "source_batch_kind": str(entry.get("source_batch_kind", "")).strip(),
        "launch_ready": str(entry.get("launch_ready", "")).strip() or "no",
        "launch_blocker": str(entry.get("launch_blocker", "")).strip(),
        "next_action": str(entry.get("next_action", "")).strip(),
        "decision_title": str(entry.get("decision_title", "")).strip(),
    }


def _normalize_planner(run_root: Path, planner: dict[str, object]) -> dict[str, object]:
    candidates = planner.get("candidates", [])
    return {
        "planner_id": str(planner.get("planner_id", "")).strip(),
        "run_dir": _canonical_local_path(planner.get("run_dir"), base=run_root),
        "run_name": str(planner.get("run_name", "")).strip(),
        "name": str(planner.get("name", "")).strip(),
        "rationale": str(planner.get("rationale", "")).strip(),
        "status": str(planner.get("status", "")).strip() or "draft",
        "created_at": str(planner.get("created_at", "")).strip() or _now_iso(),
        "updated_at": str(planner.get("updated_at", "")).strip() or _now_iso(),
        "last_launched_slate_id": str(planner.get("last_launched_slate_id", "")).strip(),
        "last_launched_action_id": str(planner.get("last_launched_action_id", "")).strip(),
        "last_launched_at": str(planner.get("last_launched_at", "")).strip(),
        "candidates": [
            _normalize_planner_candidate(run_root, candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ] if isinstance(candidates, list) else [],
    }


def _normalize_payload(run_root: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_planners = payload.get("planners", [])
    planners_by_id: dict[str, dict[str, object]] = {}
    if isinstance(raw_planners, list):
        for raw_planner in raw_planners:
            if not isinstance(raw_planner, dict):
                continue
            planner = _normalize_planner(run_root, raw_planner)
            planner_id = str(planner.get("planner_id", "")).strip()
            if planner_id:
                planners_by_id[planner_id] = planner
    return {
        "planners": sorted(
            planners_by_id.values(),
            key=lambda item: (str(item.get("updated_at", "")), str(item.get("planner_id", ""))),
            reverse=True,
        )
    }


def load_dashboard_md_slate_planners(run_root: Path) -> dict[str, object]:
    path = dashboard_md_slate_planners_path(run_root)
    planners_by_id: dict[str, dict[str, object]] = {}
    exists = False
    for candidate in [path, _preferred_md_slate_planners_path(run_root), _fallback_md_slate_planners_path(run_root)]:
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
        for planner in normalized["planners"]:
            if isinstance(planner, dict):
                planners_by_id[str(planner.get("planner_id", ""))] = planner
    payload = {
        "planners": sorted(
            planners_by_id.values(),
            key=lambda item: (str(item.get("updated_at", "")), str(item.get("planner_id", ""))),
            reverse=True,
        )
    }
    payload["path"] = str(path)
    payload["exists"] = exists
    return payload


def save_dashboard_md_slate_planners(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_md_slate_planners_path(run_root)
    normalized = _normalize_payload(run_root, payload)
    body = {"planners": normalized["planners"]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_md_slate_planners_path(run_root)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return fallback


def list_dashboard_md_slate_planners(run_root: Path) -> list[dict[str, object]]:
    loaded = load_dashboard_md_slate_planners(run_root)
    planners = loaded.get("planners", [])
    return list(planners) if isinstance(planners, list) else []


def load_dashboard_md_slate_planner(run_root: Path, planner_id: str) -> dict[str, object]:
    for planner in list_dashboard_md_slate_planners(run_root):
        if str(planner.get("planner_id", "")) == planner_id:
            return planner
    raise FileNotFoundError(f"MD slate planner does not exist: {planner_id}")


def save_dashboard_md_slate_planner(run_root: Path, planner: dict[str, object]) -> dict[str, object]:
    payload = load_dashboard_md_slate_planners(run_root)
    planners = list(payload.get("planners", [])) if isinstance(payload.get("planners", []), list) else []
    normalized = _normalize_planner(run_root, planner)
    planner_id = str(normalized.get("planner_id", "")).strip()
    if not planner_id:
        raise ValueError("MD slate planner is missing planner_id.")
    existing = [item for item in planners if str(item.get("planner_id", "")) != planner_id]
    existing.append(normalized)
    save_dashboard_md_slate_planners(run_root, {"planners": existing})
    return load_dashboard_md_slate_planner(run_root, planner_id)
