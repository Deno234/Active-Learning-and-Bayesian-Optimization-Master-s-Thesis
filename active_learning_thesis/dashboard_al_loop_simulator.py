from __future__ import annotations

import csv
import json
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from active_learning_thesis.dashboard_md_slate import launch_md_slate_rehearsal_action
from active_learning_thesis.dashboard_md_slate_state import (
    dashboard_md_slates_path,
    load_dashboard_md_slate,
)

AL_LOOP_SIMULATIONS_FILENAME = "dashboard_al_loop_simulations.json"
AL_LOOP_SIMULATION_STAGES = [
    "candidate_snapshot",
    "md_rehearsed",
    "labels_injected",
    "ingest_simulated",
    "next_round_ready",
]
SAFETY_CONTRACT = {
    "touches_real_run_files": "no",
    "touches_model_files": "no",
    "touches_remote_clusters": "no",
    "assigns_real_cgmd_label": "no",
    "human_review_required": "yes",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _canonical_path(value: str | Path | None) -> str:
    if not value:
        return ""
    try:
        return str(Path(value).resolve())
    except Exception:
        return str(value)


def _safe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [{str(key): str(value or "") for key, value in row.items()} for row in reader]
    except Exception:
        return []


def dashboard_al_loop_simulations_path(run_root: Path) -> Path:
    return dashboard_md_slates_path(run_root).parent / AL_LOOP_SIMULATIONS_FILENAME


def _default_payload() -> dict[str, object]:
    return {"simulations": []}


def _event(stage: str, summary: str, *, detail: str = "") -> dict[str, str]:
    return {
        "time": _now_iso(),
        "stage": stage,
        "summary": summary,
        "detail": detail,
    }


def _normalize_candidate(candidate: dict[str, object], *, fallback_run_dir: str = "") -> dict[str, str]:
    sequence = str(candidate.get("sequence", "")).strip()
    run_dir = str(candidate.get("run_dir", "")).strip() or fallback_run_dir
    source_batch_csv = str(candidate.get("source_batch_csv", "")).strip()
    if source_batch_csv == "-":
        source_batch_csv = ""
    return {
        "sequence": sequence,
        "run": str(candidate.get("run", candidate.get("run_name", ""))).strip(),
        "run_dir": _canonical_path(run_dir),
        "source_batch_csv": _canonical_path(source_batch_csv) if source_batch_csv else "",
        "source": str(candidate.get("source", "")).strip(),
        "strategy": str(candidate.get("strategy", "")).strip(),
        "priority_band": str(candidate.get("priority_band", "")).strip(),
        "proposal_round": str(candidate.get("proposal_round", candidate.get("round_id", ""))).strip(),
        "candidate_status": str(candidate.get("candidate_status", "")).strip(),
        "launch_ready": str(candidate.get("launch_ready", "")).strip(),
    }


def _round_id_for_candidate(candidate: dict[str, str]) -> str:
    direct_round = str(candidate.get("proposal_round", "")).strip()
    if direct_round and direct_round != "-":
        return direct_round
    source_batch_csv = str(candidate.get("source_batch_csv", "")).strip()
    sequence = str(candidate.get("sequence", "")).strip()
    if source_batch_csv and sequence:
        for row in _safe_read_csv(Path(source_batch_csv)):
            if str(row.get("sequence", "")).strip() == sequence:
                round_id = str(row.get("round_id", "")).strip()
                if round_id:
                    return round_id
    return "1"


def _simulation_id(run_dir: Path, candidates: list[dict[str, str]]) -> str:
    return uuid.uuid4().hex[:12]


def _normalize_simulation(simulation: dict[str, object]) -> dict[str, object]:
    candidates = simulation.get("candidate_snapshot", [])
    events = simulation.get("events", [])
    labels = simulation.get("simulated_review_labels", [])
    ingest = simulation.get("simulated_ingest", {})
    retrain = simulation.get("simulated_retrain", {})
    return {
        "simulation_id": str(simulation.get("simulation_id", "")).strip(),
        "run_dir": _canonical_path(simulation.get("run_dir")),
        "run_name": str(simulation.get("run_name", "")).strip(),
        "created_at": str(simulation.get("created_at", "")).strip() or _now_iso(),
        "updated_at": str(simulation.get("updated_at", "")).strip() or _now_iso(),
        "status": str(simulation.get("status", "")).strip() or "candidate_snapshot",
        "stage": str(simulation.get("stage", "")).strip() or "candidate_snapshot",
        "md_rehearsal_slate_id": str(simulation.get("md_rehearsal_slate_id", "")).strip(),
        "md_rehearsal_action_id": str(simulation.get("md_rehearsal_action_id", "")).strip(),
        "candidate_snapshot": [
            {str(key): str(value) for key, value in candidate.items()}
            for candidate in candidates
            if isinstance(candidate, dict)
        ] if isinstance(candidates, list) else [],
        "simulated_review_labels": [
            {str(key): str(value) for key, value in row.items()}
            for row in labels
            if isinstance(row, dict)
        ] if isinstance(labels, list) else [],
        "simulated_ingest": deepcopy(ingest) if isinstance(ingest, dict) else {},
        "simulated_retrain": deepcopy(retrain) if isinstance(retrain, dict) else {},
        "events": [
            {str(key): str(value) for key, value in row.items()}
            for row in events
            if isinstance(row, dict)
        ] if isinstance(events, list) else [],
        "safety": {
            **SAFETY_CONTRACT,
            **({str(key): str(value) for key, value in simulation.get("safety", {}).items()} if isinstance(simulation.get("safety", {}), dict) else {}),
        },
    }


def load_al_loop_simulations(run_root: Path) -> dict[str, object]:
    path = dashboard_al_loop_simulations_path(run_root)
    if not path.exists():
        return {**_default_payload(), "path": str(path), "exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = _default_payload()
    if not isinstance(payload, dict):
        payload = _default_payload()
    simulations = payload.get("simulations", [])
    normalized = [
        _normalize_simulation(simulation)
        for simulation in simulations
        if isinstance(simulation, dict)
    ] if isinstance(simulations, list) else []
    normalized.sort(key=lambda row: (str(row.get("updated_at", "")), str(row.get("simulation_id", ""))), reverse=True)
    return {"simulations": normalized, "path": str(path), "exists": True}


def save_al_loop_simulations(run_root: Path, payload: dict[str, object]) -> Path:
    path = dashboard_al_loop_simulations_path(run_root)
    simulations = payload.get("simulations", [])
    normalized = [
        _normalize_simulation(simulation)
        for simulation in simulations
        if isinstance(simulation, dict)
    ] if isinstance(simulations, list) else []
    normalized.sort(key=lambda row: (str(row.get("updated_at", "")), str(row.get("simulation_id", ""))), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"simulations": normalized}, indent=2) + "\n", encoding="utf-8")
    return path


def list_al_loop_simulations(run_root: Path) -> list[dict[str, object]]:
    payload = load_al_loop_simulations(run_root)
    simulations = payload.get("simulations", [])
    return list(simulations) if isinstance(simulations, list) else []


def load_al_loop_simulation(run_root: Path, simulation_id: str) -> dict[str, object]:
    for simulation in list_al_loop_simulations(run_root):
        if str(simulation.get("simulation_id", "")) == simulation_id:
            return simulation
    raise FileNotFoundError(f"AL loop simulation does not exist: {simulation_id}")


def save_al_loop_simulation(run_root: Path, simulation: dict[str, object]) -> dict[str, object]:
    payload = load_al_loop_simulations(run_root)
    simulations = list(payload.get("simulations", [])) if isinstance(payload.get("simulations", []), list) else []
    normalized = _normalize_simulation(simulation)
    simulation_id = str(normalized.get("simulation_id", "")).strip()
    if not simulation_id:
        raise ValueError("AL loop simulation is missing simulation_id.")
    retained = [row for row in simulations if str(row.get("simulation_id", "")) != simulation_id]
    retained.append(normalized)
    save_al_loop_simulations(run_root, {"simulations": retained})
    return load_al_loop_simulation(run_root, simulation_id)


def _simulation_slate_summary(run_root: Path, slate_id: str) -> dict[str, object]:
    if not slate_id:
        return {}
    try:
        slate = load_dashboard_md_slate(run_root, slate_id)
    except Exception:
        return {}
    summary = slate.get("rehearsal_summary", {}) if isinstance(slate.get("rehearsal_summary", {}), dict) else {}
    return {
        "slate_id": str(slate.get("slate_id", "")),
        "status": str(slate.get("status", "")),
        "execution_mode": str(slate.get("execution_mode", "")),
        "peptide_count": str(summary.get("peptides", len(list(slate.get("peptides", []))))),
        "review_ready": str(summary.get("review_ready", 0)),
        "blocked": str(summary.get("blocked", 0)),
    }


def start_al_loop_simulation(
    *,
    run_root: Path,
    run_dir: Path,
    run_name: str,
    candidates: list[dict[str, object]],
    max_candidates: int | None = None,
) -> dict[str, object]:
    normalized_candidates = [
        _normalize_candidate(candidate, fallback_run_dir=str(run_dir))
        for candidate in candidates
    ]
    normalized_candidates = [row for row in normalized_candidates if str(row.get("sequence", "")).strip()]
    if max_candidates is not None:
        normalized_candidates = normalized_candidates[:max_candidates]
    if not normalized_candidates:
        raise ValueError("Choose at least one launch-ready candidate before starting an AL loop rehearsal.")
    run_dirs = {_canonical_path(row.get("run_dir")) for row in normalized_candidates}
    if len(run_dirs) != 1:
        raise ValueError("An AL loop rehearsal can only use candidates from one parent run at a time.")
    run_dir = Path(next(iter(run_dirs)))
    simulation_id = _simulation_id(run_dir, normalized_candidates)
    action = launch_md_slate_rehearsal_action(
        run_root=run_root,
        run_dir=run_dir,
        run_name=run_name,
        peptides=[
            {
                "sequence": str(row.get("sequence", "")),
                "run_dir": str(row.get("run_dir", "")),
                "source_batch_csv": str(row.get("source_batch_csv", "")),
                "source": str(row.get("source", "")),
                "strategy": str(row.get("strategy", "")),
                "priority_band": str(row.get("priority_band", "")),
            }
            for row in normalized_candidates
        ],
        operator_note=f"Dashboard AL loop simulation {simulation_id}",
    )
    action_metadata = action.get("metadata", {}) if isinstance(action.get("metadata", {}), dict) else {}
    slate_id = str(action_metadata.get("slate_id", "")).strip()
    simulation = {
        "simulation_id": simulation_id,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "md_rehearsed",
        "stage": "md_rehearsed",
        "md_rehearsal_slate_id": slate_id,
        "md_rehearsal_action_id": str(action.get("id", "")),
        "candidate_snapshot": normalized_candidates,
        "simulated_review_labels": [],
        "simulated_ingest": {},
        "simulated_retrain": {},
        "events": [
            _event(
                "candidate_snapshot",
                f"Snapshotted {len(normalized_candidates)} candidate(s) for a local AL loop rehearsal.",
                detail="Later edits to the candidate queue do not change this simulation.",
            ),
            _event(
                "md_rehearsed",
                "Ran the MD slate rehearsal through line_smoke -> production_smoke -> full.",
                detail=f"Slate {slate_id} was generated under dashboard-local state only.",
            ),
        ],
        "safety": dict(SAFETY_CONTRACT),
    }
    return save_al_loop_simulation(run_root, simulation)


def inject_simulated_review_labels(
    run_root: Path,
    simulation_id: str,
    *,
    default_label: str = "1",
) -> dict[str, object]:
    simulation = load_al_loop_simulation(run_root, simulation_id)
    candidates = list(simulation.get("candidate_snapshot", []))
    if not candidates:
        raise ValueError("This simulation has no candidate snapshot to label.")
    labels: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        sequence = str(candidate.get("sequence", "")).strip()
        label = default_label if default_label in {"0", "1"} else str(index % 2)
        labels.append(
            {
                "sequence": sequence,
                "round_id": _round_id_for_candidate({str(key): str(value) for key, value in candidate.items()}),
                "simulated_cgmd_label": label,
                "review_notes": "Simulation placeholder: replace with your real self-assembly call after inspecting MD outputs.",
                "source": "dashboard_al_loop_simulation",
            }
        )
    simulation["simulated_review_labels"] = labels
    simulation["status"] = "labels_injected"
    simulation["stage"] = "labels_injected"
    simulation["updated_at"] = _now_iso()
    events = list(simulation.get("events", []))
    events.append(
        _event(
            "labels_injected",
            f"Injected {len(labels)} simulated review label(s) into dashboard-local state.",
            detail="No md_review.csv and no cgmd_label field in a real campaign was edited.",
        )
    )
    simulation["events"] = events
    return save_al_loop_simulation(run_root, simulation)


def simulate_loop_ingest(run_root: Path, simulation_id: str) -> dict[str, object]:
    simulation = load_al_loop_simulation(run_root, simulation_id)
    labels = [row for row in list(simulation.get("simulated_review_labels", [])) if isinstance(row, dict)]
    if not labels:
        raise ValueError("Inject simulated review labels before simulating ingest.")
    run_dir = Path(str(simulation.get("run_dir", "")))
    round_ids = sorted({str(row.get("round_id", "")).strip() for row in labels if str(row.get("round_id", "")).strip()})
    round_id = round_ids[0] if len(round_ids) == 1 else "mixed"
    aggregate_name = f"round_{int(round_id):03d}_dashboard_feedback.csv" if round_id.isdigit() else "mixed_round_dashboard_feedback.csv"
    rows = [
        {
            "sequence": str(row.get("sequence", "")),
            "round_id": str(row.get("round_id", "")),
            "cgmd_label": str(row.get("simulated_cgmd_label", "")),
        }
        for row in labels
    ]
    would_write = [
        str(run_dir / "imports" / aggregate_name),
        *[
            str(run_dir / "md_campaigns" / f"{run_dir.name}_full_{str(row.get('sequence', '')).lower()}" / "cgmd_ingest.csv")
            for row in labels
        ],
    ]
    simulation["simulated_ingest"] = {
        "round_id": round_id,
        "rows": rows,
        "would_write": would_write,
        "would_run": f"python -m active_learning_thesis ingest-round --run-dir {run_dir} --import-csv {run_dir / 'imports' / aggregate_name}",
        "notes": "Simulation only: the aggregate import CSV and per-campaign cgmd_ingest.csv files were not written.",
    }
    simulation["status"] = "ingest_simulated"
    simulation["stage"] = "ingest_simulated"
    simulation["updated_at"] = _now_iso()
    events = list(simulation.get("events", []))
    events.append(
        _event(
            "ingest_simulated",
            f"Built a dry-run ingest plan for {len(rows)} reviewed candidate(s).",
            detail="The real path would write cgmd_ingest.csv/import CSV files, then run ingest-round.",
        )
    )
    simulation["events"] = events
    return save_al_loop_simulation(run_root, simulation)


def simulate_loop_retrain_and_propose(run_root: Path, simulation_id: str) -> dict[str, object]:
    simulation = load_al_loop_simulation(run_root, simulation_id)
    ingest = simulation.get("simulated_ingest", {}) if isinstance(simulation.get("simulated_ingest", {}), dict) else {}
    rows = list(ingest.get("rows", [])) if isinstance(ingest.get("rows", []), list) else []
    if not rows:
        raise ValueError("Simulate ingest before simulating retrain/propose.")
    run_dir = Path(str(simulation.get("run_dir", "")))
    round_text = str(ingest.get("round_id", "")).strip()
    try:
        next_round = int(round_text) + 1
    except Exception:
        next_round = 2
    next_batch = run_dir / "batches" / f"round_{next_round:03d}_batch.csv"
    simulation["simulated_retrain"] = {
        "would_run": [
            str(ingest.get("would_run", "")),
            f"python -m active_learning_thesis propose-round --run-dir {run_dir}",
        ],
        "would_update": [
            str(run_dir / "ledger.csv"),
            str(run_dir / "models"),
            str(run_dir / "metrics"),
            str(next_batch),
        ],
        "fake_next_round_id": str(next_round),
        "fake_next_batch_csv": str(next_batch),
        "proposed_count": str(max(len(rows), 1)),
        "notes": "Simulation only: no model retraining ran and no next batch was exported.",
    }
    simulation["status"] = "next_round_ready"
    simulation["stage"] = "next_round_ready"
    simulation["updated_at"] = _now_iso()
    events = list(simulation.get("events", []))
    events.append(
        _event(
            "next_round_ready",
            f"Simulated retrain/propose readiness for round {next_round}.",
            detail="The real path would ingest labels, retrain/evaluate, and export the next proposed batch.",
        )
    )
    simulation["events"] = events
    return save_al_loop_simulation(run_root, simulation)


def build_al_loop_simulation_rows(simulations: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for simulation in simulations:
        if not isinstance(simulation, dict):
            continue
        slate_summary = simulation.get("slate_summary", {}) if isinstance(simulation.get("slate_summary", {}), dict) else {}
        rows.append(
            {
                "simulation_id": str(simulation.get("simulation_id", "")),
                "run": Path(str(simulation.get("run_dir", ""))).name or str(simulation.get("run_name", "")),
                "stage": str(simulation.get("stage", "")),
                "status": str(simulation.get("status", "")),
                "candidates": str(len(list(simulation.get("candidate_snapshot", [])))),
                "simulated_labels": str(len(list(simulation.get("simulated_review_labels", [])))),
                "slate_id": str(simulation.get("md_rehearsal_slate_id", "")),
                "slate_status": str(slate_summary.get("status", "")),
                "updated_at": str(simulation.get("updated_at", "")),
            }
        )
    return rows


def hydrate_al_loop_simulation_summaries(run_root: Path, simulations: list[dict[str, object]]) -> list[dict[str, object]]:
    hydrated: list[dict[str, object]] = []
    for simulation in simulations:
        if not isinstance(simulation, dict):
            continue
        copy = dict(simulation)
        copy["slate_summary"] = _simulation_slate_summary(run_root, str(copy.get("md_rehearsal_slate_id", "")))
        hydrated.append(copy)
    return hydrated
