from __future__ import annotations

from pathlib import Path
from typing import Callable

from active_learning_thesis.dashboard_actions import list_dashboard_actions
from active_learning_thesis.dashboard_al_loop_simulator import (
    hydrate_al_loop_simulation_summaries,
    list_al_loop_simulations,
)
from active_learning_thesis.dashboard_curation import load_dashboard_curation
from active_learning_thesis.dashboard_decisions import list_dashboard_decisions
from active_learning_thesis.dashboard_md_recovery import build_md_slate_exception_rows
from active_learning_thesis.dashboard_md_slate import build_md_slate_rows
from active_learning_thesis.dashboard_md_slate_planner_state import list_dashboard_md_slate_planners
from active_learning_thesis.dashboard_profiles import load_cluster_profiles
from active_learning_thesis.dashboard_progress import list_dashboard_progress
from active_learning_thesis.dashboard_remote_state import (
    list_cluster_health_checks,
    list_cluster_snapshots,
    list_sync_records,
)


def discover_dashboard_run_summaries(
    run_root: Path,
    summarize_run: Callable[[Path, Path], dict[str, object]],
) -> list[dict[str, object]]:
    candidate_roots = [run_root]
    phase3_branches = run_root / "branches"
    if phase3_branches.exists():
        candidate_roots.append(phase3_branches)
    run_dirs = []
    for root in candidate_roots:
        if not root.exists():
            continue
        run_dirs.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        )
    return [
        summarize_run(run_root, run_dir)
        for run_dir in sorted(
            run_dirs,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    ]


def load_dashboard_state_records(run_root: Path, run_summaries: list[dict[str, object]]) -> dict[str, object]:
    curation = load_dashboard_curation(run_root)
    actions = list_dashboard_actions(run_root)
    md_slates = build_md_slate_rows(run_root, actions)
    md_slate_planners = list_dashboard_md_slate_planners(run_root)
    al_loop_simulations = hydrate_al_loop_simulation_summaries(run_root, list_al_loop_simulations(run_root))
    md_slate_exceptions = build_md_slate_exception_rows(md_slates, actions)
    exception_counts_by_slate: dict[str, int] = {}
    for row in md_slate_exceptions:
        slate_id = str(row.get("slate_id", "")).strip()
        if slate_id:
            exception_counts_by_slate[slate_id] = exception_counts_by_slate.get(slate_id, 0) + 1
    md_slates = [
        {
            **slate,
            "exception_count": exception_counts_by_slate.get(str(slate.get("slate_id", "")).strip(), 0),
        }
        for slate in md_slates
    ]
    return {
        "curation": curation,
        "actions": actions,
        "md_slates": md_slates,
        "md_slate_planners": md_slate_planners,
        "al_loop_simulations": al_loop_simulations,
        "md_slate_exceptions": md_slate_exceptions,
        "decisions": list_dashboard_decisions(run_root),
        "progress_events": list_dashboard_progress(run_root),
        "profiles": load_cluster_profiles(),
        "snapshots": list_cluster_snapshots(run_root),
        "sync_records": list_sync_records(run_root),
        "cluster_health": list_cluster_health_checks(run_root),
    }
