from __future__ import annotations

import json
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - exercised in lean test environments
    pd = None

METRIC_FIELDS = ["f1", "pr_auc", "roc_auc", "balanced_accuracy"]


class SimpleFrame:
    def __init__(self, rows: list[dict[str, object]] | None = None):
        self.rows = rows or []

    @property
    def empty(self) -> bool:
        return not self.rows

    @property
    def columns(self) -> list[str]:
        if not self.rows:
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    def to_dict(self, orient: str = "records") -> list[dict[str, object]]:
        if orient != "records":
            raise ValueError("SimpleFrame only supports orient='records'")
        return list(self.rows)

    def assign(self, **kwargs) -> "SimpleFrame":
        assigned: list[dict[str, object]] = []
        for row in self.rows:
            updated = dict(row)
            updated.update(kwargs)
            assigned.append(updated)
        return SimpleFrame(assigned)



def _safe_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _frame(rows: list[dict[str, object]]):
    if pd is None:
        return SimpleFrame(rows)
    return pd.DataFrame(rows)



def run_metric_history(run_summary: dict[str, object]):
    rows: list[dict[str, object]] = []
    baseline = run_summary.get("baseline_metrics", {}) if isinstance(run_summary, dict) else {}
    final_metrics = run_summary.get("final_metrics", {}) if isinstance(run_summary, dict) else {}

    if baseline:
        row = {
            "run_name": run_summary.get("run_name", ""),
            "stage": "baseline",
            "round_id": baseline.get("round_id", 0),
            "labeled_count": baseline.get("labeled_count", ""),
        }
        for field in METRIC_FIELDS:
            row[field] = _safe_float(baseline.get(field))
        rows.append(row)

    if final_metrics:
        row = {
            "run_name": run_summary.get("run_name", ""),
            "stage": "final_holdout",
            "round_id": final_metrics.get("round_id", ""),
            "labeled_count": final_metrics.get("labeled_count", ""),
        }
        for field in METRIC_FIELDS:
            row[field] = _safe_float(final_metrics.get(field))
        rows.append(row)

    return _frame(rows)


def _replay_root_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "replay",
        run_dir.parent / "_dashboard_remote_state" / "downloads" / "supek" / run_dir.name / "replay",
    ]


def _first_existing_replay_root(run_dir: Path) -> Path | None:
    for replay_root in _replay_root_candidates(run_dir):
        if replay_root.exists() and any(path.is_dir() for path in replay_root.iterdir()):
            return replay_root
    return None



def replay_curve_frame(run_summary: dict[str, object]):
    rows: list[dict[str, object]] = []
    run_dir = Path(str(run_summary.get("run_dir", "")))
    replay_root = _first_existing_replay_root(run_dir)
    if replay_root is None:
        return _frame(rows)

    for strategy_dir in sorted(path for path in replay_root.iterdir() if path.is_dir()):
        summary_path = strategy_dir / "summary.json"
        if not summary_path.exists():
            continue
        points = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            rows.append(
                {
                    "run_name": run_summary.get("run_name", ""),
                    "strategy": strategy_dir.name,
                    "round_id": point.get("round_id", ""),
                    "labeled_count": point.get("labeled_count", ""),
                    "f1": _safe_float(point.get("f1")),
                    "pr_auc": _safe_float(point.get("pr_auc")),
                    "roc_auc": _safe_float(point.get("roc_auc")),
                    "balanced_accuracy": _safe_float(point.get("balanced_accuracy")),
                }
            )
    return _frame(rows)



def discovery_frame(run_summary: dict[str, object]):
    input_rows = run_summary.get("discovery", {}).get("rows", []) if isinstance(run_summary, dict) else []
    rows: list[dict[str, object]] = []
    for raw in input_rows:
        row = dict(raw)
        for column in [
            "exported_count",
            "unique_candidate_count",
            "top_batch_mean_pred_mean",
            "top_batch_mean_pred_std",
            "top_batch_mean_utility_score",
            "top_batch_mean_nearest_labeled_distance",
            "top_batch_mean_pairwise_distance",
        ]:
            if column in row:
                row[column] = _safe_float(row.get(column))
        rows.append(row)
    return _frame(rows)



def md_ladder_summary_frame(peptides: list[dict[str, object]]):
    counts: dict[tuple[str, str], int] = {}
    for ladder in peptides:
        for campaign in ladder.get("campaigns", []):
            key = (str(campaign.get("md_profile", "")), str(campaign.get("job_root_status", "")))
            counts[key] = counts.get(key, 0) + 1
    rows = [
        {"md_profile": profile, "job_root_status": status, "count": count}
        for (profile, status), count in sorted(counts.items())
    ]
    return _frame(rows)



def action_timeline_frame(actions: list[dict[str, object]]):
    rows: list[dict[str, object]] = []
    for action in actions:
        rows.append(
            {
                "id": action.get("id", ""),
                "title": action.get("title", ""),
                "scope": action.get("scope", "local"),
                "cluster": action.get("cluster", ""),
                "status": action.get("status", ""),
                "created_at": action.get("created_at", ""),
                "started_at": action.get("started_at", ""),
                "finished_at": action.get("finished_at", ""),
                "remote_job_id": action.get("remote_job_id", ""),
            }
        )
    return _frame(rows)



def remote_job_summary_frame(snapshots: list[dict[str, object]]):
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        cluster = snapshot.get("cluster", "")
        summary = snapshot.get("summary", {}) if isinstance(snapshot, dict) else {}
        jobs = snapshot.get("jobs", []) if isinstance(snapshot, dict) else []
        rows.append(
            {
                "cluster": cluster,
                "collected_at": snapshot.get("collected_at", ""),
                "job_count": len(jobs) if isinstance(jobs, list) else 0,
                "pending": summary.get("pending", 0),
                "running": summary.get("running", 0),
                "failed": summary.get("failed", 0),
                "held": summary.get("held", 0),
            }
        )
    return _frame(rows)



def flatten_remote_jobs(snapshots: list[dict[str, object]]):
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        cluster = snapshot.get("cluster", "")
        collected_at = snapshot.get("collected_at", "")
        jobs = snapshot.get("jobs", []) if isinstance(snapshot, dict) else []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            row = {"cluster": cluster, "collected_at": collected_at}
            row.update(job)
            rows.append(row)
    return _frame(rows)
