from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import median

from active_learning_thesis.config import RunConfig
from active_learning_thesis.workflow import init_run, run_replay


LOWER_IS_BETTER_METRICS = {
    "brier_score",
    "log_loss",
    "ece_10",
    "mce_10",
}

DEFAULT_STUDY_OUTPUT_DIRNAME = "_study_evidence"
DEFAULT_STUDY_ROOT_DIRNAME = "_studies"
DEFAULT_STUDY_COMPARISON_DIRNAME = "_comparisons"
DEFAULT_STUDY_SEED_START = 20260317
DEFAULT_STUDY_SEED_STEP = 1009
STUDY_CONFIG_FIELDS = [
    "random_seed",
    "batch_size",
    "max_rounds",
    "candidate_pool_min",
    "replay_seed_size",
    "epochs",
    "real_strategy",
    "replay_strategies",
    "ensemble_size",
    "use_calibrated_acquisition",
    "generator_objective_mode",
    "use_similarity_penalty",
    "use_length_penalty",
    "binary_threshold_strategy",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "study"


def _safe_float(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _safe_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {None, ""}:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _sem(values: list[float]) -> float | None:
    std = _sample_std(values)
    if std is None:
        return None
    return float(std / math.sqrt(len(values))) if values else None


def _ci95(values: list[float]) -> tuple[float | None, float | None]:
    center = _mean(values)
    sem = _sem(values)
    if center is None or sem is None:
        return None, None
    margin = 1.96 * sem
    return center - margin, center + margin


def _higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER_METRICS


def _better(value: float, reference: float, *, higher_is_better: bool) -> bool:
    return value > reference if higher_is_better else value < reference


def _advantage(value: float, reference: float, *, higher_is_better: bool) -> float:
    return value - reference if higher_is_better else reference - value


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _study_paths(run_root: Path, study_name: str) -> dict[str, Path]:
    study_dir = run_root / DEFAULT_STUDY_ROOT_DIRNAME / _slug(study_name)
    return {
        "study_dir": study_dir,
        "manifest": study_dir / "study_manifest.json",
        "evidence": study_dir / "evidence",
    }


def _path_key(value: object) -> str:
    if value in {None, ""}:
        return ""
    try:
        return str(Path(str(value)).resolve()).replace("\\", "/").lower()
    except OSError:
        return str(value).replace("\\", "/").lower()


def _seed_values(seed_count: int, seed_start: int, seed_step: int) -> list[int]:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_step <= 0:
        raise ValueError("seed_step must be positive")
    return [seed_start + index * seed_step for index in range(seed_count)]


def _study_run_name(study_name: str, index: int, seed: int) -> str:
    return f"{_slug(study_name)}_seed_{index:02d}_{seed}"


def _study_config(
    *,
    study_name: str,
    run_root: Path,
    seed: int,
    index: int,
    epochs: int,
    max_rounds: int,
    batch_size: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    replay_strategies: list[str] | None,
    ensemble_size: int,
    train_family_for_init: bool,
    use_calibrated_acquisition: bool,
    generator_objective_mode: str,
    use_similarity_penalty: bool,
    use_length_penalty: bool,
    binary_threshold_strategy: str,
) -> RunConfig:
    kwargs = {
        "run_name": _study_run_name(study_name, index, seed),
        "output_root": str(run_root),
        "random_seed": seed,
        "epochs": epochs,
        "max_rounds": max_rounds,
        "batch_size": batch_size,
        "candidate_pool_min": candidate_pool_min,
        "replay_seed_size": replay_seed_size,
        "real_strategy": real_strategy,
        "ensemble_size": ensemble_size,
        "train_family_for_init": train_family_for_init,
        "use_calibrated_acquisition": use_calibrated_acquisition,
        "generator_objective_mode": generator_objective_mode,
        "use_similarity_penalty": use_similarity_penalty,
        "use_length_penalty": use_length_penalty,
        "binary_threshold_strategy": binary_threshold_strategy,
    }
    if replay_strategies:
        kwargs["replay_strategies"] = replay_strategies
    return RunConfig(**kwargs)


def _config_projection(config: RunConfig) -> dict[str, object]:
    payload = config.to_dict()
    return {field: payload.get(field) for field in STUDY_CONFIG_FIELDS}


def _config_mismatches(expected: RunConfig, actual: RunConfig) -> dict[str, dict[str, object]]:
    expected_projection = _config_projection(expected)
    actual_projection = _config_projection(actual)
    mismatches: dict[str, dict[str, object]] = {}
    for field in STUDY_CONFIG_FIELDS:
        if expected_projection.get(field) != actual_projection.get(field):
            mismatches[field] = {
                "expected": expected_projection.get(field),
                "actual": actual_projection.get(field),
            }
    return mismatches


def _replay_complete(run_dir: Path, strategies: list[str]) -> bool:
    if not strategies:
        return False
    replay_root = run_dir / "replay"
    if not replay_root.exists():
        return False
    for strategy in strategies:
        summary_path = replay_root / strategy / "summary.json"
        if not summary_path.exists():
            return False
        payload = _read_json(summary_path)
        if not isinstance(payload, list) or not payload:
            return False
    return True


def _pending_replay_strategies(run_dir: Path, strategies: list[str]) -> list[str]:
    return [
        strategy
        for strategy in strategies
        if not _replay_complete(run_dir, [strategy])
    ]


def _load_existing_study_manifest(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _write_study_manifest(path: Path, manifest: dict[str, object]) -> None:
    _write_json(path, manifest)


def _run_display_name(run_dir: Path) -> str:
    curation_path = run_dir / ".dashboard_label"
    if curation_path.exists():
        label = curation_path.read_text(encoding="utf-8").strip()
        if label:
            return label
    return run_dir.name


def discover_replay_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(
        [
            path
            for path in run_root.iterdir()
            if path.is_dir() and (path / "replay").exists()
        ],
        key=lambda path: path.name,
    )


def _strategy_points(strategy_dir: Path) -> list[dict[str, object]]:
    summary_path = strategy_dir / "summary.json"
    payload = _read_json(summary_path)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    csv_rows = _read_csv(strategy_dir / "learning_curve.csv")
    return [dict(row) for row in csv_rows]


def _valid_metric_points(points: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    valid = []
    for index, point in enumerate(points):
        value = _safe_float(point.get(metric))
        if value is None:
            continue
        labeled_count = _safe_int(point.get("labeled_count"))
        round_id = _safe_int(point.get("round_id"))
        valid.append(
            {
                **point,
                "_index": index,
                "_metric_value": value,
                "_labeled_count": labeled_count if labeled_count is not None else index,
                "_round_id": round_id if round_id is not None else index,
            }
        )
    return sorted(
        valid,
        key=lambda point: (
            int(point["_labeled_count"]),
            int(point["_round_id"]),
            int(point["_index"]),
        ),
    )


def area_under_learning_curve(points: list[dict[str, object]], metric: str) -> float | None:
    valid = _valid_metric_points(points, metric)
    if not valid:
        return None
    if len(valid) == 1:
        return float(valid[0]["_metric_value"])

    xs = [float(point["_labeled_count"]) for point in valid]
    ys = [float(point["_metric_value"]) for point in valid]
    span = xs[-1] - xs[0]
    if span <= 0:
        return _mean(ys)

    area = 0.0
    for index in range(1, len(valid)):
        width = xs[index] - xs[index - 1]
        area += width * (ys[index] + ys[index - 1]) / 2.0
    return float(area / span)


def labels_to_reach_target(
    points: list[dict[str, object]],
    metric: str,
    target: float | None,
) -> int | None:
    if target is None:
        return None
    higher = _higher_is_better(metric)
    for point in _valid_metric_points(points, metric):
        value = float(point["_metric_value"])
        if (higher and value >= target) or (not higher and value <= target):
            return int(point["_labeled_count"])
    return None


def _strategy_summary_row(
    run_dir: Path,
    strategy: str,
    points: list[dict[str, object]],
    metric: str,
    target: float | None,
) -> dict[str, object] | None:
    valid = _valid_metric_points(points, metric)
    if not valid:
        return None
    higher = _higher_is_better(metric)
    metric_values = [float(point["_metric_value"]) for point in valid]
    best_value = max(metric_values) if higher else min(metric_values)
    target_count = labels_to_reach_target(points, metric, target)
    return {
        "run_name": _run_display_name(run_dir),
        "run_dir": str(run_dir),
        "strategy": strategy,
        "metric": metric,
        "higher_is_better": higher,
        "round_count": len(valid),
        "first_round_id": int(valid[0]["_round_id"]),
        "final_round_id": int(valid[-1]["_round_id"]),
        "first_labeled_count": int(valid[0]["_labeled_count"]),
        "final_labeled_count": int(valid[-1]["_labeled_count"]),
        "first_metric": float(valid[0]["_metric_value"]),
        "final_metric": float(valid[-1]["_metric_value"]),
        "best_metric": best_value,
        "aulc_metric": area_under_learning_curve(points, metric),
        "target": target,
        "target_reached": target_count is not None if target is not None else None,
        "labels_to_target": target_count,
    }


def collect_run_strategy_summaries(
    run_dirs: list[Path],
    *,
    metric: str,
    target: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        replay_root = run_dir / "replay"
        if not replay_root.exists():
            continue
        for strategy_dir in sorted(path for path in replay_root.iterdir() if path.is_dir()):
            row = _strategy_summary_row(
                run_dir,
                strategy_dir.name,
                _strategy_points(strategy_dir),
                metric,
                target,
            )
            if row is not None:
                rows.append(row)
    return rows


def _paired_vs_random_rows(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    higher = _higher_is_better(metric)
    by_run: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_dir"]), {})[str(row["strategy"])] = row

    paired_rows: list[dict[str, object]] = []
    for run_rows in by_run.values():
        random_row = run_rows.get("random")
        if random_row is None:
            continue
        random_final = _safe_float(random_row.get("final_metric"))
        random_aulc = _safe_float(random_row.get("aulc_metric"))
        if random_final is None or random_aulc is None:
            continue
        for strategy, row in sorted(run_rows.items()):
            if strategy == "random":
                continue
            final_value = _safe_float(row.get("final_metric"))
            aulc_value = _safe_float(row.get("aulc_metric"))
            if final_value is None or aulc_value is None:
                continue
            final_advantage = _advantage(final_value, random_final, higher_is_better=higher)
            aulc_advantage = _advantage(aulc_value, random_aulc, higher_is_better=higher)
            paired_rows.append(
                {
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "strategy": strategy,
                    "baseline_strategy": "random",
                    "metric": metric,
                    "higher_is_better": higher,
                    "final_metric": final_value,
                    "random_final_metric": random_final,
                    "final_advantage_vs_random": final_advantage,
                    "final_wins_vs_random": _better(final_value, random_final, higher_is_better=higher),
                    "aulc_metric": aulc_value,
                    "random_aulc_metric": random_aulc,
                    "aulc_advantage_vs_random": aulc_advantage,
                    "aulc_wins_vs_random": _better(aulc_value, random_aulc, higher_is_better=higher),
                }
            )
    return paired_rows


def _aggregate_strategy_rows(
    rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    *,
    metric: str,
    target: float | None,
) -> list[dict[str, object]]:
    higher = _higher_is_better(metric)
    paired_by_strategy: dict[str, list[dict[str, object]]] = {}
    for row in paired_rows:
        paired_by_strategy.setdefault(str(row["strategy"]), []).append(row)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for strategy, strategy_rows in grouped.items():
        final_values = [
            value
            for value in (_safe_float(row.get("final_metric")) for row in strategy_rows)
            if value is not None
        ]
        aulc_values = [
            value
            for value in (_safe_float(row.get("aulc_metric")) for row in strategy_rows)
            if value is not None
        ]
        best_values = [
            value
            for value in (_safe_float(row.get("best_metric")) for row in strategy_rows)
            if value is not None
        ]
        final_ci_low, final_ci_high = _ci95(final_values)
        aulc_ci_low, aulc_ci_high = _ci95(aulc_values)

        target_counts = [
            int(row["labels_to_target"])
            for row in strategy_rows
            if row.get("labels_to_target") not in {None, ""}
        ]
        paired = paired_by_strategy.get(strategy, [])
        final_advantages = [
            value
            for value in (_safe_float(row.get("final_advantage_vs_random")) for row in paired)
            if value is not None
        ]
        aulc_advantages = [
            value
            for value in (_safe_float(row.get("aulc_advantage_vs_random")) for row in paired)
            if value is not None
        ]
        aggregate_rows.append(
            {
                "strategy": strategy,
                "metric": metric,
                "higher_is_better": higher,
                "n_runs": len(strategy_rows),
                "n_points_total": sum(int(row.get("round_count", 0) or 0) for row in strategy_rows),
                "final_mean": _mean(final_values),
                "final_std": _sample_std(final_values),
                "final_sem": _sem(final_values),
                "final_ci95_low": final_ci_low,
                "final_ci95_high": final_ci_high,
                "aulc_mean": _mean(aulc_values),
                "aulc_std": _sample_std(aulc_values),
                "aulc_sem": _sem(aulc_values),
                "aulc_ci95_low": aulc_ci_low,
                "aulc_ci95_high": aulc_ci_high,
                "best_mean": _mean(best_values),
                "target": target,
                "target_reached_runs": len(target_counts) if target is not None else None,
                "labels_to_target_median": median(target_counts) if target_counts else None,
                "paired_random_runs": len(paired),
                "final_advantage_mean_vs_random": _mean(final_advantages),
                "aulc_advantage_mean_vs_random": _mean(aulc_advantages),
                "final_win_rate_vs_random": (
                    sum(1 for row in paired if bool(row.get("final_wins_vs_random"))) / len(paired)
                    if paired
                    else None
                ),
                "aulc_win_rate_vs_random": (
                    sum(1 for row in paired if bool(row.get("aulc_wins_vs_random"))) / len(paired)
                    if paired
                    else None
                ),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["aulc_mean"] if row["aulc_mean"] is not None else -math.inf)
            if higher
            else float(row["aulc_mean"] if row["aulc_mean"] is not None else math.inf),
            -float(row["final_mean"] if row["final_mean"] is not None else -math.inf)
            if higher
            else float(row["final_mean"] if row["final_mean"] is not None else math.inf),
            str(row["strategy"]),
        )
    )
    for index, row in enumerate(aggregate_rows, start=1):
        row["rank"] = index
    return aggregate_rows


def _resolve_study_manifest_path(run_root: Path, study: str | Path) -> Path:
    study_text = str(study)
    study_path = Path(study_text)
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    if study_path.is_absolute():
        add_candidate(study_path)
    else:
        add_candidate(study_path)
        add_candidate(run_root / study_path)
    add_candidate(run_root / DEFAULT_STUDY_ROOT_DIRNAME / _slug(study_text) / "study_manifest.json")

    for base in list(candidates):
        add_candidate(base / "study_manifest.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Study manifest not found for {study_text!r}; searched: {searched}")


def _load_study_manifest(run_root: Path, study: str | Path) -> tuple[dict[str, object], Path]:
    manifest_path = _resolve_study_manifest_path(run_root, study)
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Study manifest is not valid JSON: {manifest_path}")
    return payload, manifest_path


def _study_label(manifest: dict[str, object], fallback: str | Path) -> str:
    return str(manifest.get("study_name") or manifest.get("study_slug") or fallback)


def _study_slug_from_manifest(manifest: dict[str, object], fallback: str | Path) -> str:
    return _slug(str(manifest.get("study_slug") or manifest.get("study_name") or fallback))


def _manifest_config_target(manifest: dict[str, object]) -> float | None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None
    return _safe_float(config.get("target"))


def _resolved_comparison_target(
    baseline_manifest: dict[str, object],
    candidate_manifest: dict[str, object],
    requested_target: float | None,
) -> float | None:
    if requested_target is not None:
        return requested_target
    baseline_target = _manifest_config_target(baseline_manifest)
    candidate_target = _manifest_config_target(candidate_manifest)
    if baseline_target is not None and candidate_target is not None:
        if abs(baseline_target - candidate_target) <= 1e-12:
            return baseline_target
        return None
    return baseline_target if baseline_target is not None else candidate_target


def _manifest_summary_outputs(manifest: dict[str, object]) -> dict[str, object]:
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        return {}
    outputs = summary.get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _study_evidence_dir(manifest: dict[str, object]) -> Path:
    summary = manifest.get("summary")
    if isinstance(summary, dict) and summary.get("output_dir"):
        return Path(str(summary["output_dir"]))
    study_dir = manifest.get("study_dir")
    if study_dir:
        return Path(str(study_dir)) / "evidence"
    run_root = Path(str(manifest.get("run_root", ".")))
    return run_root / DEFAULT_STUDY_OUTPUT_DIRNAME


def _manifest_run_names(manifest: dict[str, object]) -> list[str]:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return []
    return [
        str(row["run_name"])
        for row in runs
        if isinstance(row, dict) and row.get("run_name")
    ]


def _summary_rows_match_request(
    rows: list[dict[str, object]],
    *,
    metric: str,
    target: float | None,
) -> bool:
    if not rows:
        return False
    for row in rows:
        row_metric = str(row.get("metric", ""))
        if row_metric and row_metric != metric:
            return False
    if target is None:
        return True
    for row in rows:
        row_target = _safe_float(row.get("target"))
        if row_target is None or abs(row_target - target) > 1e-12:
            return False
    return True


def _load_study_run_strategy_rows(
    manifest: dict[str, object],
    *,
    metric: str,
    target: float | None,
) -> tuple[list[dict[str, object]], Path]:
    outputs = _manifest_summary_outputs(manifest)
    candidates: list[Path] = []
    if outputs.get("run_strategy_summary"):
        candidates.append(Path(str(outputs["run_strategy_summary"])))
    candidates.append(_study_evidence_dir(manifest) / f"{metric}_run_strategy_summary.csv")

    for candidate in candidates:
        rows = _read_csv(candidate)
        if _summary_rows_match_request(rows, metric=metric, target=target):
            return rows, candidate

    run_root = Path(str(manifest.get("run_root", ".")))
    output_dir = _study_evidence_dir(manifest)
    summary = summarize_study(
        run_root,
        output_dir=output_dir,
        metric=metric,
        target=target,
        run_names=_manifest_run_names(manifest) or None,
    )
    return list(summary["run_strategy_rows"]), Path(str(summary["outputs"]["run_strategy_summary"]))


def _manifest_run_lookup(manifest: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    lookup: dict[str, dict[str, dict[str, object]]] = {
        "by_run_name": {},
        "by_run_dir": {},
    }
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        return lookup
    for row in runs:
        if not isinstance(row, dict):
            continue
        run_name = str(row.get("run_name", ""))
        run_dir = str(row.get("run_dir", ""))
        if run_name:
            lookup["by_run_name"][run_name] = row
        if run_dir:
            lookup["by_run_dir"][run_dir] = row
            lookup["by_run_dir"][_path_key(run_dir)] = row
            run_dir_name = Path(run_dir).name
            if run_dir_name:
                lookup["by_run_name"].setdefault(run_dir_name, row)
    return lookup


def _seed_from_row(row: dict[str, object], lookup: dict[str, dict[str, dict[str, object]]]) -> int | None:
    run_dir = str(row.get("run_dir", ""))
    for key in [run_dir, _path_key(run_dir)]:
        manifest_row = lookup["by_run_dir"].get(key)
        if manifest_row:
            seed = _safe_int(manifest_row.get("random_seed"))
            if seed is not None:
                return seed

    run_name = str(row.get("run_name", ""))
    run_dir_name = Path(run_dir).name if run_dir else ""
    for key in [run_name, run_dir_name]:
        manifest_row = lookup["by_run_name"].get(key)
        if manifest_row:
            seed = _safe_int(manifest_row.get("random_seed"))
            if seed is not None:
                return seed
        match = re.search(r"_seed_\d+_(-?\d+)$", key)
        if match:
            return int(match.group(1))
    return None


def _comparison_pair_key(
    row: dict[str, object],
    lookup: dict[str, dict[str, dict[str, object]]],
) -> tuple[str, str, str] | None:
    strategy = str(row.get("strategy", ""))
    if not strategy:
        return None
    seed = _seed_from_row(row, lookup)
    if seed is not None:
        return ("seed", str(seed), strategy)
    run_dir = str(row.get("run_dir", ""))
    run_id = Path(run_dir).name if run_dir else str(row.get("run_name", ""))
    if not run_id:
        return None
    return ("run", run_id, strategy)


def _run_strategy_rows_by_pair_key(
    rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup = _manifest_run_lookup(manifest)
    keyed_rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        key = _comparison_pair_key(row, lookup)
        if key is not None:
            keyed_rows[key] = row
    return keyed_rows


def _pair_target(
    baseline_row: dict[str, object],
    candidate_row: dict[str, object],
    requested_target: float | None,
) -> float | None:
    if requested_target is not None:
        return requested_target
    baseline_target = _safe_float(baseline_row.get("target"))
    candidate_target = _safe_float(candidate_row.get("target"))
    if baseline_target is not None and candidate_target is not None:
        if abs(baseline_target - candidate_target) <= 1e-12:
            return baseline_target
        return None
    return baseline_target if baseline_target is not None else candidate_target


def _study_comparison_pair_row(
    key: tuple[str, str, str],
    baseline_row: dict[str, object],
    candidate_row: dict[str, object],
    *,
    baseline_study: str,
    candidate_study: str,
    metric: str,
    target: float | None,
) -> dict[str, object] | None:
    higher = _higher_is_better(metric)
    baseline_final = _safe_float(baseline_row.get("final_metric"))
    candidate_final = _safe_float(candidate_row.get("final_metric"))
    baseline_aulc = _safe_float(baseline_row.get("aulc_metric"))
    candidate_aulc = _safe_float(candidate_row.get("aulc_metric"))
    if (
        baseline_final is None
        or candidate_final is None
        or baseline_aulc is None
        or candidate_aulc is None
    ):
        return None

    baseline_best = _safe_float(baseline_row.get("best_metric"))
    candidate_best = _safe_float(candidate_row.get("best_metric"))
    baseline_labels = _safe_int(baseline_row.get("labels_to_target"))
    candidate_labels = _safe_int(candidate_row.get("labels_to_target"))
    pair_target = _pair_target(baseline_row, candidate_row, target)

    return {
        "baseline_study": baseline_study,
        "candidate_study": candidate_study,
        "pair_key": f"{key[0]}:{key[1]}|strategy:{key[2]}",
        "pair_key_type": key[0],
        "random_seed": int(key[1]) if key[0] == "seed" else None,
        "strategy": key[2],
        "metric": metric,
        "higher_is_better": higher,
        "baseline_run_name": baseline_row.get("run_name"),
        "candidate_run_name": candidate_row.get("run_name"),
        "baseline_run_dir": baseline_row.get("run_dir"),
        "candidate_run_dir": candidate_row.get("run_dir"),
        "baseline_final_metric": baseline_final,
        "candidate_final_metric": candidate_final,
        "final_advantage": _advantage(candidate_final, baseline_final, higher_is_better=higher),
        "final_win": _better(candidate_final, baseline_final, higher_is_better=higher),
        "baseline_aulc_metric": baseline_aulc,
        "candidate_aulc_metric": candidate_aulc,
        "aulc_advantage": _advantage(candidate_aulc, baseline_aulc, higher_is_better=higher),
        "aulc_win": _better(candidate_aulc, baseline_aulc, higher_is_better=higher),
        "baseline_best_metric": baseline_best,
        "candidate_best_metric": candidate_best,
        "best_advantage": (
            _advantage(candidate_best, baseline_best, higher_is_better=higher)
            if baseline_best is not None and candidate_best is not None
            else None
        ),
        "best_win": (
            _better(candidate_best, baseline_best, higher_is_better=higher)
            if baseline_best is not None and candidate_best is not None
            else None
        ),
        "target": pair_target,
        "baseline_target_reached": _safe_bool(baseline_row.get("target_reached")),
        "candidate_target_reached": _safe_bool(candidate_row.get("target_reached")),
        "baseline_labels_to_target": baseline_labels,
        "candidate_labels_to_target": candidate_labels,
        "labels_saved_to_target": (
            baseline_labels - candidate_labels
            if baseline_labels is not None and candidate_labels is not None
            else None
        ),
        "baseline_final_labeled_count": _safe_int(baseline_row.get("final_labeled_count")),
        "candidate_final_labeled_count": _safe_int(candidate_row.get("final_labeled_count")),
    }


def _values(rows: list[dict[str, object]], field: str) -> list[float]:
    return [
        value
        for value in (_safe_float(row.get(field)) for row in rows)
        if value is not None
    ]


def _win_rate(rows: list[dict[str, object]], field: str) -> float | None:
    values = [
        value
        for value in (_safe_bool(row.get(field)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _aggregate_study_comparison_rows(
    paired_rows: list[dict[str, object]],
    *,
    baseline_study: str,
    candidate_study: str,
    metric: str,
    target: float | None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in paired_rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    strategy_rows: list[dict[str, object]] = []
    for strategy, rows in grouped.items():
        final_advantages = _values(rows, "final_advantage")
        aulc_advantages = _values(rows, "aulc_advantage")
        best_advantages = _values(rows, "best_advantage")
        labels_saved = _values(rows, "labels_saved_to_target")
        final_ci_low, final_ci_high = _ci95(final_advantages)
        aulc_ci_low, aulc_ci_high = _ci95(aulc_advantages)
        strategy_rows.append(
            {
                "baseline_study": baseline_study,
                "candidate_study": candidate_study,
                "strategy": strategy,
                "metric": metric,
                "higher_is_better": _higher_is_better(metric),
                "paired_count": len(rows),
                "target": target,
                "final_advantage_mean": _mean(final_advantages),
                "final_advantage_std": _sample_std(final_advantages),
                "final_advantage_sem": _sem(final_advantages),
                "final_advantage_ci95_low": final_ci_low,
                "final_advantage_ci95_high": final_ci_high,
                "final_win_rate": _win_rate(rows, "final_win"),
                "aulc_advantage_mean": _mean(aulc_advantages),
                "aulc_advantage_std": _sample_std(aulc_advantages),
                "aulc_advantage_sem": _sem(aulc_advantages),
                "aulc_advantage_ci95_low": aulc_ci_low,
                "aulc_advantage_ci95_high": aulc_ci_high,
                "aulc_win_rate": _win_rate(rows, "aulc_win"),
                "best_advantage_mean": _mean(best_advantages),
                "best_win_rate": _win_rate(rows, "best_win"),
                "label_efficiency_pair_count": len(labels_saved),
                "labels_saved_to_target_mean": _mean(labels_saved),
                "labels_saved_to_target_median": median(labels_saved) if labels_saved else None,
            }
        )

    strategy_rows.sort(
        key=lambda row: (
            -float(row["aulc_advantage_mean"] if row["aulc_advantage_mean"] is not None else -math.inf),
            -float(row["final_advantage_mean"] if row["final_advantage_mean"] is not None else -math.inf),
            str(row["strategy"]),
        )
    )
    for index, row in enumerate(strategy_rows, start=1):
        row["rank"] = index
    return strategy_rows


def _fmt_number(value: object, *, signed: bool = False) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    prefix = "+" if signed and parsed > 0 else ""
    return f"{prefix}{parsed:.4g}"


def _comparison_narrative(summary: dict[str, object], best_row: dict[str, object] | None) -> str:
    direction = "higher is better" if summary["higher_is_better"] else "lower is better"
    lines = [
        "# Study Comparison",
        "",
        f"Baseline study: {summary['baseline_study']}",
        f"Candidate study: {summary['candidate_study']}",
        f"Metric: {summary['metric']} ({direction})",
        f"Matched seed/strategy pairs: {summary['paired_count']}",
        "",
    ]
    if best_row is None:
        lines.append("No matched seed/strategy pairs were found. Check that both study manifests share seeds and strategies.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"Best strategy by mean AULC advantage: {best_row['strategy']}",
            (
                "Mean AULC advantage: "
                f"{_fmt_number(best_row.get('aulc_advantage_mean'), signed=True)} "
                f"(95% CI {_fmt_number(best_row.get('aulc_advantage_ci95_low'), signed=True)} to "
                f"{_fmt_number(best_row.get('aulc_advantage_ci95_high'), signed=True)})"
            ),
            f"Mean final-metric advantage: {_fmt_number(best_row.get('final_advantage_mean'), signed=True)}",
            f"AULC win rate: {_fmt_number(best_row.get('aulc_win_rate'))}",
        ]
    )
    if summary.get("target") is not None:
        lines.append(
            "Median labels saved to target: "
            f"{_fmt_number(best_row.get('labels_saved_to_target_median'), signed=True)}"
        )
    lines.extend(
        [
            "",
            "Interpretation note: positive advantages mean the candidate study improved over the baseline after accounting for metric direction.",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_studies(
    run_root: Path,
    *,
    baseline_study: str | Path,
    candidate_study: str | Path,
    output_dir: Path | None = None,
    metric: str = "f1",
    target: float | None = None,
) -> dict[str, object]:
    run_root = Path(run_root)
    baseline_manifest, baseline_manifest_path = _load_study_manifest(run_root, baseline_study)
    candidate_manifest, candidate_manifest_path = _load_study_manifest(run_root, candidate_study)
    if baseline_manifest_path.resolve() == candidate_manifest_path.resolve():
        raise ValueError("Baseline and candidate study manifests must be different.")
    resolved_target = _resolved_comparison_target(
        baseline_manifest,
        candidate_manifest,
        target,
    )
    baseline_label = _study_label(baseline_manifest, baseline_study)
    candidate_label = _study_label(candidate_manifest, candidate_study)
    baseline_slug = _study_slug_from_manifest(baseline_manifest, baseline_study)
    candidate_slug = _study_slug_from_manifest(candidate_manifest, candidate_study)

    baseline_rows, baseline_summary_path = _load_study_run_strategy_rows(
        baseline_manifest,
        metric=metric,
        target=resolved_target,
    )
    candidate_rows, candidate_summary_path = _load_study_run_strategy_rows(
        candidate_manifest,
        metric=metric,
        target=resolved_target,
    )
    baseline_by_key = _run_strategy_rows_by_pair_key(baseline_rows, baseline_manifest)
    candidate_by_key = _run_strategy_rows_by_pair_key(candidate_rows, candidate_manifest)

    paired_rows = []
    for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
        pair_row = _study_comparison_pair_row(
            key,
            baseline_by_key[key],
            candidate_by_key[key],
            baseline_study=baseline_label,
            candidate_study=candidate_label,
            metric=metric,
            target=resolved_target,
        )
        if pair_row is not None:
            paired_rows.append(pair_row)

    strategy_rows = _aggregate_study_comparison_rows(
        paired_rows,
        baseline_study=baseline_label,
        candidate_study=candidate_label,
        metric=metric,
        target=resolved_target,
    )
    best_row = strategy_rows[0] if strategy_rows else None

    resolved_output_dir = output_dir or (
        run_root
        / DEFAULT_STUDY_ROOT_DIRNAME
        / DEFAULT_STUDY_COMPARISON_DIRNAME
        / f"{baseline_slug}_vs_{candidate_slug}"
        / _slug(metric)
    )
    paired_path = resolved_output_dir / f"{metric}_paired_study_comparison.csv"
    strategy_path = resolved_output_dir / f"{metric}_study_comparison_by_strategy.csv"
    summary_path = resolved_output_dir / f"{metric}_study_comparison_summary.json"
    narrative_path = resolved_output_dir / f"{metric}_thesis_narrative.md"

    _write_csv(
        paired_path,
        [
            "baseline_study",
            "candidate_study",
            "pair_key",
            "pair_key_type",
            "random_seed",
            "strategy",
            "metric",
            "higher_is_better",
            "baseline_run_name",
            "candidate_run_name",
            "baseline_run_dir",
            "candidate_run_dir",
            "baseline_final_metric",
            "candidate_final_metric",
            "final_advantage",
            "final_win",
            "baseline_aulc_metric",
            "candidate_aulc_metric",
            "aulc_advantage",
            "aulc_win",
            "baseline_best_metric",
            "candidate_best_metric",
            "best_advantage",
            "best_win",
            "target",
            "baseline_target_reached",
            "candidate_target_reached",
            "baseline_labels_to_target",
            "candidate_labels_to_target",
            "labels_saved_to_target",
            "baseline_final_labeled_count",
            "candidate_final_labeled_count",
        ],
        paired_rows,
    )
    _write_csv(
        strategy_path,
        [
            "rank",
            "baseline_study",
            "candidate_study",
            "strategy",
            "metric",
            "higher_is_better",
            "paired_count",
            "target",
            "final_advantage_mean",
            "final_advantage_std",
            "final_advantage_sem",
            "final_advantage_ci95_low",
            "final_advantage_ci95_high",
            "final_win_rate",
            "aulc_advantage_mean",
            "aulc_advantage_std",
            "aulc_advantage_sem",
            "aulc_advantage_ci95_low",
            "aulc_advantage_ci95_high",
            "aulc_win_rate",
            "best_advantage_mean",
            "best_win_rate",
            "label_efficiency_pair_count",
            "labels_saved_to_target_mean",
            "labels_saved_to_target_median",
        ],
        strategy_rows,
    )

    summary = {
        "baseline_study": baseline_label,
        "candidate_study": candidate_label,
        "baseline_manifest": str(baseline_manifest_path),
        "candidate_manifest": str(candidate_manifest_path),
        "baseline_run_strategy_summary": str(baseline_summary_path),
        "candidate_run_strategy_summary": str(candidate_summary_path),
        "run_root": str(run_root),
        "output_dir": str(resolved_output_dir),
        "metric": metric,
        "higher_is_better": _higher_is_better(metric),
        "target": resolved_target,
        "paired_count": len(paired_rows),
        "strategy_count": len(strategy_rows),
        "unmatched_baseline_count": len(set(baseline_by_key) - set(candidate_by_key)),
        "unmatched_candidate_count": len(set(candidate_by_key) - set(baseline_by_key)),
        "best_strategy_by_aulc_advantage": best_row["strategy"] if best_row else "",
        "generated_at": _now_iso(),
        "outputs": {
            "paired_comparison": str(paired_path),
            "strategy_summary": str(strategy_path),
            "comparison_summary": str(summary_path),
            "thesis_narrative": str(narrative_path),
        },
    }
    narrative = _comparison_narrative(summary, best_row)
    _write_json(summary_path, summary)
    narrative_path.write_text(narrative, encoding="utf-8")
    return {
        **summary,
        "paired_rows": paired_rows,
        "strategy_rows": strategy_rows,
        "narrative": narrative,
    }


def summarize_study(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    metric: str = "f1",
    target: float | None = None,
    run_names: list[str] | None = None,
) -> dict[str, object]:
    selected_names = set(run_names or [])
    run_dirs = discover_replay_run_dirs(run_root)
    if selected_names:
        run_dirs = [path for path in run_dirs if path.name in selected_names]

    resolved_output_dir = output_dir or run_root / DEFAULT_STUDY_OUTPUT_DIRNAME
    run_strategy_rows = collect_run_strategy_summaries(
        run_dirs,
        metric=metric,
        target=target,
    )
    paired_rows = _paired_vs_random_rows(run_strategy_rows, metric)
    strategy_rows = _aggregate_strategy_rows(
        run_strategy_rows,
        paired_rows,
        metric=metric,
        target=target,
    )

    run_strategy_path = resolved_output_dir / f"{metric}_run_strategy_summary.csv"
    strategy_path = resolved_output_dir / f"{metric}_strategy_summary.csv"
    paired_path = resolved_output_dir / f"{metric}_paired_vs_random.csv"
    manifest_path = resolved_output_dir / f"{metric}_study_summary.json"

    _write_csv(
        run_strategy_path,
        [
            "run_name",
            "run_dir",
            "strategy",
            "metric",
            "higher_is_better",
            "round_count",
            "first_round_id",
            "final_round_id",
            "first_labeled_count",
            "final_labeled_count",
            "first_metric",
            "final_metric",
            "best_metric",
            "aulc_metric",
            "target",
            "target_reached",
            "labels_to_target",
        ],
        run_strategy_rows,
    )
    _write_csv(
        paired_path,
        [
            "run_name",
            "run_dir",
            "strategy",
            "baseline_strategy",
            "metric",
            "higher_is_better",
            "final_metric",
            "random_final_metric",
            "final_advantage_vs_random",
            "final_wins_vs_random",
            "aulc_metric",
            "random_aulc_metric",
            "aulc_advantage_vs_random",
            "aulc_wins_vs_random",
        ],
        paired_rows,
    )
    _write_csv(
        strategy_path,
        [
            "rank",
            "strategy",
            "metric",
            "higher_is_better",
            "n_runs",
            "n_points_total",
            "final_mean",
            "final_std",
            "final_sem",
            "final_ci95_low",
            "final_ci95_high",
            "aulc_mean",
            "aulc_std",
            "aulc_sem",
            "aulc_ci95_low",
            "aulc_ci95_high",
            "best_mean",
            "target",
            "target_reached_runs",
            "labels_to_target_median",
            "paired_random_runs",
            "final_advantage_mean_vs_random",
            "aulc_advantage_mean_vs_random",
            "final_win_rate_vs_random",
            "aulc_win_rate_vs_random",
        ],
        strategy_rows,
    )

    best_strategy = strategy_rows[0]["strategy"] if strategy_rows else ""
    manifest = {
        "run_root": str(run_root),
        "output_dir": str(resolved_output_dir),
        "metric": metric,
        "higher_is_better": _higher_is_better(metric),
        "target": target,
        "run_count": len(run_dirs),
        "run_strategy_count": len(run_strategy_rows),
        "strategy_count": len(strategy_rows),
        "paired_vs_random_count": len(paired_rows),
        "best_strategy_by_aulc": best_strategy,
        "outputs": {
            "run_strategy_summary": str(run_strategy_path),
            "strategy_summary": str(strategy_path),
            "paired_vs_random": str(paired_path),
            "study_summary": str(manifest_path),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "run_strategy_rows": run_strategy_rows,
        "strategy_rows": strategy_rows,
        "paired_vs_random_rows": paired_rows,
    }


def run_study(
    *,
    study_name: str,
    run_root: Path,
    seed_count: int = 5,
    seed_start: int = DEFAULT_STUDY_SEED_START,
    seed_step: int = DEFAULT_STUDY_SEED_STEP,
    seed_index_start: int = 1,
    epochs: int = 70,
    max_rounds: int = 10,
    batch_size: int = 5,
    candidate_pool_min: int = 50,
    replay_seed_size: int = 40,
    real_strategy: str = "ensemble_mi",
    replay_strategies: list[str] | None = None,
    ensemble_size: int = 5,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    metric: str = "f1",
    target: float | None = None,
    summarize: bool = True,
    dry_run: bool = False,
    force_replay: bool = False,
    allow_config_mismatch: bool = False,
) -> dict[str, object]:
    if not str(study_name or "").strip():
        raise ValueError("study_name must not be empty")
    run_root = Path(run_root)
    paths = _study_paths(run_root, study_name)
    seeds = _seed_values(seed_count, seed_start, seed_step)
    started_at = _now_iso()
    existing_manifest = _load_existing_study_manifest(paths["manifest"])
    previous_runs = {
        str(row.get("run_name", "")): row
        for row in existing_manifest.get("runs", [])
        if isinstance(row, dict)
    }

    run_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if seed_index_start <= 0:
        raise ValueError("seed_index_start must be positive")

    planned_configs = [
        _study_config(
            study_name=study_name,
            run_root=run_root,
            seed=seed,
            index=index,
            epochs=epochs,
            max_rounds=max_rounds,
            batch_size=batch_size,
            candidate_pool_min=candidate_pool_min,
            replay_seed_size=replay_seed_size,
            real_strategy=real_strategy,
            replay_strategies=replay_strategies,
            ensemble_size=ensemble_size,
            train_family_for_init=train_family_for_init,
            use_calibrated_acquisition=use_calibrated_acquisition,
            generator_objective_mode=generator_objective_mode,
            use_similarity_penalty=use_similarity_penalty,
            use_length_penalty=use_length_penalty,
            binary_threshold_strategy=binary_threshold_strategy,
        )
        for index, seed in enumerate(seeds, start=seed_index_start)
    ]

    run_root.mkdir(parents=True, exist_ok=True)
    paths["study_dir"].mkdir(parents=True, exist_ok=True)

    for config in planned_configs:
        run_dir = config.run_dir
        replay_plan = list(config.replay_strategies)
        row: dict[str, object] = {
            "run_name": config.run_name,
            "run_dir": str(run_dir),
            "random_seed": config.random_seed,
            "status": "planned" if dry_run else "running",
            "started_at": _now_iso() if not dry_run else "",
            "completed_at": "",
            "replay_strategies": replay_plan,
            "previous_status": str(previous_runs.get(config.run_name, {}).get("status", "")),
        }
        try:
            config_path = run_dir / "config.json"
            if dry_run:
                row["status"] = "planned"
                row["action"] = "would-init-and-replay" if not config_path.exists() else "would-resume"
                run_rows.append(row)
                continue

            if config_path.exists():
                existing_config = RunConfig.load(config_path)
                mismatches = _config_mismatches(config, existing_config)
                if mismatches and not allow_config_mismatch:
                    raise ValueError(
                        "Existing run config does not match this study plan: "
                        + json.dumps(mismatches, sort_keys=True)
                    )
                row["init_action"] = "reused-existing-run"
                config = existing_config
                replay_plan = list(config.replay_strategies if replay_strategies is None else replay_strategies)
            else:
                init_run(config)
                row["init_action"] = "created-run"

            pending_strategies = replay_plan if force_replay else _pending_replay_strategies(run_dir, replay_plan)
            row["pending_replay_strategies"] = pending_strategies
            if pending_strategies:
                run_replay(run_dir, pending_strategies)
                row["replay_action"] = "ran-replay"
            else:
                row["replay_action"] = "reused-existing-replay"

            row["status"] = "replay_complete" if _replay_complete(run_dir, replay_plan) else "partial"
            row["completed_at"] = _now_iso()
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["completed_at"] = _now_iso()
            failures.append(row)
        run_rows.append(row)

        manifest = {
            "study_name": study_name,
            "study_slug": _slug(study_name),
            "run_root": str(run_root),
            "study_dir": str(paths["study_dir"]),
            "manifest_path": str(paths["manifest"]),
            "started_at": started_at,
            "updated_at": _now_iso(),
            "dry_run": dry_run,
            "status": "failed" if failures else "running",
            "config": {
                "seed_count": seed_count,
                "seed_start": seed_start,
                "seed_step": seed_step,
                "seed_index_start": seed_index_start,
                "seeds": seeds,
                "epochs": epochs,
                "max_rounds": max_rounds,
                "batch_size": batch_size,
                "candidate_pool_min": candidate_pool_min,
                "replay_seed_size": replay_seed_size,
                "real_strategy": real_strategy,
                "replay_strategies": replay_strategies or RunConfig().replay_strategies,
                "ensemble_size": ensemble_size,
                "train_family_for_init": train_family_for_init,
                "use_calibrated_acquisition": use_calibrated_acquisition,
                "generator_objective_mode": generator_objective_mode,
                "use_similarity_penalty": use_similarity_penalty,
                "use_length_penalty": use_length_penalty,
                "binary_threshold_strategy": binary_threshold_strategy,
                "metric": metric,
                "target": target,
            },
            "runs": run_rows,
            "failure_count": len(failures),
        }
        _write_study_manifest(paths["manifest"], manifest)

    completed_rows = [row for row in run_rows if str(row.get("status", "")) == "replay_complete"]
    summary_payload: dict[str, object] = {}
    if summarize and not dry_run and completed_rows:
        summary_payload = summarize_study(
            run_root,
            output_dir=paths["evidence"],
            metric=metric,
            target=target,
            run_names=[str(row["run_name"]) for row in completed_rows],
        )

    final_status = "planned" if dry_run else ("failed" if failures else "completed")
    if not dry_run and completed_rows and len(completed_rows) < len(run_rows) and not failures:
        final_status = "partial"
    final_manifest = {
        "study_name": study_name,
        "study_slug": _slug(study_name),
        "run_root": str(run_root),
        "study_dir": str(paths["study_dir"]),
        "manifest_path": str(paths["manifest"]),
        "started_at": started_at,
        "updated_at": _now_iso(),
        "dry_run": dry_run,
        "status": final_status,
        "config": {
            "seed_count": seed_count,
            "seed_start": seed_start,
            "seed_step": seed_step,
            "seed_index_start": seed_index_start,
            "seeds": seeds,
            "epochs": epochs,
            "max_rounds": max_rounds,
            "batch_size": batch_size,
            "candidate_pool_min": candidate_pool_min,
            "replay_seed_size": replay_seed_size,
            "real_strategy": real_strategy,
            "replay_strategies": replay_strategies or RunConfig().replay_strategies,
            "ensemble_size": ensemble_size,
            "train_family_for_init": train_family_for_init,
            "use_calibrated_acquisition": use_calibrated_acquisition,
            "generator_objective_mode": generator_objective_mode,
            "use_similarity_penalty": use_similarity_penalty,
            "use_length_penalty": use_length_penalty,
            "binary_threshold_strategy": binary_threshold_strategy,
            "metric": metric,
            "target": target,
        },
        "runs": run_rows,
        "run_count": len(run_rows),
        "completed_run_count": len(completed_rows),
        "failure_count": len(failures),
        "summary": {
            key: value
            for key, value in summary_payload.items()
            if key not in {"run_strategy_rows", "strategy_rows", "paired_vs_random_rows"}
        },
    }
    _write_study_manifest(paths["manifest"], final_manifest)
    return final_manifest
