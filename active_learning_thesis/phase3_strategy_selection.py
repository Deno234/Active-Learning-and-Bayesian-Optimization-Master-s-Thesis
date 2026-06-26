from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct
import zlib
from typing import Iterable

from active_learning_thesis.config import THESIS_FULL_REPLAY_STRATEGIES


DEFAULT_PHASE2_ROOT = Path("thesis_results") / "02_replay"
DEFAULT_OUTPUT_ROOT = Path("thesis_results") / "03_real_al_strategy_selection"
TARGET_F1_VALUES = ("0.8", "0.84", "0.86")
RANDOM_AULC_FLOOR_TOLERANCE = 1e-6
ROLE_UNCERTAINTY = {"predictive_entropy", "ensemble_mi", "family_qbc", "ucb"}
ROLE_DIVERSITY = {"hybrid_mi_diverse", "cluster_diverse_representative", "similarity_penalized_mean", "oed_logdet"}
TRIO_DEFINITIONS = {
    "Trio A - performance + QBC + diversity": ("predictive_entropy", "family_qbc", "cluster_diverse_representative"),
    "Trio B - lower-overlap but weaker": ("predictive_entropy", "ucb", "cluster_diverse_representative"),
    "Trio C - ensemble uncertainty alternative": ("predictive_entropy", "ensemble_mi", "cluster_diverse_representative"),
}
PRACTICAL_COST = {
    "random": ("low", "baseline/control only"),
    "ensemble_mean": ("medium", "AP_SP ensemble; mature but exploitative"),
    "similarity_penalized_mean": ("medium", "AP_SP ensemble plus sequence-similarity penalty"),
    "predictive_entropy": ("medium", "AP_SP ensemble predictive uncertainty"),
    "ensemble_mi": ("medium", "AP_SP ensemble mutual information"),
    "ucb": ("medium", "AP_SP ensemble mean/std acquisition"),
    "family_qbc": ("high", "family committee required"),
    "cluster_diverse_representative": ("medium", "candidate clustering/diversity bookkeeping"),
    "oed_logdet": ("medium-high", "embedding/logdet diversity calculation"),
    "hybrid_mi_diverse": ("medium-high", "uncertainty plus diversity bookkeeping"),
}


@dataclass(frozen=True)
class Phase3SelectionOptions:
    phase2_root: Path = DEFAULT_PHASE2_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    top_k: int = 3
    exclude: tuple[str, ...] = ("random",)
    min_overlap_warning: float = 0.40


@dataclass
class StrategyEvidence:
    strategy: str
    role: str
    practical_cost: str
    practical_note: str
    is_baseline: bool = False
    seed_metrics: dict[int, dict[str, object]] = field(default_factory=dict)
    combined: dict[str, object] = field(default_factory=dict)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ffloat(value: object, default: float | None = None) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "not_reached", "not reached"}:
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if math.isnan(number):
        return default
    return number


def iint(value: object, default: int = 0) -> int:
    number = ffloat(value)
    return int(number) if number is not None else default


def mean(values: Iterable[float]) -> float | None:
    usable = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.fmean(usable) if usable else None


def stdev(values: Iterable[float]) -> float | None:
    usable = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.stdev(usable) if len(usable) > 1 else 0.0 if usable else None


def fmt(value: object, digits: int = 4) -> str:
    number = ffloat(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def discover_phase2_inputs(phase2_root: Path) -> dict[str, Path]:
    benchmark = phase2_root / "benchmark"
    ablation = phase2_root / "ablation"
    paths = {
        "strategy_summary": benchmark / "strategy_summary.csv",
        "paired_vs_random": benchmark / "paired_vs_random.csv",
        "labels_to_target": benchmark / "labels_to_target_summary.csv",
        "selected_sequences": benchmark / "per_run_selected_sequences.csv",
        "acquisition_log": benchmark / "per_run_acquisition_log.csv",
        "learning_curves": benchmark / "learning_curves.csv",
        "round_metrics": benchmark / "per_run_round_metrics.csv",
        "compatibility": benchmark / "strategy_compatibility_matrix.csv",
        "ablation_summary": ablation / "ablation_summary.csv",
        "ablation_compatibility": ablation / "strategy_compatibility_matrix.csv",
    }
    manifest_path = benchmark / "benchmark_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        for key in list(paths):
            value = manifest.get(key) or manifest.get(f"{key}_path")
            if isinstance(value, str):
                candidate = Path(value)
                paths[key] = candidate if candidate.is_absolute() else phase2_root / candidate
    return paths


def role_for_strategy(strategy: str) -> str:
    if strategy == "random":
        return "baseline/control"
    if strategy == "family_qbc":
        return "model/committee uncertainty"
    if strategy in ROLE_UNCERTAINTY:
        return "model/uncertainty"
    if strategy in ROLE_DIVERSITY:
        return "diversity/novelty"
    return "best replay/sample efficiency"


def auc_trapezoid(points: list[tuple[float, float]], *, normalize: bool = True) -> float | None:
    clean = sorted({(float(x), float(y)) for x, y in points if x is not None and y is not None})
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    span = clean[-1][0] - clean[0][0]
    return area / span if normalize and span > 0 else area


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def overlap_band(value: float | None) -> str:
    if value is None:
        return "not available"
    if value < 0.20:
        return "very different"
    if value < 0.40:
        return "moderately different"
    if value < 0.60:
        return "somewhat redundant"
    return "likely redundant"


def random_floor_status(delta: float | None, *, tolerance: float = RANDOM_AULC_FLOOR_TOLERANCE) -> str:
    if delta is None:
        return "unavailable"
    if delta > tolerance:
        return "meaningfully better than random"
    if delta >= -tolerance:
        return "passes floor by tolerance"
    return "below random"


def normalized_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left:
        return 1.0 if right else 0.0
    if not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if char_left == char_right else 1),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def mean_pairwise_normalized_edit_distance(sequences: list[str]) -> float | None:
    unique = list(dict.fromkeys(sequence for sequence in sequences if sequence))
    if len(unique) < 2:
        return None
    distances = [
        normalized_edit_distance(unique[i], unique[j])
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
    ]
    return mean(distances)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        cursor = 0
        while cursor < len(order):
            end = cursor
            while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
                end += 1
            rank = (cursor + end + 2) / 2.0
            for offset in range(cursor, end + 1):
                result[order[offset]] = rank
            cursor = end + 1
        return result
    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = statistics.fmean(rx)
    mean_y = statistics.fmean(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in rx))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ry))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def run_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("outer_fold_id", "")),
        str(row.get("inner_fold_id", "")),
        str(row.get("replay_seed_size", row.get("initial_label_count", ""))),
        str(row.get("run_seed", "")),
        str(row.get("round_id", "")),
    )


def seed_value(row: dict[str, str]) -> int:
    return iint(row.get("replay_seed_size", row.get("initial_label_count", "")))


def strategy_summary_by_seed(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    result: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        result[(row.get("strategy", ""), iint(row.get("initial_label_count")), row.get("evaluation_dataset", ""))] = row
    return result


def compute_auc_rows(metric_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[tuple[str, int, str, str], dict[str, float]]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, str]]] = {}
    for row in metric_rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        key = (
            row.get("strategy", ""),
            seed_value(row),
            row.get("evaluation_dataset", ""),
            row.get("outer_fold_id", ""),
        )
        groups.setdefault(key, []).append(row)
    output_rows: list[dict[str, object]] = []
    lookup: dict[tuple[str, int, str, str], dict[str, float]] = {}
    for key, rows in groups.items():
        strategy, seed, dataset, fold = key
        labeled_points = []
        round_points = []
        for row in rows:
            y = ffloat(row.get("f1"))
            labeled = ffloat(row.get("labeled_count"))
            round_id = ffloat(row.get("round_id"))
            if y is not None and labeled is not None:
                labeled_points.append((labeled, y))
            if y is not None and round_id is not None:
                round_points.append((round_id, y))
        aulc_labeled = auc_trapezoid(labeled_points, normalize=True)
        aulc_round = auc_trapezoid(round_points, normalize=True)
        final_row = max(rows, key=lambda item: ffloat(item.get("round_id"), -1) or -1)
        payload = {
            "strategy": strategy,
            "initial_label_count": seed,
            "evaluation_dataset": dataset,
            "outer_fold_id": fold,
            "AULC_F1_by_labeled_count": aulc_labeled,
            "AULC_by_round": aulc_round,
            "final_F1": ffloat(final_row.get("f1")),
            "final_PR_AUC": ffloat(final_row.get("pr_auc")),
            "final_ROC_AUC": ffloat(final_row.get("roc_auc")),
            "final_Brier": ffloat(final_row.get("brier_score")),
            "final_ECE_10": ffloat(final_row.get("ece_10")),
            "final_labeled_count": ffloat(final_row.get("labeled_count")),
        }
        lookup[key] = {k: v for k, v in payload.items() if isinstance(v, float)}
        output_rows.append(payload)
    return output_rows, lookup


def compute_rank_rows(auc_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for row in auc_rows:
        if row.get("evaluation_dataset") != "holdout":
            continue
        key = (int(row["initial_label_count"]), str(row["outer_fold_id"]), str(row["evaluation_dataset"]))
        groups.setdefault(key, []).append(row)
    ranked: list[dict[str, object]] = []
    for (_seed, _fold, _dataset), rows in groups.items():
        ordered = sorted(
            rows,
            key=lambda item: (
                ffloat(item.get("AULC_F1_by_labeled_count"), -999.0) or -999.0,
                ffloat(item.get("final_F1"), -999.0) or -999.0,
            ),
            reverse=True,
        )
        for rank, row in enumerate(ordered, start=1):
            out = dict(row)
            out["rank_by_holdout_AULC_labeled"] = rank
            ranked.append(out)
    return ranked


def aggregate_auc_by_seed(auc_rows: list[dict[str, object]]) -> dict[tuple[str, int, str], dict[str, float | None]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in auc_rows:
        key = (str(row.get("strategy", "")), iint(row.get("initial_label_count")), str(row.get("evaluation_dataset", "")))
        grouped.setdefault(key, []).append(row)
    output: dict[tuple[str, int, str], dict[str, float | None]] = {}
    for key, rows in grouped.items():
        output[key] = {
            "mean_AULC_F1_by_labeled_count": mean([ffloat(row.get("AULC_F1_by_labeled_count")) for row in rows]),
            "std_AULC_F1_by_labeled_count": stdev([ffloat(row.get("AULC_F1_by_labeled_count")) for row in rows]),
            "mean_AULC_by_round": mean([ffloat(row.get("AULC_by_round")) for row in rows]),
            "std_AULC_by_round": stdev([ffloat(row.get("AULC_by_round")) for row in rows]),
            "mean_final_F1": mean([ffloat(row.get("final_F1")) for row in rows]),
            "mean_final_PR_AUC": mean([ffloat(row.get("final_PR_AUC")) for row in rows]),
            "mean_final_ROC_AUC": mean([ffloat(row.get("final_ROC_AUC")) for row in rows]),
            "mean_final_Brier": mean([ffloat(row.get("final_Brier")) for row in rows]),
            "mean_final_ECE_10": mean([ffloat(row.get("final_ECE_10")) for row in rows]),
        }
    return output


def labels_to_target_rows(rows: list[dict[str, str]], max_labeled_by_seed: dict[int, float]) -> tuple[list[dict[str, object]], dict[tuple[str, int, str, str], float]]:
    output: list[dict[str, object]] = []
    lookup: dict[tuple[str, int, str, str], float] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        seed = iint(row.get("initial_label_count"))
        mean_labels = ffloat(row.get("mean_labels_to_target"))
        reached = iint(row.get("reached_count"))
        repeats = max(1, iint(row.get("n_repeats"), 1))
        penalty = float(max_labeled_by_seed.get(seed, seed + 100)) + max(1, repeats - reached)
        rank_value = mean_labels if mean_labels is not None and reached == repeats else penalty
        out = {
            "strategy": row.get("strategy", ""),
            "initial_label_count": seed,
            "evaluation_dataset": row.get("evaluation_dataset", ""),
            "target_f1": row.get("target_f1", ""),
            "n_repeats": repeats,
            "mean_labels_to_target": mean_labels if mean_labels is not None else "",
            "median_labels_to_target": row.get("median_labels_to_target", ""),
            "reached_count": reached,
            "not_reached_count": repeats - reached,
            "labels_to_target_rank_value": rank_value,
        }
        lookup[(out["strategy"], seed, out["evaluation_dataset"], out["target_f1"])] = rank_value
        output.append(out)
    return output, lookup


def selected_sets_by_run(rows: list[dict[str, str]]) -> dict[tuple[int, str, str], dict[str, set[str]]]:
    result: dict[tuple[int, str, str], dict[str, set[str]]] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        key = (seed_value(row), row.get("outer_fold_id", ""), row.get("run_seed", ""))
        result.setdefault(key, {}).setdefault(row.get("strategy", ""), set()).add(row.get("sequence", ""))
    return result


def compute_overlap(rows: list[dict[str, str]], strategies: list[str]) -> tuple[list[dict[str, object]], dict[int | str, dict[str, dict[str, float]]]]:
    sets_by_run = selected_sets_by_run(rows)
    pair_rows: list[dict[str, object]] = []
    for (seed, fold, run_seed), by_strategy in sets_by_run.items():
        for a in strategies:
            for b in strategies:
                set_a = by_strategy.get(a, set())
                set_b = by_strategy.get(b, set())
                pair_rows.append(
                    {
                        "initial_label_count": seed,
                        "outer_fold_id": fold,
                        "run_seed": run_seed,
                        "strategy_a": a,
                        "strategy_b": b,
                        "selected_a_count": len(set_a),
                        "selected_b_count": len(set_b),
                        "overlap_count": len(set_a & set_b),
                        "union_count": len(set_a | set_b),
                        "jaccard": jaccard(set_a, set_b),
                    }
                )
    matrices: dict[int | str, dict[str, dict[str, float]]] = {}
    for seed in sorted({int(row["initial_label_count"]) for row in pair_rows}):
        matrices[seed] = _matrix_from_pair_rows(pair_rows, strategies, seed_filter=seed)
    matrices["combined"] = _matrix_from_pair_rows(pair_rows, strategies, seed_filter=None)
    return pair_rows, matrices


def same_round_overlap_rows(rows: list[dict[str, str]], strategies: list[str]) -> list[dict[str, object]]:
    by_key_strategy_round: dict[tuple[int, str, str, str, str], set[str]] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        key = (seed_value(row), row.get("outer_fold_id", ""), row.get("run_seed", ""), row.get("strategy", ""), row.get("round_id", ""))
        by_key_strategy_round.setdefault(key, set()).add(row.get("sequence", ""))
    output: list[dict[str, object]] = []
    contexts = sorted({(seed, fold, run_seed, round_id) for seed, fold, run_seed, _strategy, round_id in by_key_strategy_round})
    for seed, fold, run_seed, round_id in contexts:
        for a in strategies:
            for b in strategies:
                set_a = by_key_strategy_round.get((seed, fold, run_seed, a, round_id), set())
                set_b = by_key_strategy_round.get((seed, fold, run_seed, b, round_id), set())
                if set_a or set_b:
                    output.append(
                        {
                            "initial_label_count": seed,
                            "outer_fold_id": fold,
                            "run_seed": run_seed,
                            "round_id": round_id,
                            "strategy_a": a,
                            "strategy_b": b,
                            "overlap_count": len(set_a & set_b),
                            "jaccard": jaccard(set_a, set_b),
                        }
                    )
    return output


def _matrix_from_pair_rows(pair_rows: list[dict[str, object]], strategies: list[str], seed_filter: int | None) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {strategy: {} for strategy in strategies}
    for a in strategies:
        for b in strategies:
            values = [
                float(row["jaccard"])
                for row in pair_rows
                if row["strategy_a"] == a and row["strategy_b"] == b and (seed_filter is None or row["initial_label_count"] == seed_filter)
            ]
            matrix[a][b] = mean(values) or 0.0
    return matrix


def same_round_overlap(rows: list[dict[str, str]], strategies: list[str]) -> dict[tuple[str, str], float]:
    by_key_strategy_round: dict[tuple[int, str, str, str, str], set[str]] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        key = (seed_value(row), row.get("outer_fold_id", ""), row.get("run_seed", ""), row.get("strategy", ""), row.get("round_id", ""))
        by_key_strategy_round.setdefault(key, set()).add(row.get("sequence", ""))
    values: dict[tuple[str, str], list[float]] = {}
    for seed, fold, run_seed, strategy, round_id in list(by_key_strategy_round):
        for a in strategies:
            for b in strategies:
                set_a = by_key_strategy_round.get((seed, fold, run_seed, a, round_id), set())
                set_b = by_key_strategy_round.get((seed, fold, run_seed, b, round_id), set())
                if set_a or set_b:
                    values.setdefault((a, b), []).append(jaccard(set_a, set_b))
    return {key: mean(items) or 0.0 for key, items in values.items()}


def acquisition_similarity(rows: list[dict[str, str]], strategies: list[str], limitations: list[str]) -> dict[str, object]:
    if not rows:
        limitations.append("Acquisition-ranking similarity skipped because per_run_acquisition_log.csv is missing or empty.")
        return {"available": False, "pairwise": []}
    if not any(row.get("acquisition_score") for row in rows):
        limitations.append("Acquisition-ranking similarity skipped because acquisition_score is not available.")
        return {"available": False, "pairwise": []}
    limitations.append("Raw acquisition score correlation was skipped because acquisition scores are not comparable across all strategy families.")
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("setup") != "ensemble_calibrated":
            continue
        key = (row.get("outer_fold_id", ""), row.get("run_seed", ""), str(seed_value(row)), row.get("round_id", ""), row.get("strategy", ""))
        grouped.setdefault(key, []).append(row)
    pair_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    contexts = sorted({key[:4] for key in grouped})
    for context in contexts:
        for a in strategies:
            rows_a = grouped.get((*context, a), [])
            scores_a = {row.get("sequence", ""): ffloat(row.get("acquisition_score")) for row in rows_a}
            top_a = {row.get("sequence", "") for row in sorted(rows_a, key=lambda r: ffloat(r.get("acquisition_score"), -999) or -999, reverse=True)[:5]}
            for b in strategies:
                rows_b = grouped.get((*context, b), [])
                scores_b = {row.get("sequence", ""): ffloat(row.get("acquisition_score")) for row in rows_b}
                common = [seq for seq in scores_a if seq in scores_b and scores_a[seq] is not None and scores_b[seq] is not None]
                if len(common) >= 3:
                    corr = spearman([scores_a[seq] for seq in common], [scores_b[seq] for seq in common])
                    if corr is not None:
                        pair_values.setdefault((a, b), {"spearman": [], "topk": []})["spearman"].append(corr)
                top_b = {row.get("sequence", "") for row in sorted(rows_b, key=lambda r: ffloat(r.get("acquisition_score"), -999) or -999, reverse=True)[:5]}
                if top_a or top_b:
                    pair_values.setdefault((a, b), {"spearman": [], "topk": []})["topk"].append(jaccard(top_a, top_b))
    pairwise = [
        {
            "strategy_a": a,
            "strategy_b": b,
            "mean_spearman_rank_correlation": mean(payload["spearman"]),
            "mean_top5_overlap": mean(payload["topk"]),
        }
        for (a, b), payload in sorted(pair_values.items())
    ]
    return {"available": True, "pairwise": pairwise}


def diversity_rows(selected_rows: list[dict[str, str]], acquisition_rows: list[dict[str, str]], strategies: list[str]) -> list[dict[str, object]]:
    distance_lookup: dict[tuple[int, str, str], list[float]] = {}
    for row in acquisition_rows:
        if row.get("setup") == "ensemble_calibrated" and row.get("selected", "").lower() in {"1", "true", "yes"}:
            distance = ffloat(row.get("distance_to_labeled"))
            if distance is not None:
                distance_lookup.setdefault((seed_value(row), row.get("strategy", ""), row.get("outer_fold_id", "")), []).append(distance)
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in selected_rows:
        if row.get("setup") == "ensemble_calibrated":
            grouped.setdefault((seed_value(row), row.get("strategy", "")), []).append(row)
    output: list[dict[str, object]] = []
    for seed in sorted({key[0] for key in grouped}):
        for strategy in strategies:
            rows = grouped.get((seed, strategy), [])
            sequences = [row.get("sequence", "") for row in rows]
            labels = [row.get("label", "") for row in rows]
            lengths = [len(seq) for seq in sequences if seq]
            near_duplicates = 0
            unique = list(dict.fromkeys(seq for seq in sequences if seq))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    if normalized_edit_distance(unique[i], unique[j]) <= 0.20:
                        near_duplicates += 1
            distances = [
                distance
                for (distance_seed, distance_strategy, _fold), vals in distance_lookup.items()
                if distance_seed == seed and distance_strategy == strategy
                for distance in vals
            ]
            positives = sum(1 for label in labels if str(label) == "1")
            output.append(
                {
                    "initial_label_count": seed,
                    "strategy": strategy,
                    "selected_rows": len(rows),
                    "unique_selected_sequences": len(set(sequences)),
                    "duplicate_count": max(0, len(sequences) - len(set(sequences))),
                    "near_duplicate_pair_count": near_duplicates,
                    "mean_pairwise_normalized_edit_distance": mean_pairwise_normalized_edit_distance(sequences),
                    "mean_length": mean([float(length) for length in lengths]),
                    "min_length": min(lengths) if lengths else "",
                    "max_length": max(lengths) if lengths else "",
                    "mean_distance_to_labeled": mean(distances),
                    "positive_rate_among_selected": positives / len(labels) if labels else None,
                    "cumulative_positives_discovered": positives,
                }
            )
    return output


def build_strategy_evidence(
    strategies: list[str],
    summary_lookup: dict[tuple[str, int, str], dict[str, str]],
    auc_summary: dict[tuple[str, int, str], dict[str, float | None]],
    rank_rows: list[dict[str, object]],
    labels_lookup: dict[tuple[str, int, str, str], float],
    label_rows: list[dict[str, object]],
    paired_rows: list[dict[str, str]],
    diversity: list[dict[str, object]],
    overlap_matrices: dict[int | str, dict[str, dict[str, float]]],
    exclude: set[str],
) -> dict[str, StrategyEvidence]:
    evidence: dict[str, StrategyEvidence] = {}
    seeds = sorted({seed for _, seed, _ in summary_lookup} | {int(row["initial_label_count"]) for row in rank_rows})
    for strategy in strategies:
        cost, note = PRACTICAL_COST.get(strategy, ("unknown", "not documented"))
        item = StrategyEvidence(
            strategy=strategy,
            role=role_for_strategy(strategy),
            practical_cost=cost,
            practical_note=note,
            is_baseline=strategy in exclude,
        )
        for seed in seeds:
            holdout_summary = summary_lookup.get((strategy, seed, "holdout"), {})
            validation_summary = summary_lookup.get((strategy, seed, "validation"), {})
            holdout_auc = auc_summary.get((strategy, seed, "holdout"), {})
            validation_auc = auc_summary.get((strategy, seed, "validation"), {})
            seed_rank_rows = [row for row in rank_rows if row["strategy"] == strategy and row["initial_label_count"] == seed]
            ranks = [float(row["rank_by_holdout_AULC_labeled"]) for row in seed_rank_rows]
            paired = [row for row in paired_rows if row.get("strategy") == strategy and iint(row.get("initial_label_count")) == seed and row.get("evaluation_dataset") == "holdout"]
            wins = sum(1 for row in paired if str(row.get("win_vs_random", "")).lower() in {"1", "true", "yes"})
            div = next((row for row in diversity if row["strategy"] == strategy and row["initial_label_count"] == seed), {})
            item.seed_metrics[seed] = {
                "holdout_mean_AULC_F1_by_labeled_count": ffloat(holdout_auc.get("mean_AULC_F1_by_labeled_count"), ffloat(holdout_summary.get("mean_AULC_F1"))),
                "holdout_mean_AULC_by_round": ffloat(holdout_auc.get("mean_AULC_by_round")),
                "validation_mean_AULC_F1_by_labeled_count": ffloat(validation_auc.get("mean_AULC_F1_by_labeled_count"), ffloat(validation_summary.get("mean_AULC_F1"))),
                "validation_mean_AULC_by_round": ffloat(validation_auc.get("mean_AULC_by_round")),
                "holdout_mean_final_F1": ffloat(holdout_auc.get("mean_final_F1"), ffloat(holdout_summary.get("mean_final_F1"))),
                "validation_mean_final_F1": ffloat(validation_auc.get("mean_final_F1"), ffloat(validation_summary.get("mean_final_F1"))),
                "holdout_mean_final_PR_AUC": ffloat(holdout_auc.get("mean_final_PR_AUC"), ffloat(holdout_summary.get("mean_final_PR_AUC"))),
                "holdout_mean_final_ROC_AUC": ffloat(holdout_auc.get("mean_final_ROC_AUC"), ffloat(holdout_summary.get("mean_final_ROC_AUC"))),
                "holdout_mean_final_Brier": ffloat(holdout_auc.get("mean_final_Brier"), ffloat(holdout_summary.get("mean_final_Brier"))),
                "holdout_mean_final_ECE_10": ffloat(holdout_auc.get("mean_final_ECE_10"), ffloat(holdout_summary.get("mean_final_ECE_10"))),
                "labels_to_f1_080": labels_lookup.get((strategy, seed, "holdout", "0.8")),
                "labels_to_f1_084": labels_lookup.get((strategy, seed, "holdout", "0.84")),
                "labels_to_f1_086": labels_lookup.get((strategy, seed, "holdout", "0.86")),
                "win_count_vs_random": wins,
                "mean_rank": mean(ranks),
                "rank_std": stdev(ranks),
                "worst_fold_rank": max(ranks) if ranks else None,
                "worst_fold_AULC_F1_by_labeled_count": min([ffloat(row.get("AULC_F1_by_labeled_count")) for row in seed_rank_rows if ffloat(row.get("AULC_F1_by_labeled_count")) is not None], default=None),
                "mean_pairwise_normalized_edit_distance": ffloat(div.get("mean_pairwise_normalized_edit_distance")),
                "positive_rate_among_selected": ffloat(div.get("positive_rate_among_selected")),
                "cumulative_positives_discovered": ffloat(div.get("cumulative_positives_discovered")),
                "near_duplicate_pair_count": ffloat(div.get("near_duplicate_pair_count")),
            }
        all_holdout = [ffloat(metrics.get("holdout_mean_AULC_F1_by_labeled_count")) for metrics in item.seed_metrics.values()]
        all_holdout_round = [ffloat(metrics.get("holdout_mean_AULC_by_round")) for metrics in item.seed_metrics.values()]
        all_labels_084 = [ffloat(metrics.get("labels_to_f1_084")) for metrics in item.seed_metrics.values()]
        all_labels = [ffloat(metrics.get("labels_to_f1_086")) for metrics in item.seed_metrics.values()]
        all_ranks = [ffloat(metrics.get("mean_rank")) for metrics in item.seed_metrics.values()]
        all_diversity = [ffloat(metrics.get("mean_pairwise_normalized_edit_distance")) for metrics in item.seed_metrics.values()]
        others = [s for s in strategies if s != strategy and s not in exclude]
        combined_overlap = mean([overlap_matrices.get("combined", {}).get(strategy, {}).get(other, 0.0) for other in others])
        item.combined = {
            "mean_holdout_AULC_F1_by_labeled_count": mean([v for v in all_holdout if v is not None]),
            "mean_holdout_AULC_by_round": mean([v for v in all_holdout_round if v is not None]),
            "mean_labels_to_f1_084": mean([v for v in all_labels_084 if v is not None]),
            "mean_labels_to_f1_086": mean([v for v in all_labels if v is not None]),
            "mean_rank": mean([v for v in all_ranks if v is not None]),
            "mean_diversity": mean([v for v in all_diversity if v is not None]),
            "mean_overlap_to_nonrandom": combined_overlap,
            "seed_size_consistency_delta_AULC": abs((all_holdout[0] or 0) - (all_holdout[-1] or 0)) if len(all_holdout) >= 2 and all_holdout[0] is not None and all_holdout[-1] is not None else None,
        }
        evidence[strategy] = item
    return evidence


def percentile_scores(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    usable = {key: value for key, value in values.items() if value is not None}
    if not usable:
        return {}
    min_value = min(usable.values())
    max_value = max(usable.values())
    if max_value == min_value:
        return {key: 1.0 for key in usable}
    result = {}
    for key, value in usable.items():
        score = (value - min_value) / (max_value - min_value)
        result[key] = score if higher_is_better else 1.0 - score
    return result


def score_evidence(evidence: dict[str, StrategyEvidence], exclude: set[str]) -> None:
    strategies = [strategy for strategy in evidence if strategy not in exclude]
    aulc = percentile_scores({s: ffloat(evidence[s].combined.get("mean_holdout_AULC_F1_by_labeled_count")) for s in strategies}, higher_is_better=True)
    labels = percentile_scores({s: ffloat(evidence[s].combined.get("mean_labels_to_f1_086")) for s in strategies}, higher_is_better=False)
    ranks = percentile_scores({s: ffloat(evidence[s].combined.get("mean_rank")) for s in strategies}, higher_is_better=False)
    diversity = percentile_scores({s: ffloat(evidence[s].combined.get("mean_diversity")) for s in strategies}, higher_is_better=True)
    overlap = percentile_scores({s: ffloat(evidence[s].combined.get("mean_overlap_to_nonrandom")) for s in strategies}, higher_is_better=False)
    for strategy in strategies:
        role_bonus = 0.04 if evidence[strategy].role in {"model/committee uncertainty", "diversity/novelty"} else 0.0
        composite = (
            0.35 * aulc.get(strategy, 0.0)
            + 0.25 * labels.get(strategy, 0.0)
            + 0.15 * ranks.get(strategy, 0.0)
            + 0.15 * diversity.get(strategy, 0.0)
            + 0.10 * overlap.get(strategy, 0.0)
            + role_bonus
        )
        evidence[strategy].combined.update(
            {
                "score_holdout_AULC": aulc.get(strategy, 0.0),
                "score_labels_to_target": labels.get(strategy, 0.0),
                "score_rank": ranks.get(strategy, 0.0),
                "score_diversity": diversity.get(strategy, 0.0),
                "score_nonredundancy": overlap.get(strategy, 0.0),
                "composite_score": composite,
            }
        )


def apply_random_floor(evidence: dict[str, StrategyEvidence], exclude: set[str], tolerance: float = RANDOM_AULC_FLOOR_TOLERANCE) -> None:
    random_item = evidence.get("random")
    random_aulc = ffloat(random_item.combined.get("mean_holdout_AULC_F1_by_labeled_count")) if random_item else None
    for strategy, item in evidence.items():
        strategy_aulc = ffloat(item.combined.get("mean_holdout_AULC_F1_by_labeled_count"))
        delta = strategy_aulc - random_aulc if strategy_aulc is not None and random_aulc is not None else None
        status = "baseline/control" if strategy in exclude else random_floor_status(delta, tolerance=tolerance)
        passes_floor = strategy not in exclude and delta is not None and delta >= -tolerance
        if strategy in exclude:
            eligibility = "baseline/control"
        elif passes_floor:
            eligibility = "eligible"
        else:
            eligibility = "exploratory/control"
        item.combined.update(
            {
                "random_floor_AULC": random_aulc,
                "delta_vs_random_AULC": delta,
                "random_floor_status": status,
                "passes_random_AULC_floor": "yes" if passes_floor else "no",
                "recommendation_eligibility": eligibility,
            }
        )


def too_redundant(candidate: str, selected: list[str], overlap_matrix: dict[str, dict[str, float]], threshold: float) -> bool:
    return any(overlap_matrix.get(candidate, {}).get(strategy, 0.0) >= threshold for strategy in selected)


def select_recommendations(evidence: dict[str, StrategyEvidence], exclude: set[str], overlap_matrix: dict[str, dict[str, float]], threshold: float, top_k: int) -> tuple[list[str], str]:
    eligible = [
        strategy
        for strategy in evidence
        if strategy not in exclude and evidence[strategy].combined.get("recommendation_eligibility") == "eligible"
    ]
    ranked = sorted(eligible, key=lambda s: ffloat(evidence[s].combined.get("composite_score"), -999) or -999, reverse=True)
    selected: list[str] = []
    if not eligible:
        return [], ""
    best = max(eligible, key=lambda s: ffloat(evidence[s].combined.get("mean_holdout_AULC_F1_by_labeled_count"), -999) or -999)
    selected.append(best)
    uncertainty = [s for s in ranked if s in ROLE_UNCERTAINTY and s not in selected]
    if uncertainty:
        selected.append(uncertainty[0])
    diversity = [s for s in ranked if s in ROLE_DIVERSITY and s not in selected]
    for candidate in diversity:
        if not too_redundant(candidate, selected, overlap_matrix, threshold):
            selected.append(candidate)
            break
    if len(selected) < 3 and diversity:
        selected.append(diversity[0])
    for candidate in ranked:
        if len(selected) >= top_k:
            break
        if candidate not in selected and not too_redundant(candidate, selected, overlap_matrix, threshold):
            selected.append(candidate)
    for candidate in ranked:
        if len(selected) >= top_k:
            break
        if candidate not in selected:
            selected.append(candidate)
    backup = next((candidate for candidate in ranked if candidate not in selected), "")
    return selected[:top_k], backup


def recommendation_role(strategy: str, selected: list[str]) -> str:
    if selected and strategy == selected[0]:
        return "best replay/sample efficiency"
    if strategy in ROLE_UNCERTAINTY:
        return "model/committee uncertainty"
    if strategy in ROLE_DIVERSITY:
        return "diversity/novelty for CG-MD slate"
    return role_for_strategy(strategy)


def matrix_to_csv_rows(matrix: dict[str, dict[str, float]], strategies: list[str]) -> list[dict[str, object]]:
    rows = []
    for strategy in strategies:
        row = {"strategy": strategy}
        row.update({other: matrix.get(strategy, {}).get(other, 0.0) for other in strategies})
        rows.append(row)
    return rows


def write_png_heatmap(path: Path, matrix: dict[str, dict[str, float]], strategies: list[str]) -> None:
    font = {
        " ": ["000", "000", "000", "000", "000", "000", "000"],
        ".": ["0", "0", "0", "0", "0", "0", "1"],
        "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
        "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    }
    n = max(1, len(strategies))
    cell = 92
    left = 430
    top = 380
    right = 190
    bottom = 120
    width = left + n * cell + right
    height = top + n * cell + bottom
    image = bytearray([255] * width * height * 3)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                set_pixel(xx, yy, color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        if x0 == x1:
            fill_rect(x0, min(y0, y1), x0 + 2, max(y0, y1) + 1, color)
        elif y0 == y1:
            fill_rect(min(x0, x1), y0, max(x0, x1) + 1, y0 + 2, color)

    def color_for(value: float) -> tuple[int, int, int]:
        value = max(0.0, min(1.0, value))
        return (int(235 - 105 * value), int(246 - 160 * value), int(255 - 25 * value))

    def draw_text(text: str, x: int, y: int, scale: int = 2, color: tuple[int, int, int] = (20, 29, 45), rotate_ccw: bool = False) -> None:
        cursor = 0
        for char in text.upper():
            glyph = font.get(char, font[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit != "1":
                        continue
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cursor + gx * scale + sx
                            py = gy * scale + sy
                            if rotate_ccw:
                                set_pixel(x + py, y - px, color)
                            else:
                                set_pixel(x + px, y + py, color)
            cursor += 6 * scale

    fill_rect(0, 0, width, height, (255, 255, 255))
    draw_text("SELECTED-PEPTIDE JACCARD OVERLAP", 28, 26, scale=3)
    draw_text("COLUMNS AND ROWS ARE ACQUISITION STRATEGIES", 28, 74, scale=2, color=(70, 83, 105))
    for row_index, strategy_y in enumerate(strategies):
        y0 = top + row_index * cell
        label = strategy_y.upper()
        label_width = len(label) * 12
        draw_text(label, max(12, left - label_width - 18), y0 + cell // 2 - 8, scale=2)
        for col_index, strategy_x in enumerate(strategies):
            x0 = left + col_index * cell
            value = max(0.0, min(1.0, matrix.get(strategy_y, {}).get(strategy_x, 0.0)))
            fill_rect(x0, y0, x0 + cell, y0 + cell, color_for(value))
            text = f"{value:.2f}"
            draw_text(text, x0 + 20, y0 + cell // 2 - 8, scale=2, color=(0, 0, 0) if value < 0.62 else (255, 255, 255))
            line(x0, y0, x0 + cell, y0, (226, 232, 240))
            line(x0, y0, x0, y0 + cell, (226, 232, 240))
    line(left, top + n * cell, left + n * cell, top + n * cell, (148, 163, 184))
    line(left + n * cell, top, left + n * cell, top + n * cell, (148, 163, 184))
    for col_index, strategy_x in enumerate(strategies):
        x0 = left + col_index * cell + cell // 2 - 8
        draw_text(strategy_x.upper(), x0, top - 18, scale=2, rotate_ccw=True)

    colorbar_x = left + n * cell + 52
    colorbar_y = top
    colorbar_w = 34
    colorbar_h = n * cell
    for yy in range(colorbar_h):
        value = 1.0 - yy / max(1, colorbar_h - 1)
        fill_rect(colorbar_x, colorbar_y + yy, colorbar_x + colorbar_w, colorbar_y + yy + 1, color_for(value))
    line(colorbar_x, colorbar_y, colorbar_x + colorbar_w, colorbar_y, (71, 85, 105))
    line(colorbar_x, colorbar_y + colorbar_h, colorbar_x + colorbar_w, colorbar_y + colorbar_h, (71, 85, 105))
    line(colorbar_x, colorbar_y, colorbar_x, colorbar_y + colorbar_h, (71, 85, 105))
    line(colorbar_x + colorbar_w, colorbar_y, colorbar_x + colorbar_w, colorbar_y + colorbar_h, (71, 85, 105))
    draw_text("JACCARD", colorbar_x - 12, colorbar_y - 18, scale=2)
    for tick in (0.0, 0.5, 1.0):
        yy = colorbar_y + int((1.0 - tick) * colorbar_h)
        line(colorbar_x + colorbar_w + 3, yy, colorbar_x + colorbar_w + 12, yy, (15, 23, 42))
        draw_text(f"{tick:.1f}", colorbar_x + colorbar_w + 18, yy - 8, scale=2)

    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(bytes([0]) + bytes(image[y * stride : (y + 1) * stride]))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def build_summary_rows(evidence: dict[str, StrategyEvidence], selected: list[str], backup: str, overlap_matrix: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy, item in evidence.items():
        decision = "baseline/control" if strategy == "random" else "rejected"
        if item.combined.get("recommendation_eligibility") == "exploratory/control":
            decision = "exploratory/control"
        if strategy in selected:
            decision = "recommended"
        elif strategy == backup:
            decision = "backup"
        row: dict[str, object] = {
            "strategy": strategy,
            "role": recommendation_role(strategy, selected) if strategy in selected else item.role,
            "decision": decision,
            "is_baseline": "yes" if item.is_baseline else "no",
            "practical_cost": item.practical_cost,
            "practical_note": item.practical_note,
            **item.combined,
            "max_overlap_with_recommended": max([overlap_matrix.get(strategy, {}).get(other, 0.0) for other in selected if other != strategy] or [0.0]),
        }
        for seed, metrics in sorted(item.seed_metrics.items()):
            for key, value in metrics.items():
                row[f"seed_{seed}_{key}"] = value
        rows.append(row)
    return sorted(rows, key=lambda row: ffloat(row.get("composite_score"), -1) or -1, reverse=True)


def evidence_sentence(item: StrategyEvidence) -> str:
    return (
        f"holdout AULC by labeled count={fmt(item.combined.get('mean_holdout_AULC_F1_by_labeled_count'))}, "
        f"labels to F1 0.86={fmt(item.combined.get('mean_labels_to_f1_086'), 1)}, "
        f"mean rank={fmt(item.combined.get('mean_rank'), 2)}"
    )


def diversity_sentence(item: StrategyEvidence) -> str:
    return (
        f"mean edit distance={fmt(item.combined.get('mean_diversity'))}, "
        f"mean overlap={fmt(item.combined.get('mean_overlap_to_nonrandom'))}"
    )


def trio_comparison_rows(
    evidence: dict[str, StrategyEvidence],
    overlap_matrix: dict[str, dict[str, float]],
    min_overlap_warning: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trio_name, trio in TRIO_DEFINITIONS.items():
        pairwise = [
            overlap_matrix.get(left, {}).get(right, 0.0)
            for i, left in enumerate(trio)
            for right in trio[i + 1 :]
        ]
        aulcs = [ffloat(evidence[strategy].combined.get("mean_holdout_AULC_F1_by_labeled_count")) for strategy in trio]
        ranks = [ffloat(evidence[strategy].combined.get("mean_rank")) for strategy in trio]
        labels_084 = [ffloat(evidence[strategy].combined.get("mean_labels_to_f1_084")) for strategy in trio]
        labels_086 = [ffloat(evidence[strategy].combined.get("mean_labels_to_f1_086")) for strategy in trio]
        below = [
            strategy
            for strategy in trio
            if evidence[strategy].combined.get("random_floor_status") == "below random"
        ]
        tolerance_only = [
            strategy
            for strategy in trio
            if evidence[strategy].combined.get("random_floor_status") == "passes floor by tolerance"
        ]
        max_jaccard = max(pairwise) if pairwise else None
        if below:
            interpretation = (
                "Not preferred as a main Real AL trio because "
                + ", ".join(f"`{strategy}`" for strategy in below)
                + " is below random on combined holdout AULC; treat that member as exploratory/control."
            )
        elif tolerance_only:
            interpretation = (
                "Passes the random floor only by numerical tolerance for "
                + ", ".join(f"`{strategy}`" for strategy in tolerance_only)
                + "; acceptable only with cautious wording."
            )
        elif max_jaccard is not None and max_jaccard >= min_overlap_warning:
            interpretation = "Scientifically defensible: all members clear the random AULC floor, but the overlap warning should be stated explicitly."
        else:
            interpretation = "Scientifically defensible: all members clear the random AULC floor and pairwise overlap stays below the warning threshold."
        rows.append(
            {
                "trio": trio_name,
                "strategies": ";".join(trio),
                "mean_combined_holdout_AULC_F1": mean([value for value in aulcs if value is not None]),
                "minimum_selected_strategy_holdout_AULC_F1": min([value for value in aulcs if value is not None], default=None),
                "mean_rank": mean([value for value in ranks if value is not None]),
                "worst_rank": max([value for value in ranks if value is not None], default=None),
                "max_pairwise_jaccard": max_jaccard,
                "mean_pairwise_jaccard": mean(pairwise),
                "mean_labels_to_F1_0_84": mean([value for value in labels_084 if value is not None]),
                "mean_labels_to_F1_0_86": mean([value for value in labels_086 if value is not None]),
                "below_random_members": ";".join(below),
                "interpretation": interpretation,
            }
        )
    return rows


def write_markdown(
    path: Path,
    evidence: dict[str, StrategyEvidence],
    selected: list[str],
    backup: str,
    overlap_matrix: dict[str, dict[str, float]],
    summary_rows: list[dict[str, object]],
    trio_rows: list[dict[str, object]],
    limitations: list[str],
    min_overlap_warning: float,
) -> None:
    lines: list[str] = []
    nonbaseline_rows = [row for row in summary_rows if str(row.get("strategy")) != "random"]
    best_replay = max(
        nonbaseline_rows,
        key=lambda row: ffloat(row.get("mean_holdout_AULC_F1_by_labeled_count"), -999.0) or -999.0,
        default={},
    )
    most_diverse = max(nonbaseline_rows, key=lambda row: ffloat(row.get("mean_diversity"), -999.0) or -999.0, default={})
    best_seed_10 = max(
        nonbaseline_rows,
        key=lambda row: ffloat(row.get("seed_10_holdout_mean_AULC_F1_by_labeled_count"), -999.0) or -999.0,
        default={},
    )
    best_seed_40 = max(
        nonbaseline_rows,
        key=lambda row: ffloat(row.get("seed_40_holdout_mean_AULC_F1_by_labeled_count"), -999.0) or -999.0,
        default={},
    )
    robust_rows = sorted(
        [row for row in nonbaseline_rows if ffloat(row.get("seed_size_consistency_delta_AULC")) is not None],
        key=lambda row: ffloat(row.get("seed_size_consistency_delta_AULC"), 999.0) or 999.0,
    )[:3]
    lines.append("# Phase 3 Real AL Strategy Recommendation")
    lines.append("")
    lines.append("Recommended Phase 3 Real AL strategies:")
    for index, strategy in enumerate(selected, start=1):
        lines.append(f"{index}. `{strategy}` - role: {recommendation_role(strategy, selected)}")
    lines.append("")
    lines.append("Backup:")
    lines.append(f"- `{backup}`" if backup else "- not available")
    lines.append("")
    lines.append("## Final human recommendation")
    lines.append("")
    chosen = ", ".join(f"`{strategy}`" for strategy in selected)
    lines.append(f"Use {chosen} for Phase 3 Real AL.")
    lines.append("")
    for strategy in selected:
        item = evidence[strategy]
        lines.append(f"`{strategy}` is included because it fills the {recommendation_role(strategy, selected)} role with {evidence_sentence(item)} and {diversity_sentence(item)}.")
    if backup:
        lines.append("")
        lines.append(f"Backup: `{backup}`, use it if one selected strategy becomes impractical or its candidate slate overlaps too strongly with another strategy in the generated pool.")
    lines.append("")
    lines.append("## Raw Evidence Highlights")
    lines.append("")
    if best_replay:
        lines.append(
            f"- Best replay performer by combined holdout AULC over labeled peptides: `{best_replay['strategy']}` "
            f"({fmt(best_replay.get('mean_holdout_AULC_F1_by_labeled_count'))})."
        )
    if best_seed_10:
        lines.append(
            f"- Best at seed size 10 by holdout AULC over labeled peptides: `{best_seed_10['strategy']}` "
            f"({fmt(best_seed_10.get('seed_10_holdout_mean_AULC_F1_by_labeled_count'))})."
        )
    if best_seed_40:
        lines.append(
            f"- Best at seed size 40 by holdout AULC over labeled peptides: `{best_seed_40['strategy']}` "
            f"({fmt(best_seed_40.get('seed_40_holdout_mean_AULC_F1_by_labeled_count'))})."
        )
    if robust_rows:
        robust_text = ", ".join(
            f"`{row['strategy']}` (delta {fmt(row.get('seed_size_consistency_delta_AULC'))})" for row in robust_rows
        )
        lines.append(f"- Most consistent across seed sizes by AULC delta: {robust_text}.")
    if most_diverse:
        lines.append(
            f"- Highest sequence diversity among selected peptides: `{most_diverse['strategy']}` "
            f"(mean normalized edit distance {fmt(most_diverse.get('mean_diversity'))})."
        )
    lines.append("")
    lines.append("## Performance Floor vs Random")
    lines.append("")
    random_row = next((row for row in summary_rows if row.get("strategy") == "random"), {})
    if random_row:
        lines.append(f"Random combined holdout AULC-F1 by labeled count: {fmt(random_row.get('mean_holdout_AULC_F1_by_labeled_count'), 6)}.")
        lines.append("")
    lines.append("| Strategy | Combined holdout AULC-F1 | Delta vs random | Floor status | Eligibility | Decision |")
    lines.append("|---|---:|---:|---|---|---|")
    for row in summary_rows:
        strategy = str(row.get("strategy", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{strategy}`",
                    fmt(row.get("mean_holdout_AULC_F1_by_labeled_count"), 6),
                    fmt(row.get("delta_vs_random_AULC"), 6),
                    str(row.get("random_floor_status", "")),
                    str(row.get("recommendation_eligibility", "")),
                    str(row.get("decision", "")),
                ]
            )
            + " |"
        )
    exploratory = [row for row in summary_rows if row.get("decision") == "exploratory/control"]
    if exploratory:
        moved = ", ".join(f"`{row['strategy']}`" for row in exploratory)
        lines.append("")
        lines.append(f"{moved} did not enter the main recommendation because below-random strategies are treated as exploratory/control by default.")
    if "predictive_entropy" in selected and "family_qbc" in selected:
        overlap = overlap_matrix.get("predictive_entropy", {}).get("family_qbc", 0.0)
        lines.append("")
        lines.append(f"`predictive_entropy` and `family_qbc` have {overlap_band(overlap)} selected-peptide overlap (Jaccard {overlap:.2f}), but both clear the random AULC floor and represent different uncertainty mechanisms.")
    lines.append("")
    lines.append("## Decision Table")
    lines.append("")
    lines.append("| Strategy | Role | Replay evidence | Overlap warning | Diversity evidence | Practical cost | Decision |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in summary_rows:
        strategy = str(row["strategy"])
        max_overlap = ffloat(row.get("max_overlap_with_recommended"), 0.0) or 0.0
        warning = overlap_band(max_overlap)
        if max_overlap >= min_overlap_warning and strategy not in selected:
            warning += f" with recommended set (max Jaccard {max_overlap:.2f})"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{strategy}`",
                    str(row.get("role", "")),
                    evidence_sentence(evidence[strategy]),
                    warning,
                    diversity_sentence(evidence[strategy]),
                    str(row.get("practical_cost", "")),
                    str(row.get("decision", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Trio Comparison")
    lines.append("")
    lines.append("| Trio | Strategies | Mean AULC-F1 | Min AULC-F1 | Mean rank | Worst rank | Max Jaccard | Mean Jaccard | Labels to F1 0.84 | Labels to F1 0.86 | Interpretation |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in trio_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("trio", "")),
                    "`" + str(row.get("strategies", "")).replace(";", "`, `") + "`",
                    fmt(row.get("mean_combined_holdout_AULC_F1"), 6),
                    fmt(row.get("minimum_selected_strategy_holdout_AULC_F1"), 6),
                    fmt(row.get("mean_rank"), 3),
                    fmt(row.get("worst_rank"), 3),
                    fmt(row.get("max_pairwise_jaccard"), 3),
                    fmt(row.get("mean_pairwise_jaccard"), 3),
                    fmt(row.get("mean_labels_to_F1_0_84"), 1),
                    fmt(row.get("mean_labels_to_F1_0_86"), 1),
                    str(row.get("interpretation", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Overlap Warnings")
    lines.append("")
    warned = False
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            value = overlap_matrix.get(left, {}).get(right, 0.0)
            if value >= min_overlap_warning:
                warned = True
                lines.append(f"- `{left}` and `{right}` have {overlap_band(value)} selected-peptide overlap (Jaccard {value:.2f}); keeping both requires the role-diversity justification above.")
    if not warned:
        lines.append("- No recommended pair exceeded the configured overlap warning threshold.")
    lines.append("")
    rejected = [row for row in summary_rows if row.get("decision") in {"rejected", "backup", "exploratory/control"}]
    lines.append("## Rejected And Backup Strategies")
    lines.append("")
    for row in rejected:
        strategy = str(row.get("strategy", ""))
        decision = str(row.get("decision", ""))
        max_overlap = ffloat(row.get("max_overlap_with_recommended"), 0.0) or 0.0
        if decision == "exploratory/control":
            reason = f"kept only as exploratory/control because its combined holdout AULC is below random (delta {fmt(row.get('delta_vs_random_AULC'), 6)})"
        elif decision == "backup":
            reason = "kept as backup because its replay evidence is strong, but it is not needed in the first three-role slate"
        elif max_overlap >= min_overlap_warning:
            reason = f"rejected mainly because selected peptides overlap the recommended set ({overlap_band(max_overlap)}, max Jaccard {max_overlap:.2f})"
        else:
            reason = "rejected because another strategy covered the same role with stronger replay/sample-efficiency or diversity evidence"
        lines.append(f"- `{strategy}`: {reason}.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for limitation in limitations or ["No additional limitations recorded."]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("Thesis-style summary: Phase 3 uses a role-diverse Real AL slate rather than the top three strategies by mean AULC alone. The recommended set balances replay performance, sample efficiency, robustness across seed sizes, selected-sequence non-redundancy, and practical feasibility for generated peptide CG-MD validation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3_strategy_selection(options: Phase3SelectionOptions | argparse.Namespace) -> dict[str, object]:
    if isinstance(options, argparse.Namespace):
        options = Phase3SelectionOptions(
            phase2_root=Path(options.phase2_root),
            output_root=Path(options.output_root),
            top_k=int(options.top_k),
            exclude=tuple(options.exclude or ()),
            min_overlap_warning=float(options.min_overlap_warning),
        )
    phase2_root = Path(options.phase2_root)
    output_root = Path(options.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = discover_phase2_inputs(phase2_root)
    strategies = list(THESIS_FULL_REPLAY_STRATEGIES)
    exclude = set(options.exclude)
    limitations: list[str] = []

    summary_rows_raw = read_csv_rows(inputs["strategy_summary"])
    labels_rows_raw = read_csv_rows(inputs["labels_to_target"])
    paired_rows_raw = read_csv_rows(inputs["paired_vs_random"])
    selected_rows_raw = read_csv_rows(inputs["selected_sequences"])
    acquisition_rows_raw = read_csv_rows(inputs["acquisition_log"])
    round_metric_rows = read_csv_rows(inputs["round_metrics"]) or read_csv_rows(inputs["learning_curves"])
    if not round_metric_rows:
        limitations.append("Fold-level AULC by labeled count unavailable because neither per_run_round_metrics.csv nor learning_curves.csv was readable.")
    if not acquisition_rows_raw:
        limitations.append("Acquisition log unavailable; ranking similarity is limited to selected-peptide overlap.")

    summary_lookup = strategy_summary_by_seed(summary_rows_raw)
    auc_rows, _auc_lookup = compute_auc_rows(round_metric_rows)
    auc_summary = aggregate_auc_by_seed(auc_rows)
    rank_rows = compute_rank_rows(auc_rows)
    max_labeled_by_seed: dict[int, float] = {}
    for row in round_metric_rows:
        if row.get("evaluation_dataset") == "holdout":
            seed = seed_value(row)
            labeled = ffloat(row.get("labeled_count"))
            if labeled is not None:
                max_labeled_by_seed[seed] = max(max_labeled_by_seed.get(seed, labeled), labeled)
    label_rows, labels_lookup = labels_to_target_rows(labels_rows_raw, max_labeled_by_seed)
    pair_rows, matrices = compute_overlap(selected_rows_raw, strategies)
    same_round_rows = same_round_overlap_rows(selected_rows_raw, strategies)
    diversity = diversity_rows(selected_rows_raw, acquisition_rows_raw, strategies)
    acquisition = acquisition_similarity(acquisition_rows_raw, strategies, limitations)
    if not any("embedding" in key for row in acquisition_rows_raw[:1] for key in row):
        limitations.append("Embedding-based diversity was not computed because embeddings are not present in the Phase 2 logs.")

    evidence = build_strategy_evidence(
        strategies,
        summary_lookup,
        auc_summary,
        rank_rows,
        labels_lookup,
        label_rows,
        paired_rows_raw,
        diversity,
        matrices,
        exclude,
    )
    score_evidence(evidence, exclude)
    apply_random_floor(evidence, exclude)
    selected, backup = select_recommendations(evidence, exclude, matrices.get("combined", {}), options.min_overlap_warning, options.top_k)
    summary_rows = build_summary_rows(evidence, selected, backup, matrices.get("combined", {}))
    trio_rows = trio_comparison_rows(evidence, matrices.get("combined", {}), options.min_overlap_warning)

    outputs = {
        "strategy_selection_summary_csv": output_root / "strategy_selection_summary.csv",
        "strategy_selection_summary_json": output_root / "strategy_selection_summary.json",
        "strategy_overlap_matrix_csv": output_root / "strategy_overlap_matrix.csv",
        "strategy_overlap_matrix_json": output_root / "strategy_overlap_matrix.json",
        "strategy_jaccard_heatmap_png": output_root / "strategy_jaccard_heatmap.png",
        "strategy_performance_vs_diversity_csv": output_root / "strategy_performance_vs_diversity.csv",
        "strategy_trio_comparison_csv": output_root / "strategy_trio_comparison.csv",
        "strategy_rank_by_fold_csv": output_root / "strategy_rank_by_fold.csv",
        "strategy_labels_to_target_csv": output_root / "strategy_labels_to_target.csv",
        "strategy_positive_discovery_csv": output_root / "strategy_positive_discovery.csv",
        "recommendation_md": output_root / "real_al_strategy_recommendation.md",
    }
    write_csv_rows(outputs["strategy_selection_summary_csv"], summary_rows)
    write_json(outputs["strategy_selection_summary_json"], {"recommended": selected, "backup": backup, "summary_rows": summary_rows, "trio_comparison": trio_rows, "limitations": limitations, "acquisition_similarity": acquisition})
    combined_matrix = matrices.get("combined", {})
    write_csv_rows(outputs["strategy_overlap_matrix_csv"], matrix_to_csv_rows(combined_matrix, strategies), ["strategy", *strategies])
    write_json(
        outputs["strategy_overlap_matrix_json"],
        {
            "jaccard_matrices": matrices,
            "pairwise_overlap_rows": pair_rows,
            "same_round_overlap_rows": same_round_rows,
        },
    )
    write_png_heatmap(outputs["strategy_jaccard_heatmap_png"], combined_matrix, strategies)
    write_csv_rows(outputs["strategy_performance_vs_diversity_csv"], [
        {
            "strategy": row["strategy"],
            "decision": row["decision"],
            "mean_holdout_AULC_F1_by_labeled_count": row.get("mean_holdout_AULC_F1_by_labeled_count", ""),
            "mean_holdout_AULC_by_round": row.get("mean_holdout_AULC_by_round", ""),
            "delta_vs_random_AULC": row.get("delta_vs_random_AULC", ""),
            "random_floor_status": row.get("random_floor_status", ""),
            "recommendation_eligibility": row.get("recommendation_eligibility", ""),
            "mean_labels_to_f1_086": row.get("mean_labels_to_f1_086", ""),
            "mean_diversity": row.get("mean_diversity", ""),
            "mean_overlap_to_nonrandom": row.get("mean_overlap_to_nonrandom", ""),
            "composite_score": row.get("composite_score", ""),
        }
        for row in summary_rows
    ])
    write_csv_rows(outputs["strategy_trio_comparison_csv"], trio_rows)
    write_csv_rows(outputs["strategy_rank_by_fold_csv"], rank_rows)
    write_csv_rows(outputs["strategy_labels_to_target_csv"], label_rows)
    write_csv_rows(outputs["strategy_positive_discovery_csv"], diversity)
    write_markdown(outputs["recommendation_md"], evidence, selected, backup, combined_matrix, summary_rows, trio_rows, limitations, options.min_overlap_warning)

    return {
        "recommended_strategies": selected,
        "backup_strategy": backup,
        "output_root": str(output_root),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "limitations": limitations,
    }
