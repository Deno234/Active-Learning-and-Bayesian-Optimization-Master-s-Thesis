from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_thesis_results_handoff import (
    BRANCH_ORDER,
    CAMPAIGN_ROOTS,
    PHASE1,
    PHASE2,
    PHASE3,
    PHASE4,
    PHASE5,
    POLICY_ORDER,
    ROOT,
    phase4_campaign_rows,
)


REPORTING = ROOT / "thesis_reporting"
SCRIPT_NAME = "build_thesis_reporting_evidence.py"
COMMIT_SHA = "7cbfa95d8d56ce4b558238341ed5e19354ffcec2"
BRANCH = "codex/active-learning-thesis"
CREATED_AT = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

THESIS_TEX = Path(r"<local_thesis_dir>\main.tex")
THESIS_PDF = Path(r"<local_thesis_dir>\Denis_Ibiši_Master_s_thesis.pdf")
BIB = Path(r"<local_thesis_dir>\references_rsc.bib")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(value: object, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def i(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value: object, digits: int = 4) -> str:
    parsed = f(value)
    return "" if parsed is None or not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def trapezoid(points: list[tuple[float, float]], start: float | None = None, end: float | None = None) -> float:
    ordered = sorted(points)
    if start is not None:
        ordered = [point for point in ordered if point[0] >= start]
    if end is not None:
        ordered = [point for point in ordered if point[0] <= end]
    if len(ordered) < 2:
        return float("nan")
    area = sum(
        (ordered[index][1] + ordered[index + 1][1])
        * 0.5
        * (ordered[index + 1][0] - ordered[index][0])
        for index in range(len(ordered) - 1)
    )
    span = ordered[-1][0] - ordered[0][0]
    return area / span if span else float("nan")


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein(left: str, right: str) -> float:
    return levenshtein(left, right) / max(len(left), len(right), 1)


def diversity(sequences: list[str]) -> tuple[float, float]:
    unique = list(dict.fromkeys(sequences))
    values = [
        normalized_levenshtein(unique[a], unique[b])
        for a in range(len(unique))
        for b in range(a + 1, len(unique))
    ]
    return (mean(values), min(values)) if values else (0.0, 0.0)


PROVENANCE: list[dict[str, object]] = []


def register(
    output: Path,
    sources: list[Path],
    formula: str,
    filters: str,
    denominator: str,
) -> None:
    PROVENANCE.append(
        {
            "derived_file": rel(output),
            "source_files": "; ".join(rel(path) for path in sources),
            "aggregation_formula": formula,
            "filtering_rules": filters,
            "denominator": denominator,
            "creation_script": SCRIPT_NAME,
            "commit_sha": COMMIT_SHA,
            "created_at": CREATED_AT,
        }
    )


def thesis_metadata() -> dict[str, object]:
    metadata = {
        "thesis_source_reviewed": str(THESIS_TEX),
        "thesis_source_sha256": sha256(THESIS_TEX),
        "thesis_source_last_modified": datetime.fromtimestamp(THESIS_TEX.stat().st_mtime).astimezone().isoformat(),
        "compiled_pdf_reviewed": str(THESIS_PDF),
        "compiled_pdf_sha256": sha256(THESIS_PDF),
        "compiled_pdf_last_modified": datetime.fromtimestamp(THESIS_PDF.stat().st_mtime).astimezone().isoformat(),
        "compiled_pdf_pages": 27,
        "bibliography_reviewed": str(BIB),
        "bibliography_sha256": sha256(BIB),
        "repository_branch": BRANCH,
        "repository_commit_sha": COMMIT_SHA,
        "working_tree_state": "dirty_and_contains_unresolved_conflicts",
        "documentation_update_date": CREATED_AT,
        "research_questions": [
            "Which acquisition objectives improve retrospective label efficiency relative to random selection?",
            "How do complementary acquisition strategies behave when new computational labels are obtained through isolated CG-MD feedback loops?",
            "How do fixed-surrogate candidate optimisation and progressively relaxed familiarity restrictions affect proposal quality, candidate diversity, and retrospective learning efficiency?",
        ],
    }
    path = REPORTING / "thesis_version_metadata.json"
    write_json(path, metadata)
    register(path, [THESIS_TEX, THESIS_PDF, BIB], "Direct file metadata and SHA-256 digest.", "Latest files at the user-supplied paths.", "One source, one PDF, and one bibliography file.")
    return metadata


def phase1_summaries() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = PHASE1 / "tables" / "nested_cv_outer_predictions.csv"
    rows = [row for row in read_csv(source) if row["threshold_type"] == "PR"]
    fold_rows: list[dict[str, object]] = []
    metrics = ["Accuracy", "F1", "ROC-AUC", "PR-AUC", "gmean", "Brier", "ECE-10", "MCE-10"]
    for row in rows:
        fold_rows.append(
            {
                "model": row["model"],
                "outer_fold": i(row["outer_fold_id"]),
                "threshold_type": "validation_PR_F1_maximising",
                "decision_threshold": f(row["threshold_value"]),
                "num_cells": row["num_cells"],
                "kernel_size": row["kernel_size"],
                **{metric.lower().replace("-", "_"): f(row[metric]) for metric in metrics},
            }
        )
    fold_path = REPORTING / "phase1_outer_fold_metrics.csv"
    write_csv(fold_path, fold_rows)
    register(
        fold_path,
        [source],
        "Direct transcription of PR-threshold rows.",
        "threshold_type=PR; five outer-test folds per model.",
        "Five overlapping outer-fold test partitions per model family.",
    )

    summary_rows: list[dict[str, object]] = []
    hyper = {row["Model"]: row for row in read_csv(PHASE1 / "tables" / "hyperparameter_summary.csv")}
    for model in ["AP", "SP", "AP_SP", "TSNE_SP", "TSNE_AP_SP"]:
        subset = [row for row in fold_rows if row["model"] == model]
        summary: dict[str, object] = {
            "model": model,
            "fold_count": len(subset),
            "evidence_type": "fold_matched_outer_test",
            "selected_num_cells": hyper[model]["Best num-cells"],
            "selected_kernel_size": hyper[model]["Best kernel-size"],
            "outer_fold_settings": hyper[model]["Outer-fold selections"],
        }
        for metric in ["accuracy", "f1", "roc_auc", "pr_auc", "gmean", "brier", "ece_10", "mce_10", "decision_threshold"]:
            values = [float(row[metric]) for row in subset]
            summary[f"mean_{metric}"] = mean(values)
            summary[f"sd_{metric}"] = sd(values)
            summary[f"min_{metric}"] = min(values)
            summary[f"max_{metric}"] = max(values)
        summary["worst_f1_fold"] = min(subset, key=lambda row: float(row["f1"]))["outer_fold"]
        summary_rows.append(summary)
    summary_path = REPORTING / "phase1_model_family_summary.csv"
    write_csv(summary_path, summary_rows)
    register(
        summary_path,
        [source, PHASE1 / "tables" / "hyperparameter_summary.csv"],
        "Arithmetic mean, sample SD, minimum, and maximum over five PR-threshold outer-fold metrics.",
        "One PR-threshold row per model and outer fold.",
        "Five fold-level repetitions; training partitions overlap.",
    )
    return fold_rows, summary_rows


def phase2_summaries() -> tuple[list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    metrics_source = PHASE2 / "benchmark" / "per_run_round_metrics.csv"
    selected_source = PHASE2 / "benchmark" / "per_run_selected_sequences.csv"
    metric_rows = [
        row
        for row in read_csv(metrics_source)
        if row["evaluation_dataset"] == "holdout" and row["setup"] == "ensemble_calibrated"
    ]
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(i(row["initial_label_count"]), row["strategy"], i(row["outer_fold_id"]))].append(row)

    foldwise: list[dict[str, object]] = []
    for (n0, strategy, fold), group in grouped.items():
        ordered = sorted(group, key=lambda row: i(row["labeled_count"]))
        points = [(f(row["labeled_count"], 0.0) or 0.0, f(row["f1"], 0.0) or 0.0) for row in ordered]
        terminal = ordered[-1]
        foldwise.append(
            {
                "initial_label_count": n0,
                "strategy": strategy,
                "outer_fold": fold,
                "aulc_f1": trapezoid(points),
                "terminal_labeled_count": i(terminal["labeled_count"]),
                "terminal_f1": f(terminal["f1"]),
                "terminal_pr_auc": f(terminal["pr_auc"]),
                "terminal_roc_auc": f(terminal["roc_auc"]),
                "terminal_brier": f(terminal["brier_score"]),
                "terminal_ece_10": f(terminal["ece_10"]),
            }
        )
    foldwise_path = REPORTING / "phase2_foldwise_metrics.csv"
    write_csv(foldwise_path, sorted(foldwise, key=lambda row: (row["initial_label_count"], row["strategy"], row["outer_fold"])))
    register(
        foldwise_path,
        [metrics_source],
        "Normalised discrete trapezoidal AULC-F1 over labelled count; terminal row is maximum labelled count.",
        "mode=benchmark, setup=ensemble_calibrated, evaluation_dataset=holdout.",
        "Five overlapping outer-fold conditions per strategy and n0.",
    )

    selected_rows = [row for row in read_csv(selected_source) if row["setup"] == "ensemble_calibrated"]
    per_run_selected: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        per_run_selected[(i(row["initial_label_count"]), row["strategy"], i(row["outer_fold_id"]))].append(row)
    yield_diversity: dict[tuple[int, str], dict[str, object]] = {}
    for n0 in (10, 40):
        for strategy in sorted({row["strategy"] for row in selected_rows}):
            runs = [rows for (seed, name, _fold), rows in per_run_selected.items() if seed == n0 and name == strategy]
            labels = [i(row["label"]) for run in runs for row in run]
            diversities = [diversity([row["sequence"] for row in run])[0] for run in runs]
            yield_diversity[(n0, strategy)] = {
                "positive_selected": sum(labels),
                "selected_records": len(labels),
                "positive_yield": sum(labels) / len(labels) if labels else "",
                "mean_run_diversity": mean(diversities),
                "sd_run_diversity": sd(diversities),
                "min_run_diversity": min(diversities),
                "max_run_diversity": max(diversities),
            }

    overlap_source = PHASE2 / "benchmark" / "overlap" / "pairwise_strategy_overlap_summary.csv"
    overlap_rows = read_csv(overlap_source)
    overlap_mean: dict[tuple[int, str], float] = {}
    strategies = sorted({row["strategy"] for row in foldwise})
    for n0 in (10, 40):
        for strategy in strategies:
            values = [
                f(row["mean_jaccard"], 0.0) or 0.0
                for row in overlap_rows
                if i(row["initial_label_count"]) == n0
                and strategy in {row["strategy_a"], row["strategy_b"]}
                and row["strategy_a"] != row["strategy_b"]
            ]
            overlap_mean[(n0, strategy)] = mean(values)

    labels_source = PHASE2 / "evidence" / "labels_to_target_summary.csv"
    labels_rows = [
        row
        for row in read_csv(labels_source)
        if row["setup"] == "ensemble_calibrated" and row["evaluation_dataset"] == "holdout"
    ]
    summary_by_n0: dict[int, list[dict[str, object]]] = {10: [], 40: []}
    for n0 in (10, 40):
        for strategy in strategies:
            folds = [row for row in foldwise if row["initial_label_count"] == n0 and row["strategy"] == strategy]
            result: dict[str, object] = {
                "initial_label_count": n0,
                "strategy": strategy,
                "fold_count": len(folds),
            }
            for metric in ["aulc_f1", "terminal_f1", "terminal_pr_auc", "terminal_roc_auc", "terminal_brier", "terminal_ece_10"]:
                values = [float(row[metric]) for row in folds]
                result[f"mean_{metric}"] = mean(values)
                result[f"sd_{metric}"] = sd(values)
                result[f"min_{metric}"] = min(values)
                result[f"max_{metric}"] = max(values)
            result.update(yield_diversity[(n0, strategy)])
            result["mean_jaccard_to_other_strategies"] = overlap_mean[(n0, strategy)]
            for target in ("0.8", "0.84", "0.86"):
                match = [
                    row
                    for row in labels_rows
                    if i(row["initial_label_count"]) == n0
                    and row["strategy"] == strategy
                    and row["target_f1"] == target
                ]
                if match:
                    result[f"target_{target}_reached_count"] = i(match[0]["reached_count"])
                    result[f"target_{target}_conditional_mean_labels"] = f(match[0]["mean_labels_to_target"])
                    result[f"target_{target}_median_labels"] = f(match[0]["median_labels_to_target"])
            summary_by_n0[n0].append(result)
        path = REPORTING / f"phase2_strategy_summary_n0_{n0}.csv"
        write_csv(path, summary_by_n0[n0])
        register(
            path,
            [metrics_source, selected_source, labels_source, overlap_source],
            "Foldwise mean/sample-SD/range plus pooled selected-label yield and mean per-run Levenshtein diversity.",
            f"benchmark ensemble_calibrated holdout; initial_label_count={n0}.",
            "Five overlapping folds; 500 selected records per strategy across folds.",
        )

    paired_path = REPORTING / "phase2_paired_vs_random.csv"
    paired_rows = read_csv(PHASE2 / "evidence" / "paired_vs_random.csv")
    write_csv(paired_path, paired_rows)
    register(
        paired_path,
        [PHASE2 / "evidence" / "paired_vs_random.csv"],
        "Canonical corrected paired differences copied without modification.",
        "Benchmark rows as stored.",
        "Five matched fold conditions per initial-label condition.",
    )
    return foldwise, summary_by_n0


def phase3_summaries() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    round_rows: list[dict[str, object]] = []
    branch_sequences: dict[str, list[str]] = defaultdict(list)
    for branch in BRANCH_ORDER:
        cumulative = 0
        for round_id in range(1, 9):
            selected_path = PHASE3 / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "selected_batch.csv"
            review_path = PHASE3 / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "review" / "md_review.csv"
            selected = {row["sequence"]: row for row in read_csv(selected_path)}
            for review in read_csv(review_path):
                label = i(review["cgmd_label"])
                cumulative += label
                sequence = review["sequence"]
                branch_sequences[branch].append(sequence)
                round_rows.append(
                    {
                        "branch": branch,
                        "round": round_id,
                        "sequence": sequence,
                        "selection_rank": i(selected[sequence]["selection_rank"]),
                        "acquisition_score": f(selected[sequence]["acquisition_score"]),
                        "pred_mean": f(selected[sequence]["pred_mean"]),
                        "pred_std": f(selected[sequence]["pred_std"]),
                        "cgmd_label": label,
                        "ap_sasa_200ns": f(review["AP_sasa"]),
                        "paper_path_apcontact_last10ns": f(review["paper_path_APcontact_last10ns"]),
                        "cumulative_positive_outcomes": cumulative,
                        "label_confidence": review["label_confidence"],
                        "reviewer": review["reviewer"],
                        "reviewed_at": review["reviewed_at"],
                        "review_csv": rel(review_path),
                    }
                )
    round_path = REPORTING / "phase3_round_outcomes.csv"
    write_csv(round_path, round_rows)
    register(
        round_path,
        [
            PHASE3 / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "selected_batch.csv"
            for branch in BRANCH_ORDER
            for round_id in range(1, 9)
        ]
        + [
            PHASE3 / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "review" / "md_review.csv"
            for branch in BRANCH_ORDER
            for round_id in range(1, 9)
        ],
        "Exact join by canonical sequence between selected batch and reviewed CG-MD row; cumulative positives ordered by round and file order.",
        "Three branches, rounds 1-8, five rows per round.",
        "40 acquisitions per branch; 120 total.",
    )

    branch_summary: list[dict[str, object]] = []
    for branch in BRANCH_ORDER:
        rows = [row for row in round_rows if row["branch"] == branch]
        div_mean, div_min = diversity([str(row["sequence"]) for row in rows])
        branch_summary.append(
            {
                "branch": branch,
                "initial_labeled_count": 235,
                "round_count": 8,
                "acquisitions_per_round": 5,
                "acquired_count": len(rows),
                "final_labeled_count": 275,
                "positive_outcomes": sum(i(row["cgmd_label"]) for row in rows),
                "positive_yield": sum(i(row["cgmd_label"]) for row in rows) / len(rows),
                "mean_pairwise_normalized_levenshtein": div_mean,
                "minimum_pairwise_normalized_levenshtein": div_min,
            }
        )
    branch_summary_path = REPORTING / "phase3_branch_summary.csv"
    write_csv(branch_summary_path, branch_summary)
    register(
        branch_summary_path,
        [round_path],
        "Counts, positive fraction, and pairwise normalized Levenshtein diversity over each branch's 40 selected sequences.",
        "All eight completed rounds.",
        "40 unique sequences per branch.",
    )

    overlap_rows: list[dict[str, object]] = []
    for left in BRANCH_ORDER:
        for right in BRANCH_ORDER:
            set_left, set_right = set(branch_sequences[left]), set(branch_sequences[right])
            overlap_rows.append(
                {
                    "branch_a": left,
                    "branch_b": right,
                    "intersection_count": len(set_left & set_right),
                    "union_count": len(set_left | set_right),
                    "jaccard": len(set_left & set_right) / len(set_left | set_right),
                }
            )
    overlap_path = REPORTING / "phase3_branch_overlap.csv"
    write_csv(overlap_path, overlap_rows)
    register(
        overlap_path,
        [round_path],
        "Exact-sequence Jaccard overlap.",
        "All 40 selected sequences per branch.",
        "Set union for each branch pair.",
    )

    terminal_rows: list[dict[str, object]] = []
    for branch in BRANCH_ORDER:
        path = PHASE3 / "branches" / branch / "metrics" / "final_holdout.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        terminal_rows.append({"branch": branch, **data, "source_file": rel(path)})
    terminal_path = REPORTING / "phase3_terminal_holdout.csv"
    write_csv(terminal_path, terminal_rows)
    register(
        terminal_path,
        [PHASE3 / "branches" / branch / "metrics" / "final_holdout.json" for branch in BRANCH_ORDER],
        "Direct transcription of terminal post-ingest holdout JSON metrics.",
        "surrogate_stage=post_ingest, round_id=8.",
        "Same frozen 74-peptide holdout per branch.",
    )
    return round_rows, branch_summary, terminal_rows


def phase3_selection_summary() -> list[dict[str, object]]:
    source = ROOT / "thesis_results" / "03_real_al_strategy_selection" / "strategy_selection_summary.csv"
    rows = read_csv(source)
    output = REPORTING / "phase3_strategy_selection_evidence.csv"
    write_csv(output, rows)
    register(
        output,
        [source, ROOT / "active_learning_thesis" / "phase3_strategy_selection.py"],
        "Canonical executable strategy-selection table copied without numerical alteration.",
        "Ten Phase 2 benchmark strategies; random retained as baseline/control.",
        "Combined n0=10 and n0=40 evidence.",
    )
    return rows


def shared_cgmd_inventory(phase3_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in phase3_rows:
        expected = int(float(row["ap_sasa_200ns"]) >= 1.75 and float(row["paper_path_apcontact_last10ns"]) >= 0.5)
        rows.append(
            {
                "canonical_sequence": row["sequence"],
                "phase": "3",
                "branch_or_policy": row["branch"],
                "round_or_rank": row["round"],
                "simulation_identifier": f"phase3_{row['branch']}_r{int(row['round']):03d}_{row['sequence']}",
                "ap_sasa_200ns": row["ap_sasa_200ns"],
                "paper_path_apcontact_last10ns": row["paper_path_apcontact_last10ns"],
                "expected_criterion_outcome": expected,
                "reviewed_outcome": row["cgmd_label"],
                "outcome_status": "human_review_recorded_and_ingested",
                "reviewer": row["reviewer"],
                "review_status": "reviewed",
                "source_files": row["review_csv"],
                "audit_notes": "One trajectory; no replica uncertainty.",
            }
        )
    phase4_rows, _conflicts = phase4_campaign_rows()
    for row in phase4_rows:
        if row["simulation_status"] != "complete":
            continue
        rows.append(
            {
                "canonical_sequence": row["sequence"],
                "phase": "4",
                "branch_or_policy": row["policy"],
                "round_or_rank": row["selection_rank"],
                "simulation_identifier": row["source_campaign"],
                "ap_sasa_200ns": row["ap_sasa_200ns"],
                "paper_path_apcontact_last10ns": row["paper_path_apcontact_last10ns"],
                "expected_criterion_outcome": row["operational_label"],
                "reviewed_outcome": "",
                "outcome_status": "threshold_derived_from_complete_metrics_formal_review_fields_blank",
                "reviewer": "",
                "review_status": "not_formally_reviewed_in_source_csv",
                "source_files": row["source_review_csv"],
                "audit_notes": "Same Phase 3 threshold rule; one trajectory; no replica uncertainty.",
            }
        )
    output = REPORTING / "shared_cgmd_inventory.csv"
    write_csv(output, rows)
    register(
        output,
        [REPORTING / "phase3_round_outcomes.csv"]
        + [root for root in CAMPAIGN_ROOTS],
        "One row per unique canonical simulated sequence; expected outcome is the conjunction AP_sasa(200 ns)>=1.75 and final-10-ns path-contact>=0.5.",
        "120 Phase 3 reviewed simulations plus 29 complete primary Phase 4 simulations.",
        "149 unique trajectories; one retained trajectory per peptide.",
    )
    return rows


def phase4_summaries() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows, conflicts = phase4_campaign_rows()
    output = REPORTING / "phase4_assessment_inventory.csv"
    write_csv(output, rows)
    register(
        output,
        [
            PHASE4 / "comparison" / "round_001" / "all_selected_peptides.csv",
            *CAMPAIGN_ROOTS,
        ],
        "Exact-sequence reconciliation; complete campaigns require analysis_complete plus both operational metrics; outcome derived by the Phase 3 threshold conjunction.",
        "All 30 archived policy-selection records.",
        "Five records per policy; 29 assessed, one unassessed.",
    )
    conflict_path = REPORTING / "phase4_metric_conflicts.csv"
    write_csv(conflict_path, conflicts, ["sequence", "complete_sources", "metric_pairs"])
    register(
        conflict_path,
        CAMPAIGN_ROOTS,
        "Reports only conflicting complete metric pairs for the same canonical sequence.",
        "Primary Phase 4 selected sequences.",
        "Zero conflicts found.",
    )
    policy_summary: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        subset = [row for row in rows if row["policy"] == policy]
        assessed = [row for row in subset if row["simulation_status"] == "complete"]
        labels = [i(row["operational_label"]) for row in assessed]
        sasa = [float(row["ap_sasa_200ns"]) for row in assessed]
        contact = [float(row["paper_path_apcontact_last10ns"]) for row in assessed]
        seqs = [str(row["sequence"]) for row in subset]
        div_mean, div_min = diversity(seqs)
        policy_summary.append(
            {
                "policy": policy,
                "archived_selected_records": len(subset),
                "assessed_records": len(assessed),
                "coverage": len(assessed) / 5,
                "positive_operational_outcomes": sum(labels),
                "cgmd_yield_over_assessed": sum(labels) / len(labels) if labels else "",
                "mean_ap_sasa_200ns": mean(sasa) if sasa else "",
                "min_ap_sasa_200ns": min(sasa) if sasa else "",
                "max_ap_sasa_200ns": max(sasa) if sasa else "",
                "mean_path_contact_last10ns": mean(contact) if contact else "",
                "min_path_contact_last10ns": min(contact) if contact else "",
                "max_path_contact_last10ns": max(contact) if contact else "",
                "mean_pairwise_normalized_levenshtein": div_mean,
                "minimum_pairwise_normalized_levenshtein": div_min,
                "excluded_or_missing_records": 5 - len(assessed),
                "formal_reviewed_labels_in_source": 0,
            }
        )
    summary_path = REPORTING / "phase4_policy_summary.csv"
    write_csv(summary_path, policy_summary)
    register(
        summary_path,
        [output],
        "Yield=sum(threshold outcome)/assessed records; coverage=assessed/5; descriptive metric ranges and selected-set Levenshtein diversity.",
        "Grouped by originating policy; unassessed record excluded from yield and retained in coverage.",
        "Five archived records per policy.",
    )

    overlap_rows: list[dict[str, object]] = []
    by_policy = {policy: {str(row["sequence"]) for row in rows if row["policy"] == policy} for policy in POLICY_ORDER}
    for left in POLICY_ORDER:
        for right in POLICY_ORDER:
            union = by_policy[left] | by_policy[right]
            overlap_rows.append(
                {
                    "policy_a": left,
                    "policy_b": right,
                    "intersection_count": len(by_policy[left] & by_policy[right]),
                    "jaccard": len(by_policy[left] & by_policy[right]) / len(union),
                }
            )
    overlap_path = REPORTING / "phase4_policy_exact_overlap.csv"
    write_csv(overlap_path, overlap_rows)
    register(
        overlap_path,
        [PHASE4 / "comparison" / "round_001" / "all_selected_peptides.csv"],
        "Exact-sequence Jaccard overlap among archived policy batches.",
        "Five archived sequences per policy, before simulation exclusion.",
        "Set union for each policy pair.",
    )
    return rows, policy_summary


def phase4d_summary() -> list[dict[str, str]]:
    source = PHASE4 / "phase4d" / "run_001" / "all_policy_tradeoffs.csv"
    rows = read_csv(source)
    output = REPORTING / "phase4d_tradeoffs.csv"
    write_csv(output, rows)
    register(
        output,
        [source],
        "Canonical paired within-retained-pool trade-off table copied without modification.",
        "Five guided policies; random excluded from paired selector comparison.",
        "Five utility-only and five similarity-aware candidates per guided policy.",
    )
    return rows


def phase5_summaries() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = PHASE5 / "tables" / "learning_curves.csv"
    rows = [
        row
        for row in read_csv(source)
        if row["evaluation_dataset"] == "holdout" and i(row["initial_label_count"]) == 10
    ]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], i(row["outer_fold"]))].append(row)
    foldwise: list[dict[str, object]] = []
    for (strategy, fold), group in grouped.items():
        ordered = sorted(group, key=lambda row: i(row["labeled_count"]))
        points = [(f(row["labeled_count"], 0.0) or 0.0, f(row["f1"], 0.0) or 0.0) for row in ordered]
        terminal = ordered[-1]
        foldwise.append(
            {
                "strategy": strategy,
                "outer_fold": fold,
                "aulc_f1_10_60": trapezoid(points, 10, 60),
                "aulc_f1_10_110": trapezoid(points, 10, 110),
                "aulc_f1_10_160": trapezoid(points, 10, 160),
                "aulc_f1_10_235": trapezoid(points, 10, 235),
                "terminal_f1": f(terminal["f1"]),
                "terminal_pr_auc": f(terminal["pr_auc"]),
            }
        )
    fold_path = REPORTING / "phase5_foldwise_aulc.csv"
    write_csv(fold_path, sorted(foldwise, key=lambda row: (row["strategy"], row["outer_fold"])))
    register(
        fold_path,
        [source],
        "Normalised discrete trapezoidal AULC-F1 over preregistered intervals; terminal metrics at 235 labels.",
        "holdout, initial_label_count=10, corrected completed Phase 5 aggregation.",
        "Three overlapping outer-fold conditions per strategy.",
    )
    labels = read_csv(PHASE5 / "tables" / "labels_to_target_summary.csv")
    summary: list[dict[str, object]] = []
    for strategy in ["random", "predictive_entropy", "static_easy_entropy", "self_paced_entropy"]:
        folds = [row for row in foldwise if row["strategy"] == strategy]
        item: dict[str, object] = {"strategy": strategy, "fold_count": len(folds)}
        for metric in [
            "aulc_f1_10_60",
            "aulc_f1_10_110",
            "aulc_f1_10_160",
            "aulc_f1_10_235",
            "terminal_f1",
            "terminal_pr_auc",
        ]:
            values = [float(row[metric]) for row in folds]
            item[f"mean_{metric}"] = mean(values)
            item[f"sd_{metric}"] = sd(values)
            item[f"min_{metric}"] = min(values)
            item[f"max_{metric}"] = max(values)
        for target in ("0.8", "0.84", "0.86"):
            match = [row for row in labels if row["strategy"] == strategy and row["target_f1"] == target]
            if match:
                item[f"target_{target}_reach_count"] = i(match[0]["reach_count"])
                item[f"target_{target}_conditional_mean_labels"] = f(match[0]["conditional_mean_labels_to_target"])
        summary.append(item)
    summary_path = REPORTING / "phase5_strategy_summary.csv"
    write_csv(summary_path, summary)
    register(
        summary_path,
        [fold_path, PHASE5 / "tables" / "labels_to_target_summary.csv"],
        "Arithmetic mean, sample SD, minimum, maximum across three folds; target summaries transcribed from corrected aggregate.",
        "fold_count=3 corrected canonical aggregation.",
        "Three overlapping fold conditions.",
    )

    proxy_source = PHASE5 / "tables" / "proxy_validity_summary.csv"
    proxy_rows = read_csv(proxy_source)
    proxy_summary: list[dict[str, object]] = []
    for strategy in ["random", "predictive_entropy", "static_easy_entropy", "self_paced_entropy"]:
        values = [
            f(row["spearman"])
            for row in proxy_rows
            if row["strategy"] == strategy
            and row["summary_type"] == "spearman"
            and row["metric"] == "post_hoc_pre_query_log_loss"
            and row["spearman"].strip()
        ]
        clean = [value for value in values if value is not None]
        proxy_summary.append(
            {
                "strategy": strategy,
                "fold_step_correlations": len(clean),
                "mean_spearman_distance_vs_log_loss": mean(clean),
                "sd_spearman_distance_vs_log_loss": sd(clean),
                "minimum_spearman": min(clean),
                "maximum_spearman": max(clean),
                "positive_correlation_fraction": sum(value > 0 for value in clean) / len(clean),
            }
        )
    proxy_path = REPORTING / "phase5_proxy_summary.csv"
    write_csv(proxy_path, proxy_summary)
    register(
        proxy_path,
        [proxy_source],
        "Descriptive mean/sample-SD/range and positive fraction over per-fold, per-step tied-rank Spearman correlations.",
        "metric=post_hoc_pre_query_log_loss, summary_type=spearman.",
        "135 fold-step correlations per strategy; repeated steps are not independent.",
    )

    runtime_source = PHASE5 / "tables" / "compute_time.csv"
    runtime = read_csv(runtime_source)
    runtime_summary: list[dict[str, object]] = []
    for strategy in ["random", "predictive_entropy", "static_easy_entropy", "self_paced_entropy"]:
        values = [f(row["walltime_seconds"], 0.0) or 0.0 for row in runtime if row["strategy"] == strategy]
        runtime_summary.append(
            {
                "strategy": strategy,
                "job_count": len(values),
                "mean_walltime_minutes": mean(values) / 60,
                "minimum_walltime_minutes": min(values) / 60,
                "maximum_walltime_minutes": max(values) / 60,
                "summed_gpu_hours": sum(values) / 3600,
            }
        )
    runtime_path = REPORTING / "phase5_runtime_summary.csv"
    write_csv(runtime_path, runtime_summary)
    register(
        runtime_path,
        [runtime_source],
        "Wall-time descriptive summaries; summed wall hours treated as approximate GPU-hours because each job requested one GPU.",
        "Twelve successful Phase 5 replay jobs.",
        "Three jobs per strategy.",
    )
    return foldwise, summary


def cross_phase_matrix() -> list[dict[str, object]]:
    rows = [
        {
            "question": "Which strategies improved retrospective label efficiency?",
            "empirical_observation": "Phase 2 predictive entropy had the strongest combined holdout AULC among eligible strategies; Phase 5 predictive entropy had the highest full AULC, while self-paced entropy exceeded static easy entropy but not predictive entropy or random.",
            "interpretation": "Uncertainty sampling was useful, but additional pacing restrictions did not consistently improve the whole learning trajectory.",
            "limitation": "Retrospective replay with overlapping folds; no independent replication.",
            "strength": "supported descriptively",
        },
        {
            "question": "Did retrospective performance translate into computational discovery?",
            "empirical_observation": "Phase 3 branches produced similar overall operational positive yields (19/40, 21/40, 19/40). Primary Phase 4 threshold-derived yields varied from 1/5 to 5/5, with very small policy batches.",
            "interpretation": "Replay ranking did not map monotonically onto branch-local CG-MD yield; fixed-surrogate policies explored very different sequence regimes.",
            "limitation": "One trajectory per peptide, no wet-lab validation, and only five Phase 4 records per policy.",
            "strength": "suggestive",
        },
        {
            "question": "How did uncertainty, disagreement, representativeness, diversity, and exploitation differ?",
            "empirical_observation": "The Phase 3 selection deliberately retained predictive entropy, family QBC, and cluster-diverse roles. Phase 4 generated policy-specific sequence families and Phase 4-D increased diversity at small utility cost within fresh pools.",
            "interpretation": "Acquisition objectives shape candidate composition and redundancy as well as predictive score.",
            "limitation": "Phase 3 role selection was post hoc and Phase 4-D differs from primary Phase 4 in generation seed and retained pool.",
            "strength": "supported descriptively",
        },
        {
            "question": "What did fixed-surrogate ranking show?",
            "empirical_observation": "Primary Phase 4 generated 30 unique archived records, assessed 29, and yielded 15 threshold-positive outcomes; PI selected five closely related long sequences and all five met the operational criterion.",
            "interpretation": "A fixed surrogate can concentrate proposals in high-scoring motifs, but apparent yield may coincide with low within-policy diversity.",
            "limitation": "Single round, policy-specific utility scales, small batches, and no retraining.",
            "strength": "suggestive",
        },
        {
            "question": "What did Phase 5 show about familiarity restrictions?",
            "empirical_observation": "Self-paced entropy improved over permanently easy-only entropy in full AULC but remained below predictive entropy and random; familiarity distance often correlated positively with pre-query loss.",
            "interpretation": "The proxy contains useful difficulty information, but restricting selection by it can sacrifice informative hard examples.",
            "limitation": "Model-dependent proxy, single AP_SP member per replay point, and three overlapping folds.",
            "strength": "supported descriptively",
        },
    ]
    path = REPORTING / "cross_phase_evidence_matrix.csv"
    write_csv(path, rows)
    register(
        path,
        [
            REPORTING / "phase2_strategy_summary_n0_10.csv",
            REPORTING / "phase3_branch_summary.csv",
            REPORTING / "phase4_policy_summary.csv",
            REPORTING / "phase4d_tradeoffs.csv",
            REPORTING / "phase5_strategy_summary.csv",
        ],
        "Narrative synthesis of verified phase-specific summaries; no new statistical inference.",
        "Only completed canonical or corrected-canonical evidence.",
        "Not applicable; qualitative evidence matrix.",
    )
    return rows


def superseded_and_unresolved() -> list[dict[str, str]]:
    rows = [
        {
            "item": "Short top-level Phase 3 working tree",
            "status": "superseded",
            "authoritative_replacement": rel(PHASE3),
            "reason": "Completed scientific archive is nested under phase3_results_20260620.",
        },
        {
            "item": "Short top-level Phase 4 working tree and stale md_inventory",
            "status": "superseded",
            "authoritative_replacement": rel(PHASE4),
            "reason": "Completed archive plus reconciled local campaign evidence is authoritative.",
        },
        {
            "item": "Original Phase 5 duplicated paired-comparison aggregate",
            "status": "superseded",
            "authoritative_replacement": "thesis_results/05_self_paced_active_learning/tables/paired_aulc_summary.csv",
            "reason": "Corrected aggregate has fold_count=3.",
        },
        {
            "item": "Original Phase 5 placeholder SVG figures",
            "status": "superseded",
            "authoritative_replacement": "thesis_results/THESIS_RESULTS_HANDOFF_20260623/05_phase5_self_paced/figures/",
            "reason": "Original SVGs stated Matplotlib unavailable; real figures were regenerated from corrected tables.",
        },
        {
            "item": "Phase 4 formal human review fields",
            "status": "unresolved",
            "authoritative_replacement": "thesis_reporting/phase4_assessment_inventory.csv",
            "reason": "All 29 complete campaign rows have blank cgmd_label/reviewer/reviewed_at; outcomes are reproducibly threshold-derived.",
        },
        {
            "item": "Phase 3 validation trajectories",
            "status": "unresolved",
            "authoritative_replacement": "Only terminal holdout JSON and round outcomes are available in completed archive.",
            "reason": "No canonical per-round validation metric trajectory was located.",
        },
        {
            "item": "Repository commit versus current evidence",
            "status": "unresolved_provenance_boundary",
            "authoritative_replacement": "Commit plus dirty-working-tree disclosure and file hashes.",
            "reason": "Current worktree contains many modifications and unresolved conflicts not represented by commit SHA.",
        },
    ]
    path = REPORTING / "superseded_and_unresolved.csv"
    write_csv(path, rows)
    register(
        path,
        [],
        "Repository evidence audit classification.",
        "Known superseded, contradictory, or unresolved items.",
        "Not applicable.",
    )
    return rows


def canonical_index() -> None:
    rows = [
        ["1", "Predictive reproduction", "Outer-fold model-family metrics", "thesis_reporting/phase1_model_family_summary.csv", "one row/model", "mean, sample SD, range", "5 folds/model", "5 overlapping folds", "derived reproducibly", "PR-threshold outer-test evidence"],
        ["1", "Architecture selection", "Selected cells/kernel", "thesis_results/01_reproduction/tables/hyperparameter_summary.csv", "one row/model", "lowest mean inner-validation loss", "inner folds", "nested CV", "canonical", "Deployment fits must remain separate"],
        ["2", "Strategy replay n0=10", "AULC, terminal metrics, yield, diversity", "thesis_reporting/phase2_strategy_summary_n0_10.csv", "one row/strategy", "fold summaries plus selected-record yield", "500 selected records/strategy", "5 overlapping folds", "derived reproducibly", "n0 intervals kept separate"],
        ["2", "Strategy replay n0=40", "AULC, terminal metrics, yield, diversity", "thesis_reporting/phase2_strategy_summary_n0_40.csv", "one row/strategy", "fold summaries plus selected-record yield", "500 selected records/strategy", "5 overlapping folds", "derived reproducibly", "n0 intervals kept separate"],
        ["2", "Paired comparison", "Differences versus random", "thesis_reporting/phase2_paired_vs_random.csv", "canonical rows", "matched fold differences", "5 fold pairs", "5 overlapping folds", "corrected canonical", "No independent-fold significance claim"],
        ["2", "Calibration/ensemble ablation", "F1, AULC, Brier, ECE", "thesis_results/02_replay/evidence/ablation_summary.csv", "setup x n0 x dataset", "canonical aggregate", "5 folds", "5 overlapping folds", "canonical", "Separate from policy benchmark"],
        ["3 selection", "Role-constrained recommendation", "Composite score and decision", "thesis_reporting/phase3_strategy_selection_evidence.csv", "one row/strategy", "executable min-max composite plus role assembly", "10 strategies", "combined n0 evidence", "canonical", "Post hoc; not preregistered"],
        ["3", "CG-MD campaign", "Round outcomes and acquisition scores", "thesis_reporting/phase3_round_outcomes.csv", "branch/round/sequence", "exact selected-review join", "120 simulations", "one trajectory/sequence", "derived reproducibly", "Human-reviewed and ingested"],
        ["3", "Terminal comparison", "Holdout F1, PR-AUC, calibration", "thesis_reporting/phase3_terminal_holdout.csv", "one row/branch", "direct JSON transcription", "74 holdout peptides", "same holdout per branch", "canonical", "Descriptive, not wholly independent external validation"],
        ["3+4", "Shared CG-MD", "SASA, path contact, criterion outcome", "thesis_reporting/shared_cgmd_inventory.csv", "one row/unique simulated sequence", "threshold conjunction", "149 trajectories", "one trajectory/sequence", "derived reproducibly", "Phase 4 formal review fields blank"],
        ["4", "Primary assessment", "Policy yield and coverage", "thesis_reporting/phase4_policy_summary.csv", "one row/policy", "positives/assessed; assessed/5", "5 archived records/policy", "29 assessed total", "derived reproducibly", "Unassessed MES record is not negative"],
        ["4", "Primary record inventory", "Sequence-level utilities and outcomes", "thesis_reporting/phase4_assessment_inventory.csv", "30 archived records", "exact-sequence reconciliation", "30 selected, 29 assessed", "single fixed surrogate", "derived reproducibly", "No cross-policy exact duplicates"],
        ["4-D", "Diversity-aware replicate", "Utility/diversity trade-off", "thesis_reporting/phase4d_tradeoffs.csv", "one row/guided policy", "same-pool paired selectors", "5+5 sequences/policy", "fresh pool", "canonical", "No CG-MD yield"],
        ["5", "Self-paced replay", "Full/partial AULC, terminal metrics", "thesis_reporting/phase5_strategy_summary.csv", "one row/strategy", "corrected fold_count=3 summary", "3 folds/strategy", "3 overlapping folds", "corrected canonical", "Not exact SPAL"],
        ["5", "Proxy validity", "Distance-log-loss Spearman", "thesis_reporting/phase5_proxy_summary.csv", "one row/strategy", "descriptive fold-step aggregation", "135 correlations/strategy", "repeated steps non-independent", "derived reproducibly", "Familiarity is model-dependent"],
    ]
    text = """# Thesis Canonical Results Index

Status vocabulary: **canonical**, **corrected canonical**, **derived reproducibly**, **superseded**, and **unresolved**.

""" + markdown_table(
        ["Phase", "Result family", "Metric/outcome", "Canonical file", "Row/key/filter", "Aggregation", "Denominator", "Fold/replicate count", "Status", "Notes"],
        rows,
    ) + """

## Superseded and unresolved items

See `thesis_reporting/superseded_and_unresolved.csv`. Raw or canonical files were not overwritten.
"""
    write_text(ROOT / "THESIS_CANONICAL_RESULTS_INDEX.md", text)


def figure_table_plan() -> None:
    rows = [
        ["Fig. 1", "Predictive model-family comparison", "RQ1 foundation", "phase1_model_family_summary.csv", "Grouped bars/dots: F1 and PR-AUC by family", "Fold-level points + mean/SD; folds labelled non-independent", "AP_SP is competitive and becomes the downstream surrogate", "Do not imply independent replication", "Main"],
        ["Fig. 2", "Phase 2 holdout learning curves, n0=10", "RQ1", "Phase 2 learning_curves.csv", "F1 vs labelled count", "Mean with fold range or SD", "Strategy trajectories differ most at low budgets", "Retrospective replay", "Main"],
        ["Fig. 3", "Phase 2 AULC and labels-to-target", "RQ1", "phase2_strategy_summary_n0_10.csv and n0_40.csv", "Two-panel dot/bar plot", "Fold points; target reach count annotated", "Separate n0 conditions and distinguish reach from conditional mean", "No cross-interval pooling", "Main"],
        ["Table 1", "Phase 2 ten-strategy exact summary", "RQ1", "phase2_strategy_summary_n0_10.csv; n0_40.csv", "Exact table", "Mean, SD, range, reach counts", "Preserves full quantitative benchmark", "PI/EI/MES excluded", "Main"],
        ["Fig. 4", "Calibration and ensemble ablation", "RQ1 support", "ablation_summary.csv", "F1/AULC plus Brier/ECE panels", "Five fold conditions", "Calibration/ensembling affect performance and probability quality", "Separate from policy ranking", "Appendix or main if space"],
        ["Fig. 5", "Phase 3 cumulative operational positives", "RQ2", "phase3_round_outcomes.csv", "Cumulative step lines by branch", "Raw counts, no CI", "All three branches produced computational positives over eight rounds", "Branch-local labels; one trajectory each", "Main"],
        ["Table 2", "Phase 3 terminal branch comparison", "RQ2", "phase3_terminal_holdout.csv", "Exact table", "F1, PR-AUC, Brier, ECE, threshold", "Terminal branch models are similar but not identical", "Same holdout reused descriptively", "Main"],
        ["Fig. 6", "Shared CG-MD outcome plane", "RQ2 and RQ3", "shared_cgmd_inventory.csv", "AP-SASA vs final-10-ns path contact with thresholds", "Raw points; phase/branch/policy encoding", "Shows operational conjunction and outcome distribution", "Not experimental validation", "Main"],
        ["Table 3", "Primary Phase 4 policy assessment", "RQ3", "phase4_policy_summary.csv", "Counts and percentages", "Report positives/assessed and assessed/5", "Fixed-surrogate policies produced different operational yields", "Only five records/policy; MES coverage 4/5", "Main"],
        ["Fig. 7", "Primary Phase 4 utility versus CG-MD outcome", "RQ3", "phase4_assessment_inventory.csv", "Policy-faceted scatter", "Raw records, no cross-policy utility comparison", "Within-policy ranking can be compared with outcomes", "Utility scales differ", "Appendix"],
        ["Fig. 8", "Phase 4-D utility-diversity trade-off", "RQ3", "phase4d_tradeoffs.csv", "Delta diversity vs delta utility", "One point/guided policy", "Similarity-aware selection raises diversity at utility cost", "Fresh-pool comparison with primary is descriptive only", "Main"],
        ["Fig. 9", "Phase 5 full and partial AULC", "RQ3", "phase5_strategy_summary.csv", "Grouped intervals", "Three fold-level points + means", "Predictive entropy leads full AULC; self-paced exceeds static easy only", "Overlapping folds, corrected aggregation", "Main"],
        ["Fig. 10", "Phase 5 familiarity and eligibility", "RQ3", "Phase 5 proxy and eligibility tables", "Two-panel trajectory/scatter", "Descriptive fold-step summaries", "Proxy carries difficulty information but pacing can exclude useful hard points", "Not intrinsic chemical difficulty", "Main"],
        ["Table 4", "Phase 5 exact result summary", "RQ3", "phase5_strategy_summary.csv", "Exact table", "Full/partial AULC, terminal metrics, targets", "Preserves mixed/null finding", "Three overlapping folds", "Main"],
    ]
    text = """# Thesis Figure And Table Plan

The plan intentionally favours one visual per scientific question plus exact-value tables. Existing files in `thesis_results/THESIS_RESULTS_HANDOFF_20260623/` provide many of these visuals; the machine-readable sources below remain authoritative.

""" + markdown_table(
        ["No.", "Working title", "RQ", "Data source", "Type/axes", "Uncertainty", "Caption message", "Boundary", "Placement"],
        rows,
    )
    write_text(ROOT / "THESIS_FIGURE_TABLE_PLAN.md", text)


def results_discussion_handoff(
    metadata: dict[str, object],
    phase1: list[dict[str, object]],
    phase2: dict[int, list[dict[str, object]]],
    phase3_branches: list[dict[str, object]],
    phase3_terminal: list[dict[str, object]],
    phase4_summary: list[dict[str, object]],
    phase5: list[dict[str, object]],
    selection: list[dict[str, object]],
    cross_phase: list[dict[str, object]],
) -> None:
    p1_rows = [
        [
            row["model"],
            fmt(row["mean_f1"]),
            fmt(row["sd_f1"]),
            f"{fmt(row['min_f1'])}-{fmt(row['max_f1'])}",
            fmt(row["mean_pr_auc"]),
            fmt(row["mean_roc_auc"]),
            fmt(row["mean_brier"]),
            fmt(row["mean_ece_10"]),
        ]
        for row in phase1
    ]
    p2_10 = sorted(phase2[10], key=lambda row: float(row["mean_aulc_f1"]), reverse=True)
    p2_40 = sorted(phase2[40], key=lambda row: float(row["mean_aulc_f1"]), reverse=True)
    p2_rows_10 = [[row["strategy"], fmt(row["mean_aulc_f1"]), fmt(row["mean_terminal_f1"]), fmt(row["mean_terminal_pr_auc"]), f"{row.get('target_0.86_reached_count','')}/5", fmt(row.get("target_0.86_conditional_mean_labels")), f"{row['positive_selected']}/{row['selected_records']}"] for row in p2_10]
    p2_rows_40 = [[row["strategy"], fmt(row["mean_aulc_f1"]), fmt(row["mean_terminal_f1"]), fmt(row["mean_terminal_pr_auc"]), f"{row.get('target_0.86_reached_count','')}/5", fmt(row.get("target_0.86_conditional_mean_labels")), f"{row['positive_selected']}/{row['selected_records']}"] for row in p2_40]
    selection_by = {row["strategy"]: row for row in selection}
    selected_table = [
        [
            strategy,
            selection_by[strategy]["decision"],
            selection_by[strategy]["role"],
            fmt(selection_by[strategy]["mean_holdout_AULC_F1_by_labeled_count"]),
            fmt(selection_by[strategy]["composite_score"]),
            fmt(selection_by[strategy]["mean_diversity"]),
            fmt(selection_by[strategy]["max_overlap_with_recommended"]),
        ]
        for strategy in ["predictive_entropy", "family_qbc", "cluster_diverse_representative", "ensemble_mi"]
    ]
    phase3_table = [[row["branch"], row["positive_outcomes"], fmt(row["positive_yield"]), fmt(row["mean_pairwise_normalized_levenshtein"])] for row in phase3_branches]
    terminal_by = {row["branch"]: row for row in phase3_terminal}
    phase3_terminal_table = [[branch, fmt(terminal_by[branch]["f1"]), fmt(terminal_by[branch]["pr_auc"]), fmt(terminal_by[branch]["brier_score"]), fmt(terminal_by[branch]["ece_10"]), fmt(terminal_by[branch]["decision_threshold"])] for branch in BRANCH_ORDER]
    phase4_table = [[row["policy"], f"{row['positive_operational_outcomes']}/{row['assessed_records']}", fmt(row["cgmd_yield_over_assessed"]), fmt(row["coverage"]), fmt(row["mean_pairwise_normalized_levenshtein"])] for row in phase4_summary]
    phase5_table = [[row["strategy"], fmt(row["mean_aulc_f1_10_60"]), fmt(row["mean_aulc_f1_10_110"]), fmt(row["mean_aulc_f1_10_160"]), fmt(row["mean_aulc_f1_10_235"]), fmt(row["mean_terminal_f1"]), fmt(row["mean_terminal_pr_auc"])] for row in phase5]

    rq_table = [
        ["RQ1: Which acquisition objectives improve retrospective label efficiency relative to random selection?", "Phases 1-2 and Phase 5", "Normalised AULC-F1; paired delta vs random", "Terminal F1/PR-AUC, labels-to-target, calibration, diversity", "Overlapping folds; retrospective replay"],
        ["RQ2: How do complementary acquisition strategies behave under isolated CG-MD feedback?", "Phase 3", "Branch-local positive operational outcomes; terminal holdout metrics", "Acquisition scores, class balance, sequence diversity and overlap", "One trajectory/peptide; no cross-branch label sharing; reused holdout"],
        ["RQ3: How do fixed-surrogate optimisation and familiarity restrictions affect proposals, diversity, and learning?", "Phases 4, 4-D, 5", "Phase 4 yield+coverage; Phase 4-D utility/diversity delta; Phase 5 AULC", "Lengths, overlap, familiarity correlations, eligibility, runtime", "Small fixed batches; Phase 4 not closed loop; Phase 5 not exact SPAL"],
    ]

    discussion_rows = [
        ["RQ1", "Predictive entropy was consistently competitive and led Phase 5 full AULC; Phase 2 effects depended on n0 and strategy.", "Phase 2 summaries and Phase 5 corrected AULCs.", "Uncertainty can focus labels near the decision boundary.", "Non-monotonic trajectories and initialization can favour random.", "Overlapping folds and small dataset.", "Settles; Barrett & White; Evans et al.", "supported descriptively"],
        ["RQ2", "All three Phase 3 branches found computational positives, with similar overall yield and modest terminal metric differences.", "19/40, 21/40, and 19/40 positives plus terminal holdout metrics.", "Complementary acquisition roles explore different regions while maintaining model performance.", "Threshold and sequence-length effects may dominate branch strategy.", "One trajectory, fixed CG model, no wet lab.", "Shmilovich et al.; Talluri et al.; Thapa et al.", "supported descriptively"],
        ["RQ3", "Primary Phase 4 produced 15 threshold positives among 29 assessed records; Phase 4-D exposed utility-diversity trade-offs; self-pacing did not beat predictive entropy overall.", "Phase 4 inventory, Phase 4-D trade-offs, Phase 5 corrected summaries.", "Goal-directed utilities strongly shape motif concentration and diversity.", "Surrogate bias, fresh-pool confounding, and familiarity-proxy artifacts.", "Single fixed-surrogate round and three overlapping Phase 5 folds.", "Njirjak et al.; Di Fiore et al.; Tang & Huang.", "suggestive"],
    ]

    literature_rows = [
        ["Hybrid AP_SP reproduction supports surrogate use", "Njirjak et al. (2024)", "Reproduction/extension", "The thesis reproduces a competitive AP_SP predictor and extends it into calibrated active-learning and proposal workflows.", "Verify exact paper metric wording before final prose."],
        ["Retrospective uncertainty sampling can improve label efficiency but is condition-dependent", "Settles (2009); Barrett & White (2021); Evans et al. (2014)", "Agreement with conditional AL benefit", "Results support strategy- and initialization-dependent gains rather than universal AL superiority.", "Targeted quotation check recommended."],
        ["CG-MD can support peptide discovery prioritization", "Shmilovich et al. (2020); Thapa et al. (2024); Talluri et al. (2025)", "Methodological extension", "The thesis uses CG-MD as computational evidence in iterative and fixed-surrogate selection workflows.", "Literature details and experimental validation rates require targeted search."],
        ["BO-style utilities emphasize goal-directed candidate ranking", "Jones et al. (1998); Srinivas et al. (2010); Wang & Jegelka (2017); Di Fiore et al. (2024)", "Adaptation", "PI, EI, UCB and MES were adapted to calibrated neural-ensemble probability space in a single fixed-surrogate round.", "Do not imply exact GP BO or closed-loop BO."],
        ["Self-paced restriction embodies easy-to-hard acquisition", "Tang & Huang (2019)", "SPAL-inspired adaptation, not reproduction", "The familiarity-percentile schedule tests an easy-to-hard principle with neural embeddings.", "Exact SPAL objective and solver differ; discuss explicitly."],
    ]

    abstract_results = [
        f"Phase 1 AP_SP achieved mean outer-fold F1 {fmt(next(row for row in phase1 if row['model']=='AP_SP')['mean_f1'])} and PR-AUC {fmt(next(row for row in phase1 if row['model']=='AP_SP')['mean_pr_auc'])}.",
        f"Phase 2 predictive entropy had combined strategy-selection AULC-F1 {fmt(selection_by['predictive_entropy']['mean_holdout_AULC_F1_by_labeled_count'])}, compared with random {fmt(selection_by['predictive_entropy']['random_floor_AULC'])}.",
        f"Phase 3 yielded 19/40, 21/40, and 19/40 operational positives for predictive entropy, family QBC, and cluster-diverse branches.",
        "Primary Phase 4 assessed 29/30 archived records and produced 15 threshold-positive outcomes; policy yields ranged from 1/5 to 5/5, with MES coverage 4/5.",
        f"Phase 5 full AULC-F1 was {fmt(next(row for row in phase5 if row['strategy']=='predictive_entropy')['mean_aulc_f1_10_235'])} for predictive entropy and {fmt(next(row for row in phase5 if row['strategy']=='self_paced_entropy')['mean_aulc_f1_10_235'])} for self-paced entropy.",
    ]

    text = f"""# Thesis Results And Discussion Handoff

## 1. Purpose and authority

This is the authoritative handoff for writing the **Results, Discussion,
Conclusions, and Abstract**. It does not replace canonical raw artifacts.

- Thesis source reviewed: `{metadata['thesis_source_reviewed']}`
- Thesis source SHA-256: `{metadata['thesis_source_sha256']}`
- Compiled PDF reviewed: `{metadata['compiled_pdf_reviewed']}` ({metadata['compiled_pdf_pages']} pages)
- PDF SHA-256: `{metadata['compiled_pdf_sha256']}`
- Bibliography reviewed: `{metadata['bibliography_reviewed']}`
- Repository branch: `{BRANCH}`
- Commit SHA: `{COMMIT_SHA}`
- Worktree: dirty, with unresolved conflicts; file hashes and canonical paths are therefore part of the evidence contract.
- Documentation update: `{CREATED_AT}`

Source hierarchy: canonical raw/reviewed outputs, executable code/configuration,
manifests, corrected aggregates, technical documentation, then thesis prose.

## 2. Research questions

{markdown_table(["Research question", "Phases", "Primary metrics", "Supporting metrics", "Limitations"], rq_table)}

## 3. Global interpretation boundaries

- Experimental labels are distinct from reviewed or threshold-derived CG-MD operational outcomes.
- Phases 2 and 5 are retrospective replay; Phase 3 is prospective computational acquisition with branch-local feedback.
- Phase 4 is fixed-surrogate proposal ranking, not adaptive improvement or closed-loop Bayesian optimisation.
- Phase 3 branches have separate ledgers even though simulations share one physical protocol.
- Phase 4-D is unsimulated and separate from primary Phase 4.
- Phase 5 is SPAL-inspired, not exact SPAL; familiarity is a model-dependent manifold-distance proxy.
- Outer folds overlap. Fold aggregates are descriptive and are not independent replications.
- One 200 ns CG-MD trajectory does not establish an experimental hit or aggregation probability.

## 4. Phase 1 canonical results

{markdown_table(["Model", "Mean F1", "SD", "Range", "Mean PR-AUC", "Mean ROC-AUC", "Mean Brier", "Mean ECE-10"], p1_rows)}

Each row summarises five PR-threshold outer-test folds. Threshold-dependent
accuracy/F1 and threshold-independent ROC-AUC/PR-AUC are kept separate in
`thesis_reporting/phase1_outer_fold_metrics.csv`. AP_SP was retained as the
later surrogate because it reproduced the hybrid architecture central to the
project, achieved mean F1 {fmt(next(row for row in phase1 if row['model']=='AP_SP')['mean_f1'])},
and exposes the joint 384-dimensional penultimate representation used by later
workflows. SP has slightly higher mean F1 and PR-AUC in this reproduction; AP_SP
must not be called uniquely best on Phase 1 metrics.

Deployment-style all-data fits are not fold-matched evidence and must be
reported separately if used.

## 5. Phase 2 canonical results

### n0=10

{markdown_table(["Strategy", "AULC-F1", "Terminal F1", "PR-AUC", "F1=.86 reach", "Conditional labels", "Positive selections"], p2_rows_10)}

### n0=40

{markdown_table(["Strategy", "AULC-F1", "Terminal F1", "PR-AUC", "F1=.86 reach", "Conditional labels", "Positive selections"], p2_rows_40)}

The two conditions cover different labelled-count intervals and are never
pooled as a single AULC experiment. Fold-wise values, ranges, worst folds,
labels-to-target reach counts, positive yield, diversity, overlap, and paired
differences from random are in `thesis_reporting/`.

The calibration/ensemble ablation is a separate result family. PI, EI and MES
were not principal Phase 2 benchmark strategies.

## 6. Phase 3 strategy-selection evidence

The executable composite was:

`C = 0.35*N_up(AULC_F1) + 0.25*N_down(L_0.86) + 0.15*N_down(R_fold) + 0.15*N_up(D_Lev) + 0.10*N_down(J_mean) + B_role`.

`N` is min-max normalisation over non-excluded strategies; a constant component
receives 1.0. The role bonus was 0.04 for committee-uncertainty or
diversity/novelty roles. Labels-to-target used the implementation's incomplete
target penalty before scoring. Strategies below random combined AULC by more
than the `1e-6` tolerance were exploratory/control. Recommendation assembly was
sequential: best eligible AULC, then an eligible committee-uncertainty role,
then an eligible diversity role satisfying the Jaccard gate where possible.

{markdown_table(["Strategy", "Decision", "Role", "Combined AULC", "Composite", "Diversity", "Max overlap"], selected_table)}

Predictive entropy was the best eligible combined-AULC strategy. Family QBC was
chosen as the committee representative, not simply the second-highest global
score. Cluster-diverse was the highest-composite eligible diversity-oriented
choice satisfying the assembly logic, but OED logdet had the largest measured
Levenshtein diversity. Ensemble MI was the highest-ranked unselected backup.

The code generated evidence and a recommendation, followed by an explicit human
adoption in `real_al_strategy_recommendation.md`. There is no evidence that this
rule was preregistered before inspecting Phase 2 results.

## 7. Phase 3 campaign results

{markdown_table(["Branch", "Positives", "Yield", "Mean diversity"], phase3_table)}

{markdown_table(["Branch", "Terminal F1", "PR-AUC", "Brier", "ECE-10", "Validation threshold"], phase3_terminal_table)}

Each branch started with 235 labels, acquired five labels in each of eight
rounds, and ended with 275. All 120 sequences were unique across branches.
Round-level sequences, acquisition scores, predictions, CG-MD metrics, labels,
and cumulative positives are in `phase3_round_outcomes.csv`.

The same 74-peptide holdout was used once after terminal retraining. Its outer
fold condition had also contributed to Phase 2 strategy-family selection, so
the terminal branch comparison is descriptive rather than a wholly independent
external validation. Canonical per-round validation trajectories were not
located in the completed archive.

## 8. Shared CG-MD evidence

The operational criterion was:

`AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`.

The inventory contains 120 Phase 3 reviewed/ingested simulations and 29 primary
Phase 4 complete simulations. Every canonical sequence has one retained
trajectory; replica uncertainty is absent. The source metrics, expected
criterion outcome, formal review state, and provenance are recorded in
`shared_cgmd_inventory.csv`. These are computational outcomes, not experimental
validation.

## 9. Primary Phase 4 canonical results

{markdown_table(["Policy", "Positive/assessed", "CG-MD yield", "Coverage", "Selected diversity"], phase4_table)}

The 30 archived records are exact-unique across policies. One 25-residue MES
record (`VLNINNMGAKWRRTCNQRLTPTALP`) lies outside the intended 3-24 range and
was archived, excluded from simulation, unassessed, not replaced, and is not a
negative. Therefore 29 eligible records correspond to 29 unique simulations.

The campaign files contain complete metrics but blank formal label/reviewer
fields. Phase 4 outcomes are consequently described as **threshold-derived
operational outcomes**, not human-reviewed labels. The evidence is sufficient
to write Phase 4 Results if this distinction and the raw counts are preserved.
Utility magnitudes must not be compared across policies with different scales.

## 10. Phase 4-D canonical results

`phase4d_tradeoffs.csv` preserves the controlled within-pool comparison between
utility-only and similarity-aware final selectors. Utility and diversity
changes are attributable to final selection because each pair shares a retained
pool and frozen utilities. Comparisons with primary Phase 4 are descriptive
only because Phase 4-D used fresh GA seeds and fresh pools. No Phase 4-D
candidate has CG-MD evidence, so no yield is reported.

## 11. Phase 5 canonical results

{markdown_table(["Strategy", "AULC 10-60", "AULC 10-110", "AULC 10-160", "AULC 10-235", "Terminal F1", "PR-AUC"], phase5_table)}

Only the corrected aggregation with `fold_count=3` is authoritative. The prior
paired-comparison aggregation duplicated fold comparisons; raw jobs were
unaffected. Predictive entropy led full AULC. Self-paced entropy exceeded
static easy entropy over the full interval, but did not exceed predictive
entropy or random. Proxy-validity, eligibility, runtime, target reach, and
terminal convergence records are indexed in `thesis_reporting/`.

## 12. Cross-phase synthesis

{markdown_table(["Question", "Observation", "Interpretation", "Limitation", "Strength"], [[row["question"], row["empirical_observation"], row["interpretation"], row["limitation"], row["strength"]] for row in cross_phase])}

## 13. Discussion framework

{markdown_table(["RQ", "Supported claim", "Evidence", "Interpretation", "Alternative", "Limitation", "Literature", "Strength"], discussion_rows)}

Required limitations: small and imbalanced dataset; overlapping folds; no
independent external experimental test set; calibration dependence; no
cross-member latent-axis averaging; post hoc role-constrained Phase 3 strategy
selection; branch-local feedback; one trajectory per peptide; sequence-length
effects; fixed force field and operational thresholds; 200 ns duration; no
wet-lab validation; single-round fixed-surrogate Phase 4; 25-residue boundary
artifact; Phase 4-D fresh-pool confounding; model-dependent familiarity proxy;
and corrected Phase 5 aggregation.

## 14. Literature-comparison map

{markdown_table(["Thesis finding", "Related citation", "Relation", "Permitted claim", "Gap"], literature_rows)}

Bibliography entries were inspected, but paper-specific numerical claims were
not re-audited here. Rows marked with a gap require targeted literature
verification before detailed comparison prose.

## 15. Conclusions contract

### Permitted

- The five predictive families were reproduced, and AP_SP provided a documented joint surrogate representation.
- Some acquisition strategies improved retrospective label efficiency under specific replay conditions.
- Phase 3 completed three isolated eight-round computational feedback loops and found operational positives in every branch.
- Primary Phase 4 demonstrates policy-dependent fixed-surrogate proposal concentration and threshold-derived CG-MD outcomes.
- Phase 4-D demonstrates a controlled within-pool utility-diversity trade-off.
- Phase 5 shows that progressive familiarity pacing is better than permanently easy-only selection over the full trajectory, but not better than predictive entropy or random in this reduced replay.

### Prohibited

- One strategy is universally best.
- CG-MD proves experimental self-assembly.
- Phase 4 is closed-loop Bayesian optimisation.
- Phase 4-D candidates were simulated or have yield.
- Overlapping folds are independent replications.
- Phase 5 measures intrinsic chemical difficulty or exactly reproduces SPAL.
- The 25-residue MES record is an in-range proposal or a negative outcome.
- One trajectory estimates aggregation probability.

## 16. Abstract evidence sheet

- **Background:** Data-efficient discovery of self-assembling peptides requires predictive models, acquisition strategies, and computational validation under limited labelled data.
- **Objective:** Evaluate retrospective label efficiency, branch-local CG-MD feedback, fixed-surrogate proposal policies, diversity-aware selection, and self-paced familiarity restrictions.
- **Methods:** Nested predictive reproduction, calibrated ensemble replay, three isolated eight-round CG-MD loops, one-round fixed-surrogate proposal comparison, a diversity-aware replicate, and reduced SPAL-inspired replay.
- **Quantitative results:**
""" + "\n".join(f"  - {item}" for item in abstract_results) + """
- **Interpretation:** Acquisition objectives changed both learning efficiency and the composition of proposed peptide sets, but benefits were strategy-, budget-, and protocol-dependent.
- **Principal limitation:** Evidence is computational and retrospective, with overlapping folds, one trajectory per simulated peptide, and no wet-laboratory validation.
- **Conclusion:** The project supports a calibrated, uncertainty-aware and diversity-conscious computational workflow while rejecting claims of universal strategy superiority.

The final abstract must be written only after Results, Discussion, and
Conclusions are approved.

## Writing-model safeguards

Do not invent missing values, estimate from figures when tables exist, imply
causation from descriptive comparisons, call CG-MD experimental ground truth,
call Phase 4 closed-loop BO, call Phase 5 exact SPAL, call folds independent,
compare cross-policy utility magnitudes, treat unassessed records as negative,
report Phase 4-D yield, claim cluster-diverse was most sequence-diverse, call
the Phase 3 rule preregistered, or hide null/contradictory results.
"""
    write_text(ROOT / "THESIS_RESULTS_DISCUSSION_HANDOFF.md", text)


def reporting_readme() -> None:
    text = f"""# Thesis Reporting Derived Evidence

Generated: `{CREATED_AT}`

Script: `{SCRIPT_NAME}`

Repository branch: `{BRANCH}`

Commit: `{COMMIT_SHA}`

The repository worktree was dirty and contained unresolved conflicts. The
commit is recorded for orientation, while each canonical source and derived
file is additionally identified by path and the final package is hashable.

No raw scientific output, checkpoint, selected-candidate archive, or completed
simulation was modified. This directory contains only reproducible derived
summaries and provenance.

`DERIVED_SUMMARY_PROVENANCE.csv` records source files, formulas, filters,
denominators, script, commit, and timestamp for every generated file.
"""
    write_text(REPORTING / "README.md", text)


def main() -> None:
    REPORTING.mkdir(parents=True, exist_ok=True)
    metadata = thesis_metadata()
    _p1_folds, p1_summary = phase1_summaries()
    _p2_folds, p2_summary = phase2_summaries()
    p3_rounds, p3_branches, p3_terminal = phase3_summaries()
    selection = phase3_selection_summary()
    shared_cgmd_inventory(p3_rounds)
    _p4_rows, p4_summary = phase4_summaries()
    phase4d_summary()
    _p5_folds, p5_summary = phase5_summaries()
    cross_phase = cross_phase_matrix()
    superseded_and_unresolved()
    canonical_index()
    figure_table_plan()
    results_discussion_handoff(
        metadata,
        p1_summary,
        p2_summary,
        p3_branches,
        p3_terminal,
        p4_summary,
        p5_summary,
        selection,
        cross_phase,
    )
    reporting_readme()
    provenance_path = REPORTING / "DERIVED_SUMMARY_PROVENANCE.csv"
    write_csv(provenance_path, PROVENANCE)
    manifest = []
    for path in sorted(
        list(REPORTING.rglob("*"))
        + [
            ROOT / "THESIS_RESULTS_DISCUSSION_HANDOFF.md",
            ROOT / "THESIS_CANONICAL_RESULTS_INDEX.md",
            ROOT / "THESIS_FIGURE_TABLE_PLAN.md",
        ]
    ):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest.append(
                {
                    "file": rel(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(REPORTING / "SHA256_MANIFEST.csv", manifest)
    print(
        json.dumps(
            {
                "created_at": CREATED_AT,
                "derived_files": len([path for path in REPORTING.iterdir() if path.is_file()]),
                "handoff": "THESIS_RESULTS_DISCUSSION_HANDOFF.md",
                "canonical_index": "THESIS_CANONICAL_RESULTS_INDEX.md",
                "figure_plan": "THESIS_FIGURE_TABLE_PLAN.md",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
