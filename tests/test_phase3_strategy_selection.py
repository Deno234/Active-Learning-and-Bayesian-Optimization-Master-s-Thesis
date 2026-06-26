from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from active_learning_thesis.config import THESIS_FULL_REPLAY_STRATEGIES
from active_learning_thesis.phase3_strategy_selection import (
    Phase3SelectionOptions,
    compute_overlap,
    jaccard,
    random_floor_status,
    run_phase3_strategy_selection,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Phase3StrategySelectionTests(unittest.TestCase):
    def _make_phase2_fixture(self, root: Path, *, include_acquisition_log: bool = True, ucb_below_random: bool = False) -> Path:
        phase2 = root / "02_replay"
        benchmark = phase2 / "benchmark"
        ablation = phase2 / "ablation"
        strategies = list(THESIS_FULL_REPLAY_STRATEGIES)
        perf = {
            "random": 0.60,
            "ensemble_mean": 0.69,
            "similarity_penalized_mean": 0.74,
            "predictive_entropy": 0.88,
            "ensemble_mi": 0.83,
            "ucb": 0.73,
            "family_qbc": 0.87,
            "cluster_diverse_representative": 0.82,
            "oed_logdet": 0.80,
            "hybrid_mi_diverse": 0.85,
        }
        if ucb_below_random:
            perf["random"] = 0.74
            perf["ucb"] = 0.73
        summary_rows = []
        round_rows = []
        paired_rows = []
        label_rows = []
        selected_rows = []
        acquisition_rows = []
        selections = {
            "random": ["RAA", "RBB", "RCC"],
            "ensemble_mean": ["AAA", "AAB", "AAC"],
            "similarity_penalized_mean": ["VAA", "VAB", "VAC"],
            "predictive_entropy": ["AAA", "AAB", "AAD"],
            "ensemble_mi": ["AAA", "AAC", "AAE"],
            "ucb": ["AAA", "AAF", "AAG"],
            "family_qbc": ["AAA", "AAB", "AAH"],
            "cluster_diverse_representative": ["VAA", "WBB", "XCC"],
            "oed_logdet": ["VAA", "YDD", "ZEE"],
            "hybrid_mi_diverse": ["LMM", "NQQ", "PRR"],
        }
        for seed in (10, 40):
            for strategy in strategies:
                base = perf[strategy] - 0.10 + (0.01 if seed == 40 else 0.0)
                final = perf[strategy] + (0.01 if seed == 40 else 0.0)
                for dataset in ("holdout", "validation"):
                    summary_rows.append(
                        {
                            "setup": "ensemble_calibrated",
                            "initial_label_count": seed,
                            "strategy": strategy,
                            "evaluation_dataset": dataset,
                            "mean_AULC_F1": base - 0.50,
                            "std_AULC_F1": 0.01,
                            "mean_final_F1": final,
                            "mean_final_PR_AUC": final - 0.02,
                            "mean_final_ROC_AUC": final + 0.02,
                            "mean_final_Brier": 1.0 - final,
                            "mean_final_ECE_10": 0.10,
                        }
                    )
                    for fold in (1, 2):
                        round_rows.append(
                            {
                                "setup": "ensemble_calibrated",
                                "initial_label_count": seed,
                                "replay_seed_size": seed,
                                "outer_fold_id": fold,
                                "run_seed": fold,
                                "strategy": strategy,
                                "evaluation_dataset": dataset,
                                "round_id": 0,
                                "labeled_count": seed,
                                "f1": base + fold * 0.001,
                                "pr_auc": base,
                                "roc_auc": base,
                                "brier_score": 1.0 - base,
                                "ece_10": 0.10,
                            }
                        )
                        round_rows.append(
                            {
                                "setup": "ensemble_calibrated",
                                "initial_label_count": seed,
                                "replay_seed_size": seed,
                                "outer_fold_id": fold,
                                "run_seed": fold,
                                "strategy": strategy,
                                "evaluation_dataset": dataset,
                                "round_id": 1,
                                "labeled_count": seed + 5,
                                "f1": final + fold * 0.001,
                                "pr_auc": final - 0.02,
                                "roc_auc": final + 0.02,
                                "brier_score": 1.0 - final,
                                "ece_10": 0.08,
                            }
                        )
                for fold in (1, 2):
                    paired_rows.append(
                        {
                            "setup": "ensemble_calibrated",
                            "initial_label_count": seed,
                            "outer_fold_id": fold,
                            "strategy": strategy,
                            "evaluation_dataset": "holdout",
                            "random_final_F1": perf["random"],
                            "strategy_final_F1": perf[strategy],
                            "final_F1_delta_vs_random": perf[strategy] - perf["random"],
                            "win_vs_random": str(perf[strategy] > perf["random"]),
                        }
                    )
                for target in ("0.8", "0.84", "0.86"):
                    reached = 2 if perf[strategy] >= float(target) else 0
                    label_rows.append(
                        {
                            "setup": "ensemble_calibrated",
                            "initial_label_count": seed,
                            "strategy": strategy,
                            "evaluation_dataset": "holdout",
                            "target_f1": target,
                            "n_repeats": 2,
                            "mean_labels_to_target": seed + (4 if strategy == "predictive_entropy" else 7),
                            "median_labels_to_target": seed + 5,
                            "reached_count": reached,
                        }
                    )
                for fold in (1, 2):
                    for rank, sequence in enumerate(selections[strategy], start=1):
                        selected_rows.append(
                            {
                                "setup": "ensemble_calibrated",
                                "initial_label_count": seed,
                                "replay_seed_size": seed,
                                "outer_fold_id": fold,
                                "run_seed": fold,
                                "strategy": strategy,
                                "round_id": rank,
                                "selection_rank": rank,
                                "sequence": f"{sequence}{fold}",
                                "label": "1" if rank % 2 else "0",
                                "acquisition_score": 1.0 / rank,
                            }
                        )
                        if include_acquisition_log:
                            acquisition_rows.append(
                                {
                                    "setup": "ensemble_calibrated",
                                    "initial_label_count": seed,
                                    "replay_seed_size": seed,
                                    "outer_fold_id": fold,
                                    "run_seed": fold,
                                    "strategy": strategy,
                                    "round_id": rank,
                                    "sequence": f"{sequence}{fold}",
                                    "selected": "true",
                                    "label_revealed_after_selection": "1" if rank % 2 else "0",
                                    "acquisition_score": 1.0 / rank,
                                    "distance_to_labeled": 0.2 + rank / 10,
                                }
                            )
        _write_csv(benchmark / "strategy_summary.csv", summary_rows)
        _write_csv(benchmark / "per_run_round_metrics.csv", round_rows)
        _write_csv(benchmark / "learning_curves.csv", round_rows)
        _write_csv(benchmark / "paired_vs_random.csv", paired_rows)
        _write_csv(benchmark / "labels_to_target_summary.csv", label_rows)
        _write_csv(benchmark / "per_run_selected_sequences.csv", selected_rows)
        if include_acquisition_log:
            _write_csv(benchmark / "per_run_acquisition_log.csv", acquisition_rows)
        _write_csv(benchmark / "strategy_compatibility_matrix.csv", [{"strategy": "random", "compatible": "true"}])
        _write_csv(ablation / "ablation_summary.csv", [{"strategy": "random", "note": "fixture"}])
        _write_csv(ablation / "strategy_compatibility_matrix.csv", [{"strategy": "random", "compatible": "true"}])
        return phase2

    def test_jaccard_overlap_calculation(self):
        self.assertAlmostEqual(jaccard({"A", "B"}, {"B", "C"}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_pairwise_overlap_matrix_symmetry(self):
        rows = [
            {"setup": "ensemble_calibrated", "initial_label_count": "10", "outer_fold_id": "1", "run_seed": "1", "strategy": "a", "sequence": "A"},
            {"setup": "ensemble_calibrated", "initial_label_count": "10", "outer_fold_id": "1", "run_seed": "1", "strategy": "a", "sequence": "B"},
            {"setup": "ensemble_calibrated", "initial_label_count": "10", "outer_fold_id": "1", "run_seed": "1", "strategy": "b", "sequence": "B"},
            {"setup": "ensemble_calibrated", "initial_label_count": "10", "outer_fold_id": "1", "run_seed": "1", "strategy": "b", "sequence": "C"},
        ]
        _pair_rows, matrices = compute_overlap(rows, ["a", "b"])
        self.assertAlmostEqual(matrices["combined"]["a"]["b"], matrices["combined"]["b"]["a"])
        self.assertAlmostEqual(matrices["combined"]["a"]["b"], 1 / 3)

    def test_report_outputs_and_recommendation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2 = self._make_phase2_fixture(root)
            before = _file_hashes(phase2)
            output = root / "03_real_al_strategy_selection"
            summary = run_phase3_strategy_selection(
                Phase3SelectionOptions(phase2_root=phase2, output_root=output, top_k=3, exclude=("random",))
            )
            after = _file_hashes(phase2)
            self.assertEqual(before, after)
            self.assertEqual(len(summary["recommended_strategies"]), 3)
            self.assertNotIn("random", summary["recommended_strategies"])
            for filename in (
                "strategy_selection_summary.csv",
                "strategy_selection_summary.json",
                "strategy_overlap_matrix.csv",
                "strategy_overlap_matrix.json",
                "strategy_jaccard_heatmap.png",
                "strategy_performance_vs_diversity.csv",
                "strategy_trio_comparison.csv",
                "strategy_rank_by_fold.csv",
                "strategy_labels_to_target.csv",
                "strategy_positive_discovery.csv",
                "real_al_strategy_recommendation.md",
            ):
                self.assertTrue((output / filename).exists(), filename)
            md = (output / "real_al_strategy_recommendation.md").read_text(encoding="utf-8")
            self.assertIn("Final human recommendation", md)
            self.assertIn("Strategy | Role | Replay evidence | Overlap warning | Diversity evidence | Practical cost | Decision", md)
            with (output / "strategy_selection_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            random_row = next(row for row in rows if row["strategy"] == "random")
            self.assertEqual(random_row["decision"], "baseline/control")
            self.assertIn("seed_10_holdout_mean_AULC_F1_by_labeled_count", random_row)
            self.assertIn("seed_40_holdout_mean_AULC_F1_by_labeled_count", random_row)
            self.assertIn("mean_holdout_AULC_F1_by_labeled_count", random_row)
            self.assertIn("mean_holdout_AULC_by_round", random_row)
            self.assertIn("delta_vs_random_AULC", random_row)
            with (output / "strategy_trio_comparison.csv").open(encoding="utf-8") as handle:
                trio_rows = list(csv.DictReader(handle))
            self.assertEqual({row["trio"] for row in trio_rows}, {
                "Trio A - performance + QBC + diversity",
                "Trio B - lower-overlap but weaker",
                "Trio C - ensemble uncertainty alternative",
            })
            heatmap_bytes = (output / "strategy_jaccard_heatmap.png").read_bytes()
            width = int.from_bytes(heatmap_bytes[16:20], "big")
            height = int.from_bytes(heatmap_bytes[20:24], "big")
            self.assertGreaterEqual(width, 1200)
            self.assertGreaterEqual(height, 1200)

    def test_aulc_and_not_reached_labels_are_reported_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2 = self._make_phase2_fixture(root)
            output = root / "out"
            run_phase3_strategy_selection(Phase3SelectionOptions(phase2_root=phase2, output_root=output))
            with (output / "strategy_rank_by_fold.csv").open(encoding="utf-8") as handle:
                rank_rows = list(csv.DictReader(handle))
            self.assertIn("AULC_F1_by_labeled_count", rank_rows[0])
            self.assertIn("AULC_by_round", rank_rows[0])
            with (output / "strategy_labels_to_target.csv").open(encoding="utf-8") as handle:
                label_rows = list(csv.DictReader(handle))
            random_086 = next(
                row for row in label_rows if row["strategy"] == "random" and row["initial_label_count"] == "10" and row["target_f1"] == "0.86"
            )
            entropy_086 = next(
                row for row in label_rows if row["strategy"] == "predictive_entropy" and row["initial_label_count"] == "10" and row["target_f1"] == "0.86"
            )
            self.assertGreater(float(random_086["labels_to_target_rank_value"]), float(entropy_086["labels_to_target_rank_value"]))

    def test_missing_acquisition_log_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2 = self._make_phase2_fixture(root, include_acquisition_log=False)
            output = root / "out"
            summary = run_phase3_strategy_selection(Phase3SelectionOptions(phase2_root=phase2, output_root=output))
            self.assertTrue((output / "real_al_strategy_recommendation.md").exists())
            self.assertTrue(any("Acquisition log unavailable" in item for item in summary["limitations"]))

    def test_non_comparable_acquisition_scores_skip_score_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2 = self._make_phase2_fixture(root, include_acquisition_log=True)
            output = root / "out"
            summary = run_phase3_strategy_selection(Phase3SelectionOptions(phase2_root=phase2, output_root=output))
            self.assertTrue(any("Raw acquisition score correlation was skipped" in item for item in summary["limitations"]))
            payload = json.loads((output / "strategy_selection_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["acquisition_similarity"]["available"])

    def test_random_floor_tolerance_status(self):
        self.assertEqual(random_floor_status(0.00001), "meaningfully better than random")
        self.assertEqual(random_floor_status(0.0), "passes floor by tolerance")
        self.assertEqual(random_floor_status(-0.0000005), "passes floor by tolerance")
        self.assertEqual(random_floor_status(-0.00001), "below random")

    def test_below_random_ucb_is_exploratory_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase2 = self._make_phase2_fixture(root, ucb_below_random=True)
            output = root / "out"
            summary = run_phase3_strategy_selection(Phase3SelectionOptions(phase2_root=phase2, output_root=output))
            self.assertNotIn("ucb", summary["recommended_strategies"])
            with (output / "strategy_selection_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            ucb_row = next(row for row in rows if row["strategy"] == "ucb")
            self.assertEqual(ucb_row["decision"], "exploratory/control")
            self.assertEqual(ucb_row["recommendation_eligibility"], "exploratory/control")
            self.assertEqual(ucb_row["random_floor_status"], "below random")
            self.assertLess(float(ucb_row["delta_vs_random_AULC"]), 0.0)
            md = (output / "real_al_strategy_recommendation.md").read_text(encoding="utf-8")
            self.assertIn("Performance Floor vs Random", md)
            self.assertIn("Trio Comparison", md)


if __name__ == "__main__":
    unittest.main()
