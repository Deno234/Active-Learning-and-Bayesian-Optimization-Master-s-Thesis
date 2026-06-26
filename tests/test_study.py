from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

from active_learning_thesis.metrics import evaluate_binary_classifier
from active_learning_thesis.predictive import ManagedModel, score_sequences_with_ensemble
from active_learning_thesis.study import (
    area_under_learning_curve,
    compare_studies,
    labels_to_reach_target,
    run_study,
    summarize_study,
)


class StudyEvidenceTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"{name}_{uuid.uuid4().hex}_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _write_replay_summary(
        self,
        run_root: Path,
        run_name: str,
        strategy: str,
        points: list[dict[str, float | int]],
    ) -> None:
        strategy_dir = run_root / run_name / "replay" / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "summary.json").write_text(
            json.dumps(points, indent=2),
            encoding="utf-8",
        )

    def _write_study_manifest_with_summary(
        self,
        run_root: Path,
        study_name: str,
        metric: str,
        seeds: list[int],
        rows_by_seed_strategy: dict[int, dict[str, dict[str, object]]],
        target: float | None = None,
    ) -> Path:
        study_dir = run_root / "_studies" / study_name
        evidence_dir = study_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        summary_path = evidence_dir / f"{metric}_run_strategy_summary.csv"
        runs = []
        rows = []
        for index, seed in enumerate(seeds, start=1):
            run_name = f"{study_name}_seed_{index:02d}_{seed}"
            run_dir = run_root / run_name
            runs.append(
                {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "random_seed": seed,
                    "status": "replay_complete",
                }
            )
            for strategy, values in rows_by_seed_strategy[seed].items():
                rows.append(
                    {
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "strategy": strategy,
                        "metric": metric,
                        "higher_is_better": metric != "brier_score",
                        "round_count": 2,
                        "first_round_id": 0,
                        "final_round_id": 1,
                        "first_labeled_count": 40,
                        "final_labeled_count": 45,
                        "first_metric": values.get("first_metric", values["final_metric"]),
                        "final_metric": values["final_metric"],
                        "best_metric": values.get("best_metric", values["final_metric"]),
                        "aulc_metric": values["aulc_metric"],
                        "target": target,
                        "target_reached": values.get("target_reached", False),
                        "labels_to_target": values.get("labels_to_target"),
                    }
                )

        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
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
            )
            writer.writeheader()
            writer.writerows(rows)

        manifest_path = study_dir / "study_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "study_name": study_name,
                    "study_slug": study_name,
                    "run_root": str(run_root),
                    "study_dir": str(study_dir),
                    "manifest_path": str(manifest_path),
                    "status": "completed",
                    "config": {"target": target},
                    "runs": runs,
                    "summary": {
                        "output_dir": str(evidence_dir),
                        "outputs": {"run_strategy_summary": str(summary_path)},
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_learning_curve_area_and_target_label_count(self):
        points = [
            {"round_id": 0, "labeled_count": 40, "f1": 0.70},
            {"round_id": 1, "labeled_count": 45, "f1": 0.80},
            {"round_id": 2, "labeled_count": 50, "f1": 0.90},
        ]

        self.assertAlmostEqual(area_under_learning_curve(points, "f1"), 0.80)
        self.assertEqual(labels_to_reach_target(points, "f1", 0.85), 50)

    def test_summarize_study_ranks_strategies_and_writes_outputs(self):
        run_root = self._scratch_dir("study_runs")
        output_dir = run_root / "study_out"

        self._write_replay_summary(
            run_root,
            "seed_a",
            "random",
            [
                {"round_id": 0, "labeled_count": 40, "f1": 0.70},
                {"round_id": 1, "labeled_count": 45, "f1": 0.75},
            ],
        )
        self._write_replay_summary(
            run_root,
            "seed_a",
            "ensemble_mi",
            [
                {"round_id": 0, "labeled_count": 40, "f1": 0.70},
                {"round_id": 1, "labeled_count": 45, "f1": 0.84},
            ],
        )
        self._write_replay_summary(
            run_root,
            "seed_b",
            "random",
            [
                {"round_id": 0, "labeled_count": 40, "f1": 0.72},
                {"round_id": 1, "labeled_count": 45, "f1": 0.76},
            ],
        )
        self._write_replay_summary(
            run_root,
            "seed_b",
            "ensemble_mi",
            [
                {"round_id": 0, "labeled_count": 40, "f1": 0.72},
                {"round_id": 1, "labeled_count": 45, "f1": 0.86},
            ],
        )

        summary = summarize_study(
            run_root,
            output_dir=output_dir,
            metric="f1",
            target=0.85,
        )

        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["strategy_count"], 2)
        self.assertEqual(summary["best_strategy_by_aulc"], "ensemble_mi")
        strategy_rows = {row["strategy"]: row for row in summary["strategy_rows"]}
        self.assertGreater(
            strategy_rows["ensemble_mi"]["aulc_mean"],
            strategy_rows["random"]["aulc_mean"],
        )
        self.assertEqual(strategy_rows["ensemble_mi"]["target_reached_runs"], 1)
        self.assertEqual(len(summary["paired_vs_random_rows"]), 2)
        self.assertTrue((output_dir / "f1_strategy_summary.csv").exists())
        self.assertTrue((output_dir / "f1_run_strategy_summary.csv").exists())
        self.assertTrue((output_dir / "f1_paired_vs_random.csv").exists())
        self.assertTrue((output_dir / "f1_study_summary.json").exists())

    def test_evaluation_includes_calibration_metrics(self):
        metrics = evaluate_binary_classifier(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
        )

        self.assertIn("brier_score", metrics)
        self.assertIn("log_loss", metrics)
        self.assertIn("ece_10", metrics)
        self.assertIn("mce_10", metrics)
        self.assertLess(metrics["brier_score"], 0.1)
        self.assertGreaterEqual(metrics["ece_10"], 0.0)

    def test_run_study_dry_run_writes_resumable_manifest(self):
        run_root = self._scratch_dir("study_dry_run")

        manifest = run_study(
            study_name="Thesis Full Replay",
            run_root=run_root,
            seed_count=2,
            seed_start=101,
            seed_step=10,
            epochs=70,
            max_rounds=10,
            dry_run=True,
        )

        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["run_count"], 2)
        self.assertEqual(manifest["completed_run_count"], 0)
        self.assertEqual(manifest["config"]["seeds"], [101, 111])
        self.assertTrue(manifest["config"]["use_calibrated_acquisition"])
        self.assertTrue(Path(str(manifest["manifest_path"])).exists())
        self.assertTrue(
            all(str(row["status"]) == "planned" for row in manifest["runs"])
        )

    def test_run_study_can_plan_raw_acquisition_ablation(self):
        run_root = self._scratch_dir("study_raw_ablation")

        manifest = run_study(
            study_name="raw-ablation",
            run_root=run_root,
            seed_count=1,
            use_calibrated_acquisition=False,
            dry_run=True,
        )

        self.assertFalse(manifest["config"]["use_calibrated_acquisition"])

    def test_run_study_initializes_replays_and_summarizes_completed_runs(self):
        run_root = self._scratch_dir("study_run")

        def fake_init(config):
            config.save(config.run_dir / "config.json")
            return config.run_dir

        def fake_replay(run_dir, strategies):
            for strategy in strategies:
                summary_dir = Path(run_dir) / "replay" / strategy
                summary_dir.mkdir(parents=True, exist_ok=True)
                (summary_dir / "summary.json").write_text(
                    json.dumps(
                        [
                            {"round_id": 0, "labeled_count": 40, "f1": 0.70},
                            {"round_id": 1, "labeled_count": 45, "f1": 0.82},
                        ]
                    ),
                    encoding="utf-8",
                )
            return {strategy: [] for strategy in strategies}

        with mock.patch(
            "active_learning_thesis.study.init_run",
            side_effect=fake_init,
        ) as init_mock, mock.patch(
            "active_learning_thesis.study.run_replay",
            side_effect=fake_replay,
        ) as replay_mock, mock.patch(
            "active_learning_thesis.study.summarize_study",
            return_value={
                "strategy_count": 1,
                "outputs": {"strategy_summary": "strategy.csv"},
            },
        ) as summary_mock:
            manifest = run_study(
                study_name="study",
                run_root=run_root,
                seed_count=2,
                seed_start=7,
                seed_step=1,
                epochs=1,
                max_rounds=1,
                replay_strategies=["random"],
                target=0.8,
            )

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["completed_run_count"], 2)
        self.assertEqual(init_mock.call_count, 2)
        self.assertEqual(replay_mock.call_count, 2)
        summary_mock.assert_called_once()
        self.assertEqual(
            summary_mock.call_args.kwargs["run_names"],
            ["study_seed_01_7", "study_seed_02_8"],
        )

    def test_run_study_resume_skips_completed_replay(self):
        run_root = self._scratch_dir("study_resume")

        manifest = run_study(
            study_name="resume",
            run_root=run_root,
            seed_count=1,
            seed_start=42,
            epochs=1,
            max_rounds=1,
            replay_strategies=["random"],
            dry_run=True,
        )
        run_name = str(manifest["runs"][0]["run_name"])
        run_dir = run_root / run_name
        from active_learning_thesis.config import RunConfig

        RunConfig(
            run_name=run_name,
            output_root=str(run_root),
            random_seed=42,
            epochs=1,
            max_rounds=1,
            replay_strategies=["random"],
        ).save(run_dir / "config.json")
        replay_dir = run_dir / "replay" / "random"
        replay_dir.mkdir(parents=True, exist_ok=True)
        (replay_dir / "summary.json").write_text(
            json.dumps([{"round_id": 0, "labeled_count": 40, "f1": 0.7}]),
            encoding="utf-8",
        )

        with mock.patch("active_learning_thesis.study.init_run") as init_mock, mock.patch(
            "active_learning_thesis.study.run_replay"
        ) as replay_mock:
            resumed = run_study(
                study_name="resume",
                run_root=run_root,
                seed_count=1,
                seed_start=42,
                epochs=1,
                max_rounds=1,
                replay_strategies=["random"],
                summarize=False,
            )

        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["runs"][0]["init_action"], "reused-existing-run")
        self.assertEqual(resumed["runs"][0]["replay_action"], "reused-existing-replay")
        init_mock.assert_not_called()
        replay_mock.assert_not_called()

    def test_compare_studies_pairs_matched_seeds_and_writes_outputs(self):
        run_root = self._scratch_dir("study_compare")
        seeds = [101, 111]

        self._write_study_manifest_with_summary(
            run_root,
            "raw",
            "f1",
            seeds,
            {
                101: {
                    "random": {"final_metric": 0.74, "aulc_metric": 0.72, "labels_to_target": 50},
                    "ensemble_mi": {"final_metric": 0.82, "aulc_metric": 0.78, "labels_to_target": 45},
                },
                111: {
                    "random": {"final_metric": 0.75, "aulc_metric": 0.73, "labels_to_target": 50},
                    "ensemble_mi": {"final_metric": 0.83, "aulc_metric": 0.79, "labels_to_target": 45},
                },
            },
            target=0.85,
        )
        self._write_study_manifest_with_summary(
            run_root,
            "calibrated",
            "f1",
            seeds,
            {
                101: {
                    "random": {"final_metric": 0.75, "aulc_metric": 0.73, "labels_to_target": 49},
                    "ensemble_mi": {"final_metric": 0.87, "aulc_metric": 0.84, "labels_to_target": 40},
                },
                111: {
                    "random": {"final_metric": 0.76, "aulc_metric": 0.74, "labels_to_target": 49},
                    "ensemble_mi": {"final_metric": 0.88, "aulc_metric": 0.85, "labels_to_target": 40},
                },
            },
            target=0.85,
        )

        comparison = compare_studies(
            run_root,
            baseline_study="raw",
            candidate_study="calibrated",
            metric="f1",
            target=0.85,
        )

        self.assertEqual(comparison["paired_count"], 4)
        self.assertEqual(comparison["strategy_count"], 2)
        self.assertEqual(comparison["best_strategy_by_aulc_advantage"], "ensemble_mi")
        ensemble_row = {
            row["strategy"]: row for row in comparison["strategy_rows"]
        }["ensemble_mi"]
        self.assertAlmostEqual(ensemble_row["aulc_advantage_mean"], 0.06)
        self.assertEqual(ensemble_row["labels_saved_to_target_median"], 5.0)
        self.assertTrue(Path(comparison["outputs"]["paired_comparison"]).exists())
        self.assertTrue(Path(comparison["outputs"]["strategy_summary"]).exists())
        self.assertTrue(Path(comparison["outputs"]["comparison_summary"]).exists())
        self.assertTrue(Path(comparison["outputs"]["thesis_narrative"]).exists())
        self.assertIn("Best strategy by mean AULC advantage", comparison["narrative"])

    def test_compare_studies_reports_positive_advantage_for_lower_is_better_metrics(self):
        run_root = self._scratch_dir("study_compare_lower")
        seeds = [7]

        self._write_study_manifest_with_summary(
            run_root,
            "raw",
            "brier_score",
            seeds,
            {
                7: {
                    "ensemble_mi": {"final_metric": 0.20, "aulc_metric": 0.22, "best_metric": 0.18},
                },
            },
        )
        self._write_study_manifest_with_summary(
            run_root,
            "calibrated",
            "brier_score",
            seeds,
            {
                7: {
                    "ensemble_mi": {"final_metric": 0.12, "aulc_metric": 0.14, "best_metric": 0.11},
                },
            },
        )

        comparison = compare_studies(
            run_root,
            baseline_study="raw",
            candidate_study="calibrated",
            metric="brier_score",
        )

        self.assertFalse(comparison["higher_is_better"])
        self.assertEqual(comparison["paired_count"], 1)
        self.assertGreater(comparison["paired_rows"][0]["final_advantage"], 0.0)
        self.assertGreater(comparison["paired_rows"][0]["aulc_advantage"], 0.0)

    def test_compare_studies_rejects_same_manifest(self):
        run_root = self._scratch_dir("study_compare_same")
        self._write_study_manifest_with_summary(
            run_root,
            "baseline",
            "f1",
            [7],
            {
                7: {
                    "ensemble_mi": {"final_metric": 0.82, "aulc_metric": 0.78},
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "must be different"):
            compare_studies(
                run_root,
                baseline_study="baseline",
                candidate_study="baseline",
                metric="f1",
            )

    def test_ensemble_scoring_can_return_calibrated_and_raw_probabilities(self):
        ensemble = [
            ManagedModel(
                "AP_SP",
                11,
                None,
                object(),
                calibration={
                    "method": "platt_logit",
                    "coef": 0.0,
                    "intercept": 0.0,
                    "center": 0.0,
                    "scale": 1.0,
                },
            )
        ]
        with mock.patch(
            "active_learning_thesis.predictive._prepare_inference_tensors",
            return_value="prepared_inputs",
        ), mock.patch(
            "active_learning_thesis.predictive._predict_probabilities_from_inputs",
            return_value=[0.2, 0.8],
        ):
            scored = score_sequences_with_ensemble(
                ensemble,
                ["AAAAA", "CCCCC"],
                use_calibration=True,
                include_raw=True,
            )

        self.assertEqual(scored["pred_mean"].tolist(), [0.5, 0.5])
        self.assertEqual(scored["raw_pred_mean"].tolist(), [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
