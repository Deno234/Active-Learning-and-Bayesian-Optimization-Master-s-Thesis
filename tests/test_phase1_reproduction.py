from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from active_learning_thesis import phase1_reproduction as p1


class Phase1ReproductionTests(unittest.TestCase):
    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _write_dataset(self, root: Path, count_per_class: int = 10) -> Path:
        path = root / "data_SA.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["peptide_sequence", "peptide_label"], delimiter=";")
            writer.writeheader()
            for index in range(count_per_class):
                writer.writerow({"peptide_sequence": f"SA{index:02d}A", "peptide_label": "1"})
                writer.writerow({"peptide_sequence": f"NS{index:02d}G", "peptide_label": "0"})
        return path

    def test_models_accept_multi_and_single_shortcut(self):
        self.assertEqual(p1.normalize_models(["AP", "SP"]), ("AP", "SP"))
        self.assertEqual(p1.normalize_models(model="AP_SP"), ("AP_SP",))
        self.assertEqual(p1.normalize_models(["AP"], "AP_SP"), ("AP", "AP_SP"))
        with self.assertRaisesRegex(ValueError, "Invalid Phase 1 model"):
            p1.normalize_models(["NOPE"])

    def test_dataset_sanity_and_preprocessing_exports_on_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_dataset(root)
            options = p1.Phase1Options(
                output_root=root / "out",
                dataset_path=dataset,
                models=("AP", "AP_SP"),
                skip_heavy=True,
            )
            p1.ensure_phase1_dirs(options.output_root)
            summary = p1.run_sanity(options)
            self.assertEqual(summary["status"], "complete")
            sanity = self._read_csv(options.output_root / "tables" / "dataset_sanity.csv")
            self.assertEqual(sanity[0]["total peptides"], "20")
            self.assertEqual(sanity[0]["SA"], "10")
            preprocessing = self._read_csv(options.output_root / "tables" / "preprocessing_shapes.csv")
            self.assertEqual([row["model name"] for row in preprocessing], ["AP", "AP_SP"])

    def test_nested_fold_manifests_repeat_outer_test_per_inner_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_dataset(root)
            options = p1.Phase1Options(output_root=root / "out", dataset_path=dataset, models=("AP_SP",))
            p1.ensure_phase1_dirs(options.output_root)
            p1.write_nested_cv_folds(options)
            rows = self._read_csv(options.output_root / "folds" / "nested_cv_fold_assignments.csv")
            self.assertEqual(len(rows), 20 * 25)
            pair_rows = [row for row in rows if row["outer_fold_id"] == "1" and row["inner_fold_id"] == "1"]
            role_counts = {role: sum(1 for row in pair_rows if row["role_for_this_outer_inner_pair"] == role) for role in ["train", "validation", "test"]}
            self.assertEqual(role_counts, {"train": 12, "validation": 4, "test": 4})
            manifest = json.loads((options.output_root / "folds" / "replay_manifest_outer_1_inner_1.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_mode"], "paper_nested_cv_replay")
            self.assertEqual(len(manifest["rows"]), 20)
            self.assertIn("holdout", {row["split"] for row in manifest["rows"]})

    def test_hyperparameter_selection_uses_inner_validation_loss_only(self):
        inner_rows = [
            {
                "model": "AP_SP",
                "outer_fold_id": 1,
                "inner_fold_id": 1,
                "num_cells": 32,
                "kernel_size": "4",
                "validation_loss": 0.2,
                "validation_accuracy": 0.7,
                "validation_f1_fixed_0_5": 0.6,
            },
            {
                "model": "AP_SP",
                "outer_fold_id": 1,
                "inner_fold_id": 1,
                "num_cells": 64,
                "kernel_size": "8",
                "validation_loss": 0.5,
                "validation_accuracy": 0.99,
                "validation_f1_fixed_0_5": 0.99,
                "outer_test_accuracy": 1.0,
            },
        ]
        frozen, rows = p1.summarize_hyperparameters(inner_rows, ("AP_SP",))
        self.assertEqual(frozen["AP_SP"]["num_cells"], 32)
        self.assertEqual(frozen["AP_SP"]["kernel_size"], "4")
        self.assertIn("inner validation", rows[0]["Selection rule"])

    def test_single_model_nested_cv_writes_model_specific_outputs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = p1.Phase1Options(output_root=root / "out", models=("AP_SP",))
            p1.ensure_phase1_dirs(options.output_root)
            inner_rows = [
                {
                    "model": "AP_SP",
                    "outer_fold_id": 1,
                    "inner_fold_id": 1,
                    "num_cells": 32,
                    "kernel_size": "4",
                    "validation_loss": 0.2,
                    "validation_accuracy": 0.8,
                    "validation_f1_fixed_0_5": 0.7,
                }
            ]
            outer_rows = [
                {
                    "model": "AP_SP",
                    "outer_fold_id": 1,
                    "threshold_type": "PR",
                    "threshold_value": 0.5,
                    "Accuracy": 0.8,
                    "F1": 0.7,
                    "ROC-AUC": 0.9,
                    "PR-AUC": 0.85,
                    "gmean": 0.75,
                    "Brier": 0.2,
                    "ECE-10": 0.1,
                    "MCE-10": 0.1,
                    "decision threshold": 0.5,
                }
            ]
            with mock.patch.object(p1, "read_experimental_dataset", return_value=[]), mock.patch.object(
                p1,
                "train_nested_cv_models",
                return_value=(inner_rows, outer_rows),
            ):
                result = p1.run_nested_cv(options)
            self.assertEqual(result["status"], "complete")
            self.assertTrue((options.output_root / "tables" / "nested_cv_inner_results_AP_SP.csv").exists())
            self.assertTrue((options.output_root / "tables" / "nested_cv_outer_predictions_AP_SP.csv").exists())
            self.assertFalse((options.output_root / "tables" / "nested_cv_inner_results.csv").exists())
            self.assertFalse((options.output_root / "frozen_model_config.json").exists())

    def test_thresholds_aggregates_per_model_nested_cv_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            options = p1.Phase1Options(output_root=output_root, models=("AP", "AP_SP"))
            p1.ensure_phase1_dirs(output_root)
            for model_name in options.models:
                p1._write_csv(
                    output_root / "tables" / f"nested_cv_inner_results_{model_name}.csv",
                    [
                        {
                            "model": model_name,
                            "outer_fold_id": 1,
                            "inner_fold_id": 1,
                            "num_cells": 32,
                            "kernel_size": "n/a" if model_name == "AP" else "4",
                            "validation_loss": 0.2,
                            "validation_accuracy": 0.8,
                            "validation_f1_fixed_0_5": 0.7,
                        }
                    ],
                )
                p1._write_csv(
                    output_root / "tables" / f"nested_cv_outer_predictions_{model_name}.csv",
                    [
                        {
                            "model": model_name,
                            "outer_fold_id": 1,
                            "threshold_type": "PR",
                            "threshold_value": 0.5,
                            "Accuracy": 0.8,
                            "F1": 0.7,
                            "ROC-AUC": 0.9,
                            "PR-AUC": 0.85,
                            "gmean": 0.75,
                            "Brier": 0.2,
                            "ECE-10": 0.1,
                            "MCE-10": 0.1,
                            "decision threshold": 0.5,
                        }
                    ],
                )
            result = p1.run_thresholds(options)
            self.assertEqual(result["status"], "complete")
            combined_inner = self._read_csv(output_root / "tables" / "nested_cv_inner_results.csv")
            self.assertEqual({row["model"] for row in combined_inner}, {"AP", "AP_SP"})
            threshold_rows = self._read_csv(output_root / "tables" / "threshold_summary.csv")
            self.assertEqual({row["Model"] for row in threshold_rows}, {"AP", "AP_SP"})

    def test_thresholds_are_selected_from_validation_and_metrics_include_calibration(self):
        validation_truth = np.asarray([1, 1, 0, 0])
        validation_probs = np.asarray([0.9, 0.8, 0.7, 0.1])
        thresholds = p1.select_thresholds_from_validation(validation_truth, validation_probs)
        self.assertIn("PR", thresholds)
        self.assertIn("ROC", thresholds)
        test_truth = np.asarray([1, 0, 0, 1])
        test_probs = np.asarray([0.6, 0.55, 0.2, 0.1])
        metrics = p1.evaluate_binary_classifier(test_truth, test_probs, threshold=thresholds["PR"])
        summary = p1._metrics_for_summary(metrics)
        self.assertIn("Brier", summary)
        self.assertIn("ECE-10", summary)
        self.assertIn("MCE-10", summary)

    def test_generated_similarity_summary_reports_training_and_generated_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = ["AAAA", "AAAT", "GGGG"]
            training = ["AAAA", "CCCC"]
            rows = p1.generated_similarity_summary_rows(generated, training)
            by_sequence = {str(row["Sequence"]): row for row in rows}
            self.assertEqual(by_sequence["AAAA"]["Nearest_training_sequence"], "AAAA")
            self.assertAlmostEqual(float(by_sequence["AAAA"]["Simtrain_max_percent"]), 100.0)
            self.assertEqual(by_sequence["AAAA"]["Nearest_generated_sequence"], "AAAT")
            self.assertAlmostEqual(float(by_sequence["AAAA"]["Simgen_max_percent"]), 75.0)
            self.assertAlmostEqual(p1._needleman_wunsch_identity_percent("AAAA", "AAAT"), 75.0)

            csv_path, json_path = p1.write_generated_similarity_summary(root, generated, training)
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            exported_rows = self._read_csv(csv_path)
            self.assertEqual(len(exported_rows), 3)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["overall"]["method"], "Needleman-Wunsch global identity percent")
            self.assertEqual(payload["overall"]["generated_count"], 3)
            self.assertEqual(payload["overall"]["training_count"], 2)
            self.assertIn("Simtrain_avg_percent", payload["overall"])
            self.assertIn("Simgen_avg_percent", payload["overall"])

    def test_generation_settings_record_configurable_candidate_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = p1.Phase1Options(
                output_root=Path(tmp) / "out",
                generation_target_unique=50,
                generation_minimum_return_count=50,
                generation_ga_max_attempts=25,
            )
            p1.ensure_phase1_dirs(options.output_root)
            settings_path = p1.write_generation_settings(options, skipped=False)
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["target_unique"], 50)
            self.assertEqual(payload["minimum_return_count"], 50)
            self.assertEqual(payload["ga_max_attempts"], 25)
            self.assertTrue(payload["use_similarity_penalty"])
            self.assertTrue(payload["use_length_penalty"])

    def test_status_reports_readiness_and_missing_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            p1.ensure_phase1_dirs(root)
            status = p1.phase1_status(root, ("AP_SP",))
            self.assertFalse(status["ready_for_phase2"])
            self.assertIn("Ready for Phase 2: no", p1.format_status_report(status))
            for relative_paths in p1.STATUS_ARTIFACTS.values():
                for relative in relative_paths:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("x", encoding="utf-8")
            (root / "models" / "AP_SP.h5").write_text("model", encoding="utf-8")
            status = p1.phase1_status(root, ("AP_SP",))
            self.assertTrue(status["ready_for_phase2"])
            self.assertEqual(status["verdict"], "Ready for Phase 2: yes")

    def test_write_supek_pbs_creates_per_model_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = p1.Phase1Options(output_root=root / "out", dataset_path=root / "data.csv", models=("AP", "AP_SP"))
            p1.ensure_phase1_dirs(options.output_root)
            paths = p1.write_supek_pbs_scripts(options)
            names = {path.name for path in paths}
            self.assertIn("phase1_nested_cv_AP.pbs", names)
            self.assertIn("phase1_nested_cv_AP_SP.pbs", names)
            self.assertIn("phase1_train_final_AP.pbs", names)
            self.assertIn("phase1_train_final_AP_SP.pbs", names)

    def test_write_supek_pbs_uses_absolute_log_paths_and_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pbs_root = root / "supek_repo"
            options = p1.Phase1Options(
                output_root=Path("relative_phase1_for_test"),
                dataset_path=Path("SA_ML_predictive/data/data_SA.csv"),
                models=("AP",),
                pbs_repo_root=pbs_root,
            )
            paths = p1.write_supek_pbs_scripts(options)
            try:
                self.assertTrue((options.output_root / "logs" / "supek_runtime").is_dir())
                text = (options.output_root / "logs" / "supek_pbs" / "phase1_nested_cv_AP.pbs").read_text(encoding="utf-8")
                output_lines = [line for line in text.splitlines() if line.startswith("#PBS -o ") or line.startswith("#PBS -e ")]
                self.assertEqual(len(output_lines), 2)
                for line in output_lines:
                    raw_path = line.split(maxsplit=2)[2]
                    self.assertTrue(Path(raw_path).is_absolute(), raw_path)
                    self.assertTrue(raw_path.startswith(pbs_root.as_posix()), raw_path)
                    self.assertNotIn("$PBS_O_WORKDIR", raw_path)
                self.assertIn(f'cd "{pbs_root.as_posix()}"', text)
                self.assertNotIn('cd "$PBS_O_WORKDIR"', text)
                self.assertIn(f"--output-root {p1._shell_quote((pbs_root / options.output_root).as_posix())}", text)
                self.assertIn(f"--dataset-path {p1._shell_quote((pbs_root / options.dataset_path).as_posix())}", text)
            finally:
                for path in paths:
                    self.assertTrue(path.exists())
                shutil.rmtree(options.output_root, ignore_errors=True)

    def test_resource_log_rows_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = p1.Phase1Options(output_root=Path(tmp) / "out")
            p1.ensure_phase1_dirs(options.output_root)
            with p1.resource_logger(options, "nested-cv", "AP") as artifacts:
                artifacts.append("artifact.csv")
            rows = self._read_csv(options.output_root / "logs" / "phase1_resource_log.csv")
            self.assertEqual(rows[0]["step"], "nested-cv")
            self.assertEqual(rows[0]["model"], "AP")
            self.assertIn("artifact.csv", rows[0]["output_artifacts"])

    def test_existing_h5_models_are_backed_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source_models"
            source_dir.mkdir()
            (source_dir / "AP_SP.h5").write_text("weights", encoding="utf-8")
            output_root = root / "out"
            with mock.patch.object(p1, "PREDICTIVE_MODEL_DIR", source_dir):
                copied = p1.backup_existing_models(output_root)
            self.assertEqual(len(copied), 1)
            self.assertTrue((output_root / "backups" / "models_before_phase1" / "AP_SP.h5").exists())

    def test_phase1_skip_heavy_does_not_call_replay_or_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_dataset(root)
            options = p1.Phase1Options(
                output_root=root / "out",
                dataset_path=dataset,
                mode="full",
                models=("AP_SP",),
                skip_heavy=True,
            )
            with mock.patch("active_learning_thesis.workflow.run_replay") as replay, mock.patch("active_learning_thesis.workflow.propose_round") as propose:
                p1.run_phase1_reproduce(options)
            replay.assert_not_called()
            propose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
