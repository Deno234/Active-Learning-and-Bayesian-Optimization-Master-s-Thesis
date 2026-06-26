from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from active_learning_thesis import phase2_replay as p2
from active_learning_thesis.cli import _build_parser


class Phase2ReplayTests(unittest.TestCase):
    def _phase1_root(self, root: Path) -> Path:
        phase1 = root / "01_reproduction"
        (phase1 / "folds").mkdir(parents=True)
        (phase1 / "tables").mkdir()
        (phase1 / "models").mkdir()
        frozen = {
            "AP": {"num_cells": 32, "kernel_size": "n/a"},
            "SP": {"num_cells": 32, "kernel_size": "6"},
            "AP_SP": {"num_cells": 48, "kernel_size": "8"},
            "TSNE_SP": {"num_cells": 48, "kernel_size": "6"},
            "TSNE_AP_SP": {"num_cells": 64, "kernel_size": "6"},
        }
        (phase1 / "frozen_model_config.json").write_text(json.dumps(frozen), encoding="utf-8")
        (phase1 / "tables" / "reproduced_predictive_performance.csv").write_text("Model,F1\nAP_SP,0.86\n", encoding="utf-8")
        rows = []
        for index in range(8):
            rows.append({"split": "holdout", "sequence": f"H{index}", "label": str(index % 2)})
        for index in range(8):
            rows.append({"split": "validation", "sequence": f"V{index}", "label": str(index % 2)})
        for index in range(30):
            rows.append({"split": "train_pool", "sequence": f"T{index}", "label": str(index % 2)})
        manifest = {
            "split_mode": "paper_nested_cv_replay",
            "outer_fold_id": 1,
            "inner_fold_id": 1,
            "rows": [
                {
                    "split_mode": "paper_nested_cv_replay",
                    "split": row["split"],
                    "replay_role": "none",
                    "outer_fold_id": 1,
                    "inner_fold_id": 1,
                    "sequence": row["sequence"],
                    "label": row["label"],
                }
                for row in rows
            ],
        }
        (phase1 / "folds" / "replay_manifest_outer_1_inner_1.json").write_text(json.dumps(manifest), encoding="utf-8")
        return phase1

    def test_required_smoke_command_parses(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "phase2-replay",
                "--mode",
                "smoke",
                "--phase1-root",
                "thesis_results/01_reproduction",
                "--output-root",
                "thesis_results/02_replay",
                "--outer-folds",
                "1",
                "--inner-fold",
                "1",
                "--replay-seed-sizes",
                "10",
                "--batch-size",
                "5",
                "--max-rounds",
                "2",
                "--strategies",
                "random",
                "ensemble_mean",
                "predictive_entropy",
                "--setup",
                "ensemble_calibrated",
            ]
        )
        self.assertEqual(args.command, "phase2-replay")
        self.assertEqual(args.mode, "smoke")
        self.assertEqual(args.replay_seed_sizes, [10])
        self.assertEqual(args.strategies, ["random", "ensemble_mean", "predictive_entropy"])

    def test_manifest_split_construction_is_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase1 = self._phase1_root(Path(tmp))
            manifest = p2.load_replay_manifest(phase1, 1, 1)
            rows = p2.construct_replay_rows(manifest, replay_seed_size=10, run_seed=123)
            p2.validate_no_within_repeat_overlap(rows)
            self.assertEqual(len(rows.replay_seed), 10)
            self.assertEqual({row["label"] for row in rows.replay_seed}, {"0", "1"})
            self.assertEqual(len(rows.replay_hidden), 20)

    def test_split_overlap_validation_catches_within_repeat_overlap(self):
        row = {"sequence": "AAA", "label": "1", "split": "x", "replay_role": "none"}
        rows = p2.ReplayRows(
            holdout=[row],
            validation=[row],
            train_pool=[],
            replay_seed=[],
            replay_hidden=[],
        )
        with self.assertRaisesRegex(ValueError, "Illegal split overlap"):
            p2.validate_no_within_repeat_overlap(rows)

    def test_run_seed_and_member_seeds_are_deterministic_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1 = self._phase1_root(root)
            options = p2.Phase2Options(
                phase1_root=phase1,
                output_root=root / "02_replay",
                mode="smoke",
                outer_folds=(1,),
                inner_fold=1,
                replay_seed_sizes=(10,),
                max_rounds=0,
                strategies=("random",),
                setup="ensemble_calibrated",
                base_seed=100,
            )
            spec = p2.build_run_specs(options)[0]
            self.assertEqual(spec.run_seed, 1210)
            config = p2.config_for_phase2(spec, p2.load_frozen_model_config(phase1))
            self.assertEqual(config.ensemble_seeds[:5], [1210, 1211, 1212, 1213, 1214])

    def test_incompatible_strategies_are_recorded_without_crashing(self):
        rows = p2.compatibility_rows(
            "ablation",
            "single_raw",
            ["random", "ensemble_mi", "nope"],
            replay_seed_size=10,
        )
        by_strategy = {row["strategy"]: row for row in rows}
        self.assertTrue(by_strategy["random"]["compatible"])
        self.assertFalse(by_strategy["ensemble_mi"]["compatible"])
        self.assertEqual(by_strategy["nope"]["skip_reason"], "unsupported_strategy")

    def test_mocked_smoke_run_writes_outputs_and_does_not_touch_phase1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1 = self._phase1_root(root)
            before = sorted(path.relative_to(phase1).as_posix() for path in phase1.rglob("*"))
            options = p2.Phase2Options(
                phase1_root=phase1,
                output_root=root / "02_replay",
                mode="smoke",
                outer_folds=(1,),
                inner_fold=1,
                replay_seed_sizes=(10,),
                batch_size=5,
                max_rounds=1,
                strategies=("random", "ensemble_mean", "predictive_entropy"),
                setup="ensemble_calibrated",
            )

            def fake_score(_ensemble, sequences, include_embeddings=False, use_calibration=True, include_raw=False):
                count = len(sequences)
                probs = np.linspace(0.2, 0.9, count) if count else np.empty(0)
                return {
                    "pred_mean": probs,
                    "pred_std": np.zeros(count),
                    "pred_entropy": np.linspace(0.9, 0.1, count) if count else np.empty(0),
                    "pred_expected_entropy": np.zeros(count),
                    "pred_mutual_information": np.zeros(count),
                    "raw_pred_mean": probs,
                    "ensemble_member_probs": probs.reshape(-1, 1) if count else np.empty((0, 1)),
                    "raw_ensemble_member_probs": probs.reshape(-1, 1) if count else np.empty((0, 1)),
                }

            with mock.patch.object(p2, "train_ensemble", return_value=["model"]), mock.patch.object(
                p2,
                "score_sequences_with_ensemble",
                side_effect=fake_score,
            ):
                result = p2.run_phase2_mode(options)

            self.assertEqual(result["status"], "complete")
            self.assertTrue((options.output_root / "smoke" / "per_run_round_metrics.csv").exists())
            self.assertTrue((options.output_root / "smoke" / "per_run_selected_sequences.csv").exists())
            self.assertTrue((options.output_root / "smoke" / "strategy_summary.csv").exists())
            after = sorted(path.relative_to(phase1).as_posix() for path in phase1.rglob("*"))
            self.assertEqual(before, after)

    def test_pbs_generation_uses_absolute_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pbs_root = root / "repo"
            options = p2.Phase2Options(
                phase1_root=Path("thesis_results/01_reproduction"),
                output_root=root / "02_replay",
                mode="smoke",
                outer_folds=(1,),
                replay_seed_sizes=(10,),
                strategies=("random",),
                setup="ensemble_calibrated",
                pbs_repo_root=pbs_root,
            )
            paths = p2.write_supek_pbs_scripts(options)
            pbs = [path for path in paths if path.name == "supek_phase2_smoke.pbs"][0]
            text = pbs.read_text(encoding="utf-8")
            output_lines = [line for line in text.splitlines() if line.startswith("#PBS -o ") or line.startswith("#PBS -e ")]
            self.assertEqual(len(output_lines), 2)
            for line in output_lines:
                raw = line.split(maxsplit=2)[2]
                self.assertTrue(Path(raw).is_absolute(), raw)
            self.assertIn("phase2-replay --mode smoke", text)

    def test_ablation_pbs_submit_script_is_posix_and_logs_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                options = p2.Phase2Options(
                    phase1_root=Path("thesis_results/01_reproduction"),
                    output_root=Path("02_replay"),
                    mode="ablation",
                    outer_folds=(1,),
                    replay_seed_sizes=(10,),
                    strategies=("random",),
                    pbs_repo_root=Path("<supek_home>/repo"),
                )
                paths = p2.write_supek_pbs_scripts(options)
                paths = [path.resolve() for path in paths]
            finally:
                os.chdir(previous_cwd)
            submit = [path for path in paths if path.name == "supek_phase2_ablation_submit_all.sh"][0]
            submit_text = submit.read_text(encoding="utf-8")
            self.assertIn("<supek_home>/repo", submit_text)
            self.assertNotIn("\\", submit_text)
            self.assertNotIn("C:/", submit_text)
            group_scripts = sorted(path for path in paths if "submit_group_" in path.name)
            self.assertEqual(len(group_scripts), 1)
            group_text = group_scripts[0].read_text(encoding="utf-8")
            self.assertIn("submitting 4 ablation jobs in group 01", group_text)
            self.assertNotIn("depend=afterok", group_text)
            self.assertNotIn("\\", group_text)
            aggregate_submit = [path for path in paths if path.name == "supek_phase2_ablation_submit_aggregate_after_groups.sh"][0]
            aggregate_text = aggregate_submit.read_text(encoding="utf-8")
            self.assertIn("use only after all groups finish", aggregate_text)

            pbs_files = [
                path
                for path in paths
                if path.name.startswith("supek_phase2_ablation_outer_") and path.suffix == ".pbs"
            ]
            output_paths = []
            for path in pbs_files:
                text = path.read_text(encoding="utf-8")
                output_paths.extend(
                    line.split(maxsplit=2)[2]
                    for line in text.splitlines()
                    if line.startswith("#PBS -o ") or line.startswith("#PBS -e ")
                )
            self.assertEqual(len(output_paths), len(set(output_paths)))

    def test_write_supek_pbs_does_not_run_replay_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = p2.Phase2Options(
                phase1_root=Path("thesis_results/01_reproduction"),
                output_root=Path(tmp) / "02_replay",
                mode="smoke",
                write_supek_pbs=True,
                outer_folds=(1,),
                replay_seed_sizes=(10,),
                strategies=("random",),
                setup="ensemble_calibrated",
            )
            with mock.patch.object(p2, "run_phase2_mode") as run_mode:
                result = p2.run_phase2_replay(options)
            run_mode.assert_not_called()
            self.assertEqual(result["status"], "pbs-written")
            self.assertTrue(any(path.endswith("supek_phase2_smoke.pbs") for path in result["outputs"]))

    def test_status_reports_failed_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "02_replay"
            log = root / "logs" / "failed.err"
            log.parent.mkdir(parents=True)
            log.write_text("Traceback (most recent call last)\nboom\n", encoding="utf-8")
            status = p2.phase2_status(root)
            self.assertFalse(status["ready_for_thesis_figures"])
            self.assertIn(str(log), status["failed_logs"])
            self.assertIn("Failed/error logs", p2.format_phase2_status(status))

    def test_phase2_svg_export_writes_dataset_specific_benchmark_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "02_replay"
            benchmark = root / "benchmark"
            benchmark.mkdir(parents=True)
            learning_header = (
                "mode,initial_label_count,setup,strategy,round_id,labeled_count,"
                "evaluation_dataset,f1,pr_auc\n"
            )
            learning_rows = [
                "benchmark,10,ensemble_calibrated,random,0,10,validation,0.70,0.72\n",
                "benchmark,10,ensemble_calibrated,random,1,15,validation,0.74,0.76\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,0,10,validation,0.75,0.78\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,1,15,validation,0.82,0.84\n",
                "benchmark,10,ensemble_calibrated,random,0,10,holdout,0.68,0.70\n",
                "benchmark,10,ensemble_calibrated,random,1,15,holdout,0.72,0.74\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,0,10,holdout,0.73,0.75\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,1,15,holdout,0.80,0.82\n",
            ]
            (benchmark / "learning_curves.csv").write_text(learning_header + "".join(learning_rows), encoding="utf-8")
            summary_header = (
                "mode,initial_label_count,setup,strategy,evaluation_dataset,n_repeats,"
                "mean_final_F1,std_final_F1,mean_AULC_F1,std_AULC_F1,mean_final_PR_AUC,mean_final_Brier\n"
            )
            summary_rows = [
                "benchmark,10,ensemble_calibrated,random,validation,1,0.74,0,0.72,0,0.76,0.18\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,validation,1,0.82,0,0.79,0,0.84,0.12\n",
                "benchmark,10,ensemble_calibrated,random,holdout,1,0.72,0,0.70,0,0.74,0.20\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,holdout,1,0.80,0,0.77,0,0.82,0.14\n",
            ]
            (benchmark / "strategy_summary.csv").write_text(summary_header + "".join(summary_rows), encoding="utf-8")
            labels_header = (
                "mode,initial_label_count,setup,strategy,evaluation_dataset,target_f1,n_repeats,"
                "mean_labels_to_target,median_labels_to_target,reached_count\n"
            )
            labels_rows = [
                "benchmark,10,ensemble_calibrated,random,validation,0.86,1,55,55,1\n",
                "benchmark,10,ensemble_calibrated,ensemble_mi,validation,0.86,1,35,35,1\n",
            ]
            (benchmark / "labels_to_target_summary.csv").write_text(labels_header + "".join(labels_rows), encoding="utf-8")

            outputs = p2.write_phase2_svg_figures(root, root / "evidence" / "figures")

            self.assertIn("benchmark_validation_f1_initial_10_vs_labeled_peptides.svg", outputs)
            self.assertIn("benchmark_holdout_f1_initial_10_vs_labeled_peptides.svg", outputs)
            self.assertIn("benchmark_validation_mean_AULC_F1_initial_10.svg", outputs)
            self.assertIn("benchmark_validation_labels_to_f1_086_initial_10.svg", outputs)
            self.assertIn("benchmark_validation_labels_saved_vs_random_to_f1_086_initial_10.svg", outputs)


if __name__ == "__main__":
    unittest.main()
