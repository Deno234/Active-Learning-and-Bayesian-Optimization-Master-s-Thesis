from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from active_learning_thesis import phase5_self_paced as p5
from active_learning_thesis.cli import _build_parser
from active_learning_thesis.config import RunConfig
from active_learning_thesis.phase2_replay import ReplayRows, ReplayRunSpec
from active_learning_thesis.predictive import (
    ManagedModel,
    extract_ap_sp_member_embeddings_strict,
)


class _FakeLayer:
    def __init__(self, name="concatenate", shape=(None, 384), class_name="Concatenate"):
        self.name = name
        self.output_shape = shape
        self.output = object()
        self.__class__ = type(class_name, (), {})


class _FakeModel:
    def __init__(self, layer):
        self.layers = [object(), layer, object()]
        self.inputs = [object()]


class Phase5SelfPacedTests(unittest.TestCase):
    def test_cli_parses_initialization_and_job_commands(self):
        parser = _build_parser()
        init = parser.parse_args(["phase5-self-paced", "init"])
        self.assertEqual(init.phase5_action, "init")
        self.assertEqual(init.outer_folds, [1, 2, 3])
        self.assertEqual(init.initial_label_counts, [10])
        self.assertEqual(init.max_rounds, 45)
        self.assertEqual(init.ensemble_size, 1)
        job = parser.parse_args(
            [
                "phase5-self-paced",
                "run-job",
                "--outer-fold",
                "2",
                "--initial-label-count",
                "40",
                "--strategy",
                "self_paced_entropy",
            ]
        )
        self.assertEqual(job.outer_fold, 2)
        self.assertEqual(job.strategy, "self_paced_entropy")

    def test_pace_schedule_endpoints_and_monotonicity(self):
        values = [p5.pace_lambda(index, 45) for index in range(45)]
        self.assertAlmostEqual(values[0], 0.30)
        self.assertAlmostEqual(values[-1], 1.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        with self.assertRaises(ValueError):
            p5.pace_lambda(45, 45)

    def test_paired_aulc_rows_emit_one_comparison_per_fold_condition(self):
        rows = []
        strategy_offsets = {
            "random": 0.00,
            "predictive_entropy": 0.01,
            "static_easy_entropy": 0.02,
            "self_paced_entropy": 0.03,
        }
        for outer_fold in (1, 2, 3):
            for strategy, offset in strategy_offsets.items():
                for labelled_count in (10, 60, 110, 160, 235):
                    rows.append(
                        {
                            "evaluation_dataset": "holdout",
                            "outer_fold_id": str(outer_fold),
                            "initial_label_count": "10",
                            "strategy": strategy,
                            "labeled_count": str(labelled_count),
                            "f1": str(0.5 + offset + labelled_count / 1000.0),
                        }
                    )

        paired = p5._paired_aulc_rows(rows)

        self.assertEqual(len(paired), 3 * 4 * 4)
        full_self_paced_vs_random = [
            row
            for row in paired
            if row["interval_end"] == 235
            and row["comparison"] == "self_paced_entropy - random"
        ]
        self.assertEqual(len(full_self_paced_vs_random), 3)
        summary = p5._paired_aulc_summary(paired)
        summary_row = next(
            row
            for row in summary
            if row["interval_end"] == "235"
            and row["comparison"] == "self_paced_entropy - random"
        )
        self.assertEqual(summary_row["fold_count"], 3)

    def test_terminal_convergence_resolves_imported_run_local_parameter_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            round_rows = []
            for strategy in ("random", "predictive_entropy"):
                run_dir = (
                    output_root
                    / "replay"
                    / "outer_1"
                    / "initial_10"
                    / strategy
                )
                run_dir.mkdir(parents=True)
                parameter_path = run_dir / "terminal_parameters.npy"
                np.save(parameter_path, np.asarray([1.0, 2.0]))
                (run_dir / "terminal_state.json").write_text(
                    json.dumps(
                        {
                            "outer_fold": 1,
                            "initial_label_count": 10,
                            "strategy": strategy,
                            "ordered_row_ids_checksum_sha256": "same",
                            "model_seed": 123,
                            "parameter_vector_path": (
                                "/lustre/remote/phase5/"
                                f"{strategy}/terminal_parameters.npy"
                            ),
                            "holdout_predictions": [0.2, 0.8],
                            "numerical_tolerance": 1e-6,
                        }
                    ),
                    encoding="utf-8",
                )
                round_rows.append(
                    {
                        "evaluation_dataset": "holdout",
                        "outer_fold_id": "1",
                        "initial_label_count": "10",
                        "strategy": strategy,
                        "labeled_count": "235",
                        "f1": "0.8",
                        "pr_auc": "0.9",
                    }
                )

            audit = p5._terminal_convergence_rows(output_root, round_rows)

        self.assertEqual(len(audit), 2)
        self.assertTrue(all(row["parameters_within_tolerance"] for row in audit))

    def test_zero_based_stable_percentiles_preserve_ties(self):
        result = p5.stable_difficulty_percentiles([0.2, 0.1, 0.1, 0.4])
        np.testing.assert_allclose(result, [2 / 3, 0.0, 1 / 3, 1.0])
        np.testing.assert_allclose(p5.stable_difficulty_percentiles([0.5]), [0.0])
        self.assertEqual(p5.operational_difficulty_quintile(0.0), 1)
        self.assertEqual(p5.operational_difficulty_quintile(1.0), 5)

    def test_memberwise_distance_normalizes_members_independently(self):
        candidates = [
            np.asarray([[3.0, 0.0], [0.0, 2.0]]),
            np.asarray([[0.0, 4.0], [5.0, 0.0]]),
        ]
        labelled = [
            np.asarray([[1.0, 0.0]]),
            np.asarray([[0.0, 1.0]]),
        ]
        per_member, mean = p5.memberwise_familiarity_distances(candidates, labelled)
        np.testing.assert_allclose(per_member[:, 0], [0.0, np.sqrt(2.0)])
        np.testing.assert_allclose(per_member[:, 1], [0.0, np.sqrt(2.0)])
        np.testing.assert_allclose(mean, [0.0, np.sqrt(2.0)])
        zeros = p5.l2_normalize_embeddings(np.zeros((2, 3)))
        np.testing.assert_allclose(zeros, np.zeros((2, 3)))

    def test_strategy_independent_model_seed_schedule(self):
        expected = p5.model_seed_schedule(100, 2, 40, 3, ensemble_size=5)
        self.assertEqual(expected, [240130, 240131, 240132, 240133, 240134])
        self.assertEqual(
            expected, p5.model_seed_schedule(100, 2, 40, 3, ensemble_size=5)
        )
        self.assertEqual(p5.model_seed_schedule(100, 2, 10, 45), [210550])

    def test_canonical_training_order_ignores_acquisition_order(self):
        rows = [
            {"sequence": "CCC", "original_dataset_row_id": "3"},
            {"sequence": "AAA", "original_dataset_row_id": "1"},
            {"sequence": "BBB", "original_dataset_row_id": "2"},
        ]
        ordered = p5.canonical_training_order(rows)
        self.assertEqual([row["sequence"] for row in ordered], ["AAA", "BBB", "CCC"])
        self.assertEqual(
            p5.ordered_row_id_checksum(ordered),
            p5.ordered_row_id_checksum(
                p5.canonical_training_order(list(reversed(ordered)))
            ),
        )

    def test_predictive_entropy_matches_phase2_stable_selector(self):
        scores = {
            "pred_mean": np.asarray([0.2, 0.3, 0.4, 0.5]),
            "pred_entropy": np.asarray([0.7, 0.9, 0.9, 0.1]),
        }
        config = RunConfig()
        selected, acquisition, eligible, used, fallback, pace = p5.select_phase5_batch(
            "predictive_entropy",
            3,
            scores,
            np.asarray([0.4, 0.3, 0.2, 0.1]),
            np.asarray([1.0, 2 / 3, 1 / 3, 0.0]),
            0,
            20,
            config,
            7,
        )
        self.assertEqual(selected, [1, 2, 0])
        np.testing.assert_allclose(acquisition, scores["pred_entropy"])
        self.assertTrue(eligible.all())
        self.assertFalse(used)
        self.assertFalse(fallback)
        self.assertEqual(pace, 1.0)

        self_paced = p5.select_phase5_batch(
            "self_paced_entropy",
            3,
            scores,
            np.asarray([0.4, 0.3, 0.2, 0.1]),
            np.asarray([1.0, 2 / 3, 1 / 3, 0.0]),
            19,
            20,
            config,
            7,
        )
        self.assertEqual(self_paced[0], selected)
        self.assertTrue(self_paced[2].all())
        self.assertEqual(self_paced[5], 1.0)

    def test_static_and_self_paced_export_real_eligibility(self):
        scores = {
            "pred_mean": np.linspace(0.1, 0.9, 6),
            "pred_entropy": np.asarray([0.1, 0.9, 0.8, 0.7, 0.6, 0.5]),
        }
        percentiles = np.linspace(0.0, 1.0, 6)
        selected, _, eligible, used, fallback, pace = p5.select_phase5_batch(
            "static_easy_entropy",
            2,
            scores,
            np.arange(6, dtype=float),
            percentiles,
            10,
            20,
            RunConfig(),
            9,
        )
        self.assertTrue(used)
        self.assertEqual(pace, 0.30)
        self.assertEqual(list(eligible), [True, True, False, False, False, False])
        self.assertEqual(selected, [1, 0])
        self.assertFalse(fallback)

        selected, _, eligible, used, fallback, pace = p5.select_phase5_batch(
            "self_paced_entropy",
            5,
            scores,
            np.arange(6, dtype=float),
            percentiles,
            0,
            20,
            RunConfig(),
            9,
        )
        self.assertTrue(used)
        self.assertTrue(fallback)
        self.assertEqual(sum(eligible), 2)
        self.assertEqual(len(selected), 5)
        self.assertEqual(pace, 0.30)

    def test_random_uses_inherited_selector_and_has_no_familiarity_filter(self):
        scores = {
            "pred_mean": np.zeros(8),
            "pred_entropy": np.arange(8, dtype=float),
        }
        expected, _ = p5.select_batch(
            "random",
            5,
            scores,
            np.empty((0, 0)),
            RunConfig(),
            44,
        )
        actual, _, eligible, used, fallback, pace = p5.select_phase5_batch(
            "random",
            5,
            scores,
            np.arange(8, dtype=float)[::-1],
            np.linspace(0, 1, 8),
            0,
            20,
            RunConfig(),
            44,
        )
        self.assertEqual(actual, expected)
        self.assertTrue(eligible.all())
        self.assertFalse(used)
        self.assertFalse(fallback)
        self.assertEqual(pace, 1.0)

    def test_post_hoc_join_uses_fixed_half_and_one_e_minus_six_clip(self):
        rows = [
            {
                "sequence": "AAA",
                "pred_mean": 1.0,
                "predictive_entropy": 0.0,
                "ensemble_mi": 0.0,
            },
            {
                "sequence": "CCC",
                "pred_mean": 0.49,
                "predictive_entropy": 0.4,
                "ensemble_mi": 0.1,
            },
        ]
        joined = p5._join_post_hoc_labels(
            rows,
            [{"sequence": "AAA", "label": "0"}, {"sequence": "CCC", "label": "1"}],
        )
        self.assertEqual(joined[0]["post_hoc_pre_query_error_fixed_0_5"], 1)
        self.assertEqual(joined[1]["post_hoc_pre_query_error_fixed_0_5"], 1)
        self.assertAlmostEqual(
            joined[0]["post_hoc_pre_query_log_loss"],
            -np.log(1e-6),
        )

    def test_spearman_uses_average_tied_ranks_and_handles_undefined(self):
        ranks = p5.average_tied_ranks([1.0, 1.0, 3.0])
        np.testing.assert_allclose(ranks, [0.5, 0.5, 2.0])
        self.assertAlmostEqual(p5.spearman_with_ties([1, 1, 2], [2, 2, 3]), 1.0)
        self.assertEqual(p5.spearman_with_ties([1, 1], [2, 3]), "")
        self.assertEqual(p5.spearman_with_ties([1], [2]), "")

    def test_preregistered_partial_aulc_uses_discrete_trapezoids(self):
        points = [
            {"labeled_count": count, "f1": value}
            for count, value in ((10, 0.2), (60, 0.6), (110, 0.8), (160, 0.9), (235, 1.0))
        ]
        self.assertAlmostEqual(
            p5.normalized_aulc_interval(points, "f1", 10, 60),
            0.4,
        )
        expected_full = (
            0.4 * 50 + 0.7 * 50 + 0.85 * 50 + 0.95 * 75
        ) / 225
        self.assertAlmostEqual(
            p5.normalized_aulc_interval(points, "f1", 10, 235),
            expected_full,
        )
        self.assertIsNone(p5.normalized_aulc_interval(points[:-1], "f1", 10, 235))

    def test_budget_aware_yield_and_diversity_mark_terminal_as_consistency_only(self):
        rows = []
        for step in range(45):
            for rank in range(5):
                rows.append(
                    {
                        "outer_fold": "1",
                        "initial_label_count": "10",
                        "strategy": "random",
                        "batch_size": "5",
                        "acquisition_step": str(step),
                        "sequence": f"S{step:02d}{rank}",
                        "label": str((step + rank) % 2),
                    }
                )
        yield_rows = p5._positive_yield_rows(rows)
        fixed = [
            row for row in yield_rows
            if row["scope"] == "cumulative" and row["labelled_count"] == 60
        ]
        self.assertEqual(len(fixed), 1)
        self.assertTrue(fixed[0]["fixed_budget_summary"])
        terminal = [
            row for row in yield_rows
            if row["scope"] == "cumulative" and row["labelled_count"] == 235
        ]
        self.assertTrue(terminal[0]["terminal_consistency_only"])
        diversity = p5._diversity_rows(rows)
        self.assertTrue(
            any(
                row["scope"] == "cumulative"
                and row["labelled_count"] == 160
                and row["fixed_budget_summary"]
                for row in diversity
            )
        )

    def test_strict_embedding_contract_rejects_wrong_layer_or_shape(self):
        wrong_layer = ManagedModel(
            "AP_SP",
            1,
            None,
            _FakeModel(_FakeLayer(class_name="Dense")),
        )
        with self.assertRaisesRegex(ValueError, "not Concatenate"):
            extract_ap_sp_member_embeddings_strict([wrong_layer], ["AAA"])

        wrong_shape = ManagedModel(
            "AP_SP",
            1,
            None,
            _FakeModel(_FakeLayer(shape=(None, 383))),
        )
        with self.assertRaisesRegex(ValueError, "configured embedding shape"):
            extract_ap_sp_member_embeddings_strict([wrong_shape], ["AAA"])

    def test_strict_embedding_contract_records_runtime_metadata(self):
        layer = _FakeLayer()
        member = ManagedModel("AP_SP", 17, None, _FakeModel(layer))
        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(Model=lambda inputs, outputs: object())
        )
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}), mock.patch(
            "active_learning_thesis.predictive._prepare_inference_tensors",
            return_value=np.zeros((2, 1)),
        ), mock.patch(
            "active_learning_thesis.predictive._batched_forward_pass",
            return_value=np.ones((2, 384)),
        ), mock.patch(
            "active_learning_thesis.predictive._predictive_modules",
            return_value=(None, None, types.SimpleNamespace(BATCH_SIZE=16)),
        ):
            embeddings, metadata = extract_ap_sp_member_embeddings_strict(
                [member], ["AAA", "CCC"]
            )
        self.assertEqual(embeddings[0].shape, (2, 384))
        self.assertEqual(metadata[0]["embedding_layer_name"], "concatenate")
        self.assertEqual(metadata[0]["member_seed"], 17)

    def test_strict_embedding_contract_rejects_runtime_row_mismatch_and_nonfinite(self):
        member = ManagedModel("AP_SP", 17, None, _FakeModel(_FakeLayer()))
        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(Model=lambda inputs, outputs: object())
        )
        common = [
            mock.patch.dict(sys.modules, {"tensorflow": fake_tf}),
            mock.patch(
                "active_learning_thesis.predictive._prepare_inference_tensors",
                return_value=np.zeros((2, 1)),
            ),
            mock.patch(
                "active_learning_thesis.predictive._predictive_modules",
                return_value=(None, None, types.SimpleNamespace(BATCH_SIZE=16)),
            ),
        ]
        with common[0], common[1], common[2], mock.patch(
            "active_learning_thesis.predictive._batched_forward_pass",
            return_value=np.ones((1, 384)),
        ):
            with self.assertRaisesRegex(ValueError, "embedding output shape"):
                extract_ap_sp_member_embeddings_strict([member], ["AAA", "CCC"])
        values = np.ones((2, 384))
        values[0, 0] = np.nan
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}), mock.patch(
            "active_learning_thesis.predictive._prepare_inference_tensors",
            return_value=np.zeros((2, 1)),
        ), mock.patch(
            "active_learning_thesis.predictive._predictive_modules",
            return_value=(None, None, types.SimpleNamespace(BATCH_SIZE=16)),
        ), mock.patch(
            "active_learning_thesis.predictive._batched_forward_pass",
            return_value=values,
        ):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                extract_ap_sp_member_embeddings_strict([member], ["AAA", "CCC"])

    def test_runtime_walltime_is_derived_and_override_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase2 = Path(tmp) / "02_replay"
            run = phase2 / "benchmark" / "runs" / "run_a"
            run.mkdir(parents=True)
            (run / "resource_log.csv").write_text(
                "run_id,walltime_seconds,exit_status\nrun_a,36000,success\n",
                encoding="utf-8",
            )
            derived = p5.derive_phase5_walltime(phase2, None)
            self.assertEqual(derived["requested_walltime"], "02:00:00")
            explicit = p5.derive_phase5_walltime(phase2, "06:30:00")
            self.assertEqual(explicit["requested_walltime"], "06:30:00")
            with self.assertRaisesRegex(ValueError, "provide --supek-walltime"):
                p5.derive_phase5_walltime(Path(tmp) / "missing", None)

    def test_mocked_replay_keeps_labels_out_of_scoring_records(self):
        rows = ReplayRows(
            holdout=[{"sequence": "HAA", "label": "0", "original_dataset_row_id": "0"}],
            validation=[
                {"sequence": "VAA", "label": "0", "original_dataset_row_id": "1"},
                {"sequence": "VCC", "label": "1", "original_dataset_row_id": "2"},
            ],
            train_pool=[],
            replay_seed=[
                {"sequence": "LAA", "label": "0", "original_dataset_row_id": "3"},
                {"sequence": "LCC", "label": "1", "original_dataset_row_id": "4"},
            ],
            replay_hidden=[
                {
                    "sequence": f"C{index}A",
                    "label": str(index % 2),
                    "original_dataset_row_id": str(index + 5),
                }
                for index in range(6)
            ],
        )
        spec = ReplayRunSpec(
            mode="phase5",
            setup="ensemble_calibrated",
            outer_fold_id=1,
            inner_fold_id=1,
            replay_seed_size=2,
            batch_size=2,
            max_rounds=1,
            strategies=("self_paced_entropy",),
            base_seed=100,
            run_seed=1202,
            ensemble_size=5,
            use_calibrated_acquisition=True,
            run_dir=Path("unused"),
        )
        options = p5.Phase5Options(
            action="run-job",
            outer_fold=1,
            initial_label_count=2,
            strategy="self_paced_entropy",
            batch_size=2,
            max_rounds=1,
        )
        score_calls = []
        embedding_calls = []

        def fake_score(_ensemble, sequences, **_kwargs):
            score_calls.append(list(sequences))
            count = len(sequences)
            probs = np.linspace(0.2, 0.8, count)
            members = np.repeat(probs[:, None], 5, axis=1)
            return {
                "pred_mean": probs,
                "pred_std": np.zeros(count),
                "pred_entropy": np.linspace(0.1, 0.9, count),
                "pred_expected_entropy": np.zeros(count),
                "pred_mutual_information": np.zeros(count),
                "ensemble_member_probs": members,
                "raw_ensemble_member_probs": members,
            }

        def fake_embeddings(_ensemble, sequences, expected_width=384):
            embedding_calls.append(list(sequences))
            count = len(sequences)
            values = []
            metadata = []
            for member in range(5):
                array = np.zeros((count, expected_width))
                array[:, member] = np.arange(1, count + 1)
                values.append(array)
                metadata.append(
                    {
                        "member_index": member,
                        "runtime_shape": [count, expected_width],
                    }
                )
            return values, metadata

        metrics = {
            "f1": 0.5,
            "pr_auc": 0.5,
            "brier_score": 0.25,
            "ece_10": 0.1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            spec.run_dir = Path(tmp)
            fake_model = types.SimpleNamespace(
                model=types.SimpleNamespace(get_weights=lambda: [np.asarray([1.0])])
            )
            with mock.patch.object(p5, "train_ensemble", return_value=[fake_model] * 5), mock.patch.object(
                p5,
                "score_sequences_with_ensemble",
                side_effect=fake_score,
            ), mock.patch.object(
                p5,
                "extract_ap_sp_member_embeddings_strict",
                side_effect=fake_embeddings,
            ), mock.patch.object(
                p5,
                "evaluate_with_validation_threshold",
                return_value=(metrics, 0.5),
            ), mock.patch.object(
                p5,
                "evaluate_with_fixed_threshold",
                return_value=metrics,
            ), mock.patch.object(
                p5,
                "calibration_metric_rows",
                return_value=[],
            ):
                result = p5._run_strategy(options, spec, RunConfig(ensemble_size=5), rows)
        self.assertEqual(len(result["round_metrics"]), 4)
        self.assertEqual(len(result["selected_sequences"]), 2)
        self.assertNotIn("true_label", result["candidate_scoring"][0])
        self.assertIn("true_label", result["proxy_records"][0])
        self.assertTrue(all(set(call).isdisjoint({"0", "1"}) for call in score_calls))
        self.assertTrue(all("VAA" not in call and "VCC" not in call for call in embedding_calls))
        self.assertTrue(all("HAA" not in call for call in embedding_calls))
        self.assertTrue(all("LAA" in call and "LCC" in call for call in embedding_calls))

    def test_initialization_generates_12_jobs_without_submitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1 = root / "01_reproduction"
            (phase1 / "folds").mkdir(parents=True)
            (phase1 / "frozen_model_config.json").write_text(
                json.dumps({"AP_SP": {"num_cells": 48, "kernel_size": 8}}),
                encoding="utf-8",
            )
            sequences = [f"SEQ{index:03d}" for index in range(235)]
            manifest_rows = [
                {"sequence": sequence, "label": str(index % 2), "split": "train_pool"}
                for index, sequence in enumerate(sequences)
            ]
            for outer in range(1, 4):
                (phase1 / "folds" / f"replay_manifest_outer_{outer}_inner_1.json").write_text(
                    json.dumps({"rows": manifest_rows}),
                    encoding="utf-8",
                )
            phase2_run = root / "02_replay" / "benchmark" / "runs" / "run"
            phase2_run.mkdir(parents=True)
            (phase2_run / "resource_log.csv").write_text(
                "run_id,walltime_seconds,exit_status\nrun,40000,success\n",
                encoding="utf-8",
            )
            options = p5.Phase5Options(
                action="init",
                phase1_root=phase1,
                phase2_root=root / "02_replay",
                output_root=root / "05_phase5",
                pbs_repo_root=Path("/lustre/home/test/repo"),
            )
            before = {
                path.relative_to(phase1).as_posix(): path.read_bytes()
                for path in phase1.rglob("*")
                if path.is_file()
            }
            with mock.patch.object(
                p5,
                "canonical_dataset_row_ids",
                return_value={sequence: index for index, sequence in enumerate(sequences)},
            ):
                result = p5.initialize_phase5(options)
            self.assertEqual(result["job_count"], 12)
            self.assertFalse(result["jobs_submitted"])
            jobs = list((options.output_root / "pbs").glob("p5_o*.pbs"))
            self.assertEqual(len(jobs), 12)
            job_text = jobs[0].read_text(encoding="utf-8")
            self.assertIn(
                'cd "/lustre/home/test/repo"',
                job_text,
            )
            self.assertNotIn("G:/lustre", job_text)
            aggregate = options.output_root / "pbs" / "p5_aggregate.pbs"
            self.assertTrue(aggregate.exists())
            submit = (options.output_root / "pbs" / "submit_phase5_all.sh").read_text(
                encoding="utf-8"
            )
            self.assertIn("depend=afterany", submit)
            self.assertIn("PHASE5_MAX_ACTIVE_JOBS", submit)
            self.assertIn("wait_for_queue_slot", submit)
            after = {
                path.relative_to(phase1).as_posix(): path.read_bytes()
                for path in phase1.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
