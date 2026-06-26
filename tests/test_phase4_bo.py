from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

from active_learning_thesis import phase4_bo as p4
from active_learning_thesis.acquisition import _descending_indices, select_batch
from active_learning_thesis.cli import _build_parser
from active_learning_thesis.config import RunConfig
from active_learning_thesis.generative import (
    _population_fitness_from_utilities,
    _validated_policy_utilities,
    calculate_length_penalties,
    calculate_similarity_penalties,
    generation_fitness_components,
)
from active_learning_thesis.predictive import (
    CalibrationInputError,
    _apply_calibration,
    _fit_platt_calibration,
)


class Phase4BOTests(unittest.TestCase):
    def test_candidate_schema_separates_generator_and_final_utilities(self):
        required = {
            "generator_utility_score",
            "generator_utility_scope",
            "similarity_penalty",
            "length_penalty",
            "generator_fitness",
            "final_acquisition_utility",
            "final_acquisition_utility_applicable",
            "final_acquisition_utility_scope",
            "selection_rank",
            "random_shuffle_audit_key",
        }
        self.assertTrue(required.issubset(set(p4.CANDIDATE_FIELDS)))

    def test_cli_exposes_phase4_actions_without_phase4_penalty_arguments(self):
        parser = _build_parser()
        args = parser.parse_args(["phase4-bo", "status"])
        self.assertEqual(args.command, "phase4-bo")
        self.assertEqual(args.phase4_bo_action, "status")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["phase4-bo", "init", "--lambda-similarity", "1.0"]
            )

    def test_calibration_rejects_malformed_inputs(self):
        config = RunConfig()
        for probabilities, labels in [
            (None, [0, 1]),
            ([], []),
            ([[0.2, 0.8]], [0, 1]),
            ([0.2], [0, 1]),
            ([np.nan, 0.5], [0, 1]),
            ([0.2, 0.8], [0, 2]),
        ]:
            with self.subTest(probabilities=probabilities, labels=labels):
                with self.assertRaises(CalibrationInputError):
                    _fit_platt_calibration(probabilities, labels, config)

    def test_calibration_low_variation_uses_clipped_identity(self):
        config = RunConfig()
        calibration = _fit_platt_calibration(
            np.asarray([0.5, 0.5, 0.5, 0.5]),
            np.asarray([0, 1, 0, 1]),
            config,
        )
        self.assertEqual(calibration["method"], "identity_low_logit_variation")
        self.assertFalse(calibration["fitting_attempted"])
        self.assertTrue(calibration["fallback_used"])
        raw = np.asarray([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            _apply_calibration(raw, calibration),
            [1e-6, 0.5, 1 - 1e-6],
        )

    def test_calibration_single_class_uses_identity(self):
        calibration = _fit_platt_calibration(
            np.asarray([0.2, 0.3, 0.4]),
            np.asarray([1, 1, 1]),
            RunConfig(),
        )
        self.assertEqual(calibration["method"], "identity_single_class")
        self.assertEqual(calibration["fallback_reason"], "validation_labels_single_class")

    def test_valid_calibration_records_phase3_settings(self):
        calibration = _fit_platt_calibration(
            np.asarray([0.1, 0.3, 0.7, 0.9]),
            np.asarray([0, 0, 1, 1]),
            RunConfig(),
        )
        self.assertEqual(calibration["method"], "platt_logit")
        self.assertEqual(calibration["learning_rate"], 0.05)
        self.assertEqual(calibration["l2"], 1e-3)
        self.assertEqual(calibration["max_iterations"], 500)
        self.assertEqual(calibration["probability_clipping_epsilon"], 1e-6)

    def test_utility_callback_contract(self):
        sequences = ["AAAAA", "CCCCC"]
        np.testing.assert_allclose(
            _validated_policy_utilities(sequences, lambda values: [0.2, 0.8]),
            [0.2, 0.8],
        )
        with self.assertRaises(ValueError):
            _validated_policy_utilities(sequences, lambda values: [0.2])
        with self.assertRaises(ValueError):
            _validated_policy_utilities(sequences, lambda values: [0.2, np.nan])
        with self.assertRaises(ValueError):
            _validated_policy_utilities(
                sequences,
                lambda values: {
                    "sequences": list(reversed(values)),
                    "utilities": [0.2, 0.8],
                },
            )
        with self.assertRaises(ValueError):
            _validated_policy_utilities(
                sequences,
                lambda values: {
                    "utilities": [0.2, 0.8],
                    "metadata": {"similarity_penalty_applied": True},
                },
            )

    def test_no_callback_and_equivalent_callback_have_identical_fitness(self):
        sequences = ["AAAAA", "AAAAC", "CCCCCC"]
        scores = {"pred_mean": np.asarray([0.2, 0.6, 0.9])}
        config = RunConfig(use_similarity_penalty=True, use_length_penalty=True)
        original = generation_fitness_components(
            sequences,
            scores,
            "ensemble_mean",
            config,
        )
        callback = generation_fitness_components(
            sequences,
            scores,
            "ensemble_mean",
            config,
            policy_utility_callback=lambda ordered: scores["pred_mean"],
        )
        for field in (
            "generator_utility_score",
            "similarity_penalty",
            "length_penalty",
            "generator_fitness",
        ):
            np.testing.assert_allclose(original[field], callback[field])

    def test_phase3_penalties_and_complete_fitness_are_reused(self):
        sequences = ["AAAA", "AAAA", "CCCCCCCCCCC"]
        config = RunConfig(
            preferred_length_min=5,
            preferred_length_max=10,
            allowed_amino_acids="ACDEFGHIKLMNPQRSTVWY",
        )
        similarity = calculate_similarity_penalties(
            sequences, config.allowed_amino_acids
        )
        length = calculate_length_penalties(sequences, config)
        utility = np.asarray([0.5, 0.5, 0.5])
        expected = utility - similarity - length
        np.testing.assert_allclose(
            _population_fitness_from_utilities(sequences, utility, config),
            expected,
        )
        self.assertAlmostEqual(length[0], 0.175)
        self.assertAlmostEqual(length[2], 0.175)
        self.assertGreater(similarity[0], 0.0)

    def test_greedy_ucb_pi_ei_and_zero_variance_equations(self):
        means = np.asarray([0.9, 0.6, 0.5])
        stds = np.asarray([0.1, 0.0, 0.0])
        members = np.tile(means[:, None], (1, 5))
        greedy, _ = p4.phase4_acquisition_scores(
            "greedy", means, stds, members, 0.5, 1.0, 0.0, 1e-8
        )
        ucb, _ = p4.phase4_acquisition_scores(
            "ucb", means, stds, members, 0.5, 1.0, 0.0, 1e-8
        )
        pi, _ = p4.phase4_acquisition_scores(
            "pi", means, stds, members, 0.5, 1.0, 0.0, 1e-8
        )
        ei, _ = p4.phase4_acquisition_scores(
            "ei", means, stds, members, 0.5, 1.0, 0.0, 1e-8
        )
        np.testing.assert_allclose(greedy, means)
        np.testing.assert_allclose(ucb, means + stds)
        self.assertEqual(pi[1], 1.0)
        self.assertEqual(pi[2], 0.0)
        self.assertAlmostEqual(ei[1], 0.1)
        self.assertEqual(ei[2], 0.0)

    def test_mes_uses_five_common_pool_member_maxima(self):
        members = np.asarray(
            [
                [0.1, 0.4, 0.2, 0.05, 0.3],
                [0.5, 0.1, 0.6, 0.2, 0.05],
                [0.2, 0.3, 0.1, 0.7, 0.4],
            ]
        )
        mean = members.mean(axis=1)
        std = members.std(axis=1, ddof=0)
        scores, maxima = p4.phase4_acquisition_scores(
            "mes", mean, std, members, 0.0, 1.0, 0.0, 1e-8
        )
        np.testing.assert_allclose(maxima, [0.5, 0.4, 0.6, 0.7, 0.4])
        self.assertTrue(np.isfinite(scores).all())

    def test_final_mes_rescoring_receives_complete_retained_pool(self):
        sequences = ["AAAAA", "CCCCC", "DDDDD"]
        rows = [
            {
                "final_acquisition_utility": float(index),
                "final_acquisition_utility_applicable": True,
                "final_acquisition_utility_scope": "final",
            }
            for index in range(len(sequences))
        ]
        with mock.patch.object(
            p4,
            "_score_model_guided_sequences",
            return_value=rows,
        ) as scorer:
            exported = p4._final_candidate_rows(
                "mes",
                sequences,
                {},
                object(),
                {"incumbent": {"value": 0.5}},
                {},
            )
        self.assertEqual(
            scorer.call_args.args[3],
            sequences,
        )
        self.assertEqual(len(exported), len(sequences))
        self.assertTrue(
            all(
                row["final_acquisition_utility_scope"] == "final"
                for row in exported
            )
        )

    def test_training_incumbent_uses_only_calibrated_235_row_predictions(self):
        rows = [
            {"sequence": f"SEQ{index}", "label": str(index % 2)}
            for index in range(235)
        ]
        members = np.full((235, 5), 0.2)
        members[17] = [0.8, 0.9, 0.85, 0.95, 0.9]
        scores = {
            "ensemble_member_probs": members,
            "pred_mean": members.mean(axis=1),
        }
        incumbent = p4._training_incumbent(rows, scores)
        self.assertEqual(incumbent["sequence"], "SEQ17")
        self.assertAlmostEqual(incumbent["value"], 0.88)
        with self.assertRaises(ValueError):
            p4._training_incumbent(rows[:-1], scores)

    def test_guided_scoring_requests_memberwise_calibration_before_acquisition(self):
        calibrated = np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.7, 0.8, 0.9, 0.6, 0.75],
            ]
        )
        raw = np.asarray(
            [
                [0.01, 0.02, 0.03, 0.04, 0.05],
                [0.2, 0.3, 0.4, 0.1, 0.25],
            ]
        )
        summary = {
            "ensemble_member_probs": calibrated,
            "raw_ensemble_member_probs": raw,
            "pred_mean": calibrated.mean(axis=1),
            "pred_std": calibrated.std(axis=1, ddof=0),
            "pred_entropy": np.asarray([0.6, 0.5]),
            "pred_expected_entropy": np.asarray([0.5, 0.4]),
            "pred_mutual_information": np.asarray([0.1, 0.1]),
        }
        with mock.patch.object(
            p4,
            "score_sequences_with_ensemble",
            return_value=summary,
        ) as scorer:
            rows = p4._score_model_guided_sequences(
                "greedy",
                [object()] * 5,
                {"incumbent": {"value": 0.9}},
                ["AAAAA", "CCCCC"],
                {
                    "ucb_kappa": 1.0,
                    "improvement_xi": 0.0,
                    "zero_variance_epsilon": 1e-8,
                    "random_seed": 20260317,
                },
            )
        scorer.assert_called_once()
        self.assertTrue(scorer.call_args.kwargs["use_calibration"])
        self.assertAlmostEqual(
            rows[0]["final_acquisition_utility"], calibrated[0].mean()
        )
        self.assertAlmostEqual(
            rows[1]["calibrated_ensemble_std_probability"],
            calibrated[1].std(ddof=0),
        )

    def test_reporting_threshold_cannot_change_acquisition_scores(self):
        calibrated = np.asarray([[0.2, 0.3, 0.4, 0.5, 0.6]], dtype=float)
        raw = calibrated - 0.01
        scored = {
            "ensemble_member_probs": calibrated,
            "raw_ensemble_member_probs": raw,
            "pred_mean": calibrated.mean(axis=1),
            "pred_std": calibrated.std(axis=1, ddof=0),
            "pred_entropy": np.asarray([0.6]),
            "pred_expected_entropy": np.asarray([0.5]),
            "pred_mutual_information": np.asarray([0.1]),
        }
        config = {
            "random_seed": 20260317,
            "ucb_kappa": 1.0,
            "improvement_xi": 0.0,
            "zero_variance_epsilon": 1e-8,
        }
        context_low = {
            "incumbent": {"value": 0.55},
            "validation_report": {"decision_threshold": 0.1},
        }
        context_high = {
            "incumbent": {"value": 0.55},
            "validation_report": {"decision_threshold": 0.9},
        }
        with mock.patch.object(
            p4,
            "score_sequences_with_ensemble",
            return_value=scored,
        ):
            low = p4._score_model_guided_sequences(
                "ei", object(), context_low, ["AAAAA"], config
            )
            high = p4._score_model_guided_sequences(
                "ei", object(), context_high, ["AAAAA"], config
            )
        self.assertEqual(
            low[0]["final_acquisition_utility"],
            high[0]["final_acquisition_utility"],
        )

    def test_random_has_zero_generation_utility_and_null_final_utility(self):
        callback = p4._phase4_utility_callback(
            "random", None, None, {"random_seed": 20260317}
        )
        payload = callback(["AAAAA", "CCCCC"])
        np.testing.assert_allclose(payload["utilities"], [0.0, 0.0])
        rows = p4._final_candidate_rows(
            "random",
            ["AAAAA"],
            {
                "AAAAA": {
                    "generator_objective": "broad_pool",
                    "generator_utility_score": 0.0,
                    "similarity_penalty": 0.02,
                    "length_penalty": 0.0,
                    "generator_fitness": -0.02,
                }
            },
            None,
            None,
            {},
        )
        self.assertIsNone(rows[0]["final_acquisition_utility"])
        self.assertFalse(rows[0]["final_acquisition_utility_applicable"])
        self.assertEqual(rows[0]["final_acquisition_utility_scope"], "not_applicable")

    def test_random_null_utility_serializes_as_empty_numeric_csv_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "random.csv"
            p4._write_csv(
                path,
                [
                    {
                        "final_acquisition_utility": None,
                        "final_acquisition_utility_applicable": False,
                        "final_acquisition_utility_scope": "not_applicable",
                    }
                ],
            )
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["final_acquisition_utility"], "")
        self.assertEqual(row["final_acquisition_utility_applicable"], "False")
        self.assertEqual(row["final_acquisition_utility_scope"], "not_applicable")

    def test_random_and_pointwise_final_selectors_are_inherited(self):
        config = RunConfig(batch_size=2)
        candidate_scores = {"pred_mean": np.zeros(4)}
        selected, _ = select_batch(
            "random",
            2,
            candidate_scores,
            np.empty((0, 0)),
            config,
            123,
        )
        expected = list(range(4))
        random.Random(123).shuffle(expected)
        self.assertEqual(selected, expected[:2])
        self.assertEqual(_descending_indices(np.asarray([0.2, 0.9, 0.5])), [1, 2, 0])

    def test_threshold_tie_break_prefers_higher_threshold(self):
        from active_learning_thesis.metrics import pr_best_f1_threshold

        threshold, _ = pr_best_f1_threshold([0, 1], [0.2, 0.8])
        self.assertEqual(threshold, 0.8)

    def test_compare_reports_failed_blocked_and_missing_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for policy, status in {
                "random": "completed",
                "greedy": "failed",
                "ucb": "blocked",
            }.items():
                round_dir = root / "branches" / policy / "rounds" / "round_001"
                round_dir.mkdir(parents=True)
                (round_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "error": "fixture" if status != "completed" else "",
                        }
                    ),
                    encoding="utf-8",
                )
                if status == "completed":
                    p4._write_csv(
                        round_dir / "selected_batch.csv",
                        [
                            {
                                "sequence": f"AAAA{index}",
                                "final_acquisition_utility": None,
                            }
                            for index in range(5)
                        ],
                    )
            result = p4.compare_phase4(
                argparse.Namespace(output_root=str(root), round=1)
            )
            self.assertEqual(result["branch_status"]["random"], "completed")
            self.assertEqual(result["branch_status"]["greedy"], "failed")
            self.assertEqual(result["branch_status"]["ucb"], "blocked")
            self.assertEqual(result["branch_status"]["pi"], "missing")

    def test_compare_marks_afterok_guided_jobs_blocked_when_training_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models" / "ap_sp_fixed_split_ensemble"
            model_dir.mkdir(parents=True)
            (model_dir / "model_manifest.json").write_text(
                json.dumps({"status": "not_trained"}),
                encoding="utf-8",
            )
            for policy in p4.PHASE4_POLICIES:
                round_dir = root / "branches" / policy / "rounds" / "round_001"
                round_dir.mkdir(parents=True)
                status = "failed" if policy == "random" else "preview_ready"
                (round_dir / "status.json").write_text(
                    json.dumps({"status": status}),
                    encoding="utf-8",
                )
            result = p4.compare_phase4(
                argparse.Namespace(output_root=str(root), round=1)
            )
            self.assertEqual(result["branch_status"]["random"], "failed")
            for policy in p4.MODEL_GUIDED_POLICIES:
                self.assertEqual(result["branch_status"][policy], "blocked")

    def test_pbs_submit_graph_uses_afterok_and_afterany(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(parents=True, exist_ok=True)
            p4._write_json(
                root / "config.json",
                {
                    "supek": {
                        "queue": "gpu",
                        "ncpus": 4,
                        "ngpus": 1,
                        "mem": "40GB",
                    }
                },
            )
            paths = p4.write_phase4_pbs_previews(root, p4.PHASE4_POLICIES)
            submit = Path(paths["submit_all"]).read_text(encoding="utf-8")
            self.assertIn("depend=afterok:$TRAIN_ID", submit)
            self.assertIn("depend=afterany:$RANDOM_ID:", submit)
            compare_pbs = Path(paths["compare"]).read_text(encoding="utf-8")
            self.assertIn("phase4-bo compare", compare_pbs)


if __name__ == "__main__":
    unittest.main()
