from __future__ import annotations

import csv
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
import tempfile
import uuid
from unittest import mock

import numpy as np

from active_learning_thesis.acquisition import (
    _oed_indices,
    acquisition_diagnostics,
    candidate_objective_scores,
    embedding_novelty_scores,
    generator_objective_for_strategy,
    generator_objective_requires_embeddings,
    requires_embeddings,
    requires_family_models,
    select_batch,
)
from active_learning_thesis.config import (
    DEFAULT_REPLAY_STRATEGIES,
    THESIS_FULL_REPLAY_STRATEGIES,
    RunConfig,
)
from active_learning_thesis.dataset import build_split_manifest, read_experimental_dataset
from active_learning_thesis.dashboard_run_setup import RUN_SETUP_PRESETS
from active_learning_thesis.dashboard_study_setup import STUDY_PRESETS
from active_learning_thesis.discovery import (
    discovery_utility_scores,
    mean_pairwise_distance,
    min_distances_to_reference,
)
from active_learning_thesis.generative import (
    _hybrid_two_pool_targets,
    _population_fitness_from_probabilities,
    _population_fitness_from_utilities,
    calculate_length_penalty,
    calculate_similarity_penalty,
    generate_candidate_sequences,
    generation_fitness_components,
)
from active_learning_thesis.ledger import create_initial_ledger, save_ledger
from active_learning_thesis.metrics import evaluate_binary_classifier, summarize_ensemble
from active_learning_thesis.predictive import (
    ManagedModel,
    _InMemoryBestWeights,
    _batched_forward_pass,
    _training_fingerprint,
    score_sequences_with_ensemble,
    score_sequences_with_family,
    train_model,
)
from active_learning_thesis.workflow import (
    _resolve_discovery_ensemble_dir,
    _validate_cgmd_import,
    evaluate_final,
    init_run,
    propose_round,
    run_discovery,
    run_replay,
)


@contextmanager
def workspace_tempdir(prefix: str):
    root = Path(tempfile.mkdtemp(prefix=f"{prefix}_{uuid.uuid4().hex}_"))
    try:
        yield str(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


class DatasetSplitTests(unittest.TestCase):
    def test_old_configs_load_with_backward_compatible_generator_defaults(self):
        with workspace_tempdir("altest") as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                json.dumps({"run_name": "old_run", "output_root": "."}),
                encoding="utf-8",
            )
            config = RunConfig.load(path)
        self.assertEqual(config.generator_objective_mode, "fixed_mean")
        self.assertTrue(config.use_similarity_penalty)
        self.assertTrue(config.use_length_penalty)
        self.assertEqual(config.binary_threshold_strategy, "fixed_0_5")

    def test_new_configs_default_to_pr_best_f1_thresholding(self):
        config = RunConfig(run_name="new_threshold_default")
        self.assertEqual(config.binary_threshold_strategy, "pr_best_f1")

    def test_thesis_full_presets_include_extended_strategy_set_without_changing_defaults(self):
        self.assertEqual(
            DEFAULT_REPLAY_STRATEGIES,
            [
                "random",
                "ensemble_mi",
                "similarity_penalized_mean",
                "family_qbc",
                "cluster_diverse_representative",
                "oed_logdet",
                "hybrid_mi_diverse",
            ],
        )
        self.assertEqual(RunConfig().diversity_prefilter_multiplier, 3)
        self.assertEqual(len(THESIS_FULL_REPLAY_STRATEGIES), 10)
        for strategy in [
            "ensemble_mean",
            "similarity_penalized_mean",
            "predictive_entropy",
            "ucb",
            "cluster_diverse_representative",
        ]:
            self.assertIn(strategy, THESIS_FULL_REPLAY_STRATEGIES)
        self.assertNotIn("cluster_representative", THESIS_FULL_REPLAY_STRATEGIES)
        self.assertNotIn("embedding_farthest_first", THESIS_FULL_REPLAY_STRATEGIES)
        self.assertNotIn("cluster_representative", DEFAULT_REPLAY_STRATEGIES)
        self.assertNotIn("embedding_farthest_first", DEFAULT_REPLAY_STRATEGIES)
        self.assertEqual(
            RUN_SETUP_PRESETS["Thesis full"]["replay_strategies"],
            THESIS_FULL_REPLAY_STRATEGIES,
        )
        self.assertEqual(
            STUDY_PRESETS["Thesis full"]["strategies"],
            THESIS_FULL_REPLAY_STRATEGIES,
        )

    def test_split_manifest_is_mutually_exclusive_and_reproducible(self):
        records = read_experimental_dataset()
        config = RunConfig(run_name="unit_test_run")
        manifest_a = build_split_manifest(records, config)
        manifest_b = build_split_manifest(records, config)
        self.assertEqual(manifest_a, manifest_b)

        split_sets = {
            split: {
                sequence
                for sequence, assigned in manifest_a["splits"].items()
                if assigned == split
            }
            for split in {"holdout", "validation", "train_pool"}
        }
        self.assertFalse(split_sets["holdout"] & split_sets["validation"])
        self.assertFalse(split_sets["holdout"] & split_sets["train_pool"])
        self.assertFalse(split_sets["validation"] & split_sets["train_pool"])

        self.assertEqual(len(split_sets["holdout"]), 74)
        self.assertEqual(len(split_sets["validation"]), 29)
        self.assertEqual(len(split_sets["train_pool"]), 265)
        replay_seed = {
            sequence
            for sequence, role in manifest_a["replay_roles"].items()
            if role == "seed"
        }
        self.assertEqual(len(replay_seed), 40)

    def test_initial_ledger_contains_split_and_replay_metadata(self):
        records = read_experimental_dataset()
        manifest = build_split_manifest(records, RunConfig(run_name="ledger_test"))
        ledger = create_initial_ledger(records, manifest)
        self.assertEqual(len(ledger), len(records))
        sample = ledger[0]
        self.assertIn(sample["split"], {"holdout", "validation", "train_pool"})
        self.assertIn(sample["replay_role"], {"seed", "hidden", "none"})


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.config = RunConfig(run_name="acquisition_test")
        self.embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
                [9.0, 0.0],
                [9.1, 0.1],
            ],
            dtype=float,
        )
        self.candidate_scores = {
            "pred_mean": np.array([0.2, 0.3, 0.8, 0.75, 0.6, 0.62]),
            "pred_std": np.array([0.05, 0.03, 0.1, 0.08, 0.12, 0.11]),
            "pred_entropy": np.array([0.4, 0.5, 0.6, 0.55, 0.68, 0.69]),
            "pred_expected_entropy": np.array([0.1, 0.2, 0.22, 0.21, 0.24, 0.24]),
            "pred_mutual_information": np.array([0.3, 0.3, 0.38, 0.34, 0.44, 0.45]),
            "avg_embedding": self.embeddings,
            "committee_vote_entropy": np.array([0.0, 0.0, 0.67, 0.67, 0.67, 0.67]),
            "committee_prob_std": np.array([0.02, 0.03, 0.1, 0.12, 0.14, 0.13]),
        }

    def test_ensemble_summary_outputs_expected_keys(self):
        summary = summarize_ensemble(
            np.array(
                [
                    [0.2, 0.3, 0.4],
                    [0.8, 0.85, 0.9],
                ]
            )
        )
        self.assertEqual(
            set(summary),
            {
                "pred_mean",
                "pred_std",
                "pred_entropy",
                "pred_expected_entropy",
                "pred_mutual_information",
            },
        )

    def test_cluster_representative_returns_unique_indices(self):
        indices, _ = select_batch(
            "cluster_representative",
            batch_size=3,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
        )
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(set(indices)), 3)

    def test_cluster_diverse_representative_selects_novel_cluster_representative(self):
        config = RunConfig(run_name="cluster_diverse_test")
        config.diversity_prefilter_multiplier = 2
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["avg_embedding"] = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
                [10.2, 0.0],
            ],
            dtype=float,
        )
        for key in [
            "pred_mean",
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:4], dtype=float)
        indices, scores = select_batch(
            "cluster_diverse_representative",
            batch_size=1,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.array([[0.0, 0.0]], dtype=float),
            config=config,
            seed=7,
        )
        self.assertEqual(indices, [2])
        self.assertGreater(scores[2], scores[0])
        self.assertEqual(scores[2], scores[3])

    def test_cluster_diverse_representative_reports_deterministic_fallback(self):
        config = RunConfig(run_name="cluster_diverse_fallback")
        embeddings = np.zeros((4, 2), dtype=float)
        candidate_scores = {
            **self.candidate_scores,
            "avg_embedding": embeddings,
        }
        for key in [
            "pred_mean",
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:4], dtype=float)
        indices, scores = select_batch(
            "cluster_diverse_representative",
            batch_size=3,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.array([[1.0, 1.0]], dtype=float),
            config=config,
            seed=7,
        )
        diagnostics = acquisition_diagnostics(
            "cluster_diverse_representative",
            indices,
            candidate_scores,
            np.array([[1.0, 1.0]], dtype=float),
            scores,
            config,
            seed=7,
        )
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(len(set(indices)), 3)
        self.assertEqual(diagnostics[0]["requested_batch_size"], 3)
        self.assertEqual(diagnostics[0]["candidate_count"], 4)
        self.assertEqual(diagnostics[0]["requested_cluster_count"], 4)
        self.assertEqual(diagnostics[0]["non_empty_cluster_count"], 1)
        self.assertEqual(diagnostics[0]["selected_cluster_count"], 1)
        self.assertEqual(diagnostics[0]["fallback_fill_count"], 2)

    def test_oed_scores_are_monotonic_for_selected_points(self):
        selected, scores = _oed_indices(
            self.embeddings,
            labeled_embeddings=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            batch_size=3,
            regularization=1e-3,
        )
        chosen_scores = [scores[index] for index in selected]
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(score >= 0 for score in chosen_scores))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_hybrid_selection_splits_uncertainty_then_diversity(self):
        labeled = np.array([[0.0, 0.0], [9.2, 0.2]], dtype=float)
        indices, _ = select_batch(
            "hybrid_mi_diverse",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=11,
        )
        self.assertEqual(indices, [5, 2])

    def test_hybrid_selection_gives_uncertainty_extra_slot_for_odd_batches(self):
        labeled = np.array([[0.0, 0.0]], dtype=float)
        indices, _ = select_batch(
            "hybrid_mi_diverse",
            batch_size=5,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=11,
        )
        self.assertEqual(indices[:3], [5, 4, 2])
        self.assertEqual(len(indices[3:]), 2)
        self.assertFalse(set(indices[:3]) & set(indices[3:]))

    def test_hybrid_selection_splits_even_batches_equally(self):
        labeled = np.array([[0.0, 0.0]], dtype=float)
        indices, _ = select_batch(
            "hybrid_mi_diverse",
            batch_size=4,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=11,
        )
        self.assertEqual(indices[:2], [5, 4])
        self.assertEqual(len(indices[2:]), 2)
        self.assertFalse(set(indices[:2]) & set(indices[2:]))

    def test_embedding_farthest_first_selects_dynamic_diverse_indices(self):
        labeled = np.array([[0.0, 0.0]], dtype=float)
        indices, scores = select_batch(
            "embedding_farthest_first",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=7,
        )
        self.assertEqual(indices, [5, 2])
        self.assertGreater(scores[indices[0]], scores[indices[1]])
        self.assertGreater(scores[indices[1]], 0.0)

    def test_pointwise_acquisition_strategies_rank_expected_scores(self):
        mean_indices, mean_scores = select_batch(
            "ensemble_mean",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
        )
        entropy_indices, entropy_scores = select_batch(
            "predictive_entropy",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
        )
        ucb_indices, ucb_scores = select_batch(
            "ucb",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
        )
        self.assertEqual(mean_indices, [2, 3])
        self.assertEqual(entropy_indices, [5, 4])
        self.assertEqual(ucb_indices, [2, 3])
        np.testing.assert_allclose(mean_scores, self.candidate_scores["pred_mean"])
        np.testing.assert_allclose(entropy_scores, self.candidate_scores["pred_entropy"])
        np.testing.assert_allclose(
            ucb_scores,
            self.candidate_scores["pred_mean"]
            + self.config.discovery_ucb_beta * self.candidate_scores["pred_std"],
        )

    def test_similarity_penalized_mean_requires_sequence_context(self):
        with self.assertRaisesRegex(ValueError, "candidate_sequences"):
            select_batch(
                "similarity_penalized_mean",
                batch_size=2,
                candidate_scores=self.candidate_scores,
                labeled_embeddings=np.empty((0, 0), dtype=float),
                config=self.config,
                seed=7,
            )

    def test_similarity_penalized_mean_ranks_by_mean_minus_similarity(self):
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["pred_mean"] = np.array([0.95, 0.90, 0.96], dtype=float)
        for key in [
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:3], dtype=float)
        indices, scores = select_batch(
            "similarity_penalized_mean",
            batch_size=1,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
            candidate_sequences=["AAAAA", "CCCCC", "AAAAC"],
            reference_sequences=["AAAAA"],
        )
        self.assertEqual(indices, [1])
        self.assertAlmostEqual(scores[0], 0.85)
        self.assertAlmostEqual(scores[1], 0.90)
        self.assertAlmostEqual(scores[2], 0.88)

    def test_similarity_penalized_mean_greedy_updates_reference_set(self):
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["pred_mean"] = np.array([0.95, 0.90, 0.96], dtype=float)
        for key in [
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:3], dtype=float)
        sequences = ["AAAAA", "CCCCC", "AAAAC"]
        indices, scores = select_batch(
            "similarity_penalized_mean",
            batch_size=2,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
            candidate_sequences=sequences,
            reference_sequences=["AAAAA"],
        )
        diagnostics = acquisition_diagnostics(
            "similarity_penalized_mean",
            indices,
            candidate_scores,
            np.empty((0, 0), dtype=float),
            scores,
            self.config,
            seed=7,
            candidate_sequences=sequences,
            reference_sequences=["AAAAA"],
        )
        self.assertEqual(indices, [1, 2])
        self.assertAlmostEqual(diagnostics[1]["similarity_penalty"], 0.0)
        self.assertAlmostEqual(diagnostics[2]["similarity_penalty"], 0.05)
        self.assertAlmostEqual(diagnostics[2]["selection_score"], 0.91)

    def test_similarity_penalized_mean_replay_ignores_length_penalty(self):
        config = RunConfig(run_name="similarity_no_length_penalty")
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["pred_mean"] = np.array([0.95, 0.96], dtype=float)
        for key in [
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:2], dtype=float)
        indices, scores = select_batch(
            "similarity_penalized_mean",
            batch_size=1,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=config,
            seed=7,
            candidate_sequences=["AAAAAAAAAAAAAAAAAAAA", "CCCCC"],
            reference_sequences=["CCCCC"],
        )
        self.assertEqual(indices, [0])
        self.assertAlmostEqual(scores[0], 0.95)

    def test_similarity_penalized_mean_can_select_unpenalized_mean(self):
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["pred_mean"] = np.array([0.95, 0.90, 0.96], dtype=float)
        for key in [
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:3], dtype=float)
        indices, scores = select_batch(
            "similarity_penalized_mean",
            batch_size=2,
            candidate_scores=candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
            candidate_sequences=["AAAAA", "CCCCC", "AAAAC"],
            reference_sequences=["AAAAA"],
            apply_similarity_penalty=False,
        )
        self.assertEqual(indices, [2, 0])
        np.testing.assert_allclose(scores, candidate_scores["pred_mean"])

    def test_new_pointwise_strategies_need_no_extra_models_or_embeddings(self):
        for strategy in ["ensemble_mean", "predictive_entropy", "ucb"]:
            self.assertFalse(requires_family_models(strategy))
            self.assertFalse(requires_embeddings(strategy))

    def test_similarity_penalized_mean_needs_no_extra_models_or_embeddings(self):
        self.assertFalse(requires_family_models("similarity_penalized_mean"))
        self.assertFalse(requires_embeddings("similarity_penalized_mean"))

    def test_embedding_farthest_first_requires_embeddings_only(self):
        self.assertFalse(requires_family_models("embedding_farthest_first"))
        self.assertTrue(requires_embeddings("embedding_farthest_first"))

    def test_cluster_diverse_representative_requires_embeddings_only(self):
        self.assertFalse(requires_family_models("cluster_diverse_representative"))
        self.assertTrue(requires_embeddings("cluster_diverse_representative"))

    def test_pointwise_diagnostics_are_reporting_only(self):
        indices, scores = select_batch(
            "ensemble_mi",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.empty((0, 0), dtype=float),
            config=self.config,
            seed=7,
        )
        diagnostics = acquisition_diagnostics(
            "ensemble_mi",
            indices,
            self.candidate_scores,
            np.empty((0, 0), dtype=float),
            scores,
            self.config,
            seed=7,
        )
        self.assertEqual(indices, [5, 4])
        self.assertEqual(diagnostics[5]["selection_rank"], 1)
        self.assertEqual(diagnostics[4]["selection_rank"], 2)
        np.testing.assert_allclose(
            [diagnostics[index]["pointwise_score"] for index in range(len(scores))],
            scores,
        )
        np.testing.assert_allclose(
            [diagnostics[index]["selection_score"] for index in range(len(scores))],
            scores,
        )

    def test_cluster_diagnostics_report_centroid_distances(self):
        indices, scores = select_batch(
            "cluster_representative",
            batch_size=3,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=np.array([[0.0, 0.0]], dtype=float),
            config=self.config,
            seed=7,
        )
        diagnostics = acquisition_diagnostics(
            "cluster_representative",
            indices,
            self.candidate_scores,
            np.array([[0.0, 0.0]], dtype=float),
            scores,
            self.config,
            seed=7,
        )
        for index in indices:
            self.assertNotEqual(diagnostics[index]["cluster_id"], "")
            self.assertGreaterEqual(diagnostics[index]["distance_to_centroid"], 0.0)
            self.assertLessEqual(diagnostics[index]["selection_score"], 0.0)
            self.assertNotEqual(diagnostics[index]["distance_to_labeled"], "")
        self.assertEqual(diagnostics[indices[0]]["selection_rank"], 1)

    def test_cluster_diverse_diagnostics_report_cluster_novelty(self):
        config = RunConfig(run_name="cluster_diverse_diagnostics")
        config.diversity_prefilter_multiplier = 2
        candidate_scores = dict(self.candidate_scores)
        candidate_scores["avg_embedding"] = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
                [10.2, 0.0],
            ],
            dtype=float,
        )
        for key in [
            "pred_mean",
            "pred_std",
            "pred_entropy",
            "pred_expected_entropy",
            "pred_mutual_information",
            "committee_vote_entropy",
            "committee_prob_std",
        ]:
            candidate_scores[key] = np.asarray(candidate_scores[key][:4], dtype=float)
        labeled = np.array([[0.0, 0.0]], dtype=float)
        indices, scores = select_batch(
            "cluster_diverse_representative",
            batch_size=1,
            candidate_scores=candidate_scores,
            labeled_embeddings=labeled,
            config=config,
            seed=7,
        )
        diagnostics = acquisition_diagnostics(
            "cluster_diverse_representative",
            indices,
            candidate_scores,
            labeled,
            scores,
            config,
            seed=7,
        )
        selected = indices[0]
        self.assertEqual(diagnostics[selected]["selection_rank"], 1)
        self.assertNotEqual(diagnostics[selected]["cluster_id"], "")
        self.assertGreaterEqual(diagnostics[selected]["distance_to_centroid"], 0.0)
        self.assertEqual(diagnostics[selected]["selection_score"], scores[selected])
        self.assertEqual(
            diagnostics[selected]["pointwise_score"],
            candidate_scores["pred_mean"][selected],
        )
        self.assertEqual(diagnostics[selected]["fallback_fill_count"], 0)

    def test_oed_diagnostics_report_selected_greedy_gains(self):
        labeled = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        indices, scores = select_batch(
            "oed_logdet",
            batch_size=3,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=7,
        )
        diagnostics = acquisition_diagnostics(
            "oed_logdet",
            indices,
            self.candidate_scores,
            labeled,
            scores,
            self.config,
            seed=7,
        )
        for index in indices:
            self.assertGreaterEqual(diagnostics[index]["oed_gain"], 0.0)
            self.assertEqual(diagnostics[index]["selection_score"], scores[index])
            self.assertNotEqual(diagnostics[index]["distance_to_labeled"], "")

    def test_hybrid_diagnostics_keep_mi_score_separate_from_diversity_rank(self):
        labeled = np.array([[0.0, 0.0], [9.2, 0.2]], dtype=float)
        indices, scores = select_batch(
            "hybrid_mi_diverse",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=11,
        )
        diagnostics = acquisition_diagnostics(
            "hybrid_mi_diverse",
            indices,
            self.candidate_scores,
            labeled,
            scores,
            self.config,
            seed=11,
        )
        uncertainty_index, diversity_index = indices
        self.assertEqual(diagnostics[uncertainty_index]["selection_score"], scores[uncertainty_index])
        self.assertEqual(diagnostics[uncertainty_index]["pointwise_score"], scores[uncertainty_index])
        self.assertEqual(diagnostics[uncertainty_index]["diversity_rank"], "")
        self.assertEqual(diagnostics[diversity_index]["pointwise_score"], scores[diversity_index])
        self.assertEqual(diagnostics[diversity_index]["diversity_rank"], 1)
        self.assertNotEqual(diagnostics[diversity_index]["distance_to_labeled"], "")
        self.assertGreater(diagnostics[diversity_index]["selection_score"], scores[diversity_index])

    def test_embedding_farthest_first_diagnostics_report_dynamic_distances(self):
        labeled = np.array([[0.0, 0.0]], dtype=float)
        indices, scores = select_batch(
            "embedding_farthest_first",
            batch_size=2,
            candidate_scores=self.candidate_scores,
            labeled_embeddings=labeled,
            config=self.config,
            seed=11,
        )
        diagnostics = acquisition_diagnostics(
            "embedding_farthest_first",
            indices,
            self.candidate_scores,
            labeled,
            scores,
            self.config,
            seed=11,
        )
        self.assertEqual(indices, [5, 2])
        for rank, index in enumerate(indices, start=1):
            self.assertEqual(diagnostics[index]["diversity_rank"], rank)
            self.assertEqual(diagnostics[index]["selection_score"], scores[index])
            self.assertEqual(diagnostics[index]["pointwise_score"], "")
            self.assertNotEqual(diagnostics[index]["distance_to_labeled"], "")
        self.assertGreater(
            diagnostics[indices[1]]["distance_to_labeled"],
            diagnostics[indices[1]]["selection_score"],
        )

    def test_generator_objective_scores_match_acquisition_quantities(self):
        mi_scores = candidate_objective_scores(
            "ensemble_mi",
            self.candidate_scores,
            np.empty((0, 0), dtype=float),
            self.config,
            seed=7,
        )
        qbc_scores = candidate_objective_scores(
            "family_qbc",
            self.candidate_scores,
            np.empty((0, 0), dtype=float),
            self.config,
            seed=7,
        )
        entropy_scores = candidate_objective_scores(
            "predictive_entropy",
            self.candidate_scores,
            np.empty((0, 0), dtype=float),
            self.config,
            seed=7,
        )
        np.testing.assert_allclose(
            mi_scores,
            self.candidate_scores["pred_mutual_information"],
        )
        np.testing.assert_allclose(
            qbc_scores,
            self.candidate_scores["committee_vote_entropy"],
        )
        np.testing.assert_allclose(
            entropy_scores,
            self.candidate_scores["pred_entropy"],
        )

    def test_embedding_novelty_scores_normalize_nearest_labeled_distances(self):
        labeled = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
        raw_scores, normalized_scores = embedding_novelty_scores(
            np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=float),
            labeled,
        )
        np.testing.assert_allclose(raw_scores, np.array([0.0, 5.0, 0.0]))
        np.testing.assert_allclose(normalized_scores, np.array([0.0, 1.0, 0.0]))

    def test_embedding_novelty_normalization_handles_zero_and_equal_scores(self):
        zero_raw, zero_normalized = embedding_novelty_scores(
            np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float),
            np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float),
        )
        np.testing.assert_allclose(zero_raw, np.zeros(2))
        np.testing.assert_allclose(zero_normalized, np.zeros(2))

        equal_raw, equal_normalized = embedding_novelty_scores(
            np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=float),
            np.array([[0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(equal_raw, np.ones(2))
        np.testing.assert_allclose(equal_normalized, np.ones(2))

    def test_embedding_novelty_falls_back_to_candidate_centroid(self):
        raw_scores, normalized_scores = embedding_novelty_scores(
            np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=float),
            np.empty((0, 2), dtype=float),
        )
        np.testing.assert_allclose(raw_scores, np.array([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(normalized_scores, np.array([1.0, 1.0, 0.0]))

    def test_embedding_novelty_generator_objective_returns_normalized_scores(self):
        scores = candidate_objective_scores(
            "embedding_novelty",
            self.candidate_scores,
            np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float),
            self.config,
            seed=7,
        )
        raw_scores, expected = embedding_novelty_scores(
            self.candidate_scores["avg_embedding"],
            np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(scores, expected)
        self.assertGreater(float(raw_scores.max()), 1.0)

    def test_pointwise_strategies_map_to_matching_generator_objectives(self):
        for strategy in ["ensemble_mean", "predictive_entropy", "ucb"]:
            self.assertEqual(
                generator_objective_for_strategy(
                    strategy,
                    "match_acquisition",
                    self.config,
                ),
                strategy,
            )

    def test_broad_pool_mapping_avoids_mean_probability_bias(self):
        self.assertEqual(
            generator_objective_for_strategy("random", "match_acquisition", self.config),
            "broad_pool",
        )
        self.assertEqual(
            generator_objective_for_strategy(
                "cluster_representative",
                "match_acquisition",
                self.config,
            ),
            "broad_pool",
        )
        self.assertEqual(
            generator_objective_for_strategy(
                "cluster_diverse_representative",
                "match_acquisition",
                self.config,
            ),
            "embedding_novelty",
        )
        self.assertTrue(generator_objective_requires_embeddings("embedding_novelty"))
        self.assertEqual(
            generator_objective_for_strategy(
                "embedding_farthest_first",
                "match_acquisition",
                self.config,
            ),
            "broad_pool",
        )
        self.assertEqual(
            generator_objective_for_strategy(
                "hybrid_mi_diverse",
                "match_acquisition",
                self.config,
            ),
            "hybrid_two_pool",
        )
        self.assertTrue(generator_objective_requires_embeddings("hybrid_two_pool"))
        self.assertEqual(
            generator_objective_for_strategy(
                "similarity_penalized_mean",
                "match_acquisition",
                self.config,
            ),
            "ensemble_mean",
        )
        scores = candidate_objective_scores(
            "broad_pool",
            self.candidate_scores,
            np.empty((0, 0), dtype=float),
            self.config,
            seed=7,
        )
        np.testing.assert_allclose(scores, np.zeros(len(scores)))

    def test_oed_generator_objective_is_finite_single_candidate_gain(self):
        scores = candidate_objective_scores(
            "oed_logdet",
            self.candidate_scores,
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            self.config,
            seed=7,
        )
        self.assertEqual(scores.shape, (len(self.embeddings),))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_embedding_novelty_fitness_components_export_raw_and_normalized_scores(self):
        config = RunConfig(
            run_name="embedding_novelty_components",
            use_similarity_penalty=False,
            use_length_penalty=True,
            preferred_length_min=4,
            preferred_length_max=4,
        )
        candidate_scores = {
            **self.candidate_scores,
            "avg_embedding": np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float),
        }
        components = generation_fitness_components(
            ["AAAA", "AAAAAA"],
            candidate_scores,
            "embedding_novelty",
            config,
            labeled_embeddings=np.array([[0.0, 0.0]], dtype=float),
        )
        np.testing.assert_allclose(
            components["embedding_novelty_raw"],
            np.array([0.0, 5.0]),
        )
        np.testing.assert_allclose(
            components["generator_utility_score"],
            np.array([0.0, 1.0]),
        )
        np.testing.assert_allclose(
            components["generator_fitness"],
            components["generator_utility_score"] - components["length_penalty"],
        )

    def test_hybrid_two_pool_splits_deduplicates_and_refills_underrepresented_pool(self):
        config = RunConfig(run_name="hybrid_two_pool_test", candidate_pool_min=5)
        self.assertEqual(_hybrid_two_pool_targets(config.candidate_pool_min), (3, 2))

        def meta(sequence: str, objective: str, utility: float) -> dict[str, float | str]:
            payload: dict[str, float | str] = {
                "generator_objective": objective,
                "generator_utility_score": utility,
                "similarity_penalty": 0.0,
                "length_penalty": 0.0,
                "generator_fitness": utility,
            }
            if objective == "ensemble_mi":
                payload["normalized_mi"] = utility
            if objective == "embedding_novelty":
                payload["embedding_novelty_raw"] = utility * 10
                payload["normalized_embedding_novelty"] = utility
            return payload

        calls = []

        def fake_single(_ensemble, existing_sequences, _config, **kwargs):
            calls.append((set(existing_sequences), kwargs))
            objective = kwargs["objective"]
            target = kwargs["min_unique"]
            if objective == "ensemble_mi" and target == 3:
                sequences = ["MI1", "DUP", "MI3"]
            elif objective == "embedding_novelty" and target == 2:
                sequences = ["DUP", "NOV2"]
            elif objective == "embedding_novelty" and target == 1:
                self.assertIn("MI1", existing_sequences)
                self.assertIn("NOV2", existing_sequences)
                sequences = ["NOV_FILL"]
            else:
                raise AssertionError(f"Unexpected call: {objective=} {target=}")
            return sequences, {
                sequence: meta(sequence, objective, float(index + 1) / 10)
                for index, sequence in enumerate(sequences)
            }

        with mock.patch(
            "active_learning_thesis.generative._generate_candidate_sequences_single",
            side_effect=fake_single,
        ):
            sequences, metadata = generate_candidate_sequences(
                object(),
                {"KNOWN"},
                config,
                objective="hybrid_two_pool",
                return_metadata=True,
            )

        self.assertEqual(sequences, ["MI1", "DUP", "MI3", "NOV2", "NOV_FILL"])
        self.assertEqual([call[1]["objective"] for call in calls], [
            "ensemble_mi",
            "embedding_novelty",
            "embedding_novelty",
        ])
        self.assertEqual(metadata["DUP"]["generator_subpool"], "ensemble_mi")
        self.assertEqual(metadata["NOV_FILL"]["generator_subpool"], "embedding_novelty_fill")
        self.assertEqual(metadata["NOV_FILL"]["generator_objective"], "hybrid_two_pool")
        self.assertEqual(metadata["NOV_FILL"]["subpool_target"], 1)
        self.assertEqual(metadata["NOV_FILL"]["subpool_fill_count"], 1)
        self.assertEqual(metadata["NOV_FILL"]["deduplicated_count"], 1)
        self.assertEqual(metadata["NOV_FILL"]["subpool_unique_count_after_dedup"], 2)

    def test_hybrid_two_pool_uses_broad_pool_fallback_when_refill_fails(self):
        config = RunConfig(run_name="hybrid_two_pool_fallback_test", candidate_pool_min=3)
        novelty_calls = 0

        def fake_single(_ensemble, _existing_sequences, _config, **kwargs):
            nonlocal novelty_calls
            objective = kwargs["objective"]
            target = kwargs["min_unique"]
            if objective == "ensemble_mi":
                return ["DUP", "MI2"][:target], {
                    sequence: {
                        "generator_objective": objective,
                        "generator_utility_score": 0.1,
                        "similarity_penalty": 0.0,
                        "length_penalty": 0.0,
                        "generator_fitness": 0.1,
                    }
                    for sequence in ["DUP", "MI2"][:target]
                }
            if objective == "embedding_novelty":
                novelty_calls += 1
                if target == 1 and novelty_calls == 1:
                    return ["DUP"], {
                        "DUP": {
                            "generator_objective": objective,
                            "generator_utility_score": 0.2,
                            "similarity_penalty": 0.0,
                            "length_penalty": 0.0,
                            "generator_fitness": 0.2,
                        }
                    }
                raise RuntimeError("refill failed")
            if objective == "broad_pool":
                return ["FALLBACK"], {
                    "FALLBACK": {
                        "generator_objective": objective,
                        "generator_utility_score": 0.0,
                        "similarity_penalty": 0.0,
                        "length_penalty": 0.0,
                        "generator_fitness": 0.0,
                    }
                }
            raise AssertionError(f"Unexpected objective: {objective}")

        with mock.patch(
            "active_learning_thesis.generative._generate_candidate_sequences_single",
            side_effect=fake_single,
        ):
            sequences, metadata = generate_candidate_sequences(
                object(),
                {"KNOWN"},
                config,
                objective="hybrid_two_pool",
                return_metadata=True,
            )

        self.assertEqual(sequences, ["DUP", "MI2", "FALLBACK"])
        self.assertEqual(metadata["FALLBACK"]["generator_subpool"], "broad_pool_fallback")
        self.assertEqual(metadata["FALLBACK"]["generator_objective"], "hybrid_two_pool")


class BinaryThresholdTests(unittest.TestCase):
    def test_pr_best_f1_can_improve_over_fixed_half_threshold(self):
        metrics = evaluate_binary_classifier(
            [1, 1, 0, 0],
            [0.4, 0.35, 0.3, 0.2],
            threshold_strategy="pr_best_f1",
        )
        self.assertEqual(metrics["threshold_strategy"], "pr_best_f1")
        self.assertEqual(metrics["threshold_source"], "evaluation_dataset")
        self.assertGreaterEqual(metrics["f1"], metrics["f1_fixed_0_5"])
        self.assertEqual(metrics["f1_fixed_0_5"], 0.0)

    def test_pr_best_f1_tie_breaks_to_higher_threshold(self):
        metrics = evaluate_binary_classifier(
            [1, 0, 0],
            [0.8, 0.7, 0.6],
            threshold_strategy="pr_best_f1",
        )
        self.assertEqual(metrics["decision_threshold"], 0.8)
        self.assertEqual(metrics["f1"], 1.0)

    def test_applied_validation_threshold_keeps_selection_metadata(self):
        metrics = evaluate_binary_classifier(
            [1, 0, 0],
            [0.9, 0.8, 0.1],
            threshold=0.75,
            threshold_strategy="pr_best_f1",
            threshold_source="validation",
            threshold_selection_f1=0.72,
        )
        self.assertEqual(metrics["decision_threshold"], 0.75)
        self.assertEqual(metrics["threshold_source"], "validation")
        self.assertEqual(metrics["threshold_selection_f1"], 0.72)
        self.assertIn("precision_fixed_0_5", metrics)

    def test_threshold_independent_metrics_match_fixed_strategy(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.9, 0.4, 0.8]
        fixed = evaluate_binary_classifier(labels, scores, threshold_strategy="fixed_0_5")
        pr_best = evaluate_binary_classifier(labels, scores, threshold_strategy="pr_best_f1")
        for key in ["roc_auc", "pr_auc", "brier_score", "log_loss", "ece_10", "mce_10"]:
            self.assertAlmostEqual(fixed[key], pr_best[key])


class DiscoveryTests(unittest.TestCase):
    def test_discovery_utilities_are_finite_and_ucb_prefers_higher_std(self):
        config = RunConfig(run_name="discovery_test")
        mean = np.array([0.6, 0.6], dtype=float)
        std = np.array([0.05, 0.2], dtype=float)
        ucb = discovery_utility_scores("ucb", mean, std, 0.5, config, seed=7)
        self.assertGreater(ucb[1], ucb[0])
        for strategy in config.discovery_strategies:
            scores = discovery_utility_scores(strategy, mean, std, 0.5, config, seed=11)
            self.assertTrue(np.all(np.isfinite(scores)))

    def test_ei_and_pi_respect_incumbent_and_xi(self):
        config = RunConfig(
            run_name="ei_pi_test",
            discovery_improvement_xi=0.01,
        )
        mean = np.array([0.45, 0.75], dtype=float)
        std = np.array([0.1, 0.1], dtype=float)
        ei = discovery_utility_scores("ei", mean, std, 0.5, config, seed=3)
        pi = discovery_utility_scores("pi", mean, std, 0.5, config, seed=3)
        self.assertGreater(ei[1], ei[0])
        self.assertGreater(pi[1], pi[0])

    def test_mes_is_reproducible_for_fixed_seed(self):
        config = RunConfig(run_name="mes_test", discovery_mes_samples=32)
        mean = np.array([0.55, 0.7, 0.8], dtype=float)
        std = np.array([0.05, 0.1, 0.15], dtype=float)
        scores_a = discovery_utility_scores("mes", mean, std, 0.0, config, seed=17)
        scores_b = discovery_utility_scores("mes", mean, std, 0.0, config, seed=17)
        np.testing.assert_allclose(scores_a, scores_b)

    def test_ensemble_mean_population_fitness_matches_probability_path(self):
        config = RunConfig(run_name="discovery_regression_test")
        sequences = ["AAAAA", "AAAAC", "CCCCCCCCCCC"]
        probabilities = np.array([0.61, 0.42, 0.88], dtype=float)
        old_scores = _population_fitness_from_probabilities(sequences, probabilities, config)
        new_scores = _population_fitness_from_utilities(sequences, probabilities, config)
        np.testing.assert_allclose(old_scores, new_scores, atol=1e-12)

    def test_generation_fitness_can_disable_similarity_penalty(self):
        config = RunConfig(run_name="penalty_toggle_test")
        sequences = ["AAAAA", "AAAAC"]
        candidate_scores = {
            "pred_mean": np.array([0.6, 0.6], dtype=float),
            "pred_std": np.array([0.0, 0.0], dtype=float),
        }
        components = generation_fitness_components(
            sequences,
            candidate_scores,
            "ensemble_mean",
            config,
            use_similarity_penalty=False,
            use_length_penalty=True,
        )
        np.testing.assert_allclose(components["similarity_penalty"], np.zeros(2))
        np.testing.assert_allclose(
            components["generator_fitness"],
            components["generator_utility_score"] - components["length_penalty"],
        )

    def test_generation_fitness_reports_penalized_score_separately(self):
        config = RunConfig(run_name="penalty_reporting_test")
        sequences = ["AAAAA", "AAAAC"]
        candidate_scores = {
            "pred_mean": np.array([0.6, 0.6], dtype=float),
            "pred_std": np.array([0.0, 0.0], dtype=float),
        }
        components = generation_fitness_components(
            sequences,
            candidate_scores,
            "ensemble_mean",
            config,
            use_similarity_penalty=True,
            use_length_penalty=True,
        )
        self.assertTrue(np.any(components["similarity_penalty"] > 0))
        self.assertLess(components["generator_fitness"][0], components["generator_utility_score"][0])

    def test_distance_helpers_return_expected_shapes(self):
        candidates = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=float)
        reference = np.array([[0.0, 0.0]], dtype=float)
        min_distances = min_distances_to_reference(candidates, reference)
        self.assertEqual(min_distances.shape, (2,))
        self.assertEqual(mean_pairwise_distance(candidates), 5.0)


class WorkflowDiscoveryTests(unittest.TestCase):
    def test_resolve_discovery_ensemble_dir_prefers_latest_post_ingest_else_baseline(self):
        with workspace_tempdir("altest") as temp_dir:
            models_root = Path(temp_dir) / "models"
            baseline = models_root / "real_al" / "round_000" / "baseline" / "ensemble"
            latest = models_root / "real_al" / "round_003" / "post_ingest" / "ensemble"
            baseline.mkdir(parents=True)
            latest.mkdir(parents=True)
            (baseline / "ap_sp_member_00.h5").write_text("", encoding="utf-8")
            (latest / "ap_sp_member_00.h5").write_text("", encoding="utf-8")

            chosen_dir, info = _resolve_discovery_ensemble_dir(models_root)

        self.assertEqual(chosen_dir, latest)
        self.assertEqual(info["surrogate_stage"], "post_ingest")
        self.assertEqual(info["surrogate_round_id"], 3)

    def test_resolve_discovery_ensemble_dir_falls_back_to_baseline(self):
        with workspace_tempdir("altest") as temp_dir:
            models_root = Path(temp_dir) / "models"
            baseline = models_root / "real_al" / "round_000" / "baseline" / "ensemble"
            baseline.mkdir(parents=True)
            (baseline / "ap_sp_member_00.h5").write_text("", encoding="utf-8")

            chosen_dir, info = _resolve_discovery_ensemble_dir(models_root)

        self.assertEqual(chosen_dir, baseline)
        self.assertEqual(info["surrogate_stage"], "baseline")
        self.assertEqual(info["surrogate_round_id"], 0)

    def test_run_discovery_writes_artifacts_without_mutating_ledger(self):
        config = RunConfig(
            run_name="discovery_run_test",
            output_root=".",
            candidate_pool_min=3,
            batch_size=2,
            discovery_export_count=2,
            discovery_strategies=["ensemble_mean"],
        )
        with workspace_tempdir("altest") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True)
            config.save(run_dir / "config.json")
            ledger_rows = [
                {
                    "sequence": "TRAINA",
                    "label": "1",
                    "label_source": "experimental",
                    "split": "train_pool",
                    "mode": "experimental",
                    "round_id": "0",
                    "status": "train_pool",
                    "generator_origin": "experimental_dataset",
                    "replay_role": "seed",
                },
                {
                    "sequence": "TRAINB",
                    "label": "0",
                    "label_source": "experimental",
                    "split": "train_pool",
                    "mode": "experimental",
                    "round_id": "0",
                    "status": "train_pool",
                    "generator_origin": "experimental_dataset",
                    "replay_role": "hidden",
                },
            ]
            save_ledger(run_dir / "ledger.csv", ledger_rows)
            baseline = run_dir / "models" / "real_al" / "round_000" / "baseline" / "ensemble"
            latest = run_dir / "models" / "real_al" / "round_001" / "post_ingest" / "ensemble"
            baseline.mkdir(parents=True)
            latest.mkdir(parents=True)
            (baseline / "ap_sp_member_00.h5").write_text("", encoding="utf-8")
            (latest / "ap_sp_member_00.h5").write_text("", encoding="utf-8")
            ledger_before = (run_dir / "ledger.csv").read_text(encoding="utf-8")

            training_summary = {
                "pred_mean": np.array([0.3, 0.8], dtype=float),
                "pred_std": np.array([0.05, 0.1], dtype=float),
                "pred_entropy": np.array([0.0, 0.0], dtype=float),
                "pred_expected_entropy": np.array([0.0, 0.0], dtype=float),
                "pred_mutual_information": np.array([0.0, 0.0], dtype=float),
                "ensemble_member_probs": np.array([[0.3], [0.8]], dtype=float),
                "avg_embedding": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            }
            candidate_summary = {
                "pred_mean": np.array([0.9, 0.7, 0.6], dtype=float),
                "pred_std": np.array([0.2, 0.1, 0.05], dtype=float),
                "pred_entropy": np.array([0.0, 0.0, 0.0], dtype=float),
                "pred_expected_entropy": np.array([0.0, 0.0, 0.0], dtype=float),
                "pred_mutual_information": np.array([0.0, 0.0, 0.0], dtype=float),
                "ensemble_member_probs": np.array([[0.9], [0.7], [0.6]], dtype=float),
                "avg_embedding": np.array([[2.0, 2.0], [2.5, 2.5], [3.0, 3.0]], dtype=float),
            }

            def fake_score(_ensemble, sequences, include_embeddings=False, **_kwargs):
                if sequences == ["TRAINA", "TRAINB"]:
                    return training_summary
                if sequences == ["DISC1", "DISC2", "DISC3"]:
                    return candidate_summary
                raise AssertionError(f"Unexpected sequence request: {sequences}")

            with mock.patch("active_learning_thesis.workflow.ensure_predictive_runtime"), mock.patch(
                "active_learning_thesis.workflow.load_ensemble_from_dir",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ) as load_mock, mock.patch(
                "active_learning_thesis.workflow.generate_candidate_sequences",
                return_value=(
                    ["DISC1", "DISC2", "DISC3"],
                    {
                        sequence: {
                            "generator_objective": "ensemble_mean",
                            "generator_utility_score": score,
                            "similarity_penalty": 0.0,
                            "length_penalty": 0.0,
                            "generator_fitness": score,
                        }
                        for sequence, score in zip(
                            ["DISC1", "DISC2", "DISC3"],
                            [0.9, 0.7, 0.6],
                        )
                    },
                ),
            ), mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                side_effect=fake_score,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                results = run_discovery(run_dir)

            self.assertEqual(
                load_mock.call_args[0][0],
                latest,
            )
            self.assertIn("ensemble_mean", results)
            self.assertEqual(
                (run_dir / "ledger.csv").read_text(encoding="utf-8"),
                ledger_before,
            )
            with (run_dir / "discovery" / "ensemble_mean" / "top_batch.csv").open("r", encoding="utf-8") as handle:
                top_rows = list(csv.DictReader(handle))
            self.assertEqual(len(top_rows), 2)
            self.assertEqual(top_rows[0]["generator_objective"], "ensemble_mean")
            self.assertEqual(top_rows[0]["similarity_penalty"], "0.0")
            with (run_dir / "discovery" / "aggregate_summary.csv").open("r", encoding="utf-8") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate_rows), 1)


class WorkflowEvaluationPolicyTests(unittest.TestCase):
    @staticmethod
    def _fixed_metrics() -> dict[str, float]:
        return {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.7466666667,
            "balanced_accuracy": 0.72,
            "gmean": 0.71,
            "roc_auc": 0.81,
            "pr_auc": 0.84,
            "decision_threshold": 0.42,
            "threshold_strategy": "pr_best_f1",
            "threshold_selection_f1": 0.7466666667,
            "threshold_source": "evaluation_dataset",
            "f1_fixed_0_5": 0.70,
        }

    def test_init_run_writes_validation_baseline_metrics(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="init_validation_policy",
                output_root=temp_dir,
                epochs=1,
            )
            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_holdout",
                side_effect=AssertionError("holdout should not be used during init-run"),
            ):
                run_dir = init_run(config)

            metrics = json.loads(
                (Path(run_dir) / "metrics" / "baseline_round_000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["evaluation_dataset"], "validation")
            self.assertEqual(metrics["stage"], "baseline")

    def test_init_run_can_skip_local_baseline_training(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="config_only_remote_seed",
                output_root=temp_dir,
                epochs=1,
            )
            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime",
                side_effect=AssertionError("predictive runtime should be skipped"),
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                side_effect=AssertionError("baseline training should be skipped"),
            ):
                run_dir = init_run(config, train_baseline=False)

            run_path = Path(run_dir)
            self.assertTrue((run_path / "config.json").exists())
            self.assertTrue((run_path / "ledger.csv").exists())
            self.assertFalse((run_path / "metrics" / "baseline_round_000.json").exists())

    def test_run_replay_uses_validation_metrics(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="replay_validation_policy",
                output_root=temp_dir,
                epochs=1,
                max_rounds=0,
            )
            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_holdout",
                side_effect=AssertionError("holdout should not be used during replay"),
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                results = run_replay(Path(run_dir))

            for strategy_metrics in results.values():
                self.assertEqual(len(strategy_metrics), 1)
                self.assertEqual(strategy_metrics[0]["evaluation_dataset"], "validation")

    def test_run_replay_writes_selected_batch_trace_only(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="replay_trace_policy",
                output_root=temp_dir,
                epochs=1,
                max_rounds=1,
                batch_size=2,
                replay_strategies=["ensemble_mean"],
            )

            def fake_score(_ensemble, sequences, **_kwargs):
                count = len(sequences)
                return {
                    "pred_mean": np.linspace(0.1, 0.9, count),
                    "pred_std": np.linspace(0.01, 0.05, count),
                    "pred_entropy": np.linspace(0.2, 0.6, count),
                    "pred_expected_entropy": np.zeros(count),
                    "pred_mutual_information": np.linspace(0.01, 0.09, count),
                }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                side_effect=fake_score,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                run_replay(Path(run_dir))

            trace_path = (
                Path(run_dir)
                / "replay"
                / "ensemble_mean"
                / "round_000_selected_batch.csv"
            )
            with trace_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("selection_rank", rows[0])
            self.assertIn("pointwise_score", rows[0])
            self.assertIn("selection_score", rows[0])
            self.assertEqual(rows[0]["selection_rank"], "1")
            self.assertEqual(rows[0]["pointwise_score"], rows[0]["acquisition_score"])

    def test_run_replay_writes_similarity_penalized_trace(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="replay_similarity_penalty_trace",
                output_root=temp_dir,
                epochs=1,
                max_rounds=1,
                batch_size=2,
                replay_strategies=["similarity_penalized_mean"],
            )

            def fake_score(_ensemble, sequences, **_kwargs):
                count = len(sequences)
                return {
                    "pred_mean": np.linspace(0.1, 0.9, count),
                    "pred_std": np.linspace(0.01, 0.05, count),
                    "pred_entropy": np.linspace(0.2, 0.6, count),
                    "pred_expected_entropy": np.zeros(count),
                    "pred_mutual_information": np.linspace(0.01, 0.09, count),
                }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                side_effect=fake_score,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                run_replay(Path(run_dir))

            trace_path = (
                Path(run_dir)
                / "replay"
                / "similarity_penalized_mean"
                / "round_000_selected_batch.csv"
            )
            with trace_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("similarity_penalty", rows[0])
            self.assertNotEqual(rows[0]["similarity_penalty"], "")
            self.assertEqual(rows[0]["selection_score"], rows[0]["acquisition_score"])
            self.assertNotEqual(rows[0]["pointwise_score"], "")

    def test_propose_round_outputs_interpretability_columns(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="proposal_diagnostics_policy",
                output_root=temp_dir,
                epochs=1,
                batch_size=2,
                candidate_pool_min=3,
                real_strategy="ensemble_mean",
            )
            candidate_summary = {
                "pred_mean": np.array([0.3, 0.9, 0.7], dtype=float),
                "pred_std": np.array([0.02, 0.05, 0.04], dtype=float),
                "pred_entropy": np.array([0.4, 0.5, 0.45], dtype=float),
                "pred_expected_entropy": np.array([0.1, 0.1, 0.1], dtype=float),
                "pred_mutual_information": np.array([0.3, 0.4, 0.35], dtype=float),
                "ensemble_member_probs": np.array(
                    [[0.3], [0.9], [0.7]],
                    dtype=float,
                ),
            }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.generate_candidate_sequences",
                return_value=(
                    ["CAND_A", "CAND_B", "CAND_C"],
                    {
                        sequence: {
                            "generator_objective": "ensemble_mean",
                            "generator_utility_score": score,
                            "similarity_penalty": 0.0,
                            "length_penalty": 0.0,
                            "generator_fitness": score,
                        }
                        for sequence, score in zip(
                            ["CAND_A", "CAND_B", "CAND_C"],
                            [0.3, 0.9, 0.7],
                        )
                    },
                ),
            ), mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                return_value=candidate_summary,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                batch_path = propose_round(Path(run_dir))

            scored_path = Path(run_dir) / "candidates" / "round_001_scored.csv"
            with scored_path.open("r", encoding="utf-8", newline="") as handle:
                scored_rows = list(csv.DictReader(handle))
            with Path(batch_path).open("r", encoding="utf-8", newline="") as handle:
                batch_rows = list(csv.DictReader(handle))
            self.assertEqual(len(scored_rows), 3)
            self.assertEqual(len(batch_rows), 2)
            for field in [
                "selection_rank",
                "pointwise_score",
                "selection_score",
                "distance_to_labeled",
                "oed_gain",
                "diversity_rank",
            ]:
                self.assertIn(field, scored_rows[0])
                self.assertIn(field, batch_rows[0])
            selected_by_sequence = {row["sequence"]: row for row in batch_rows}
            self.assertEqual(selected_by_sequence["CAND_B"]["selection_rank"], "1")
            self.assertEqual(
                selected_by_sequence["CAND_B"]["pointwise_score"],
                selected_by_sequence["CAND_B"]["acquisition_score"],
            )

    def test_propose_round_cluster_diverse_uses_embedding_novelty_generation_metadata(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="proposal_embedding_novelty_policy",
                output_root=temp_dir,
                epochs=1,
                batch_size=2,
                candidate_pool_min=3,
                real_strategy="cluster_diverse_representative",
                use_similarity_penalty=False,
                use_length_penalty=True,
            )
            candidate_summary = {
                "pred_mean": np.array([0.3, 0.9, 0.7], dtype=float),
                "pred_std": np.array([0.02, 0.05, 0.04], dtype=float),
                "pred_entropy": np.array([0.4, 0.5, 0.45], dtype=float),
                "pred_expected_entropy": np.array([0.1, 0.1, 0.1], dtype=float),
                "pred_mutual_information": np.array([0.3, 0.4, 0.35], dtype=float),
                "avg_embedding": np.array(
                    [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
                    dtype=float,
                ),
                "ensemble_member_probs": np.array(
                    [[0.3], [0.9], [0.7]],
                    dtype=float,
                ),
            }
            generation_metadata = {
                "CAND_A": {
                    "generator_objective": "embedding_novelty",
                    "embedding_novelty_raw": 0.0,
                    "generator_utility_score": 0.0,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": -0.1,
                },
                "CAND_B": {
                    "generator_objective": "embedding_novelty",
                    "embedding_novelty_raw": 5.0,
                    "generator_utility_score": 0.5,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": 0.4,
                },
                "CAND_C": {
                    "generator_objective": "embedding_novelty",
                    "embedding_novelty_raw": 10.0,
                    "generator_utility_score": 1.0,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": 0.9,
                },
            }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow._labeled_embeddings",
                return_value=np.array([[0.0, 0.0]], dtype=float),
            ), mock.patch(
                "active_learning_thesis.workflow.generate_candidate_sequences",
                return_value=(["CAND_A", "CAND_B", "CAND_C"], generation_metadata),
            ) as generate_mock, mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                return_value=candidate_summary,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                propose_round(Path(run_dir))

            self.assertEqual(generate_mock.call_args.kwargs["objective"], "embedding_novelty")
            np.testing.assert_allclose(
                generate_mock.call_args.kwargs["labeled_embeddings"],
                np.array([[0.0, 0.0]], dtype=float),
            )

            scored_path = Path(run_dir) / "candidates" / "round_001_scored.csv"
            with scored_path.open("r", encoding="utf-8", newline="") as handle:
                scored_rows = list(csv.DictReader(handle))
            scored_by_sequence = {row["sequence"]: row for row in scored_rows}
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_objective"], "embedding_novelty")
            self.assertEqual(scored_by_sequence["CAND_C"]["embedding_novelty_raw"], "10.0")
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_utility_score"], "1.0")
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_fitness"], "0.9")

    def test_propose_round_hybrid_uses_two_pool_generation_metadata_and_rescores_once(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="proposal_hybrid_two_pool_policy",
                output_root=temp_dir,
                epochs=1,
                batch_size=2,
                candidate_pool_min=3,
                real_strategy="hybrid_mi_diverse",
                use_similarity_penalty=False,
                use_length_penalty=True,
            )
            candidate_summary = {
                "pred_mean": np.array([0.3, 0.9, 0.7], dtype=float),
                "pred_std": np.array([0.02, 0.05, 0.04], dtype=float),
                "pred_entropy": np.array([0.4, 0.5, 0.45], dtype=float),
                "pred_expected_entropy": np.array([0.1, 0.1, 0.1], dtype=float),
                "pred_mutual_information": np.array([0.3, 0.9, 0.35], dtype=float),
                "avg_embedding": np.array(
                    [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
                    dtype=float,
                ),
                "ensemble_member_probs": np.array(
                    [[0.3], [0.9], [0.7]],
                    dtype=float,
                ),
            }
            generation_metadata = {
                "CAND_A": {
                    "generator_objective": "hybrid_two_pool",
                    "generator_subpool": "ensemble_mi",
                    "subpool_target": 2,
                    "subpool_unique_count_after_dedup": 2,
                    "subpool_fill_count": 0,
                    "deduplicated_count": 1,
                    "subpool_rank": 1,
                    "normalized_mi": 0.8,
                    "generator_utility_score": 0.4,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": 0.3,
                },
                "CAND_B": {
                    "generator_objective": "hybrid_two_pool",
                    "generator_subpool": "ensemble_mi",
                    "subpool_target": 2,
                    "subpool_unique_count_after_dedup": 2,
                    "subpool_fill_count": 0,
                    "deduplicated_count": 1,
                    "subpool_rank": 2,
                    "normalized_mi": 1.0,
                    "generator_utility_score": 0.9,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": 0.8,
                },
                "CAND_C": {
                    "generator_objective": "hybrid_two_pool",
                    "generator_subpool": "embedding_novelty_fill",
                    "subpool_target": 1,
                    "subpool_unique_count_after_dedup": 1,
                    "subpool_fill_count": 1,
                    "deduplicated_count": 1,
                    "subpool_rank": 1,
                    "embedding_novelty_raw": 10.0,
                    "normalized_embedding_novelty": 1.0,
                    "generator_utility_score": 1.0,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.1,
                    "generator_fitness": 0.9,
                },
            }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow._labeled_embeddings",
                return_value=np.array([[0.0, 0.0]], dtype=float),
            ), mock.patch(
                "active_learning_thesis.workflow.generate_candidate_sequences",
                return_value=(["CAND_A", "CAND_B", "CAND_C"], generation_metadata),
            ) as generate_mock, mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                return_value=candidate_summary,
            ) as score_mock, mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                propose_round(Path(run_dir))

            self.assertEqual(generate_mock.call_args.kwargs["objective"], "hybrid_two_pool")
            self.assertEqual(score_mock.call_count, 1)
            self.assertEqual(score_mock.call_args.kwargs["include_embeddings"], True)

            scored_path = Path(run_dir) / "candidates" / "round_001_scored.csv"
            with scored_path.open("r", encoding="utf-8", newline="") as handle:
                scored_rows = list(csv.DictReader(handle))
            scored_by_sequence = {row["sequence"]: row for row in scored_rows}
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_objective"], "hybrid_two_pool")
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_subpool"], "embedding_novelty_fill")
            self.assertEqual(scored_by_sequence["CAND_C"]["subpool_fill_count"], "1")
            self.assertEqual(scored_by_sequence["CAND_C"]["deduplicated_count"], "1")
            self.assertEqual(scored_by_sequence["CAND_C"]["normalized_embedding_novelty"], "1.0")
            self.assertEqual(scored_by_sequence["CAND_C"]["generator_fitness"], "0.9")

    def test_propose_round_similarity_penalized_mean_forces_generation_penalty_only(self):
        with workspace_tempdir("altest") as temp_dir:
            config = RunConfig(
                run_name="proposal_similarity_penalty_policy",
                output_root=temp_dir,
                epochs=1,
                batch_size=2,
                candidate_pool_min=3,
                real_strategy="similarity_penalized_mean",
                use_similarity_penalty=False,
                use_length_penalty=False,
            )
            candidate_summary = {
                "pred_mean": np.array([0.3, 0.9, 0.7], dtype=float),
                "pred_std": np.array([0.02, 0.05, 0.04], dtype=float),
                "pred_entropy": np.array([0.4, 0.5, 0.45], dtype=float),
                "pred_expected_entropy": np.array([0.1, 0.1, 0.1], dtype=float),
                "pred_mutual_information": np.array([0.3, 0.4, 0.35], dtype=float),
                "ensemble_member_probs": np.array(
                    [[0.3], [0.9], [0.7]],
                    dtype=float,
                ),
            }
            generation_metadata = {
                sequence: {
                    "generator_objective": "ensemble_mean",
                    "generator_utility_score": score,
                    "similarity_penalty": penalty,
                    "length_penalty": 0.0,
                    "generator_fitness": score - penalty,
                }
                for sequence, score, penalty in zip(
                    ["CAND_A", "CAND_B", "CAND_C"],
                    [0.3, 0.9, 0.7],
                    [0.01, 0.22, 0.13],
                )
            }

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.train_ensemble",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value=self._fixed_metrics(),
            ), mock.patch(
                "active_learning_thesis.workflow.generate_candidate_sequences",
                return_value=(
                    ["CAND_A", "CAND_B", "CAND_C"],
                    generation_metadata,
                ),
            ) as generate_mock, mock.patch(
                "active_learning_thesis.workflow.score_sequences_with_ensemble",
                return_value=candidate_summary,
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                run_dir = init_run(config)
                batch_path = propose_round(Path(run_dir))

            self.assertTrue(generate_mock.call_args.kwargs["use_similarity_penalty"])
            self.assertFalse(generate_mock.call_args.kwargs["use_length_penalty"])
            self.assertEqual(generate_mock.call_args.kwargs["objective"], "ensemble_mean")

            scored_path = Path(run_dir) / "candidates" / "round_001_scored.csv"
            with scored_path.open("r", encoding="utf-8", newline="") as handle:
                scored_rows = list(csv.DictReader(handle))
            with Path(batch_path).open("r", encoding="utf-8", newline="") as handle:
                batch_rows = list(csv.DictReader(handle))

            selected_by_sequence = {row["sequence"]: row for row in batch_rows}
            self.assertEqual(selected_by_sequence["CAND_B"]["selection_rank"], "1")
            self.assertEqual(selected_by_sequence["CAND_B"]["acquisition_score"], "0.9")
            self.assertEqual(selected_by_sequence["CAND_B"]["selection_score"], "0.9")
            self.assertEqual(selected_by_sequence["CAND_B"]["similarity_penalty"], "0.22")
            scored_by_sequence = {row["sequence"]: row for row in scored_rows}
            self.assertEqual(scored_by_sequence["CAND_B"]["generator_objective"], "ensemble_mean")
            self.assertEqual(scored_by_sequence["CAND_B"]["similarity_penalty"], "0.22")

    def test_evaluate_final_writes_holdout_metrics(self):
        config = RunConfig(run_name="final_eval_config", output_root=".")
        with workspace_tempdir("altest") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True)
            config.save(run_dir / "config.json")
            save_ledger(
                run_dir / "ledger.csv",
                [
                    {
                        "sequence": "TRAINSEQ",
                        "label": "1",
                        "label_source": "experimental",
                        "split": "train_pool",
                        "mode": "experimental",
                        "round_id": "0",
                        "status": "train_pool",
                    },
                    {
                        "sequence": "HOLDOUTSEQ",
                        "label": "0",
                        "label_source": "experimental",
                        "split": "holdout",
                        "mode": "experimental",
                        "round_id": "0",
                        "status": "holdout",
                    },
                ],
            )
            baseline = run_dir / "models" / "real_al" / "round_000" / "baseline" / "ensemble"
            latest = run_dir / "models" / "real_al" / "round_002" / "post_ingest" / "ensemble"
            baseline.mkdir(parents=True)
            latest.mkdir(parents=True)
            (baseline / "ap_sp_member_00.h5").write_text("", encoding="utf-8")
            (latest / "ap_sp_member_00.h5").write_text("", encoding="utf-8")

            with mock.patch(
                "active_learning_thesis.workflow.ensure_predictive_runtime"
            ), mock.patch(
                "active_learning_thesis.workflow.load_ensemble_from_dir",
                return_value=[ManagedModel("AP_SP", 11, None, object())],
            ) as load_mock, mock.patch(
                "active_learning_thesis.workflow.evaluate_rows",
                return_value={
                    **self._fixed_metrics(),
                    "decision_threshold": 0.42,
                    "threshold_selection_f1": 0.73,
                    "threshold_source": "validation",
                },
            ), mock.patch(
                "active_learning_thesis.workflow.evaluate_holdout",
                return_value={
                    **self._fixed_metrics(),
                    "decision_threshold": 0.42,
                    "threshold_source": "validation",
                    "threshold_selection_f1": 0.73,
                },
            ), mock.patch(
                "active_learning_thesis.workflow._cleanup_tensorflow_runtime"
            ):
                metrics = evaluate_final(run_dir)

            self.assertEqual(load_mock.call_args[0][0], latest)
            self.assertEqual(metrics["evaluation_dataset"], "holdout")
            self.assertEqual(metrics["threshold_source"], "validation")
            self.assertEqual(metrics["validation_decision_threshold"], 0.42)
            self.assertEqual(metrics["surrogate_stage"], "post_ingest")
            saved_metrics = json.loads(
                (run_dir / "metrics" / "final_holdout.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_metrics["evaluation_dataset"], "holdout")



class CgmdImportTests(unittest.TestCase):
    def test_import_validation_rejects_missing_sequences(self):
        proposed_ledger = [
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "status": "proposed",
            },
            {
                "sequence": "CCCCC",
                "round_id": "1",
                "status": "proposed",
            },
        ]
        with workspace_tempdir("altest") as temp_dir:
            path = Path(temp_dir) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sequence", "round_id", "cgmd_label"],
                )
                writer.writeheader()
                writer.writerow(
                    {"sequence": "AAAAA", "round_id": "1", "cgmd_label": "1"}
                )
            with self.assertRaises(ValueError):
                _validate_cgmd_import(proposed_ledger, path)


class PerformanceOptimizationTests(unittest.TestCase):
    def test_batched_forward_pass_matches_direct_concatenation(self):
        class FakeModel:
            def __call__(self, batch_inputs, training=False):
                self.last_training = training
                return np.sum(batch_inputs, axis=1, keepdims=True)

        model = FakeModel()
        inputs = np.arange(15, dtype=float).reshape(5, 3)
        outputs = _batched_forward_pass(model, inputs, batch_size=2)
        expected = np.sum(inputs, axis=1, keepdims=True)
        np.testing.assert_allclose(outputs, expected)
        self.assertFalse(model.last_training)

    def test_batched_population_fitness_matches_scalar_utility(self):
        config = RunConfig(run_name="generative_perf_test")
        sequences = ["AAAAA", "AAAAC", "CCCCCCCCCCC"]
        mean_probabilities = np.array([0.61, 0.42, 0.88], dtype=float)
        population = [SimpleNamespace(sequence=sequence) for sequence in sequences]

        scalar = np.array(
            [
                probability
                - calculate_similarity_penalty(
                    sequence,
                    population,
                    config.allowed_amino_acids,
                )
                - calculate_length_penalty(sequence, config)
                for sequence, probability in zip(sequences, mean_probabilities)
            ],
            dtype=float,
        )
        batched = _population_fitness_from_probabilities(
            sequences,
            mean_probabilities,
            config,
        )
        np.testing.assert_allclose(batched, scalar, atol=1e-12)

    def test_ensemble_scoring_prepares_inputs_once_per_call(self):
        sequences = ["AAAAA", "CCCCC"]
        ensemble = [
            ManagedModel("AP_SP", 11, None, object()),
            ManagedModel("AP_SP", 23, None, object()),
        ]
        with mock.patch(
            "active_learning_thesis.predictive._prepare_inference_tensors",
            return_value="prepared_inputs",
        ) as prepare_mock, mock.patch(
            "active_learning_thesis.predictive._predict_probabilities_from_inputs",
            side_effect=[
                np.array([0.2, 0.4], dtype=float),
                np.array([0.6, 0.8], dtype=float),
            ],
        ) as predict_mock:
            summary = score_sequences_with_ensemble(ensemble, sequences)

        prepare_mock.assert_called_once_with("AP_SP", sequences)
        self.assertEqual(predict_mock.call_count, 2)
        np.testing.assert_allclose(
            summary["pred_mean"],
            np.array([0.4, 0.6], dtype=float),
        )

    def test_family_scoring_prepares_once_per_model_type(self):
        sequences = ["AAAAA", "CCCCC"]
        family_models = [
            ManagedModel("AP", 1, None, object()),
            ManagedModel("SP", 2, None, object()),
            ManagedModel("AP", 3, None, object()),
        ]

        def fake_predict(member, prepared_inputs):
            self.assertEqual(prepared_inputs, f"inputs-{member.model_name}")
            outputs = {
                1: np.array([0.1, 0.2], dtype=float),
                2: np.array([0.3, 0.4], dtype=float),
                3: np.array([0.5, 0.6], dtype=float),
            }
            return outputs[member.seed]

        with mock.patch(
            "active_learning_thesis.predictive._prepare_inference_tensors",
            side_effect=lambda model_name, _: f"inputs-{model_name}",
        ) as prepare_mock, mock.patch(
            "active_learning_thesis.predictive._predict_probabilities_from_inputs",
            side_effect=fake_predict,
        ):
            summary = score_sequences_with_family(family_models, sequences)

        self.assertEqual(prepare_mock.call_count, 2)
        np.testing.assert_allclose(
            summary["family_member_probs"],
            np.array(
                [
                    [0.1, 0.3, 0.5],
                    [0.2, 0.4, 0.6],
                ],
                dtype=float,
            ),
        )

    def test_in_memory_best_weights_matches_checkpoint_min_semantics(self):
        class DummyCallback:
            def set_model(self, model):
                self.model = model

        fake_tf = SimpleNamespace(
            keras=SimpleNamespace(callbacks=SimpleNamespace(Callback=DummyCallback))
        )

        class FakeModel:
            def __init__(self):
                self._weights = [np.array([-1.0], dtype=float)]

            def get_weights(self):
                return [weight.copy() for weight in self._weights]

            def set_weights(self, weights):
                self._weights = [weight.copy() for weight in weights]

        tracker = _InMemoryBestWeights(fake_tf, "val_loss")
        callback = tracker.callback()
        model = FakeModel()
        callback.set_model(model)

        monitored_values = [0.8, 0.4, 0.4, 0.2, 0.25]
        for epoch, value in enumerate(monitored_values):
            model._weights = [np.array([epoch], dtype=float)]
            callback.on_epoch_end(epoch, {"val_loss": value})

        self.assertEqual(tracker.best_epoch, 3)
        tracker.restore(model)
        np.testing.assert_allclose(model.get_weights()[0], np.array([3.0], dtype=float))

    def test_train_model_restores_best_weights_and_avoids_reload(self):
        class DummyCallback:
            def set_model(self, model):
                self.model = model

        class DummyLearningRateScheduler(DummyCallback):
            def __init__(self, schedule):
                self.schedule = schedule

        class DummyOptimizer:
            def __init__(self, learning_rate):
                self.learning_rate = learning_rate

        load_model_mock = mock.Mock(side_effect=AssertionError("train_model should not reload the model after fit"))
        fake_tf = SimpleNamespace(
            keras=SimpleNamespace(
                backend=SimpleNamespace(clear_session=lambda: None),
                utils=SimpleNamespace(set_random_seed=lambda seed: None),
                optimizers=SimpleNamespace(Adam=DummyOptimizer),
                callbacks=SimpleNamespace(
                    Callback=DummyCallback,
                    LearningRateScheduler=DummyLearningRateScheduler,
                ),
                models=SimpleNamespace(load_model=load_model_mock),
            )
        )

        class FakeModel:
            def __init__(self, epoch_logs):
                self.epoch_logs = epoch_logs
                self.current_weights = [np.array([-1.0], dtype=float)]
                self.saved_paths = []
                self.set_weights_calls = []
                self.compiled = False

            def compile(self, **kwargs):
                self.compiled = True
                self.compile_kwargs = kwargs

            def fit(self, *_args, callbacks=None, epochs=None, **_kwargs):
                callbacks = callbacks or []
                for callback in callbacks:
                    if hasattr(callback, "set_model"):
                        callback.set_model(self)
                for epoch in range(epochs):
                    self.current_weights = [np.array([epoch + 1], dtype=float)]
                    for callback in callbacks:
                        if hasattr(callback, "on_epoch_end"):
                            callback.on_epoch_end(epoch, self.epoch_logs[epoch])

            def get_weights(self):
                return [weight.copy() for weight in self.current_weights]

            def set_weights(self, weights):
                self.set_weights_calls.append([weight.copy() for weight in weights])
                self.current_weights = [weight.copy() for weight in weights]

            def save(self, path):
                self.saved_paths.append(path)

        fake_model = FakeModel(
            [
                {"val_loss": 0.8},
                {"val_loss": 0.3},
                {"val_loss": 0.3},
                {"val_loss": 0.5},
            ]
        )
        automate_training = SimpleNamespace(
            LEARNING_RATE_SET=0.01,
            BATCH_SIZE=8,
            scheduler=lambda epoch, learning_rate=None: learning_rate,
            DROPOUT=0.5,
            LSTM=5,
            CONV=5,
            LAMBDA=0.0,
        )
        train_rows = [{"sequence": "AAAAA", "label": "1", "label_source": "experimental"}]
        validation_rows = [{"sequence": "CCCCC", "label": "0", "label_source": "experimental"}]
        config = RunConfig(run_name="train_model_perf_test", epochs=4)

        with workspace_tempdir("altest") as temp_dir, mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            output_path = Path(temp_dir) / "model.h5"
            with mock.patch(
                "active_learning_thesis.predictive._predictive_modules",
                return_value=(None, None, automate_training),
            ), mock.patch(
                "active_learning_thesis.predictive._try_load_reusable_model",
                return_value=None,
            ), mock.patch(
                "active_learning_thesis.predictive._prepare_training_tensors",
                side_effect=[("train_inputs", np.array([1.0])), ("validation_inputs", np.array([0.0]))],
            ), mock.patch(
                "active_learning_thesis.predictive._build_model",
                return_value=fake_model,
            ):
                managed = train_model(
                    "AP_SP",
                    train_rows,
                    validation_rows,
                    seed=11,
                    output_path=output_path,
                    config=config,
                    cache_dir=None,
                )

        self.assertTrue(fake_model.compiled)
        self.assertEqual(len(fake_model.saved_paths), 1)
        self.assertEqual(fake_model.saved_paths[0], str(output_path))
        self.assertEqual(load_model_mock.call_count, 0)
        self.assertEqual(len(fake_model.set_weights_calls), 1)
        np.testing.assert_allclose(
            fake_model.set_weights_calls[0][0],
            np.array([2.0], dtype=float),
        )
        np.testing.assert_allclose(
            managed.model.get_weights()[0],
            np.array([2.0], dtype=float),
        )

    def test_training_fingerprint_is_stable_across_row_order(self):
        config = RunConfig(run_name="fingerprint_test", epochs=1)
        train_rows_a = [
            {"sequence": "AAAAA", "label": "1", "label_source": "experimental"},
            {"sequence": "CCCCC", "label": "0", "label_source": "experimental"},
        ]
        train_rows_b = list(reversed(train_rows_a))
        validation_rows = [
            {"sequence": "GGGGG", "label": "1", "label_source": "experimental"},
        ]

        fingerprint_a = _training_fingerprint(
            "AP_SP",
            train_rows_a,
            validation_rows,
            seed=11,
            config=config,
        )
        fingerprint_b = _training_fingerprint(
            "AP_SP",
            train_rows_b,
            validation_rows,
            seed=11,
            config=config,
        )
        self.assertEqual(fingerprint_a, fingerprint_b)


if __name__ == "__main__":
    unittest.main()
