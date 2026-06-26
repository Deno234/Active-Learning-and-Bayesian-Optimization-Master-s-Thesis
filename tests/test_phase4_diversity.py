from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from active_learning_thesis.cli import _build_parser
from active_learning_thesis import phase4_diversity as p4d
from active_learning_thesis import generative
from active_learning_thesis.config import RunConfig


class Phase4DiversityTests(unittest.TestCase):
    def test_cli_uses_run_not_round(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "phase4-bo",
                "run-diversity-aware",
                "--phase4d-run",
                "1",
            ]
        )
        self.assertEqual(args.phase4d_run, 1)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "phase4-bo",
                    "run-diversity-aware",
                    "--phase4d-round",
                    "1",
                ]
            )

    def test_seed_blocks_are_zero_based_and_non_overlapping(self):
        p4d._validate_seed_blocks(dict(p4d.PHASE4D_DEFAULT_SEEDS))
        for left, right in zip(
            p4d.PHASE4_POLICIES,
            p4d.PHASE4_POLICIES[1:],
        ):
            left_seed = p4d.PHASE4D_DEFAULT_SEEDS[left]
            right_seed = p4d.PHASE4D_DEFAULT_SEEDS[right]
            self.assertEqual(left_seed + 99, right_seed - 1)
        overlapping = dict(p4d.PHASE4D_DEFAULT_SEEDS)
        overlapping["greedy"] = overlapping["random"] + 99
        with self.assertRaises(ValueError):
            p4d._validate_seed_blocks(overlapping)

    def test_exact_composition_similarity(self):
        self.assertAlmostEqual(
            p4d.phase4d_composition_similarity("AAAA", "AAAA"),
            0.1,
        )
        self.assertAlmostEqual(
            p4d.phase4d_composition_similarity("AAAA", "CCCC"),
            0.0,
        )
        self.assertAlmostEqual(
            p4d.phase4d_composition_similarity("AAAC", "AACC"),
            0.075,
        )

    def test_similarity_aware_selection_updates_reference_and_removes_selected(self):
        rows = [
            {"sequence": "AAAAA", "final_acquisition_utility": 0.90},
            {"sequence": "AAAAC", "final_acquisition_utility": 0.89},
            {"sequence": "CCCCC", "final_acquisition_utility": 0.84},
            {"sequence": "DDDDD", "final_acquisition_utility": 0.83},
            {"sequence": "EEEEE", "final_acquisition_utility": 0.82},
            {"sequence": "FFFFF", "final_acquisition_utility": 0.81},
        ]
        selected, trace = p4d.phase4d_similarity_aware_selection(rows, batch_size=5)
        self.assertEqual(selected[0], 0)
        self.assertEqual(len(selected), len(set(selected)))
        first = [
            row for row in trace
            if row["selection_step"] == 1 and row["selected_at_step"]
        ][0]
        self.assertEqual(first["phase4d_final_similarity_penalty"], 0.0)
        self.assertEqual(
            first["phase4d_selection_score"],
            first["original_final_acquisition_utility"],
        )
        for step in range(2, 6):
            candidates = [row for row in trace if row["selection_step"] == step]
            self.assertEqual(len(candidates), len(rows) - step + 1)

    def test_tie_break_prefers_utility_then_penalty_then_input_order(self):
        rows = [
            {"sequence": "AAAAA", "final_acquisition_utility": 1.0},
            {"sequence": "AAAAC", "final_acquisition_utility": 0.90},
            {"sequence": "CCCCC", "final_acquisition_utility": 0.80},
        ]
        def similarity(left, right, allowed_amino_acids):
            if left == "AAAAC":
                return 0.10
            if left == "CCCCC":
                return 0.0
            return 0.0

        with mock.patch.object(
            p4d,
            "phase4d_composition_similarity",
            side_effect=similarity,
        ):
            selected, _ = p4d.phase4d_similarity_aware_selection(
                rows, batch_size=2
            )
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[1], 1)

    def test_attempt_manifest_records_complete_multi_attempt_history(self):
        history = [
            {
                "attempt_index": 0,
                "attempt_seed": 100,
                "candidates_generated": 50,
                "novel_candidates_generated": 20,
                "candidates_accepted": 20,
                "retained_pool_size_after_attempt": 20,
            },
            {
                "attempt_index": 1,
                "attempt_seed": 101,
                "candidates_generated": 50,
                "novel_candidates_generated": 30,
                "candidates_accepted": 30,
                "retained_pool_size_after_attempt": 50,
            },
        ]
        manifest = p4d._attempt_manifest(
            "greedy", 100, history, 50, p4d._now_iso(), 0.0
        )
        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(manifest["attempt_indices_used"], [0, 1])
        self.assertEqual(manifest["attempt_seeds_used"], [100, 101])
        self.assertNotIn("successful_generation_seed", manifest)
        single = p4d._attempt_manifest(
            "greedy", 100, [dict(history[0], candidates_accepted=50)], 50,
            p4d._now_iso(), 0.0
        )
        self.assertEqual(single["successful_generation_seed"], 100)

    def test_shared_generator_records_zero_based_attempt_history(self):
        class FakeGA:
            class Peptide:
                def __init__(self, sequence):
                    self.sequence = sequence

            def __init__(self, **kwargs):
                self.event_callback = kwargs.get("event_callback")

            def find_peptides(self):
                peptides = [self.Peptide("AAAAA"), self.Peptide("CCCCC")]
                if self.event_callback:
                    for peptide in peptides:
                        self.event_callback(
                            {"event": "created", "sequence": peptide.sequence}
                        )
                return peptides

        history = []
        config = RunConfig(
            random_seed=500,
            candidate_pool_min=2,
            ga_max_attempts=3,
            population_size=2,
            offspring_count=0,
            max_num_generations=0,
        )
        with mock.patch.object(
            generative,
            "load_genetic_algorithm_class",
            return_value=FakeGA,
        ):
            sequences, _ = generative.generate_candidate_sequences(
                None,
                set(),
                config,
                min_unique=2,
                minimum_return_count=2,
                objective="greedy",
                return_metadata=True,
                policy_utility_callback=lambda values: np.asarray([0.2, 0.8]),
                attempt_history=history,
            )
        self.assertEqual(sequences, ["CCCCC", "AAAAA"])
        self.assertEqual(history[0]["attempt_index"], 0)
        self.assertEqual(history[0]["attempt_seed"], 500)
        self.assertEqual(history[0]["candidates_generated"], 2)
        self.assertEqual(history[0]["candidates_accepted"], 2)

    def test_random_validation_accepts_null_utility(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
            pool = [
                {
                    "sequence": (
                        "AA"
                        + alphabet[(index // 20) % 20]
                        + alphabet[index % 20]
                        + "A"
                    ),
                    "final_acquisition_utility": None,
                    "final_acquisition_utility_applicable": False,
                    "phase4d_similarity_aware_applicable": False,
                }
                for index in range(50)
            ]
            self._write_csv(policy_dir / "new_candidate_pool.csv", pool)
            self._write_csv(policy_dir / "random_selected_batch.csv", pool[:5])
            (policy_dir / "status.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "policy_base_seed": 20270417,
                    }
                ),
                encoding="utf-8",
            )
            (policy_dir / "checksums.json").write_text(
                json.dumps(p4d._directory_checksums(policy_dir)),
                encoding="utf-8",
            )
            config = {
                "candidate_pool_target": 50,
                "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY",
            }
            p4d._validate_policy_directory(
                policy_dir, "random", 20270417, config
            )

    def test_primary_artifact_checksums_detect_change_and_ignore_phase4d(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "config.json"
            primary.write_text("original", encoding="utf-8")
            before = p4d._primary_artifact_manifest(root, p4d._now_iso())
            phase4d_file = root / "phase4d" / "run_001" / "result.csv"
            phase4d_file.parent.mkdir(parents=True)
            phase4d_file.write_text("allowed", encoding="utf-8")
            verified = p4d._verify_primary_artifacts_unchanged(root, before)
            self.assertEqual(verified["verification"], "passed")
            primary.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                p4d._verify_primary_artifacts_unchanged(root, before)

    def test_primary_artifact_checksums_reject_unclassified_addition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("original", encoding="utf-8")
            before = p4d._primary_artifact_manifest(root, p4d._now_iso())
            (root / "new_scientific.csv").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                p4d._verify_primary_artifacts_unchanged(root, before)

    def test_runtime_walltime_uses_measured_logs_and_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs" / "supek_runtime"
            logs.mkdir(parents=True)
            start = datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc)
            end = start + timedelta(minutes=10)
            for policy in p4d.PHASE4_POLICIES:
                (logs / f"p4_{policy[:7]}.out").write_text(
                    f"[phase4] start {start.isoformat()} host=node\n"
                    f"[phase4] end {end.isoformat()} exit_status=0\n",
                    encoding="utf-8",
                )
            timing = p4d._derive_phase4d_walltime(root, None)
            self.assertEqual(timing["measured_seconds"], 3600)
            self.assertEqual(timing["requested_seconds"], 9000)
            self.assertEqual(timing["pbs_walltime"], "03:00:00")

    def test_pbs_preview_runs_sequential_command_without_qsub(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            output = repo / "thesis_results" / "04_bayesian_optimization"
            output.mkdir(parents=True)
            (output / "config.json").write_text(
                json.dumps(
                    {
                        "supek": {
                            "queue": "gpu",
                            "ncpus": 4,
                            "ngpus": 1,
                            "mem": "40GB",
                        }
                    }
                ),
                encoding="utf-8",
            )
            path = p4d._write_phase4d_pbs(
                output,
                1,
                str(repo),
                "03:00:00",
                None,
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("phase4-bo run-diversity-aware", text)
            self.assertIn("--phase4d-run 1", text)
            self.assertNotIn("qsub", text)

    def test_penalty_scale_audit_reports_penalty_exceeding_utility(self):
        utility_rows = [
            {
                "sequence": sequence,
                "original_final_acquisition_utility": utility,
                "original_final_utility_rank": index + 1,
                "phase4d_final_similarity_penalty": 0.0,
                "selection_step": index + 1,
            }
            for index, (sequence, utility) in enumerate(
                zip(("AAAAA", "CCCCC", "DDDDD", "EEEEE", "FFFFF"),
                    (0.05, 0.04, 0.03, 0.02, 0.01))
            )
        ]
        diversity_rows = [dict(row) for row in utility_rows]
        for index, row in enumerate(diversity_rows):
            row["phase4d_final_similarity_penalty"] = 0.0 if index == 0 else 0.08
        trace = [
            {
                "original_final_acquisition_utility": 0.01,
                "phase4d_final_similarity_penalty": 0.08,
            },
            {
                "original_final_acquisition_utility": 0.05,
                "phase4d_final_similarity_penalty": 0.0,
            },
        ]
        comparison = p4d._guided_policy_comparison(
            "pi", utility_rows, diversity_rows, trace
        )
        self.assertEqual(comparison["penalty_exceeds_utility_count"], 1)
        self.assertEqual(comparison["first_step_similarity_penalty"], 0.0)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
