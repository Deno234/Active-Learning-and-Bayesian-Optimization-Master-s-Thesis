from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from active_learning_thesis.cli import _build_parser
from active_learning_thesis.config import RunConfig
from active_learning_thesis.ledger import empty_row, load_ledger, save_ledger
from active_learning_thesis import phase3_real_al as p3


class Phase3RealALTests(unittest.TestCase):
    def _phase1_root(self, root: Path) -> Path:
        phase1 = root / "01_reproduction"
        phase1.mkdir(parents=True)
        frozen = {
            "AP": {"num_cells": 32, "kernel_size": "n/a"},
            "SP": {"num_cells": 64, "kernel_size": "4"},
            "AP_SP": {"num_cells": 48, "kernel_size": "8"},
            "TSNE_SP": {"num_cells": 48, "kernel_size": "6"},
            "TSNE_AP_SP": {"num_cells": 64, "kernel_size": "6"},
        }
        (phase1 / "frozen_model_config.json").write_text(json.dumps(frozen), encoding="utf-8")
        return phase1

    def _phase2_root(self, root: Path) -> Path:
        phase2 = root / "02_replay"
        phase2.mkdir(parents=True)
        (phase2 / "marker.txt").write_text("phase2", encoding="utf-8")
        return phase2

    def _hash_tree(self, root: Path) -> dict[str, str]:
        hashes = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def _fake_init_run(self, config: RunConfig, *, train_baseline: bool = True) -> Path:
        run_dir = config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save(run_dir / "config.json")
        rows = [
            empty_row(
                {
                    "sequence": "AAAAA",
                    "label": "1",
                    "label_source": "experimental",
                    "split": "train_pool",
                    "mode": "experimental",
                    "round_id": "0",
                    "status": "train_pool",
                }
            ),
            empty_row(
                {
                    "sequence": "CCCCC",
                    "label": "0",
                    "label_source": "experimental",
                    "split": "validation",
                    "mode": "experimental",
                    "round_id": "0",
                    "status": "validation",
                }
            ),
        ]
        save_ledger(run_dir / "ledger.csv", rows)
        return run_dir

    def _fake_propose_round(self, branch_dir: Path, branch: str) -> Path:
        (branch_dir / "batches").mkdir(exist_ok=True)
        (branch_dir / "candidates").mkdir(exist_ok=True)
        round_id = p3.next_real_round_id(load_ledger(branch_dir / "ledger.csv"))
        batch = branch_dir / "batches" / f"round_{round_id:03d}_batch.csv"
        with batch.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sequence", "round_id", "acquisition_strategy", "acquisition_score"],
            )
            writer.writeheader()
            for index in range(5):
                writer.writerow(
                    {
                        "sequence": f"CAND{round_id}{index}",
                        "round_id": str(round_id),
                        "acquisition_strategy": branch,
                        "acquisition_score": str(1.0 - index * 0.1),
                    }
                )
        candidate_rows = []
        with (branch_dir / "candidates" / f"round_{round_id:03d}_scored.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence", "round_id", "acquisition_score"])
            writer.writeheader()
            for index in range(8):
                scored = {"sequence": f"CAND{round_id}{index}", "round_id": str(round_id), "acquisition_score": str(index)}
                writer.writerow(scored)
                candidate_rows.append(
                    empty_row(
                        {
                            **scored,
                            "split": "generated",
                            "mode": "real_al",
                            "status": "proposed" if index < 5 else "candidate_scored",
                            "acquisition_strategy": branch,
                            "selection_rank": str(index + 1) if index < 5 else "",
                        }
                    )
                )
        ledger_rows = load_ledger(branch_dir / "ledger.csv")
        ledger_rows.extend(candidate_rows)
        save_ledger(branch_dir / "ledger.csv", ledger_rows)
        return batch

    def _seed_phase3_selected_rows(self, output_root: Path, branch: str, sequences: list[str], *, round_id: int = 1) -> Path:
        branch_dir = output_root / "branches" / branch
        ledger_path = branch_dir / "ledger.csv"
        rows = load_ledger(ledger_path)
        selected_rows = []
        for index, sequence in enumerate(sequences, start=1):
            row = empty_row(
                {
                    "sequence": sequence,
                    "label": "",
                    "label_source": "",
                    "split": "generated",
                    "mode": "real_al",
                    "round_id": str(round_id),
                    "status": "proposed",
                    "acquisition_strategy": branch,
                    "acquisition_score": str(1.0 - index * 0.1),
                    "selection_rank": str(index),
                    "candidate_source": "phase3_test_fixture",
                    "generator_origin": "phase3_test_fixture",
                    "replay_role": "phase3_selected",
                }
            )
            rows.append(row)
            selected_rows.append(
                {
                    "sequence": sequence,
                    "round_id": "1",
                    "acquisition_strategy": branch,
                    "acquisition_score": row["acquisition_score"],
                    "selection_rank": row["selection_rank"],
                    "campaign_name": f"phase3_{branch}_r{round_id:03d}_{sequence}",
                    "campaign_dir": str(branch_dir / "rounds" / f"round_{round_id:03d}" / "md_campaigns" / f"phase3_{branch}_r{round_id:03d}_{sequence}"),
                }
            )
        save_ledger(ledger_path, rows)
        round_dir = branch_dir / "rounds" / f"round_{round_id:03d}"
        p3._write_csv(round_dir / "selected_batch.csv", selected_rows)
        p3._write_csv(round_dir / "proposal.csv", selected_rows)
        p3._write_current_labeled_ledger(branch_dir)
        return round_dir / "selected_batch.csv"

    def _write_phase3_review(self, output_root: Path, branch: str, rows: list[dict[str, str]], *, round_id: int = 1) -> Path:
        review_path = output_root / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "review" / "md_review.csv"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sequence",
            "cgmd_label",
            "label_rubric",
            "label_confidence",
            "label_evidence_summary",
            "review_notes",
            "reviewer",
            "reviewed_at",
            "campaign_name",
            "campaign_dir",
        ]
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload = {field: "" for field in fieldnames}
                payload.update(row)
                writer.writerow(payload)
        return review_path

    def _fully_ingest_round(self, output_root: Path, branch: str, sequences: list[str], *, round_id: int = 1) -> None:
        self._write_phase3_review(
            output_root,
            branch,
            [
                {
                    "sequence": sequence,
                    "cgmd_label": "1" if index % 2 == 0 else "0",
                    "label_rubric": "self_assembling" if index % 2 == 0 else "not_self_assembling",
                    "label_confidence": "high",
                    "label_evidence_summary": "Human-reviewed CG-MD evidence summary.",
                    "review_notes": "Ready for branch-local ingestion.",
                    "reviewer": "tester",
                }
                for index, sequence in enumerate(sequences)
            ],
            round_id=round_id,
        )
        p3.make_phase3_ingest_csv(
            type("Args", (), {"output_root": str(output_root), "branch": branch, "round": round_id, "force": False})()
        )
        import_csv = output_root / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "ingest" / "cgmd_ingest.csv"
        p3.ingest_phase3_labels(
            type(
                "Args",
                (),
                {
                    "output_root": str(output_root),
                    "branch": branch,
                    "round": round_id,
                    "import_csv": str(import_csv),
                    "dry_run": False,
                    "force": False,
                },
            )()
        )

    def _args(self, root: Path, **overrides):
        defaults = {
            "phase3_real_al_action": "init",
            "phase1_root": str(self._phase1_root(root)),
            "phase2_root": str(self._phase2_root(root)),
            "output_root": str(root / "03_real_al"),
            "strategies": list(p3.DEFAULT_STRATEGIES),
            "backup_strategy": "ensemble_mi",
            "force": False,
            "random_seed": 20260317,
            "replay_seed_size": 40,
            "batch_size": 5,
            "max_rounds": 10,
            "candidate_pool_min": 50,
            "ga_max_attempts": 100,
            "ensemble_size": 5,
            "epochs": 70,
            "raw_acquisition": False,
            "generator_objective_mode": "match_acquisition",
            "use_similarity_penalty": True,
            "no_length_penalty": False,
            "binary_threshold_strategy": "pr_best_f1",
            "pbs_repo_root": None,
            "supek_walltime": None,
            "supek_queue": None,
            "supek_ncpus": None,
            "supek_ngpus": None,
            "supek_mem": None,
        }
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_cli_parses_phase3_real_al_subcommands(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "phase3-real-al",
                "init",
                "--phase1-root",
                "thesis_results/01_reproduction",
                "--phase2-root",
                "thesis_results/02_replay",
                "--output-root",
                "thesis_results/03_real_al",
                "--strategies",
                "predictive_entropy",
                "family_qbc",
                "cluster_diverse_representative",
                "--supek-walltime",
                "06:00:00",
            ]
        )
        self.assertEqual(args.command, "phase3-real-al")
        self.assertEqual(args.phase3_real_al_action, "init")
        self.assertEqual(args.supek_walltime, "06:00:00")

    def test_init_creates_isolated_branches_inventory_and_preserves_phase_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            phase1_hashes = self._hash_tree(Path(args.phase1_root))
            phase2_hashes = self._hash_tree(Path(args.phase2_root))
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run) as init_mock:
                summary = p3.init_phase3_real_al(args)
            self.assertEqual(summary["status"], "initialized")
            self.assertEqual(init_mock.call_count, 3)
            output_root = Path(args.output_root)
            self.assertTrue((output_root / "phase3_real_al_manifest.json").exists())
            self.assertTrue((output_root / "md_inventory" / "md_inventory_events.csv").exists())
            for strategy in p3.DEFAULT_STRATEGIES:
                branch = output_root / "branches" / strategy
                self.assertTrue((branch / "config.json").exists())
                config = json.loads((branch / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["candidate_pool_min"], 50)
                self.assertEqual(config["ga_max_attempts"], 100)
                self.assertTrue((branch / "ledger.csv").exists())
                self.assertTrue((branch / "current_labeled_ledger.csv").exists())
                self.assertTrue((branch / "rounds" / "round_001" / "command_preview.txt").exists())
            self.assertEqual(phase1_hashes, self._hash_tree(Path(args.phase1_root)))
            self.assertEqual(phase2_hashes, self._hash_tree(Path(args.phase2_root)))

    def test_existing_branch_requires_force_and_force_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
                with self.assertRaises(FileExistsError):
                    p3.init_phase3_real_al(args)
                args.force = True
                p3.init_phase3_real_al(args)
            archives = list((Path(args.output_root) / "logs" / "archived_branches").glob("predictive_entropy_*"))
            self.assertEqual(len(archives), 1)

    def test_supek_previews_include_resources_no_set_u_and_no_automatic_qsub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            args = self._args(
                root,
                strategies=["predictive_entropy"],
                pbs_repo_root=str(repo_root),
                supek_walltime="05:00:00",
                supek_queue="gpu_short",
                supek_ncpus=2,
                supek_ngpus=1,
                supek_mem="24GB",
            )
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            pbs = Path(args.output_root) / "logs" / "supek_pbs" / "supek_phase3_propose_predictive_entropy_r001.pbs"
            text = pbs.read_text(encoding="utf-8")
            self.assertIn("#PBS -q gpu_short", text)
            self.assertIn("select=1:ncpus=2:ngpus=1:mem=24GB", text)
            self.assertIn("#PBS -l walltime=05:00:00", text)
            self.assertIn("set -eo pipefail", text)
            self.assertNotIn("set -euo pipefail", text)
            self.assertNotIn("qsub ", text)
            self.assertIn("python -m active_learning_thesis phase3-real-al propose", text)
            self.assertIn("--branch predictive_entropy --round 1", text)
            self.assertNotIn("--dry-run", text)
            preview = Path(args.output_root) / "branches" / "predictive_entropy" / "rounds" / "round_001" / "command_preview.txt"
            preview_text = preview.read_text(encoding="utf-8")
            self.assertIn("Queue: gpu_short", preview_text)
            self.assertIn("Walltime: 05:00:00", preview_text)
            status = json.loads((preview.parent / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "preview_ready")
            self.assertIn("phase3-real-al propose", status["exact_command"])

    def test_per_branch_supek_config_overrides_are_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["family_qbc"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            config_path = Path(args.output_root) / "branches" / "family_qbc" / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["supek_walltime"] = "18:00:00"
            config["supek_mem"] = "64GB"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            p3.write_supek_proposal_preview(Path(args.output_root), "family_qbc", 1, args=None)
            text = (Path(args.output_root) / "logs" / "supek_pbs" / "supek_phase3_propose_family_qbc_r001.pbs").read_text(encoding="utf-8")
            self.assertIn("#PBS -l walltime=18:00:00", text)
            self.assertIn("mem=64GB", text)

    def test_dry_run_preview_does_not_call_real_proposal_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            propose_args = type(
                "Args",
                (),
                {
                    "output_root": args.output_root,
                    "branch": "predictive_entropy",
                    "round": 1,
                    "dry_run": True,
                    "write_supek_pbs": False,
                    "pbs_repo_root": None,
                    "supek_walltime": None,
                    "supek_queue": None,
                    "supek_ncpus": None,
                    "supek_ngpus": None,
                    "supek_mem": None,
                },
            )()
            with mock.patch.object(p3, "propose_round", side_effect=AssertionError("should not run")):
                result = p3.propose_phase3_real_al(propose_args)
            self.assertEqual(result["status"], "preview-written")

    def test_real_proposal_path_calls_engine_and_writes_round_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            propose_args = type(
                "Args",
                (),
                {
                    "output_root": args.output_root,
                    "branch": "predictive_entropy",
                    "round": 1,
                    "dry_run": False,
                    "write_supek_pbs": False,
                },
            )()
            with mock.patch.object(p3, "propose_round", side_effect=self._fake_propose_round) as propose_mock:
                result = p3.propose_phase3_real_al(propose_args)
            self.assertEqual(result["status"], "proposal-complete")
            propose_mock.assert_called_once()
            round_dir = Path(args.output_root) / "branches" / "predictive_entropy" / "rounds" / "round_001"
            for name in ["proposal.csv", "scored_candidates.csv", "selected_batch.csv", "acquisition_log.csv", "config.json", "status.json"]:
                self.assertTrue((round_dir / name).exists(), name)
            status = json.loads((round_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["branch_strategy"], "predictive_entropy")
            self.assertEqual(status["round_id"], "round_001")
            self.assertIn("phase3-real-al propose", status["exact_command"])
            self.assertTrue(status["phase1_frozen_config_path"].endswith("frozen_model_config.json"))
            self.assertTrue(status["branch_config_path"].endswith("branches\\predictive_entropy\\config.json") or status["branch_config_path"].endswith("branches/predictive_entropy/config.json"))
            self.assertTrue(status["current_labeled_ledger_path"].endswith("current_labeled_ledger.csv"))
            self.assertTrue(status["selected_batch_path"].endswith("selected_batch.csv"))
            self.assertTrue(status["scored_candidates_path"].endswith("scored_candidates.csv"))
            self.assertEqual(status["candidate_count"], 8)
            self.assertEqual(status["selected_count"], 5)
            self.assertEqual(p3.detect_round_status(Path(args.output_root), "predictive_entropy", 1)["status"], "completed")
            config = json.loads((round_dir / "config.json").read_text(encoding="utf-8"))
            self.assertIn("phase3_round_metadata", config)
            self.assertEqual(config["phase3_round_metadata"]["selected_count"], 5)

    def test_failed_proposal_writes_failed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            propose_args = type(
                "Args",
                (),
                {
                    "output_root": args.output_root,
                    "branch": "predictive_entropy",
                    "round": 1,
                    "dry_run": False,
                    "write_supek_pbs": False,
                },
            )()
            with mock.patch.object(p3, "propose_round", side_effect=RuntimeError("training failed")):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    p3.propose_phase3_real_al(propose_args)
            status_path = Path(args.output_root) / "branches" / "predictive_entropy" / "rounds" / "round_001" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertIn("training failed", status["error"])
            self.assertEqual(p3.detect_round_status(Path(args.output_root), "predictive_entropy", 1)["status"], "failed")

    def test_proposal_mirroring_preserves_duplicates_and_inventory_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy", "family_qbc"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            for branch in ["predictive_entropy", "family_qbc"]:
                branch_dir = output_root / "branches" / branch
                (branch_dir / "batches").mkdir(exist_ok=True)
                (branch_dir / "candidates").mkdir(exist_ok=True)
                batch = branch_dir / "batches" / "round_001_batch.csv"
                with batch.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["sequence", "round_id", "acquisition_strategy", "acquisition_score"])
                    writer.writeheader()
                    writer.writerow({"sequence": "DUPSEQ", "round_id": "1", "acquisition_strategy": branch, "acquisition_score": "0.9"})
                (branch_dir / "candidates" / "round_001_scored.csv").write_text(
                    "sequence,acquisition_score\nDUPSEQ,0.9\n",
                    encoding="utf-8",
                )
                p3.mirror_round_outputs(output_root, branch, 1, batch)
            first_selected = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "selected_batch.csv"
            second_selected = output_root / "branches" / "family_qbc" / "rounds" / "round_001" / "selected_batch.csv"
            self.assertIn("DUPSEQ", first_selected.read_text(encoding="utf-8"))
            with second_selected.open("r", encoding="utf-8") as handle:
                second_rows = list(csv.DictReader(handle))
            self.assertIn("already present", second_rows[0]["duplicate_md_warning"])
            with (output_root / "md_inventory" / "md_inventory_events.csv").open("r", encoding="utf-8") as handle:
                events = list(csv.DictReader(handle))
            self.assertEqual(len([row for row in events if row["sequence"] == "DUPSEQ"]), 2)
            with (output_root / "md_inventory" / "md_inventory.csv").open("r", encoding="utf-8") as handle:
                snapshot = list(csv.DictReader(handle))
            self.assertIn("predictive_entropy:round_001", snapshot[0]["selected_by_branches"])
            self.assertIn("family_qbc:round_001", snapshot[0]["selected_by_branches"])
            p3.compare_phase3_real_al(type("Args", (), {"output_root": args.output_root})())
            duplicates = (output_root / "comparison" / "duplicate_sequences_across_branches.csv").read_text(encoding="utf-8")
            self.assertIn("DUPSEQ", duplicates)

    def test_compare_and_status_are_read_only_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            summary = p3.compare_phase3_real_al(type("Args", (), {"output_root": args.output_root})())
            self.assertEqual(summary["status"], "comparison-written")
            comparison = Path(args.output_root) / "comparison"
            for name in [
                "branch_proposal_overlap.csv",
                "branch_selected_sequences.csv",
                "branch_md_status_summary.csv",
                "duplicate_sequences_across_branches.csv",
                "branch_label_summary.csv",
                "branch_round_metrics.csv",
                "branch_comparison.md",
            ]:
                self.assertTrue((comparison / name).exists())
            status = p3.status_phase3_real_al(type("Args", (), {"output_root": args.output_root})())
            self.assertEqual(status["status"], "ok")

    def test_make_ingest_csv_includes_only_ready_rows_and_reports_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["READYSEQ", "BLOCKSEQ"])
            self._write_phase3_review(
                output_root,
                "predictive_entropy",
                [
                    {
                        "sequence": "READYSEQ",
                        "cgmd_label": "1",
                        "label_rubric": "self_assembling",
                        "label_confidence": "high",
                        "label_evidence_summary": "AP/SASA/contact evidence supports assembly.",
                        "review_notes": "Human reviewed trajectory.",
                        "reviewer": "tester",
                    },
                    {
                        "sequence": "BLOCKSEQ",
                        "cgmd_label": "0",
                        "label_rubric": "not_self_assembling",
                        "label_confidence": "",
                        "label_evidence_summary": "",
                        "review_notes": "",
                        "reviewer": "tester",
                    },
                ],
            )
            result = p3.make_phase3_ingest_csv(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "force": False})())
            self.assertEqual(result["ingest_ready_count"], 1)
            self.assertEqual(result["blocked_rows_count"], 1)
            with (output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "ingest" / "cgmd_ingest.csv").open(encoding="utf-8") as handle:
                ingest_rows = list(csv.DictReader(handle))
            self.assertEqual([row["sequence"] for row in ingest_rows], ["READYSEQ"])
            preview = json.loads((output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "ingest" / "ingest_preview.json").read_text(encoding="utf-8"))
            self.assertIn("label_confidence", preview["blockers"][0]["missing"])

    def test_phase3_ingest_updates_only_selected_branch_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy", "family_qbc"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["SAMESEQ", "ONLYPE"])
            self._seed_phase3_selected_rows(output_root, "family_qbc", ["SAMESEQ"])
            other_before = (output_root / "branches" / "family_qbc" / "ledger.csv").read_bytes()
            self._write_phase3_review(
                output_root,
                "predictive_entropy",
                [
                    {
                        "sequence": "SAMESEQ",
                        "cgmd_label": "1",
                        "label_rubric": "self_assembling",
                        "label_confidence": "medium",
                        "label_evidence_summary": "Evidence summary.",
                        "review_notes": "Reviewed for PE only.",
                        "reviewer": "tester",
                    }
                ],
            )
            p3.make_phase3_ingest_csv(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "force": False})())
            import_csv = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "ingest" / "cgmd_ingest.csv"
            result = p3.ingest_phase3_labels(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "import_csv": str(import_csv), "dry_run": False, "force": False})())
            self.assertEqual(result["status"], "partially_ingested")
            pe_rows = {row["sequence"]: row for row in load_ledger(output_root / "branches" / "predictive_entropy" / "ledger.csv")}
            self.assertEqual(pe_rows["SAMESEQ"]["label"], "1")
            self.assertEqual(pe_rows["SAMESEQ"]["label_source"], "cgmd")
            self.assertEqual(pe_rows["SAMESEQ"]["status"], "acquired")
            self.assertEqual(pe_rows["SAMESEQ"]["acquisition_score"], "0.9")
            self.assertEqual(pe_rows["SAMESEQ"]["selection_rank"], "1")
            self.assertEqual(other_before, (output_root / "branches" / "family_qbc" / "ledger.csv").read_bytes())
            fqbc_rows = {row["sequence"]: row for row in load_ledger(output_root / "branches" / "family_qbc" / "ledger.csv")}
            self.assertEqual(fqbc_rows["SAMESEQ"]["label"], "")
            status = json.loads((output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "partially_ingested")
            self.assertTrue(status["next_proposal_blocked"])
            inventory = (output_root / "md_inventory" / "md_inventory.csv").read_text(encoding="utf-8")
            self.assertIn("predictive_entropy:round_001", inventory)
            self.assertNotIn("family_qbc:round_001", inventory.split("ingested_branches", 1)[-1])

    def test_phase3_ingest_dry_run_and_repeat_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["DRYSEQ"])
            self._write_phase3_review(
                output_root,
                "predictive_entropy",
                [
                    {
                        "sequence": "DRYSEQ",
                        "cgmd_label": "0",
                        "label_rubric": "not_self_assembling",
                        "label_confidence": "high",
                        "label_evidence_summary": "Evidence summary.",
                        "review_notes": "Reviewed trajectory.",
                        "reviewer": "tester",
                    }
                ],
            )
            p3.make_phase3_ingest_csv(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "force": False})())
            import_csv = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "ingest" / "cgmd_ingest.csv"
            ledger_before = (output_root / "branches" / "predictive_entropy" / "ledger.csv").read_bytes()
            dry = p3.ingest_phase3_labels(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "import_csv": str(import_csv), "dry_run": True, "force": False})())
            self.assertEqual(dry["status"], "dry-run")
            self.assertEqual(ledger_before, (output_root / "branches" / "predictive_entropy" / "ledger.csv").read_bytes())
            p3.ingest_phase3_labels(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "import_csv": str(import_csv), "dry_run": False, "force": False})())
            with self.assertRaisesRegex(ValueError, "already ingested|already acquired"):
                p3.ingest_phase3_labels(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "import_csv": str(import_csv), "dry_run": False, "force": False})())

    def test_round2_proposal_blocks_until_round1_selected_batch_is_fully_ingested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            propose_args = type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "dry_run": False, "write_supek_pbs": False})()
            with mock.patch.object(p3, "propose_round", side_effect=self._fake_propose_round):
                p3.propose_phase3_real_al(propose_args)
            blocked_args = type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 2, "dry_run": False, "write_supek_pbs": False})()
            with mock.patch.object(p3, "propose_round", side_effect=AssertionError("round 2 should be blocked")):
                with self.assertRaisesRegex(ValueError, "unresolved proposed|not fully ingested|not ingested"):
                    p3.propose_phase3_real_al(blocked_args)

    def test_round2_proposal_succeeds_after_full_ingest_and_ignores_scored_only_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy", "family_qbc"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            round1_args = type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "dry_run": False, "write_supek_pbs": False})()
            with mock.patch.object(p3, "propose_round", side_effect=self._fake_propose_round):
                p3.propose_phase3_real_al(round1_args)
            scored_only = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "scored_candidates.csv"
            self.assertIn("CAND17", scored_only.read_text(encoding="utf-8"))
            self._fully_ingest_round(output_root, "predictive_entropy", [f"CAND1{index}" for index in range(5)], round_id=1)
            other_before = (output_root / "branches" / "family_qbc" / "ledger.csv").read_bytes()
            training_counts: list[int] = []

            def fake_round2(branch_dir: Path, branch: str) -> Path:
                training_counts.append(len(p3.current_real_training_rows(load_ledger(branch_dir / "ledger.csv"))))
                return self._fake_propose_round(branch_dir, branch)

            round2_args = type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 2, "dry_run": False, "write_supek_pbs": False})()
            with mock.patch.object(p3, "propose_round", side_effect=fake_round2) as propose_mock:
                result = p3.propose_phase3_real_al(round2_args)
            self.assertEqual(result["status"], "proposal-complete")
            propose_mock.assert_called_once()
            self.assertGreaterEqual(training_counts[0], 6)
            round2_dir = output_root / "branches" / "predictive_entropy" / "rounds" / "round_002"
            self.assertTrue((round2_dir / "selected_batch.csv").exists())
            with (round2_dir / "selected_batch.csv").open("r", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle))
            self.assertTrue(selected)
            self.assertEqual({row["round_id"] for row in selected}, {"2"})
            self.assertEqual(other_before, (output_root / "branches" / "family_qbc" / "ledger.csv").read_bytes())
            pbs = p3.write_supek_proposal_preview(output_root, "predictive_entropy", 2, args=None)
            self.assertIn("r002", pbs.name)
            self.assertIn("--round 2", pbs.read_text(encoding="utf-8"))
            p3.compare_phase3_real_al(type("Args", (), {"output_root": args.output_root, "round": 2})())
            self.assertTrue((output_root / "comparison" / "round_002" / "branch_selected_sequences.csv").exists())
            self.assertTrue((output_root / "comparison" / "all_rounds_branch_summary.csv").exists())

    def test_partial_ingestion_blocks_next_round_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["PARTA", "PARTB"])
            self._write_phase3_review(
                output_root,
                "predictive_entropy",
                [
                    {
                        "sequence": "PARTA",
                        "cgmd_label": "1",
                        "label_rubric": "self_assembling",
                        "label_confidence": "high",
                        "label_evidence_summary": "Evidence summary.",
                        "review_notes": "Reviewed.",
                        "reviewer": "tester",
                    }
                ],
            )
            p3.make_phase3_ingest_csv(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "force": False})())
            import_csv = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "ingest" / "cgmd_ingest.csv"
            p3.ingest_phase3_labels(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 1, "import_csv": str(import_csv), "dry_run": False, "force": False})())
            with mock.patch.object(p3, "propose_round", side_effect=AssertionError("partial should block")):
                with self.assertRaisesRegex(ValueError, "unresolved proposed|not fully ingested|partially_ingested"):
                    p3.propose_phase3_real_al(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 2, "dry_run": False, "write_supek_pbs": False})())

    def test_round2_make_ingest_csv_uses_selected_batch_not_scored_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["SEL2"], round_id=2)
            p3._write_csv(
                output_root / "branches" / "predictive_entropy" / "rounds" / "round_002" / "scored_candidates.csv",
                [
                    {"sequence": "SEL2", "round_id": "2", "acquisition_score": "1"},
                    {"sequence": "SCORED_ONLY", "round_id": "2", "acquisition_score": "0.5"},
                ],
            )
            self._write_phase3_review(
                output_root,
                "predictive_entropy",
                [
                    {
                        "sequence": "SEL2",
                        "cgmd_label": "1",
                        "label_rubric": "self_assembling",
                        "label_confidence": "high",
                        "label_evidence_summary": "Evidence summary.",
                        "review_notes": "Reviewed selected row.",
                        "reviewer": "tester",
                    },
                    {
                        "sequence": "SCORED_ONLY",
                        "cgmd_label": "1",
                        "label_rubric": "self_assembling",
                        "label_confidence": "high",
                        "label_evidence_summary": "Evidence summary.",
                        "review_notes": "Should be ignored because it was not selected.",
                        "reviewer": "tester",
                    },
                ],
                round_id=2,
            )
            p3.make_phase3_ingest_csv(type("Args", (), {"output_root": args.output_root, "branch": "predictive_entropy", "round": 2, "force": False})())
            with (output_root / "branches" / "predictive_entropy" / "rounds" / "round_002" / "ingest" / "cgmd_ingest.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sequence"] for row in rows], ["SEL2"])

    def test_dashboard_discovers_phase3_branches_from_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            from active_learning_thesis.dashboard import collect_dashboard_state

            state = collect_dashboard_state(Path(args.output_root))
            self.assertEqual(len(state["runs"]), 1)
            self.assertEqual(state["runs"][0]["branch_strategy"], "predictive_entropy")
            self.assertIn("Phase 3 branch", state["runs"][0]["run_display_name"])
            self.assertEqual(state["runs"][0]["phase3_round_status"]["status"], "preview_ready")
            self.assertIn("phase3_ingest_status", state["runs"][0])
            self.assertEqual(state["runs"][0]["phase3_ingest_status"]["branch_strategy"], "predictive_entropy")

    def test_cli_parses_phase3_finalize(self):
        parser = _build_parser()
        args = parser.parse_args([
            "phase3-real-al", "finalize", "--output-root", "thesis_results/03_real_al",
            "--branch", "predictive_entropy", "--round", "8",
            "--evaluate-holdout", "--write-supek-pbs",
        ])
        self.assertEqual(args.phase3_real_al_action, "finalize")
        self.assertEqual(args.round, 8)
        self.assertTrue(args.evaluate_holdout)
        self.assertTrue(args.write_supek_pbs)

    def test_finalize_retrains_fully_ingested_round_and_preserves_ingest_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["FINALSEQ"])
            self._fully_ingest_round(output_root, "predictive_entropy", ["FINALSEQ"])
            finalize_args = type("Args", (), {
                "output_root": args.output_root, "branch": "predictive_entropy", "round": 1,
                "evaluate_holdout": True, "dry_run": False,
                "write_supek_pbs": False, "force": False,
            })()
            validation = {"round_id": 1, "labeled_count": 2, "f1": 0.8, "stage": "post_ingest"}
            holdout = {"round_id": 1, "labeled_count": 2, "f1": 0.75, "evaluation_dataset": "holdout"}
            with mock.patch.object(p3, "retrain_after_ingest", return_value=validation) as retrain_mock, mock.patch.object(
                p3, "evaluate_final", return_value=holdout
            ) as evaluate_mock:
                result = p3.finalize_phase3_real_al(finalize_args)
            self.assertEqual(result["status"], "finalized")
            retrain_mock.assert_called_once()
            evaluate_mock.assert_called_once()
            status_path = output_root / "branches" / "predictive_entropy" / "rounds" / "round_001" / "finalization" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["labeled_count"], 2)
            round_status = json.loads((status_path.parent.parent / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(round_status["status"], "ingested")

    def test_finalize_preview_and_post_ingest_metric_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root, strategies=["predictive_entropy"])
            with mock.patch.object(p3, "init_run", side_effect=self._fake_init_run):
                p3.init_phase3_real_al(args)
            output_root = Path(args.output_root)
            self._seed_phase3_selected_rows(output_root, "predictive_entropy", ["FINALSEQ"])
            self._fully_ingest_round(output_root, "predictive_entropy", ["FINALSEQ"])
            finalize_args = type("Args", (), {
                "output_root": args.output_root, "branch": "predictive_entropy", "round": 1,
                "evaluate_holdout": True, "dry_run": True, "write_supek_pbs": False, "force": False,
                "pbs_repo_root": None, "supek_walltime": None, "supek_queue": None,
                "supek_ncpus": None, "supek_ngpus": None, "supek_mem": None,
            })()
            result = p3.finalize_phase3_real_al(finalize_args)
            pbs = Path(result["pbs"])
            text = pbs.read_text(encoding="utf-8")
            self.assertIn("finalize_predictive_entropy_r001", pbs.name)
            self.assertIn("phase3-real-al finalize", text)
            self.assertIn("--round 1", text)
            self.assertIn("--evaluate-holdout", text)
            self.assertNotIn("qsub ", text)
            model_root = output_root / "branches" / "predictive_entropy" / "models" / "real_al" / "round_001"
            (model_root / "pre_proposal").mkdir(parents=True, exist_ok=True)
            (model_root / "post_ingest").mkdir(parents=True, exist_ok=True)
            (model_root / "pre_proposal" / "metrics.json").write_text(json.dumps({"f1": 0.5}), encoding="utf-8")
            (model_root / "post_ingest" / "metrics.json").write_text(json.dumps({"f1": 0.8}), encoding="utf-8")
            rows = p3._branch_round_metric_rows(output_root, ["predictive_entropy"], 1)
            self.assertEqual(rows[0]["metrics_stage"], "post_ingest")
            self.assertEqual(rows[0]["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
