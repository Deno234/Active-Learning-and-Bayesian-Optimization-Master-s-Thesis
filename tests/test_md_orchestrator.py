from __future__ import annotations

import csv
import json
import shutil
import unittest
import tempfile
import uuid
from pathlib import Path
from unittest import mock

from active_learning_thesis.config import RunConfig
from active_learning_thesis.md_orchestrator import (
    NEXT_COMMANDS_FILENAME,
    STAGE_META_FILENAME,
    finalize_md_stage,
    md_ladder_status,
    prepare_md_stage,
)


class MdOrchestratorTests(unittest.TestCase):
    def _scratch_dir(self, name: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"{name}_{uuid.uuid4().hex}_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _write_batch_csv(self, path: Path, rows: list[dict[str, str]] | None = None) -> None:
        rows = rows or [
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.91",
                "pred_std": "0.08",
                "pred_entropy": "0.31",
                "pred_mutual_information": "0.05",
                "acquisition_score": "0.72",
            },
            {
                "sequence": "CCCCC",
                "round_id": "1",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.84",
                "pred_std": "0.06",
                "pred_entropy": "0.28",
                "pred_mutual_information": "0.04",
                "acquisition_score": "0.63",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _create_run_dir(self, temp_dir: Path, run_name: str = "md_orchestrator_test") -> tuple[Path, Path]:
        run_dir = temp_dir / "run"
        run_dir.mkdir(parents=True)
        RunConfig(run_name=run_name, output_root=str(temp_dir)).save(run_dir / "config.json")
        batch_csv = run_dir / "round_001_batch.csv"
        self._write_batch_csv(batch_csv)
        return run_dir, batch_csv

    def _read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _create_reuse_campaign(self, root: Path, sequence: str = "AAAAA") -> Path:
        campaign_dir = root / "previous_campaign"
        pdb_dir = campaign_dir / "PDBs"
        pdb_dir.mkdir(parents=True)
        (pdb_dir / f"{sequence}.pdb").write_text("ATOM\n", encoding="utf-8")
        return campaign_dir

    def _seed_smoke_success(self, campaign_dir: Path, sequence: str = "AAAAA") -> None:
        package_dir = campaign_dir / "packages" / sequence
        (package_dir / f"{sequence}_CG_150_CG.gro").write_text("gro", encoding="utf-8")

    def _seed_full_analysis_success(self, campaign_dir: Path, sequence: str = "AAAAA") -> None:
        package_dir = campaign_dir / "packages" / sequence
        (package_dir / f"{sequence}_CG_150_CG.xtc").write_text("xtc", encoding="utf-8")
        (package_dir / f"{sequence}_sasa.xvg").write_text(
            "@ title \"SASA\"\n0.000 10.0\n5.000 5.0\n12.000 4.0\n25.000 2.0\n50.000 1.0\n100.000 0.5\n200.000 0.25\n",
            encoding="utf-8",
        )
        (package_dir / f"{sequence}_AP_SASA.txt").write_text(
            "The AP for 5 ns is: 2.0\nThe AP for 12 ns is: 2.5\n",
            encoding="utf-8",
        )

    def _seed_full_sasa_only(self, campaign_dir: Path, sequence: str = "AAAAA") -> None:
        package_dir = campaign_dir / "packages" / sequence
        (package_dir / f"{sequence}_CG_150_CG.xtc").write_text("xtc", encoding="utf-8")
        (package_dir / f"{sequence}_sasa.xvg").write_text(
            "@ title \"SASA\"\n0.000 10.0\n",
            encoding="utf-8",
        )
        (package_dir / f"{sequence}_AP_SASA.txt").write_text("", encoding="utf-8")

    def test_prepare_md_stage_filters_sequence_and_writes_artifacts(self):
        temp_dir = self._scratch_dir("md_orchestrator_prepare")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        build_calls: list[bool] = []

        def fake_build(campaign_dir: Path, validate_only: bool = False) -> Path:
            build_calls.append(validate_only)
            pdb_dir = campaign_dir / "PDBs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            sequence = "AAAAA"
            (pdb_dir / f"{sequence}.pdb").write_text("ATOM\n", encoding="utf-8")
            package_dir = campaign_dir / "packages" / sequence
            package_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdb_dir / f"{sequence}.pdb", package_dir / f"{sequence}.pdb")
            manifest_rows = self._read_csv(campaign_dir / "manifest.csv")
            for row in manifest_rows:
                row["pdb_status"] = "built_local"
            with (campaign_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            return pdb_dir

        with mock.patch("active_learning_thesis.md_orchestrator.build_pdbs", side_effect=fake_build):
            campaign_dir, next_commands_path = prepare_md_stage(
                run_dir,
                batch_csv,
                "AAAAA",
                "guided_line",
                "line_smoke",
                exclude_nodes="bura201, bura202",
            )

        self.assertEqual(build_calls, [False])
        manifest_rows = self._read_csv(campaign_dir / "manifest.csv")
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["sequence"], "AAAAA")
        self.assertEqual(manifest_rows[0]["pdb_status"], "built_local")

        selected_rows = self._read_csv(campaign_dir / "selected_batch.csv")
        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(selected_rows[0]["sequence"], "AAAAA")

        meta = json.loads((campaign_dir / STAGE_META_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(meta["sequence"], "AAAAA")
        self.assertEqual(meta["md_profile"], "line_smoke")
        self.assertEqual(meta["exclude_nodes"], "bura201,bura202")
        self.assertEqual(meta["expected_terminal_status"], "dynamics_complete")
        self.assertEqual(meta["next_profile_on_success"], "production_smoke")

        next_commands = next_commands_path.read_text(encoding="utf-8")
        self.assertIn("cd ~/guided_line", next_commands)
        self.assertIn("bash ./submit_chain.sh --exclude bura201,bura202 AAAAA", next_commands)
        self.assertTrue((campaign_dir / NEXT_COMMANDS_FILENAME).exists())

    def test_prepare_md_stage_reuses_prior_pdb_and_validates_package(self):
        temp_dir = self._scratch_dir("md_orchestrator_reuse")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")

        campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_prod",
            "production_smoke",
            reuse_pdb_from=reuse_campaign_dir,
        )

        self.assertTrue((campaign_dir / "PDBs" / "AAAAA.pdb").exists())
        self.assertTrue((campaign_dir / "packages" / "AAAAA" / "AAAAA.pdb").exists())
        self.assertEqual(
            (campaign_dir / "PDBs" / "AAAAA.pdb").read_text(encoding="utf-8"),
            (reuse_campaign_dir / "PDBs" / "AAAAA.pdb").read_text(encoding="utf-8"),
        )
        manifest_rows = self._read_csv(campaign_dir / "manifest.csv")
        self.assertEqual(manifest_rows[0]["pdb_status"], "manual_ready")
        meta = json.loads((campaign_dir / STAGE_META_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(meta["reuse_pdb_from"], str(reuse_campaign_dir))

    def test_prepare_md_stage_rejects_missing_or_duplicate_sequence(self):
        temp_dir = self._scratch_dir("md_orchestrator_errors")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")

        with self.assertRaisesRegex(ValueError, "Sequence not found"):
            prepare_md_stage(run_dir, batch_csv, "VVVVV", "missing_seq", "line_smoke", reuse_pdb_from=reuse_campaign_dir)

        duplicate_rows = [
            {
                "sequence": "AAAAA",
                "round_id": "1",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.91",
                "pred_std": "0.08",
                "pred_entropy": "0.31",
                "pred_mutual_information": "0.05",
                "acquisition_score": "0.72",
            },
            {
                "sequence": "AAAAA",
                "round_id": "2",
                "acquisition_strategy": "ensemble_mi",
                "pred_mean": "0.93",
                "pred_std": "0.07",
                "pred_entropy": "0.29",
                "pred_mutual_information": "0.06",
                "acquisition_score": "0.75",
            },
        ]
        duplicate_batch = run_dir / "duplicate_batch.csv"
        self._write_batch_csv(duplicate_batch, duplicate_rows)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            prepare_md_stage(run_dir, duplicate_batch, "AAAAA", "duplicate_seq", "line_smoke", reuse_pdb_from=reuse_campaign_dir)

    def test_finalize_md_stage_reports_next_stage_for_smoke_profiles(self):
        temp_dir = self._scratch_dir("md_orchestrator_finalize_smoke")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")

        line_campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_line",
            "line_smoke",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_smoke_success(line_campaign_dir)
        _, line_review, line_message = finalize_md_stage(line_campaign_dir)
        self.assertEqual(line_review["job_root_status"], "dynamics_complete")
        self.assertIn("production_smoke", line_message)

        prod_campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_prod",
            "production_smoke",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_smoke_success(prod_campaign_dir)
        _, prod_review, prod_message = finalize_md_stage(prod_campaign_dir)
        self.assertEqual(prod_review["job_root_status"], "dynamics_complete")
        self.assertIn("Next recommended stage: full", prod_message)

    def test_finalize_md_stage_reports_full_analysis_and_sasa_only(self):
        temp_dir = self._scratch_dir("md_orchestrator_finalize_full")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")

        full_analysis_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_full_analysis",
            "full",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_full_analysis_success(full_analysis_dir)
        _, analysis_review, analysis_message = finalize_md_stage(full_analysis_dir)
        self.assertEqual(analysis_review["job_root_status"], "analysis_complete")
        self.assertIn("make-md-ingest-csv", analysis_message)

        full_sasa_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_full_sasa",
            "full",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_full_sasa_only(full_sasa_dir)
        _, sasa_review, sasa_message = finalize_md_stage(full_sasa_dir)
        self.assertEqual(sasa_review["job_root_status"], "sasa_complete")
        self.assertIn("not ingest-ready", sasa_message)

    def test_finalize_md_stage_imports_staged_package_before_parse(self):
        temp_dir = self._scratch_dir("md_orchestrator_finalize_staged")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")
        campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_full_staged",
            "full",
            reuse_pdb_from=reuse_campaign_dir,
        )
        staged_package_dir = temp_dir / "downloads" / "bura" / "guided_full_staged" / "packages" / "AAAAA"
        staged_package_dir.mkdir(parents=True)
        (staged_package_dir / "AAAAA_CG_150_CG.xtc").write_text("xtc", encoding="utf-8")
        (staged_package_dir / "AAAAA_sasa.xvg").write_text("@ title \"SASA\"\n0.000 10.0\n", encoding="utf-8")
        (staged_package_dir / "AAAAA_AP_SASA.txt").write_text(
            "The AP for 5 ns is: 2.0\nThe AP for 200 ns is: 3.0\n",
            encoding="utf-8",
        )
        (staged_package_dir / "AAAAA_CG_150_CG.log").write_text(
            """
               Core t (s)   Wall t (s)        (%)
       Time:     480.000       10.000     4800.0
                 (ns/day)    (hour/ns)
Performance:      123.456        0.194
Finished mdrun on rank 0
""".strip()
            + "\n",
            encoding="utf-8",
        )

        _, review, _ = finalize_md_stage(campaign_dir, staged_package_dir)

        self.assertEqual(review["job_root_status"], "analysis_complete")
        self.assertEqual(review["ap_5ns"], "2.0")
        self.assertEqual(review["ap_200ns"], "3.0")
        self.assertEqual(review["md_runtime_wall_hms"], "00:10")
        self.assertEqual(review["md_runtime_wall_seconds"], "10.000")
        self.assertEqual(review["md_runtime_core_seconds"], "480.000")
        self.assertEqual(review["md_runtime_ns_per_day"], "123.456")
        self.assertTrue((campaign_dir / "packages" / "AAAAA" / "AAAAA_AP_SASA.txt").exists())

    def test_md_ladder_status_tracks_next_profile_and_review_readiness(self):
        temp_dir = self._scratch_dir("md_orchestrator_ladder")
        run_dir, batch_csv = self._create_run_dir(temp_dir)
        reuse_campaign_dir = self._create_reuse_campaign(temp_dir, "AAAAA")

        line_campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_line",
            "line_smoke",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_smoke_success(line_campaign_dir)
        finalize_md_stage(line_campaign_dir)

        prod_campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_prod",
            "production_smoke",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_smoke_success(prod_campaign_dir)
        finalize_md_stage(prod_campaign_dir)

        status = md_ladder_status(run_dir, "AAAAA")
        self.assertEqual(status["next_profile"], "full")
        self.assertFalse(status["ready_for_review"])
        self.assertEqual(len(status["campaigns"]), 2)

        full_campaign_dir, _ = prepare_md_stage(
            run_dir,
            batch_csv,
            "AAAAA",
            "guided_full",
            "full",
            reuse_pdb_from=reuse_campaign_dir,
        )
        self._seed_full_analysis_success(full_campaign_dir)
        finalize_md_stage(full_campaign_dir)

        status = md_ladder_status(run_dir, "AAAAA")
        self.assertEqual(status["next_profile"], "")
        self.assertTrue(status["ready_for_review"])
        self.assertEqual(len(status["campaigns"]), 3)


if __name__ == "__main__":
    unittest.main()
