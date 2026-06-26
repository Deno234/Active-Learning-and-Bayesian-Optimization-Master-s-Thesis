from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from active_learning_thesis.cli import main as cli_main
from active_learning_thesis.md_review_evidence import review_evidence_status
from active_learning_thesis.thesis_canary import run_thesis_canary


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


class ThesisCanaryTests(unittest.TestCase):
    def test_run_thesis_canary_creates_evidence_backed_loop_report(self):
        with tempfile.TemporaryDirectory(prefix="thesis_canary_") as temp_dir:
            report = run_thesis_canary(
                run_root=Path(temp_dir),
                name="unit_canary",
                seed=1234,
                peptide_count=2,
            )

            self.assertEqual(report["status"], "passed")
            outputs = report["outputs"]
            review_rows = _read_csv(Path(outputs["review_csv"]))
            self.assertEqual(len(review_rows), 2)
            self.assertTrue(all(row["job_root_status"] == "analysis_complete" for row in review_rows))
            self.assertTrue(all(review_evidence_status(row)["ingest_ready"] for row in review_rows))

            ingest_rows = _read_csv(Path(outputs["ingest_csv"]))
            import_rows = _read_csv(Path(outputs["import_csv"]))
            self.assertEqual(ingest_rows, import_rows)
            self.assertEqual(list(ingest_rows[0]), ["sequence", "round_id", "cgmd_label"])
            self.assertTrue(Path(outputs["post_ingest_metrics"]).exists())
            self.assertTrue(Path(outputs["next_batch_csv"]).exists())
            self.assertTrue(Path(outputs["report_json"]).exists())
            self.assertTrue(Path(outputs["report_markdown"]).exists())

    def test_thesis_canary_force_recreates_same_seed_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="thesis_canary_force_") as temp_dir:
            first = run_thesis_canary(
                run_root=Path(temp_dir),
                name="repeatable",
                seed=2222,
                peptide_count=2,
            )
            second = run_thesis_canary(
                run_root=Path(temp_dir),
                name="repeatable",
                seed=2222,
                peptide_count=2,
                force=True,
            )

            self.assertEqual(first["sequences"], second["sequences"])
            self.assertEqual(first["fingerprint"]["batch_csv"], second["fingerprint"]["batch_csv"])
            self.assertEqual(first["fingerprint"]["review_csv"], second["fingerprint"]["review_csv"])
            self.assertEqual(first["fingerprint"]["ingest_csv"], second["fingerprint"]["ingest_csv"])

    def test_thesis_canary_cli_prints_json_report(self):
        with tempfile.TemporaryDirectory(prefix="thesis_canary_cli_") as temp_dir:
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = cli_main(
                    [
                        "thesis-canary",
                        "--run-root",
                        temp_dir,
                        "--name",
                        "cli_canary",
                        "--seed",
                        "777",
                        "--peptides",
                        "1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["peptide_count"], 1)
            self.assertTrue(Path(payload["outputs"]["report_json"]).exists())


if __name__ == "__main__":
    unittest.main()
