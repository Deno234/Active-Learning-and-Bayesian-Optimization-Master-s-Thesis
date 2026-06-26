from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from active_learning_thesis.cli import main as cli_main
from active_learning_thesis.thesis_canary import run_thesis_canary
from active_learning_thesis.thesis_packet import export_thesis_packet


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


class ThesisPacketTests(unittest.TestCase):
    def test_export_thesis_packet_collects_canary_and_review_evidence(self):
        with tempfile.TemporaryDirectory(prefix="thesis_packet_") as temp_dir:
            run_root = Path(temp_dir)
            run_thesis_canary(run_root=run_root, name="packet_canary", seed=1212, peptide_count=2)

            packet = export_thesis_packet(
                run_root,
                output_dir=run_root / "packet",
                title="Unit Packet",
                metric="f1",
                include_dashboard=False,
            )

            self.assertEqual(packet["counts"]["runs"], 1)
            self.assertEqual(packet["counts"]["canaries"], 1)
            self.assertEqual(packet["counts"]["md_review_rows"], 2)
            self.assertEqual(packet["counts"]["evidence_backed_reviews"], 2)
            outputs = packet["outputs"]
            for key in ["runs", "md_review_evidence", "metrics", "canary_reports", "manifest", "index", "reproducibility"]:
                self.assertTrue(Path(outputs[key]).exists(), key)

            review_rows = _read_csv(Path(outputs["md_review_evidence"]))
            self.assertTrue(all(row["review_evidence_state"] == "Evidence-backed label" for row in review_rows))
            canary_rows = _read_csv(Path(outputs["canary_reports"]))
            self.assertEqual(canary_rows[0]["status"], "passed")

    def test_export_thesis_packet_collects_study_artifacts_and_learning_curves(self):
        with tempfile.TemporaryDirectory(prefix="thesis_packet_study_") as temp_dir:
            run_root = Path(temp_dir)
            run_dir = run_root / "study_run"
            strategy_dir = run_dir / "replay" / "random"
            strategy_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps({"run_name": "study_run", "output_root": str(run_root)}),
                encoding="utf-8",
            )
            with (strategy_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["round_id", "labeled_count", "f1"])
                writer.writeheader()
                writer.writerow({"round_id": "0", "labeled_count": "40", "f1": "0.7"})
            study_dir = run_root / "_study_evidence"
            study_dir.mkdir()
            (study_dir / "f1_study_summary.json").write_text(json.dumps({"metric": "f1"}), encoding="utf-8")
            with (study_dir / "f1_strategy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["strategy", "aulc_mean"])
                writer.writeheader()
                writer.writerow({"strategy": "random", "aulc_mean": "0.7"})

            packet = export_thesis_packet(
                run_root,
                output_dir=run_root / "packet",
                include_dashboard=False,
            )

            learning_rows = _read_csv(Path(packet["outputs"]["learning_curves"]))
            study_rows = _read_csv(Path(packet["outputs"]["study_artifacts"]))
            self.assertEqual(learning_rows[0]["strategy"], "random")
            self.assertTrue(any(row["artifact"] == "f1_strategy_summary.csv" for row in study_rows))
            self.assertTrue(any(row["artifact"] == "f1_study_summary.json" for row in study_rows))

    def test_export_thesis_packet_cli_prints_json(self):
        with tempfile.TemporaryDirectory(prefix="thesis_packet_cli_") as temp_dir:
            run_root = Path(temp_dir)
            run_thesis_canary(run_root=run_root, name="cli_packet_canary", seed=3434, peptide_count=1)
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = cli_main(
                    [
                        "export-thesis-packet",
                        "--run-root",
                        str(run_root),
                        "--output-dir",
                        str(run_root / "packet"),
                        "--skip-dashboard",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["counts"]["canaries"], 1)
            self.assertTrue(Path(payload["outputs"]["index"]).exists())


if __name__ == "__main__":
    unittest.main()
