from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from active_learning_thesis.cli import main as cli_main
from active_learning_thesis.thesis_figures import build_thesis_figures


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _seed_packet(root: Path) -> Path:
    packet_dir = root / "packet"
    run_dir = root / "run_a"
    (run_dir / "discovery" / "ucb").mkdir(parents=True)
    (run_dir / "discovery" / "ucb" / "summary.json").write_text(
        json.dumps(
            {
                "strategy": "ucb",
                "exported_count": 2,
                "top_batch_mean_utility_score": 0.91,
                "top_batch_mean_pred_mean": 0.78,
                "top_batch_mean_pred_std": 0.13,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    study_summary = root / "_study_evidence" / "f1_strategy_summary.csv"
    _write_csv(
        study_summary,
        ["rank", "strategy", "n_runs", "final_mean", "aulc_mean", "labels_to_target_median"],
        [
            {"rank": 1, "strategy": "ensemble_mi", "n_runs": 3, "final_mean": 0.84, "aulc_mean": 0.79, "labels_to_target_median": 45},
            {"rank": 2, "strategy": "random", "n_runs": 3, "final_mean": 0.72, "aulc_mean": 0.68, "labels_to_target_median": 60},
        ],
    )
    _write_csv(
        packet_dir / "tables" / "runs.csv",
        ["run_name", "run_dir", "baseline_f1", "final_f1"],
        [
            {"run_name": "run_a", "run_dir": str(run_dir), "baseline_f1": 0.5, "final_f1": 0.82},
            {"run_name": "run_b", "run_dir": str(root / "run_b"), "baseline_f1": 0.55, "final_f1": 0.77},
        ],
    )
    _write_csv(
        packet_dir / "tables" / "final_freezes.csv",
        ["run_name", "status", "metric", "final_f1"],
        [
            {"run_name": "run_a", "status": "frozen", "metric": "f1", "final_f1": 0.83},
            {"run_name": "run_b", "status": "frozen_with_warnings", "metric": "f1", "final_f1": 0.76},
        ],
    )
    _write_csv(
        packet_dir / "figure_data" / "learning_curves.csv",
        ["run_name", "strategy", "round_id", "labeled_count", "f1"],
        [
            {"run_name": "run_a", "strategy": "ensemble_mi", "round_id": 0, "labeled_count": 40, "f1": 0.62},
            {"run_name": "run_a", "strategy": "ensemble_mi", "round_id": 1, "labeled_count": 45, "f1": 0.70},
            {"run_name": "run_a", "strategy": "random", "round_id": 0, "labeled_count": 40, "f1": 0.58},
            {"run_name": "run_a", "strategy": "random", "round_id": 1, "labeled_count": 45, "f1": 0.61},
        ],
    )
    _write_csv(
        packet_dir / "tables" / "md_review_evidence.csv",
        ["run_name", "sequence", "review_evidence_state", "evidence_ready_for_ingest"],
        [
            {"run_name": "run_a", "sequence": "KFFAKK", "review_evidence_state": "Evidence-backed label", "evidence_ready_for_ingest": "yes"},
            {"run_name": "run_a", "sequence": "DDDDD", "review_evidence_state": "Needs review / label", "evidence_ready_for_ingest": "no"},
        ],
    )
    _write_csv(
        packet_dir / "tables" / "study_artifacts.csv",
        ["artifact", "kind", "path", "row_count", "columns"],
        [{"artifact": "f1_strategy_summary.csv", "kind": "csv", "path": str(study_summary), "row_count": 2, "columns": "strategy, aulc_mean"}],
    )
    _write_csv(packet_dir / "tables" / "metrics.csv", ["run_name"], [])
    manifest = {
        "title": "unit_packet",
        "metric": "f1",
        "output_dir": str(packet_dir),
        "outputs": {
            "runs": str(packet_dir / "tables" / "runs.csv"),
            "final_freezes": str(packet_dir / "tables" / "final_freezes.csv"),
            "learning_curves": str(packet_dir / "figure_data" / "learning_curves.csv"),
            "md_review_evidence": str(packet_dir / "tables" / "md_review_evidence.csv"),
            "study_artifacts": str(packet_dir / "tables" / "study_artifacts.csv"),
            "metrics": str(packet_dir / "tables" / "metrics.csv"),
        },
    }
    (packet_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return packet_dir


class ThesisFiguresTests(unittest.TestCase):
    def test_build_thesis_figures_writes_svg_tables_and_captions(self):
        with tempfile.TemporaryDirectory(prefix="thesis_figures_") as temp_dir:
            packet_dir = _seed_packet(Path(temp_dir))

            bundle = build_thesis_figures(packet_dir)

            self.assertEqual(bundle["status"], "ready")
            self.assertGreaterEqual(bundle["counts"]["figures"], 5)
            outputs = bundle["outputs"]
            for key in [
                "figure_final_scorecard",
                "figure_learning_curves",
                "figure_md_review_evidence",
                "figure_strategy_aulc",
                "figure_discovery_utility",
                "figure_captions",
                "manifest",
                "readme",
            ]:
                self.assertTrue(Path(outputs[key]).exists(), key)
            self.assertIn("<svg", Path(outputs["figure_final_scorecard"]).read_text(encoding="utf-8"))
            captions = Path(outputs["figure_captions"]).read_text(encoding="utf-8")
            self.assertIn("Replay learning curves", captions)
            scorecard_rows = _read_csv(Path(outputs["table_final_scorecard"]))
            self.assertEqual(scorecard_rows[0]["run"], "run_a")
            self.assertEqual(scorecard_rows[0]["source"], "final_freeze")

    def test_build_thesis_figures_cli_prints_json(self):
        with tempfile.TemporaryDirectory(prefix="thesis_figures_cli_") as temp_dir:
            packet_dir = _seed_packet(Path(temp_dir))
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = cli_main(["build-thesis-figures", "--packet-dir", str(packet_dir), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "ready")
            self.assertTrue(Path(payload["outputs"]["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
