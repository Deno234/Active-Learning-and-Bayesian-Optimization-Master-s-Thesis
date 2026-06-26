from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from active_learning_thesis.cli import main as cli_main
from active_learning_thesis.ledger import empty_row, save_ledger
from active_learning_thesis.optional_evaluator_study import run_optional_evaluator_study


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _seed_optional_run(root: Path) -> Path:
    run_dir = root / "optional_run"
    ledger_rows = [
        empty_row(
            {
                "sequence": "AAAAA",
                "label": "0",
                "label_source": "experimental",
                "split": "train_pool",
                "status": "train_pool",
                "round_id": "0",
            }
        ),
        empty_row(
            {
                "sequence": "KFFAKK",
                "label": "",
                "split": "generated",
                "status": "proposed",
                "round_id": "1",
                "pred_mean": "0.82",
                "pred_std": "0.12",
                "pred_mutual_information": "0.09",
                "acquisition_strategy": "ensemble_mi",
                "acquisition_score": "0.91",
            }
        ),
        empty_row(
            {
                "sequence": "DDDDD",
                "label": "",
                "split": "generated",
                "status": "candidate_scored",
                "round_id": "1",
                "pred_mean": "0.22",
                "pred_std": "0.05",
                "pred_mutual_information": "0.02",
                "acquisition_strategy": "ensemble_mi",
                "acquisition_score": "0.24",
            }
        ),
    ]
    save_ledger(run_dir / "ledger.csv", ledger_rows)
    _write_csv(
        run_dir / "discovery" / "ucb" / "candidates.csv",
        ["sequence", "strategy", "pred_mean", "pred_std", "utility_score"],
        [
            {"sequence": "FWYKK", "strategy": "ucb", "pred_mean": "0.74", "pred_std": "0.18", "utility_score": "0.92"},
            {"sequence": "GGGGG", "strategy": "ucb", "pred_mean": "0.15", "pred_std": "0.03", "utility_score": "0.18"},
        ],
    )
    return run_dir


class OptionalEvaluatorStudyTests(unittest.TestCase):
    def test_optional_evaluator_study_exports_disagreement_and_complexity_tables(self):
        with tempfile.TemporaryDirectory(prefix="optional_eval_") as temp_dir:
            root = Path(temp_dir)
            run_dir = _seed_optional_run(root)
            external_scores = root / "external_scores.csv"
            _write_csv(
                external_scores,
                ["sequence", "evaluator", "score", "confidence"],
                [
                    {"sequence": "KFFAKK", "evaluator": "external_model_a", "score": "0.35", "confidence": "high"},
                    {"sequence": "DDDDD", "evaluator": "external_model_a", "score": "0.18", "confidence": "medium"},
                    {"sequence": "FWYKK", "evaluator": "external_model_a", "score": "0.91", "confidence": "high"},
                ],
            )

            report = run_optional_evaluator_study(run_dir, external_scores=external_scores)

            self.assertEqual(report["status"], "ready")
            outputs = report["outputs"]
            for key in [
                "internal_predictions",
                "external_scores",
                "evaluator_disagreement",
                "complexity_bins",
                "complexity_summary",
                "manifest",
                "readme",
            ]:
                self.assertTrue(Path(outputs[key]).exists(), key)

            disagreement_rows = _read_csv(Path(outputs["evaluator_disagreement"]))
            self.assertEqual(len(disagreement_rows), 3)
            kffakk = next(row for row in disagreement_rows if row["sequence"] == "KFFAKK")
            self.assertEqual(kffakk["label_disagreement"], "yes")
            self.assertEqual(kffakk["internal_label"], "1")
            self.assertEqual(kffakk["external_label"], "0")

            complexity_rows = _read_csv(Path(outputs["complexity_bins"]))
            self.assertTrue(any(row["sequence"] == "FWYKK" for row in complexity_rows))
            self.assertTrue(all(row["complexity_bin"] for row in complexity_rows))

    def test_optional_evaluator_study_without_external_scores_stays_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="optional_eval_no_external_") as temp_dir:
            run_dir = _seed_optional_run(Path(temp_dir))

            report = run_optional_evaluator_study(run_dir)

            self.assertEqual(report["status"], "no_external_scores")
            self.assertEqual(report["counts"]["external_rows"], 0)
            self.assertEqual(report["counts"]["disagreement_rows"], 0)
            self.assertTrue(Path(report["outputs"]["complexity_bins"]).exists())

    def test_optional_evaluator_study_cli_prints_json(self):
        with tempfile.TemporaryDirectory(prefix="optional_eval_cli_") as temp_dir:
            root = Path(temp_dir)
            run_dir = _seed_optional_run(root)
            external_scores = root / "external_scores.csv"
            _write_csv(
                external_scores,
                ["sequence", "evaluator", "external_score"],
                [{"sequence": "KFFAKK", "evaluator": "external_model_b", "external_score": "0.9"}],
            )

            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = cli_main(
                    [
                        "optional-evaluator-study",
                        "--run-dir",
                        str(run_dir),
                        "--external-scores",
                        str(external_scores),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "ready")
            self.assertTrue(Path(payload["outputs"]["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
