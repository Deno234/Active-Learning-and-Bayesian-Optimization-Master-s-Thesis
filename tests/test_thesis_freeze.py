from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from active_learning_thesis.cli import main as cli_main
from active_learning_thesis.config import RunConfig
from active_learning_thesis.ledger import empty_row, save_ledger
from active_learning_thesis.thesis_freeze import freeze_final_result
from active_learning_thesis.thesis_packet import export_thesis_packet


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _seed_freezable_run(run_root: Path, *, pending: bool = False) -> Path:
    run_dir = run_root / ("pending_run" if pending else "freezable_run")
    RunConfig(
        run_name=run_dir.name,
        output_root=str(run_root),
        random_seed=4242,
        batch_size=1,
        max_rounds=1,
        epochs=1,
    ).save(run_dir / "config.json")
    rows = [
        empty_row(
            {
                "sequence": "AAAAA",
                "label": "0",
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
                "label": "1",
                "label_source": "experimental",
                "split": "validation",
                "mode": "experimental",
                "round_id": "0",
                "status": "validation",
            }
        ),
        empty_row(
            {
                "sequence": "DDDDD",
                "label": "0",
                "label_source": "experimental",
                "split": "holdout",
                "mode": "experimental",
                "round_id": "0",
                "status": "holdout",
            }
        ),
        empty_row(
            {
                "sequence": "EEEEE",
                "label": "" if pending else "1",
                "label_source": "" if pending else "cgmd",
                "split": "generated",
                "mode": "real_al",
                "round_id": "1",
                "status": "proposed" if pending else "acquired",
            }
        ),
    ]
    save_ledger(run_dir / "ledger.csv", rows)
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "metrics" / "final_holdout.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "strategy": "final_evaluation",
                "labeled_count": 2,
                "evaluation_dataset": "holdout",
                "surrogate_stage": "post_ingest",
                "surrogate_round_id": 1,
                "f1": 0.8125,
                "pr_auc": 0.875,
                "roc_auc": 0.9,
                "balanced_accuracy": 0.8,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ensemble_dir = run_dir / "models" / "real_al" / "round_001" / "post_ingest" / "ensemble"
    ensemble_dir.mkdir(parents=True)
    (ensemble_dir / "ap_sp_member_00.h5").write_text("synthetic model artifact\n", encoding="utf-8")
    _write_csv(
        run_dir / "imports" / "round_001_labels.csv",
        ["sequence", "round_id", "cgmd_label"],
        [{"sequence": "EEEEE", "round_id": "1", "cgmd_label": "1"}],
    )
    review_fields = [
        "sequence",
        "round_id",
        "campaign",
        "cluster",
        "md_profile",
        "package_dir",
        "job_root_status",
        "ap_5ns",
        "ap_12ns",
        "ap_25ns",
        "ap_50ns",
        "ap_100ns",
        "ap_200ns",
        "sasa_file",
        "ap_file",
        "review_notes",
        "cgmd_label",
        "label_review_schema",
        "label_rubric",
        "label_confidence",
        "label_evidence_tags",
        "label_evidence_summary",
        "reviewer",
        "reviewed_at",
    ]
    campaign_dir = run_dir / "md_campaigns" / "round_001_full"
    _write_csv(
        campaign_dir / "md_review.csv",
        review_fields,
        [
            {
                "sequence": "EEEEE",
                "round_id": "1",
                "campaign": "round_001_full",
                "cluster": "bura",
                "md_profile": "full",
                "package_dir": "packages/EEEEE",
                "job_root_status": "analysis_complete",
                "ap_5ns": "2.1",
                "ap_12ns": "2.4",
                "ap_25ns": "2.8",
                "ap_50ns": "3.2",
                "ap_100ns": "3.5",
                "ap_200ns": "3.8",
                "sasa_file": "packages/EEEEE/EEEEE_sasa.xvg",
                "ap_file": "packages/EEEEE/EEEEE_AP_SASA.txt",
                "review_notes": "AP and SASA support an assembling label.",
                "cgmd_label": "1",
                "label_review_schema": "structured",
                "label_rubric": "self_assembling",
                "label_confidence": "high",
                "label_evidence_tags": "ap_supports_label, sasa_supports_label",
                "label_evidence_summary": "Sustained high AP and compact SASA support assembly.",
                "reviewer": "unit-test",
                "reviewed_at": "2026-01-01T00:00:00",
            }
        ],
    )
    _write_csv(
        campaign_dir / "cgmd_ingest.csv",
        ["sequence", "round_id", "cgmd_label"],
        [{"sequence": "EEEEE", "round_id": "1", "cgmd_label": "1"}],
    )
    return run_dir


class ThesisFreezeTests(unittest.TestCase):
    def test_freeze_final_result_writes_manifest_model_card_and_checks(self):
        with tempfile.TemporaryDirectory(prefix="thesis_freeze_") as temp_dir:
            run_dir = _seed_freezable_run(Path(temp_dir))

            report = freeze_final_result(run_dir)

            self.assertEqual(report["status"], "frozen")
            outputs = report["outputs"]
            for key in ["freeze_json", "model_card", "checks_csv", "model_manifest_csv"]:
                self.assertTrue(Path(outputs[key]).exists(), key)
            self.assertEqual(report["artifact_hashes"]["final_metrics"], report["artifact_hashes"]["final_metrics"].lower())
            self.assertEqual(report["counts"]["pending_proposals"], 0)
            self.assertEqual(report["counts"]["model_artifacts"], 1)
            checks = _read_csv(Path(outputs["checks_csv"]))
            self.assertTrue(any(row["check_id"] == "md_reviews_resolved" and row["status"] == "pass" for row in checks))

    def test_freeze_final_result_blocks_unresolved_proposals(self):
        with tempfile.TemporaryDirectory(prefix="thesis_freeze_blocked_") as temp_dir:
            run_dir = _seed_freezable_run(Path(temp_dir), pending=True)

            report = freeze_final_result(run_dir)

            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["counts"]["pending_proposals"], 1)
            self.assertTrue(Path(report["outputs"]["freeze_json"]).exists())

    def test_freeze_final_cli_prints_json(self):
        with tempfile.TemporaryDirectory(prefix="thesis_freeze_cli_") as temp_dir:
            run_dir = _seed_freezable_run(Path(temp_dir))
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = cli_main(["freeze-final", "--run-dir", str(run_dir), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "frozen")
            self.assertTrue(Path(payload["outputs"]["model_card"]).exists())

    def test_thesis_packet_collects_final_freezes(self):
        with tempfile.TemporaryDirectory(prefix="thesis_freeze_packet_") as temp_dir:
            run_root = Path(temp_dir)
            run_dir = _seed_freezable_run(run_root)
            freeze_final_result(run_dir)

            packet = export_thesis_packet(run_root, output_dir=run_root / "packet", include_dashboard=False)

            self.assertEqual(packet["counts"]["final_freezes"], 1)
            freeze_rows = _read_csv(Path(packet["outputs"]["final_freezes"]))
            self.assertEqual(freeze_rows[0]["status"], "frozen")
            self.assertEqual(freeze_rows[0]["final_f1"], "0.8125")


if __name__ == "__main__":
    unittest.main()
