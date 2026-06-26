from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence

from active_learning_thesis.config import RunConfig
from active_learning_thesis.dashboard_actions import submit_update_md_review_action
from active_learning_thesis.ledger import empty_row, load_ledger, save_ledger, snapshot_ledger
from active_learning_thesis.md_review_evidence import review_evidence_status
from active_learning_thesis.md_workflow import (
    make_md_ingest_csv,
    parse_md_results,
    prepare_md_campaign,
)
from active_learning_thesis.workflow import _validate_cgmd_import

CANARY_DIRNAME = "_thesis_canaries"
CANARY_CAMPAIGN = "seeded_full_canary"
CANARY_IMPORT_NAME = "round_001_canary_labels.csv"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _safe_remove_run_dir(run_dir: Path, canary_root: Path) -> None:
    resolved_run_dir = run_dir.resolve()
    resolved_canary_root = canary_root.resolve()
    if resolved_run_dir == resolved_canary_root or resolved_canary_root not in resolved_run_dir.parents:
        raise ValueError(f"Refusing to remove path outside canary root: {run_dir}")
    shutil.rmtree(resolved_run_dir)


def _candidate_sequences(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequences: list[str] = []
    seen: set[str] = {"AAAAA", "CCCCC", "DDDDD", "FFFFF"}
    while len(sequences) < count:
        length = rng.randint(5, 8)
        sequence = "".join(rng.choice(amino_acids) for _ in range(length))
        if sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    return sequences


def _training_rows() -> list[dict[str, str]]:
    seeds = [
        ("AAAAA", "0", "train_pool"),
        ("CCCCC", "1", "train_pool"),
        ("DDDDD", "0", "validation"),
        ("FFFFF", "1", "holdout"),
    ]
    return [
        empty_row(
            {
                "sequence": sequence,
                "label": label,
                "label_source": "experimental",
                "split": split,
                "mode": "experimental",
                "round_id": "0",
                "status": split,
                "generator_origin": "thesis_canary_seed",
                "replay_role": "seed" if split == "train_pool" else "none",
            }
        )
        for sequence, label, split in seeds
    ]


def _score_row(sequence: str, index: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed + index * 101)
    pred_mean = 0.42 + 0.08 * index + rng.random() * 0.03
    pred_std = 0.09 + rng.random() * 0.02
    pred_entropy = 0.55 + rng.random() * 0.05
    pred_mi = 0.08 + rng.random() * 0.04
    acquisition_score = pred_mean + pred_std + pred_mi
    return {
        "sequence": sequence,
        "round_id": "1",
        "acquisition_strategy": "seeded_canary",
        "pred_mean": f"{pred_mean:.6f}",
        "pred_std": f"{pred_std:.6f}",
        "pred_entropy": f"{pred_entropy:.6f}",
        "pred_mutual_information": f"{pred_mi:.6f}",
        "raw_pred_mean": f"{pred_mean:.6f}",
        "raw_pred_std": f"{pred_std:.6f}",
        "raw_pred_entropy": f"{pred_entropy:.6f}",
        "raw_pred_mutual_information": f"{pred_mi:.6f}",
        "acquisition_score": f"{acquisition_score:.6f}",
    }


def _proposed_ledger_rows(batch_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        empty_row(
            {
                **row,
                "split": "generated",
                "mode": "real_al",
                "status": "proposed",
                "generator_origin": "seeded_thesis_canary",
            }
        )
        for row in batch_rows
    ]


def _write_manifest(run_dir: Path, sequences: list[str]) -> None:
    manifest = {
        "created_by": "thesis-canary",
        "splits": {
            "AAAAA": "train_pool",
            "CCCCC": "train_pool",
            "DDDDD": "validation",
            "FFFFF": "holdout",
            **{sequence: "generated" for sequence in sequences},
        },
        "replay_roles": {
            "AAAAA": "seed",
            "CCCCC": "seed",
            "DDDDD": "none",
            "FFFFF": "none",
            **{sequence: "none" for sequence in sequences},
        },
    }
    _save_json(run_dir / "split_manifest.json", manifest)


def _write_baseline_metrics(run_dir: Path) -> None:
    _save_json(
        run_dir / "metrics" / "baseline_round_000.json",
        {
            "round_id": 0,
            "strategy": "baseline",
            "stage": "baseline",
            "evaluation_dataset": "validation",
            "labeled_count": 2,
            "f1": 0.5,
            "pr_auc": 0.5,
            "roc_auc": 0.5,
            "balanced_accuracy": 0.5,
        },
    )


def _write_pdbs(campaign_dir: Path, sequences: list[str]) -> None:
    pdb_root = campaign_dir / "PDBs"
    pdb_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        pdb_text = (
            "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\n"
            "TER\n"
            "END\n"
        )
        (pdb_root / f"{sequence}.pdb").write_text(pdb_text, encoding="utf-8")
        package_dir = campaign_dir / "packages" / sequence
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / f"{sequence}.pdb").write_text(pdb_text, encoding="utf-8")


def _label_for_sequence(sequence: str, index: int) -> str:
    hydrophobic = sum(1 for aa in sequence if aa in {"A", "F", "I", "L", "M", "V", "W", "Y"})
    if hydrophobic >= max(2, len(sequence) // 2):
        return "1"
    return "0" if index % 2 else "1"


def _write_synthetic_md_outputs(campaign_dir: Path, sequences: list[str]) -> None:
    for index, sequence in enumerate(sequences):
        package_dir = campaign_dir / "packages" / sequence
        label = _label_for_sequence(sequence, index)
        base = 2.0 + index * 0.25
        if label == "1":
            ap_values = {
                5: base + 0.6,
                12: base + 0.9,
                25: base + 1.2,
                50: base + 1.5,
                100: base + 1.8,
                200: base + 2.0,
            }
        else:
            ap_values = {
                5: 0.55 + index * 0.05,
                12: 0.50 + index * 0.04,
                25: 0.45 + index * 0.03,
                50: 0.40 + index * 0.02,
                100: 0.35 + index * 0.02,
                200: 0.30 + index * 0.01,
            }
        (package_dir / f"{sequence}_canary_CG.xtc").write_text("synthetic trajectory\n", encoding="utf-8")
        (package_dir / f"{sequence}_sasa.xvg").write_text(
            "@ title \"Synthetic canary SASA\"\n"
            + "\n".join(f"{time:.3f} {10.0 / (1.0 + value):.6f}" for time, value in ap_values.items())
            + "\n",
            encoding="utf-8",
        )
        (package_dir / f"{sequence}_AP_SASA.txt").write_text(
            "\n".join(f"The AP for {time} ns is: {value:.6f}" for time, value in ap_values.items()) + "\n",
            encoding="utf-8",
        )


def _save_evidence_backed_reviews(
    *,
    run_root: Path,
    run_dir: Path,
    campaign_dir: Path,
    review_csv: Path,
    sequences: list[str],
) -> list[dict[str, str]]:
    saved: list[dict[str, str]] = []
    for index, sequence in enumerate(sequences):
        label = _label_for_sequence(sequence, index)
        rubric = "self_assembling" if label == "1" else "not_self_assembling"
        summary = (
            "Synthetic canary AP/SASA evidence shows sustained aggregation propensity."
            if label == "1"
            else "Synthetic canary AP/SASA evidence stays low and supports a non-assembling call."
        )
        submit_update_md_review_action(
            run_root=run_root,
            review_csv=review_csv,
            sequence=sequence,
            cgmd_label=label,
            review_notes="Seeded canary label generated from deterministic synthetic AP/SASA evidence.",
            label_rubric=rubric,
            label_confidence="high",
            label_evidence_tags="ap_supports_label, sasa_supports_label, trajectory_visual_check",
            label_evidence_summary=summary,
            reviewer="thesis-canary",
            related_run=str(run_dir),
            related_campaign=str(campaign_dir),
        )
        saved.append({"sequence": sequence, "cgmd_label": label, "label_rubric": rubric})
    return saved


def _normalize_canary_review_times(review_csv: Path, sequences: list[str], seed: int) -> None:
    rows = _read_csv(review_csv)
    if not rows:
        return
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    fieldnames = list(rows[0])
    if "reviewed_at" not in fieldnames:
        fieldnames.append("reviewed_at")
    for row in rows:
        sequence = str(row.get("sequence", ""))
        if sequence in sequence_index:
            row["reviewed_at"] = f"2026-01-01T00:{seed % 60:02d}:{sequence_index[sequence]:02d}"
    _save_csv(review_csv, fieldnames, rows)


def _copy_import_and_mark_acquired(run_dir: Path, ingest_csv: Path) -> tuple[Path, list[dict[str, str]]]:
    import_path = run_dir / "imports" / CANARY_IMPORT_NAME
    import_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ingest_csv, import_path)
    import_rows = _read_csv(import_path)
    ledger_rows = load_ledger(run_dir / "ledger.csv")
    by_sequence = {row["sequence"]: row for row in ledger_rows}
    for row in import_rows:
        ledger_row = by_sequence[row["sequence"]]
        ledger_row["label"] = row["cgmd_label"]
        ledger_row["label_source"] = "cgmd"
        ledger_row["status"] = "acquired"
    save_ledger(run_dir / "ledger.csv", ledger_rows)
    snapshot_ledger(run_dir / "snapshots", ledger_rows, "ledger_round_001_canary_acquired")
    return import_path, import_rows


def _write_post_ingest_metrics(run_dir: Path, import_rows: list[dict[str, str]]) -> Path:
    metrics_path = run_dir / "models" / "real_al" / "round_001" / "post_ingest" / "metrics.json"
    labels = [int(row["cgmd_label"]) for row in import_rows]
    positive_rate = sum(labels) / max(len(labels), 1)
    _save_json(
        metrics_path,
        {
            "round_id": 1,
            "strategy": "seeded_canary",
            "stage": "post_ingest",
            "evaluation_dataset": "synthetic_canary",
            "labeled_count": len(import_rows) + 2,
            "f1": 0.75 + positive_rate * 0.05,
            "pr_auc": 0.70 + positive_rate * 0.05,
            "roc_auc": 0.72 + positive_rate * 0.05,
            "balanced_accuracy": 0.70,
            "note": "Synthetic canary metric. This verifies artifact flow, not scientific performance.",
        },
    )
    return metrics_path


def _write_next_round_stub(run_dir: Path, seed: int, count: int) -> Path:
    next_sequences = _candidate_sequences(seed + 7919, count)
    rows = [_score_row(sequence, index, seed + 2) | {"round_id": "2"} for index, sequence in enumerate(next_sequences)]
    batch_path = run_dir / "batches" / "round_002_canary_batch.csv"
    _save_csv(
        batch_path,
        [
            "sequence",
            "round_id",
            "acquisition_strategy",
            "pred_mean",
            "pred_std",
            "pred_entropy",
            "pred_mutual_information",
            "raw_pred_mean",
            "raw_pred_std",
            "raw_pred_entropy",
            "raw_pred_mutual_information",
            "acquisition_score",
        ],
        rows,
    )
    return batch_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_markdown_report(report_path: Path, report: dict[str, object]) -> None:
    checks = report.get("checks", {}) if isinstance(report.get("checks", {}), dict) else {}
    lines = [
        "# Seeded Thesis Canary Report",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Seed: `{report.get('seed', '')}`",
        f"- Run directory: `{report.get('run_dir', '')}`",
        f"- Proposed peptides: `{report.get('peptide_count', '')}`",
        f"- Evidence-backed reviews: `{checks.get('evidence_backed_reviews', '')}`",
        f"- Ingest rows validated: `{checks.get('ingest_rows_validated', '')}`",
        f"- Next-round stub: `{report.get('outputs', {}).get('next_batch_csv', '') if isinstance(report.get('outputs', {}), dict) else ''}`",
        "",
        "This is a synthetic canary. It verifies the local thesis-loop contracts and artifact handoffs, not physical MD truth or model performance.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_thesis_canary(
    *,
    run_root: Path,
    name: str = "seeded_thesis_canary",
    seed: int = 20260425,
    peptide_count: int = 2,
    force: bool = False,
) -> dict[str, object]:
    if peptide_count <= 0:
        raise ValueError("peptide_count must be positive.")
    run_root = Path(run_root)
    canary_root = run_root / CANARY_DIRNAME
    run_name = f"{name}_seed_{seed}"
    run_dir = canary_root / run_name
    if run_dir.exists():
        if not force:
            raise FileExistsError(f"Canary run already exists: {run_dir}")
        _safe_remove_run_dir(run_dir, canary_root)

    sequences = _candidate_sequences(seed, peptide_count)
    config = RunConfig(
        run_name=run_name,
        output_root=str(canary_root),
        random_seed=seed,
        batch_size=peptide_count,
        max_rounds=1,
        epochs=1,
        candidate_pool_min=max(peptide_count * 2, 10),
        real_strategy="seeded_canary",
        replay_strategies=["random"],
        train_family_for_init=False,
    )
    config.save(run_dir / "config.json")
    _write_manifest(run_dir, sequences)
    _write_baseline_metrics(run_dir)

    batch_rows = [_score_row(sequence, index, seed) for index, sequence in enumerate(sequences)]
    ledger_rows = [*_training_rows(), *_proposed_ledger_rows(batch_rows)]
    save_ledger(run_dir / "ledger.csv", ledger_rows)
    snapshot_ledger(run_dir / "snapshots", ledger_rows, "ledger_round_001_canary_proposed")
    batch_path = run_dir / "batches" / "round_001_canary_batch.csv"
    _save_csv(batch_path, list(batch_rows[0]), batch_rows)

    campaign_dir = prepare_md_campaign(
        run_dir,
        batch_path,
        CANARY_CAMPAIGN,
        cluster="bura",
        md_profile="full",
    )
    _write_pdbs(campaign_dir, sequences)
    _write_synthetic_md_outputs(campaign_dir, sequences)
    review_csv = parse_md_results(campaign_dir)
    saved_reviews = _save_evidence_backed_reviews(
        run_root=canary_root,
        run_dir=run_dir,
        campaign_dir=campaign_dir,
        review_csv=review_csv,
        sequences=sequences,
    )
    _normalize_canary_review_times(review_csv, sequences, seed)
    review_rows = _read_csv(review_csv)
    evidence_statuses = [review_evidence_status(row) for row in review_rows]
    ingest_csv = make_md_ingest_csv(campaign_dir, review_csv)
    validated_round_id, validated_import_rows = _validate_cgmd_import(load_ledger(run_dir / "ledger.csv"), ingest_csv)
    import_csv, import_rows = _copy_import_and_mark_acquired(run_dir, ingest_csv)
    metrics_path = _write_post_ingest_metrics(run_dir, import_rows)
    next_batch_csv = _write_next_round_stub(run_dir, seed, peptide_count)

    checks = {
        "proposed_batch_rows": len(batch_rows),
        "analysis_complete_rows": sum(1 for row in review_rows if row.get("job_root_status") == "analysis_complete"),
        "evidence_backed_reviews": sum(1 for status in evidence_statuses if bool(status.get("ingest_ready", False))),
        "ingest_rows_validated": len(validated_import_rows),
        "validated_round_id": validated_round_id,
        "import_rows_written": len(import_rows),
        "post_ingest_metrics_exists": metrics_path.exists(),
        "next_batch_rows": len(_read_csv(next_batch_csv)),
    }
    status = "passed" if all(
        [
            checks["proposed_batch_rows"] == peptide_count,
            checks["analysis_complete_rows"] == peptide_count,
            checks["evidence_backed_reviews"] == peptide_count,
            checks["ingest_rows_validated"] == peptide_count,
            checks["import_rows_written"] == peptide_count,
            checks["post_ingest_metrics_exists"],
            checks["next_batch_rows"] == peptide_count,
        ]
    ) else "failed"

    outputs = {
        "config": str(run_dir / "config.json"),
        "ledger": str(run_dir / "ledger.csv"),
        "batch_csv": str(batch_path),
        "campaign_dir": str(campaign_dir),
        "review_csv": str(review_csv),
        "ingest_csv": str(ingest_csv),
        "import_csv": str(import_csv),
        "post_ingest_metrics": str(metrics_path),
        "next_batch_csv": str(next_batch_csv),
    }
    report = {
        "status": status,
        "created_at": _now_iso(),
        "seed": seed,
        "run_root": str(run_root),
        "canary_root": str(canary_root),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "peptide_count": peptide_count,
        "sequences": sequences,
        "saved_reviews": saved_reviews,
        "checks": checks,
        "outputs": outputs,
        "fingerprint": {
            "batch_csv": _sha256(batch_path),
            "review_csv": _sha256(review_csv),
            "ingest_csv": _sha256(ingest_csv),
            "import_csv": _sha256(import_csv),
            "next_batch_csv": _sha256(next_batch_csv),
        },
        "notes": [
            "Synthetic canary verifies local workflow contracts, not MD physics.",
            "The post-ingest metric is synthetic and should not be used as scientific evidence.",
        ],
    }
    report_json = run_dir / "canary_report.json"
    report_md = run_dir / "canary_report.md"
    report["outputs"]["report_json"] = str(report_json)
    report["outputs"]["report_markdown"] = str(report_md)
    _save_json(report_json, report)
    _write_markdown_report(report_md, report)
    return report
