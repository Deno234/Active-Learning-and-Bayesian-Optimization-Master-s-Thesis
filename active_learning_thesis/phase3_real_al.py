from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable, Sequence

from active_learning_thesis.config import RunConfig
from active_learning_thesis.ledger import (
    current_real_training_rows,
    load_ledger,
    next_real_round_id,
    save_ledger,
    snapshot_ledger,
    unresolved_proposals,
)
from active_learning_thesis.md_review_evidence import review_evidence_status
from active_learning_thesis.phase1_reproduction import PHASE1_MODELS
from active_learning_thesis.phase2_replay import load_frozen_model_config
from active_learning_thesis.workflow import evaluate_final, init_run, propose_round, retrain_after_ingest


DEFAULT_PHASE1_ROOT = Path("thesis_results") / "01_reproduction"
DEFAULT_PHASE2_ROOT = Path("thesis_results") / "02_replay"
DEFAULT_OUTPUT_ROOT = Path("thesis_results") / "03_real_al"
DEFAULT_STRATEGIES = (
    "predictive_entropy",
    "family_qbc",
    "cluster_diverse_representative",
)
DEFAULT_BACKUP_STRATEGY = "ensemble_mi"
DEFAULT_SUPEK_WALLTIME = {
    "predictive_entropy": "05:00:00",
    "cluster_diverse_representative": "05:00:00",
    "family_qbc": "05:00:00",
}
DEFAULT_PHASE3_GA_MAX_ATTEMPTS = 100
INVENTORY_EVENT_FIELDS = [
    "event_id",
    "sequence",
    "branch_strategy",
    "round_id",
    "event_type",
    "event_status",
    "proposal_csv",
    "selected_batch_csv",
    "campaign_name",
    "campaign_dir",
    "md_profile",
    "timestamp",
    "source_action_id",
    "notes",
]
INVENTORY_SNAPSHOT_FIELDS = [
    "sequence",
    "current_status",
    "selected_by_branches",
    "active_campaigns",
    "reviewed_labels",
    "ingested_branches",
    "latest_event_type",
    "latest_event_status",
    "latest_timestamp",
    "notes",
]
PHASE3_INGEST_FIELDS = [
    "sequence",
    "round_id",
    "cgmd_label",
    "branch_strategy",
    "label_confidence",
    "label_rubric",
    "review_notes",
    "reviewer",
    "campaign_name",
    "campaign_dir",
    "review_csv",
    "source_branch",
    "source_round_id",
]
PHASE3_INGEST_LOG_FIELDS = [
    "timestamp",
    "branch_strategy",
    "round_id",
    "sequence",
    "cgmd_label",
    "label_confidence",
    "label_rubric",
    "reviewer",
    "campaign_name",
    "campaign_dir",
    "import_csv",
    "status",
    "notes",
]


@dataclass(frozen=True)
class SupekResources:
    queue: str = "gpu"
    ncpus: int = 4
    ngpus: int = 1
    mem: str = "40GB"
    walltime: str = ""


def run_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    action = getattr(args, "phase3_real_al_action", None)
    if action == "init":
        return init_phase3_real_al(args)
    if action == "propose":
        return propose_phase3_real_al(args)
    if action == "compare":
        return compare_phase3_real_al(args)
    if action == "status":
        return status_phase3_real_al(args)
    if action == "make-ingest-csv":
        return make_phase3_ingest_csv(args)
    if action == "ingest":
        return ingest_phase3_labels(args)
    if action == "finalize":
        return finalize_phase3_real_al(args)
    raise ValueError(f"Unsupported phase3-real-al action: {action}")


def init_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    phase1_root = Path(args.phase1_root)
    phase2_root = Path(args.phase2_root)
    strategies = tuple(args.strategies or DEFAULT_STRATEGIES)
    frozen_config = load_frozen_model_config(phase1_root)
    output_root.mkdir(parents=True, exist_ok=True)
    branches_root = output_root / "branches"
    branches_root.mkdir(parents=True, exist_ok=True)
    _ensure_inventory_files(output_root)

    created_branches: list[str] = []
    for strategy in strategies:
        branch_dir = branches_root / strategy
        if branch_dir.exists() and any(branch_dir.iterdir()):
            if not args.force:
                raise FileExistsError(f"Phase 3 branch already exists: {branch_dir}")
            _archive_existing_branch(output_root, branch_dir)
        config = _branch_config(
            strategy=strategy,
            branches_root=branches_root,
            output_root=output_root,
            phase1_root=phase1_root,
            phase2_root=phase2_root,
            frozen_config=frozen_config,
            args=args,
        )
        init_run(config, train_baseline=False)
        _write_phase3_branch_scaffold(branch_dir, strategy, frozen_config)
        write_supek_proposal_preview(output_root, strategy, 1, args=args)
        created_branches.append(strategy)

    manifest = {
        "phase": "phase3_real_al",
        "created_at": _now_iso(),
        "phase1_root": str(phase1_root),
        "phase2_root": str(phase2_root),
        "output_root": str(output_root),
        "strategies": list(strategies),
        "backup_strategy": getattr(args, "backup_strategy", DEFAULT_BACKUP_STRATEGY),
        "branches_root": str(branches_root),
        "md_inventory_root": str(output_root / "md_inventory"),
        "supek_pbs_root": str(output_root / "logs" / "supek_pbs"),
    }
    _write_json(output_root / "phase3_real_al_manifest.json", manifest)
    submit = write_supek_submit_preview(output_root, strategies)
    compare_phase3_real_al(argparse.Namespace(output_root=str(output_root)))
    return {
        "status": "initialized",
        "output_root": str(output_root),
        "branches": created_branches,
        "manifest": str(output_root / "phase3_real_al_manifest.json"),
        "supek_submit_preview": str(submit),
    }


def propose_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    branch = str(args.branch)
    round_id = int(args.round)
    _validate_round_id(round_id)
    _validate_phase3_proposal_ready(output_root, branch, round_id)
    command = _exact_or_constructed_command(output_root, branch, round_id)
    if getattr(args, "dry_run", False) or getattr(args, "write_supek_pbs", False):
        pbs = write_supek_proposal_preview(output_root, branch, round_id, args=args)
        _write_round_status(
            output_root,
            branch,
            round_id,
            "preview_ready",
            exact_command=_proposal_cli_command(output_root, branch, round_id),
            completed_at=_now_iso(),
            pbs_path=pbs,
        )
        return {"status": "preview-written", "branch": branch, "round_id": round_id, "pbs": str(pbs)}
    branch_dir = _branch_dir(output_root, branch)
    started_at = _now_iso()
    _write_round_status(
        output_root,
        branch,
        round_id,
        "running",
        exact_command=command,
        started_at=started_at,
    )
    try:
        batch_path = propose_round(branch_dir, branch)
        _validate_batch_round(batch_path, round_id)
        outputs = mirror_round_outputs(
            output_root,
            branch,
            round_id,
            batch_path,
            exact_command=command,
            started_at=started_at,
        )
        compare_phase3_real_al(argparse.Namespace(output_root=str(output_root), round=round_id))
        return {"status": "proposal-complete", "branch": branch, "round_id": round_id, "outputs": outputs}
    except Exception as exc:
        _write_round_status(
            output_root,
            branch,
            round_id,
            "failed",
            exact_command=command,
            started_at=started_at,
            completed_at=_now_iso(),
            error=str(exc),
        )
        raise


def compare_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    round_id = int(getattr(args, "round", 1) or 1)
    _validate_round_id(round_id)
    comparison_root = output_root / "comparison" / f"round_{round_id:03d}"
    comparison_root.mkdir(parents=True, exist_ok=True)
    branches = _branch_names(output_root)
    selected_rows, by_branch = _round_selected_rows_for_comparison(output_root, branches, round_id)
    _write_csv(comparison_root / "branch_selected_sequences.csv", selected_rows)

    overlap_rows = _overlap_rows(branches, by_branch)
    _write_csv(comparison_root / "branch_proposal_overlap.csv", overlap_rows)
    duplicate_rows = _duplicate_sequence_rows(selected_rows)
    _write_csv(comparison_root / "duplicate_sequences_across_branches.csv", duplicate_rows)
    _write_csv(
        comparison_root / "branch_md_status_summary.csv",
        _read_csv(output_root / "md_inventory" / "md_inventory.csv"),
        fieldnames=INVENTORY_SNAPSHOT_FIELDS,
    )
    _write_csv(comparison_root / "branch_label_summary.csv", _branch_label_summary_rows(output_root, branches))
    _write_csv(comparison_root / "branch_round_metrics.csv", _branch_round_metric_rows(output_root, branches, round_id))
    markdown = _branch_comparison_markdown(branches, overlap_rows, selected_rows, duplicate_rows, round_id=round_id)
    (comparison_root / "branch_comparison.md").write_text(markdown, encoding="utf-8")
    if round_id == 1:
        _write_legacy_round_comparison_aliases(output_root / "comparison", comparison_root)
    _write_all_rounds_comparison(output_root, branches)
    _write_all_round_metrics(output_root, branches)
    _write_final_holdout_summary(output_root, branches)
    return {
        "status": "comparison-written",
        "output_root": str(comparison_root),
        "branches": branches,
        "round_id": round_id,
    }


def status_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    manifest = _safe_read_json(output_root / "phase3_real_al_manifest.json")
    inventory = _read_csv(output_root / "md_inventory" / "md_inventory.csv")
    return {
        "status": "ok" if manifest else "not_initialized",
        "output_root": str(output_root),
        "manifest": manifest,
        "branches": _branch_names(output_root),
        "branch_continuation": {
            branch: summarize_phase3_branch_continuation(output_root, branch)
            for branch in _branch_names(output_root)
        },
        "round_status": {
            branch: detect_round_status(output_root, branch, _latest_known_round(output_root, branch) or 1)
            for branch in _branch_names(output_root)
        },
        "ingest_status": {
            branch: summarize_phase3_ingest_status(output_root, branch, _latest_known_round(output_root, branch) or 1)
            for branch in _branch_names(output_root)
        },
        "finalization_status": {
            branch: detect_finalization_status(output_root, branch, _latest_known_round(output_root, branch) or 1)
            for branch in _branch_names(output_root)
        },
        "inventory_sequences": len(inventory),
    }


def make_phase3_ingest_csv(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    branch = str(args.branch)
    round_id = int(args.round)
    _validate_round_id(round_id)
    round_dir = _round_dir(output_root, branch, round_id)
    ingest_dir = round_dir / "ingest"
    review_dir = round_dir / "review"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    ingest_csv = ingest_dir / "cgmd_ingest.csv"
    if ingest_csv.exists() and not bool(getattr(args, "force", False)):
        raise FileExistsError(f"Phase 3 ingest CSV already exists: {ingest_csv}")

    selected_rows = _selected_batch_rows(output_root, branch, round_id)
    review_rows = _collect_phase3_review_rows(output_root, branch, round_id, selected_rows)
    ready_rows: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    reviewed_count = 0
    selected_sequences = [_normalize_sequence(row.get("sequence", "")) for row in selected_rows]
    review_by_sequence = _best_review_by_sequence(review_rows)
    for sequence in selected_sequences:
        if not sequence:
            continue
        review = review_by_sequence.get(sequence)
        if review is None:
            blockers.append(
                {
                    "sequence": sequence,
                    "reason": "missing_review_row",
                    "missing": "review row",
                    "blockers": "-",
                }
            )
            continue
        status = review_evidence_status(review)
        if str(status.get("label", "")).strip() in {"0", "1"}:
            reviewed_count += 1
        if bool(status.get("ingest_ready", False)):
            ready_rows.append(_phase3_ingest_row(review, branch, round_id))
        else:
            blockers.append(
                {
                    "sequence": sequence,
                    "reason": str(status.get("state", "")),
                    "missing": str(status.get("missing_text", "")),
                    "blockers": str(status.get("blocker_text", "")),
                    "review_csv": str(review.get("review_csv", "")),
                }
            )

    _write_csv(ingest_csv, ready_rows, fieldnames=PHASE3_INGEST_FIELDS)
    preview = {
        "status": "ready" if ready_rows else "blocked",
        "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}",
        "selected_count": len([seq for seq in selected_sequences if seq]),
        "review_rows_found": len(review_rows),
        "reviewed_count": reviewed_count,
        "ingest_ready_count": len(ready_rows),
        "blocked_rows_count": len(blockers),
        "ingest_csv": str(ingest_csv),
        "blockers": blockers,
        "rows": ready_rows,
    }
    _write_json(ingest_dir / "ingest_preview.json", preview)
    _write_json(
        ingest_dir / "ingest_status.json",
        {
            "status": "csv_created" if ready_rows else "blocked_no_ready_rows",
            "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}",
            "timestamp": _now_iso(),
            "ingest_csv": str(ingest_csv),
            "ingest_ready_count": len(ready_rows),
            "blocked_rows_count": len(blockers),
        },
    )
    _write_json(
        review_dir / "review_status.json",
        {
            "status": "review_ready" if ready_rows else "review_blocked",
            "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}",
            "review_rows_found": len(review_rows),
            "reviewed_count": reviewed_count,
            "ingest_ready_count": len(ready_rows),
            "blocked_rows_count": len(blockers),
            "blockers": blockers,
        },
    )
    _append_phase3_inventory_events(
        output_root,
        branch,
        round_id,
        ready_rows,
        event_type="reviewed_label",
        event_status="reviewed_label_ready",
        notes_prefix="ready for branch-local ingest",
    )
    return {
        "status": "ingest-csv-written",
        "branch": branch,
        "round_id": round_id,
        "ingest_csv": str(ingest_csv),
        "ingest_ready_count": len(ready_rows),
        "blocked_rows_count": len(blockers),
    }


def ingest_phase3_labels(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    branch = str(args.branch)
    round_id = int(args.round)
    _validate_round_id(round_id)
    import_csv = Path(args.import_csv)
    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))
    rows = _read_csv(import_csv)
    validation = _validate_phase3_ingest_rows(output_root, branch, round_id, rows, force=force)
    if dry_run:
        return {
            "status": "dry-run",
            "branch": branch,
            "round_id": round_id,
            "import_csv": str(import_csv),
            "valid": not validation["blockers"],
            "would_ingest_count": len(validation["ready_rows"]),
            "blockers": validation["blockers"],
        }

    ingest_dir = _round_dir(output_root, branch, round_id) / "ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    status_path = ingest_dir / "ingest_status.json"
    previous_status = _safe_read_json(status_path)
    if (
        isinstance(previous_status, dict)
        and str(previous_status.get("status", "")) in {"ingested", "partially_ingested"}
        and not force
    ):
        raise ValueError("Phase 3 labels were already ingested for this branch/round. Use --force to repeat safely.")
    if validation["blockers"]:
        _write_json(
            status_path,
            {
                "status": "blocked",
                "branch_strategy": branch,
                "round_id": f"round_{round_id:03d}",
                "timestamp": _now_iso(),
                "import_csv": str(import_csv),
                "blockers": validation["blockers"],
            },
        )
        raise ValueError(f"Phase 3 ingest validation failed with {len(validation['blockers'])} blocker(s).")

    branch_dir = _branch_dir(output_root, branch)
    ledger_path = branch_dir / "ledger.csv"
    ledger_rows = load_ledger(ledger_path)
    by_sequence = {_normalize_sequence(row.get("sequence", "")): row for row in ledger_rows}
    timestamp = _now_iso()
    log_rows: list[dict[str, object]] = []
    for row in validation["ready_rows"]:
        sequence = _normalize_sequence(row.get("sequence", ""))
        ledger_row = by_sequence[sequence]
        ledger_row["label"] = str(row.get("cgmd_label", "")).strip()
        ledger_row["label_source"] = "cgmd"
        ledger_row["status"] = "acquired"
        log_rows.append(
            {
                "timestamp": timestamp,
                "branch_strategy": branch,
                "round_id": f"round_{round_id:03d}",
                "sequence": sequence,
                "cgmd_label": str(row.get("cgmd_label", "")).strip(),
                "label_confidence": str(row.get("label_confidence", "")),
                "label_rubric": str(row.get("label_rubric", "")),
                "reviewer": str(row.get("reviewer", "")),
                "campaign_name": str(row.get("campaign_name", "")),
                "campaign_dir": str(row.get("campaign_dir", "")),
                "import_csv": str(import_csv),
                "status": "ingested",
                "notes": "branch-local ledger update only",
            }
        )
    save_ledger(ledger_path, ledger_rows)
    snapshot_ledger(branch_dir / "snapshots", ledger_rows, f"ledger_round_{round_id:03d}_phase3_ingest")
    _write_current_labeled_ledger(branch_dir)

    selected_sequences = validation["selected_sequences"]
    acquired_selected = [
        seq
        for seq in selected_sequences
        if by_sequence.get(seq, {}).get("status") == "acquired" and by_sequence.get(seq, {}).get("label") in {"0", "1"}
    ]
    round_status = "ingested" if len(acquired_selected) == len(selected_sequences) else "partially_ingested"
    _write_csv(ingest_dir / "ingest_log.csv", log_rows, fieldnames=PHASE3_INGEST_LOG_FIELDS)
    status_payload = {
        "status": round_status,
        "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}",
        "timestamp": timestamp,
        "import_csv": str(import_csv),
        "ingested_count": len(log_rows),
        "selected_count": len(selected_sequences),
        "remaining_selected_count": len(selected_sequences) - len(acquired_selected),
        "next_proposal_blocked": round_status == "partially_ingested",
        "notes": "Phase 3 branch-isolated ledger/provenance ingestion; no retraining was run.",
    }
    _write_json(status_path, status_payload)
    round_status_payload = _round_status_payload(
        output_root,
        branch,
        round_id,
        round_status,
        completed_at=timestamp,
        selected_count=len(selected_sequences),
    )
    round_status_payload["ingest_status_path"] = str(status_path)
    round_status_payload["next_proposal_blocked"] = round_status == "partially_ingested"
    round_status_payload["ingested_count"] = len(acquired_selected)
    _write_json(_round_dir(output_root, branch, round_id) / "status.json", round_status_payload)
    _append_phase3_inventory_events(
        output_root,
        branch,
        round_id,
        validation["ready_rows"],
        event_type="ingested_label",
        event_status="ingested_branch_label",
        notes_prefix=f"branch-local ingest from {import_csv}",
    )
    compare_phase3_real_al(argparse.Namespace(output_root=str(output_root), round=round_id))
    return {
        "status": round_status,
        "branch": branch,
        "round_id": round_id,
        "ingested_count": len(log_rows),
        "selected_count": len(selected_sequences),
        "current_labeled_ledger": str(branch_dir / "current_labeled_ledger.csv"),
        "ingest_status": str(status_path),
    }


def finalize_phase3_real_al(args: argparse.Namespace) -> dict[str, object]:
    """Retrain one fully ingested branch and optionally evaluate the frozen holdout."""
    output_root = Path(args.output_root)
    branch = str(args.branch)
    round_id = int(args.round)
    _validate_round_id(round_id)
    readiness = _previous_round_readiness(output_root, branch, round_id + 1)
    if not readiness.get("ready"):
        raise ValueError(
            f"Cannot finalize round {round_id} for {branch}: {readiness.get('blocked_reason', 'round is not fully ingested')}"
        )
    finalization_dir = _round_dir(output_root, branch, round_id) / "finalization"
    status_path = finalization_dir / "status.json"
    previous = _safe_read_json(status_path)
    force = bool(getattr(args, "force", False))
    if isinstance(previous, dict) and previous.get("status") == "completed" and not force:
        raise ValueError(f"Round {round_id} for {branch} is already finalized. Use --force to repeat.")
    evaluate_holdout_flag = bool(getattr(args, "evaluate_holdout", False))
    command = _finalize_cli_command(output_root, branch, round_id, evaluate_holdout_flag)
    if bool(getattr(args, "dry_run", False)) or bool(getattr(args, "write_supek_pbs", False)):
        pbs = write_supek_finalize_preview(output_root, branch, round_id, args=args)
        _write_json(status_path, {
            "status": "preview_ready", "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}", "round_number": round_id,
            "evaluate_holdout": evaluate_holdout_flag, "exact_command": command,
            "pbs_path": str(pbs), "timestamp": _now_iso(),
        })
        return {"status": "preview-written", "branch": branch, "round_id": round_id, "pbs": str(pbs)}
    finalization_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now_iso()
    _write_json(status_path, {
        "status": "running", "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}", "round_number": round_id,
        "evaluate_holdout": evaluate_holdout_flag, "exact_command": command,
        "timestamp_started": started_at,
    })
    try:
        branch_dir = _branch_dir(output_root, branch)
        validation_metrics = retrain_after_ingest(branch_dir, round_id)
        holdout_metrics = evaluate_final(branch_dir) if evaluate_holdout_flag else {}
        validation_path = branch_dir / "models" / "real_al" / f"round_{round_id:03d}" / "post_ingest" / "metrics.json"
        holdout_path = branch_dir / "metrics" / "final_holdout.json"
        payload = {
            "status": "completed", "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}", "round_number": round_id,
            "evaluate_holdout": evaluate_holdout_flag, "exact_command": command,
            "timestamp_started": started_at, "timestamp_completed": _now_iso(),
            "labeled_count": validation_metrics.get("labeled_count", ""),
            "validation_metrics_path": str(validation_path),
            "holdout_metrics_path": str(holdout_path) if evaluate_holdout_flag else "",
            "validation_metrics": validation_metrics, "holdout_metrics": holdout_metrics,
        }
        _write_json(status_path, payload)
        compare_phase3_real_al(argparse.Namespace(output_root=str(output_root), round=round_id))
        return {
            "status": "finalized", "branch": branch, "round_id": round_id,
            "labeled_count": validation_metrics.get("labeled_count", ""),
            "validation_metrics": str(validation_path),
            "holdout_metrics": str(holdout_path) if evaluate_holdout_flag else "",
            "finalization_status": str(status_path),
        }
    except Exception as exc:
        _write_json(status_path, {
            "status": "failed", "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}", "round_number": round_id,
            "evaluate_holdout": evaluate_holdout_flag, "exact_command": command,
            "timestamp_started": started_at, "timestamp_completed": _now_iso(),
            "error": str(exc),
        })
        raise


def mirror_round_outputs(
    output_root: Path,
    branch: str,
    round_id: int,
    batch_path: Path,
    *,
    exact_command: str = "",
    started_at: str = "",
) -> dict[str, str]:
    branch_dir = _branch_dir(output_root, branch)
    round_dir = _round_dir(output_root, branch, round_id)
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "review").mkdir(parents=True, exist_ok=True)
    (round_dir / "ingest").mkdir(parents=True, exist_ok=True)
    (round_dir / "md_campaigns").mkdir(parents=True, exist_ok=True)
    scored_path = branch_dir / "candidates" / f"round_{round_id:03d}_scored.csv"
    selected_rows = _read_csv(batch_path)
    _validate_selected_rows_round(selected_rows, round_id, batch_path)
    snapshot_before = {row.get("sequence", ""): row for row in _read_csv(output_root / "md_inventory" / "md_inventory.csv")}
    proposal_path = round_dir / "proposal.csv"
    selected_path = round_dir / "selected_batch.csv"
    scored_copy = round_dir / "scored_candidates.csv"
    acquisition_log = round_dir / "acquisition_log.csv"
    command_preview = round_dir / "command_preview.txt"
    round_config_path = round_dir / "config.json"
    campaign_root = branch_dir / "md_campaigns"
    augmented_rows = []
    for rank, row in enumerate(selected_rows, start=1):
        sequence = str(row.get("sequence", "")).strip()
        previous = snapshot_before.get(sequence, {})
        campaign_name = _campaign_name(branch, round_id, sequence)
        campaign_dir = campaign_root / campaign_name
        warning = ""
        next_action = "prepare_md_stage"
        if previous:
            status = str(previous.get("current_status", "known_in_inventory"))
            warning = f"already present in shared MD inventory: {status}"
            next_action = "review_existing_inventory_before_md"
        augmented = {
            **row,
            "selection_rank": row.get("selection_rank") or str(rank),
            "md_inventory_status": previous.get("current_status", "not_seen"),
            "duplicate_md_warning": warning,
            "recommended_next_action": next_action,
            "campaign_name": campaign_name,
            "campaign_dir": str(campaign_dir),
        }
        augmented_rows.append(augmented)
    _write_csv(proposal_path, augmented_rows)
    _write_csv(selected_path, augmented_rows)
    if scored_path.exists():
        _validate_scored_rows_round(scored_path, round_id)
        shutil.copyfile(scored_path, scored_copy)
    else:
        _write_csv(scored_copy, [])
    _write_csv(acquisition_log, augmented_rows)
    command_preview.write_text(_proposal_command_preview(output_root, branch, round_id, selected_path), encoding="utf-8")
    _append_selection_events(output_root, branch, round_id, proposal_path, selected_path, augmented_rows)
    _write_current_labeled_ledger(branch_dir)
    candidate_count = len(_read_csv(scored_copy))
    selected_count = len(augmented_rows)
    completed_at = _now_iso()
    status_payload = _round_status_payload(
        output_root,
        branch,
        round_id,
        "completed",
        exact_command=exact_command,
        started_at=started_at,
        completed_at=completed_at,
        selected_batch_path=selected_path,
        scored_candidates_path=scored_copy,
        candidate_count=candidate_count,
        selected_count=selected_count,
    )
    _write_json(round_config_path, _round_config_payload(output_root, branch, round_id, status_payload))
    _write_json(round_dir / "status.json", status_payload)
    _validate_round_outputs(output_root, branch, round_id)
    return {
        "proposal": str(proposal_path),
        "scored_candidates": str(scored_copy),
        "selected_batch": str(selected_path),
        "acquisition_log": str(acquisition_log),
        "config": str(round_config_path),
        "status": str(round_dir / "status.json"),
        "command_preview": str(command_preview),
    }


def write_supek_proposal_preview(
    output_root: Path,
    branch: str,
    round_id: int,
    *,
    args: argparse.Namespace | None = None,
) -> Path:
    pbs_root = output_root / "logs" / "supek_pbs"
    runtime_root = output_root / "logs" / "supek_runtime"
    pbs_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    resources = _supek_resources(output_root, branch, args=args)
    repo_root = _pbs_repo_root(args)
    absolute_output = _target_path(output_root, repo_root)
    absolute_runtime = absolute_output / "logs" / "supek_runtime"
    command = (
        _proposal_cli_command(absolute_output, branch, round_id)
    )
    job_name = _job_name(branch, round_id)
    pbs_path = pbs_root / f"supek_phase3_propose_{branch}_r{round_id:03d}.pbs"
    pbs_path.write_text(
        _pbs_text(job_name, command, absolute_runtime, repo_root, resources),
        encoding="utf-8",
    )
    preview = _round_dir(output_root, branch, round_id) / "command_preview.txt"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(
        _supek_preview_text(command, pbs_path, repo_root, absolute_runtime, resources),
        encoding="utf-8",
    )
    _write_round_status(
        output_root,
        branch,
        round_id,
        "preview_ready",
        exact_command=command,
        completed_at=_now_iso(),
        pbs_path=pbs_path,
    )
    return pbs_path


def write_supek_finalize_preview(
    output_root: Path,
    branch: str,
    round_id: int,
    *,
    args: argparse.Namespace | None = None,
) -> Path:
    pbs_root = output_root / "logs" / "supek_pbs"
    pbs_root.mkdir(parents=True, exist_ok=True)
    resources = _supek_resources(output_root, branch, args=args)
    repo_root = _pbs_repo_root(args)
    absolute_output = _target_path(output_root, repo_root)
    absolute_runtime = absolute_output / "logs" / "supek_runtime"
    evaluate_holdout_flag = bool(getattr(args, "evaluate_holdout", False)) if args is not None else False
    command = _finalize_cli_command(absolute_output, branch, round_id, evaluate_holdout_flag)
    job_name = f"{_job_name(branch, round_id)}_fin"
    pbs_path = pbs_root / f"supek_phase3_finalize_{branch}_r{round_id:03d}.pbs"
    pbs_path.write_text(_pbs_text(job_name, command, absolute_runtime, repo_root, resources), encoding="utf-8")
    preview = _round_dir(output_root, branch, round_id) / "finalization" / "command_preview.txt"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_text(_supek_preview_text(command, pbs_path, repo_root, absolute_runtime, resources), encoding="utf-8")
    return pbs_path


def write_supek_submit_preview(output_root: Path, branches: Sequence[str]) -> Path:
    pbs_root = output_root / "logs" / "supek_pbs"
    pbs_root.mkdir(parents=True, exist_ok=True)
    submit_path = pbs_root / "supek_phase3_submit_proposals_round_001.sh"
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "# Preview-only helper. Inspect each PBS file before running this manually.",
    ]
    for branch in branches:
        pbs = (output_root / "logs" / "supek_pbs" / f"supek_phase3_propose_{branch}_r001.pbs").resolve()
        lines.append(f'# qsub "{pbs.as_posix()}"')
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return submit_path


def _branch_config(
    *,
    strategy: str,
    branches_root: Path,
    output_root: Path,
    phase1_root: Path,
    phase2_root: Path,
    frozen_config: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> RunConfig:
    model_num_cells = dict(RunConfig().model_num_cells)
    model_kernel_size = dict(RunConfig().model_kernel_size)
    for model_name, payload in frozen_config.items():
        if model_name not in PHASE1_MODELS:
            continue
        model_num_cells[model_name] = int(payload["num_cells"])
        kernel = payload.get("kernel_size", model_kernel_size.get(model_name, 4))
        model_kernel_size[model_name] = 4 if str(kernel) == "n/a" else int(kernel)
    resources = _supek_resources(output_root, strategy, args=args, read_config=False)
    return RunConfig(
        run_name=strategy,
        output_root=str(branches_root),
        random_seed=int(getattr(args, "random_seed", 20260317)),
        replay_seed_size=int(getattr(args, "replay_seed_size", 40)),
        batch_size=int(getattr(args, "batch_size", 5)),
        max_rounds=int(getattr(args, "max_rounds", 10)),
        candidate_pool_min=int(getattr(args, "candidate_pool_min", 50)),
        ga_max_attempts=int(getattr(args, "ga_max_attempts", DEFAULT_PHASE3_GA_MAX_ATTEMPTS)),
        ensemble_size=int(getattr(args, "ensemble_size", 5)),
        ensemble_seeds=[int(getattr(args, "random_seed", 20260317)) + index for index in range(int(getattr(args, "ensemble_size", 5)))],
        epochs=int(getattr(args, "epochs", 70)),
        real_strategy=strategy,
        replay_strategies=[],
        generator_objective_mode=str(getattr(args, "generator_objective_mode", "match_acquisition")),
        use_similarity_penalty=bool(getattr(args, "use_similarity_penalty", True)),
        use_length_penalty=not bool(getattr(args, "no_length_penalty", False)),
        binary_threshold_strategy=str(getattr(args, "binary_threshold_strategy", "pr_best_f1")),
        use_calibrated_acquisition=not bool(getattr(args, "raw_acquisition", False)),
        train_family_for_init=False,
        phase="phase3_real_al",
        branch_strategy=strategy,
        backup_strategy=str(getattr(args, "backup_strategy", DEFAULT_BACKUP_STRATEGY)),
        phase1_root=str(phase1_root),
        phase2_root=str(phase2_root),
        phase3_output_root=str(output_root),
        md_inventory_root=str(output_root / "md_inventory"),
        supek_queue=resources.queue,
        supek_ncpus=resources.ncpus,
        supek_ngpus=resources.ngpus,
        supek_mem=resources.mem,
        supek_walltime=resources.walltime,
        model_num_cells=model_num_cells,
        model_kernel_size=model_kernel_size,
    )


def _write_phase3_branch_scaffold(branch_dir: Path, strategy: str, frozen_config: dict[str, dict[str, object]]) -> None:
    for path in [
        branch_dir / "rounds" / "round_001" / "md_campaigns",
        branch_dir / "rounds" / "round_001" / "review",
        branch_dir / "rounds" / "round_001" / "ingest",
        branch_dir / "actions",
        branch_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _write_current_labeled_ledger(branch_dir)
    _write_json(branch_dir / "phase1_frozen_model_config_used.json", frozen_config)
    _write_json(
        branch_dir / "phase3_branch_manifest.json",
        {
            "phase": "phase3_real_al",
            "branch_strategy": strategy,
            "rounds": ["round_001"],
            "branch_isolation": "labels ingested here update only this branch by default",
        },
    )


def _write_current_labeled_ledger(branch_dir: Path) -> None:
    ledger_path = branch_dir / "ledger.csv"
    if not ledger_path.exists():
        return
    rows = load_ledger(ledger_path)
    current = list(current_real_training_rows(rows))
    seen = {_normalize_sequence(row.get("sequence", "")) for row in current}
    for row in rows:
        sequence = _normalize_sequence(row.get("sequence", ""))
        if (
            sequence
            and sequence not in seen
            and row.get("label")
            and row.get("label_source") == "cgmd"
            and row.get("status") == "acquired"
        ):
            current.append(row)
            seen.add(sequence)
    save_ledger(branch_dir / "current_labeled_ledger.csv", current)


def _ensure_inventory_files(output_root: Path) -> None:
    inventory_root = output_root / "md_inventory"
    inventory_root.mkdir(parents=True, exist_ok=True)
    events = inventory_root / "md_inventory_events.csv"
    snapshot = inventory_root / "md_inventory.csv"
    if not events.exists():
        _write_csv(events, [], fieldnames=INVENTORY_EVENT_FIELDS)
    if not snapshot.exists():
        _write_csv(snapshot, [], fieldnames=INVENTORY_SNAPSHOT_FIELDS)


def _append_selection_events(
    output_root: Path,
    branch: str,
    round_id: int,
    proposal_path: Path,
    selected_path: Path,
    selected_rows: list[dict[str, object]],
) -> None:
    events_path = output_root / "md_inventory" / "md_inventory_events.csv"
    existing = _read_csv(events_path)
    events = list(existing)
    timestamp = _now_iso()
    for row in selected_rows:
        sequence = str(row.get("sequence", "")).strip()
        if not sequence:
            continue
        events.append(
            {
                "event_id": f"{timestamp}_{branch}_r{round_id:03d}_{sequence}",
                "sequence": sequence,
                "branch_strategy": branch,
                "round_id": f"round_{round_id:03d}",
                "event_type": "selected_for_md",
                "event_status": "selected_pending_md",
                "proposal_csv": str(proposal_path),
                "selected_batch_csv": str(selected_path),
                "campaign_name": str(row.get("campaign_name", "")),
                "campaign_dir": str(row.get("campaign_dir", "")),
                "md_profile": str(row.get("md_profile", "")),
                "timestamp": timestamp,
                "source_action_id": "",
                "notes": str(row.get("duplicate_md_warning", "")),
            }
        )
    _write_csv(events_path, events, fieldnames=INVENTORY_EVENT_FIELDS)
    _write_inventory_snapshot(output_root)


def _write_inventory_snapshot(output_root: Path) -> None:
    events = _read_csv(output_root / "md_inventory" / "md_inventory_events.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in events:
        sequence = row.get("sequence", "")
        if sequence:
            grouped.setdefault(sequence, []).append(row)
    snapshot_rows = []
    for sequence, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item.get("timestamp", ""))
        selected_by = []
        campaigns = []
        labels = []
        ingested = []
        notes = []
        for row in rows:
            branch_round = f"{row.get('branch_strategy', '')}:{row.get('round_id', '')}"
            if row.get("event_type") == "selected_for_md" and branch_round not in selected_by:
                selected_by.append(branch_round)
            if row.get("campaign_name") and row.get("campaign_name") not in campaigns:
                campaigns.append(row.get("campaign_name", ""))
            if row.get("event_type") == "reviewed_label":
                labels.append(row.get("event_status", ""))
            if row.get("event_type") == "ingested_label":
                ingested.append(branch_round)
            if row.get("notes"):
                notes.append(row.get("notes", ""))
        latest = rows[-1]
        snapshot_rows.append(
            {
                "sequence": sequence,
                "current_status": latest.get("event_status", ""),
                "selected_by_branches": ";".join(selected_by),
                "active_campaigns": ";".join(campaigns),
                "reviewed_labels": ";".join(labels),
                "ingested_branches": ";".join(ingested),
                "latest_event_type": latest.get("event_type", ""),
                "latest_event_status": latest.get("event_status", ""),
                "latest_timestamp": latest.get("timestamp", ""),
                "notes": " | ".join(notes),
            }
        )
    _write_csv(output_root / "md_inventory" / "md_inventory.csv", snapshot_rows, fieldnames=INVENTORY_SNAPSHOT_FIELDS)


def summarize_phase3_ingest_status(output_root: Path, branch: str, round_id: int) -> dict[str, object]:
    round_dir = _round_dir(output_root, branch, round_id)
    selected_rows = _selected_batch_rows(output_root, branch, round_id)
    review_rows = _collect_phase3_review_rows(output_root, branch, round_id, selected_rows)
    ready_count = 0
    reviewed_count = 0
    blocked_count = 0
    for row in review_rows:
        status = review_evidence_status(row)
        if str(status.get("label", "")).strip() in {"0", "1"}:
            reviewed_count += 1
        if bool(status.get("ingest_ready", False)):
            ready_count += 1
        else:
            blocked_count += 1
    ledger = load_ledger(_branch_dir(output_root, branch) / "ledger.csv") if (_branch_dir(output_root, branch) / "ledger.csv").exists() else []
    selected_sequences = {_normalize_sequence(row.get("sequence", "")) for row in selected_rows if row.get("sequence")}
    ingested_count = len(
        [
            row
            for row in ledger
            if _normalize_sequence(row.get("sequence", "")) in selected_sequences
            and row.get("status") == "acquired"
            and row.get("label_source") == "cgmd"
            and row.get("label")
        ]
    )
    ingest_status = _safe_read_json(round_dir / "ingest" / "ingest_status.json")
    status_text = str(ingest_status.get("status", "")) if isinstance(ingest_status, dict) else ""
    if ingested_count and ingested_count < len(selected_sequences):
        next_action = "finish review/ingest before next proposal"
    elif selected_sequences and ingested_count == len(selected_sequences):
        next_action = "round labels ingested"
    elif ready_count:
        next_action = "ingest labels"
    else:
        next_action = "create ingest CSV after human review"
    return {
        "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}",
        "proposal_status": detect_round_status(output_root, branch, round_id).get("status", "missing"),
        "selected_count": len(selected_sequences),
        "md_returned_count": len(review_rows),
        "reviewed_count": reviewed_count,
        "ingest_ready_count": ready_count,
        "ingested_count": ingested_count,
        "blocked_rows_count": blocked_count,
        "ingest_status": status_text or "missing",
        "ingest_csv": str(round_dir / "ingest" / "cgmd_ingest.csv"),
        "next_action": next_action,
    }


def _selected_or_proposed_rows(output_root: Path, branch: str, round_id: int) -> list[dict[str, str]]:
    round_dir = _round_dir(output_root, branch, round_id)
    for candidate in [round_dir / "selected_batch.csv", round_dir / "proposal.csv"]:
        rows = _read_csv(candidate)
        if rows:
            return rows
    ledger_path = _branch_dir(output_root, branch) / "ledger.csv"
    if not ledger_path.exists():
        return []
    selected_statuses = {"proposed", "acquired"}
    return [
        row
        for row in load_ledger(ledger_path)
        if _round_number(row.get("round_id", "")) == round_id and row.get("status") in selected_statuses
    ]


def _selected_batch_rows(output_root: Path, branch: str, round_id: int) -> list[dict[str, str]]:
    return _read_csv(_round_dir(output_root, branch, round_id) / "selected_batch.csv")


def _collect_phase3_review_rows(
    output_root: Path,
    branch: str,
    round_id: int,
    selected_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    round_dir = _round_dir(output_root, branch, round_id)
    selected_by_sequence = {
        _normalize_sequence(row.get("sequence", "")): row
        for row in selected_rows
        if row.get("sequence")
    }
    review_paths: list[Path] = []
    local_review = round_dir / "review" / "md_review.csv"
    if local_review.exists():
        review_paths.append(local_review)
    for row in selected_rows:
        campaign_dir = str(row.get("campaign_dir", "")).strip()
        if campaign_dir:
            path = Path(campaign_dir) / "md_review.csv"
            if path.exists() and path not in review_paths:
                review_paths.append(path)
    campaigns_root = round_dir / "md_campaigns"
    if campaigns_root.exists():
        for path in sorted(campaigns_root.glob("*/md_review.csv")):
            if path not in review_paths:
                review_paths.append(path)
    branch_campaigns_root = _branch_dir(output_root, branch) / "md_campaigns"
    if branch_campaigns_root.exists():
        for path in sorted(branch_campaigns_root.glob("*/md_review.csv")):
            if path not in review_paths:
                review_paths.append(path)

    rows: list[dict[str, str]] = []
    for path in review_paths:
        for raw in _read_csv(path):
            sequence = _normalize_sequence(raw.get("sequence", ""))
            if not sequence:
                continue
            selected = selected_by_sequence.get(sequence, {})
            campaign_dir = str(raw.get("campaign_dir") or selected.get("campaign_dir") or path.parent)
            campaign_name = str(raw.get("campaign_name") or selected.get("campaign_name") or Path(campaign_dir).name)
            row = {
                **selected,
                **raw,
                "sequence": sequence,
                "branch_strategy": str(raw.get("branch_strategy") or branch),
                "source_branch": str(raw.get("source_branch") or branch),
                "round_id": str(raw.get("round_id") or f"round_{round_id:03d}"),
                "source_round_id": str(raw.get("source_round_id") or f"round_{round_id:03d}"),
                "campaign_name": campaign_name,
                "campaign_dir": campaign_dir,
                "review_csv": str(path),
            }
            rows.append(row)
    return rows


def _best_review_by_sequence(review_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in review_rows:
        sequence = _normalize_sequence(row.get("sequence", ""))
        if not sequence:
            continue
        current = best.get(sequence)
        if current is None:
            best[sequence] = row
            continue
        if review_evidence_status(row).get("ingest_ready") and not review_evidence_status(current).get("ingest_ready"):
            best[sequence] = row
    return best


def _phase3_ingest_row(review: dict[str, str], branch: str, round_id: int) -> dict[str, object]:
    return {
        "sequence": _normalize_sequence(review.get("sequence", "")),
        "round_id": f"round_{round_id:03d}",
        "cgmd_label": str(review.get("cgmd_label", "")).strip(),
        "branch_strategy": branch,
        "label_confidence": str(review.get("label_confidence", "")).strip(),
        "label_rubric": str(review.get("label_rubric", "")).strip(),
        "review_notes": str(review.get("review_notes", "")).strip(),
        "reviewer": str(review.get("reviewer", "")).strip(),
        "campaign_name": str(review.get("campaign_name", "")).strip(),
        "campaign_dir": str(review.get("campaign_dir", "")).strip(),
        "review_csv": str(review.get("review_csv", "")).strip(),
        "source_branch": str(review.get("source_branch") or branch).strip(),
        "source_round_id": str(review.get("source_round_id") or f"round_{round_id:03d}").strip(),
    }


def _validate_phase3_ingest_rows(
    output_root: Path,
    branch: str,
    round_id: int,
    rows: list[dict[str, str]],
    *,
    force: bool = False,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    ready_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    selected_rows = _selected_batch_rows(output_root, branch, round_id)
    selected_sequences = {_normalize_sequence(row.get("sequence", "")) for row in selected_rows if row.get("sequence")}
    ledger_path = _branch_dir(output_root, branch) / "ledger.csv"
    ledger_rows = load_ledger(ledger_path) if ledger_path.exists() else []
    by_sequence = {_normalize_sequence(row.get("sequence", "")): row for row in ledger_rows}
    for index, row in enumerate(rows, start=2):
        sequence = _normalize_sequence(row.get("sequence", ""))
        row_blockers = []
        if not sequence:
            row_blockers.append("missing sequence")
        if sequence in seen:
            row_blockers.append("duplicate sequence in import CSV")
        seen.add(sequence)
        label = str(row.get("cgmd_label", "")).strip()
        if label not in {"0", "1"}:
            row_blockers.append("cgmd_label must be 0 or 1")
        row_branch = str(row.get("branch_strategy", "") or row.get("source_branch", "")).strip()
        if row_branch and row_branch != branch:
            row_blockers.append(f"row belongs to branch {row_branch}, not {branch}")
        source_branch = str(row.get("source_branch", "")).strip()
        if source_branch and source_branch != branch:
            row_blockers.append(f"cross-branch import is not enabled: {source_branch}")
        row_round = _round_number(row.get("round_id", "") or row.get("source_round_id", ""))
        source_round = _round_number(row.get("source_round_id", "") or row.get("round_id", ""))
        if row_round != round_id or source_round != round_id:
            row_blockers.append("row round_id/source_round_id does not match requested round")
        if sequence not in selected_sequences:
            row_blockers.append("sequence is not in this branch round selected/proposed rows")
        ledger_row = by_sequence.get(sequence)
        if ledger_row is None:
            row_blockers.append("sequence is missing from this branch ledger")
        elif ledger_row.get("status") == "acquired" and ledger_row.get("label_source") == "cgmd" and not force:
            row_blockers.append("sequence is already acquired from cgmd in this branch; use --force to repeat")
        if row_blockers:
            blockers.append({"csv_line": index, "sequence": sequence, "blockers": "; ".join(row_blockers)})
        else:
            ready_rows.append(row)
    if not rows:
        blockers.append({"csv_line": "", "sequence": "", "blockers": "import CSV has no rows"})
    return {
        "ready_rows": ready_rows,
        "blockers": blockers,
        "selected_sequences": sorted(selected_sequences),
    }


def _append_phase3_inventory_events(
    output_root: Path,
    branch: str,
    round_id: int,
    rows: list[dict[str, object]],
    *,
    event_type: str,
    event_status: str,
    notes_prefix: str,
) -> None:
    if not rows:
        return
    _ensure_inventory_files(output_root)
    events_path = output_root / "md_inventory" / "md_inventory_events.csv"
    existing = _read_csv(events_path)
    timestamp = _now_iso()
    events = list(existing)
    for row in rows:
        sequence = _normalize_sequence(row.get("sequence", ""))
        if not sequence:
            continue
        label = str(row.get("cgmd_label", "")).strip()
        reviewer = str(row.get("reviewer", "")).strip()
        rubric = str(row.get("label_rubric", "")).strip()
        confidence = str(row.get("label_confidence", "")).strip()
        notes = (
            f"{notes_prefix}; label={label}; reviewer={reviewer}; "
            f"rubric={rubric}; confidence={confidence}; review_csv={row.get('review_csv', '')}"
        )
        events.append(
            {
                "event_id": f"{timestamp}_{branch}_r{round_id:03d}_{event_type}_{sequence}",
                "sequence": sequence,
                "branch_strategy": branch,
                "round_id": f"round_{round_id:03d}",
                "event_type": event_type,
                "event_status": event_status,
                "proposal_csv": str(_round_dir(output_root, branch, round_id) / "proposal.csv"),
                "selected_batch_csv": str(_round_dir(output_root, branch, round_id) / "selected_batch.csv"),
                "campaign_name": str(row.get("campaign_name", "")),
                "campaign_dir": str(row.get("campaign_dir", "")),
                "md_profile": str(row.get("md_profile", "")),
                "timestamp": timestamp,
                "source_action_id": "",
                "notes": notes,
            }
        )
    _write_csv(events_path, events, fieldnames=INVENTORY_EVENT_FIELDS)
    _write_inventory_snapshot(output_root)


def _branch_label_summary_rows(output_root: Path, branches: Sequence[str]) -> list[dict[str, object]]:
    rows = []
    for branch in branches:
        ledger_path = _branch_dir(output_root, branch) / "ledger.csv"
        ledger = load_ledger(ledger_path) if ledger_path.exists() else []
        acquired = [row for row in ledger if row.get("status") == "acquired" and row.get("label_source") == "cgmd"]
        positives = [row for row in acquired if row.get("label") == "1"]
        rows.append(
            {
                "branch_strategy": branch,
                "acquired_labels": len(acquired),
                "positive_labels": len(positives),
                "positive_rate": (len(positives) / len(acquired)) if acquired else "",
            }
        )
    return rows


def _branch_round_metric_rows(output_root: Path, branches: Sequence[str], round_id: int) -> list[dict[str, object]]:
    rows = []
    for branch in branches:
        round_models = _branch_dir(output_root, branch) / "models" / "real_al" / f"round_{round_id:03d}"
        metrics_path = round_models / "post_ingest" / "metrics.json"
        stage = "post_ingest"
        if not metrics_path.exists():
            metrics_path = round_models / "pre_proposal" / "metrics.json"
            stage = "pre_proposal"
        payload = _safe_read_json(metrics_path)
        if isinstance(payload, dict):
            rows.append({"branch_strategy": branch, "metrics_stage": stage, "metrics_path": str(metrics_path), **payload})
        else:
            rows.append({"branch_strategy": branch, "round_id": round_id, "status": "not_available"})
    return rows


def _branch_comparison_markdown(
    branches: Sequence[str],
    overlap_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    *,
    round_id: int = 1,
) -> str:
    lines = [
        f"# Phase 3 Real AL Branch Comparison: round_{round_id:03d}",
        "",
        "This report is read-only. It preserves duplicate strategy picks as overlap evidence and does not merge branch histories.",
        "",
        "## Branches",
    ]
    for branch in branches:
        count = len([row for row in selected_rows if row.get("branch_strategy") == branch])
        lines.append(f"- `{branch}`: {count} selected peptides recorded")
    lines.extend(["", "## Pairwise Selected-Peptide Overlap", "", "| Left | Right | Overlap | Jaccard | Sequences |", "|---|---|---:|---:|---|"])
    for row in overlap_rows:
        if row.get("left_branch") == row.get("right_branch"):
            continue
        jaccard = row.get("jaccard", "")
        jaccard_text = f"{float(jaccard):.3f}" if isinstance(jaccard, float) else str(jaccard)
        lines.append(
            f"| `{row.get('left_branch', '')}` | `{row.get('right_branch', '')}` | "
            f"{row.get('overlap_count', 0)} | {jaccard_text} | {row.get('overlap_sequences', '')} |"
        )
    lines.extend(["", "## Duplicate Selections Across Branches"])
    if duplicate_rows:
        lines.extend(["", "| Sequence | Branches | Count |", "|---|---|---:|"])
        for row in duplicate_rows:
            lines.append(f"| `{row.get('sequence', '')}` | {row.get('branches', '')} | {row.get('branch_count', '')} |")
    else:
        lines.extend(["", "No duplicate round-1 selected sequences across branches are recorded yet."])
    return "\n".join(lines) + "\n"


def _duplicate_sequence_rows(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, set[str]] = {}
    selected_paths: dict[str, set[str]] = {}
    for row in selected_rows:
        sequence = str(row.get("sequence", "")).strip()
        branch = str(row.get("branch_strategy", "")).strip()
        if not sequence or not branch:
            continue
        grouped.setdefault(sequence, set()).add(branch)
        path = str(row.get("selected_batch_csv", "") or row.get("source_selected_batch_csv", "")).strip()
        if path:
            selected_paths.setdefault(sequence, set()).add(path)
    rows = []
    for sequence, branches in sorted(grouped.items()):
        if len(branches) < 2:
            continue
        rows.append(
            {
                "sequence": sequence,
                "branch_count": len(branches),
                "branches": ";".join(sorted(branches)),
                "selected_batch_paths": ";".join(sorted(selected_paths.get(sequence, set()))),
            }
        )
    return rows


def _round_selected_rows_for_comparison(
    output_root: Path,
    branches: Sequence[str],
    round_id: int,
) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    selected_rows: list[dict[str, object]] = []
    by_branch: dict[str, set[str]] = {}
    for branch in branches:
        selected_path = _round_dir(output_root, branch, round_id) / "selected_batch.csv"
        selected = _read_csv(selected_path)
        by_branch[branch] = {_normalize_sequence(row.get("sequence", "")) for row in selected if row.get("sequence")}
        for row in selected:
            selected_rows.append(
                {
                    "branch_strategy": branch,
                    "comparison_round_id": f"round_{round_id:03d}",
                    "source_selected_batch_csv": str(selected_path),
                    **row,
                }
            )
    return selected_rows, by_branch


def _overlap_rows(branches: Sequence[str], by_branch: dict[str, set[str]]) -> list[dict[str, object]]:
    overlap_rows = []
    for left in branches:
        for right in branches:
            overlap = by_branch.get(left, set()) & by_branch.get(right, set())
            union = by_branch.get(left, set()) | by_branch.get(right, set())
            overlap_rows.append(
                {
                    "left_branch": left,
                    "right_branch": right,
                    "overlap_count": len(overlap),
                    "left_count": len(by_branch.get(left, set())),
                    "right_count": len(by_branch.get(right, set())),
                    "jaccard": (len(overlap) / len(union)) if union else "",
                    "overlap_sequences": ";".join(sorted(overlap)),
                }
            )
    return overlap_rows


def _write_legacy_round_comparison_aliases(comparison_root: Path, round_root: Path) -> None:
    for name in [
        "branch_selected_sequences.csv",
        "branch_proposal_overlap.csv",
        "duplicate_sequences_across_branches.csv",
        "branch_md_status_summary.csv",
        "branch_label_summary.csv",
        "branch_round_metrics.csv",
        "branch_comparison.md",
    ]:
        source = round_root / name
        if source.exists():
            shutil.copyfile(source, comparison_root / name)


def _write_all_rounds_comparison(output_root: Path, branches: Sequence[str]) -> None:
    comparison_root = output_root / "comparison"
    rounds = _all_known_rounds(output_root)
    summary_rows: list[dict[str, object]] = []
    markdown = [
        "# Phase 3 Real AL All-Rounds Branch Comparison",
        "",
        "This summary is read-only and preserves each branch trajectory independently.",
        "",
        "| Round | Branch | Selected | Ingest Status | Proposal Status |",
        "|---|---|---:|---|---|",
    ]
    for round_id in rounds:
        for branch in branches:
            selected_count = len(_read_csv(_round_dir(output_root, branch, round_id) / "selected_batch.csv"))
            proposal_status = str(detect_round_status(output_root, branch, round_id).get("status", "missing"))
            ingest_status = str(summarize_phase3_ingest_status(output_root, branch, round_id).get("ingest_status", "missing"))
            row = {
                "round_id": f"round_{round_id:03d}",
                "round_number": round_id,
                "branch_strategy": branch,
                "selected_count": selected_count,
                "proposal_status": proposal_status,
                "ingest_status": ingest_status,
            }
            summary_rows.append(row)
            markdown.append(
                f"| round_{round_id:03d} | `{branch}` | {selected_count} | {ingest_status} | {proposal_status} |"
            )
    _write_csv(comparison_root / "all_rounds_branch_summary.csv", summary_rows)
    (comparison_root / "all_rounds_branch_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _write_all_round_metrics(output_root: Path, branches: Sequence[str]) -> None:
    rows: list[dict[str, object]] = []
    for round_id in _all_known_rounds(output_root):
        for branch in branches:
            round_models = _branch_dir(output_root, branch) / "models" / "real_al" / f"round_{round_id:03d}"
            for stage in ["pre_proposal", "post_ingest"]:
                metrics_path = round_models / stage / "metrics.json"
                payload = _safe_read_json(metrics_path)
                if isinstance(payload, dict):
                    rows.append({
                        "branch_strategy": branch, "round_number": round_id,
                        "round_id_text": f"round_{round_id:03d}",
                        "metrics_stage": stage, "metrics_path": str(metrics_path), **payload,
                    })
    _write_csv(output_root / "comparison" / "all_rounds_branch_metrics.csv", rows)


def _write_final_holdout_summary(output_root: Path, branches: Sequence[str]) -> None:
    rows: list[dict[str, object]] = []
    for branch in branches:
        metrics_path = _branch_dir(output_root, branch) / "metrics" / "final_holdout.json"
        payload = _safe_read_json(metrics_path)
        if isinstance(payload, dict):
            rows.append({"branch_strategy": branch, "metrics_path": str(metrics_path), **payload})
    _write_csv(output_root / "comparison" / "final_branch_holdout_metrics.csv", rows)


def detect_finalization_status(output_root: Path, branch: str, round_id: int) -> dict[str, object]:
    status_path = _round_dir(output_root, branch, round_id) / "finalization" / "status.json"
    payload = _safe_read_json(status_path)
    if isinstance(payload, dict):
        return {**payload, "status_path": str(status_path)}
    return {
        "status": "missing", "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}", "status_path": str(status_path),
    }


def detect_round_status(output_root: Path, branch: str, round_id: int) -> dict[str, object]:
    round_dir = _round_dir(output_root, branch, round_id)
    status_path = round_dir / "status.json"
    payload = _safe_read_json(status_path)
    if isinstance(payload, dict) and payload.get("status"):
        return {
            **payload,
            "status_path": str(status_path),
        }
    required = _required_round_outputs(round_dir)
    if all(path.exists() for path in required.values()):
        return {
            "status": "completed",
            "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}",
            "status_path": str(status_path),
            "inferred": True,
        }
    if (round_dir / "command_preview.txt").exists():
        return {
            "status": "preview_ready",
            "branch_strategy": branch,
            "round_id": f"round_{round_id:03d}",
            "status_path": str(status_path),
            "inferred": True,
        }
    return {
        "status": "missing",
        "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}",
        "status_path": str(status_path),
        "inferred": True,
    }


def summarize_phase3_branch_continuation(output_root: Path, branch: str) -> dict[str, object]:
    branch_dir = _branch_dir(output_root, branch)
    ledger_path = branch_dir / "ledger.csv"
    ledger = load_ledger(ledger_path) if ledger_path.exists() else []
    known_rounds = _known_rounds_for_branch(output_root, branch)
    latest_round = max(known_rounds, default=0)
    latest_fully_ingested = 0
    blocked_reason = ""
    for round_id in known_rounds:
        readiness = _previous_round_readiness(output_root, branch, round_id + 1)
        if readiness["ready"]:
            latest_fully_ingested = round_id
    next_round = max(1, next_real_round_id(ledger) if ledger else 1)
    if next_round > 1:
        readiness = _previous_round_readiness(output_root, branch, next_round)
        if not readiness["ready"]:
            blocked_reason = str(readiness["blocked_reason"])
    unresolved = unresolved_proposals(ledger)
    if unresolved and not blocked_reason:
        blocked_reason = f"{len(unresolved)} unresolved proposed ledger row(s)"
    if blocked_reason:
        next_action = f"resolve blockers before proposing round {next_round}"
    elif detect_round_status(output_root, branch, next_round).get("status") == "preview_ready":
        next_action = f"submit proposal job for round {next_round}"
    else:
        next_action = f"propose round {next_round}"
    acquired_cgmd = [
        row
        for row in ledger
        if row.get("status") == "acquired" and row.get("label_source") == "cgmd" and row.get("label") in {"0", "1"}
    ]
    return {
        "branch_strategy": branch,
        "known_rounds": [f"round_{round_id:03d}" for round_id in known_rounds],
        "current_round": f"round_{next_round:03d}",
        "next_round_number": next_round,
        "latest_completed_proposal_round": f"round_{latest_round:03d}" if latest_round else "",
        "latest_fully_ingested_round": f"round_{latest_fully_ingested:03d}" if latest_fully_ingested else "",
        "latest_proposal_status": detect_round_status(output_root, branch, latest_round or 1).get("status", "missing"),
        "latest_ingest_status": summarize_phase3_ingest_status(output_root, branch, latest_round or 1).get("ingest_status", "missing"),
        "labeled_count": len(current_real_training_rows(ledger)),
        "acquired_cgmd_labels_count": len(acquired_cgmd),
        "next_action": next_action,
        "blocked_reason": blocked_reason,
    }


def _write_round_status(
    output_root: Path,
    branch: str,
    round_id: int,
    status: str,
    *,
    exact_command: str = "",
    started_at: str = "",
    completed_at: str = "",
    error: str = "",
    pbs_path: Path | None = None,
    selected_batch_path: Path | None = None,
    scored_candidates_path: Path | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
) -> dict[str, object]:
    payload = _round_status_payload(
        output_root,
        branch,
        round_id,
        status,
        exact_command=exact_command,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
        pbs_path=pbs_path,
        selected_batch_path=selected_batch_path,
        scored_candidates_path=scored_candidates_path,
        candidate_count=candidate_count,
        selected_count=selected_count,
    )
    _write_json(_round_dir(output_root, branch, round_id) / "status.json", payload)
    return payload


def _round_status_payload(
    output_root: Path,
    branch: str,
    round_id: int,
    status: str,
    *,
    exact_command: str = "",
    started_at: str = "",
    completed_at: str = "",
    error: str = "",
    pbs_path: Path | None = None,
    selected_batch_path: Path | None = None,
    scored_candidates_path: Path | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
) -> dict[str, object]:
    branch_dir = _branch_dir(output_root, branch)
    round_dir = _round_dir(output_root, branch, round_id)
    config_path = branch_dir / "config.json"
    config = _safe_read_json(config_path)
    phase1_root = Path(str(config.get("phase1_root", DEFAULT_PHASE1_ROOT))) if isinstance(config, dict) else DEFAULT_PHASE1_ROOT
    selected_path = selected_batch_path or round_dir / "selected_batch.csv"
    scored_path = scored_candidates_path or round_dir / "scored_candidates.csv"
    payload: dict[str, object] = {
        "status": status,
        "branch_strategy": branch,
        "round_id": f"round_{round_id:03d}",
        "round_number": round_id,
        "exact_command": exact_command,
        "timestamp_started": started_at,
        "timestamp_completed": completed_at,
        "git_commit": _git_commit_hash(),
        "phase1_frozen_config_path": str(phase1_root / "frozen_model_config.json"),
        "branch_config_path": str(config_path),
        "current_labeled_ledger_path": str(branch_dir / "current_labeled_ledger.csv"),
        "proposal_csv": str(round_dir / "proposal.csv"),
        "selected_batch_path": str(selected_path),
        "scored_candidates_path": str(scored_path),
        "acquisition_log_path": str(round_dir / "acquisition_log.csv"),
        "candidate_count": candidate_count if candidate_count is not None else _csv_row_count(scored_path),
        "selected_count": selected_count if selected_count is not None else _csv_row_count(selected_path),
    }
    if pbs_path is not None:
        payload["pbs_path"] = str(pbs_path)
    if error:
        payload["error"] = error
    return payload


def _round_config_payload(
    output_root: Path,
    branch: str,
    round_id: int,
    status_payload: dict[str, object],
) -> dict[str, object]:
    config_path = _branch_dir(output_root, branch) / "config.json"
    config = _safe_read_json(config_path)
    payload = dict(config) if isinstance(config, dict) else {}
    payload["phase3_round_metadata"] = {
        key: status_payload.get(key, "")
        for key in [
            "exact_command",
            "branch_strategy",
            "round_id",
            "timestamp_started",
            "timestamp_completed",
            "git_commit",
            "phase1_frozen_config_path",
            "branch_config_path",
            "current_labeled_ledger_path",
            "selected_batch_path",
            "scored_candidates_path",
            "candidate_count",
            "selected_count",
        ]
    }
    return payload


def _validate_round_outputs(output_root: Path, branch: str, round_id: int) -> None:
    round_dir = _round_dir(output_root, branch, round_id)
    missing = [name for name, path in _required_round_outputs(round_dir).items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Round {round_id} for {branch} is missing required outputs: {', '.join(missing)}"
        )


def _required_round_outputs(round_dir: Path) -> dict[str, Path]:
    return {
        "proposal.csv": round_dir / "proposal.csv",
        "scored_candidates.csv": round_dir / "scored_candidates.csv",
        "selected_batch.csv": round_dir / "selected_batch.csv",
        "acquisition_log.csv": round_dir / "acquisition_log.csv",
        "config.json": round_dir / "config.json",
        "status.json": round_dir / "status.json",
    }


def _validate_round_id(round_id: int) -> None:
    if round_id < 1:
        raise ValueError(f"Phase 3 round must be >= 1, got {round_id}.")


def _validate_phase3_proposal_ready(output_root: Path, branch: str, round_id: int) -> None:
    branch_dir = _branch_dir(output_root, branch)
    ledger_path = branch_dir / "ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 branch ledger: {ledger_path}")
    ledger = load_ledger(ledger_path)
    unresolved = unresolved_proposals(ledger)
    if unresolved:
        raise ValueError(
            f"Cannot propose round {round_id} for {branch}: "
            f"{len(unresolved)} unresolved proposed ledger row(s) remain."
        )
    expected_round = next_real_round_id(ledger)
    if expected_round != round_id:
        raise ValueError(
            f"Requested Phase 3 round {round_id} for {branch}, but branch ledger next round is {expected_round}."
        )
    if round_id == 1:
        return
    readiness = _previous_round_readiness(output_root, branch, round_id)
    if not readiness["ready"]:
        raise ValueError(
            f"Cannot propose round {round_id} for {branch}: {readiness['blocked_reason']}"
        )


def _previous_round_readiness(output_root: Path, branch: str, requested_round: int) -> dict[str, object]:
    previous_round = requested_round - 1
    status = detect_round_status(output_root, branch, previous_round)
    status_text = str(status.get("status", "missing"))
    if status_text != "ingested":
        return {
            "ready": False,
            "blocked_reason": f"previous round round_{previous_round:03d} status is {status_text}, not ingested",
        }
    selected_path = _round_dir(output_root, branch, previous_round) / "selected_batch.csv"
    selected_rows = _read_csv(selected_path)
    if not selected_rows:
        return {
            "ready": False,
            "blocked_reason": f"previous round round_{previous_round:03d} has no selected_batch.csv rows",
        }
    ledger_path = _branch_dir(output_root, branch) / "ledger.csv"
    ledger_rows = load_ledger(ledger_path) if ledger_path.exists() else []
    by_sequence = {_normalize_sequence(row.get("sequence", "")): row for row in ledger_rows}
    blockers = []
    for row in selected_rows:
        sequence = _normalize_sequence(row.get("sequence", ""))
        ledger_row = by_sequence.get(sequence)
        if ledger_row is None:
            blockers.append(f"{sequence}: missing from branch ledger")
            continue
        if _round_number(ledger_row.get("round_id", "")) != previous_round:
            blockers.append(f"{sequence}: ledger round_id={ledger_row.get('round_id', '')}")
            continue
        if _is_selected_row_complete(ledger_row):
            continue
        blockers.append(
            f"{sequence}: status={ledger_row.get('status', '') or '-'} label={ledger_row.get('label', '') or '-'}"
        )
    if blockers:
        return {
            "ready": False,
            "blocked_reason": (
                f"previous round round_{previous_round:03d} selected rows are not fully ingested: "
                + "; ".join(blockers[:8])
            ),
        }
    return {"ready": True, "blocked_reason": ""}


def _is_selected_row_complete(ledger_row: dict[str, str]) -> bool:
    return ledger_row.get("status") == "acquired" and ledger_row.get("label") in {"0", "1"}


def _validate_batch_round(batch_path: Path, round_id: int) -> None:
    _validate_selected_rows_round(_read_csv(batch_path), round_id, batch_path)


def _validate_selected_rows_round(rows: list[dict[str, str]], round_id: int, path: Path) -> None:
    mismatched = [
        str(row.get("sequence", ""))
        for row in rows
        if _round_number(row.get("round_id", "")) != round_id
    ]
    if mismatched:
        raise ValueError(
            f"Selected batch {path} contains rows that do not match requested round {round_id}: "
            + ", ".join(mismatched[:8])
        )


def _validate_scored_rows_round(path: Path, round_id: int) -> None:
    rows = _read_csv(path)
    mismatched = [
        str(row.get("sequence", ""))
        for row in rows
        if row.get("round_id") and _round_number(row.get("round_id", "")) != round_id
    ]
    if mismatched:
        raise ValueError(
            f"Scored candidates {path} contain rows that do not match requested round {round_id}: "
            + ", ".join(mismatched[:8])
        )


def _known_rounds_for_branch(output_root: Path, branch: str) -> list[int]:
    rounds_root = _branch_dir(output_root, branch) / "rounds"
    found = set()
    if rounds_root.exists():
        for path in rounds_root.iterdir():
            if path.is_dir():
                round_id = _round_number(path.name)
                if round_id > 0:
                    found.add(round_id)
    ledger_path = _branch_dir(output_root, branch) / "ledger.csv"
    if ledger_path.exists():
        for row in load_ledger(ledger_path):
            round_id = _round_number(row.get("round_id", ""))
            if round_id > 0 and row.get("mode") == "real_al":
                found.add(round_id)
    return sorted(found)


def _latest_known_round(output_root: Path, branch: str) -> int:
    return max(_known_rounds_for_branch(output_root, branch), default=0)


def _all_known_rounds(output_root: Path) -> list[int]:
    found = set()
    for branch in _branch_names(output_root):
        found.update(_known_rounds_for_branch(output_root, branch))
    return sorted(found) or [1]


def _supek_resources(
    output_root: Path,
    branch: str,
    *,
    args: argparse.Namespace | None = None,
    read_config: bool = True,
) -> SupekResources:
    walltime = DEFAULT_SUPEK_WALLTIME.get(branch, "05:00:00")
    queue = "gpu"
    ncpus = 4
    ngpus = 1
    mem = "40GB"
    config_path = _branch_dir(output_root, branch) / "config.json"
    if read_config and config_path.exists():
        config = _safe_read_json(config_path)
        if isinstance(config, dict):
            walltime = str(config.get("supek_walltime") or walltime)
            queue = str(config.get("supek_queue") or queue)
            ncpus = int(config.get("supek_ncpus") or ncpus)
            ngpus = int(config.get("supek_ngpus") or ngpus)
            mem = str(config.get("supek_mem") or mem)
    if args is not None:
        walltime = str(getattr(args, "supek_walltime", None) or walltime)
        queue = str(getattr(args, "supek_queue", None) or queue)
        ncpus = int(getattr(args, "supek_ncpus", None) or ncpus)
        ngpus = int(getattr(args, "supek_ngpus", None) or ngpus)
        mem = str(getattr(args, "supek_mem", None) or mem)
    return SupekResources(queue=queue, ncpus=ncpus, ngpus=ngpus, mem=mem, walltime=walltime)


def _pbs_text(job_name: str, command: str, log_dir: Path, repo_root: Path, resources: SupekResources) -> str:
    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q {resources.queue}
#PBS -l select=1:ncpus={resources.ncpus}:ngpus={resources.ngpus}:mem={resources.mem}
#PBS -l walltime={resources.walltime}
#PBS -o {log_dir.as_posix()}/{job_name}.out
#PBS -e {log_dir.as_posix()}/{job_name}.err

set -eo pipefail
cd "{repo_root.as_posix()}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
echo "[phase3] start $(date -Is) host=$(hostname)"
echo "[phase3] workdir=$(pwd)"
echo "[phase3] resources queue={resources.queue} ncpus={resources.ncpus} ngpus={resources.ngpus} mem={resources.mem} walltime={resources.walltime}"
echo "[phase3] command: {command}"
{command}
echo "[phase3] end $(date -Is)"
"""


def _supek_preview_text(
    command: str,
    pbs_path: Path,
    repo_root: Path,
    log_dir: Path,
    resources: SupekResources,
) -> str:
    return "\n".join(
        [
            "Phase 3 SUPEK proposal preview",
            f"PBS: {pbs_path}",
            f"Repository root: {repo_root}",
            f"Log directory: {log_dir}",
            f"Queue: {resources.queue}",
            f"Walltime: {resources.walltime}",
            f"CPUs: {resources.ncpus}",
            f"GPUs: {resources.ngpus}",
            f"Memory: {resources.mem}",
            f"Command: {command}",
            "No qsub was run by this command.",
            "",
        ]
    )


def _proposal_command_preview(output_root: Path, branch: str, round_id: int, selected_path: Path) -> str:
    return "\n".join(
        [
            "Phase 3 Real AL round proposal completed",
            f"Branch: {branch}",
            f"Round: round_{round_id:03d}",
            f"Selected batch: {selected_path}",
            f"Shared MD inventory: {output_root / 'md_inventory' / 'md_inventory.csv'}",
            "Duplicate MD submissions should be reviewed against the shared inventory before BURA staging.",
            "",
        ]
    )


def _campaign_name(branch: str, round_id: int, sequence: str) -> str:
    short_branch = "cluster_diverse" if branch == "cluster_diverse_representative" else branch
    return f"phase3_{short_branch}_r{round_id:03d}_{sequence}"


def _job_name(branch: str, round_id: int) -> str:
    codes = {
        "predictive_entropy": "p3_pe",
        "family_qbc": "p3_fqbc",
        "cluster_diverse_representative": "p3_cdr",
    }
    return f"{codes.get(branch, 'p3_' + branch[:8])}_r{round_id:03d}"


def _archive_existing_branch(output_root: Path, branch_dir: Path) -> None:
    archive_root = output_root / "logs" / "archived_branches"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{branch_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(branch_dir), str(destination))


def _branch_names(output_root: Path) -> list[str]:
    branches_root = output_root / "branches"
    if not branches_root.exists():
        return []
    return [
        path.name
        for path in sorted(branches_root.iterdir())
        if path.is_dir() and (path / "config.json").exists()
    ]


def _branch_dir(output_root: Path, branch: str) -> Path:
    return output_root / "branches" / branch


def _round_dir(output_root: Path, branch: str, round_id: int) -> Path:
    return _branch_dir(output_root, branch) / "rounds" / f"round_{round_id:03d}"


def _pbs_repo_root(args: argparse.Namespace | None) -> Path:
    value = getattr(args, "pbs_repo_root", None) if args is not None else None
    return Path(value).resolve() if value else Path.cwd().resolve()


def _target_path(path: Path, repo_root: Path | None) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve() if repo_root else (Path.cwd() / path).resolve()


def _proposal_cli_command(output_root: Path, branch: str, round_id: int) -> str:
    return (
        "python -m active_learning_thesis phase3-real-al propose "
        f"--output-root {shlex.quote(output_root.as_posix())} "
        f"--branch {shlex.quote(branch)} --round {round_id}"
    )


def _finalize_cli_command(
    output_root: Path,
    branch: str,
    round_id: int,
    evaluate_holdout_flag: bool,
) -> str:
    command = (
        "python -m active_learning_thesis phase3-real-al finalize "
        f"--output-root {shlex.quote(output_root.as_posix())} "
        f"--branch {shlex.quote(branch)} --round {round_id}"
    )
    if evaluate_holdout_flag:
        command += " --evaluate-holdout"
    return command


def _exact_or_constructed_command(output_root: Path, branch: str, round_id: int) -> str:
    argv = list(sys.argv)
    if "phase3-real-al" in argv and "propose" in argv:
        return " ".join(shlex.quote(item) for item in argv)
    return _proposal_cli_command(output_root, branch, round_id)


def _git_commit_hash() -> str:
    cwd = Path.cwd().resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={cwd.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _csv_row_count(path: Path) -> int:
    return len(_read_csv(path))


def _normalize_sequence(value: object) -> str:
    return str(value or "").strip().upper()


def _round_number(value: object) -> int:
    text = str(value or "").strip().lower()
    if not text:
        return -1
    if text.startswith("round_"):
        text = text.replace("round_", "", 1)
    try:
        return int(text)
    except ValueError:
        return -1


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_read_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
