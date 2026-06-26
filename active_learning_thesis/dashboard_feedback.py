from __future__ import annotations

import csv
import json
from pathlib import Path

from active_learning_thesis.ledger import load_ledger, unresolved_proposals
from active_learning_thesis.md_workflow import make_md_ingest_csv


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pending_round_summary(run_dir: Path) -> dict[str, object]:
    ledger_path = run_dir / "ledger.csv"
    if not ledger_path.exists():
        return {
            "status": "missing-ledger",
            "pending_round_id": "",
            "pending_sequences": [],
            "summary": "This run has no ledger yet, so there is no active-learning feedback queue to continue.",
        }
    proposed_rows = unresolved_proposals(load_ledger(ledger_path))
    if not proposed_rows:
        return {
            "status": "no-pending-batch",
            "pending_round_id": "",
            "pending_sequences": [],
            "summary": "There is no proposed AL batch currently waiting for MD feedback.",
        }
    round_ids = sorted({str(row.get("round_id", "")).strip() for row in proposed_rows if str(row.get("round_id", "")).strip()})
    if len(round_ids) != 1:
        return {
            "status": "invalid-pending-batch",
            "pending_round_id": "",
            "pending_sequences": sorted(str(row.get("sequence", "")).strip() for row in proposed_rows if str(row.get("sequence", "")).strip()),
            "summary": "Multiple proposed AL rounds are pending at once. Resolve them one round at a time before continuing the feedback loop.",
        }
    round_id = round_ids[0]
    return {
        "status": "pending",
        "pending_round_id": round_id,
        "pending_sequences": sorted(str(row.get("sequence", "")).strip() for row in proposed_rows if str(row.get("sequence", "")).strip()),
        "summary": f"Round {round_id} is waiting for MD feedback.",
        "pending_batch_csv": str(run_dir / "batches" / f"round_{int(round_id):03d}_batch.csv") if round_id.isdigit() else "",
    }


def _full_campaign_rows(run_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_sequence: dict[str, list[dict[str, str]]] = {}
    campaigns_root = run_dir / "md_campaigns"
    if not campaigns_root.exists():
        return rows_by_sequence
    for campaign_dir in sorted(path for path in campaigns_root.iterdir() if path.is_dir()):
        review_path = campaign_dir / "md_review.csv"
        if not review_path.exists():
            continue
        meta = _load_json(campaign_dir / "md_stage_meta.json")
        meta_profile = str(meta.get("md_profile", "")).strip()
        meta_source_batch = str(meta.get("source_batch_csv", "")).strip()
        for row in _read_csv(review_path):
            md_profile = str(row.get("md_profile", "")).strip() or meta_profile
            if md_profile != "full":
                continue
            sequence = str(row.get("sequence", "")).strip()
            if not sequence:
                continue
            rows_by_sequence.setdefault(sequence, []).append(
                {
                    "sequence": sequence,
                    "campaign": campaign_dir.name,
                    "campaign_dir": str(campaign_dir),
                    "review_csv": str(review_path),
                    "round_id": str(row.get("round_id", "")).strip(),
                    "job_root_status": str(row.get("job_root_status", "")).strip(),
                    "cgmd_label": str(row.get("cgmd_label", "")).strip(),
                    "review_notes": str(row.get("review_notes", "")).strip(),
                    "source_batch_csv": meta_source_batch,
                    "promoted_to_real_batch_at": str(meta.get("promoted_to_real_batch_at", "")).strip(),
                    "promoted_round_id": str(meta.get("promoted_round_id", "")).strip(),
                    "mtime": f"{campaign_dir.stat().st_mtime:.6f}",
                }
            )
    for rows in rows_by_sequence.values():
        rows.sort(key=lambda row: (str(row.get("promoted_to_real_batch_at", "")), str(row.get("mtime", ""))), reverse=True)
    return rows_by_sequence


def build_feedback_queue(run_dir: Path) -> dict[str, object]:
    pending = _pending_round_summary(run_dir)
    status = str(pending.get("status", ""))
    if status != "pending":
        return {
            **pending,
            "can_continue": False,
            "ready_campaigns": [],
            "blocked_rows": [],
            "ready_count": 0,
            "blocked_count": 0,
            "all_ingest_csvs_present": False,
        }

    pending_round_id = str(pending.get("pending_round_id", "")).strip()
    pending_sequences = list(pending.get("pending_sequences", []))
    campaign_rows = _full_campaign_rows(run_dir)
    ready_campaigns: list[dict[str, str]] = []
    blocked_rows: list[dict[str, str]] = []
    all_ingest_csvs_present = True

    for sequence in pending_sequences:
        candidates = campaign_rows.get(sequence, [])
        ready = next(
            (
                row
                for row in candidates
                if str(row.get("round_id", "")).strip() == pending_round_id
                and str(row.get("job_root_status", "")).strip() == "analysis_complete"
                and str(row.get("cgmd_label", "")).strip() in {"0", "1"}
            ),
            None,
        )
        if ready is not None:
            ingest_csv = Path(str(ready["campaign_dir"])) / "cgmd_ingest.csv"
            ready_campaigns.append(
                {
                    **ready,
                    "ingest_csv": str(ingest_csv),
                    "ingest_csv_exists": "yes" if ingest_csv.exists() else "no",
                }
            )
            if not ingest_csv.exists():
                all_ingest_csvs_present = False
            continue

        latest = candidates[0] if candidates else None
        if latest is None:
            blocker = "No full-analysis campaign exists for this peptide yet."
        elif str(latest.get("job_root_status", "")).strip() != "analysis_complete":
            blocker = "Full analysis is not complete yet."
        elif str(latest.get("cgmd_label", "")).strip() not in {"0", "1"}:
            blocker = "Human review label is still missing."
        elif str(latest.get("round_id", "")).strip() != pending_round_id:
            blocker = (
                f"The latest reviewed campaign still points at round {latest.get('round_id', '-')}. "
                "Use the promotion bridge before continuing AL."
            )
        else:
            blocker = "This peptide is not yet ready for model feedback."
        blocked_rows.append(
            {
                "sequence": sequence,
                "campaign": str(latest.get("campaign", "")) if latest else "-",
                "state": str(latest.get("job_root_status", "")) if latest else "missing",
                "label": str(latest.get("cgmd_label", "")) if latest else "-",
                "blocker": blocker,
            }
        )

    ready_count = len(ready_campaigns)
    blocked_count = len(blocked_rows)
    can_continue = bool(pending_sequences) and ready_count == len(pending_sequences) and blocked_count == 0
    if can_continue:
        summary = (
            f"Round {pending_round_id} is fully reviewed for all {len(pending_sequences)} proposed peptide(s) "
            "and can continue back into active learning now."
        )
        queue_status = "ready"
    else:
        summary = (
            f"Round {pending_round_id} has {ready_count}/{len(pending_sequences)} proposed peptide(s) ready for model feedback. "
            f"{blocked_count} still need review, promotion, or finished full-analysis outputs."
        )
        queue_status = "blocked"
    return {
        **pending,
        "status": queue_status,
        "can_continue": can_continue,
        "ready_campaigns": ready_campaigns,
        "blocked_rows": blocked_rows,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "all_ingest_csvs_present": all_ingest_csvs_present if can_continue else False,
        "summary": summary,
    }


def run_feedback_loop(run_dir: Path, *, propose_next_batch: bool = False) -> dict[str, object]:
    feedback_queue = build_feedback_queue(run_dir)
    if not bool(feedback_queue.get("can_continue", False)):
        raise ValueError(str(feedback_queue.get("summary", "The active-learning feedback loop is not ready yet.")))

    pending_round_id = str(feedback_queue.get("pending_round_id", "")).strip()
    pending_sequences = [str(item) for item in feedback_queue.get("pending_sequences", [])]
    ready_campaigns = list(feedback_queue.get("ready_campaigns", []))
    aggregate_rows: list[dict[str, str]] = []
    synced_ingest_csvs: list[str] = []

    for campaign in ready_campaigns:
        campaign_dir = Path(str(campaign.get("campaign_dir", "")))
        review_csv = Path(str(campaign.get("review_csv", "")))
        ingest_path = make_md_ingest_csv(campaign_dir, review_csv)
        synced_ingest_csvs.append(str(ingest_path))
        aggregate_rows.append(
            {
                "sequence": str(campaign.get("sequence", "")).strip(),
                "round_id": pending_round_id,
                "cgmd_label": str(campaign.get("cgmd_label", "")).strip(),
            }
        )

    aggregate_rows.sort(key=lambda row: str(row.get("sequence", "")))
    seen = {str(row.get("sequence", "")) for row in aggregate_rows}
    missing = sorted(set(pending_sequences) - seen)
    if missing:
        raise ValueError("The feedback runner could not build a complete import CSV. Missing sequences: " + ", ".join(missing))

    aggregate_import_csv = run_dir / "imports" / f"round_{int(pending_round_id):03d}_dashboard_feedback.csv"
    _write_csv(aggregate_import_csv, ["sequence", "round_id", "cgmd_label"], aggregate_rows)

    from active_learning_thesis.workflow import ingest_round, propose_round

    metrics = ingest_round(run_dir, aggregate_import_csv)
    next_batch_csv = ""
    if propose_next_batch:
        next_batch_csv = str(propose_round(run_dir))
    return {
        "pending_round_id": pending_round_id,
        "pending_sequences": pending_sequences,
        "synced_ingest_csvs": synced_ingest_csvs,
        "aggregate_import_csv": str(aggregate_import_csv),
        "metrics": metrics,
        "next_batch_csv": next_batch_csv,
        "proposed_next_batch": propose_next_batch,
    }
