from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from active_learning_thesis.config import RunConfig
from active_learning_thesis.ledger import (
    current_real_training_rows,
    holdout_rows,
    load_ledger,
    unresolved_proposals,
)
from active_learning_thesis.md_review_evidence import review_evidence_status
from active_learning_thesis.workflow import evaluate_final

FREEZE_DIRNAME = "final_freeze"
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_FIELDS = ["check_id", "status", "severity", "message", "evidence"]
MODEL_MANIFEST_FIELDS = ["path", "relative_path", "size_bytes", "sha256"]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_read_json(path: Path) -> dict[str, object] | list[object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]
    except Exception:
        return []


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _add_check(
    checks: list[dict[str, object]],
    check_id: str,
    *,
    passed: bool,
    severity: str,
    message: str,
    evidence: object = "",
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "message": message,
            "evidence": json.dumps(evidence, sort_keys=True) if isinstance(evidence, (dict, list)) else str(evidence),
        }
    )


def _git_metadata() -> dict[str, object]:
    def run_git(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return ""
        return (result.stdout or result.stderr).strip()

    status = run_git(["status", "--short"])
    return {
        "repo_root": str(REPO_ROOT),
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": status,
        "dirty": bool(status.strip()),
    }


def _resolve_model_source(run_dir: Path, final_metrics: Mapping[str, object]) -> Path | None:
    stage = str(final_metrics.get("surrogate_stage", "") or "").strip()
    round_value = final_metrics.get("surrogate_round_id", final_metrics.get("round_id", ""))
    try:
        round_id = int(round_value)
    except (TypeError, ValueError):
        return None
    if stage == "baseline":
        return run_dir / "models" / "real_al" / "round_000" / "baseline" / "ensemble"
    if stage:
        return run_dir / "models" / "real_al" / f"round_{round_id:03d}" / stage / "ensemble"
    return None


def _model_manifest(model_dir: Path | None, run_dir: Path) -> tuple[list[dict[str, object]], str]:
    if model_dir is None or not model_dir.exists():
        return [], ""
    rows: list[dict[str, object]] = []
    manifest_digest = hashlib.sha256()
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        file_hash = _sha256_file(path)
        relative_path = _relative(path, run_dir)
        size_bytes = path.stat().st_size
        rows.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": file_hash,
            }
        )
        manifest_digest.update(f"{relative_path}\t{size_bytes}\t{file_hash}\n".encode("utf-8"))
    return rows, manifest_digest.hexdigest() if rows else ""


def _import_hashes(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((run_dir / "imports").glob("*.csv")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path),
                "relative_path": _relative(path, run_dir),
                "sha256": _sha256_file(path),
                "rows": len(_safe_read_csv(path)),
            }
        )
    return rows


def _scan_md_reviews(run_dir: Path, ledger_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ledger_by_sequence = {str(row.get("sequence", "")): row for row in ledger_rows}
    review_rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    for review_csv in sorted((run_dir / "md_campaigns").glob("*/md_review.csv")):
        campaign_dir = review_csv.parent
        ingest_csv = campaign_dir / "cgmd_ingest.csv"
        for index, row in enumerate(_safe_read_csv(review_csv), start=1):
            sequence = str(row.get("sequence", "") or "")
            status = review_evidence_status(row)
            md_profile = str(row.get("md_profile", "") or "")
            job_status = str(row.get("job_root_status", "") or "")
            ledger_row = ledger_by_sequence.get(sequence, {})
            ingested = (
                str(ledger_row.get("status", "")) == "acquired"
                and str(ledger_row.get("label_source", "")) == "cgmd"
                and str(ledger_row.get("label", "")) in {"0", "1"}
            )
            terminal_review = md_profile == "full" or job_status == "analysis_complete" or bool(str(row.get("cgmd_label", "")).strip())
            review_rows.append(
                {
                    "sequence": sequence,
                    "campaign": campaign_dir.name,
                    "review_csv": str(review_csv),
                    "row_number": index,
                    "md_profile": md_profile,
                    "job_root_status": job_status,
                    "evidence_state": status.get("state", ""),
                    "ingest_ready": bool(status.get("ingest_ready")),
                    "ledger_status": ledger_row.get("status", ""),
                    "ledger_label_source": ledger_row.get("label_source", ""),
                    "ingested": ingested,
                    "ingest_csv_exists": ingest_csv.exists(),
                }
            )
            if not terminal_review:
                continue
            if not status.get("ingest_ready"):
                issues.append(
                    {
                        "sequence": sequence,
                        "campaign": campaign_dir.name,
                        "reason": "terminal MD review is not evidence-backed",
                        "state": status.get("state", ""),
                        "missing": status.get("missing", []),
                        "blockers": status.get("blockers", []),
                    }
                )
                continue
            if not ingested and not ingest_csv.exists():
                issues.append(
                    {
                        "sequence": sequence,
                        "campaign": campaign_dir.name,
                        "reason": "evidence-backed MD label has not been converted to cgmd_ingest.csv or acquired in the ledger",
                    }
                )
    return review_rows, issues


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, object]:
    hashes: dict[str, object] = {}
    for name, path in paths.items():
        hashes[name] = _sha256_file(path) if path.exists() and path.is_file() else ""
    return hashes


def _write_model_card(path: Path, report: Mapping[str, object]) -> None:
    config = report.get("config", {}) if isinstance(report.get("config", {}), dict) else {}
    final_metrics = report.get("final_metrics", {}) if isinstance(report.get("final_metrics", {}), dict) else {}
    counts = report.get("counts", {}) if isinstance(report.get("counts", {}), dict) else {}
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
    status = str(report.get("status", ""))
    metric_lines = []
    for metric in ["f1", "pr_auc", "roc_auc", "balanced_accuracy", "brier_score", "log_loss", "ece_10"]:
        if metric in final_metrics:
            metric_lines.append(f"- `{metric}`: `{final_metrics.get(metric)}`")
    if not metric_lines:
        metric_lines.append("- No final metric values were available.")

    lines = [
        "# Thesis Final Model Card",
        "",
        f"- Freeze status: `{status}`",
        f"- Frozen at: `{report.get('frozen_at', '')}`",
        f"- Run name: `{config.get('run_name', report.get('run_name', ''))}`",
        f"- Run directory: `{report.get('run_dir', '')}`",
        f"- Random seed: `{config.get('random_seed', '')}`",
        f"- Real AL strategy: `{config.get('real_strategy', '')}`",
        f"- Acquisition calibration: `{config.get('use_calibrated_acquisition', '')}`",
        f"- Final evaluation dataset: `{final_metrics.get('evaluation_dataset', '')}`",
        f"- Surrogate source: `{final_metrics.get('surrogate_stage', '')}` round `{final_metrics.get('surrogate_round_id', final_metrics.get('round_id', ''))}`",
        "",
        "## Final Metrics",
        "",
        *metric_lines,
        "",
        "## Training And Feedback Evidence",
        "",
        f"- Ledger rows: `{counts.get('ledger_rows', 0)}`",
        f"- Real training rows: `{counts.get('real_training_rows', 0)}`",
        f"- Holdout rows: `{counts.get('holdout_rows', 0)}`",
        f"- Acquired CG-MD rows: `{counts.get('acquired_cgmd_rows', 0)}`",
        f"- Pending proposed rows: `{counts.get('pending_proposals', 0)}`",
        f"- Terminal MD review issues: `{counts.get('md_review_issues', 0)}`",
        "",
        "## Intended Use",
        "",
        "This freeze is intended as the official local evidence checkpoint for thesis reporting. It preserves the final holdout metrics, ledger state, model artifact fingerprint, imported label fingerprints, and review-evidence checks.",
        "",
        "## Limitations",
        "",
        "The freeze verifies local workflow consistency and artifact reproducibility. It does not independently validate MD physical correctness, experimental truth, or whether the held-out dataset is representative beyond the run configuration.",
        "",
        "## Reproducibility Files",
        "",
        f"- Freeze manifest: `{outputs.get('freeze_json', '')}`",
        f"- Freeze checks: `{outputs.get('checks_csv', '')}`",
        f"- Model manifest: `{outputs.get('model_manifest_csv', '')}`",
        f"- Model card: `{outputs.get('model_card', '')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def freeze_final_result(
    run_dir: Path,
    *,
    run_evaluation: bool = False,
    force: bool = False,
    allow_unresolved: bool = False,
    metric: str = "f1",
) -> dict[str, object]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    freeze_dir = run_dir / FREEZE_DIRNAME
    outputs = {
        "freeze_json": str(freeze_dir / "final_freeze.json"),
        "model_card": str(freeze_dir / "model_card.md"),
        "checks_csv": str(freeze_dir / "freeze_checks.csv"),
        "model_manifest_csv": str(freeze_dir / "model_artifacts.csv"),
    }
    freeze_json = Path(outputs["freeze_json"])
    if freeze_json.exists() and not force:
        raise FileExistsError(f"Final freeze already exists: {freeze_json}. Use --force to replace it.")

    checks: list[dict[str, object]] = []
    config_path = run_dir / "config.json"
    ledger_path = run_dir / "ledger.csv"
    metrics_path = run_dir / "metrics" / "final_holdout.json"

    config: RunConfig | None = None
    try:
        config = RunConfig.load(config_path)
        _add_check(checks, "config_loads", passed=True, severity="blocker", message="Run configuration loaded.", evidence=str(config_path))
    except Exception as exc:
        _add_check(checks, "config_loads", passed=False, severity="blocker", message="Run configuration could not be loaded.", evidence=str(exc))

    ledger_rows: list[dict[str, str]] = []
    try:
        ledger_rows = load_ledger(ledger_path)
        _add_check(
            checks,
            "ledger_loads",
            passed=True,
            severity="blocker",
            message="Ledger loaded.",
            evidence={"path": str(ledger_path), "rows": len(ledger_rows)},
        )
    except Exception as exc:
        _add_check(checks, "ledger_loads", passed=False, severity="blocker", message="Ledger could not be loaded.", evidence=str(exc))

    final_metrics: dict[str, object] = {}
    if run_evaluation:
        try:
            final_metrics = evaluate_final(run_dir)
            _add_check(checks, "final_evaluation_runs", passed=True, severity="blocker", message="Final holdout evaluation completed.", evidence=str(metrics_path))
        except Exception as exc:
            _add_check(checks, "final_evaluation_runs", passed=False, severity="blocker", message="Final holdout evaluation failed.", evidence=str(exc))
    payload = _safe_read_json(metrics_path)
    if isinstance(payload, dict):
        final_metrics = payload
        _add_check(checks, "final_metrics_exist", passed=True, severity="blocker", message="Final holdout metrics found.", evidence=str(metrics_path))
    else:
        _add_check(
            checks,
            "final_metrics_exist",
            passed=False,
            severity="blocker",
            message="Final holdout metrics are missing. Run evaluate-final first or use --run-evaluation.",
            evidence=str(metrics_path),
        )

    _add_check(
        checks,
        "final_metrics_are_holdout",
        passed=str(final_metrics.get("evaluation_dataset", "")) == "holdout",
        severity="blocker",
        message="Final metrics are explicitly marked as holdout evaluation.",
        evidence=final_metrics.get("evaluation_dataset", ""),
    )
    _add_check(
        checks,
        "focus_metric_present",
        passed=metric in final_metrics,
        severity="warning",
        message=f"Focus metric `{metric}` is present in final metrics.",
        evidence=final_metrics.get(metric, ""),
    )

    pending = unresolved_proposals(ledger_rows) if ledger_rows else []
    _add_check(
        checks,
        "no_pending_proposals",
        passed=not pending,
        severity="blocker",
        message="No proposed CG-MD rows remain unresolved in the ledger.",
        evidence=[row.get("sequence", "") for row in pending],
    )

    training_rows = current_real_training_rows(ledger_rows) if ledger_rows else []
    holdout = holdout_rows(ledger_rows) if ledger_rows else []
    _add_check(
        checks,
        "holdout_rows_available",
        passed=bool(holdout),
        severity="blocker",
        message="Holdout rows are available for final evaluation.",
        evidence=len(holdout),
    )
    _add_check(
        checks,
        "real_training_rows_available",
        passed=bool(training_rows),
        severity="warning",
        message="Real training rows are available for interpreting the frozen model.",
        evidence=len(training_rows),
    )

    md_review_rows, md_review_issues = _scan_md_reviews(run_dir, ledger_rows)
    _add_check(
        checks,
        "md_reviews_resolved",
        passed=not md_review_issues,
        severity="blocker",
        message="Terminal MD review rows are evidence-backed and either ingested or converted to ingest CSV.",
        evidence=md_review_issues,
    )

    model_dir = _resolve_model_source(run_dir, final_metrics)
    model_rows, model_manifest_hash = _model_manifest(model_dir, run_dir)
    _add_check(
        checks,
        "model_artifacts_fingerprinted",
        passed=bool(model_rows),
        severity="warning",
        message="Model source artifacts were fingerprinted.",
        evidence=str(model_dir or ""),
    )
    import_rows = _import_hashes(run_dir)

    failed_blockers = [check for check in checks if check["severity"] == "blocker" and check["status"] == "fail"]
    failed_warnings = [check for check in checks if check["severity"] == "warning" and check["status"] == "fail"]
    if failed_blockers:
        status = "frozen_with_unresolved" if allow_unresolved else "blocked"
    elif failed_warnings:
        status = "frozen_with_warnings"
    else:
        status = "frozen"

    counts = {
        "ledger_rows": len(ledger_rows),
        "real_training_rows": len(training_rows),
        "holdout_rows": len(holdout),
        "acquired_cgmd_rows": sum(1 for row in ledger_rows if row.get("status") == "acquired" and row.get("label_source") == "cgmd"),
        "pending_proposals": len(pending),
        "md_review_rows": len(md_review_rows),
        "md_review_issues": len(md_review_issues),
        "model_artifacts": len(model_rows),
        "import_csvs": len(import_rows),
        "failed_blockers": len(failed_blockers),
        "failed_warnings": len(failed_warnings),
    }
    paths_to_hash = {
        "config": config_path,
        "ledger": ledger_path,
        "final_metrics": metrics_path,
    }
    config_dict = config.to_dict() if config else {}
    report: dict[str, object] = {
        "status": status,
        "frozen_at": _now_iso(),
        "run_name": config.run_name if config else run_dir.name,
        "run_dir": str(run_dir),
        "allow_unresolved": allow_unresolved,
        "run_evaluation": run_evaluation,
        "metric": metric,
        "config": config_dict,
        "final_metrics": final_metrics,
        "counts": counts,
        "checks": checks,
        "md_reviews": md_review_rows,
        "md_review_issues": md_review_issues,
        "artifact_hashes": _artifact_hashes(paths_to_hash),
        "model_source_dir": str(model_dir or ""),
        "model_manifest_sha256": model_manifest_hash,
        "model_artifacts": model_rows,
        "import_hashes": import_rows,
        "git": _git_metadata(),
        "outputs": outputs,
        "notes": [
            "Blocked freezes are written for audit, but should not be used as official thesis results.",
            "Use --allow-unresolved only when intentionally preserving an unresolved state.",
        ],
    }

    _write_csv(Path(outputs["checks_csv"]), checks, CHECK_FIELDS)
    _write_csv(Path(outputs["model_manifest_csv"]), model_rows, MODEL_MANIFEST_FIELDS)
    _write_json(freeze_json, report)
    _write_model_card(Path(outputs["model_card"]), report)
    return report
