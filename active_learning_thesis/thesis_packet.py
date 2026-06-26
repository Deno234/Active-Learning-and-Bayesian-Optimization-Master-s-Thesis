from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from active_learning_thesis.config import RunConfig
from active_learning_thesis.dashboard import collect_dashboard_state
from active_learning_thesis.ledger import load_ledger
from active_learning_thesis.md_review_evidence import LABEL_REVIEW_FIELDS, review_evidence_status

PACKET_DIRNAME = "_thesis_packets"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "thesis_packet"


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


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fields = list(fieldnames or _fieldnames(rows))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _fieldnames(rows: Sequence[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _path_name(path: str | Path) -> str:
    text = str(path or "").strip()
    return Path(text).name if text else ""


def _discover_run_dirs(run_root: Path) -> list[Path]:
    ignored_parts = {"_dashboard_actions", "_dashboard_remote_state", PACKET_DIRNAME}
    run_dirs: list[Path] = []
    for config_path in sorted(run_root.rglob("config.json")):
        if any(part in ignored_parts for part in config_path.parts):
            continue
        run_dirs.append(config_path.parent)
    return sorted(set(run_dirs), key=lambda path: str(path))


def _safe_config(run_dir: Path) -> RunConfig | None:
    try:
        return RunConfig.load(run_dir / "config.json")
    except Exception:
        return None


def _latest_json_metric(run_dir: Path, pattern: str) -> dict[str, object]:
    candidates = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    for path in reversed(candidates):
        payload = _safe_read_json(path)
        if isinstance(payload, dict):
            return payload | {"_metric_path": str(path)}
    return {}


def _run_summary_rows(run_dirs: Sequence[Path], run_root: Path, metric: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = _safe_config(run_dir)
        baseline = _safe_read_json(run_dir / "metrics" / "baseline_round_000.json")
        if not isinstance(baseline, dict):
            baseline = {}
        final = _safe_read_json(run_dir / "metrics" / "final_holdout.json")
        if not isinstance(final, dict):
            final = {}
        post_ingest = _latest_json_metric(run_dir, "models/real_al/round_*/post_ingest/metrics.json")
        ledger_rows = _safe_read_csv(run_dir / "ledger.csv")
        proposed = [row for row in ledger_rows if row.get("status") == "proposed"]
        acquired = [row for row in ledger_rows if row.get("status") == "acquired"]
        md_campaigns = list((run_dir / "md_campaigns").glob("*")) if (run_dir / "md_campaigns").exists() else []
        rows.append(
            {
                "run_name": config.run_name if config else run_dir.name,
                "run_dir": str(run_dir),
                "relative_run_dir": str(run_dir.relative_to(run_root)) if run_dir.is_relative_to(run_root) else str(run_dir),
                "random_seed": config.random_seed if config else "",
                "real_strategy": config.real_strategy if config else "",
                "batch_size": config.batch_size if config else "",
                "max_rounds": config.max_rounds if config else "",
                "ledger_rows": len(ledger_rows),
                "proposed_rows": len(proposed),
                "acquired_rows": len(acquired),
                "md_campaigns": len([path for path in md_campaigns if path.is_dir()]),
                f"baseline_{metric}": baseline.get(metric, ""),
                f"post_ingest_{metric}": post_ingest.get(metric, ""),
                f"final_{metric}": final.get(metric, ""),
                "baseline_ready": "yes" if baseline else "no",
                "post_ingest_ready": "yes" if post_ingest else "no",
                "final_ready": "yes" if final else "no",
            }
        )
    return rows


def _md_review_rows(run_dirs: Sequence[Path], run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = _safe_config(run_dir)
        run_name = config.run_name if config else run_dir.name
        for review_csv in sorted((run_dir / "md_campaigns").glob("*/md_review.csv")):
            campaign_dir = review_csv.parent
            ingest_csv = campaign_dir / "cgmd_ingest.csv"
            for row in _safe_read_csv(review_csv):
                status = review_evidence_status(row)
                output = {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "relative_run_dir": str(run_dir.relative_to(run_root)) if run_dir.is_relative_to(run_root) else str(run_dir),
                    "campaign": campaign_dir.name,
                    "campaign_dir": str(campaign_dir),
                    "sequence": row.get("sequence", ""),
                    "round_id": row.get("round_id", ""),
                    "md_profile": row.get("md_profile", ""),
                    "job_root_status": row.get("job_root_status", ""),
                    "cgmd_label": row.get("cgmd_label", ""),
                    "review_notes": row.get("review_notes", ""),
                    "review_evidence_state": status.get("state", ""),
                    "evidence_ready_for_ingest": "yes" if status.get("ingest_ready") else "no",
                    "missing_or_blocked": ", ".join(
                        [str(item) for item in [*list(status.get("missing", [])), *list(status.get("blockers", []))] if str(item).strip()]
                    ) or "-",
                    "ingest_csv_exists": "yes" if ingest_csv.exists() else "no",
                    "review_csv": str(review_csv),
                    "ingest_csv": str(ingest_csv) if ingest_csv.exists() else "",
                    "sasa_file": row.get("sasa_file", ""),
                    "ap_file": row.get("ap_file", ""),
                    "ap_5ns": row.get("ap_5ns", ""),
                    "ap_12ns": row.get("ap_12ns", ""),
                    "ap_25ns": row.get("ap_25ns", ""),
                    "ap_50ns": row.get("ap_50ns", ""),
                    "ap_100ns": row.get("ap_100ns", ""),
                    "ap_200ns": row.get("ap_200ns", ""),
                }
                for field in LABEL_REVIEW_FIELDS:
                    output[field] = row.get(field, "")
                rows.append(output)
    return rows


def _learning_curve_rows(run_dirs: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = _safe_config(run_dir)
        run_name = config.run_name if config else run_dir.name
        for curve_path in sorted((run_dir / "replay").glob("*/learning_curve.csv")):
            strategy = curve_path.parent.name
            for row in _safe_read_csv(curve_path):
                rows.append({"run_name": run_name, "run_dir": str(run_dir), "strategy": strategy, **row})
    return rows


def _metric_rows(run_dirs: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patterns = [
        ("baseline", "metrics/baseline_round_000.json"),
        ("final_holdout", "metrics/final_holdout.json"),
        ("post_ingest", "models/real_al/round_*/post_ingest/metrics.json"),
        ("pre_proposal", "models/real_al/round_*/pre_proposal/metrics.json"),
    ]
    for run_dir in run_dirs:
        config = _safe_config(run_dir)
        run_name = config.run_name if config else run_dir.name
        for metric_kind, pattern in patterns:
            for metric_path in sorted(run_dir.glob(pattern)):
                payload = _safe_read_json(metric_path)
                if not isinstance(payload, dict):
                    continue
                rows.append(
                    {
                        "run_name": run_name,
                        "run_dir": str(run_dir),
                        "metric_kind": metric_kind,
                        "metric_path": str(metric_path),
                        **payload,
                    }
                )
    return rows


def _canary_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report_path in sorted((run_root / "_thesis_canaries").glob("*/canary_report.json")):
        payload = _safe_read_json(report_path)
        if not isinstance(payload, dict):
            continue
        checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
        outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs", {}), dict) else {}
        rows.append(
            {
                "run_name": payload.get("run_name", ""),
                "status": payload.get("status", ""),
                "seed": payload.get("seed", ""),
                "peptide_count": payload.get("peptide_count", ""),
                "analysis_complete_rows": checks.get("analysis_complete_rows", ""),
                "evidence_backed_reviews": checks.get("evidence_backed_reviews", ""),
                "ingest_rows_validated": checks.get("ingest_rows_validated", ""),
                "report_json": str(report_path),
                "report_markdown": outputs.get("report_markdown", ""),
                "run_dir": payload.get("run_dir", ""),
            }
        )
    return rows


def _freeze_rows(run_dirs: Sequence[Path], run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = _safe_config(run_dir)
        freeze_path = run_dir / "final_freeze" / "final_freeze.json"
        payload = _safe_read_json(freeze_path)
        if not isinstance(payload, dict):
            continue
        final_metrics = payload.get("final_metrics", {}) if isinstance(payload.get("final_metrics", {}), dict) else {}
        counts = payload.get("counts", {}) if isinstance(payload.get("counts", {}), dict) else {}
        outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs", {}), dict) else {}
        rows.append(
            {
                "run_name": payload.get("run_name", config.run_name if config else run_dir.name),
                "run_dir": str(run_dir),
                "relative_run_dir": str(run_dir.relative_to(run_root)) if run_dir.is_relative_to(run_root) else str(run_dir),
                "status": payload.get("status", ""),
                "frozen_at": payload.get("frozen_at", ""),
                "metric": payload.get("metric", ""),
                "final_f1": final_metrics.get("f1", ""),
                "final_pr_auc": final_metrics.get("pr_auc", ""),
                "final_roc_auc": final_metrics.get("roc_auc", ""),
                "evaluation_dataset": final_metrics.get("evaluation_dataset", ""),
                "surrogate_stage": final_metrics.get("surrogate_stage", ""),
                "surrogate_round_id": final_metrics.get("surrogate_round_id", final_metrics.get("round_id", "")),
                "ledger_rows": counts.get("ledger_rows", ""),
                "real_training_rows": counts.get("real_training_rows", ""),
                "holdout_rows": counts.get("holdout_rows", ""),
                "acquired_cgmd_rows": counts.get("acquired_cgmd_rows", ""),
                "pending_proposals": counts.get("pending_proposals", ""),
                "md_review_issues": counts.get("md_review_issues", ""),
                "failed_blockers": counts.get("failed_blockers", ""),
                "failed_warnings": counts.get("failed_warnings", ""),
                "model_artifacts": counts.get("model_artifacts", ""),
                "model_manifest_sha256": payload.get("model_manifest_sha256", ""),
                "freeze_json": str(freeze_path),
                "model_card": outputs.get("model_card", ""),
                "checks_csv": outputs.get("checks_csv", ""),
            }
        )
    return rows


def _study_artifact_rows(run_root: Path) -> list[dict[str, object]]:
    roots = [run_root / "_study_evidence", run_root / "_studies"]
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted([*root.rglob("*.csv"), *root.rglob("*.json"), *root.rglob("*.md")]):
            if not path.is_file():
                continue
            row_count = ""
            columns = ""
            if path.suffix == ".csv":
                csv_rows = _safe_read_csv(path)
                row_count = len(csv_rows)
                columns = ", ".join(_fieldnames(csv_rows)) if csv_rows else ""
            rows.append(
                {
                    "artifact": path.name,
                    "kind": path.suffix.lstrip("."),
                    "path": str(path),
                    "relative_path": str(path.relative_to(run_root)) if path.is_relative_to(run_root) else str(path),
                    "row_count": row_count,
                    "columns": columns,
                }
            )
    return rows


def _dashboard_rows(state: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    inventory = state.get("peptide_inventory", {}) if isinstance(state.get("peptide_inventory", {}), dict) else {}
    return {
        "dashboard_peptide_lifecycle": list(inventory.get("ledger", [])) if isinstance(inventory.get("ledger", []), list) else [],
        "dashboard_review_pipeline": list(inventory.get("review_pipeline", [])) if isinstance(inventory.get("review_pipeline", []), list) else [],
        "dashboard_readiness": list(state.get("execution_readiness", [])) if isinstance(state.get("execution_readiness", []), list) else [],
        "remote_watchdog": list(state.get("remote_watchdog", [])) if isinstance(state.get("remote_watchdog", []), list) else [],
    }


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


def _write_markdown_index(path: Path, manifest: dict[str, object]) -> None:
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs", {}), dict) else {}
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts", {}), dict) else {}
    lines = [
        "# Thesis Evidence Packet",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Run root: `{manifest.get('run_root', '')}`",
        f"- Metric focus: `{manifest.get('metric', '')}`",
        f"- Runs scanned: `{counts.get('runs', 0)}`",
        f"- MD review rows: `{counts.get('md_review_rows', 0)}`",
        f"- Evidence-backed MD reviews: `{counts.get('evidence_backed_reviews', 0)}`",
        f"- Canary reports: `{counts.get('canaries', 0)}`",
        f"- Final freezes: `{counts.get('final_freezes', 0)}`",
        "",
        "## Packet Files",
        "",
    ]
    for label, value in outputs.items():
        lines.append(f"- `{label}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "This packet consolidates local workflow evidence for thesis writing. Synthetic canary rows verify workflow contracts, not physical MD truth or final model performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_thesis_packet(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    title: str = "thesis_packet",
    metric: str = "f1",
    include_dashboard: bool = True,
) -> dict[str, object]:
    run_root = Path(run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    resolved_output = output_dir or run_root / PACKET_DIRNAME / f"{_slug(title)}_{_timestamp_slug()}"
    tables_dir = resolved_output / "tables"
    figure_data_dir = resolved_output / "figure_data"
    metadata_dir = resolved_output / "metadata"
    resolved_output.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(run_root)
    run_rows = _run_summary_rows(run_dirs, run_root, metric)
    md_review_rows = _md_review_rows(run_dirs, run_root)
    learning_rows = _learning_curve_rows(run_dirs)
    metric_rows = _metric_rows(run_dirs)
    canary_rows = _canary_rows(run_root)
    freeze_rows = _freeze_rows(run_dirs, run_root)
    study_rows = _study_artifact_rows(run_root)
    dashboard_tables: dict[str, list[dict[str, object]]] = {}
    dashboard_error = ""
    if include_dashboard:
        try:
            dashboard_tables = _dashboard_rows(collect_dashboard_state(run_root))
        except Exception as exc:
            dashboard_error = str(exc)

    outputs = {
        "runs": str(tables_dir / "runs.csv"),
        "md_review_evidence": str(tables_dir / "md_review_evidence.csv"),
        "metrics": str(tables_dir / "metrics.csv"),
        "learning_curves": str(figure_data_dir / "learning_curves.csv"),
        "canary_reports": str(tables_dir / "canary_reports.csv"),
        "final_freezes": str(tables_dir / "final_freezes.csv"),
        "study_artifacts": str(tables_dir / "study_artifacts.csv"),
        "reproducibility": str(metadata_dir / "reproducibility.json"),
        "manifest": str(resolved_output / "packet_manifest.json"),
        "index": str(resolved_output / "README.md"),
    }
    _write_csv(Path(outputs["runs"]), run_rows)
    _write_csv(Path(outputs["md_review_evidence"]), md_review_rows)
    _write_csv(Path(outputs["metrics"]), metric_rows)
    _write_csv(Path(outputs["learning_curves"]), learning_rows)
    _write_csv(Path(outputs["canary_reports"]), canary_rows)
    _write_csv(Path(outputs["final_freezes"]), freeze_rows)
    _write_csv(Path(outputs["study_artifacts"]), study_rows)

    for table_name, rows in dashboard_tables.items():
        table_path = tables_dir / f"{table_name}.csv"
        outputs[table_name] = str(table_path)
        _write_csv(table_path, rows)

    reproducibility = {
        "generated_at": _now_iso(),
        "run_root": str(run_root),
        "output_dir": str(resolved_output),
        "metric": metric,
        "git": _git_metadata(),
        "dashboard_error": dashboard_error,
    }
    _write_json(Path(outputs["reproducibility"]), reproducibility)

    counts = {
        "runs": len(run_rows),
        "md_review_rows": len(md_review_rows),
        "evidence_backed_reviews": sum(1 for row in md_review_rows if row.get("evidence_ready_for_ingest") == "yes"),
        "learning_curve_rows": len(learning_rows),
        "metric_rows": len(metric_rows),
        "canaries": len(canary_rows),
        "final_freezes": len(freeze_rows),
        "study_artifacts": len(study_rows),
        "dashboard_tables": len(dashboard_tables),
    }
    manifest = {
        "title": title,
        "generated_at": _now_iso(),
        "run_root": str(run_root),
        "output_dir": str(resolved_output),
        "metric": metric,
        "counts": counts,
        "outputs": outputs,
        "dashboard_error": dashboard_error,
    }
    _write_json(Path(outputs["manifest"]), manifest)
    _write_markdown_index(Path(outputs["index"]), manifest)
    return manifest
