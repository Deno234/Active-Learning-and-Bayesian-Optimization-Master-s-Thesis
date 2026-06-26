from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

MD_SOURCE_BATCHES_DIRNAME = "md_source_batches"
MD_SOURCE_BATCH_FIELDS = [
    "sequence",
    "round_id",
    "acquisition_strategy",
    "pred_mean",
    "pred_std",
    "pred_entropy",
    "pred_mutual_information",
    "acquisition_score",
]


def _run_key(value: str | Path) -> str:
    return hashlib.sha1(str(Path(value).resolve()).encode("utf-8")).hexdigest()[:12]


def _preferred_md_source_batches_root(run_root: Path) -> Path:
    user_root = Path.home() / ".active_learning_thesis" / "dashboard_state"
    return user_root / _run_key(run_root) / MD_SOURCE_BATCHES_DIRNAME


def _fallback_md_source_batches_root(run_root: Path) -> Path:
    temp_root = Path(tempfile.gettempdir()) / "active_learning_thesis" / "dashboard_state"
    return temp_root / _run_key(run_root) / MD_SOURCE_BATCHES_DIRNAME


def dashboard_md_source_batches_root(run_root: Path) -> Path:
    preferred = _preferred_md_source_batches_root(run_root)
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = _fallback_md_source_batches_root(run_root)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def candidate_md_source_batch_path(run_root: Path, run_dir: Path, sequence: str) -> Path:
    sequence_key = str(sequence).strip().upper()
    return dashboard_md_source_batches_root(run_root) / _run_key(run_dir) / f"{sequence_key}.csv"


def find_dashboard_md_source_batch(run_root: Path, run_dir: Path, sequence: str) -> str:
    sequence_key = str(sequence).strip().upper()
    relative = Path(_run_key(run_dir)) / f"{sequence_key}.csv"
    for root in (_preferred_md_source_batches_root(run_root), _fallback_md_source_batches_root(run_root)):
        candidate = root / relative
        if candidate.exists():
            return str(candidate)
    return ""


def find_run_md_source_batch(run_dir: Path, sequence: str) -> str:
    batches_root = run_dir / "batches"
    if not batches_root.exists():
        return ""
    sequence_key = str(sequence).strip().upper()
    for candidate in sorted(batches_root.glob("round_*_batch.csv"), reverse=True):
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if str(row.get("sequence", "")).strip().upper() == sequence_key:
                        return str(candidate)
        except Exception:
            continue
    return ""


def load_md_source_batch_row(batch_csv: Path, sequence: str) -> dict[str, str]:
    sequence_key = str(sequence).strip().upper()
    with batch_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("sequence", "")).strip().upper() == sequence_key:
                return {str(key): str(value or "") for key, value in row.items()}
    raise ValueError(f"Sequence {sequence_key} was not found in batch CSV {batch_csv}")


def is_dashboard_md_source_batch(run_root: Path, batch_csv: str | Path | None) -> bool:
    if not batch_csv:
        return False
    try:
        resolved = Path(batch_csv).resolve()
    except Exception:
        return False
    for root in (_preferred_md_source_batches_root(run_root), _fallback_md_source_batches_root(run_root)):
        try:
            resolved.relative_to(root.resolve())
            return True
        except Exception:
            continue
    return False


def export_dashboard_md_source_batch(
    run_root: Path,
    *,
    run_dir: Path,
    sequence: str,
    round_id: str,
    acquisition_strategy: str,
) -> Path:
    path = candidate_md_source_batch_path(run_root, run_dir, sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "sequence": str(sequence).strip().upper(),
        "round_id": str(round_id).strip() or "0",
        "acquisition_strategy": str(acquisition_strategy).strip() or "dashboard_md_export",
        "pred_mean": "",
        "pred_std": "",
        "pred_entropy": "",
        "pred_mutual_information": "",
        "acquisition_score": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MD_SOURCE_BATCH_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path
