from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

OPTIONAL_OUTPUT_DIRNAME = "_optional_evaluator_study"

INTERNAL_FIELDS = [
    "sequence",
    "source_kind",
    "source_path",
    "row_number",
    "round_id",
    "split",
    "status",
    "acquisition_strategy",
    "pred_mean",
    "pred_std",
    "pred_mutual_information",
    "raw_pred_mean",
    "raw_pred_std",
    "raw_pred_mutual_information",
    "acquisition_score",
    "label",
    "label_source",
]
EXTERNAL_FIELDS = [
    "sequence",
    "evaluator",
    "external_score",
    "external_label",
    "confidence",
    "source",
    "row_number",
    "external_scores_csv",
]
COMPLEXITY_FIELDS = [
    "sequence",
    "length",
    "unique_aa_count",
    "composition_diversity",
    "hydrophobic_fraction",
    "aromatic_fraction",
    "positive_count",
    "negative_count",
    "net_charge_proxy",
    "charge_density",
    "internal_uncertainty",
    "complexity_score",
    "complexity_bin",
    "source_kind",
    "source_path",
]
DISAGREEMENT_FIELDS = [
    "sequence",
    "evaluator",
    "internal_score",
    "external_score",
    "absolute_delta",
    "internal_label",
    "external_label",
    "label_disagreement",
    "internal_rank",
    "external_rank",
    "rank_delta",
    "internal_uncertainty",
    "complexity_bin",
    "complexity_score",
    "source_kind",
    "source_path",
]
SUMMARY_FIELDS = [
    "complexity_bin",
    "sequence_count",
    "mean_complexity_score",
    "mean_length",
    "mean_internal_uncertainty",
    "matched_external_count",
    "mean_absolute_delta",
    "label_disagreement_count",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value: object) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _source_rows(path: Path, source_kind: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for index, row in enumerate(_read_csv(path), start=1):
        sequence = str(row.get("sequence", "") or "").strip()
        if not sequence:
            continue
        rows.append(
            {
                **row,
                "sequence": sequence,
                "source_kind": source_kind,
                "source_path": str(path),
                "row_number": index,
            }
        )
    return rows


def _collect_internal_rows(run_dir: Path) -> list[dict[str, object]]:
    sources: list[tuple[str, Path]] = [
        ("ledger", run_dir / "ledger.csv"),
    ]
    sources.extend(("candidate_pool", path) for path in sorted((run_dir / "candidates").glob("*.csv")))
    sources.extend(("batch", path) for path in sorted((run_dir / "batches").glob("*.csv")))
    for strategy_dir in sorted((run_dir / "discovery").glob("*")):
        if not strategy_dir.is_dir():
            continue
        sources.append(("discovery_candidates", strategy_dir / "candidates.csv"))
        sources.append(("discovery_top_batch", strategy_dir / "top_batch.csv"))

    rows: list[dict[str, object]] = []
    for source_kind, path in sources:
        rows.extend(_source_rows(path, source_kind))
    return rows


def _internal_score(row: Mapping[str, object]) -> float | None:
    for field in ["pred_mean", "raw_pred_mean"]:
        value = _safe_float(row.get(field))
        if value is not None:
            return value
    return None


def _internal_uncertainty(row: Mapping[str, object]) -> float | None:
    for field in ["pred_mutual_information", "raw_pred_mutual_information", "pred_std", "raw_pred_std"]:
        value = _safe_float(row.get(field))
        if value is not None:
            return value
    return None


def _best_internal_rows(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    priority = {
        "ledger": 50,
        "candidate_pool": 40,
        "discovery_top_batch": 35,
        "discovery_candidates": 30,
        "batch": 20,
    }

    def sort_key(row: Mapping[str, object]) -> tuple[int, int, int, int]:
        has_score = 1 if _internal_score(row) is not None else 0
        source_priority = priority.get(str(row.get("source_kind", "")), 0)
        round_id = _safe_int(row.get("round_id")) or 0
        row_number = _safe_int(row.get("row_number")) or 0
        return has_score, source_priority, round_id, row_number

    best: dict[str, dict[str, object]] = {}
    for row in rows:
        sequence = str(row.get("sequence", "") or "")
        if not sequence:
            continue
        if sequence not in best or sort_key(row) > sort_key(best[sequence]):
            best[sequence] = dict(row)
    return best


def _score_field(row: Mapping[str, object]) -> str | None:
    for field in ["external_score", "score", "probability", "prob_self_assembly", "pred_mean"]:
        if _safe_float(row.get(field)) is not None:
            return field
    return None


def _label_from_value(value: object, score: float | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"0", "1"}:
        return raw
    if raw in {"false", "negative", "not_self_assembling", "not self assembling", "non_assembling"}:
        return "0"
    if raw in {"true", "positive", "self_assembling", "self assembling", "assembling"}:
        return "1"
    if score is not None:
        return "1" if score >= 0.5 else "0"
    return ""


def _load_external_scores(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"External score CSV does not exist: {path}")
    rows: list[dict[str, object]] = []
    default_evaluator = path.stem
    for index, row in enumerate(_read_csv(path), start=1):
        sequence = str(row.get("sequence", "") or "").strip()
        if not sequence:
            continue
        score_name = _score_field(row)
        if score_name is None:
            continue
        score = _safe_float(row.get(score_name))
        evaluator = str(row.get("evaluator", "") or "").strip() or default_evaluator
        label_value = row.get("external_label", row.get("prediction", row.get("label", "")))
        rows.append(
            {
                "sequence": sequence,
                "evaluator": evaluator,
                "external_score": score,
                "external_label": _label_from_value(label_value, score),
                "confidence": row.get("confidence", ""),
                "source": row.get("source", ""),
                "row_number": index,
                "external_scores_csv": str(path),
            }
        )
    return rows


def _composition_features(sequence: str) -> dict[str, object]:
    length = len(sequence)
    if length == 0:
        return {
            "length": 0,
            "unique_aa_count": 0,
            "composition_diversity": 0.0,
            "hydrophobic_fraction": 0.0,
            "aromatic_fraction": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "net_charge_proxy": 0,
            "charge_density": 0.0,
        }
    hydrophobic = sum(1 for aa in sequence if aa in {"A", "F", "I", "L", "M", "V", "W", "Y"})
    aromatic = sum(1 for aa in sequence if aa in {"F", "W", "Y"})
    positive = sum(1 for aa in sequence if aa in {"K", "R", "H"})
    negative = sum(1 for aa in sequence if aa in {"D", "E"})
    net_charge = positive - negative
    return {
        "length": length,
        "unique_aa_count": len(set(sequence)),
        "composition_diversity": len(set(sequence)) / length,
        "hydrophobic_fraction": hydrophobic / length,
        "aromatic_fraction": aromatic / length,
        "positive_count": positive,
        "negative_count": negative,
        "net_charge_proxy": net_charge,
        "charge_density": abs(net_charge) / length,
    }


def _bin_labels(bin_count: int) -> list[str]:
    if bin_count == 1:
        return ["all"]
    if bin_count == 2:
        return ["low", "high"]
    if bin_count == 3:
        return ["low", "medium", "high"]
    if bin_count == 4:
        return ["low", "medium", "high", "very_high"]
    return [f"bin_{index:02d}" for index in range(1, bin_count + 1)]


def _assign_complexity_bins(rows: list[dict[str, object]], bin_count: int) -> None:
    if not rows:
        return
    labels = _bin_labels(max(1, bin_count))
    ordered = sorted(rows, key=lambda row: (float(row["complexity_score"]), str(row["sequence"])))
    for index, row in enumerate(ordered):
        bin_index = min(len(labels) - 1, int(index * len(labels) / len(ordered)))
        row["complexity_bin"] = labels[bin_index]


def _complexity_rows(best_rows: Mapping[str, dict[str, object]], bin_count: int) -> list[dict[str, object]]:
    if not best_rows:
        return []
    max_length = max((len(sequence) for sequence in best_rows), default=1)
    uncertainties = [value for value in (_internal_uncertainty(row) for row in best_rows.values()) if value is not None]
    max_uncertainty = max(uncertainties, default=0.0)
    rows: list[dict[str, object]] = []
    for sequence, row in sorted(best_rows.items()):
        features = _composition_features(sequence)
        uncertainty = _internal_uncertainty(row)
        uncertainty_norm = (uncertainty / max_uncertainty) if uncertainty is not None and max_uncertainty > 0 else 0.0
        length_norm = features["length"] / max(max_length, 1)
        score = (
            0.35 * float(length_norm)
            + 0.25 * float(features["composition_diversity"])
            + 0.20 * float(features["charge_density"])
            + 0.20 * float(uncertainty_norm)
        )
        rows.append(
            {
                **features,
                "sequence": sequence,
                "internal_uncertainty": uncertainty,
                "complexity_score": score,
                "complexity_bin": "",
                "source_kind": row.get("source_kind", ""),
                "source_path": row.get("source_path", ""),
            }
        )
    _assign_complexity_bins(rows, bin_count)
    return rows


def _rank_map(rows: Sequence[Mapping[str, object]], score_key: str) -> dict[tuple[str, str], int]:
    by_evaluator: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_evaluator.setdefault(str(row.get("evaluator", "")), []).append(row)
    ranks: dict[tuple[str, str], int] = {}
    for evaluator, evaluator_rows in by_evaluator.items():
        ordered = sorted(
            evaluator_rows,
            key=lambda row: (-(float(row.get(score_key, 0.0) or 0.0)), str(row.get("sequence", ""))),
        )
        for rank, row in enumerate(ordered, start=1):
            ranks[(evaluator, str(row.get("sequence", "")))] = rank
    return ranks


def _disagreement_rows(
    external_rows: Sequence[dict[str, object]],
    best_internal: Mapping[str, dict[str, object]],
    complexity_by_sequence: Mapping[str, dict[str, object]],
) -> list[dict[str, object]]:
    prelim: list[dict[str, object]] = []
    for external in external_rows:
        sequence = str(external.get("sequence", ""))
        internal = best_internal.get(sequence)
        if not internal:
            continue
        internal_score = _internal_score(internal)
        external_score = _safe_float(external.get("external_score"))
        if internal_score is None or external_score is None:
            continue
        internal_label = "1" if internal_score >= 0.5 else "0"
        external_label = str(external.get("external_label", "")) or ("1" if external_score >= 0.5 else "0")
        complexity = complexity_by_sequence.get(sequence, {})
        prelim.append(
            {
                "sequence": sequence,
                "evaluator": external.get("evaluator", ""),
                "internal_score": internal_score,
                "external_score": external_score,
                "absolute_delta": abs(internal_score - external_score),
                "internal_label": internal_label,
                "external_label": external_label,
                "label_disagreement": internal_label != external_label,
                "internal_rank": "",
                "external_rank": "",
                "rank_delta": "",
                "internal_uncertainty": _internal_uncertainty(internal),
                "complexity_bin": complexity.get("complexity_bin", ""),
                "complexity_score": complexity.get("complexity_score", ""),
                "source_kind": internal.get("source_kind", ""),
                "source_path": internal.get("source_path", ""),
            }
        )
    internal_ranks = _rank_map(prelim, "internal_score")
    external_ranks = _rank_map(prelim, "external_score")
    for row in prelim:
        key = (str(row["evaluator"]), str(row["sequence"]))
        internal_rank = internal_ranks.get(key)
        external_rank = external_ranks.get(key)
        row["internal_rank"] = internal_rank
        row["external_rank"] = external_rank
        row["rank_delta"] = abs(internal_rank - external_rank) if internal_rank is not None and external_rank is not None else ""
    return sorted(prelim, key=lambda row: (str(row["evaluator"]), -float(row["absolute_delta"]), str(row["sequence"])))


def _mean_or_blank(values: Sequence[float]) -> float | str:
    return mean(values) if values else ""


def _summary_rows(
    complexity_rows: Sequence[dict[str, object]],
    disagreement_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    disagreements_by_bin: dict[str, list[dict[str, object]]] = {}
    for row in disagreement_rows:
        disagreements_by_bin.setdefault(str(row.get("complexity_bin", "")), []).append(row)
    rows: list[dict[str, object]] = []
    for bin_name in _bin_labels(len({str(row.get("complexity_bin", "")) for row in complexity_rows}) or 1):
        bin_complexity = [row for row in complexity_rows if str(row.get("complexity_bin", "")) == bin_name]
        bin_disagreements = disagreements_by_bin.get(bin_name, [])
        if not bin_complexity and not bin_disagreements:
            continue
        rows.append(
            {
                "complexity_bin": bin_name,
                "sequence_count": len(bin_complexity),
                "mean_complexity_score": _mean_or_blank([float(row["complexity_score"]) for row in bin_complexity]),
                "mean_length": _mean_or_blank([float(row["length"]) for row in bin_complexity]),
                "mean_internal_uncertainty": _mean_or_blank(
                    [
                        float(row["internal_uncertainty"])
                        for row in bin_complexity
                        if _safe_float(row.get("internal_uncertainty")) is not None
                    ]
                ),
                "matched_external_count": len(bin_disagreements),
                "mean_absolute_delta": _mean_or_blank([float(row["absolute_delta"]) for row in bin_disagreements]),
                "label_disagreement_count": sum(1 for row in bin_disagreements if bool(row.get("label_disagreement"))),
            }
        )
    return rows


def _write_readme(path: Path, manifest: Mapping[str, object]) -> None:
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs", {}), dict) else {}
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts", {}), dict) else {}
    lines = [
        "# Optional External Evaluator Study",
        "",
        "This is a sidecar analysis. It does not change the active-learning ledger, model artifacts, dashboard state, thesis packet, or final freeze.",
        "",
        f"- Status: `{manifest.get('status', '')}`",
        f"- Run directory: `{manifest.get('run_dir', '')}`",
        f"- Internal peptide rows: `{counts.get('internal_rows', 0)}`",
        f"- External score rows: `{counts.get('external_rows', 0)}`",
        f"- Matched disagreement rows: `{counts.get('disagreement_rows', 0)}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "Use this only if you decide to report external-evaluator disagreement or curriculum/complexity analysis as an optional thesis result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_optional_evaluator_study(
    run_dir: Path,
    *,
    external_scores: Path | None = None,
    output_dir: Path | None = None,
    bin_count: int = 4,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")

    resolved_output = output_dir or run_dir / OPTIONAL_OUTPUT_DIRNAME
    tables_dir = resolved_output / "tables"
    internal_rows = _collect_internal_rows(run_dir)
    best_internal = _best_internal_rows(internal_rows)
    external_rows = _load_external_scores(Path(external_scores) if external_scores else None)
    complexity = _complexity_rows(best_internal, bin_count)
    complexity_by_sequence = {str(row["sequence"]): row for row in complexity}
    disagreements = _disagreement_rows(external_rows, best_internal, complexity_by_sequence)
    summaries = _summary_rows(complexity, disagreements)

    outputs = {
        "internal_predictions": str(tables_dir / "internal_predictions.csv"),
        "external_scores": str(tables_dir / "external_scores.csv"),
        "evaluator_disagreement": str(tables_dir / "evaluator_disagreement.csv"),
        "complexity_bins": str(tables_dir / "complexity_bins.csv"),
        "complexity_summary": str(tables_dir / "complexity_summary.csv"),
        "manifest": str(resolved_output / "optional_evaluator_study_manifest.json"),
        "readme": str(resolved_output / "README.md"),
    }
    _write_csv(Path(outputs["internal_predictions"]), INTERNAL_FIELDS, internal_rows)
    _write_csv(Path(outputs["external_scores"]), EXTERNAL_FIELDS, external_rows)
    _write_csv(Path(outputs["evaluator_disagreement"]), DISAGREEMENT_FIELDS, disagreements)
    _write_csv(Path(outputs["complexity_bins"]), COMPLEXITY_FIELDS, complexity)
    _write_csv(Path(outputs["complexity_summary"]), SUMMARY_FIELDS, summaries)

    status = "ready" if external_rows else "no_external_scores"
    if external_rows and not disagreements:
        status = "no_matched_external_scores"
    manifest: dict[str, object] = {
        "status": status,
        "generated_at": _now_iso(),
        "run_dir": str(run_dir),
        "output_dir": str(resolved_output),
        "external_scores": str(external_scores) if external_scores else "",
        "bin_count": bin_count,
        "counts": {
            "internal_rows": len(internal_rows),
            "unique_internal_sequences": len(best_internal),
            "external_rows": len(external_rows),
            "complexity_rows": len(complexity),
            "disagreement_rows": len(disagreements),
            "complexity_summary_rows": len(summaries),
        },
        "outputs": outputs,
        "notes": [
            "Optional sidecar analysis only; it does not mutate the active-learning workflow.",
            "Complexity score is a transparent proxy combining length, composition diversity, charge density, and internal uncertainty.",
        ],
    }
    _write_json(Path(outputs["manifest"]), manifest)
    _write_readme(Path(outputs["readme"]), manifest)
    return manifest
