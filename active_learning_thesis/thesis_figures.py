from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

FIGURE_DIRNAME = "thesis_figures"
PALETTE = ["#1F6F8B", "#D97706", "#4C7C2B", "#B91C1C", "#6B5B95", "#0F766E", "#9F1239", "#475569"]


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


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [{str(key): str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]
    except Exception:
        return []


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _packet_manifest(packet_dir: Path) -> dict[str, object]:
    manifest = _read_json(packet_dir / "packet_manifest.json")
    if manifest:
        return manifest
    return {
        "title": packet_dir.name,
        "metric": "f1",
        "output_dir": str(packet_dir),
        "outputs": {
            "runs": str(packet_dir / "tables" / "runs.csv"),
            "md_review_evidence": str(packet_dir / "tables" / "md_review_evidence.csv"),
            "metrics": str(packet_dir / "tables" / "metrics.csv"),
            "learning_curves": str(packet_dir / "figure_data" / "learning_curves.csv"),
            "final_freezes": str(packet_dir / "tables" / "final_freezes.csv"),
            "study_artifacts": str(packet_dir / "tables" / "study_artifacts.csv"),
        },
    }


def _resolve_packet_path(packet_dir: Path, raw_path: object) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return packet_dir / "__missing__"
    path = Path(text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = packet_dir / path
    if candidate.exists():
        return candidate
    return path


def _output_path(manifest: Mapping[str, object], packet_dir: Path, key: str) -> Path:
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs", {}), dict) else {}
    return _resolve_packet_path(packet_dir, outputs.get(key, ""))


def _label(value: object, max_len: int = 22) -> str:
    text = str(value or "").strip() or "n/a"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _nice_domain(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        padding = abs(low) * 0.1 or 1.0
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#FBFAF6"/>',
    ]


def _axis_lines(x0: float, y0: float, plot_w: float, plot_h: float) -> list[str]:
    return [
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0 + plot_w:.1f}" y2="{y0:.1f}" stroke="#1F2937" stroke-width="1"/>',
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 - plot_h:.1f}" stroke="#1F2937" stroke-width="1"/>',
    ]


def _bar_chart_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    value_key: str,
    title: str,
    y_label: str,
) -> bool:
    chart_rows = [(str(row.get(label_key, "")), _safe_float(row.get(value_key))) for row in rows]
    chart_rows = [(label, value) for label, value in chart_rows if label and value is not None]
    if not chart_rows:
        return False
    width = max(760, 120 + len(chart_rows) * 84)
    height = 480
    margin_left = 82
    margin_right = 34
    margin_top = 58
    margin_bottom = 118
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_min = min(0.0, min(value for _, value in chart_rows))
    y_max = max(value for _, value in chart_rows)
    if y_min == y_max:
        y_max = y_min + 1.0
    y_span = y_max - y_min
    baseline_y = margin_top + plot_h - ((0.0 - y_min) / y_span) * plot_h if y_min < 0 else margin_top + plot_h
    slot = plot_w / len(chart_rows)
    bar_w = max(20, min(56, slot * 0.62))

    lines = _svg_header(width, height)
    lines.extend(
        [
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Georgia, serif" font-size="20" fill="#111827">{_escape(title)}</text>',
            f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22,{margin_top + plot_h / 2:.1f})" text-anchor="middle" font-family="Verdana, sans-serif" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        ]
    )
    for tick in range(5):
        value = y_min + (y_span * tick / 4)
        y = margin_top + plot_h - ((value - y_min) / y_span) * plot_h
        lines.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{margin_left + plot_w:.1f}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-family="Verdana, sans-serif" font-size="11" fill="#4B5563">{value:.2f}</text>')
    lines.extend(_axis_lines(margin_left, margin_top + plot_h, plot_w, plot_h))
    lines.append(f'<line x1="{margin_left:.1f}" y1="{baseline_y:.1f}" x2="{margin_left + plot_w:.1f}" y2="{baseline_y:.1f}" stroke="#6B7280" stroke-width="1" stroke-dasharray="4 4"/>')
    for index, (label, value) in enumerate(chart_rows):
        x = margin_left + index * slot + slot / 2 - bar_w / 2
        value_y = margin_top + plot_h - ((value - y_min) / y_span) * plot_h
        y = min(value_y, baseline_y)
        bar_h = abs(baseline_y - value_y)
        color = PALETTE[index % len(PALETTE)]
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Verdana, sans-serif" font-size="11" fill="#111827">{value:.2f}</text>')
        label_x = x + bar_w / 2
        lines.append(f'<text x="{label_x:.1f}" y="{margin_top + plot_h + 20:.1f}" text-anchor="end" transform="rotate(-35 {label_x:.1f},{margin_top + plot_h + 20:.1f})" font-family="Verdana, sans-serif" font-size="11" fill="#374151">{_escape(_label(label, 24))}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _line_chart_svg(
    path: Path,
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> bool:
    clean_series: dict[str, list[tuple[float, float]]] = {
        name: sorted([(float(x), float(y)) for x, y in points], key=lambda item: item[0])
        for name, points in series.items()
        if points
    }
    if not clean_series:
        return False
    all_x = [x for points in clean_series.values() for x, _ in points]
    all_y = [y for points in clean_series.values() for _, y in points]
    x_min, x_max = _nice_domain(all_x)
    y_min, y_max = _nice_domain(all_y)
    width = 860
    height = 520
    margin_left = 78
    margin_right = 190
    margin_top = 58
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_pos(value: float) -> float:
        return margin_left + ((value - x_min) / max(x_max - x_min, 1e-12)) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + plot_h - ((value - y_min) / max(y_max - y_min, 1e-12)) * plot_h

    lines = _svg_header(width, height)
    lines.extend(
        [
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Georgia, serif" font-size="20" fill="#111827">{_escape(title)}</text>',
            f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 20:.1f}" text-anchor="middle" font-family="Verdana, sans-serif" font-size="12" fill="#374151">{_escape(x_label)}</text>',
            f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22,{margin_top + plot_h / 2:.1f})" text-anchor="middle" font-family="Verdana, sans-serif" font-size="12" fill="#374151">{_escape(y_label)}</text>',
        ]
    )
    for tick in range(5):
        x_value = x_min + (x_max - x_min) * tick / 4
        x = x_pos(x_value)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top:.1f}" x2="{x:.1f}" y2="{margin_top + plot_h:.1f}" stroke="#EEF2F7" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 18:.1f}" text-anchor="middle" font-family="Verdana, sans-serif" font-size="11" fill="#4B5563">{x_value:.0f}</text>')
        y_value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(y_value)
        lines.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{margin_left + plot_w:.1f}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-family="Verdana, sans-serif" font-size="11" fill="#4B5563">{y_value:.2f}</text>')
    lines.extend(_axis_lines(margin_left, margin_top + plot_h, plot_w, plot_h))
    for index, (name, points) in enumerate(clean_series.items()):
        color = PALETTE[index % len(PALETTE)]
        polyline = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in points)
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3.5" fill="{color}" stroke="#FBFAF6" stroke-width="1"/>')
        legend_y = margin_top + index * 24
        lines.append(f'<rect x="{margin_left + plot_w + 28:.1f}" y="{legend_y - 10:.1f}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{margin_left + plot_w + 46:.1f}" y="{legend_y:.1f}" font-family="Verdana, sans-serif" font-size="12" fill="#374151">{_escape(_label(name, 26))}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _final_scorecard_rows(packet_dir: Path, manifest: Mapping[str, object], metric: str) -> list[dict[str, object]]:
    run_rows = _read_csv(_output_path(manifest, packet_dir, "runs"))
    freeze_rows = _read_csv(_output_path(manifest, packet_dir, "final_freezes"))
    runs_by_name = {row.get("run_name", ""): row for row in run_rows}
    rows: list[dict[str, object]] = []
    if freeze_rows:
        for freeze in freeze_rows:
            run_name = freeze.get("run_name", "")
            final_value = _safe_float(freeze.get(f"final_{metric}"))
            run = runs_by_name.get(run_name, {})
            baseline_value = _safe_float(run.get(f"baseline_{metric}"))
            if final_value is None:
                continue
            rows.append(
                {
                    "run": run_name,
                    "metric": metric,
                    "baseline_metric": baseline_value,
                    "final_metric": final_value,
                    "delta_metric": final_value - baseline_value if baseline_value is not None else "",
                    "freeze_status": freeze.get("status", ""),
                    "source": "final_freeze",
                }
            )
    if not rows:
        for run in run_rows:
            final_value = _safe_float(run.get(f"final_{metric}"))
            baseline_value = _safe_float(run.get(f"baseline_{metric}"))
            if final_value is None:
                continue
            rows.append(
                {
                    "run": run.get("run_name", ""),
                    "metric": metric,
                    "baseline_metric": baseline_value,
                    "final_metric": final_value,
                    "delta_metric": final_value - baseline_value if baseline_value is not None else "",
                    "freeze_status": "not frozen",
                    "source": "runs_table",
                }
            )
    return sorted(rows, key=lambda row: (-float(row["final_metric"]), str(row["run"])))


def _learning_curve_rows(packet_dir: Path, manifest: Mapping[str, object], metric: str) -> tuple[list[dict[str, object]], dict[str, list[tuple[float, float]]]]:
    rows = []
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in _read_csv(_output_path(manifest, packet_dir, "learning_curves")):
        value = _safe_float(row.get(metric))
        x_value = _safe_float(row.get("labeled_count"))
        if x_value is None:
            x_value = _safe_float(row.get("round_id"))
        if value is None or x_value is None:
            continue
        output = {
            "run": row.get("run_name", ""),
            "strategy": row.get("strategy", ""),
            "round_id": row.get("round_id", ""),
            "labeled_count": x_value,
            metric: value,
        }
        rows.append(output)
        grouped.setdefault((str(output["strategy"]), x_value), []).append(value)
    series: dict[str, list[tuple[float, float]]] = {}
    for (strategy, x_value), values in grouped.items():
        series.setdefault(strategy, []).append((x_value, mean(values)))
    return rows, series


def _md_feedback_rows(packet_dir: Path, manifest: Mapping[str, object]) -> list[dict[str, object]]:
    counts: dict[str, dict[str, int]] = {}
    for row in _read_csv(_output_path(manifest, packet_dir, "md_review_evidence")):
        state = row.get("review_evidence_state", "") or "Unspecified"
        entry = counts.setdefault(state, {"count": 0, "ingest_ready": 0})
        entry["count"] += 1
        if row.get("evidence_ready_for_ingest") == "yes":
            entry["ingest_ready"] += 1
    return [
        {"review_evidence_state": state, "count": values["count"], "ingest_ready_count": values["ingest_ready"]}
        for state, values in sorted(counts.items())
    ]


def _strategy_summary_rows(packet_dir: Path, manifest: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for artifact in _read_csv(_output_path(manifest, packet_dir, "study_artifacts")):
        artifact_name = artifact.get("artifact", "")
        if not artifact_name.endswith("_strategy_summary.csv"):
            continue
        artifact_path = _resolve_packet_path(packet_dir, artifact.get("path", ""))
        for row in _read_csv(artifact_path):
            if "strategy" not in row:
                continue
            rows.append(
                {
                    "rank": row.get("rank", ""),
                    "strategy": row.get("strategy", ""),
                    "n_runs": row.get("n_runs", row.get("paired_count", "")),
                    "final_mean": row.get("final_mean", row.get("final_advantage_mean", "")),
                    "aulc_mean": row.get("aulc_mean", row.get("aulc_advantage_mean", "")),
                    "labels_to_target_median": row.get("labels_to_target_median", row.get("labels_saved_to_target_median", "")),
                    "source_artifact": str(artifact_path),
                }
            )
    return rows


def _discovery_rows(packet_dir: Path, manifest: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in _read_csv(_output_path(manifest, packet_dir, "runs")):
        run_dir = _resolve_packet_path(packet_dir, run.get("run_dir", ""))
        for path in sorted((run_dir / "discovery").glob("*/summary.json")):
            payload = _read_json(path)
            if not payload:
                continue
            rows.append(
                {
                    "run": run.get("run_name", run_dir.name),
                    "strategy": payload.get("strategy", path.parent.name),
                    "exported_count": payload.get("exported_count", ""),
                    "top_batch_mean_utility_score": payload.get("top_batch_mean_utility_score", ""),
                    "top_batch_mean_pred_mean": payload.get("top_batch_mean_pred_mean", ""),
                    "top_batch_mean_pred_std": payload.get("top_batch_mean_pred_std", ""),
                    "source_artifact": str(path),
                }
            )
    return rows


def _caption_markdown(captions: Sequence[Mapping[str, object]]) -> str:
    lines = ["# Thesis Figure Captions", ""]
    for index, caption in enumerate(captions, start=1):
        lines.extend(
            [
                f"## Figure {index}. {caption.get('title', '')}",
                "",
                str(caption.get("caption", "")),
                "",
                f"Source table: `{caption.get('source_table', '')}`",
                f"Figure file: `{caption.get('figure', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _write_readme(path: Path, manifest: Mapping[str, object]) -> None:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts", {}), dict) else {}
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs", {}), dict) else {}
    lines = [
        "# Thesis Figure Bundle",
        "",
        "This bundle is output-only. It is derived from an exported thesis packet and does not change any run, ledger, dashboard, or freeze artifact.",
        "",
        f"- Packet directory: `{manifest.get('packet_dir', '')}`",
        f"- Metric focus: `{manifest.get('metric', '')}`",
        f"- Figures written: `{counts.get('figures', 0)}`",
        f"- Tables written: `{counts.get('tables', 0)}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_thesis_figures(
    packet_dir: Path,
    *,
    output_dir: Path | None = None,
    metric: str | None = None,
) -> dict[str, object]:
    packet_dir = Path(packet_dir)
    if not packet_dir.exists():
        raise FileNotFoundError(f"Thesis packet directory does not exist: {packet_dir}")
    manifest = _packet_manifest(packet_dir)
    selected_metric = metric or str(manifest.get("metric", "") or "f1")
    resolved_output = output_dir or packet_dir / FIGURE_DIRNAME
    figures_dir = resolved_output / "figures"
    tables_dir = resolved_output / "tables"
    captions_dir = resolved_output / "captions"

    outputs: dict[str, str] = {}
    figure_rows: list[dict[str, object]] = []
    captions: list[dict[str, object]] = []

    final_rows = _final_scorecard_rows(packet_dir, manifest, selected_metric)
    final_table = tables_dir / "table_final_scorecard.csv"
    _write_csv(final_table, final_rows, ["run", "metric", "baseline_metric", "final_metric", "delta_metric", "freeze_status", "source"])
    outputs["table_final_scorecard"] = str(final_table)
    final_figure = figures_dir / "figure_final_scorecard.svg"
    if _bar_chart_svg(final_figure, final_rows, label_key="run", value_key="final_metric", title=f"Frozen Final {selected_metric.upper()} By Run", y_label=selected_metric):
        outputs["figure_final_scorecard"] = str(final_figure)
        figure_rows.append({"figure_id": "final_scorecard", "status": "written", "path": str(final_figure), "source_table": str(final_table)})
        captions.append(
            {
                "title": f"Frozen final {selected_metric.upper()} by run",
                "caption": f"Final holdout {selected_metric} values for thesis packet runs. Frozen results are preferred when `final_freezes.csv` is present; otherwise the packet run summary is used.",
                "source_table": str(final_table),
                "figure": str(final_figure),
            }
        )
    else:
        figure_rows.append({"figure_id": "final_scorecard", "status": "skipped_no_data", "path": "", "source_table": str(final_table)})

    learning_rows, learning_series = _learning_curve_rows(packet_dir, manifest, selected_metric)
    learning_table = tables_dir / "table_learning_curve_points.csv"
    _write_csv(learning_table, learning_rows, ["run", "strategy", "round_id", "labeled_count", selected_metric])
    outputs["table_learning_curve_points"] = str(learning_table)
    learning_figure = figures_dir / "figure_learning_curves.svg"
    if _line_chart_svg(learning_figure, learning_series, title=f"Replay Learning Curves ({selected_metric.upper()})", x_label="Labeled peptides", y_label=selected_metric):
        outputs["figure_learning_curves"] = str(learning_figure)
        figure_rows.append({"figure_id": "learning_curves", "status": "written", "path": str(learning_figure), "source_table": str(learning_table)})
        captions.append(
            {
                "title": f"Replay learning curves for {selected_metric.upper()}",
                "caption": f"Mean replay {selected_metric} across labeled-peptide counts, grouped by active-learning strategy. This plot supports the acquisition-strategy comparison.",
                "source_table": str(learning_table),
                "figure": str(learning_figure),
            }
        )
    else:
        figure_rows.append({"figure_id": "learning_curves", "status": "skipped_no_data", "path": "", "source_table": str(learning_table)})

    md_rows = _md_feedback_rows(packet_dir, manifest)
    md_table = tables_dir / "table_md_feedback_summary.csv"
    _write_csv(md_table, md_rows, ["review_evidence_state", "count", "ingest_ready_count"])
    outputs["table_md_feedback_summary"] = str(md_table)
    md_figure = figures_dir / "figure_md_review_evidence.svg"
    if _bar_chart_svg(md_figure, md_rows, label_key="review_evidence_state", value_key="count", title="MD Review Evidence States", y_label="Peptide count"):
        outputs["figure_md_review_evidence"] = str(md_figure)
        figure_rows.append({"figure_id": "md_review_evidence", "status": "written", "path": str(md_figure), "source_table": str(md_table)})
        captions.append(
            {
                "title": "MD review evidence states",
                "caption": "Counts of reviewed MD rows by evidence state, highlighting how many labels are evidence-backed and ready for model feedback.",
                "source_table": str(md_table),
                "figure": str(md_figure),
            }
        )
    else:
        figure_rows.append({"figure_id": "md_review_evidence", "status": "skipped_no_data", "path": "", "source_table": str(md_table)})

    strategy_rows = _strategy_summary_rows(packet_dir, manifest)
    strategy_table = tables_dir / "table_strategy_summary.csv"
    _write_csv(strategy_table, strategy_rows, ["rank", "strategy", "n_runs", "final_mean", "aulc_mean", "labels_to_target_median", "source_artifact"])
    outputs["table_strategy_summary"] = str(strategy_table)
    strategy_figure = figures_dir / "figure_strategy_aulc.svg"
    if _bar_chart_svg(strategy_figure, strategy_rows, label_key="strategy", value_key="aulc_mean", title=f"Strategy AULC Summary ({selected_metric.upper()})", y_label="AULC"):
        outputs["figure_strategy_aulc"] = str(strategy_figure)
        figure_rows.append({"figure_id": "strategy_aulc", "status": "written", "path": str(strategy_figure), "source_table": str(strategy_table)})
        captions.append(
            {
                "title": "Acquisition strategy AULC summary",
                "caption": "Area-under-learning-curve summary by acquisition strategy from packet-linked study artifacts. Higher AULC indicates better label efficiency for metrics where higher is better.",
                "source_table": str(strategy_table),
                "figure": str(strategy_figure),
            }
        )
    else:
        figure_rows.append({"figure_id": "strategy_aulc", "status": "skipped_no_data", "path": "", "source_table": str(strategy_table)})

    discovery_rows = _discovery_rows(packet_dir, manifest)
    discovery_table = tables_dir / "table_discovery_summary.csv"
    _write_csv(discovery_table, discovery_rows, ["run", "strategy", "exported_count", "top_batch_mean_utility_score", "top_batch_mean_pred_mean", "top_batch_mean_pred_std", "source_artifact"])
    outputs["table_discovery_summary"] = str(discovery_table)
    discovery_figure = figures_dir / "figure_discovery_utility.svg"
    if _bar_chart_svg(discovery_figure, discovery_rows, label_key="strategy", value_key="top_batch_mean_utility_score", title="Discovery Strategy Utility", y_label="Mean utility"):
        outputs["figure_discovery_utility"] = str(discovery_figure)
        figure_rows.append({"figure_id": "discovery_utility", "status": "written", "path": str(discovery_figure), "source_table": str(discovery_table)})
        captions.append(
            {
                "title": "Discovery strategy utility",
                "caption": "Mean utility of the exported discovery shortlist by Bayesian-optimization strategy, derived from run discovery summaries referenced by the packet.",
                "source_table": str(discovery_table),
                "figure": str(discovery_figure),
            }
        )
    else:
        figure_rows.append({"figure_id": "discovery_utility", "status": "skipped_no_data", "path": "", "source_table": str(discovery_table)})

    figure_index = tables_dir / "figure_index.csv"
    _write_csv(figure_index, figure_rows, ["figure_id", "status", "path", "source_table"])
    outputs["figure_index"] = str(figure_index)
    captions_path = captions_dir / "figure_captions.md"
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(_caption_markdown(captions), encoding="utf-8")
    outputs["figure_captions"] = str(captions_path)
    readme_path = resolved_output / "README.md"
    outputs["readme"] = str(readme_path)
    manifest_path = resolved_output / "thesis_figures_manifest.json"
    outputs["manifest"] = str(manifest_path)

    result = {
        "status": "ready",
        "generated_at": _now_iso(),
        "packet_dir": str(packet_dir),
        "output_dir": str(resolved_output),
        "metric": selected_metric,
        "counts": {
            "figures": sum(1 for row in figure_rows if row.get("status") == "written"),
            "skipped_figures": sum(1 for row in figure_rows if row.get("status") != "written"),
            "tables": 6,
            "captions": len(captions),
            "final_scorecard_rows": len(final_rows),
            "learning_curve_rows": len(learning_rows),
            "md_feedback_summary_rows": len(md_rows),
            "strategy_summary_rows": len(strategy_rows),
            "discovery_summary_rows": len(discovery_rows),
        },
        "figures": figure_rows,
        "outputs": outputs,
        "notes": [
            "Output-only figure bundle derived from an exported thesis packet.",
            "SVG files are dependency-free and can be edited in vector graphics tools if needed.",
        ],
    }
    _write_json(manifest_path, result)
    _write_readme(readme_path, result)
    return result
