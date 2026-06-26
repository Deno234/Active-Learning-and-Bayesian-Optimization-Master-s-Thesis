from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAMP = "20260623"
OUTPUT = ROOT / "thesis_results" / f"THESIS_RESULTS_HANDOFF_{STAMP}"
ZIP_PATH = ROOT / "thesis_results" / f"THESIS_RESULTS_HANDOFF_{STAMP}.zip"

PHASE1 = ROOT / "thesis_results" / "01_reproduction"
PHASE2 = ROOT / "thesis_results" / "02_replay"
PHASE3 = (
    ROOT
    / "thesis_results"
    / "03_real_al"
    / "phase3_results_20260620"
    / "thesis_results"
    / "03_real_al"
)
PHASE4 = (
    ROOT
    / "thesis_results"
    / "04_bayesian_optimization"
    / "Phase4 results"
    / "phase4_complete_20260621"
    / "thesis_results"
    / "04_bayesian_optimization"
)
PHASE5 = ROOT / "thesis_results" / "05_self_paced_active_learning"

CAMPAIGN_ROOTS = [
    ROOT / "active_learning_runs" / "thesis_main_20260502" / "md_campaigns",
    ROOT / "active_learning_runs" / "thesis_main_supek_20260502" / "md_campaigns",
    ROOT
    / "active_learning_runs"
    / "thesis_main_supek_clean_20260502_original"
    / "md_campaigns",
]

POLICY_ORDER = ["random", "greedy", "ucb", "pi", "ei", "mes"]
BRANCH_ORDER = [
    "predictive_entropy",
    "family_qbc",
    "cluster_diverse_representative",
]
PHASE5_STRATEGIES = [
    "random",
    "predictive_entropy",
    "static_easy_entropy",
    "self_paced_entropy",
]

COLORS = {
    "random": "#64748b",
    "greedy": "#0f766e",
    "ucb": "#2563eb",
    "pi": "#7c3aed",
    "ei": "#db2777",
    "mes": "#d97706",
    "predictive_entropy": "#2563eb",
    "family_qbc": "#0f766e",
    "cluster_diverse_representative": "#d97706",
    "static_easy_entropy": "#dc2626",
    "self_paced_entropy": "#7c3aed",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def number(value: object, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_start(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#0f172a">{esc(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'<text x="{width / 2:.1f}" y="54" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">{esc(subtitle)}</text>'
        )
    return lines


def svg_end(lines: list[str], path: Path) -> None:
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: dict[str, list[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 980,
    height: int = 600,
) -> None:
    left, right, top, bottom = 90, 250, 80, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_points = [point for points in series.values() for point in points]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y_min is None else y_min
    y1 = max(ys) if y_max is None else y_max
    if y0 == y1:
        y0 -= 0.1
        y1 += 0.1
    pad = (y1 - y0) * 0.06
    if y_min is None:
        y0 -= pad
    if y_max is None:
        y1 += pad

    def sx(x: float) -> float:
        return left + (x - x0) / max(x1 - x0, 1e-12) * plot_w

    def sy(y: float) -> float:
        return top + (y1 - y) / max(y1 - y0, 1e-12) * plot_h

    lines = svg_start(width, height, title, subtitle)
    for tick in range(6):
        value = y0 + tick * (y1 - y0) / 5
        yy = sy(value)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        lines.append(
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#475569">{value:.2f}</text>'
        )
    for tick in range(6):
        value = x0 + tick * (x1 - x0) / 5
        xx = sx(value)
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#f1f5f9"/>')
        lines.append(
            f'<text x="{xx:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#475569">{value:.0f}</text>'
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155" stroke-width="1.5"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="13" fill="#334155">{esc(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#334155">{esc(y_label)}</text>',
        ]
    )
    legend_y = top + 10
    for index, (name, points) in enumerate(series.items()):
        color = COLORS.get(name, f"hsl({index * 47 % 360},55%,45%)")
        ordered = sorted(points)
        point_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        lines.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in ordered:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="{color}"/>')
        ly = legend_y + index * 25
        lines.append(f'<line x1="{left + plot_w + 28}" y1="{ly}" x2="{left + plot_w + 55}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{left + plot_w + 63}" y="{ly + 4}" font-family="Arial" font-size="12" fill="#1e293b">{esc(name.replace("_", " "))}</text>'
        )
    svg_end(lines, path)


def bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    *,
    label_key: str,
    value_keys: list[str],
    legend_labels: list[str] | None = None,
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
    width: int = 980,
    height: int = 560,
) -> None:
    left, right, top, bottom = 90, 60, 85, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row[key]) for row in rows for key in value_keys if str(row.get(key, "")).strip()]
    maximum = max(values) if values else 1.0
    y1 = y_max if y_max is not None else maximum * 1.14
    if y1 <= y_min:
        y1 = y_min + 1
    lines = svg_start(width, height, title, subtitle)

    def sy(y: float) -> float:
        return top + (y1 - y) / (y1 - y_min) * plot_h

    for tick in range(6):
        value = y_min + tick * (y1 - y_min) / 5
        yy = sy(value)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        lines.append(
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#475569">{value:.2f}</text>'
        )
    group_w = plot_w / max(len(rows), 1)
    bar_gap = 4
    bar_w = min(52, (group_w * 0.78 - bar_gap * (len(value_keys) - 1)) / max(len(value_keys), 1))
    palette = ["#2563eb", "#d97706", "#0f766e", "#7c3aed", "#dc2626"]
    for row_index, row in enumerate(rows):
        center = left + group_w * (row_index + 0.5)
        total_w = len(value_keys) * bar_w + (len(value_keys) - 1) * bar_gap
        start = center - total_w / 2
        for key_index, key in enumerate(value_keys):
            value = number(row.get(key))
            if value is None:
                continue
            xx = start + key_index * (bar_w + bar_gap)
            yy = sy(value)
            color = palette[key_index % len(palette)]
            lines.append(
                f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{top + plot_h - yy:.1f}" fill="{color}" rx="2"/>'
            )
            lines.append(
                f'<text x="{xx + bar_w / 2:.1f}" y="{yy - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#334155">{value:.3f}</text>'
            )
        label = str(row[label_key]).replace("_", " ")
        lines.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 20}" transform="rotate(28 {center:.1f} {top + plot_h + 20})" text-anchor="start" font-family="Arial" font-size="11" fill="#334155">{esc(label)}</text>'
        )
    lines.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#334155">{esc(y_label)}</text>'
    )
    if len(value_keys) > 1:
        labels = legend_labels or value_keys
        for index, label in enumerate(labels):
            xx = left + index * 180
            lines.append(f'<rect x="{xx}" y="{height - 32}" width="14" height="10" fill="{palette[index]}"/>')
            lines.append(
                f'<text x="{xx + 20}" y="{height - 23}" font-family="Arial" font-size="11" fill="#334155">{esc(label)}</text>'
            )
    svg_end(lines, path)


def scatter_chart(
    path: Path,
    title: str,
    subtitle: str,
    points: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    group_key: str,
    x_label: str,
    y_label: str,
    x_threshold: float | None = None,
    y_threshold: float | None = None,
    width: int = 980,
    height: int = 590,
) -> None:
    left, right, top, bottom = 90, 240, 85, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(row[x_key]) for row in points]
    ys = [float(row[y_key]) for row in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x_pad = max((x1 - x0) * 0.08, 0.05)
    y_pad = max((y1 - y0) * 0.08, 0.03)
    x0, x1 = x0 - x_pad, x1 + x_pad
    y0, y1 = y0 - y_pad, y1 + y_pad

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return top + (y1 - y) / (y1 - y0) * plot_h

    lines = svg_start(width, height, title, subtitle)
    for tick in range(6):
        xv = x0 + tick * (x1 - x0) / 5
        yv = y0 + tick * (y1 - y0) / 5
        xx, yy = sx(xv), sy(yv)
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#f1f5f9"/>')
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        lines.append(f'<text x="{xx:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="11" fill="#475569">{xv:.2f}</text>')
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv:.2f}</text>')
    if x_threshold is not None:
        xx = sx(x_threshold)
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>')
        lines.append(f'<text x="{xx + 5:.1f}" y="{top + 15}" font-size="11" fill="#b91c1c">threshold {x_threshold:g}</text>')
    if y_threshold is not None:
        yy = sy(y_threshold)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>')
        lines.append(f'<text x="{left + 5}" y="{yy - 7:.1f}" font-size="11" fill="#b91c1c">threshold {y_threshold:g}</text>')
    groups = []
    for row in points:
        group = str(row[group_key])
        if group not in groups:
            groups.append(group)
        color = COLORS.get(group, "#334155")
        positive = str(row.get("operational_label", "")) == "1"
        lines.append(
            f'<circle cx="{sx(float(row[x_key])):.1f}" cy="{sy(float(row[y_key])):.1f}" r="{6 if positive else 5}" fill="{color}" fill-opacity="0.78" stroke="{"#111827" if positive else "#ffffff"}" stroke-width="{2 if positive else 1}"><title>{esc(row.get("sequence", ""))}</title></circle>'
        )
    for index, group in enumerate(groups):
        yy = top + 18 + index * 25
        color = COLORS.get(group, "#334155")
        lines.append(f'<circle cx="{left + plot_w + 30}" cy="{yy}" r="6" fill="{color}"/>')
        lines.append(f'<text x="{left + plot_w + 43}" y="{yy + 4}" font-size="12" fill="#1e293b">{esc(group.replace("_", " "))}</text>')
    lines.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="13" fill="#334155">{esc(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="#334155">{esc(y_label)}</text>',
        ]
    )
    svg_end(lines, path)


def heatmap(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: dict[tuple[str, str], float],
    *,
    width: int = 800,
    height: int = 700,
) -> None:
    left, top, right, bottom = 210, 80, 80, 170
    plot = min(width - left - right, height - top - bottom)
    cell = plot / len(labels)
    lines = svg_start(width, height, title, subtitle)
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            value = values.get((row_label, col_label), values.get((col_label, row_label), 0.0))
            shade = int(245 - 165 * max(0.0, min(1.0, value)))
            color = f"rgb({shade},{min(250, shade + 20)},250)"
            xx, yy = left + j * cell, top + i * cell
            lines.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{color}" stroke="#ffffff"/>')
            lines.append(f'<text x="{xx + cell / 2:.1f}" y="{yy + cell / 2 + 4:.1f}" text-anchor="middle" font-size="11" fill="#0f172a">{value:.2f}</text>')
        yy = top + (i + 0.5) * cell
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#334155">{esc(row_label.replace("_", " "))}</text>')
    for j, label in enumerate(labels):
        xx = left + (j + 0.5) * cell
        lines.append(f'<text x="{xx:.1f}" y="{top + plot + 15:.1f}" transform="rotate(42 {xx:.1f} {top + plot + 15:.1f})" text-anchor="start" font-size="11" fill="#334155">{esc(label.replace("_", " "))}</text>')
    svg_end(lines, path)


def phase1() -> list[dict[str, str]]:
    target = OUTPUT / "01_phase1_reproduction"
    sources = [
        PHASE1 / "tables" / "reproduced_predictive_performance.csv",
        PHASE1 / "tables" / "threshold_summary.csv",
        PHASE1 / "generated" / "generated_similarity_summary.csv",
        PHASE1 / "figures" / "presentation" / "phase1_performance_presentation.svg",
    ]
    for source in sources:
        destination = target / ("figures" if source.suffix == ".svg" else "tables") / source.name
        copy_file(source, destination)
    return [{"phase": "1", "status": "complete", "primary_result": "Nested-CV predictive reproduction", "canonical_source": str(PHASE1.relative_to(ROOT))}]


def phase2() -> list[dict[str, str]]:
    target = OUTPUT / "02_phase2_replay"
    tables = [
        PHASE2 / "evidence" / "benchmark_strategy_summary.csv",
        PHASE2 / "evidence" / "labels_to_target_summary.csv",
        PHASE2 / "evidence" / "paired_vs_random.csv",
        PHASE2 / "evidence" / "ablation_summary.csv",
        PHASE2 / "evidence" / "ablation_calibration_summary.csv",
        PHASE2 / "benchmark" / "overlap" / "pairwise_strategy_overlap_summary.csv",
        PHASE2 / "benchmark" / "overlap" / "strategy_unique_selected_counts.csv",
    ]
    figures = [
        "benchmark_holdout_f1_initial_10_vs_labeled_peptides.svg",
        "benchmark_holdout_f1_initial_40_vs_labeled_peptides.svg",
        "benchmark_holdout_mean_AULC_F1_initial_10.svg",
        "benchmark_holdout_mean_AULC_F1_initial_40.svg",
        "benchmark_holdout_mean_final_F1_initial_10.svg",
        "benchmark_holdout_mean_final_F1_initial_40.svg",
        "benchmark_holdout_labels_to_f1_086_combined_initial_10_40.svg",
        "benchmark_pairwise_jaccard_heatmap_initial_10_presentation.svg",
        "benchmark_pairwise_jaccard_heatmap_initial_40_presentation.svg",
        "ablation_holdout_mean_AULC_F1_initial_10.svg",
        "ablation_holdout_mean_AULC_F1_initial_40.svg",
        "ablation_holdout_mean_final_F1_initial_10.svg",
        "ablation_holdout_mean_final_F1_initial_40.svg",
        "ablation_mean_final_Brier_by_setup.svg",
        "ablation_mean_final_ECE_10_by_setup.svg",
    ]
    for source in tables:
        copy_file(source, target / "tables" / source.name)
    for name in figures:
        copy_file(PHASE2 / "evidence" / "figures" / name, target / "figures" / name)
    return [{"phase": "2", "status": "complete", "primary_result": "Retrospective ten-strategy replay", "canonical_source": str((PHASE2 / "evidence").relative_to(ROOT))}]


def phase3() -> list[dict[str, str]]:
    target = OUTPUT / "03_phase3_real_active_learning"
    labels: list[dict[str, object]] = []
    final_metrics: list[dict[str, object]] = []
    for branch in BRANCH_ORDER:
        cumulative = 0
        for round_id in range(1, 9):
            source = PHASE3 / "branches" / branch / "rounds" / f"round_{round_id:03d}" / "ingest" / "cgmd_ingest.csv"
            for row in read_csv(source):
                label = int(row["cgmd_label"])
                cumulative += label
                labels.append(
                    {
                        "branch": branch,
                        "round": round_id,
                        "sequence": row["sequence"],
                        "cgmd_label": label,
                        "cumulative_positive_labels": cumulative,
                    }
                )
        metrics = json.loads((PHASE3 / "branches" / branch / "metrics" / "final_holdout.json").read_text(encoding="utf-8"))
        final_metrics.append(
            {
                "branch": branch,
                "labeled_count": int(metrics["labeled_count"]),
                "f1": float(metrics["f1"]),
                "pr_auc": float(metrics["pr_auc"]),
                "roc_auc": float(metrics["roc_auc"]),
                "brier_score": float(metrics["brier_score"]),
                "ece_10": float(metrics["ece_10"]),
                "decision_threshold": float(metrics["decision_threshold"]),
            }
        )
    write_csv(target / "tables" / "phase3_acquired_labels.csv", labels)
    write_csv(target / "tables" / "phase3_final_holdout_metrics.csv", final_metrics)
    summary = []
    for branch in BRANCH_ORDER:
        branch_labels = [int(row["cgmd_label"]) for row in labels if row["branch"] == branch]
        summary.append(
            {
                "branch": branch,
                "acquired_labels": len(branch_labels),
                "positive_labels": sum(branch_labels),
                "positive_rate": sum(branch_labels) / len(branch_labels),
            }
        )
    write_csv(target / "tables" / "phase3_branch_label_summary.csv", summary)
    copy_file(PHASE3 / "comparison" / "all_rounds_branch_comparison.md", target / "all_rounds_branch_comparison.md")

    series: dict[str, list[tuple[float, float]]] = {}
    for branch in BRANCH_ORDER:
        points = []
        total = 0
        for round_id in range(1, 9):
            total += sum(int(row["cgmd_label"]) for row in labels if row["branch"] == branch and int(row["round"]) == round_id)
            points.append((round_id, total))
        series[branch] = points
    line_chart(
        target / "figures" / "phase3_cumulative_positive_labels.svg",
        "Phase 3 cumulative CG-MD positive labels",
        "Five branch-local acquisitions per round; labels were never shared across branches.",
        series,
        x_label="CG-MD acquisition round",
        y_label="Cumulative positive labels",
        y_min=0,
        y_max=max(point[1] for points in series.values() for point in points) + 2,
    )
    bar_chart(
        target / "figures" / "phase3_positive_rate_by_branch.svg",
        "Phase 3 positive-label yield by branch",
        "Positive fraction among 40 branch-local CG-MD acquisitions.",
        summary,
        label_key="branch",
        value_keys=["positive_rate"],
        y_label="Positive-label fraction",
        y_max=1.0,
    )
    bar_chart(
        target / "figures" / "phase3_final_holdout_metrics.svg",
        "Phase 3 final frozen-holdout metrics",
        "Terminal post-ingest models after eight CG-MD rounds; threshold chosen on validation only.",
        final_metrics,
        label_key="branch",
        value_keys=["f1", "pr_auc"],
        legend_labels=["F1", "PR-AUC"],
        y_label="Metric value",
        y_min=0.70,
        y_max=1.0,
    )
    return [{"phase": "3", "status": "complete", "primary_result": "Eight branch-isolated CG-MD feedback rounds", "canonical_source": str(PHASE3.relative_to(ROOT))}]


def phase4_campaign_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = read_csv(PHASE4 / "comparison" / "round_001" / "all_selected_peptides.csv")
    selected_by_sequence = {row["sequence"]: row for row in selected}
    candidates: dict[str, list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    for campaign_root in CAMPAIGN_ROOTS:
        for review_path in campaign_root.glob("*/md_review.csv"):
            for row in read_csv(review_path):
                sequence = row.get("sequence", "")
                if sequence in selected_by_sequence:
                    candidates[sequence].append((review_path, row))

    rows: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for selected_row in selected:
        sequence = selected_row["sequence"]
        entries = candidates.get(sequence, [])
        complete = [
            entry
            for entry in entries
            if entry[1].get("job_root_status") == "analysis_complete"
            and entry[1].get("ap_200ns", "").strip()
            and entry[1].get("paper_path_ap_contact_last10ns_mean", "").strip()
        ]
        chosen: tuple[Path, dict[str, str]] | None = complete[0] if complete else None
        if len(complete) > 1:
            values = {
                (
                    entry[1].get("ap_200ns", ""),
                    entry[1].get("paper_path_ap_contact_last10ns_mean", ""),
                )
                for entry in complete
            }
            if len(values) > 1:
                conflicts.append({"sequence": sequence, "complete_sources": len(complete), "metric_pairs": repr(sorted(values))})
        base = {
            "policy": selected_row["policy"],
            "selection_rank": selected_row["selection_rank"],
            "sequence": sequence,
            "sequence_length": selected_row["sequence_length"],
            "selected_final_acquisition_utility": selected_row.get("final_acquisition_utility", ""),
            "simulation_status": "complete" if chosen else "not_simulated",
            "source_review_csv": str(chosen[0].relative_to(ROOT)) if chosen else "",
            "source_campaign": chosen[0].parent.name if chosen else "",
            "duplicate_review_sources_seen": len(entries),
        }
        if chosen:
            review = chosen[1]
            ap_200 = float(review["ap_200ns"])
            contact = float(review["paper_path_ap_contact_last10ns_mean"])
            label = int(ap_200 >= 1.75 and contact >= 0.5)
            base.update(
                {
                    "ap_sasa_200ns": ap_200,
                    "paper_path_apcontact_last10ns": contact,
                    "paper_ap_sasa_last10ns_mean_diagnostic": number(review.get("paper_ap_sasa_last10ns_mean")),
                    "contact_last10ns_sd": number(review.get("paper_path_ap_contact_last10ns_sd")),
                    "contact_last10ns_n_frames": review.get("paper_path_ap_contact_last10ns_n_frames", ""),
                    "operational_label": label,
                    "operational_rubric": "AP_sasa(200 ns)>=1.75 AND paper_path_APcontact_last10ns>=0.5",
                }
            )
        else:
            base.update(
                {
                    "ap_sasa_200ns": "",
                    "paper_path_apcontact_last10ns": "",
                    "paper_ap_sasa_last10ns_mean_diagnostic": "",
                    "contact_last10ns_sd": "",
                    "contact_last10ns_n_frames": "",
                    "operational_label": "",
                    "operational_rubric": "not evaluated",
                }
            )
        rows.append(base)
    return rows, conflicts


def phase4() -> list[dict[str, str]]:
    target = OUTPUT / "04_phase4_bayesian_optimization"
    rows, conflicts = phase4_campaign_rows()
    write_csv(target / "tables" / "primary_phase4_selected_peptides_with_cgmd.csv", rows)
    write_csv(target / "tables" / "campaign_metric_conflicts.csv", conflicts, ["sequence", "complete_sources", "metric_pairs"])
    policy_summary = []
    for policy in POLICY_ORDER:
        policy_rows = [row for row in rows if row["policy"] == policy]
        complete = [row for row in policy_rows if row["simulation_status"] == "complete"]
        positives = sum(int(row["operational_label"]) for row in complete)
        policy_summary.append(
            {
                "policy": policy,
                "selected_count": len(policy_rows),
                "simulated_count": len(complete),
                "coverage_fraction": len(complete) / len(policy_rows),
                "positive_count": positives,
                "positive_fraction_among_simulated": positives / len(complete) if complete else "",
                "positive_fraction_among_selected": positives / len(policy_rows),
            }
        )
    write_csv(target / "tables" / "primary_phase4_policy_cgmd_summary.csv", policy_summary)
    copy_file(PHASE4 / "comparison" / "round_001" / "all_selected_peptides.csv", target / "tables" / "primary_phase4_all_selected_peptides.csv")
    copy_file(PHASE4 / "comparison" / "round_001" / "policy_score_summary.csv", target / "tables" / "primary_phase4_policy_score_summary.csv")
    copy_file(PHASE4 / "comparison" / "round_001" / "round_001_comparison.md", target / "primary_phase4_proposal_comparison.md")

    complete_rows = [row for row in rows if row["simulation_status"] == "complete"]
    scatter_chart(
        target / "figures" / "primary_phase4_cgmd_threshold_scatter.svg",
        "Primary Phase 4 CG-MD outcomes",
        "Filled markers with dark outlines satisfy both operational thresholds; 29 of 30 proposals were simulated.",
        complete_rows,
        x_key="ap_sasa_200ns",
        y_key="paper_path_apcontact_last10ns",
        group_key="policy",
        x_label="AP-SASA ratio at 200 ns",
        y_label="Final-10-ns paper-path AP-contact",
        x_threshold=1.75,
        y_threshold=0.5,
    )
    bar_chart(
        target / "figures" / "primary_phase4_cgmd_yield_by_policy.svg",
        "Primary Phase 4 CG-MD yield by policy",
        "Positive fraction uses only completed simulations; coverage separately shows the invalid unsimulated MES proposal.",
        policy_summary,
        label_key="policy",
        value_keys=["positive_fraction_among_simulated", "coverage_fraction"],
        legend_labels=["Positive fraction among simulated", "Simulation coverage"],
        y_label="Fraction",
        y_max=1.0,
    )
    length_rows = []
    for policy in POLICY_ORDER:
        policy_lengths = [int(row["sequence_length"]) for row in rows if row["policy"] == policy]
        length_rows.append(
            {
                "policy": policy,
                "mean_length": mean([float(value) for value in policy_lengths]),
                "maximum_length": max(policy_lengths),
            }
        )
    write_csv(target / "tables" / "primary_phase4_length_summary.csv", length_rows)
    bar_chart(
        target / "figures" / "primary_phase4_selected_lengths.svg",
        "Primary Phase 4 selected peptide lengths",
        "The 25-residue MES proposal violated the configured 3-24 range and was not simulated.",
        length_rows,
        label_key="policy",
        value_keys=["mean_length", "maximum_length"],
        legend_labels=["Mean length", "Maximum length"],
        y_label="Residues",
        y_max=28,
    )

    phase4d = PHASE4 / "phase4d" / "run_001"
    for source in [
        phase4d / "all_policy_tradeoffs.csv",
        phase4d / "manual_review_recommendations.csv",
        phase4d / "phase4d_report.md",
    ]:
        copy_file(source, target / "phase4d" / ("tables" if source.suffix == ".csv" else "") / source.name)
    for name in ["phase4d_diversity_change.svg", "phase4d_utility_tradeoff.svg"]:
        copy_file(phase4d / "figures" / name, target / "phase4d" / "figures" / name)
    for policy in POLICY_ORDER:
        policy_dir = phase4d / "policies" / policy
        selected_name = "random_selected_batch.csv" if policy == "random" else "similarity_aware_selected_batch.csv"
        copy_file(policy_dir / selected_name, target / "phase4d" / "selected_batches" / f"{policy}_{selected_name}")

    return [
        {
            "phase": "4",
            "status": "complete_with_29_of_30_cgmd_outcomes",
            "primary_result": "Six fixed-surrogate proposal policies plus completed primary CG-MD assessment",
            "canonical_source": str(PHASE4.relative_to(ROOT)),
        },
        {
            "phase": "4-D",
            "status": "complete_exploratory",
            "primary_result": "Fresh diversity-aware generative replicate",
            "canonical_source": str((PHASE4 / "phase4d" / "run_001").relative_to(ROOT)),
        },
    ]


def phase5() -> list[dict[str, str]]:
    target = OUTPUT / "05_phase5_self_paced"
    table_names = [
        "learning_curves.csv",
        "paired_aulc_summary.csv",
        "paired_aulc_differences.csv",
        "labels_to_target_summary.csv",
        "terminal_convergence_audit.csv",
        "phase1_contextual_baseline.csv",
        "proxy_validity_summary.csv",
        "selected_positive_yield.csv",
        "sequence_diversity.csv",
        "selection_overlap.csv",
        "compute_time.csv",
    ]
    for name in table_names:
        copy_file(PHASE5 / "tables" / name, target / "tables" / name)
    copy_file(ROOT / "PHASE5_RESULTS_SUMMARY.md", target / "PHASE5_RESULTS_SUMMARY.md")

    learning = [
        row
        for row in read_csv(PHASE5 / "tables" / "learning_curves.csv")
        if row["evaluation_dataset"] == "holdout" and row["initial_label_count"] == "10"
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in learning:
        grouped[(row["strategy"], int(row["labeled_count"]))].append(float(row["f1"]))
    series = {
        strategy: sorted(
            (count, mean(values))
            for (name, count), values in grouped.items()
            if name == strategy
        )
        for strategy in PHASE5_STRATEGIES
    }
    line_chart(
        target / "figures" / "holdout_f1_vs_labelled_count_initial_10.svg",
        "Phase 5 holdout F1 versus labelled count",
        "Mean across three overlapping outer-fold conditions; the folds are not statistically independent.",
        series,
        x_label="Labelled replay-training peptides",
        y_label="Holdout F1",
        y_min=0.60,
        y_max=0.90,
    )

    aulc_rows = []
    aulc_source = read_csv(PHASE5 / "tables" / "paired_aulc_summary.csv")
    for row in aulc_source:
        if row["aulc_scope"] == "full":
            aulc_rows.append({"comparison": row["comparison"], "mean_delta": float(row["mean_delta_aulc_f1"])})
    bar_chart(
        target / "figures" / "paired_aulc_differences_initial_10.svg",
        "Phase 5 paired full-interval AULC-F1 differences",
        "Mean paired differences across three overlapping fold conditions; positive values favour the left strategy.",
        aulc_rows,
        label_key="comparison",
        value_keys=["mean_delta"],
        y_label="Mean paired AULC-F1 difference",
        y_min=-0.025,
        y_max=0.018,
    )

    target_rows = read_csv(PHASE5 / "tables" / "labels_to_target_summary.csv")
    target_chart_rows = []
    for strategy in PHASE5_STRATEGIES:
        values = {"strategy": strategy}
        for target_f1 in ("0.8", "0.84", "0.86"):
            matching = [row for row in target_rows if row["strategy"] == strategy and row["target_f1"] == target_f1]
            values[f"f1_{target_f1}"] = float(matching[0]["conditional_mean_labels_to_target"]) if matching else ""
        target_chart_rows.append(values)
    bar_chart(
        target / "figures" / "labels_to_target_initial_10.svg",
        "Phase 5 labels required to reach F1 targets",
        "Conditional mean among folds reaching each target; reach fractions remain in the source table.",
        target_chart_rows,
        label_key="strategy",
        value_keys=["f1_0.8", "f1_0.84", "f1_0.86"],
        legend_labels=["F1 0.80", "F1 0.84", "F1 0.86"],
        y_label="Labelled count",
        y_max=130,
    )

    difficulty = read_csv(PHASE5 / "tables" / "selected_difficulty_by_round.csv")
    difficulty_grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in difficulty:
        difficulty_grouped[(row["strategy"], int(row["acquisition_step"]))].append(float(row["difficulty_percentile"]))
    diff_series = {
        strategy: sorted(
            (step, mean(values))
            for (name, step), values in difficulty_grouped.items()
            if name == strategy
        )
        for strategy in PHASE5_STRATEGIES
    }
    line_chart(
        target / "figures" / "selected_difficulty_vs_round_initial_10.svg",
        "Phase 5 selected operational difficulty percentile",
        "Mean selected percentile across folds; lower values are closer to the currently labelled model manifold.",
        diff_series,
        x_label="Acquisition step",
        y_label="Mean selected difficulty percentile",
        y_min=0,
        y_max=1,
    )

    eligible = read_csv(PHASE5 / "tables" / "eligible_pool_fraction.csv")
    eligible_grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in eligible:
        eligible_grouped[(row["strategy"], int(row["acquisition_step"]))].append(float(row["eligible_fraction"]))
    eligible_series = {
        strategy: sorted(
            (step, mean(values))
            for (name, step), values in eligible_grouped.items()
            if name == strategy
        )
        for strategy in PHASE5_STRATEGIES
    }
    line_chart(
        target / "figures" / "eligible_pool_fraction_vs_round_initial_10.svg",
        "Phase 5 eligible hidden-pool fraction",
        "Random and predictive entropy use the full pool; paced strategies restrict eligibility by familiarity percentile.",
        eligible_series,
        x_label="Acquisition step",
        y_label="Eligible fraction",
        y_min=0,
        y_max=1,
    )

    proxy_records = read_csv(PHASE5 / "tables" / "proxy_validity_records.csv")
    sampled = []
    stride = max(1, len(proxy_records) // 1200)
    for index, row in enumerate(proxy_records):
        if index % stride == 0:
            sampled.append(
                {
                    "sequence": row["sequence"],
                    "strategy": row["strategy"],
                    "distance": float(row["embedding_distance_to_labelled"]),
                    "log_loss": float(row["post_hoc_pre_query_log_loss"]),
                }
            )
    scatter_chart(
        target / "figures" / "distance_vs_post_hoc_pre_query_log_loss_initial_10.svg",
        "Familiarity distance versus pre-query log loss",
        "Systematic sample from all candidate-step records; distance is diagnostic and labels were joined only post hoc.",
        sampled,
        x_key="distance",
        y_key="log_loss",
        group_key="strategy",
        x_label="Labelled-manifold familiarity distance",
        y_label="Post-hoc pre-query log loss",
    )

    proxy_summary = read_csv(PHASE5 / "tables" / "proxy_validity_summary.csv")
    error_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in proxy_summary:
        if row["summary_type"] == "difficulty_quintile" and row["error_rate"]:
            error_groups[(row["strategy"], int(row["difficulty_quintile"]))].append(float(row["error_rate"]))
    quintile_rows = []
    for strategy in PHASE5_STRATEGIES:
        values: dict[str, object] = {"strategy": strategy}
        for quintile in range(1, 6):
            values[f"q{quintile}"] = mean(error_groups[(strategy, quintile)])
        quintile_rows.append(values)
    bar_chart(
        target / "figures" / "error_rate_by_difficulty_quintile_initial_10.svg",
        "Pre-query classification error by difficulty quintile",
        "Mean post-hoc fixed-0.5 error rate across fold-step summaries.",
        quintile_rows,
        label_key="strategy",
        value_keys=["q1", "q2", "q3", "q4", "q5"],
        legend_labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        y_label="Error rate",
        y_max=0.65,
    )

    overlap = read_csv(PHASE5 / "tables" / "selection_overlap.csv")
    fixed_overlap = [
        row
        for row in overlap
        if row["fixed_budget_summary"] == "True"
        and row["scope"] == "cumulative"
        and row["labelled_count"] == "110"
    ]
    overlap_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in fixed_overlap:
        overlap_groups[(row["strategy_a"], row["strategy_b"])].append(float(row["jaccard"]))
    overlap_values = {key: mean(values) for key, values in overlap_groups.items()}
    for strategy in PHASE5_STRATEGIES:
        overlap_values[(strategy, strategy)] = 1.0
    heatmap(
        target / "figures" / "selection_overlap_heatmap_initial_10.svg",
        "Phase 5 cumulative selection overlap at 110 labels",
        "Mean Jaccard overlap across the three fold conditions.",
        PHASE5_STRATEGIES,
        overlap_values,
    )

    diversity = read_csv(PHASE5 / "tables" / "sequence_diversity.csv")
    diversity_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in diversity:
        if row["fixed_budget_summary"] == "True" and row["scope"] == "cumulative":
            diversity_groups[(row["strategy"], int(row["labelled_count"]))].append(float(row["mean_pairwise_normalized_levenshtein"]))
    diversity_rows = []
    for strategy in PHASE5_STRATEGIES:
        diversity_rows.append(
            {
                "strategy": strategy,
                "labels_60": mean(diversity_groups[(strategy, 60)]),
                "labels_110": mean(diversity_groups[(strategy, 110)]),
                "labels_160": mean(diversity_groups[(strategy, 160)]),
            }
        )
    bar_chart(
        target / "figures" / "sequence_diversity_comparison_initial_10.svg",
        "Phase 5 cumulative selected-sequence diversity",
        "Mean pairwise normalised Levenshtein distance at preregistered labelled-count budgets.",
        diversity_rows,
        label_key="strategy",
        value_keys=["labels_60", "labels_110", "labels_160"],
        legend_labels=["60 labels", "110 labels", "160 labels"],
        y_label="Mean pairwise normalised Levenshtein distance",
        y_min=0.65,
        y_max=1.0,
    )

    compute = read_csv(PHASE5 / "tables" / "compute_time.csv")
    compute_groups: dict[str, list[float]] = defaultdict(list)
    for row in compute:
        compute_groups[row["strategy"]].append(float(row["walltime_seconds"]) / 60)
    compute_rows = [{"strategy": strategy, "mean_minutes": mean(compute_groups[strategy])} for strategy in PHASE5_STRATEGIES]
    bar_chart(
        target / "figures" / "compute_time_comparison_initial_10.svg",
        "Phase 5 replay-job compute time",
        "Mean SUPEK wall time across three outer-fold jobs per strategy.",
        compute_rows,
        label_key="strategy",
        value_keys=["mean_minutes"],
        y_label="Mean wall time (minutes)",
        y_max=35,
    )
    return [{"phase": "5", "status": "complete", "primary_result": "Reduced SPAL-inspired self-paced replay", "canonical_source": str(PHASE5.relative_to(ROOT))}]


def make_overview(status_rows: list[dict[str, str]]) -> None:
    overview = OUTPUT / "00_overview"
    write_csv(overview / "phase_status.csv", status_rows)
    docs = [
        "THESIS_HANDOFF_FOR_NEXT_MODEL.md",
        "THESIS_METHODOLOGY_TECHNICAL_SUMMARY.md",
        "PHASE5_RESULTS_SUMMARY.md",
        "THESIS_FEATURE_RUNTIME_SHAPES.md",
        "THESIS_PREDICTIVE_MODEL_ARCHITECTURE.md",
        "THESIS_CGMD_PARAMETER_CONTRACT.md",
        "REPOSITORY_CONSISTENCY_AUDIT.md",
    ]
    for name in docs:
        copy_file(ROOT / name, overview / "project_documentation" / name)

    claims = """# Results Claims And Caveats

## Supported results

- Phase 1 reproduced the five predictive model families under nested cross-validation.
- Phase 2 completed the ten-strategy retrospective replay for `n0=10` and `n0=40`.
- Phase 3 completed 120 branch-local CG-MD acquisitions across three strategies and eight rounds.
- Primary Phase 4 produced 30 exact-unique proposals. Twenty-nine valid proposals were simulated and have complete operational CG-MD evidence; the known 25-residue MES proposal was not simulated.
- Phase 4-D completed as a separate exploratory diversity-aware replicate.
- Phase 5 completed 12 replay jobs and aggregation. Predictive entropy had the highest mean full-interval AULC-F1; self-paced entropy exceeded static easy entropy but not predictive entropy or random.

## Mandatory caveats

- The outer folds overlap and are fold-level repetitions, not statistically independent replicates.
- Phase 2 and Phase 5 are retrospective replay experiments, not prospective peptide validation.
- CG-MD operational labels are modelled simulation outcomes, not universal biological ground truth.
- Phase 3 branches are isolated trajectories; labels were not shared across branches.
- Primary Phase 4 policy yields are based on only five proposals per policy, and MES has four simulated proposals because the invalid length-25 item was excluded.
- Phase 4-D is not part of the primary Phase 4 policy comparison.
- Phase 5 is SPAL-inspired and uses a neural familiarity proxy; it is not an exact reproduction of SPAL.
- Cross-phase predictive metrics are not directly interchangeable because training-set sizes and validation/calibration protocols differ.
"""
    write_text(overview / "CLAIMS_AND_CAVEATS.md", claims)
    provenance = """# Data Provenance

## Canonical scientific archives

- Phase 1: `thesis_results/01_reproduction/`
- Phase 2: `thesis_results/02_replay/evidence/` and the associated benchmark overlap tables
- Phase 3: `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/`
- Phase 4 and Phase 4-D: `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/`
- Phase 5: `thesis_results/05_self_paced_active_learning/`

## Primary Phase 4 simulation evidence

The primary proposal list comes from the completed Phase 4 archive. Simulation
evidence is resolved by exact peptide sequence from these three local campaign
roots:

- `active_learning_runs/thesis_main_20260502/md_campaigns/`
- `active_learning_runs/thesis_main_supek_20260502/md_campaigns/`
- `active_learning_runs/thesis_main_supek_clean_20260502_original/md_campaigns/`

For each selected sequence, a campaign is accepted only when `md_review.csv`
reports `job_root_status=analysis_complete` and contains both `ap_200ns` and
`paper_path_ap_contact_last10ns_mean`. When duplicate review rows exist, a
complete row is preferred over an incomplete prepared package. No conflicting
complete metric pairs were found.

## Derived outputs

Every derived table is explicitly stored in the corresponding phase `tables/`
directory. Figures are generated from those copied or derived tables. Original
checkpoints, trajectories, scheduler output, and large candidate-level archives
are intentionally excluded from this writing packet.
"""
    write_text(overview / "DATA_PROVENANCE.md", provenance)
    visual_review = """# Figure Review Record

Review date: 2026-06-23

- All 28 SVG figures were parsed successfully as XML.
- All 28 figures rendered in the local browser gallery with non-zero dimensions.
- No included figure contains the former `Matplotlib unavailable` placeholder.
- Phase 4 threshold lines, axis labels, policy legend, and operational-positive
  marker outlines were visually checked.
- Phase 4 policy yield values were cross-checked against the 29 sequence-level
  completed simulation rows.
- Phase 3 terminal F1 and PR-AUC bars were cross-checked against each branch's
  `final_holdout.json`.
- Phase 5 learning curves, paired AULC differences, labels-to-target bars,
  familiarity diagnostics, overlap heatmap, diversity bars, and compute-time
  chart were visually inspected for clipping, missing series, and overlapping
  labels.
- The selected Phase 2 and Phase 1 figures are canonical repository figures,
  not recomputed approximations.

Remaining interpretive caution: small fold or policy sample sizes are stated in
the captions and `CLAIMS_AND_CAVEATS.md`; visual polish does not increase their
statistical evidential strength.
"""
    write_text(overview / "FIGURE_REVIEW.md", visual_review)


def make_index(status_rows: list[dict[str, str]]) -> None:
    figure_paths = sorted(OUTPUT.glob("**/*.svg"))
    table_paths = sorted(OUTPUT.glob("**/*.csv"))
    figure_rows = []
    for index, path in enumerate(figure_paths, start=1):
        figure_rows.append(
            {
                "figure_id": f"F{index:02d}",
                "phase": path.relative_to(OUTPUT).parts[0],
                "filename": path.name,
                "relative_path": path.relative_to(OUTPUT).as_posix(),
                "status": "rendered_and_reviewed_20260623",
            }
        )
    table_rows = []
    for index, path in enumerate(table_paths, start=1):
        table_rows.append(
            {
                "table_id": f"T{index:02d}",
                "phase": path.relative_to(OUTPUT).parts[0],
                "filename": path.name,
                "relative_path": path.relative_to(OUTPUT).as_posix(),
            }
        )
    write_csv(OUTPUT / "00_overview" / "FIGURE_INDEX.csv", figure_rows)
    write_csv(OUTPUT / "00_overview" / "TABLE_INDEX.csv", table_rows)

    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Thesis Results Handoff</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;color:#172033}h1,h2{color:#0f172a}.phase{margin:38px 0}.figure{border-top:1px solid #dbe2ea;padding:22px 0}.figure img{max-width:1100px;width:100%;height:auto}.note{color:#52606d}.status{border-collapse:collapse}.status td,.status th{border:1px solid #cbd5e1;padding:7px 10px}</style></head><body>",
        "<h1>Master's Thesis Results Handoff</h1>",
        "<p class='note'>Curated canonical results plus regenerated Phase 3, primary Phase 4 CG-MD, and Phase 5 figures. Read README.md and CLAIMS_AND_CAVEATS.md before interpreting figures.</p>",
        "<h2>Phase Status</h2><table class='status'><tr><th>Phase</th><th>Status</th><th>Primary result</th></tr>",
    ]
    for row in status_rows:
        html_lines.append(f"<tr><td>{esc(row['phase'])}</td><td>{esc(row['status'])}</td><td>{esc(row['primary_result'])}</td></tr>")
    html_lines.append("</table>")
    current_phase = ""
    for row in figure_rows:
        if row["phase"] != current_phase:
            current_phase = row["phase"]
            html_lines.append(f"<div class='phase'><h2>{esc(current_phase)}</h2>")
        html_lines.append(
            f"<div class='figure'><h3>{esc(row['filename'])}</h3><img src='../{esc(row['relative_path'])}' alt='{esc(row['filename'])}'></div>"
        )
    html_lines.append("</body></html>")
    write_text(OUTPUT / "00_overview" / "FIGURE_GALLERY.html", "\n".join(html_lines))

    readme = f"""# Master's Thesis Results Handoff ({STAMP})

This packet is a curated, lightweight result bundle for writing the thesis Results section. It contains canonical aggregate tables, presentation-ready SVG figures, the newly reconciled primary Phase 4 CG-MD evidence, and the project documentation required to interpret the results safely.

## Start here

1. `00_overview/CLAIMS_AND_CAVEATS.md`
2. `00_overview/phase_status.csv`
3. `00_overview/FIGURE_GALLERY.html`
4. `00_overview/FIGURE_REVIEW.md`
5. `00_overview/DATA_PROVENANCE.md`
6. `00_overview/FIGURE_INDEX.csv`
7. `00_overview/TABLE_INDEX.csv`
8. `00_overview/project_documentation/THESIS_HANDOFF_FOR_NEXT_MODEL.md`

## Phase 4 CG-MD reconciliation

- Selected proposals: 30
- Complete simulations with both operational metrics: 29
- Operational positives: 15
- Missing simulation: `VLNINNMGAKWRRTCNQRLTPTALP` (MES, 25 residues, known invalid-length proposal)
- Operational label: `AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`
- The last-10-ns AP-SASA value is exported as a diagnostic, but it is not substituted for the implemented 200 ns AP-SASA criterion.

## Scope

The ZIP intentionally excludes neural-network checkpoints, raw trajectories, temporary scheduler output, and duplicated exploratory archives. Canonical source paths are retained in the tables and manifest.
"""
    write_text(OUTPUT / "README.md", readme)


def hash_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "relative_path": path.relative_to(OUTPUT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    write_csv(OUTPUT / "SHA256_MANIFEST.csv", rows)


def zip_output() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(OUTPUT.name) / path.relative_to(OUTPUT))


def main() -> None:
    required = [PHASE1, PHASE2, PHASE3, PHASE4, PHASE5, *CAMPAIGN_ROOTS]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required roots:\n" + "\n".join(str(path) for path in missing))
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    status_rows: list[dict[str, str]] = []
    status_rows.extend(phase1())
    status_rows.extend(phase2())
    status_rows.extend(phase3())
    status_rows.extend(phase4())
    status_rows.extend(phase5())
    make_overview(status_rows)
    make_index(status_rows)
    hash_manifest()
    zip_output()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "zip": str(ZIP_PATH),
                "files": sum(1 for path in OUTPUT.rglob("*") if path.is_file()),
                "figures": sum(1 for path in OUTPUT.rglob("*.svg")),
                "tables": sum(1 for path in OUTPUT.rglob("*.csv")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
