"""Generate the Phase 2 calibration/ensemble ablation thesis figure.

The figure is derived only from
``thesis_results/02_replay/evidence/ablation_summary.csv``.  It uses holdout
rows and averages the canonical strategy-level means across the six ablation
strategies for each initial-size/configuration pair, because the target figure
has no acquisition-strategy dimension.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = REPO_ROOT / "thesis_results" / "02_replay" / "evidence" / "ablation_summary.csv"
OUT_DIR = REPO_ROOT / "thesis_results" / "THESIS_RESULTS_HANDOFF_20260623" / "02_phase2_replay"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SVG_OUT = FIG_DIR / "phase2_calibration_ensemble_ablation.svg"
PDF_OUT = FIG_DIR / "phase2_calibration_ensemble_ablation.pdf"
VALUES_OUT = TABLE_DIR / "phase2_calibration_ensemble_ablation_values.csv"
MANIFEST_OUT = OUT_DIR / "phase2_calibration_ensemble_ablation_manifest.json"

SETUP_ORDER = [
    ("single_raw", "Single raw"),
    ("single_calibrated", "Single calibrated"),
    ("ensemble_raw", "Ensemble raw"),
    ("ensemble_calibrated", "Ensemble calibrated"),
]
INITIAL_ORDER = [10, 40]
METRICS = [
    ("mean_AULC_F1", "Mean normalised AULC-F1", "higher"),
    ("mean_final_F1", "Mean terminal F1", "higher"),
    ("mean_final_Brier", "Mean terminal Brier score", "lower"),
    ("mean_final_ECE_10", "Mean terminal ECE-10", "lower"),
]

COLOURS = {10: "#0072B2", 40: "#D55E00"}
MARKERS = {10: "circle", 40: "diamond"}


def repo_commit() -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_source_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_values(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    required = {
        "initial_label_count",
        "setup",
        "strategy",
        "evaluation_dataset",
        "mean_AULC_F1",
        "mean_final_F1",
        "mean_final_Brier",
        "mean_final_ECE_10",
    }
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    warnings: list[str] = []
    output: list[dict[str, str]] = []
    for initial in INITIAL_ORDER:
        for setup, label in SETUP_ORDER:
            subset = [
                row
                for row in rows
                if row["evaluation_dataset"] == "holdout"
                and int(row["initial_label_count"]) == initial
                and row["setup"] == setup
            ]
            strategies = sorted({row["strategy"] for row in subset})
            if len(subset) != 6:
                warnings.append(
                    f"{setup}, n0={initial}: expected 6 holdout strategy rows, found {len(subset)}"
                )
            if len(strategies) != len(subset):
                warnings.append(f"{setup}, n0={initial}: duplicate strategy rows detected")
            if not subset:
                raise ValueError(f"No holdout rows for setup={setup}, n0={initial}")

            record = {"initial_size": str(initial), "configuration": label}
            for metric, *_ in METRICS:
                vals = [float(row[metric]) for row in subset]
                if any(not math.isfinite(v) for v in vals):
                    raise ValueError(f"Non-finite {metric} for setup={setup}, n0={initial}")
                record[metric] = format(sum(vals) / len(vals), ".15g")
            output.append(record)
    return output, warnings


def write_values_csv(records: list[dict[str, str]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "initial_size",
        "configuration",
        "mean_AULC_F1",
        "mean_final_F1",
        "mean_final_Brier",
        "mean_final_ECE_10",
    ]
    with VALUES_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def metric_range(records: list[dict[str, str]], metric: str) -> tuple[float, float]:
    vals = [float(record[metric]) for record in records]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.28, 0.002)
    if metric in {"mean_final_Brier", "mean_final_ECE_10"}:
        lo = max(0.0, lo - pad)
    else:
        lo = lo - pad
    hi = hi + pad
    return lo, hi


def nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw_step = (hi - lo) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 2.5, 5, 10]
    step = min(candidates, key=lambda c: abs(c * power - raw_step)) * power
    start = math.ceil(lo / step) * step
    ticks = []
    val = start
    while val <= hi + step * 0.5:
        ticks.append(round(val, 10))
        val += step
    return ticks


def text(x: float, y: float, value: str, size: int = 14, anchor: str = "middle", weight: str | None = None, rotate: str | None = None, colour: str = "#0f172a") -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'text-anchor="{anchor}"',
        'font-family="Arial, sans-serif"',
        f'font-size="{size}"',
        f'fill="{colour}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if rotate:
        attrs.append(f'transform="{rotate}"')
    return f"<text {' '.join(attrs)}>{escape(value)}</text>"


def marker(x: float, y: float, initial: int) -> str:
    colour = COLOURS[initial]
    if MARKERS[initial] == "diamond":
        points = [
            (x, y - 6),
            (x + 6, y),
            (x, y + 6),
            (x - 6, y),
        ]
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        return f'<polygon points="{pts}" fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>'


def generate_svg(records: list[dict[str, str]]) -> str:
    width, height = 1120, 780
    panel_w, panel_h = 430, 250
    panels = [(95, 70), (620, 70), (95, 410), (620, 410)]
    x_offsets = {10: -13, 40: 13}
    setup_labels = [label for _, label in SETUP_ORDER]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<style>text{dominant-baseline:auto}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#0f172a;stroke-width:2.2}</style>',
    ]

    for (metric, ylabel, _direction), (px, py) in zip(METRICS, panels):
        lo, hi = metric_range(records, metric)
        ticks = nice_ticks(lo, hi, 5)
        def sx(index: int, initial: int) -> float:
            return px + 58 + index * ((panel_w - 90) / 3) + x_offsets[initial]
        def sy(value: float) -> float:
            return py + panel_h - 46 - ((value - lo) / (hi - lo)) * (panel_h - 78)

        lines.append(text(px + panel_w / 2, py + panel_h + 48, "Configuration", size=20))
        cy = py + panel_h / 2
        lines.append(text(px - 78, cy, ylabel, size=20, rotate=f"rotate(-90 {px-78:.1f} {cy:.1f})"))
        lines.append(f'<line class="axis" x1="{px}" y1="{py+panel_h-46}" x2="{px+panel_w}" y2="{py+panel_h-46}"/>')
        lines.append(f'<line class="axis" x1="{px}" y1="{py}" x2="{px}" y2="{py+panel_h-46}"/>')

        for tick in ticks:
            y = sy(tick)
            lines.append(f'<line class="grid" x1="{px}" y1="{y:.1f}" x2="{px+panel_w}" y2="{y:.1f}"/>')
            lines.append(text(px - 12, y + 6, f"{tick:.3f}", size=16, anchor="end", colour="#0f172a"))

        for i, label in enumerate(setup_labels):
            x_mid = px + 58 + i * ((panel_w - 90) / 3)
            lines.append(f'<line x1="{x_mid:.1f}" y1="{py}" x2="{x_mid:.1f}" y2="{py+panel_h-46}" stroke="#f1f5f9" stroke-width="1"/>')
            display = label.replace(" ", "\n")
            parts = display.split("\n")
            for j, part in enumerate(parts):
                lines.append(text(x_mid, py + panel_h - 23 + j * 18, part, size=15, colour="#0f172a"))

        rec_lookup = {(int(r["initial_size"]), r["configuration"]): r for r in records}
        for i, label in enumerate(setup_labels):
            for initial in INITIAL_ORDER:
                value = float(rec_lookup[(initial, label)][metric])
                lines.append(marker(sx(i, initial), sy(value), initial))

    # Shared legend.
    lx, ly = 430, 12
    lines.append(marker(lx, ly, 10))
    lines.append(text(lx + 18, ly + 6, "n0 = 10", size=18, anchor="start"))
    lines.append(marker(lx + 120, ly, 40))
    lines.append(text(lx + 138, ly + 6, "n0 = 40", size=18, anchor="start"))

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def export_pdf(records: list[dict[str, str]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - local dependency check
        raise RuntimeError("Matplotlib is required for local PDF export") from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    x = list(range(len(SETUP_ORDER)))
    setup_labels = [label.replace(" ", "\n") for _, label in SETUP_ORDER]
    marker_lookup = {10: "o", 40: "D"}
    rec_lookup = {(int(r["initial_size"]), r["configuration"]): r for r in records}

    for ax, (metric, ylabel, _direction) in zip(axes.ravel(), METRICS):
        for initial in INITIAL_ORDER:
            values = [
                float(rec_lookup[(initial, label)][metric])
                for _, label in SETUP_ORDER
            ]
            ax.plot(
                x,
                values,
                linestyle="none",
                marker=marker_lookup[initial],
                markersize=5.6,
                color=COLOURS[initial],
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=f"n0 = {initial}",
            )
        lo, hi = metric_range(records, metric)
        ax.set_ylim(lo, hi)
        ax.set_xticks(x)
        ax.set_xticklabels(setup_labels)
        ax.set_xlabel("Configuration", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.grid(axis="x", color="#f1f5f9", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="#0f172a", width=0.9, length=3.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)


def verify_values(records: list[dict[str, str]]) -> None:
    with VALUES_OUT.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    if records != written:
        raise AssertionError("Written values CSV does not match computed records")
    for record in records:
        for metric, *_ in METRICS:
            float(record[metric])


def main() -> None:
    rows = read_source_rows()
    records, warnings = aggregate_values(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_values_csv(records)
    SVG_OUT.write_text(generate_svg(records), encoding="utf-8")
    export_pdf(records)
    verify_values(records)

    manifest = {
        "source_file": str(SOURCE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generation_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
        "repository_commit": repo_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_svg": str(SVG_OUT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_pdf": str(PDF_OUT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "values_csv": str(VALUES_OUT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "filter": {
            "evaluation_dataset": "holdout",
            "aggregation": "arithmetic mean across the six canonical ablation strategy rows for each setup and initial_label_count",
            "error_bars": "not plotted; no canonical variability columns correspond exactly to these setup-level means",
        },
        "warnings": warnings,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
