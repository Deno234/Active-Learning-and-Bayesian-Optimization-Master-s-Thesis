from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from active_learning_thesis.config import (
    DEFAULT_REPLAY_STRATEGIES,
    THESIS_FULL_REPLAY_STRATEGIES,
)
from active_learning_thesis.study import (
    DEFAULT_STUDY_COMPARISON_DIRNAME,
    DEFAULT_STUDY_OUTPUT_DIRNAME,
    DEFAULT_STUDY_ROOT_DIRNAME,
)


STUDY_PRESETS: dict[str, dict[str, object]] = {
    "Quick smoke": {
        "description": "Tiny multi-seed check for the study machinery before spending real compute.",
        "study_name_prefix": "gui_study_smoke",
        "seeds": 2,
        "seed_start": 20260317,
        "seed_step": 1009,
        "epochs": 3,
        "max_rounds": 1,
        "batch_size": 2,
        "candidate_pool_min": 20,
        "replay_seed_size": 12,
        "ensemble_size": 3,
        "real_strategy": "ensemble_mi",
        "strategies": ["random", "ensemble_mi"],
        "metric": "f1",
        "target": "",
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Strategy comparison": {
        "description": "Normal thesis comparison of acquisition strategies across multiple seeds.",
        "study_name_prefix": "strategy_comparison",
        "seeds": 5,
        "seed_start": 20260317,
        "seed_step": 1009,
        "epochs": 70,
        "max_rounds": 10,
        "batch_size": 5,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "ensemble_size": 5,
        "real_strategy": "ensemble_mi",
        "strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "metric": "f1",
        "target": "",
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Thesis full": {
        "description": "Full ten-strategy comparison for the final thesis acquisition-function table.",
        "study_name_prefix": "thesis_full",
        "seeds": 5,
        "seed_start": 20260317,
        "seed_step": 1009,
        "epochs": 70,
        "max_rounds": 10,
        "batch_size": 5,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "ensemble_size": 5,
        "real_strategy": "ensemble_mi",
        "strategies": list(THESIS_FULL_REPLAY_STRATEGIES),
        "metric": "f1",
        "target": "",
        "train_family_for_init": True,
        "use_calibrated_acquisition": True,
    },
    "Seed sensitivity": {
        "description": "More seeds with the standard strategy set to test robustness of the thesis conclusion.",
        "study_name_prefix": "seed_sensitivity",
        "seeds": 10,
        "seed_start": 20260317,
        "seed_step": 1009,
        "epochs": 70,
        "max_rounds": 10,
        "batch_size": 5,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "ensemble_size": 5,
        "real_strategy": "ensemble_mi",
        "strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "metric": "f1",
        "target": "",
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Calibration ablation": {
        "description": "Matched study plan for comparing calibrated versus raw acquisition/reporting behavior.",
        "study_name_prefix": "calibration_ablation",
        "seeds": 5,
        "seed_start": 20260317,
        "seed_step": 1009,
        "epochs": 70,
        "max_rounds": 10,
        "batch_size": 5,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "ensemble_size": 5,
        "real_strategy": "ensemble_mi",
        "strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "metric": "f1",
        "target": "",
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
}


def normalize_study_name(value: str, *, fallback: str = "dashboard_study") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or fallback


def parse_float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path, *, limit: int = 250) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def discover_study_manifests(run_root: Path) -> list[dict[str, object]]:
    study_root = run_root / DEFAULT_STUDY_ROOT_DIRNAME
    rows: list[dict[str, object]] = []
    for path in sorted(study_root.glob("*/study_manifest.json")):
        payload = read_json_file(path)
        if not payload:
            continue
        config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        rows.append(
            {
                "study_name": payload.get("study_name", path.parent.name),
                "status": payload.get("status", ""),
                "runs": payload.get("run_count", len(list(payload.get("runs", []))) if isinstance(payload.get("runs", []), list) else 0),
                "completed": payload.get("completed_run_count", ""),
                "failures": payload.get("failure_count", 0),
                "metric": config.get("metric", ""),
                "seeds": config.get("seed_count", ""),
                "strategies": ", ".join(str(item) for item in list(config.get("replay_strategies", []))[:6]) if isinstance(config.get("replay_strategies", []), list) else "",
                "manifest": str(path),
                "updated_at": payload.get("updated_at", ""),
            }
        )
    rows.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
    return rows


def discover_study_summaries(run_root: Path) -> list[dict[str, object]]:
    candidate_paths = list((run_root / DEFAULT_STUDY_OUTPUT_DIRNAME).glob("*_study_summary.json"))
    candidate_paths.extend((run_root / DEFAULT_STUDY_ROOT_DIRNAME).glob("*/evidence/*_study_summary.json"))
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(candidate_paths):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        payload = read_json_file(path)
        if not payload:
            continue
        rows.append(
            {
                "metric": payload.get("metric", ""),
                "run_count": payload.get("run_count", 0),
                "strategy_count": payload.get("strategy_count", 0),
                "paired_vs_random": payload.get("paired_vs_random_count", 0),
                "best_strategy": payload.get("best_strategy_by_aulc", ""),
                "summary": str(path),
                "output_dir": payload.get("output_dir", str(path.parent)),
            }
        )
    rows.sort(key=lambda row: str(row.get("summary", "")), reverse=True)
    return rows


def discover_study_comparisons(run_root: Path) -> list[dict[str, object]]:
    comparison_root = run_root / DEFAULT_STUDY_ROOT_DIRNAME / DEFAULT_STUDY_COMPARISON_DIRNAME
    rows: list[dict[str, object]] = []
    for path in sorted(comparison_root.glob("**/*_study_comparison_summary.json")):
        payload = read_json_file(path)
        if not payload:
            continue
        rows.append(
            {
                "baseline": payload.get("baseline_study", ""),
                "candidate": payload.get("candidate_study", ""),
                "metric": payload.get("metric", ""),
                "paired_count": payload.get("paired_count", 0),
                "strategy_count": payload.get("strategy_count", 0),
                "best_strategy": payload.get("best_strategy_by_aulc_advantage", ""),
                "summary": str(path),
                "output_dir": payload.get("output_dir", str(path.parent)),
            }
        )
    rows.sort(key=lambda row: str(row.get("summary", "")), reverse=True)
    return rows


def study_manifest_options(run_root: Path) -> list[str]:
    return [str(row["manifest"]) for row in discover_study_manifests(run_root)]
