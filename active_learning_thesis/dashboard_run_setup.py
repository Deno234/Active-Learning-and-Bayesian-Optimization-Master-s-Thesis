from __future__ import annotations

import re
from pathlib import Path

from active_learning_thesis.config import (
    DEFAULT_REPLAY_STRATEGIES,
    THESIS_FULL_REPLAY_STRATEGIES,
    RunConfig,
)
from active_learning_thesis.dashboard_curation import pin_dashboard_run, set_dashboard_run_label
from active_learning_thesis.workflow import init_run, run_replay


RUN_SETUP_PRESETS: dict[str, dict[str, object]] = {
    "Thesis baseline": {
        "description": "Default thesis run: full baseline training, normal batch size, and the standard replay strategy set.",
        "run_name_prefix": "thesis_baseline",
        "random_seed": 20260317,
        "batch_size": 5,
        "max_rounds": 10,
        "epochs": 70,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "real_strategy": "ensemble_mi",
        "replay_strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Quick smoke": {
        "description": "Fast local confidence check: tiny replay horizon and fewer epochs, useful before touching real thesis runs.",
        "run_name_prefix": "gui_smoke",
        "random_seed": 20260317,
        "batch_size": 2,
        "max_rounds": 1,
        "epochs": 3,
        "candidate_pool_min": 20,
        "replay_seed_size": 12,
        "real_strategy": "ensemble_mi",
        "replay_strategies": ["random", "ensemble_mi"],
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Full AL run": {
        "description": "Heavier production-style run with a larger candidate pool and the standard strategy set.",
        "run_name_prefix": "full_al_run",
        "random_seed": 20260317,
        "batch_size": 5,
        "max_rounds": 10,
        "epochs": 90,
        "candidate_pool_min": 80,
        "replay_seed_size": 40,
        "real_strategy": "ensemble_mi",
        "replay_strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "train_family_for_init": False,
        "use_calibrated_acquisition": True,
    },
    "Strategy ablation": {
        "description": "Replay-focused setup for comparing acquisition strategies from the same initial split.",
        "run_name_prefix": "strategy_ablation",
        "random_seed": 20260317,
        "batch_size": 5,
        "max_rounds": 10,
        "epochs": 70,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "real_strategy": "ensemble_mi",
        "replay_strategies": list(DEFAULT_REPLAY_STRATEGIES),
        "train_family_for_init": True,
        "use_calibrated_acquisition": True,
    },
    "Thesis full": {
        "description": "Full thesis comparison with exploitation, entropy, UCB, ensemble, committee, diversity, and OED acquisition strategies.",
        "run_name_prefix": "thesis_full",
        "random_seed": 20260317,
        "batch_size": 5,
        "max_rounds": 10,
        "epochs": 70,
        "candidate_pool_min": 50,
        "replay_seed_size": 40,
        "real_strategy": "ensemble_mi",
        "replay_strategies": list(THESIS_FULL_REPLAY_STRATEGIES),
        "train_family_for_init": True,
        "use_calibrated_acquisition": True,
    },
}


def normalize_run_name(value: str, *, fallback: str = "dashboard_run") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or fallback


def parse_strategy_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        parts = re.split(r"[,;\s]+", str(value or ""))
    strategies: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        strategies.append(part)
    return strategies


def coerce_positive_int(value: object, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)
    return max(int(minimum), parsed)


def run_setup_defaults(preset_name: str, *, clone_run_dir: Path | None = None) -> dict[str, object]:
    if clone_run_dir is not None:
        config_path = clone_run_dir / "config.json"
        if config_path.exists():
            config = RunConfig.load(config_path)
            payload = config.to_dict()
            payload["description"] = f"Cloned from {clone_run_dir.name}."
            payload["run_name_prefix"] = f"{normalize_run_name(config.run_name)}_copy"
            return payload
    return dict(RUN_SETUP_PRESETS.get(preset_name, RUN_SETUP_PRESETS["Thesis baseline"]))


def build_run_setup_readiness(run_root: Path, *, run_name: str) -> dict[str, object]:
    normalized_name = normalize_run_name(run_name, fallback="")
    target_dir = run_root / normalized_name if normalized_name else run_root
    blockers: list[str] = []
    cautions: list[str] = []
    if not normalized_name:
        blockers.append("Run name is empty.")
    if target_dir.exists():
        blockers.append(f"Run directory already exists: {target_dir}")
    if not run_root.exists():
        cautions.append(f"Run root will be created: {run_root}")
    if blockers:
        return {
            "verdict": "Blocked",
            "summary": "The new run cannot be created yet.",
            "blockers": blockers,
            "cautions": cautions,
            "fix_now": "Choose a unique run folder name before launching the setup action.",
            "disable_button": True,
        }
    return {
        "verdict": "Ready" if not cautions else "Ready with caution",
        "summary": f"Ready to create `{normalized_name}` under `{run_root}`.",
        "blockers": [],
        "cautions": cautions,
        "fix_now": "Launch the setup action when the name, preset, seed, and replay choice look right.",
        "disable_button": False,
    }


def dashboard_init_run(
    *,
    run_root: Path,
    run_name: str,
    random_seed: int,
    batch_size: int,
    max_rounds: int,
    epochs: int,
    candidate_pool_min: int,
    replay_seed_size: int,
    real_strategy: str,
    replay_strategies: list[str] | None = None,
    train_family_for_init: bool = False,
    use_calibrated_acquisition: bool = True,
    generator_objective_mode: str = "match_acquisition",
    use_similarity_penalty: bool = False,
    use_length_penalty: bool = True,
    binary_threshold_strategy: str = "pr_best_f1",
    pin_run: bool = True,
    run_label: str = "",
    train_baseline_after_init: bool = True,
    run_replay_after_init: bool = False,
) -> dict[str, object]:
    normalized_name = normalize_run_name(run_name, fallback="")
    if not normalized_name:
        raise ValueError("Run name is empty.")
    config = RunConfig(
        run_name=normalized_name,
        output_root=str(run_root),
        random_seed=int(random_seed),
        batch_size=int(batch_size),
        max_rounds=int(max_rounds),
        epochs=int(epochs),
        candidate_pool_min=int(candidate_pool_min),
        replay_seed_size=int(replay_seed_size),
        real_strategy=str(real_strategy or "ensemble_mi").strip() or "ensemble_mi",
        replay_strategies=replay_strategies or list(DEFAULT_REPLAY_STRATEGIES),
        train_family_for_init=bool(train_family_for_init),
        use_calibrated_acquisition=bool(use_calibrated_acquisition),
        generator_objective_mode=str(generator_objective_mode),
        use_similarity_penalty=bool(use_similarity_penalty),
        use_length_penalty=bool(use_length_penalty),
        binary_threshold_strategy=str(binary_threshold_strategy),
    )
    run_dir = init_run(config, train_baseline=bool(train_baseline_after_init))
    if pin_run:
        pin_dashboard_run(run_root, run_dir)
    if str(run_label).strip():
        set_dashboard_run_label(run_root, run_dir, str(run_label).strip())
    replay_results: dict[str, list[dict]] = {}
    if run_replay_after_init:
        replay_results = run_replay(run_dir, replay_strategies or None)
    return {
        "run_dir": str(run_dir),
        "run_name": normalized_name,
        "pinned": bool(pin_run),
        "label": str(run_label).strip(),
        "baseline_trained": bool(train_baseline_after_init),
        "replay_started": bool(run_replay_after_init),
        "replay_strategies": sorted(replay_results) if replay_results else [],
    }
