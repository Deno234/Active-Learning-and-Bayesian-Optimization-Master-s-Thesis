from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import html
import json
from pathlib import Path
import random
import shlex
import socket
import sys
import time
from typing import Iterable, Sequence

import numpy as np

from active_learning_thesis.acquisition import (
    acquisition_diagnostics,
    requires_embeddings,
    requires_family_models,
    select_batch,
)
from active_learning_thesis.config import RunConfig, THESIS_FULL_REPLAY_STRATEGIES
from active_learning_thesis.ledger import serialize_probabilities
from active_learning_thesis.metrics import evaluate_binary_classifier
from active_learning_thesis.phase1_reproduction import PHASE1_MODELS
from active_learning_thesis.predictive import (
    score_sequences_with_ensemble,
    score_sequences_with_family,
    train_ensemble,
    train_family,
)


DEFAULT_PHASE1_ROOT = Path("thesis_results") / "01_reproduction"
DEFAULT_PHASE2_ROOT = Path("thesis_results") / "02_replay"
DEFAULT_BASE_SEED = 20260317
TARGET_F1_VALUES = [0.80, 0.84, 0.86]
SUPEK_SUBMIT_GROUP_SIZE = 5

ABLATION_SETUPS = ["single_raw", "ensemble_raw", "single_calibrated", "ensemble_calibrated"]
ABLATION_STRATEGIES = [
    "random",
    "ensemble_mean",
    "predictive_entropy",
    "similarity_penalized_mean",
    "cluster_diverse_representative",
    "oed_logdet",
]
BENCHMARK_STRATEGIES = list(THESIS_FULL_REPLAY_STRATEGIES)
SETUP_ALIASES = {
    "single_raw": (1, False),
    "ensemble_raw": (5, False),
    "single_calibrated": (1, True),
    "ensemble_calibrated": (5, True),
}


@dataclass(frozen=True)
class Phase2Options:
    phase1_root: Path = DEFAULT_PHASE1_ROOT
    output_root: Path = DEFAULT_PHASE2_ROOT
    mode: str | None = None
    status: bool = False
    force: bool = False
    write_supek_pbs: bool = False
    pbs_repo_root: Path | None = None
    outer_folds: tuple[int, ...] = (1, 2, 3, 4, 5)
    inner_fold: int = 1
    replay_seed_sizes: tuple[int, ...] = (10, 40)
    batch_size: int = 5
    max_rounds: int = 20
    strategies: tuple[str, ...] = ()
    setup: str | None = None
    ensemble_size: int = 5
    calibrated: bool = False
    base_seed: int = DEFAULT_BASE_SEED


@dataclass(frozen=True)
class Phase2ExportOptions:
    input_root: Path = DEFAULT_PHASE2_ROOT
    output_root: Path = DEFAULT_PHASE2_ROOT / "evidence"


@dataclass(frozen=True)
class CompatibilityDecision:
    strategy: str
    canonical_strategy: str
    compatible: bool
    skip_reason: str = ""


@dataclass
class ReplayRunSpec:
    mode: str
    setup: str
    outer_fold_id: int
    inner_fold_id: int
    replay_seed_size: int
    batch_size: int
    max_rounds: int
    strategies: tuple[str, ...]
    base_seed: int
    run_seed: int
    ensemble_size: int
    use_calibrated_acquisition: bool
    run_dir: Path

    @property
    def initial_label_count(self) -> int:
        return self.replay_seed_size


@dataclass
class ReplayRows:
    holdout: list[dict[str, str]]
    validation: list[dict[str, str]]
    train_pool: list[dict[str, str]]
    replay_seed: list[dict[str, str]]
    replay_hidden: list[dict[str, str]]


@contextmanager
def resource_logger(output_root: Path, step: str, run_id: str):
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "resource_log.csv"
    start = time.time()
    start_ts = _now_iso()
    artifacts: list[str] = []
    status = "success"
    try:
        yield artifacts
    except Exception:
        status = "failed"
        raise
    finally:
        row = {
            "step": step,
            "run_id": run_id,
            "start_timestamp": start_ts,
            "end_timestamp": _now_iso(),
            "walltime_seconds": f"{time.time() - start:.6f}",
            "hostname": socket.gethostname(),
            "command": " ".join(shlex.quote(item) for item in _safe_argv()),
            "exit_status": status,
            "output_artifacts": json.dumps(artifacts),
        }
        _append_csv(log_path, row)


def options_from_args(args) -> Phase2Options:
    return Phase2Options(
        phase1_root=Path(args.phase1_root),
        output_root=Path(args.output_root),
        mode=args.mode,
        status=bool(args.status),
        force=bool(args.force),
        write_supek_pbs=bool(args.write_supek_pbs),
        pbs_repo_root=Path(args.pbs_repo_root) if getattr(args, "pbs_repo_root", None) else None,
        outer_folds=tuple(_positive_ints(args.outer_folds, "--outer-folds")),
        inner_fold=_positive_int(args.inner_fold, "--inner-fold"),
        replay_seed_sizes=tuple(_positive_ints(args.replay_seed_sizes, "--replay-seed-sizes")),
        batch_size=_positive_int(args.batch_size, "--batch-size"),
        max_rounds=_nonnegative_int(args.max_rounds, "--max-rounds"),
        strategies=tuple(args.strategies or ()),
        setup=args.setup,
        ensemble_size=_positive_int(args.ensemble_size, "--ensemble-size"),
        calibrated=bool(args.calibrated),
        base_seed=int(args.base_seed),
    )


def export_options_from_args(args) -> Phase2ExportOptions:
    return Phase2ExportOptions(input_root=Path(args.input_root), output_root=Path(args.output_root))


def run_phase2_replay(args_or_options) -> dict[str, object]:
    options = args_or_options if isinstance(args_or_options, Phase2Options) else options_from_args(args_or_options)
    if options.status:
        return phase2_status(options.output_root)
    if options.write_supek_pbs:
        paths = write_supek_pbs_scripts(options)
        return {"status": "pbs-written", "outputs": [str(path) for path in paths]}
    if options.mode is None:
        raise ValueError("--mode is required unless --status is used")
    if options.mode == "aggregate":
        return aggregate_phase2(options.output_root)
    return run_phase2_mode(options)


def run_phase2_mode(options: Phase2Options) -> dict[str, object]:
    if options.mode not in {"smoke", "ablation", "benchmark"}:
        raise ValueError("--mode must be smoke, ablation, benchmark, or aggregate")
    _assert_phase1_inputs(options.phase1_root)
    frozen_config = load_frozen_model_config(options.phase1_root)
    specs = build_run_specs(options)
    mode_root = options.output_root / str(options.mode)
    mode_root.mkdir(parents=True, exist_ok=True)
    manifest_path = mode_root / f"{options.mode}_manifest.json"
    manifest = {
        "mode": options.mode,
        "phase1_root": str(options.phase1_root),
        "output_root": str(options.output_root),
        "outer_folds": list(options.outer_folds),
        "inner_fold": options.inner_fold,
        "replay_seed_sizes": list(options.replay_seed_sizes),
        "initial_label_counts": list(options.replay_seed_sizes),
        "n_repeats": len(options.outer_folds),
        "batch_size": options.batch_size,
        "max_rounds": options.max_rounds,
        "created_at": _now_iso(),
    }
    _write_json(manifest_path, manifest)
    run_results: list[dict[str, object]] = []
    for spec in specs:
        result = run_phase2_single(spec, options.phase1_root, frozen_config, force=options.force)
        run_results.append(result)
    aggregate_payload = aggregate_phase2(options.output_root, mode=options.mode)
    return {
        "status": "complete",
        "mode": options.mode,
        "run_count": len(run_results),
        "runs": run_results,
        "manifest": str(manifest_path),
        "aggregate": aggregate_payload,
    }


def build_run_specs(options: Phase2Options) -> list[ReplayRunSpec]:
    mode = str(options.mode or "smoke")
    specs: list[ReplayRunSpec] = []
    if mode == "ablation":
        setups = ABLATION_SETUPS if options.setup is None else [options.setup]
        default_strategies = tuple(options.strategies or tuple(ABLATION_STRATEGIES))
    elif mode == "benchmark":
        setups = [options.setup or "ensemble_calibrated"]
        default_strategies = tuple(options.strategies or tuple(BENCHMARK_STRATEGIES))
    else:
        setups = [options.setup or "ensemble_calibrated"]
        default_strategies = tuple(options.strategies or ("random", "ensemble_mean", "predictive_entropy"))
    for setup in setups:
        if setup not in SETUP_ALIASES:
            raise ValueError(f"Unsupported Phase 2 setup: {setup}")
        setup_ensemble_size, setup_calibrated = SETUP_ALIASES[setup]
        if mode == "benchmark":
            setup_ensemble_size = int(options.ensemble_size)
            setup_calibrated = bool(options.calibrated or setup.endswith("_calibrated"))
        for replay_seed_size in options.replay_seed_sizes:
            for outer_fold in options.outer_folds:
                run_seed = deterministic_run_seed(options.base_seed, outer_fold, options.inner_fold, replay_seed_size)
                run_id = _run_id(mode, setup, outer_fold, options.inner_fold, replay_seed_size)
                specs.append(
                    ReplayRunSpec(
                        mode=mode,
                        setup=setup,
                        outer_fold_id=int(outer_fold),
                        inner_fold_id=int(options.inner_fold),
                        replay_seed_size=int(replay_seed_size),
                        batch_size=int(options.batch_size),
                        max_rounds=int(options.max_rounds),
                        strategies=default_strategies,
                        base_seed=int(options.base_seed),
                        run_seed=run_seed,
                        ensemble_size=setup_ensemble_size,
                        use_calibrated_acquisition=setup_calibrated,
                        run_dir=options.output_root / mode / "runs" / run_id,
                    )
                )
    return specs


def run_phase2_single(
    spec: ReplayRunSpec,
    phase1_root: Path,
    frozen_config: dict[str, dict[str, object]],
    *,
    force: bool = False,
) -> dict[str, object]:
    metrics_path = spec.run_dir / "per_run_round_metrics.csv"
    if metrics_path.exists() and not force:
        return {"run_id": spec.run_dir.name, "status": "reused-existing", "run_dir": str(spec.run_dir)}
    if spec.run_dir.exists() and force:
        # Keep this intentionally narrow: only remove Phase 2 run output, never Phase 1.
        import shutil

        shutil.rmtree(spec.run_dir)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    with resource_logger(spec.run_dir, "phase2-replay", spec.run_dir.name) as artifacts:
        manifest = load_replay_manifest(phase1_root, spec.outer_fold_id, spec.inner_fold_id)
        rows = construct_replay_rows(manifest, spec.replay_seed_size, spec.run_seed)
        validate_no_within_repeat_overlap(rows)
        config = config_for_phase2(spec, frozen_config)
        write_run_inputs(spec, config, manifest, rows)
        compatibility = compatibility_rows(spec.mode, spec.setup, spec.strategies, spec.replay_seed_size)
        _write_csv(spec.run_dir / "strategy_compatibility_matrix.csv", compatibility)
        compatible = [
            row["canonical_strategy"]
            for row in compatibility
            if str(row["compatible"]).lower() == "true"
        ]
        if not compatible:
            raise ValueError("No compatible acquisition strategies requested for this setup")
        round_rows: list[dict[str, object]] = []
        selected_rows: list[dict[str, object]] = []
        acquisition_rows: list[dict[str, object]] = []
        calibration_rows: list[dict[str, object]] = []
        for strategy in compatible:
            strategy_result = run_strategy_replay(spec, config, strategy, rows)
            round_rows.extend(strategy_result["round_metrics"])
            selected_rows.extend(strategy_result["selected_sequences"])
            acquisition_rows.extend(strategy_result["acquisition_log"])
            calibration_rows.extend(strategy_result["calibration_summary"])
        _write_csv(spec.run_dir / "per_run_round_metrics.csv", round_rows)
        _write_csv(spec.run_dir / "per_run_selected_sequences.csv", selected_rows)
        _write_csv(spec.run_dir / "per_run_acquisition_log.csv", acquisition_rows)
        _write_csv(spec.run_dir / "ablation_calibration_summary.csv", calibration_rows)
        artifacts.extend(
            [
                str(spec.run_dir / "per_run_round_metrics.csv"),
                str(spec.run_dir / "per_run_selected_sequences.csv"),
                str(spec.run_dir / "per_run_acquisition_log.csv"),
            ]
        )
    return {"run_id": spec.run_dir.name, "status": "complete", "run_dir": str(spec.run_dir)}


def run_strategy_replay(
    spec: ReplayRunSpec,
    config: RunConfig,
    strategy: str,
    rows: ReplayRows,
) -> dict[str, list[dict[str, object]]]:
    current_labeled = [dict(row) for row in rows.replay_seed]
    hidden_pool = {row["sequence"]: dict(row) for row in rows.replay_hidden}
    round_metrics: list[dict[str, object]] = []
    selected_sequences: list[dict[str, object]] = []
    acquisition_log: list[dict[str, object]] = []
    calibration_summary: list[dict[str, object]] = []
    for round_id in range(spec.max_rounds + 1):
        round_dir = spec.run_dir / "models" / strategy / f"round_{round_id:03d}"
        ensemble = train_ensemble(
            current_labeled,
            rows.validation,
            round_dir / "ensemble",
            config,
            cache_dir=spec.run_dir / "model_cache",
        )
        validation_metrics, threshold = evaluate_with_validation_threshold(
            ensemble,
            rows.validation,
            use_calibration=config.use_calibrated_acquisition,
        )
        holdout_metrics = evaluate_with_fixed_threshold(
            ensemble,
            rows.holdout,
            threshold=threshold,
            use_calibration=config.use_calibrated_acquisition,
        )
        round_metrics.append(
            _metric_row(spec, strategy, round_id, len(current_labeled), "validation", validation_metrics)
        )
        round_metrics.append(
            _metric_row(spec, strategy, round_id, len(current_labeled), "holdout", holdout_metrics)
        )
        calibration_summary.extend(
            calibration_metric_rows(spec, strategy, round_id, ensemble, rows.validation)
        )
        if round_id == spec.max_rounds or not hidden_pool:
            _cleanup_tensorflow_runtime()
            break
        candidate_rows = list(hidden_pool.values())
        candidate_sequences = [row["sequence"] for row in candidate_rows]
        need_embeddings = requires_embeddings(strategy)
        candidate_scores = score_sequences_with_ensemble(
            ensemble,
            candidate_sequences,
            include_embeddings=need_embeddings,
            use_calibration=config.use_calibrated_acquisition,
            include_raw=True,
        )
        if requires_family_models(strategy):
            family = train_family(
                current_labeled,
                rows.validation,
                round_dir / "family",
                config,
                cache_dir=spec.run_dir / "model_cache",
            )
            candidate_scores.update(
                score_sequences_with_family(
                    family,
                    candidate_sequences,
                    use_calibration=config.use_calibrated_acquisition,
                    include_raw=True,
                )
            )
        labeled_embeddings = (
            _labeled_embeddings(ensemble, current_labeled)
            if need_embeddings
            else np.empty((0, 0), dtype=float)
        )
        selection_seed = spec.run_seed + round_id
        selected_indices, acquisition_scores = select_batch(
            strategy,
            spec.batch_size,
            candidate_scores,
            labeled_embeddings,
            config,
            selection_seed,
            candidate_sequences=candidate_sequences,
            reference_sequences=[row["sequence"] for row in current_labeled],
        )
        diagnostics = acquisition_diagnostics(
            strategy,
            selected_indices,
            candidate_scores,
            labeled_embeddings,
            acquisition_scores,
            config,
            selection_seed,
            candidate_sequences=candidate_sequences,
            reference_sequences=[row["sequence"] for row in current_labeled],
        )
        selected_set = set(selected_indices)
        for index, row in enumerate(candidate_rows):
            log_row = _candidate_log_row(
                spec,
                strategy,
                round_id,
                row,
                index,
                candidate_scores,
                acquisition_scores,
                diagnostics,
                selected=index in selected_set,
            )
            acquisition_log.append(log_row)
        for rank, index in enumerate(selected_indices, start=1):
            row = candidate_rows[index]
            selected_sequences.append(
                {
                    "mode": spec.mode,
                    "setup": spec.setup,
                    "outer_fold_id": spec.outer_fold_id,
                    "inner_fold_id": spec.inner_fold_id,
                    "replay_seed_size": spec.replay_seed_size,
                    "initial_label_count": spec.initial_label_count,
                    "run_seed": spec.run_seed,
                    "strategy": strategy,
                    "round_id": round_id,
                    "selection_rank": rank,
                    "sequence": row["sequence"],
                    "label": row["label"],
                    "acquisition_score": float(acquisition_scores[index]),
                }
            )
            current_labeled.append(hidden_pool.pop(row["sequence"]))
        _cleanup_tensorflow_runtime()
    return {
        "round_metrics": round_metrics,
        "selected_sequences": selected_sequences,
        "acquisition_log": acquisition_log,
        "calibration_summary": calibration_summary,
    }


def _labeled_embeddings(ensemble, rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    scored = score_sequences_with_ensemble(
        ensemble,
        [row["sequence"] for row in rows],
        include_embeddings=True,
        use_calibration=False,
    )
    return np.asarray(scored["avg_embedding"], dtype=float)


def _cleanup_tensorflow_runtime() -> None:
    try:
        if "tensorflow" in sys.modules:
            import tensorflow as tf

            tf.keras.backend.clear_session()
    except Exception:
        pass


def evaluate_with_validation_threshold(ensemble, validation_rows, *, use_calibration: bool) -> tuple[dict[str, object], float]:
    sequences = [row["sequence"] for row in validation_rows]
    labels = [row["label"] for row in validation_rows]
    scored = score_sequences_with_ensemble(
        ensemble,
        sequences,
        use_calibration=use_calibration,
    )
    metrics = evaluate_binary_classifier(
        labels,
        scored["pred_mean"],
        threshold_strategy="pr_best_f1",
        threshold_source="validation",
    )
    return metrics, float(metrics["decision_threshold"])


def evaluate_with_fixed_threshold(ensemble, rows, *, threshold: float, use_calibration: bool) -> dict[str, object]:
    sequences = [row["sequence"] for row in rows]
    labels = [row["label"] for row in rows]
    scored = score_sequences_with_ensemble(
        ensemble,
        sequences,
        use_calibration=use_calibration,
    )
    return evaluate_binary_classifier(
        labels,
        scored["pred_mean"],
        threshold=threshold,
        threshold_strategy="pr_best_f1",
        threshold_source="validation",
    )


def calibration_metric_rows(spec: ReplayRunSpec, strategy: str, round_id: int, ensemble, validation_rows) -> list[dict[str, object]]:
    sequences = [row["sequence"] for row in validation_rows]
    labels = [row["label"] for row in validation_rows]
    scored = score_sequences_with_ensemble(ensemble, sequences, use_calibration=True, include_raw=True)
    raw = evaluate_binary_classifier(labels, scored.get("raw_pred_mean", scored["pred_mean"]), threshold_strategy="pr_best_f1")
    calibrated = evaluate_binary_classifier(labels, scored["pred_mean"], threshold_strategy="pr_best_f1")
    return [
        _calibration_row(spec, strategy, round_id, "raw", raw),
        _calibration_row(spec, strategy, round_id, "calibrated", calibrated),
    ]


def construct_replay_rows(manifest: dict[str, object], replay_seed_size: int, run_seed: int) -> ReplayRows:
    source_rows = [dict(row) for row in manifest.get("rows", []) if isinstance(row, dict)]
    holdout = [_norm_row(row, "holdout", "none") for row in source_rows if row.get("split") == "holdout"]
    validation = [_norm_row(row, "validation", "none") for row in source_rows if row.get("split") == "validation"]
    train_pool = [_norm_row(row, "train_pool", "none") for row in source_rows if row.get("split") == "train_pool"]
    seed_sequences = stratified_replay_seed_sequences(train_pool, replay_seed_size, run_seed)
    seed_set = set(seed_sequences)
    replay_seed = [_norm_row(row, "train_pool", "seed") for row in train_pool if row["sequence"] in seed_set]
    replay_hidden = [_norm_row(row, "train_pool", "hidden") for row in train_pool if row["sequence"] not in seed_set]
    return ReplayRows(
        holdout=holdout,
        validation=validation,
        train_pool=train_pool,
        replay_seed=replay_seed,
        replay_hidden=replay_hidden,
    )


def stratified_replay_seed_sequences(train_pool: list[dict[str, str]], replay_seed_size: int, run_seed: int) -> list[str]:
    if replay_seed_size >= len(train_pool):
        raise ValueError("replay_seed_size must be smaller than train_pool")
    by_label: dict[str, list[dict[str, str]]] = {"1": [], "0": []}
    for row in train_pool:
        by_label.setdefault(str(row["label"]), []).append(row)
    positives = by_label.get("1", [])
    negatives = by_label.get("0", [])
    if not positives or not negatives:
        raise ValueError("train_pool must contain both classes for stratified replay_seed")
    pos_target = int(round(replay_seed_size * len(positives) / len(train_pool)))
    pos_target = max(1, min(pos_target, replay_seed_size - 1, len(positives)))
    neg_target = replay_seed_size - pos_target
    if neg_target > len(negatives):
        neg_target = len(negatives)
        pos_target = replay_seed_size - neg_target
    rng = random.Random(run_seed)
    pos_rows = list(positives)
    neg_rows = list(negatives)
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)
    selected = [row["sequence"] for row in pos_rows[:pos_target] + neg_rows[:neg_target]]
    rng.shuffle(selected)
    if len(selected) != replay_seed_size:
        raise ValueError("Unable to construct stratified replay_seed of requested size")
    return selected


def validate_no_within_repeat_overlap(rows: ReplayRows) -> None:
    holdout = {row["sequence"] for row in rows.holdout}
    validation = {row["sequence"] for row in rows.validation}
    train_pool = {row["sequence"] for row in rows.train_pool}
    replay_seed = {row["sequence"] for row in rows.replay_seed}
    replay_hidden = {row["sequence"] for row in rows.replay_hidden}
    checks = [
        ("holdout", holdout, "validation", validation),
        ("holdout", holdout, "train_pool", train_pool),
        ("validation", validation, "train_pool", train_pool),
        ("replay_seed", replay_seed, "replay_hidden", replay_hidden),
    ]
    for left_name, left, right_name, right in checks:
        overlap = left & right
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Illegal split overlap between {left_name} and {right_name}: {preview}")
    if not replay_seed <= train_pool:
        raise ValueError("replay_seed must be a subset of train_pool")
    if not replay_hidden <= train_pool:
        raise ValueError("replay_hidden must be a subset of train_pool")
    if replay_seed | replay_hidden != train_pool:
        raise ValueError("replay_seed plus replay_hidden must equal train_pool")


def compatibility_rows(mode: str, setup: str, strategies: Sequence[str], replay_seed_size: int) -> list[dict[str, object]]:
    rows = []
    for strategy in strategies:
        decision = strategy_compatibility(mode, setup, strategy)
        rows.append(
            {
                "mode": mode,
                "setup": setup,
                "strategy": strategy,
                "canonical_strategy": decision.canonical_strategy,
                "compatible": decision.compatible,
                "skip_reason": decision.skip_reason,
                "replay_seed_size": replay_seed_size,
                "initial_label_count": replay_seed_size,
            }
        )
    return rows


def strategy_compatibility(mode: str, setup: str, strategy: str) -> CompatibilityDecision:
    canonical = "ensemble_mean" if strategy == "predictive_mean" else strategy
    known = set(BENCHMARK_STRATEGIES) | set(ABLATION_STRATEGIES) | {"predictive_mean"}
    if strategy not in known:
        return CompatibilityDecision(strategy, canonical, False, "unsupported_strategy")
    if mode == "ablation":
        if canonical in ABLATION_STRATEGIES:
            return CompatibilityDecision(strategy, canonical, True)
        return CompatibilityDecision(strategy, canonical, False, "incompatible_with_ablation_setup")
    if setup.startswith("single_") and canonical in {"ensemble_mi", "hybrid_mi_diverse", "family_qbc", "ucb"}:
        return CompatibilityDecision(strategy, canonical, False, "requires_ensemble_or_committee")
    return CompatibilityDecision(strategy, canonical, True)


def config_for_phase2(spec: ReplayRunSpec, frozen_config: dict[str, dict[str, object]]) -> RunConfig:
    model_num_cells = dict(RunConfig().model_num_cells)
    model_kernel_size = dict(RunConfig().model_kernel_size)
    for model_name, payload in frozen_config.items():
        if model_name not in PHASE1_MODELS:
            continue
        model_num_cells[model_name] = int(payload["num_cells"])
        kernel = payload.get("kernel_size", RunConfig().model_kernel_size.get(model_name, 4))
        model_kernel_size[model_name] = 4 if str(kernel) == "n/a" else int(kernel)
    return RunConfig(
        run_name=spec.run_dir.name,
        output_root=str(spec.run_dir.parent),
        random_seed=spec.run_seed,
        replay_seed_size=spec.replay_seed_size,
        batch_size=spec.batch_size,
        max_rounds=spec.max_rounds,
        ensemble_size=spec.ensemble_size,
        ensemble_seeds=[spec.run_seed + index for index in range(spec.ensemble_size)],
        epochs=70,
        replay_strategies=list(spec.strategies),
        use_calibrated_acquisition=spec.use_calibrated_acquisition,
        model_num_cells=model_num_cells,
        model_kernel_size=model_kernel_size,
    )


def write_run_inputs(spec: ReplayRunSpec, config: RunConfig, manifest: dict[str, object], rows: ReplayRows) -> None:
    config_payload = config.to_dict()
    config_payload.update(
        {
            "phase2_mode": spec.mode,
            "setup": spec.setup,
            "outer_fold_id": spec.outer_fold_id,
            "inner_fold_id": spec.inner_fold_id,
            "replay_seed_size": spec.replay_seed_size,
            "initial_label_count": spec.initial_label_count,
            "base_seed": spec.base_seed,
            "run_seed": spec.run_seed,
            "member_seeds": list(config.ensemble_seeds),
        }
    )
    _write_json(spec.run_dir / "config.json", config_payload)
    _write_json(spec.run_dir / "phase1_frozen_model_config_used.json", {
        "model_num_cells": config.model_num_cells,
        "model_kernel_size": config.model_kernel_size,
    })
    _write_json(spec.run_dir / "replay_manifest_used.json", manifest)
    split_rows = split_seed_manifest_rows(spec, rows)
    _write_json(spec.run_dir / "split_seed_manifest.json", {"rows": split_rows})
    _write_csv(spec.run_dir / "split_audit.csv", split_audit_rows(spec, rows))


def split_seed_manifest_rows(spec: ReplayRunSpec, rows: ReplayRows) -> list[dict[str, object]]:
    result = []
    for split_name in ["holdout", "validation", "replay_seed", "replay_hidden"]:
        for row in getattr(rows, split_name):
            result.append(
                {
                    "outer_fold_id": spec.outer_fold_id,
                    "inner_fold_id": spec.inner_fold_id,
                    "replay_seed_size": spec.replay_seed_size,
                    "initial_label_count": spec.initial_label_count,
                    "split": split_name,
                    "sequence": row["sequence"],
                    "label": row["label"],
                }
            )
    return result


def split_audit_rows(spec: ReplayRunSpec, rows: ReplayRows) -> list[dict[str, object]]:
    result = []
    for split_name in ["holdout", "validation", "train_pool", "replay_seed", "replay_hidden"]:
        split_rows = getattr(rows, split_name)
        positives = sum(1 for row in split_rows if str(row["label"]) == "1")
        result.append(
            {
                "mode": spec.mode,
                "setup": spec.setup,
                "outer_fold_id": spec.outer_fold_id,
                "inner_fold_id": spec.inner_fold_id,
                "replay_seed_size": spec.replay_seed_size,
                "initial_label_count": spec.initial_label_count,
                "split": split_name,
                "count": len(split_rows),
                "positive_count": positives,
                "negative_count": len(split_rows) - positives,
            }
        )
    return result


def aggregate_phase2(output_root: Path, mode: str | None = None) -> dict[str, object]:
    modes = [mode] if mode else ["smoke", "ablation", "benchmark"]
    outputs: dict[str, str] = {}
    for selected_mode in modes:
        mode_root = output_root / selected_mode
        if not mode_root.exists():
            continue
        round_rows = _collect_run_csv(mode_root, "per_run_round_metrics.csv")
        selected_rows = _collect_run_csv(mode_root, "per_run_selected_sequences.csv")
        acquisition_rows = _collect_run_csv(mode_root, "per_run_acquisition_log.csv")
        compatibility = _collect_run_csv(mode_root, "strategy_compatibility_matrix.csv")
        if round_rows:
            _write_csv(mode_root / "per_run_round_metrics.csv", round_rows)
            _write_csv(mode_root / "learning_curves.csv", round_rows)
            summary = summarize_round_metrics(round_rows)
            summary_name = "ablation_summary.csv" if selected_mode == "ablation" else "strategy_summary.csv"
            _write_csv(mode_root / summary_name, summary)
            _write_csv(mode_root / "labels_to_target_summary.csv", labels_to_target_rows(round_rows))
            _write_csv(mode_root / "paired_vs_random.csv", paired_vs_random_rows(round_rows))
            outputs[f"{selected_mode}_summary"] = str(mode_root / summary_name)
        if selected_rows:
            _write_csv(mode_root / "per_run_selected_sequences.csv", selected_rows)
        if acquisition_rows:
            _write_csv(mode_root / "per_run_acquisition_log.csv", acquisition_rows)
        if compatibility:
            _write_csv(mode_root / "strategy_compatibility_matrix.csv", compatibility)
        if selected_mode == "ablation":
            calibration = _collect_run_csv(mode_root, "ablation_calibration_summary.csv")
            if calibration:
                _write_csv(mode_root / "ablation_calibration_summary.csv", calibration)
    return {"status": "complete", "outputs": outputs}


def summarize_round_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    final_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("mode", ""),
            row.get("initial_label_count", row.get("replay_seed_size", "")),
            row.get("setup", ""),
            row.get("strategy", ""),
            row.get("evaluation_dataset", ""),
        )
        final_by_key.setdefault(key, []).append(row)
    result = []
    for (mode, initial_label_count, setup, strategy, evaluation_dataset), group in sorted(final_by_key.items()):
        finals = _final_rows_by_outer_fold(group)
        f1_values = [_float(row.get("f1")) for row in finals if _float(row.get("f1")) is not None]
        pr_values = [_float(row.get("pr_auc")) for row in finals if _float(row.get("pr_auc")) is not None]
        roc_values = [_float(row.get("roc_auc")) for row in finals if _float(row.get("roc_auc")) is not None]
        brier_values = [_float(row.get("brier_score")) for row in finals if _float(row.get("brier_score")) is not None]
        ece_values = [_float(row.get("ece_10")) for row in finals if _float(row.get("ece_10")) is not None]
        all_points = group
        result.append(
            {
                "mode": mode,
                "initial_label_count": initial_label_count,
                "setup": setup,
                "strategy": strategy,
                "evaluation_dataset": evaluation_dataset,
                "n_repeats": len({row.get("outer_fold_id", "") for row in finals}),
                "mean_final_F1": _mean(f1_values),
                "std_final_F1": _std(f1_values),
                "median_final_F1": _median(f1_values),
                "min_final_F1": min(f1_values) if f1_values else "",
                "max_final_F1": max(f1_values) if f1_values else "",
                "mean_AULC_F1": _mean(_aulc_by_fold(all_points, "f1")),
                "std_AULC_F1": _std(_aulc_by_fold(all_points, "f1")),
                "mean_final_PR_AUC": _mean(pr_values),
                "mean_final_ROC_AUC": _mean(roc_values),
                "mean_final_Brier": _mean(brier_values),
                "mean_final_ECE_10": _mean(ece_values),
            }
        )
    return result


def labels_to_target_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("mode", ""),
            row.get("initial_label_count", row.get("replay_seed_size", "")),
            row.get("setup", ""),
            row.get("strategy", ""),
            row.get("evaluation_dataset", ""),
            row.get("outer_fold_id", ""),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for target in TARGET_F1_VALUES:
        aggregate: dict[tuple[str, str, str, str, str], list[int | None]] = {}
        for (mode, initial_label_count, setup, strategy, dataset, _outer), points in groups.items():
            reached = labels_to_target(points, "f1", target)
            aggregate.setdefault((mode, initial_label_count, setup, strategy, dataset), []).append(reached)
        for (mode, initial_label_count, setup, strategy, dataset), values in sorted(aggregate.items()):
            numeric = [float(value) for value in values if value is not None]
            result.append(
                {
                    "mode": mode,
                    "initial_label_count": initial_label_count,
                    "setup": setup,
                    "strategy": strategy,
                    "evaluation_dataset": dataset,
                    "target_f1": target,
                    "n_repeats": len(values),
                    "mean_labels_to_target": _mean(numeric),
                    "median_labels_to_target": _median(numeric),
                    "reached_count": len(numeric),
                }
            )
    return result


def paired_vs_random_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    finals = _final_rows_by_group(rows)
    result = []
    for key, strategy_rows in finals.items():
        random_row = strategy_rows.get("random")
        if not random_row:
            continue
        for strategy, row in sorted(strategy_rows.items()):
            if strategy == "random":
                continue
            result.append(
                {
                    "mode": key[0],
                    "initial_label_count": key[1],
                    "setup": key[2],
                    "evaluation_dataset": key[3],
                    "outer_fold_id": key[4],
                    "strategy": strategy,
                    "random_final_F1": random_row.get("f1", ""),
                    "strategy_final_F1": row.get("f1", ""),
                    "final_F1_delta_vs_random": _none_subtract(_float(row.get("f1")), _float(random_row.get("f1"))),
                    "win_vs_random": _float(row.get("f1")) is not None and _float(random_row.get("f1")) is not None and _float(row.get("f1")) > _float(random_row.get("f1")),
                }
            )
    return result


def run_phase2_export(args_or_options) -> dict[str, object]:
    options = args_or_options if isinstance(args_or_options, Phase2ExportOptions) else export_options_from_args(args_or_options)
    aggregate_phase2(options.input_root)
    options.output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    copy_map = {
        "ablation_summary.csv": [options.input_root / "ablation" / "ablation_summary.csv"],
        "ablation_calibration_summary.csv": [options.input_root / "ablation" / "ablation_calibration_summary.csv"],
        "strategy_compatibility_matrix.csv": [options.input_root / "ablation" / "strategy_compatibility_matrix.csv"],
        "benchmark_strategy_summary.csv": [options.input_root / "benchmark" / "strategy_summary.csv"],
        "labels_to_target_summary.csv": [
            options.input_root / "benchmark" / "labels_to_target_summary.csv",
            options.input_root / "ablation" / "labels_to_target_summary.csv",
        ],
        "paired_vs_random.csv": [
            options.input_root / "benchmark" / "paired_vs_random.csv",
            options.input_root / "ablation" / "paired_vs_random.csv",
        ],
        "learning_curves.csv": [
            options.input_root / "benchmark" / "learning_curves.csv",
            options.input_root / "ablation" / "learning_curves.csv",
        ],
    }
    for name, sources in copy_map.items():
        source = next((candidate for candidate in sources if candidate.is_file()), None)
        if source is None:
            continue
        target = options.output_root / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        outputs[name] = str(target)
    figures = write_phase2_figures(options.input_root, options.output_root / "figures")
    outputs.update(figures)
    return {"status": "complete", "outputs": outputs}


def write_phase2_figures(input_root: Path, figure_root: Path) -> dict[str, str]:
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return write_phase2_svg_figures(input_root, figure_root)
    for mode in ["ablation", "benchmark"]:
        curve_path = input_root / mode / "learning_curves.csv"
        if not curve_path.exists():
            continue
        all_rows = _read_csv(curve_path)
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in all_rows if row.get("evaluation_dataset") == dataset]
            for initial in sorted({row.get("initial_label_count", "") for row in dataset_rows}, key=_sort_text_number):
                rows = [row for row in dataset_rows if row.get("initial_label_count") == initial]
                for metric, title in [
                    ("f1", "F1 vs labeled peptides"),
                    ("pr_auc", "PR-AUC vs labeled peptides"),
                ]:
                    fig_path = figure_root / f"{mode}_{dataset}_{metric}_initial_{initial}_vs_labeled_peptides.png"
                    _plot_learning_curves(
                        plt,
                        rows,
                        metric,
                        f"{mode.title()} {dataset.title()} {title}, initial labels={initial}",
                        fig_path,
                    )
                    outputs[fig_path.name] = str(fig_path)

    for mode, summary_name in [("ablation", "ablation_summary.csv"), ("benchmark", "strategy_summary.csv")]:
        summary_path = input_root / mode / summary_name
        if not summary_path.exists():
            continue
        rows = _read_csv(summary_path)
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in rows if row.get("evaluation_dataset") == dataset]
            if not dataset_rows:
                continue
            for metric, xlabel, lower_is_better in [
                ("mean_AULC_F1", "Mean AULC-F1", False),
                ("mean_final_F1", "Mean final F1", False),
                ("mean_final_PR_AUC", "Mean final PR-AUC", False),
                ("mean_final_Brier", "Mean final Brier score", True),
            ]:
                fig_path = figure_root / f"{mode}_{dataset}_{metric}_by_strategy.png"
                _plot_strategy_metric_by_initial(
                    plt,
                    dataset_rows,
                    metric,
                    f"{mode.title()} {dataset.title()} {xlabel} by strategy",
                    xlabel,
                    fig_path,
                    lower_is_better=lower_is_better,
                    limit=14 if mode == "ablation" else 12,
                )
                outputs[fig_path.name] = str(fig_path)
        if mode == "ablation":
            holdout = [row for row in rows if row.get("evaluation_dataset") == "holdout"]
            for metric, ylabel in [("mean_final_Brier", "Mean final Brier"), ("mean_final_ECE_10", "Mean final ECE-10")]:
                fig_path = figure_root / f"ablation_{metric}_by_setup.png"
                _plot_setup_metric_bar(plt, holdout, metric, ylabel, fig_path)
                outputs[fig_path.name] = str(fig_path)

    labels_path = input_root / "benchmark" / "labels_to_target_summary.csv"
    if labels_path.exists():
        rows = [
            row
            for row in _read_csv(labels_path)
            if row.get("target_f1") == "0.86"
        ]
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in rows if row.get("evaluation_dataset") == dataset]
            if not dataset_rows:
                continue
            fig_path = figure_root / f"benchmark_{dataset}_labels_to_f1_086_by_strategy.png"
            _plot_strategy_metric_by_initial(
                plt,
                dataset_rows,
                "mean_labels_to_target",
                f"Benchmark {dataset.title()} labels needed to reach F1=0.86",
                "Mean labels to target F1=0.86",
                fig_path,
                lower_is_better=True,
                limit=12,
            )
            outputs[fig_path.name] = str(fig_path)

    labels_path = input_root / "ablation" / "labels_to_target_summary.csv"
    if labels_path.exists():
        rows = [
            row
            for row in _read_csv(labels_path)
            if row.get("target_f1") == "0.84"
        ]
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in rows if row.get("evaluation_dataset") == dataset]
            if not dataset_rows:
                continue
            fig_path = figure_root / f"ablation_{dataset}_labels_to_f1_084_by_strategy.png"
            _plot_strategy_metric_by_initial(
                plt,
                dataset_rows,
                "mean_labels_to_target",
                f"Ablation {dataset.title()} labels needed to reach F1=0.84",
                "Mean labels to target F1=0.84",
                fig_path,
                lower_is_better=True,
                limit=14,
            )
            outputs[fig_path.name] = str(fig_path)

    paired_path = input_root / "benchmark" / "paired_vs_random.csv"
    if paired_path.exists():
        rows = _read_csv(paired_path)
        for dataset in ["validation", "holdout"]:
            summary_rows = _paired_delta_summary_rows(rows, dataset)
            if not summary_rows:
                continue
            fig_path = figure_root / f"benchmark_{dataset}_paired_delta_vs_random.png"
            _plot_strategy_metric_by_initial(
                plt,
                summary_rows,
                "mean_final_F1_delta_vs_random",
                f"Benchmark {dataset.title()} final F1 delta vs random",
                "Mean final F1 delta vs random",
                fig_path,
                lower_is_better=False,
                limit=12,
                annotation_field="wins_vs_random",
            )
            outputs[fig_path.name] = str(fig_path)
    outputs.update(write_phase2_svg_figures(input_root, figure_root))
    return outputs


def write_phase2_svg_figures(input_root: Path, figure_root: Path) -> dict[str, str]:
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for mode in ["ablation", "benchmark"]:
        curve_path = input_root / mode / "learning_curves.csv"
        if not curve_path.exists():
            continue
        all_rows = _read_csv(curve_path)
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in all_rows if row.get("evaluation_dataset") == dataset]
            for initial in sorted({row.get("initial_label_count", "") for row in dataset_rows}, key=_sort_text_number):
                rows = [row for row in dataset_rows if row.get("initial_label_count") == initial]
                for metric, title in [
                    ("f1", "F1 vs labeled peptides"),
                    ("pr_auc", "PR-AUC vs labeled peptides"),
                ]:
                    fig_path = figure_root / f"{mode}_{dataset}_{metric}_initial_{initial}_vs_labeled_peptides.svg"
                    _write_svg_learning_curves(
                        rows,
                        metric,
                        f"{mode.title()} {dataset.title()} {title}, initial labels={initial}",
                        fig_path,
                    )
                    outputs[fig_path.name] = str(fig_path)

    for mode, summary_name in [("ablation", "ablation_summary.csv"), ("benchmark", "strategy_summary.csv")]:
        summary_path = input_root / mode / summary_name
        if not summary_path.exists():
            continue
        rows = _read_csv(summary_path)
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in rows if row.get("evaluation_dataset") == dataset]
            for initial in sorted({row.get("initial_label_count", "") for row in dataset_rows}, key=_sort_text_number):
                points = [row for row in dataset_rows if row.get("initial_label_count") == initial]
                for metric, label, lower_is_better in [
                    ("mean_AULC_F1", "AULC-F1", False),
                    ("mean_final_F1", "Final F1", False),
                    ("mean_final_PR_AUC", "Final PR-AUC", False),
                    ("mean_final_Brier", "Final Brier score", True),
                ]:
                    fig_path = figure_root / f"{mode}_{dataset}_{metric}_initial_{initial}.svg"
                    if mode == "benchmark" and metric in {"mean_AULC_F1", "mean_final_F1"}:
                        _write_svg_ranked_dot_plot(
                            points,
                            metric,
                            f"{mode.title()} {dataset.title()} {label}, initial labels={initial}",
                            label,
                            fig_path,
                            limit=12,
                            lower_is_better=lower_is_better,
                        )
                    else:
                        _write_svg_summary_bar(
                            points,
                            metric,
                            f"{mode.title()} {dataset.title()} {label}, initial labels={initial}",
                            fig_path,
                            limit=14 if mode == "ablation" else 12,
                            lower_is_better=lower_is_better,
                        )
                    outputs[fig_path.name] = str(fig_path)
        if mode == "ablation":
            holdout = [row for row in rows if row.get("evaluation_dataset") == "holdout"]
            for metric in ["mean_final_Brier", "mean_final_ECE_10"]:
                fig_path = figure_root / f"ablation_{metric}_by_setup.svg"
                _write_svg_setup_metric_bar(holdout, metric, fig_path)
                outputs[fig_path.name] = str(fig_path)

    labels_jobs = [
        ("benchmark", "0.86", "086", 12),
        ("ablation", "0.84", "084", 14),
    ]
    for mode, target, slug, limit in labels_jobs:
        labels_path = input_root / mode / "labels_to_target_summary.csv"
        if not labels_path.exists():
            continue
        rows = [
            row
            for row in _read_csv(labels_path)
            if row.get("target_f1") == target
        ]
        for dataset in ["validation", "holdout"]:
            dataset_rows = [row for row in rows if row.get("evaluation_dataset") == dataset]
            for initial in sorted({row.get("initial_label_count", "") for row in dataset_rows}, key=_sort_text_number):
                points = [row for row in dataset_rows if row.get("initial_label_count") == initial]
                fig_path = figure_root / f"{mode}_{dataset}_labels_to_f1_{slug}_initial_{initial}.svg"
                _write_svg_summary_bar(
                    points,
                    "mean_labels_to_target",
                    f"{mode.title()} {dataset.title()} labels to F1={target}, initial labels={initial}",
                    fig_path,
                    limit=limit,
                    lower_is_better=True,
                )
                outputs[fig_path.name] = str(fig_path)
                if mode == "benchmark":
                    saved_points = _labels_saved_vs_random_rows(points)
                    saved_path = figure_root / f"{mode}_{dataset}_labels_saved_vs_random_to_f1_{slug}_initial_{initial}.svg"
                    _write_svg_labels_saved_bar(
                        saved_points,
                        f"{mode.title()} {dataset.title()} labels needed vs random to F1={target} (initial labels={initial})",
                        saved_path,
                    )
                    outputs[saved_path.name] = str(saved_path)
    paired_path = input_root / "benchmark" / "paired_vs_random.csv"
    if paired_path.exists():
        rows = _read_csv(paired_path)
        for dataset in ["validation", "holdout"]:
            summary_rows = _paired_delta_summary_rows(rows, dataset)
            for initial in sorted({row.get("initial_label_count", "") for row in summary_rows}, key=_sort_text_number):
                points = [row for row in summary_rows if row.get("initial_label_count") == initial]
                fig_path = figure_root / f"benchmark_{dataset}_paired_delta_vs_random_initial_{initial}.svg"
                _write_svg_delta_bar(
                    points,
                    "mean_final_F1_delta_vs_random",
                    f"Benchmark {dataset.title()} final F1 delta vs random, initial labels={initial}",
                    fig_path,
                )
                outputs[fig_path.name] = str(fig_path)
    outputs.update(_write_phase2_overlap_svg_figures(input_root, figure_root))
    outputs.update(_write_phase2_presentation_svg_figures(input_root, figure_root / "presentation"))
    _write_phase2_curated_figure_index(input_root, figure_root)
    return outputs


def phase2_status(output_root: Path) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    for mode in ["smoke", "ablation", "benchmark"]:
        mode_root = output_root / mode
        metrics = mode_root / "per_run_round_metrics.csv"
        checks[mode] = {
            "complete": metrics.exists(),
            "metrics": str(metrics),
            "run_count": len(list((mode_root / "runs").glob("*"))) if (mode_root / "runs").exists() else 0,
        }
    evidence_root = output_root / "evidence"
    checks["evidence"] = {"complete": evidence_root.exists(), "path": str(evidence_root)}
    failed_logs = scan_failed_logs(output_root)
    return {
        "output_root": str(output_root),
        "checks": checks,
        "failed_logs": failed_logs,
        "smoke_complete": bool(checks["smoke"]["complete"]),
        "ready_for_thesis_figures": bool(
            checks["ablation"]["complete"]
            and checks["benchmark"]["complete"]
            and not failed_logs
        ),
    }


def format_phase2_status(status: dict[str, object]) -> str:
    lines = [f"Phase 2 status for {status['output_root']}"]
    checks = status.get("checks", {})
    if isinstance(checks, dict):
        for name, payload in checks.items():
            complete = bool(payload.get("complete")) if isinstance(payload, dict) else False
            lines.append(f"- {name}: {'complete' if complete else 'missing'}")
    failed_logs = status.get("failed_logs", [])
    if failed_logs:
        lines.append("Failed/error logs:")
        for path in failed_logs:
            lines.append(f"- {path}")
    lines.append(f"Smoke complete: {'yes' if status.get('smoke_complete') else 'no'}")
    lines.append(f"Ready for thesis figures: {'yes' if status.get('ready_for_thesis_figures') else 'no'}")
    return "\n".join(lines)


def scan_failed_logs(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    failed: list[str] = []
    for path in output_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".out", ".err", ".log", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Traceback (most recent call last)" in text or "ERROR" in text or "Exit_status = 1" in text:
            failed.append(str(path))
    return failed


def write_supek_pbs_scripts(options: Phase2Options) -> list[Path]:
    output_root = options.output_root
    pbs_dir = output_root / "logs" / "supek_pbs"
    runtime_dir = output_root / "logs" / "supek_runtime"
    pbs_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pbs_repo_root = _pbs_repo_root(options)
    absolute_output = _pbs_target_path(options.output_root, options.pbs_repo_root)
    absolute_phase1 = _pbs_target_path(options.phase1_root, options.pbs_repo_root)
    absolute_runtime = absolute_output / "logs" / "supek_runtime"
    created: list[Path] = []
    modes = [options.mode] if options.mode else ["smoke", "ablation", "benchmark"]
    for mode in modes:
        if mode == "smoke":
            command = _phase2_command(options, absolute_phase1, absolute_output, mode)
            path = pbs_dir / "supek_phase2_smoke.pbs"
            path.write_text(_pbs_text("p2_smoke", "04:00:00", command, absolute_runtime, pbs_repo_root), encoding="utf-8")
            created.append(path)
            continue
        job_paths = []
        submit_job_paths = []
        sub_options = Phase2Options(**{**options.__dict__, "mode": mode})
        for spec in build_run_specs(sub_options):
            job_name = f"p2_{mode[:3]}_o{spec.outer_fold_id}_n{spec.replay_seed_size}"
            if mode == "ablation":
                job_name += f"_{_setup_job_code(spec.setup)}"
            command = _phase2_command_for_spec(options, absolute_phase1, absolute_output, spec)
            path = pbs_dir / f"supek_phase2_{mode}_outer_{spec.outer_fold_id}_seedsize_{spec.replay_seed_size}_{spec.setup}.pbs"
            path.write_text(_pbs_text(job_name, "24:00:00", command, absolute_runtime, pbs_repo_root), encoding="utf-8")
            created.append(path)
            job_paths.append(path)
            submit_job_paths.append(absolute_output / path.relative_to(output_root))
        aggregate_path = pbs_dir / f"supek_phase2_{mode}_aggregate.pbs"
        aggregate_command = f"python -m active_learning_thesis phase2-replay --mode aggregate --output-root {_shell_quote(absolute_output.as_posix())}"
        aggregate_path.write_text(_pbs_text(f"p2_{mode[:3]}_agg", "02:00:00", aggregate_command, absolute_runtime, pbs_repo_root), encoding="utf-8")
        created.append(aggregate_path)
        submit_path = pbs_dir / f"supek_phase2_{mode}_submit_all.sh"
        submit_aggregate_path = absolute_output / aggregate_path.relative_to(output_root)
        submit_path.write_text(_submit_all_text(submit_job_paths, submit_aggregate_path, absolute_output / "logs"), encoding="utf-8")
        created.append(submit_path)
        for group_index, group_paths in enumerate(_chunks(submit_job_paths, SUPEK_SUBMIT_GROUP_SIZE), start=1):
            group_path = pbs_dir / f"supek_phase2_{mode}_submit_group_{group_index:02d}.sh"
            group_path.write_text(
                _submit_group_text(group_paths, absolute_output / "logs", mode, group_index),
                encoding="utf-8",
            )
            created.append(group_path)
        aggregate_submit_path = pbs_dir / f"supek_phase2_{mode}_submit_aggregate_after_groups.sh"
        aggregate_submit_path.write_text(
            _submit_aggregate_text(submit_aggregate_path, absolute_output / "logs"),
            encoding="utf-8",
        )
        created.append(aggregate_submit_path)
    return created


def deterministic_run_seed(base_seed: int, outer_fold_id: int, inner_fold_id: int, replay_seed_size: int) -> int:
    return int(base_seed) + int(outer_fold_id) * 1000 + int(inner_fold_id) * 100 + int(replay_seed_size)


def load_frozen_model_config(phase1_root: Path) -> dict[str, dict[str, object]]:
    path = phase1_root / "frozen_model_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1 frozen model config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen_model_config.json must contain an object")
    return payload


def load_replay_manifest(phase1_root: Path, outer_fold_id: int, inner_fold_id: int) -> dict[str, object]:
    path = phase1_root / "folds" / f"replay_manifest_outer_{outer_fold_id}_inner_{inner_fold_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1 replay manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_phase1_inputs(phase1_root: Path) -> None:
    load_frozen_model_config(phase1_root)
    baseline = phase1_root / "tables" / "reproduced_predictive_performance.csv"
    if not baseline.exists():
        raise FileNotFoundError(f"Missing Phase 1 reproduced baseline table: {baseline}")


def _norm_row(row: dict[str, object], split: str, replay_role: str) -> dict[str, str]:
    return {
        "sequence": str(row["sequence"]),
        "label": str(row["label"]),
        "split": split,
        "replay_role": replay_role,
        "label_source": "experimental",
    }


def _metric_row(spec: ReplayRunSpec, strategy: str, round_id: int, labeled_count: int, dataset: str, metrics: dict[str, object]) -> dict[str, object]:
    row = {
        "mode": spec.mode,
        "setup": spec.setup,
        "outer_fold_id": spec.outer_fold_id,
        "inner_fold_id": spec.inner_fold_id,
        "replay_seed_size": spec.replay_seed_size,
        "initial_label_count": spec.initial_label_count,
        "run_seed": spec.run_seed,
        "strategy": strategy,
        "round_id": round_id,
        "labeled_count": labeled_count,
        "evaluation_dataset": dataset,
    }
    row.update(metrics)
    return row


def _calibration_row(spec: ReplayRunSpec, strategy: str, round_id: int, probability_type: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "mode": spec.mode,
        "setup": spec.setup,
        "outer_fold_id": spec.outer_fold_id,
        "inner_fold_id": spec.inner_fold_id,
        "replay_seed_size": spec.replay_seed_size,
        "initial_label_count": spec.initial_label_count,
        "strategy": strategy,
        "round_id": round_id,
        "probability_type": probability_type,
        "Brier": metrics.get("brier_score", ""),
        "ECE_10": metrics.get("ece_10", ""),
        "MCE_10": metrics.get("mce_10", ""),
    }


def _candidate_log_row(
    spec: ReplayRunSpec,
    strategy: str,
    round_id: int,
    row: dict[str, str],
    index: int,
    candidate_scores: dict[str, np.ndarray],
    acquisition_scores: np.ndarray,
    diagnostics: list[dict[str, object]],
    *,
    selected: bool,
) -> dict[str, object]:
    payload = {
        "mode": spec.mode,
        "setup": spec.setup,
        "outer_fold_id": spec.outer_fold_id,
        "inner_fold_id": spec.inner_fold_id,
        "replay_seed_size": spec.replay_seed_size,
        "initial_label_count": spec.initial_label_count,
        "strategy": strategy,
        "round_id": round_id,
        "sequence": row["sequence"],
        "label_revealed_after_selection": bool(selected),
        "selected": bool(selected),
        "pred_mean": _array_value(candidate_scores, "pred_mean", index),
        "pred_std": _array_value(candidate_scores, "pred_std", index),
        "pred_entropy": _array_value(candidate_scores, "pred_entropy", index),
        "pred_mutual_information": _array_value(candidate_scores, "pred_mutual_information", index),
        "acquisition_score": float(acquisition_scores[index]),
    }
    if "ensemble_member_probs" in candidate_scores:
        payload["ensemble_member_probs"] = serialize_probabilities(candidate_scores["ensemble_member_probs"][index])
    if "raw_ensemble_member_probs" in candidate_scores:
        payload["raw_ensemble_member_probs"] = serialize_probabilities(candidate_scores["raw_ensemble_member_probs"][index])
    payload.update(diagnostics[index])
    return payload


def _array_value(mapping: dict[str, np.ndarray], key: str, index: int) -> float:
    if key not in mapping:
        return 0.0
    return float(np.asarray(mapping[key])[index])


def _collect_run_csv(mode_root: Path, filename: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted((mode_root / "runs").glob(f"*/{filename}")):
        rows.extend(_read_csv(path))
    return rows


def _final_rows_by_outer_fold(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("outer_fold_id", ""), []).append(row)
    finals = []
    for group in grouped.values():
        finals.append(max(group, key=lambda row: int(float(row.get("round_id", 0) or 0))))
    return finals


def _final_rows_by_group(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("mode", ""),
            row.get("initial_label_count", row.get("replay_seed_size", "")),
            row.get("setup", ""),
            row.get("evaluation_dataset", ""),
            row.get("outer_fold_id", ""),
            row.get("strategy", ""),
        )
        grouped.setdefault(key, []).append(row)
    result: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for (mode, initial, setup, dataset, outer, strategy), group in grouped.items():
        final = max(group, key=lambda row: int(float(row.get("round_id", 0) or 0)))
        result.setdefault((mode, initial, setup, dataset, outer), {})[strategy] = final
    return result


def _aulc_by_fold(rows: list[dict[str, str]], metric: str) -> list[float]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("outer_fold_id", ""), []).append(row)
    return [area_under_learning_curve(group, metric) for group in grouped.values()]


def area_under_learning_curve(points: list[dict[str, str]], metric: str) -> float:
    ordered = sorted(points, key=lambda row: float(row.get("labeled_count", 0) or 0))
    xs = [_float(row.get("labeled_count")) for row in ordered]
    ys = [_float(row.get(metric)) for row in ordered]
    filtered = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(filtered) < 2:
        return filtered[0][1] if filtered else 0.0
    x_values = np.asarray([item[0] for item in filtered], dtype=float)
    y_values = np.asarray([item[1] for item in filtered], dtype=float)
    width = float(x_values[-1] - x_values[0])
    if width <= 0:
        return float(y_values[-1])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y_values, x_values) / width)


def labels_to_target(points: list[dict[str, str]], metric: str, target: float) -> int | None:
    ordered = sorted(points, key=lambda row: float(row.get("round_id", 0) or 0))
    for row in ordered:
        value = _float(row.get(metric))
        if value is not None and value >= target:
            labeled = _float(row.get("labeled_count"))
            return int(labeled) if labeled is not None else None
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean(values: Sequence[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def _std(values: Sequence[float]) -> float | str:
    return float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else "")


def _median(values: Sequence[float]) -> float | str:
    return float(np.median(values)) if values else ""


def _float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_text_number(value: str) -> tuple[float, str]:
    parsed = _float(value)
    return (parsed if parsed is not None else float("inf"), str(value))


def _none_subtract(left: float | None, right: float | None) -> float | str:
    if left is None or right is None:
        return ""
    return float(left - right)


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def _nonnegative_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _positive_ints(values: Iterable[int], name: str) -> list[int]:
    return [_positive_int(value, name) for value in values]


def _safe_argv() -> list[str]:
    import sys

    return list(sys.argv)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _run_id(mode: str, setup: str, outer_fold: int, inner_fold: int, replay_seed_size: int) -> str:
    return f"{mode}_outer_{outer_fold}_inner_{inner_fold}_initial_{replay_seed_size}_{setup}"


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _pbs_target_path(path: Path, pbs_repo_root: Path | None) -> Path:
    if path.is_absolute():
        return path
    if pbs_repo_root is not None:
        return pbs_repo_root / path
    return (Path.cwd() / path).resolve()


def _pbs_repo_root(options: Phase2Options) -> Path:
    return options.pbs_repo_root if options.pbs_repo_root is not None else Path.cwd().resolve()


def _phase2_command(options: Phase2Options, phase1_root: Path, output_root: Path, mode: str) -> str:
    strategies = " ".join(options.strategies or ("random", "ensemble_mean", "predictive_entropy"))
    outer_folds = " ".join(str(item) for item in options.outer_folds)
    replay_seed_sizes = " ".join(str(item) for item in options.replay_seed_sizes)
    return (
        f"python -m active_learning_thesis phase2-replay --mode {mode} "
        f"--phase1-root {_shell_quote(phase1_root.as_posix())} --output-root {_shell_quote(output_root.as_posix())} "
        f"--outer-folds {outer_folds} --inner-fold {options.inner_fold} --replay-seed-sizes {replay_seed_sizes} "
        f"--batch-size {options.batch_size} --max-rounds {options.max_rounds} --strategies {strategies} "
        f"--setup {options.setup or 'ensemble_calibrated'}"
    )


def _phase2_command_for_spec(options: Phase2Options, phase1_root: Path, output_root: Path, spec: ReplayRunSpec) -> str:
    strategies = " ".join(spec.strategies)
    calibrated = " --calibrated" if spec.use_calibrated_acquisition else ""
    return (
        f"python -m active_learning_thesis phase2-replay --mode {spec.mode} "
        f"--phase1-root {_shell_quote(phase1_root.as_posix())} --output-root {_shell_quote(output_root.as_posix())} "
        f"--outer-folds {spec.outer_fold_id} --inner-fold {spec.inner_fold_id} --replay-seed-sizes {spec.replay_seed_size} "
        f"--batch-size {spec.batch_size} --max-rounds {spec.max_rounds} --strategies {strategies} "
        f"--setup {spec.setup} --ensemble-size {spec.ensemble_size} --base-seed {spec.base_seed}{calibrated}"
    )


def _pbs_text(job_name: str, walltime: str, command: str, log_dir: Path, repo_root: Path) -> str:
    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=40GB
#PBS -l walltime={walltime}
#PBS -o {log_dir.as_posix()}/{job_name}.out
#PBS -e {log_dir.as_posix()}/{job_name}.err

set -euo pipefail
cd "{repo_root.as_posix()}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
echo "[phase2] start $(date -Is) host=$(hostname)"
echo "[phase2] workdir=$(pwd)"
echo "[phase2] python=$(python --version)"
echo "[phase2] conda_env=${{CONDA_DEFAULT_ENV:-}}"
git rev-parse HEAD || true
nvidia-smi || true
echo "[phase2] command: {command}"
{command}
echo "[phase2] end $(date -Is)"
"""


def _setup_job_code(setup: str) -> str:
    codes = {
        "single_raw": "sr",
        "ensemble_raw": "er",
        "single_calibrated": "sc",
        "ensemble_calibrated": "ec",
    }
    return codes.get(setup, setup.replace("_", "")[:8])


def _chunks(items: Sequence[Path], size: int) -> list[list[Path]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _submit_all_text(job_paths: list[Path], aggregate_path: Path, logs_dir: Path) -> str:
    logs_text = logs_dir.as_posix()
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(logs_text)}",
        f'JOB_LOG="{logs_text}/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"',
        "IDS=()",
    ]
    for path in job_paths:
        lines.extend(
            [
                f'jid=$(qsub "{path.as_posix()}")',
                'echo "$jid" | tee -a "$JOB_LOG"',
                'IDS+=("${jid%%.*}")',
            ]
        )
    lines.extend(
        [
            'dep=$(IFS=:; echo "${IDS[*]}")',
            f'qsub -W depend=afterok:$dep "{aggregate_path.as_posix()}" | tee -a "$JOB_LOG"',
        ]
    )
    return "\n".join(lines) + "\n"


def _submit_group_text(job_paths: list[Path], logs_dir: Path, mode: str, group_index: int) -> str:
    logs_text = logs_dir.as_posix()
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(logs_text)}",
        f'JOB_LOG="{logs_text}/submitted_jobs_{mode}_group_{group_index:02d}_$(date +%Y%m%d_%H%M%S).txt"',
        f'echo "[phase2] submitting {len(job_paths)} {mode} jobs in group {group_index:02d}" | tee -a "$JOB_LOG"',
    ]
    for path in job_paths:
        lines.extend(
            [
                f'jid=$(qsub "{path.as_posix()}")',
                'echo "$jid" | tee -a "$JOB_LOG"',
            ]
        )
    lines.append('echo "[phase2] group submitted; run aggregate only after all groups finish successfully." | tee -a "$JOB_LOG"')
    return "\n".join(lines) + "\n"


def _submit_aggregate_text(aggregate_path: Path, logs_dir: Path) -> str:
    logs_text = logs_dir.as_posix()
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(logs_text)}",
        f'JOB_LOG="{logs_text}/submitted_aggregate_$(date +%Y%m%d_%H%M%S).txt"',
        'echo "[phase2] submitting aggregate job without dependency; use only after all groups finish." | tee -a "$JOB_LOG"',
        f'qsub "{aggregate_path.as_posix()}" | tee -a "$JOB_LOG"',
    ]
    return "\n".join(lines) + "\n"


def _plot_learning_curves(plt, rows: list[dict[str, str]], metric: str, title: str, path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (row.get("setup", ""), row.get("strategy", ""), row.get("initial_label_count", "")),
            [],
        ).append(row)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for (setup, strategy, initial), points in sorted(grouped.items()):
        round_groups: dict[float, list[float]] = {}
        for row in points:
            labeled = _float(row.get("labeled_count"))
            value = _float(row.get(metric))
            if labeled is None or value is None:
                continue
            round_groups.setdefault(labeled, []).append(value)
        ordered = sorted((labeled, _mean(values)) for labeled, values in round_groups.items())
        if not ordered:
            continue
        label_parts = [part for part in [setup, strategy] if part]
        label = "/".join(label_parts) if label_parts else strategy
        ax.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            label=f"{label}, initial labels={initial}",
            alpha=0.8,
            linewidth=2.0,
        )
    ax.set_xlabel("Number of labeled peptides")
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(title)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        columns = 2 if len(handles) > 12 else 1
        ax.legend(
            handles,
            labels,
            fontsize=7,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=columns,
            borderaxespad=0.0,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_strategy_metric_by_initial(
    plt,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
    *,
    lower_is_better: bool = False,
    limit: int = 12,
    annotation_field: str | None = None,
) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    annotations: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        strategy = row.get("strategy", "")
        initial = row.get("initial_label_count", "")
        grouped.setdefault((strategy, initial), []).append(value)
        if annotation_field and row.get(annotation_field, ""):
            annotations.setdefault((strategy, initial), []).append(str(row[annotation_field]))
    if not grouped:
        return
    initials = sorted({initial for _, initial in grouped}, key=_sort_text_number)
    strategies = sorted(
        {strategy for strategy, _ in grouped},
        key=lambda strategy: _mean(
            [
                _mean(values)
                for (candidate, _), values in grouped.items()
                if candidate == strategy
            ]
        ),
        reverse=not lower_is_better,
    )[:limit]
    value_lookup = {
        key: _mean(values)
        for key, values in grouped.items()
    }
    plotted_values = [
        value_lookup[(strategy, initial)]
        for strategy in strategies
        for initial in initials
        if (strategy, initial) in value_lookup
    ]
    if not plotted_values:
        return
    spread = max(plotted_values) - min(plotted_values)
    pad = spread * 0.12 if spread > 0 else max(abs(max(plotted_values)) * 0.08, 0.02)
    x_min = min(plotted_values) - pad
    x_max = max(plotted_values) + pad
    if min(plotted_values) < 0 < max(plotted_values):
        x_min = min(x_min, -pad)
        x_max = max(x_max, pad)

    fig_height = max(4.5, 1.8 + 0.42 * len(strategies))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    y_positions = np.arange(len(strategies), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(initials)) if len(initials) > 1 else np.asarray([0.0])
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    markers = ["o", "s", "^", "D"]
    for index, initial in enumerate(initials):
        xs = []
        ys = []
        labels = []
        for y_index, strategy in enumerate(strategies):
            key = (strategy, initial)
            if key not in value_lookup:
                continue
            xs.append(value_lookup[key])
            ys.append(y_positions[y_index] + offsets[index])
            labels.append(strategy)
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=58,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            label=f"initial labels={initial}",
            zorder=3,
        )
        for x, y, strategy in zip(xs, ys, labels):
            suffix = ""
            annotation_values = annotations.get((strategy, initial), [])
            if annotation_values:
                suffix = f" ({annotation_values[0]})"
            ax.text(x + pad * 0.035, y + 0.03, f"{x:.3f}{suffix}", fontsize=7.5, color="#334155")
    if x_min < 0 < x_max:
        ax.axvline(0, color="#64748b", linewidth=1.1, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(strategies, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_bar(
    plt,
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    *,
    limit: int = 12,
    lower_is_better: bool = False,
) -> None:
    points: list[tuple[float, str]] = []
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        label = _display_phase2_label(row.get("setup", ""), row.get("strategy", ""))
        points.append((value, label))
    points.sort(key=lambda item: item[0], reverse=not lower_is_better)
    points = points[:limit]
    if not points:
        return
    values = [point[0] for point in points]
    labels = [point[1] for point in points]
    plt.figure(figsize=(10, max(4, 0.34 * len(points))))
    plt.barh(range(len(points)), values)
    plt.yticks(range(len(points)), labels, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_setup_metric_bar(plt, rows: list[dict[str, str]], metric: str, ylabel: str, path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((row.get("initial_label_count", ""), row.get("setup", "")), []).append(value)
    points = [
        (_mean(values), f"{setup}\ninitial labels={initial}")
        for (initial, setup), values in sorted(grouped.items())
        if values
    ]
    if not points:
        return
    values = [point[0] for point in points]
    labels = [point[1] for point in points]
    plt.figure(figsize=(9, 4.8))
    plt.bar(range(len(points)), values)
    plt.xticks(range(len(points)), labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(ylabel + " by setup")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _paired_delta_summary_rows(rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("evaluation_dataset") != dataset:
            continue
        grouped.setdefault((row.get("initial_label_count", ""), row.get("strategy", "")), []).append(row)
    summary: list[dict[str, str]] = []
    for (initial, strategy), group in sorted(grouped.items(), key=lambda item: (_sort_text_number(item[0][0]), item[0][1])):
        deltas = [
            value
            for value in (_float(row.get("final_F1_delta_vs_random")) for row in group)
            if value is not None
        ]
        if not deltas:
            continue
        wins = sum(1 for row in group if str(row.get("win_vs_random", "")).lower() == "true")
        summary.append(
            {
                "initial_label_count": initial,
                "strategy": strategy,
                "mean_final_F1_delta_vs_random": str(_mean(deltas)),
                "wins_vs_random": f"{wins}/{len(group)} wins",
            }
        )
    return summary


def _labels_saved_vs_random_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    random_row = next((row for row in rows if row.get("strategy") == "random"), None)
    random_labels = _float(random_row.get("mean_labels_to_target")) if random_row else None
    if random_labels is None or random_labels <= 0:
        return []
    result: list[dict[str, str]] = []
    for row in rows:
        if row.get("strategy") == "random":
            continue
        labels = _float(row.get("mean_labels_to_target"))
        if labels is None:
            continue
        saved = random_labels - labels
        result.append(
            {
                "strategy": row.get("strategy", ""),
                "labels_saved_vs_random": str(saved),
                "percent_labels_saved_vs_random": str(100.0 * saved / random_labels),
                "mean_labels_to_target": row.get("mean_labels_to_target", ""),
                "random_mean_labels_to_target": str(random_labels),
                "reached_count": row.get("reached_count", ""),
                "n_repeats": row.get("n_repeats", ""),
            }
        )
    return result


PHASE2_STRATEGY_DISPLAY_NAMES = {
    "random": "Random",
    "ensemble_mean": "Mean",
    "similarity_penalized_mean": "Similarity-penalized mean",
    "predictive_entropy": "Predictive entropy",
    "ensemble_mi": "Ensemble MI",
    "ucb": "UCB",
    "family_qbc": "Family QBC",
    "cluster_diverse_representative": "Cluster-diverse",
    "oed_logdet": "OED logdet",
    "hybrid_mi_diverse": "Hybrid MI + diverse",
}
PHASE2_STRATEGY_ORDER = [
    "random",
    "predictive_entropy",
    "family_qbc",
    "cluster_diverse_representative",
    "ensemble_mi",
    "hybrid_mi_diverse",
    "oed_logdet",
    "ensemble_mean",
    "similarity_penalized_mean",
    "ucb",
]
PHASE2_HIGHLIGHT_STRATEGIES = {"predictive_entropy", "family_qbc", "cluster_diverse_representative", "random"}


def _display_strategy(strategy: str) -> str:
    return PHASE2_STRATEGY_DISPLAY_NAMES.get(strategy, strategy.replace("_", " "))


def _display_phase2_label(setup: str, strategy: str) -> str:
    strategy_label = _display_strategy(strategy)
    if setup and setup != "ensemble_calibrated":
        return f"{setup.replace('_', ' ')} / {strategy_label}"
    return strategy_label


def _strategy_sort_key(strategy: str) -> tuple[int, str]:
    try:
        return (PHASE2_STRATEGY_ORDER.index(strategy), strategy)
    except ValueError:
        return (len(PHASE2_STRATEGY_ORDER), strategy)


def _phase2_strategy_color(strategy: str, index: int = 0) -> str:
    fixed = {
        "random": "#475569",
        "predictive_entropy": "#2563eb",
        "family_qbc": "#7c3aed",
        "cluster_diverse_representative": "#16a34a",
        "ensemble_mi": "#0891b2",
        "hybrid_mi_diverse": "#ea580c",
        "oed_logdet": "#ca8a04",
        "ensemble_mean": "#64748b",
        "similarity_penalized_mean": "#be123c",
        "ucb": "#0f766e",
    }
    return fixed.get(strategy, _svg_palette()[index % len(_svg_palette())])


def _write_svg_learning_curves(rows: list[dict[str, str]], metric: str, title: str, path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (row.get("setup", ""), row.get("strategy", ""), row.get("initial_label_count", "")),
            [],
        ).append(row)
    series: list[tuple[str, str, str, list[tuple[float, float]]]] = []
    for (setup, strategy, initial), points in sorted(grouped.items(), key=lambda item: (item[0][0], _strategy_sort_key(item[0][1]), _sort_text_number(item[0][2]))):
        by_labeled: dict[float, list[float]] = {}
        for row in points:
            labeled = _float(row.get("labeled_count"))
            value = _float(row.get(metric))
            if labeled is None or value is None:
                continue
            by_labeled.setdefault(labeled, []).append(value)
        ordered = sorted((x, _mean(values)) for x, values in by_labeled.items())
        if ordered:
            label = _display_phase2_label(setup, strategy)
            series.append((setup, strategy, label, ordered))
    if not series:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    setups = {setup for setup, _, _, _ in series if setup}
    if len(setups) > 1:
        _write_svg_learning_curves_by_setup(series, metric, title, path)
        return
    xs = [x for _, _, _, points in series for x, _ in points]
    ys = [y for _, _, _, points in series for _, y in points]
    width, height = 1220, 720
    left, right, top, bottom = 90, 360, 70, 88
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / max(y_max - y_min, 1e-9) * (height - top - bottom)

    body = [_svg_header(width, height, title)]
    body.append(f'<text x="{left}" y="54" font-size="12" fill="#64748b">Lines show mean across outer folds; random is dashed; Phase 3 strategies are emphasized.</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>')
    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5
        x = sx(value)
        body.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#334155"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="12" fill="#475569">{value:.0f}</text>')
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>')
    for index, (_setup, strategy, label, points) in enumerate(series[:24]):
        color = _phase2_strategy_color(strategy, index)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        width_px = 3.2 if strategy in PHASE2_HIGHLIGHT_STRATEGIES else 1.7
        opacity = 0.96 if strategy in PHASE2_HIGHLIGHT_STRATEGIES else 0.48
        dash = ' stroke-dasharray="5 5"' if strategy == "random" else ""
        body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width_px}" opacity="{opacity}"{dash}/>')
        legend_y = top + 20 * index
        body.append(f'<rect x="{width-right+30}" y="{legend_y-10}" width="10" height="10" fill="{color}"/>')
        weight = "600" if strategy in PHASE2_HIGHLIGHT_STRATEGIES else "400"
        body.append(f'<text x="{width-right+46}" y="{legend_y}" font-size="11" font-weight="{weight}" fill="#334155">{html.escape(label)}</text>')
    body.append(f'<text x="{(left+width-right)/2:.1f}" y="{height-25}" text-anchor="middle" font-size="13" fill="#334155">Number of labeled peptides</text>')
    body.append(f'<text x="24" y="{(top+height-bottom)/2:.1f}" transform="rotate(-90 24 {(top+height-bottom)/2:.1f})" text-anchor="middle" font-size="13" fill="#334155">{html.escape(metric.upper())}</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_learning_curves_by_setup(
    series: list[tuple[str, str, str, list[tuple[float, float]]]],
    metric: str,
    title: str,
    path: Path,
) -> None:
    xs = [x for _, _, _, points in series for x, _ in points]
    ys = [y for _, _, _, points in series for _, y in points]
    if not xs or not ys:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    width, height = 1280, 820
    left, right, top, bottom = 86, 310, 92, 78
    gap_x, gap_y = 56, 72
    cols = 2
    setup_order = [setup for setup in ABLATION_SETUPS if setup in {item[0] for item in series}]
    setup_order.extend(sorted({item[0] for item in series if item[0] not in setup_order}))
    rows = max(1, (len(setup_order) + cols - 1) // cols)
    panel_w = (width - left - right - gap_x * (cols - 1)) / cols
    panel_h = (height - top - bottom - gap_y * (rows - 1)) / rows
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def sx(value: float, panel_left: float) -> float:
        return panel_left + (value - x_min) / max(x_max - x_min, 1e-9) * panel_w

    def sy(value: float, panel_top: float) -> float:
        return panel_top + panel_h - (value - y_min) / max(y_max - y_min, 1e-9) * panel_h

    by_setup: dict[str, list[tuple[str, str, list[tuple[float, float]]]]] = {}
    for setup, strategy, label, points in series:
        by_setup.setdefault(setup, []).append((strategy, _display_strategy(strategy), points))

    body = [_svg_header(width, height, title)]
    body.append(f'<text x="{left}" y="58" font-size="12" fill="#64748b">Small multiples split the ablation setups; y-axis is shared, random is dashed, Phase 3 strategies are emphasized.</text>')
    for setup_index, setup in enumerate(setup_order):
        row = setup_index // cols
        col = setup_index % cols
        panel_left = left + col * (panel_w + gap_x)
        panel_top = top + row * (panel_h + gap_y)
        panel_bottom = panel_top + panel_h
        body.append(f'<text x="{panel_left:.1f}" y="{panel_top-18:.1f}" font-size="13" font-weight="600" fill="#0f172a">{html.escape(setup.replace("_", " "))}</text>')
        body.append(f'<rect x="{panel_left:.1f}" y="{panel_top:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#ffffff" stroke="#e2e8f0"/>')
        for i in range(4):
            value = x_min + (x_max - x_min) * i / 3
            x = sx(value, panel_left)
            body.append(f'<line x1="{x:.1f}" y1="{panel_top:.1f}" x2="{x:.1f}" y2="{panel_bottom:.1f}" stroke="#eef2f7"/>')
            body.append(f'<text x="{x:.1f}" y="{panel_bottom+17:.1f}" text-anchor="middle" font-size="10" fill="#475569">{value:.0f}</text>')
        for i in range(4):
            value = y_min + (y_max - y_min) * i / 3
            y = sy(value, panel_top)
            body.append(f'<line x1="{panel_left:.1f}" y1="{y:.1f}" x2="{panel_left+panel_w:.1f}" y2="{y:.1f}" stroke="#eef2f7"/>')
            if col == 0:
                body.append(f'<text x="{panel_left-9:.1f}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#475569">{value:.2f}</text>')
        body.append(f'<line x1="{panel_left:.1f}" y1="{panel_bottom:.1f}" x2="{panel_left+panel_w:.1f}" y2="{panel_bottom:.1f}" stroke="#334155"/>')
        body.append(f'<line x1="{panel_left:.1f}" y1="{panel_top:.1f}" x2="{panel_left:.1f}" y2="{panel_bottom:.1f}" stroke="#334155"/>')
        for index, (strategy, _label, points) in enumerate(sorted(by_setup.get(setup, []), key=lambda item: _strategy_sort_key(item[0]))):
            color = _phase2_strategy_color(strategy, index)
            coords = " ".join(f"{sx(x, panel_left):.1f},{sy(y, panel_top):.1f}" for x, y in points)
            width_px = 3.0 if strategy in PHASE2_HIGHLIGHT_STRATEGIES else 1.6
            opacity = 0.95 if strategy in PHASE2_HIGHLIGHT_STRATEGIES else 0.58
            dash = ' stroke-dasharray="5 5"' if strategy == "random" else ""
            body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width_px}" opacity="{opacity}"{dash}/>')
    legend_x = width - right + 28
    legend_y = top
    strategies = sorted({strategy for _, strategy, _, _ in series}, key=_strategy_sort_key)
    body.append(f'<text x="{legend_x}" y="{legend_y-22}" font-size="12" font-weight="600" fill="#0f172a">Strategy</text>')
    for index, strategy in enumerate(strategies):
        y = legend_y + 20 * index
        color = _phase2_strategy_color(strategy, index)
        body.append(f'<rect x="{legend_x}" y="{y-10}" width="10" height="10" fill="{color}"/>')
        weight = "600" if strategy in PHASE2_HIGHLIGHT_STRATEGIES else "400"
        body.append(f'<text x="{legend_x+18}" y="{y}" font-size="11" font-weight="{weight}" fill="#334155">{html.escape(_display_strategy(strategy))}</text>')
    body.append(f'<text x="{(left + width - right)/2:.1f}" y="{height-22}" text-anchor="middle" font-size="13" fill="#334155">Number of labeled peptides</text>')
    body.append(f'<text x="24" y="{(top + height - bottom)/2:.1f}" transform="rotate(-90 24 {(top + height - bottom)/2:.1f})" text-anchor="middle" font-size="13" fill="#334155">{html.escape(metric.upper())}</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_summary_bar(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    path: Path,
    *,
    limit: int = 12,
    lower_is_better: bool = False,
) -> None:
    points: list[tuple[float, str]] = []
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        label = _display_phase2_label(row.get("setup", ""), row.get("strategy", ""))
        points.append((value, label))
    points.sort(key=lambda item: item[0], reverse=not lower_is_better)
    _write_svg_bar(points[:limit], title, path)


def _write_svg_setup_metric_bar(rows: list[dict[str, str]], metric: str, path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((row.get("initial_label_count", ""), row.get("setup", "")), []).append(value)
    points = [
        (_mean(values), f"{setup}, initial labels={initial}")
        for (initial, setup), values in sorted(grouped.items())
        if values
    ]
    _write_svg_bar(points, metric.replace("_", " ").title(), path)


def _write_svg_bar(points: list[tuple[float, str]], title: str, path: Path) -> None:
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    width = 1000
    row_height = 28
    height = max(260, 90 + row_height * len(points))
    left, right, top = 320, 50, 58
    max_value = max(value for value, _ in points) or 1.0
    body = [_svg_header(width, height, title)]
    colors = _svg_palette()
    for index, (value, label) in enumerate(points):
        y = top + index * row_height
        bar_width = (width - left - right) * value / max_value
        color = colors[index % len(colors)]
        body.append(f'<text x="{left-10}" y="{y+16}" text-anchor="end" font-size="12" fill="#334155">{html.escape(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="18" rx="3" fill="{color}" opacity="0.82"/>')
        body.append(f'<text x="{left+bar_width+6:.1f}" y="{y+14}" font-size="12" fill="#334155">{value:.3f}</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_ranked_dot_plot(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    x_label: str,
    path: Path,
    *,
    limit: int = 12,
    lower_is_better: bool = False,
) -> None:
    points: list[tuple[float, float, str, str]] = []
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        strategy = row.get("strategy", "")
        label = _display_phase2_label(row.get("setup", ""), strategy)
        std = _float(row.get(metric.replace("mean_", "std_", 1))) if metric.startswith("mean_") else None
        points.append((value, std or 0.0, label, strategy))
    points.sort(key=lambda item: item[0], reverse=not lower_is_better)
    points = points[:limit]
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    values = [value for value, _, _, _ in points]
    errors = [std for _, std, _, _ in points]
    data_min = min(values)
    data_max = max(values)
    error_max = max(errors) if errors else 0.0
    span = max(data_max - data_min, 1e-9)
    # Leave enough room for both whiskers and text annotations.  The older
    # version zoomed tightly to the means, which made labels touch dots and
    # clipped long whiskers in presentation exports.
    x_min = max(0.0, data_min - error_max - span * 0.32)
    x_max = min(1.0, data_max + error_max + span * 0.42)
    if x_max - x_min < 0.03:
        center = (x_min + x_max) / 2
        x_min = max(0.0, center - 0.02)
        x_max = min(1.0, center + 0.02)
    width = 1180
    row_height = 36
    height = max(350, 128 + row_height * len(points))
    left, right, top, bottom = 330, 170, 82, 70

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)

    plot_right = width - right
    title_x = (left + plot_right) / 2
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="{title_x:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="18" font-weight="600" fill="#0f172a">{html.escape(title)}</text>'
        )
    ]
    body.append(f'<text x="{title_x:.1f}" y="54" text-anchor="middle" font-size="12" fill="#64748b">Dots show mean across 5 outer folds; whiskers and labels show SD. Higher is better unless stated otherwise.</text>')
    plot_bottom = height - bottom
    for index in range(5):
        tick = x_min + (x_max - x_min) * index / 4
        x = sx(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.1f}" y="{plot_bottom+18}" text-anchor="middle" font-size="11" fill="#475569">{tick:.3f}</text>')
    body.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{width-right}" y2="{plot_bottom}" stroke="#334155"/>')
    random_point = next((value for value, _, _, strategy in points if strategy == "random"), None)
    if random_point is not None and x_min <= random_point <= x_max:
        random_x = sx(random_point)
        body.append(f'<line x1="{random_x:.1f}" y1="{top-8}" x2="{random_x:.1f}" y2="{plot_bottom}" stroke="#475569" stroke-width="1.3" stroke-dasharray="5 5"/>')
        body.append(f'<text x="{random_x:.1f}" y="{top-18}" text-anchor="middle" font-size="11" fill="#475569">random</text>')
    for index, (value, std, label, strategy) in enumerate(points):
        y = top + index * row_height
        x = sx(value)
        color = _phase2_strategy_color(strategy, index)
        weight = "600" if strategy in PHASE2_HIGHLIGHT_STRATEGIES else "400"
        body.append(f'<text x="{left-12}" y="{y+15}" text-anchor="end" font-size="12" font-weight="{weight}" fill="#334155">{html.escape(label)}</text>')
        body.append(f'<line x1="{left}" y1="{y+10}" x2="{width-right}" y2="{y+10}" stroke="#f1f5f9"/>')
        if std > 0:
            x_low = sx(max(x_min, value - std))
            x_high = sx(min(x_max, value + std))
            body.append(f'<line x1="{x_low:.1f}" y1="{y+10}" x2="{x_high:.1f}" y2="{y+10}" stroke="{color}" stroke-width="2" opacity="0.45"/>')
            body.append(f'<line x1="{x_low:.1f}" y1="{y+6}" x2="{x_low:.1f}" y2="{y+14}" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
            body.append(f'<line x1="{x_high:.1f}" y1="{y+6}" x2="{x_high:.1f}" y2="{y+14}" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y+10}" r="5.5" fill="{color}"/>')
        value_y = y + (1 if index % 2 == 0 else 23)
        body.append(f'<text x="{min(x + 12, plot_right - 46):.1f}" y="{value_y:.1f}" font-size="12" fill="#0f172a">{value:.3f}</text>')
        if std > 0:
            sd_x = min(sx(min(x_max, value + std)) + 8, width - 84)
            body.append(f'<text x="{sd_x:.1f}" y="{y+14}" font-size="10.5" fill="#64748b">SD {std:.3f}</text>')
    body.append(f'<text x="{title_x:.1f}" y="{height-16}" text-anchor="middle" font-size="13" fill="#334155">{html.escape(x_label)}</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_labels_saved_bar(rows: list[dict[str, str]], title: str, path: Path) -> None:
    points: list[tuple[float, float, str, str]] = []
    for row in rows:
        saved = _float(row.get("labels_saved_vs_random"))
        percent = _float(row.get("percent_labels_saved_vs_random"))
        if saved is None or percent is None:
            continue
        delta = -saved
        percent_delta = -percent
        reached = row.get("reached_count", "")
        repeats = row.get("n_repeats", "")
        reached_text = f"{reached}/{repeats}" if reached and repeats else ""
        points.append((delta, percent_delta, _display_strategy(row.get("strategy", "")), reached_text))
    points.sort(key=lambda item: item[0])
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    width = 1280
    row_height = 44
    height = max(390, 162 + row_height * len(points))
    left, right, top, bottom = 330, 300, 100, 86
    values = [value for value, _, _, _ in points]
    min_value = min(min(values), 0.0)
    max_value = max(max(values), 0.0)
    span = max(max_value - min_value, 1e-9)
    x_min = min_value - span * 0.14
    x_max = max_value + span * 0.14

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)

    zero_x = sx(0.0)
    value_text_x = width - 34
    text_center_x = (left + value_text_x) / 2
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="{text_center_x:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="18" font-weight="600" fill="#0f172a">{html.escape(title)}</text>'
        )
    ]
    legend_y = 58
    body.append(f'<rect x="{width-right-166}" y="{legend_y-11}" width="12" height="12" fill="#2563eb" opacity="0.72"/>')
    body.append(f'<text x="{width-right-148}" y="{legend_y}" font-size="11" fill="#334155">fewer labels than random</text>')
    body.append(f'<rect x="{width-176}" y="{legend_y-11}" width="12" height="12" fill="#dc2626" opacity="0.72"/>')
    body.append(f'<text x="{width-158}" y="{legend_y}" font-size="11" fill="#334155">more labels</text>')
    plot_bottom = height - bottom
    for index in range(5):
        tick = x_min + (x_max - x_min) * index / 4
        x = sx(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.1f}" y="{plot_bottom+18}" text-anchor="middle" font-size="11" fill="#475569">{tick:+.0f}</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="{top-20}" x2="{zero_x:.1f}" y2="{plot_bottom}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4 4"/>')
    body.append(f'<text x="{zero_x:.1f}" y="{plot_bottom+36}" text-anchor="middle" font-size="12" fill="#475569">random baseline</text>')
    for index, (delta, percent_delta, label, reached_text) in enumerate(points):
        y = top + index * row_height
        line_y = y + 24
        x = sx(delta)
        color = "#2563eb" if delta <= 0 else "#dc2626"
        suffix = f", {reached_text} folds" if reached_text else ""
        delta_text = f"{delta:+.0f}" if abs(delta - round(delta)) < 0.05 else f"{delta:+.1f}"
        body.append(f'<text x="{left-18}" y="{line_y+4}" text-anchor="end" font-size="12" fill="#334155">{html.escape(label)}</text>')
        body.append(f'<line x1="{zero_x:.1f}" y1="{line_y}" x2="{x:.1f}" y2="{line_y}" stroke="{color}" stroke-width="4" opacity="0.72"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{line_y}" r="5" fill="{color}"/>')
        body.append(f'<text x="{value_text_x}" y="{line_y+4}" text-anchor="end" font-size="12" fill="{color}">{delta_text} labels ({percent_delta:+.0f}%{suffix})</text>')
    body.append(f'<text x="{text_center_x:.1f}" y="{height-16}" text-anchor="middle" font-size="13" fill="#334155">Additional labeled peptides vs random to reach target F1</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_delta_bar(rows: list[dict[str, str]], metric: str, title: str, path: Path) -> None:
    points: list[tuple[float, str, str]] = []
    for row in rows:
        value = _float(row.get(metric))
        if value is None:
            continue
        label = _display_strategy(row.get("strategy", ""))
        wins = row.get("wins_vs_random", "")
        points.append((value, label, wins))
    points.sort(key=lambda item: item[0], reverse=True)
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    width = 1160
    row_height = 42
    height = max(350, 130 + row_height * len(points))
    left, right, top, bottom = 320, 160, 76, 76
    values = [value for value, _, _ in points]
    min_value = min(min(values), 0.0)
    max_value = max(max(values), 0.0)
    span = max(max_value - min_value, 1e-9)
    x_min = min_value - span * 0.12
    x_max = max_value + span * 0.12

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)

    zero_x = sx(0.0)
    title_x = (left + width - right) / 2
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="{title_x:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="18" font-weight="600" fill="#0f172a">{html.escape(title)}</text>'
        )
    ]
    plot_bottom = height - bottom
    for index in range(5):
        tick = x_min + (x_max - x_min) * index / 4
        x = sx(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{x:.1f}" y="{plot_bottom+18}" text-anchor="middle" font-size="11" fill="#475569">{tick:+.3f}</text>')
    body.append(f'<line x1="{zero_x:.1f}" y1="{top-20}" x2="{zero_x:.1f}" y2="{plot_bottom}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4 4"/>')
    body.append(f'<text x="{zero_x:.1f}" y="{plot_bottom+36}" text-anchor="middle" font-size="12" fill="#475569">random baseline</text>')
    body.append(f'<text x="{width-22}" y="52" text-anchor="end" font-size="11" fill="#64748b">wins shown as folds beating random; n=5</text>')
    for index, (value, label, wins) in enumerate(points):
        y = top + index * row_height
        x = sx(value)
        color = "#2563eb" if value >= 0 else "#dc2626"
        body.append(f'<text x="{left-16}" y="{y+12}" text-anchor="end" font-size="12" fill="#334155">{html.escape(label)}</text>')
        if wins:
            body.append(f'<text x="{width-22}" y="{y+31}" text-anchor="end" font-size="11" fill="#64748b">{html.escape(wins)}</text>')
        body.append(f'<line x1="{zero_x:.1f}" y1="{y+26}" x2="{x:.1f}" y2="{y+26}" stroke="{color}" stroke-width="4" opacity="0.72"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y+26}" r="5" fill="{color}"/>')
        text_x = x + 8 if value >= 0 else x - 8
        anchor = "start" if value >= 0 else "end"
        body.append(f'<text x="{text_x:.1f}" y="{y+31}" text-anchor="{anchor}" font-size="12" fill="#334155">{value:+.3f}</text>')
    body.append(f'<text x="{title_x:.1f}" y="{height-16}" text-anchor="middle" font-size="13" fill="#334155">Mean final F1 delta vs random after 20 replay rounds</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_phase2_presentation_svg_figures(input_root: Path, figure_root: Path) -> dict[str, str]:
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    summary_path = input_root / "benchmark" / "strategy_summary.csv"
    if summary_path.exists():
        rows = _read_csv(summary_path)
        for initial in sorted({row.get("initial_label_count", "") for row in rows}, key=_sort_text_number):
            points = [
                row
                for row in rows
                if row.get("evaluation_dataset") == "holdout"
                and row.get("initial_label_count") == initial
                and row.get("setup") == "ensemble_calibrated"
            ]
            if not points:
                continue
            fig_path = figure_root / f"benchmark_holdout_aulc_f1_initial_{initial}_presentation.svg"
            _write_svg_presentation_aulc_plot(
                points,
                f"Benchmark holdout AULC-F1, initial labels = {initial}",
                fig_path,
            )
            outputs[fig_path.name] = str(fig_path)

    labels_path = input_root / "benchmark" / "labels_to_target_summary.csv"
    if labels_path.exists():
        rows = _read_csv(labels_path)
        for initial in sorted({row.get("initial_label_count", "") for row in rows}, key=_sort_text_number):
            points = [
                row
                for row in rows
                if row.get("evaluation_dataset") == "holdout"
                and row.get("initial_label_count") == initial
                and row.get("target_f1") == "0.84"
                and row.get("setup") == "ensemble_calibrated"
            ]
            if not points:
                continue
            fig_path = figure_root / f"benchmark_holdout_labels_to_f1_084_initial_{initial}_presentation.svg"
            _write_svg_presentation_labels_to_target(
                points,
                f"Benchmark holdout labels to F1 = 0.84, initial labels = {initial}",
                fig_path,
            )
            outputs[fig_path.name] = str(fig_path)
            if initial == "40":
                legacy_path = figure_root / "benchmark_holdout_labels_to_f1_084_presentation.svg"
                _write_svg_presentation_labels_to_target(
                    points,
                    "Benchmark holdout labels to F1 = 0.84",
                    legacy_path,
                )
                outputs[legacy_path.name] = str(legacy_path)
    return outputs


def _write_svg_presentation_aulc_plot(rows: list[dict[str, str]], title: str, path: Path) -> None:
    points: list[tuple[float, float, str, str]] = []
    for row in rows:
        value = _float(row.get("mean_AULC_F1"))
        if value is None:
            continue
        std = _float(row.get("std_AULC_F1")) or 0.0
        strategy = row.get("strategy", "")
        points.append((value, std, _display_strategy(strategy), strategy))
    points.sort(key=lambda item: item[0], reverse=True)
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return

    width = 1180
    row_height = 42
    left, right, top, bottom = 300, 210, 88, 72
    height = max(420, top + bottom + row_height * len(points))
    plot_right = width - right
    title_x = (left + plot_right) / 2
    values = [value for value, _, _, _ in points]
    errors = [std for _, std, _, _ in points]
    data_min = min(value - std for value, std, _, _ in points)
    data_max = max(value + std for value, std, _, _ in points)
    span = max(data_max - data_min, 0.01)
    x_min = max(0.0, data_min - span * 0.08)
    x_max = min(1.0, data_max + span * 0.18)

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (plot_right - left)

    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="{title_x:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="22" font-weight="700" fill="#0f172a">{html.escape(title)}</text>'
        )
    ]
    body.append(f'<text x="{title_x:.1f}" y="58" text-anchor="middle" font-size="13" fill="#334155">Dots show mean across 5 outer folds; whiskers show SD. Higher is better.</text>')
    plot_bottom = height - bottom
    for index in range(5):
        tick = x_min + (x_max - x_min) * index / 4
        x = sx(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e2e8f0"/>')
        body.append(f'<text x="{x:.1f}" y="{plot_bottom+22}" text-anchor="middle" font-size="12" fill="#334155">{tick:.2f}</text>')
    random_mean = next((value for value, _, _, strategy in points if strategy == "random"), None)
    if random_mean is not None and x_min <= random_mean <= x_max:
        random_x = sx(random_mean)
        body.append(f'<line x1="{random_x:.1f}" y1="{top-6}" x2="{random_x:.1f}" y2="{plot_bottom}" stroke="#475569" stroke-width="1.6" stroke-dasharray="6 6"/>')
        body.append(f'<text x="{random_x + 10:.1f}" y="{top+10}" font-size="12" fill="#475569">random mean</text>')
    for index, (value, std, label, strategy) in enumerate(points):
        y = top + index * row_height + 16
        x = sx(value)
        color = _phase2_strategy_color(strategy, index)
        weight = "700" if strategy in PHASE2_HIGHLIGHT_STRATEGIES else "400"
        body.append(f'<text x="{left-18}" y="{y+4}" text-anchor="end" font-size="13" font-weight="{weight}" fill="#0f172a">{html.escape(label)}</text>')
        if std > 0:
            x_low = sx(max(x_min, value - std))
            x_high = sx(min(x_max, value + std))
            body.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="{color}" stroke-width="2.2" opacity="0.72"/>')
            sd_x = min(x_high + 12, width - 88)
            body.append(f'<text x="{sd_x:.1f}" y="{y+4}" font-size="11" fill="#475569">SD {std:.3f}</text>')
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{color}"/>')
        value_y = y - 10 if index % 2 == 0 else y + 20
        body.append(f'<text x="{x + 14:.1f}" y="{value_y:.1f}" font-size="12" fill="#0f172a">{value:.3f}</text>')
    body.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#334155"/>')
    body.append(f'<text x="{title_x:.1f}" y="{height-18}" text-anchor="middle" font-size="14" fill="#0f172a">AULC-F1 over number of labeled peptides</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_svg_presentation_labels_to_target(rows: list[dict[str, str]], title: str, path: Path) -> None:
    points: list[tuple[float, int, str, str]] = []
    for row in rows:
        value = _float(row.get("mean_labels_to_target"))
        if value is None:
            continue
        reached = int(_float(row.get("reached_count")) or 0)
        strategy = row.get("strategy", "")
        points.append((value, reached, _display_strategy(strategy), strategy))
    points.sort(key=lambda item: item[0])
    if not points:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return

    width = 1120
    row_height = 38
    left, right, top, bottom = 300, 150, 90, 72
    height = max(390, top + bottom + row_height * len(points))
    plot_right = width - right
    title_x = (left + plot_right) / 2
    max_value = max(value for value, _, _, _ in points) or 1.0
    x_max = max_value * 1.16

    def sx(value: float) -> float:
        return left + value / max(x_max, 1e-9) * (plot_right - left)

    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="{title_x:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="22" font-weight="700" fill="#0f172a">{html.escape(title)}</text>'
        )
    ]
    body.append(f'<text x="{title_x:.1f}" y="58" text-anchor="middle" font-size="13" fill="#334155">Lower is better; labels are mean peptides needed, with reached folds shown as n/5.</text>')
    plot_bottom = height - bottom
    for index in range(5):
        tick = x_max * index / 4
        x = sx(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#e2e8f0"/>')
        body.append(f'<text x="{x:.1f}" y="{plot_bottom+22}" text-anchor="middle" font-size="12" fill="#334155">{tick:.0f}</text>')
    for index, (value, reached, label, strategy) in enumerate(points):
        y = top + index * row_height + 9
        color = _phase2_strategy_color(strategy, index)
        weight = "700" if strategy in PHASE2_HIGHLIGHT_STRATEGIES else "400"
        x = sx(value)
        body.append(f'<text x="{left-18}" y="{y+13}" text-anchor="end" font-size="13" font-weight="{weight}" fill="#0f172a">{html.escape(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{max(1.0, x-left):.1f}" height="18" rx="2" fill="{color}" opacity="0.78"/>')
        body.append(f'<text x="{min(x+10, plot_right-55):.1f}" y="{y+14}" font-size="12" fill="#0f172a">{value:.1f} labels ({reached}/5)</text>')
    body.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#334155"/>')
    body.append(f'<text x="{title_x:.1f}" y="{height-18}" text-anchor="middle" font-size="14" fill="#0f172a">Mean labeled peptides needed to reach target F1</text>')
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_phase2_overlap_svg_figures(input_root: Path, figure_root: Path) -> dict[str, str]:
    overlap_root = input_root / "benchmark" / "overlap"
    if not overlap_root.exists():
        return {}
    outputs: dict[str, str] = {}
    for matrix_path in sorted(overlap_root.glob("pairwise_jaccard_matrix_initial_*.csv")):
        initial = matrix_path.stem.replace("pairwise_jaccard_matrix_initial_", "")
        rows = _read_csv(matrix_path)
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        if len(fieldnames) < 2:
            continue
        strategies = [name for name in fieldnames[1:] if name]
        strategies = sorted(strategies, key=_strategy_sort_key)
        matrix: dict[str, dict[str, float]] = {}
        for row in rows:
            strategy = row.get(fieldnames[0], "")
            matrix[strategy] = {}
            for other in strategies:
                matrix[strategy][other] = _float(row.get(other)) or 0.0
        title = f"Benchmark selected-peptide Jaccard overlap, initial labels={initial}"
        figure_path = figure_root / f"benchmark_pairwise_jaccard_heatmap_initial_{initial}.svg"
        _write_svg_jaccard_heatmap(matrix, strategies, title, figure_path)
        outputs[figure_path.name] = str(figure_path)
        presentation_path = figure_root / f"benchmark_pairwise_jaccard_heatmap_initial_{initial}_presentation.svg"
        _write_svg_jaccard_heatmap(matrix, strategies, title, presentation_path)
        outputs[presentation_path.name] = str(presentation_path)
        original_path = overlap_root / f"pairwise_jaccard_heatmap_initial_{initial}.svg"
        _write_svg_jaccard_heatmap(matrix, strategies, title, original_path)
    return outputs


def _write_svg_jaccard_heatmap(matrix: dict[str, dict[str, float]], strategies: list[str], title: str, path: Path) -> None:
    if not strategies:
        path.write_text(_empty_svg(title), encoding="utf-8")
        return
    column_labels = {
        "random": "Rand",
        "predictive_entropy": "PE",
        "family_qbc": "FQBC",
        "cluster_diverse_representative": "CDR",
        "ensemble_mi": "EMI",
        "hybrid_mi_diverse": "HMI+D",
        "oed_logdet": "OED",
        "ensemble_mean": "Mean",
        "similarity_penalized_mean": "SimPen",
        "ucb": "UCB",
    }
    legend_items = [
        ("PE", "predictive entropy"),
        ("FQBC", "family QBC"),
        ("CDR", "cluster-diverse"),
        ("EMI", "ensemble MI"),
        ("HMI+D", "hybrid MI + diverse"),
        ("OED", "OED logdet"),
        ("SimPen", "similarity-penalized mean"),
    ]
    cell = 72
    left, top = 268, 142
    right, bottom = 210, 132
    width = left + cell * len(strategies) + right
    height = top + cell * len(strategies) + bottom

    def color(value: float) -> str:
        value = max(0.0, min(1.0, value))
        # White -> pale blue -> deep blue, readable in grayscale printouts too.
        if value < 0.5:
            ratio = value / 0.5
            start = (248, 250, 252)
            end = (147, 197, 253)
        else:
            ratio = (value - 0.5) / 0.5
            start = (147, 197, 253)
            end = (29, 78, 216)
        rgb = tuple(round(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    title_x = width / 2
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{title_x:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="{title_x:.1f}" y="62" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#334155">Darker cells indicate more redundant selected sets; values are mean Jaccard across replay runs.</text>',
        f'<text x="{title_x:.1f}" y="84" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#64748b">Columns use abbreviations; rows show full strategy names.</text>',
    ]
    for index, strategy in enumerate(strategies):
        x = left + index * cell + cell / 2
        label = column_labels.get(strategy, _display_strategy(strategy))
        y = top - 16
        body.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#0f172a">'
            f'{html.escape(label)}</text>'
        )
    for row_index, strategy in enumerate(strategies):
        y = top + row_index * cell + cell / 2
        body.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-size="13" fill="#0f172a">{html.escape(_display_strategy(strategy))}</text>')
        for col_index, other in enumerate(strategies):
            value = matrix.get(strategy, {}).get(other, 0.0)
            x = left + col_index * cell
            y0 = top + row_index * cell
            fill = color(value)
            text_color = "#ffffff" if value >= 0.55 else "#0f172a"
            body.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="1"/>')
            body.append(f'<text x="{x + cell/2:.1f}" y="{y0 + cell/2 + 4:.1f}" text-anchor="middle" font-size="12" fill="{text_color}">{value:.2f}</text>')
    # Method-family separators after baseline, uncertainty/QBC, and diversity blocks.
    separator_after = ["random", "family_qbc", "oed_logdet"]
    for name in separator_after:
        if name in strategies:
            idx = strategies.index(name) + 1
            if idx < len(strategies):
                x = left + idx * cell
                y = top + idx * cell
                body.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + len(strategies)*cell:.1f}" stroke="#0f172a" stroke-width="1.2" opacity="0.35"/>')
                body.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + len(strategies)*cell:.1f}" y2="{y:.1f}" stroke="#0f172a" stroke-width="1.2" opacity="0.35"/>')
    bar_x = left + len(strategies) * cell + 42
    bar_y = top
    bar_h = cell * len(strategies)
    steps = 20
    for i in range(steps):
        value = 1.0 - i / (steps - 1)
        y = bar_y + i * bar_h / steps
        body.append(f'<rect x="{bar_x}" y="{y:.1f}" width="18" height="{bar_h/steps + 0.5:.1f}" fill="{color(value)}"/>')
    body.append(f'<text x="{bar_x+28}" y="{bar_y+4}" font-size="12" fill="#334155">1.0</text>')
    body.append(f'<text x="{bar_x+28}" y="{bar_y+bar_h:.1f}" font-size="12" fill="#334155">0.0</text>')
    body.append(f'<text x="{bar_x+9}" y="{bar_y+bar_h+30:.1f}" text-anchor="middle" font-size="13" fill="#334155">Jaccard</text>')
    legend_y = top + len(strategies) * cell + 42
    legend_x = left
    body.append(
        f'<text x="{legend_x}" y="{legend_y}" font-size="12" fill="#475569">'
        f'Abbrev.: {html.escape("; ".join(f"{abbr} = {name}" for abbr, name in legend_items[:4]))}</text>'
    )
    body.append(
        f'<text x="{legend_x}" y="{legend_y + 20}" font-size="12" fill="#475569">'
        f'{html.escape("; ".join(f"{abbr} = {name}" for abbr, name in legend_items[4:]))}</text>'
    )
    body.append("</svg>\n")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_phase2_curated_figure_index(input_root: Path, figure_root: Path) -> None:
    curated = figure_root / "thesis_main"
    curated.mkdir(parents=True, exist_ok=True)
    chosen = [
        "benchmark_holdout_mean_AULC_F1_initial_40.svg",
        "benchmark_holdout_labels_saved_vs_random_to_f1_086_initial_10.svg",
        "benchmark_holdout_labels_saved_vs_random_to_f1_086_initial_40.svg",
        "benchmark_pairwise_jaccard_heatmap_initial_40.svg",
        "benchmark_holdout_f1_initial_40_vs_labeled_peptides.svg",
        "benchmark_validation_mean_AULC_F1_initial_40.svg",
    ]
    copied: list[str] = []
    for name in chosen:
        source = figure_root / name
        if not source.exists():
            continue
        target = curated / name
        target.write_bytes(source.read_bytes())
        copied.append(name)
    lines = [
        "# Curated Phase 2 Thesis Figures",
        "",
        "This folder contains a smaller, thesis-oriented subset of the Phase 2 figure exports.",
        "Use holdout figures in the main text where possible; validation figures are included for audit/appendix support.",
        "",
    ]
    for name in copied:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Recommended main-text order:",
            "1. Holdout AULC-F1 ranked summary.",
            "2. Labels saved versus random at F1 = 0.86.",
            "3. Selected-peptide overlap heatmap.",
            "4. Holdout learning curve for the 40-label starting point.",
            "",
        ]
    )
    (curated / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="18" font-weight="600" fill="#0f172a">{html.escape(title)}</text>'
    )


def _empty_svg(title: str) -> str:
    return _svg_header(800, 220, title) + '\n<text x="400" y="120" text-anchor="middle" font-size="14" fill="#64748b">No data available</text>\n</svg>\n'


def _svg_palette() -> list[str]:
    return [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#65a30d",
        "#be123c",
        "#0f766e",
        "#7c3aed",
        "#ca8a04",
    ]
