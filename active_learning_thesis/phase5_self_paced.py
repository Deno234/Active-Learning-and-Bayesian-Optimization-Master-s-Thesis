from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import hashlib
import json
import math
import os
import posixpath
from pathlib import Path
import shutil
import statistics
import time
from typing import Iterable, Sequence

import numpy as np

from active_learning_thesis.acquisition import select_batch
from active_learning_thesis.config import RunConfig
from active_learning_thesis.dataset import read_experimental_dataset
from active_learning_thesis.ledger import serialize_probabilities
from active_learning_thesis.phase2_replay import (
    BENCHMARK_STRATEGIES,
    ReplayRunSpec,
    TARGET_F1_VALUES,
    calibration_metric_rows,
    config_for_phase2,
    construct_replay_rows,
    deterministic_run_seed,
    evaluate_with_fixed_threshold,
    evaluate_with_validation_threshold,
    labels_to_target,
    load_frozen_model_config,
    load_replay_manifest,
    resource_logger,
    validate_no_within_repeat_overlap,
)
from active_learning_thesis.predictive import (
    extract_ap_sp_member_embeddings_strict,
    score_sequences_with_ensemble,
    train_ensemble,
)


DEFAULT_OUTPUT_ROOT = Path("thesis_results") / "05_self_paced_active_learning"
DEFAULT_PHASE1_ROOT = Path("thesis_results") / "01_reproduction"
DEFAULT_PHASE2_ROOT = Path("thesis_results") / "02_replay"
PHASE5_STRATEGIES = (
    "random",
    "predictive_entropy",
    "static_easy_entropy",
    "self_paced_entropy",
)
PACE_LAMBDA_0 = 0.30
PACE_MAX_ROUNDS = 45
PRIMARY_OUTER_FOLDS = (1, 2, 3)
PRIMARY_INITIAL_LABEL_COUNTS = (10,)
PREREGISTERED_AULC_ENDPOINTS = (60, 110, 160, 235)
FIXED_BUDGET_COUNTS = (60, 110, 160)
EMBEDDING_WIDTH = 384
EMBEDDING_NORM_EPSILON = 1e-12
POST_HOC_PROBABILITY_EPSILON = 1e-6
PBS_SAFETY_MULTIPLIER = 2.0
PBS_ADDITIONAL_SECONDS = 30 * 60
PBS_MINIMUM_HOURS = 2
TERMINAL_NUMERICAL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Phase5Options:
    action: str
    phase1_root: Path = DEFAULT_PHASE1_ROOT
    phase2_root: Path = DEFAULT_PHASE2_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    pbs_repo_root: Path | None = None
    outer_folds: tuple[int, ...] = PRIMARY_OUTER_FOLDS
    inner_fold: int = 1
    initial_label_counts: tuple[int, ...] = PRIMARY_INITIAL_LABEL_COUNTS
    strategies: tuple[str, ...] = PHASE5_STRATEGIES
    outer_fold: int | None = None
    initial_label_count: int | None = None
    strategy: str | None = None
    batch_size: int = 5
    max_rounds: int = PACE_MAX_ROUNDS
    ensemble_size: int = 1
    base_seed: int = 20260317
    force: bool = False
    supek_walltime: str | None = None
    supek_queue: str = "gpu"
    supek_ncpus: int = 4
    supek_ngpus: int = 1
    supek_mem: str = "40GB"


def options_from_args(args) -> Phase5Options:
    return Phase5Options(
        action=str(args.phase5_action),
        phase1_root=Path(args.phase1_root),
        phase2_root=Path(args.phase2_root),
        output_root=Path(args.output_root),
        pbs_repo_root=Path(args.pbs_repo_root) if getattr(args, "pbs_repo_root", None) else None,
        outer_folds=tuple(int(value) for value in getattr(args, "outer_folds", list(PRIMARY_OUTER_FOLDS))),
        inner_fold=int(getattr(args, "inner_fold", 1)),
        initial_label_counts=tuple(
            int(value) for value in getattr(args, "initial_label_counts", list(PRIMARY_INITIAL_LABEL_COUNTS))
        ),
        strategies=tuple(getattr(args, "strategies", list(PHASE5_STRATEGIES))),
        outer_fold=getattr(args, "outer_fold", None),
        initial_label_count=getattr(args, "initial_label_count", None),
        strategy=getattr(args, "strategy", None),
        batch_size=int(getattr(args, "batch_size", 5)),
        max_rounds=int(getattr(args, "max_rounds", PACE_MAX_ROUNDS)),
        ensemble_size=int(getattr(args, "ensemble_size", 1)),
        base_seed=int(getattr(args, "base_seed", 20260317)),
        force=bool(getattr(args, "force", False)),
        supek_walltime=getattr(args, "supek_walltime", None),
        supek_queue=str(getattr(args, "supek_queue", "gpu")),
        supek_ncpus=int(getattr(args, "supek_ncpus", 4)),
        supek_ngpus=int(getattr(args, "supek_ngpus", 1)),
        supek_mem=str(getattr(args, "supek_mem", "40GB")),
    )


def run_phase5(args_or_options) -> dict[str, object]:
    options = (
        args_or_options
        if isinstance(args_or_options, Phase5Options)
        else options_from_args(args_or_options)
    )
    if options.action == "init":
        return initialize_phase5(options)
    if options.action == "run-job":
        return run_phase5_job(options)
    if options.action == "aggregate":
        return aggregate_phase5(options.output_root)
    if options.action == "status":
        return phase5_status(options.output_root)
    raise ValueError(f"Unsupported Phase 5 action: {options.action}")


def pace_lambda(acquisition_step: int, max_rounds: int = PACE_MAX_ROUNDS) -> float:
    if acquisition_step < 0 or acquisition_step >= max_rounds:
        raise ValueError(
            f"acquisition_step must be in [0, {max_rounds - 1}], got {acquisition_step}"
        )
    if max_rounds <= 1:
        return 1.0
    value = PACE_LAMBDA_0 + (1.0 - PACE_LAMBDA_0) * acquisition_step / (max_rounds - 1)
    return float(np.clip(value, PACE_LAMBDA_0, 1.0))


def stable_difficulty_percentiles(distances: Sequence[float]) -> np.ndarray:
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1:
        raise ValueError("Difficulty distances must be a one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError("Difficulty distances contain non-finite values")
    count = len(values)
    if count <= 1:
        return np.zeros(count, dtype=float)
    order = np.argsort(values, kind="mergesort")
    positions = np.empty(count, dtype=int)
    positions[order] = np.arange(count)
    return positions.astype(float) / float(count - 1)


def operational_difficulty_quintile(percentile: float) -> int:
    return min(5, int(math.floor(5.0 * float(percentile))) + 1)


def l2_normalize_embeddings(
    embeddings: np.ndarray,
    epsilon: float = EMBEDDING_NORM_EPSILON,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=float)
    if values.ndim != 2:
        raise ValueError("Embeddings must be rank two")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, float(epsilon))


def memberwise_familiarity_distances(
    candidate_member_embeddings: Sequence[np.ndarray],
    labelled_member_embeddings: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if len(candidate_member_embeddings) != len(labelled_member_embeddings):
        raise ValueError("Candidate and labelled member embedding counts differ")
    if not candidate_member_embeddings:
        raise ValueError("At least one member embedding is required")
    columns: list[np.ndarray] = []
    candidate_count: int | None = None
    for member_index, (candidate, labelled) in enumerate(
        zip(candidate_member_embeddings, labelled_member_embeddings)
    ):
        candidate_values = l2_normalize_embeddings(candidate)
        labelled_values = l2_normalize_embeddings(labelled)
        if labelled_values.shape[0] == 0:
            raise ValueError("The labelled familiarity reference set cannot be empty")
        if candidate_values.shape[1] != labelled_values.shape[1]:
            raise ValueError(f"Member {member_index} candidate/labelled widths differ")
        if candidate_count is None:
            candidate_count = candidate_values.shape[0]
        elif candidate_values.shape[0] != candidate_count:
            raise ValueError("Candidate embedding row counts differ between members")
        pairwise = np.linalg.norm(
            candidate_values[:, None, :] - labelled_values[None, :, :],
            axis=2,
        )
        columns.append(pairwise.min(axis=1))
    member_distances = np.column_stack(columns)
    return member_distances, member_distances.mean(axis=1)


def model_seed_schedule(
    base_seed: int,
    outer_fold: int,
    initial_label_count: int,
    replay_point: int,
    ensemble_size: int = 1,
) -> list[int]:
    anchor = (
        int(base_seed)
        + int(outer_fold) * 100000
        + int(initial_label_count) * 1000
        + int(replay_point) * 10
    )
    return [anchor + member_index for member_index in range(ensemble_size)]


def canonical_dataset_row_ids() -> dict[str, int]:
    rows = read_experimental_dataset()
    mapping = {row["sequence"]: index for index, row in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Canonical experimental dataset contains duplicate sequences")
    return mapping


def attach_canonical_row_ids(rows, mapping: dict[str, int]) -> None:
    for split_name in ("holdout", "validation", "train_pool", "replay_seed", "replay_hidden"):
        for row in getattr(rows, split_name):
            sequence = row["sequence"]
            if sequence not in mapping:
                raise ValueError(
                    f"Replay sequence {sequence} is missing from the canonical experimental dataset"
                )
            row["original_dataset_row_id"] = str(mapping[sequence])


def canonical_training_order(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: int(row["original_dataset_row_id"]))
    if len({row["original_dataset_row_id"] for row in ordered}) != len(ordered):
        raise ValueError("Canonical training rows contain duplicate original dataset row identifiers")
    return [dict(row) for row in ordered]


def ordered_row_id_checksum(rows: Sequence[dict[str, str]]) -> str:
    payload = ",".join(row["original_dataset_row_id"] for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enable_phase5_determinism() -> dict[str, object]:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    enabled = False
    reason = ""
    try:
        import tensorflow as tf

        enable = getattr(tf.config.experimental, "enable_op_determinism", None)
        if callable(enable):
            enable()
            enabled = True
        else:
            reason = "tensorflow_enable_op_determinism_unavailable"
    except Exception as exc:
        reason = f"tensorflow_determinism_setup_failed:{type(exc).__name__}"
    return {
        "tf_deterministic_ops_environment": os.environ.get("TF_DETERMINISTIC_OPS", ""),
        "tf_cudnn_deterministic_environment": os.environ.get("TF_CUDNN_DETERMINISTIC", ""),
        "tensorflow_enable_op_determinism_called": enabled,
        "fallback_reason": reason,
    }


def select_phase5_batch(
    strategy: str,
    batch_size: int,
    candidate_scores: dict[str, np.ndarray],
    distances: np.ndarray,
    percentiles: np.ndarray,
    acquisition_step: int,
    max_rounds: int,
    config: RunConfig,
    seed: int,
) -> tuple[list[int], np.ndarray, np.ndarray, bool, bool, float]:
    candidate_count = len(np.asarray(candidate_scores["pred_mean"]))
    if strategy not in PHASE5_STRATEGIES:
        raise ValueError(f"Unsupported Phase 5 strategy: {strategy}")
    if candidate_count != len(distances) or candidate_count != len(percentiles):
        raise ValueError("Candidate score and familiarity row counts differ")

    if strategy == "random":
        selected, scores = select_batch(
            "random",
            batch_size,
            candidate_scores,
            np.empty((0, 0), dtype=float),
            config,
            seed,
        )
        return selected, scores, np.ones(candidate_count, dtype=bool), False, False, 1.0

    if strategy == "predictive_entropy":
        selected, scores = select_batch(
            "predictive_entropy",
            batch_size,
            candidate_scores,
            np.empty((0, 0), dtype=float),
            config,
            seed,
        )
        return selected, scores, np.ones(candidate_count, dtype=bool), False, False, 1.0

    selected_lambda = (
        PACE_LAMBDA_0
        if strategy == "static_easy_entropy"
        else pace_lambda(acquisition_step, max_rounds)
    )
    eligible = np.asarray(percentiles <= selected_lambda, dtype=bool)
    selection_eligible = eligible.copy()
    fallback = int(np.sum(eligible)) < min(batch_size, candidate_count)
    if fallback:
        fallback_order = np.argsort(percentiles, kind="mergesort")
        selection_eligible[fallback_order[: min(batch_size, candidate_count)]] = True
    eligible_indices = np.flatnonzero(selection_eligible)
    entropy = np.asarray(candidate_scores["pred_entropy"], dtype=float)
    local_order = np.argsort(-entropy[eligible_indices], kind="mergesort")
    selected = list(eligible_indices[local_order[: min(batch_size, len(eligible_indices))]])
    return selected, entropy, eligible, True, fallback, float(selected_lambda)


def initialize_phase5(options: Phase5Options) -> dict[str, object]:
    _validate_options(options)
    load_frozen_model_config(options.phase1_root)
    row_mapping = canonical_dataset_row_ids()
    train_pool_sizes = {}
    for outer_fold in options.outer_folds:
        replay_manifest = load_replay_manifest(options.phase1_root, outer_fold, options.inner_fold)
        train_pool = [
            row for row in replay_manifest.get("rows", [])
            if isinstance(row, dict) and row.get("split") == "train_pool"
        ]
        train_pool_sizes[str(outer_fold)] = len(train_pool)
        missing = [row.get("sequence", "") for row in train_pool if row.get("sequence") not in row_mapping]
        if missing:
            raise ValueError(
                f"Outer fold {outer_fold} contains sequences absent from the canonical dataset: "
                f"{missing[:5]}"
            )
        for initial in options.initial_label_counts:
            expected_steps = math.ceil((len(train_pool) - initial) / options.batch_size)
            if options.max_rounds != expected_steps:
                raise ValueError(
                    f"Outer fold {outer_fold} with initial_label_count={initial} requires "
                    f"max_rounds={expected_steps} to reveal all {len(train_pool)} training rows; "
                    f"received {options.max_rounds}"
                )
    if options.output_root.exists() and options.force:
        shutil.rmtree(options.output_root)
    options.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("config", "manifests", "replay", "tables", "figures", "audits", "pbs", "logs"):
        (options.output_root / name).mkdir(parents=True, exist_ok=True)

    timing = derive_phase5_walltime(options.phase2_root, options.supek_walltime)
    manifest = {
        "status": "implemented_but_not_yet_executed",
        "experiment_name": "SPAL-inspired self-paced active-learning replay",
        "created_at": _now_iso(),
        "phase1_root": options.phase1_root.as_posix(),
        "phase2_root": options.phase2_root.as_posix(),
        "output_root": options.output_root.as_posix(),
        "outer_folds": list(options.outer_folds),
        "outer_fold_interpretation": (
            f"{len(options.outer_folds)} fold-level repetitions with overlapping training partitions; "
            "not statistically independent replicates."
        ),
        "inner_fold": options.inner_fold,
        "initial_label_counts": list(options.initial_label_counts),
        "strategies": list(options.strategies),
        "batch_size": options.batch_size,
        "maximum_acquisition_steps": options.max_rounds,
        "terminal_replay_points": options.max_rounds + 1,
        "train_pool_sizes": train_pool_sizes,
        "primary_model_protocol": "one calibrated AP_SP model per replay point",
        "ensemble_size": options.ensemble_size,
        "pace_lambda_0": PACE_LAMBDA_0,
        "embedding_contract": {
            "model": "AP_SP",
            "semantic_layer": "penultimate Concatenate",
            "runtime_shape": ["batch_size", EMBEDDING_WIDTH],
            "member_normalization_epsilon": EMBEDDING_NORM_EPSILON,
            "aggregation": (
                "single-member nearest-labelled distance"
                if options.ensemble_size == 1
                else "mean of member-wise nearest-labelled scalar distances"
            ),
        },
        "labelled_reference_set": (
            "Currently revealed rows from the 235-row replay training pool only."
        ),
        "hidden_percentile_population": (
            "Current hidden rows from the replay training pool only."
        ),
        "post_hoc_probability_clipping_epsilon": POST_HOC_PROBABILITY_EPSILON,
        "pace_schedule": (
            f"clip(0.30 + 0.70 * acquisition_step / {options.max_rounds - 1}, "
            "0.30, 1.0)"
        ),
        "preregistered_normalized_aulc_intervals": [
            [options.initial_label_counts[0], endpoint]
            for endpoint in PREREGISTERED_AULC_ENDPOINTS
        ],
        "fixed_budget_counts": list(FIXED_BUDGET_COUNTS),
        "difficulty_percentile": (
            "zero-based stable ascending position divided by hidden_pool_size-1; "
            "zero when hidden_pool_size <= 1"
        ),
        "tie_breaking": {
            "difficulty": "stable candidate input order",
            "predictive_entropy": "stable candidate input order",
        },
        "eligibility_semantics": {
            "random": "not used; exported eligible=true",
            "predictive_entropy": "not used; exported eligible=true",
            "static_easy_entropy": "difficulty_percentile <= 0.30",
            "self_paced_entropy": "difficulty_percentile <= pace_lambda",
        },
        "model_seed_rule": (
            "base + outer_fold*100000 + initial_label_count*1000 + "
            "replay_point*10 + model_member_index"
        ),
        "canonical_training_order": {
            "source": "SA_ML_predictive/data/data_SA.csv original valid-row order",
            "identifier": "zero-based original_dataset_row_id",
            "canonical_dataset_sequence_count": len(row_mapping),
        },
        "terminal_numerical_tolerance": TERMINAL_NUMERICAL_TOLERANCE,
        "pbs_timing": timing,
        "claims": {
            "exact_spal_reproduction": False,
            "prospective_experiment": False,
            "md_workflow": False,
            "chemical_complexity_primary_selector": False,
        },
        "source_checksums": _source_checksums(options),
    }
    _write_json(options.output_root / "manifests" / "phase5_manifest.json", manifest)
    _write_json(
        options.output_root / "config" / "phase5_config.json",
        {
            **manifest,
            "supek_queue": options.supek_queue,
            "supek_ncpus": options.supek_ncpus,
            "supek_ngpus": options.supek_ngpus,
            "supek_mem": options.supek_mem,
        },
    )
    paths = write_phase5_pbs(options, timing["requested_walltime"])
    return {
        "status": "initialized",
        "output_root": str(options.output_root),
        "job_count": len(options.outer_folds)
        * len(options.initial_label_counts)
        * len(options.strategies),
        "pbs_walltime": timing["requested_walltime"],
        "pbs_outputs": [str(path) for path in paths],
        "jobs_submitted": False,
    }


def run_phase5_job(options: Phase5Options) -> dict[str, object]:
    _validate_options(options)
    if options.outer_fold is None or options.initial_label_count is None or options.strategy is None:
        raise ValueError("run-job requires --outer-fold, --initial-label-count, and --strategy")
    if options.strategy not in PHASE5_STRATEGIES:
        raise ValueError(f"Unsupported Phase 5 strategy: {options.strategy}")
    run_dir = _job_dir(
        options.output_root,
        int(options.outer_fold),
        int(options.initial_label_count),
        str(options.strategy),
    )
    status_path = run_dir / "status.json"
    if status_path.exists() and not options.force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            return {"status": "reused-existing", "run_dir": str(run_dir)}
    if run_dir.exists() and options.force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with resource_logger(run_dir, "phase5-self-paced-replay", run_dir.name):
        started = time.time()
        manifest = load_replay_manifest(
            options.phase1_root,
            int(options.outer_fold),
            options.inner_fold,
        )
        run_seed = deterministic_run_seed(
            options.base_seed,
            int(options.outer_fold),
            options.inner_fold,
            int(options.initial_label_count),
        )
        rows = construct_replay_rows(manifest, int(options.initial_label_count), run_seed)
        attach_canonical_row_ids(rows, canonical_dataset_row_ids())
        validate_no_within_repeat_overlap(rows)
        expected_steps = math.ceil(
            (len(rows.train_pool) - int(options.initial_label_count)) / options.batch_size
        )
        if options.max_rounds != expected_steps:
            raise ValueError(
                f"This replay requires {expected_steps} acquisition steps to reveal all "
                f"{len(rows.train_pool)} training rows; received {options.max_rounds}"
            )
        frozen_config = load_frozen_model_config(options.phase1_root)
        setup_name = (
            "single_calibrated" if options.ensemble_size == 1 else "ensemble_calibrated"
        )
        spec = ReplayRunSpec(
            mode="phase5",
            setup=setup_name,
            outer_fold_id=int(options.outer_fold),
            inner_fold_id=options.inner_fold,
            replay_seed_size=int(options.initial_label_count),
            batch_size=options.batch_size,
            max_rounds=options.max_rounds,
            strategies=(str(options.strategy),),
            base_seed=options.base_seed,
            run_seed=run_seed,
            ensemble_size=options.ensemble_size,
            use_calibrated_acquisition=True,
            run_dir=run_dir,
        )
        base_config = config_for_phase2(spec, frozen_config)
        _write_job_inputs(run_dir, options, spec, rows, base_config)
        result = _run_strategy(options, spec, base_config, rows)
        _write_csv(run_dir / "round_metrics.csv", result["round_metrics"])
        _write_csv(run_dir / "selected_sequences.csv", result["selected_sequences"])
        _write_csv(run_dir / "candidate_scoring.csv", result["candidate_scoring"])
        _write_csv(run_dir / "proxy_validity_records.csv", result["proxy_records"])
        _write_csv(run_dir / "proxy_validity_summary.csv", result["proxy_summary"])
        _write_csv(run_dir / "calibration_summary.csv", result["calibration_summary"])
        _write_json(run_dir / "terminal_state.json", result["terminal_state"])
        _write_json(
            status_path,
            {
                "status": "completed",
                "outer_fold": options.outer_fold,
                "initial_label_count": options.initial_label_count,
                "strategy": options.strategy,
                "replay_points": options.max_rounds + 1,
                "acquisition_steps": options.max_rounds,
                "terminal_labelled_count": result["terminal_state"]["labelled_count"],
                "terminal_hidden_count": result["terminal_state"]["hidden_count"],
                "elapsed_seconds": time.time() - started,
                "completed_at": _now_iso(),
            },
        )
    return {"status": "completed", "run_dir": str(run_dir)}


def _run_strategy(options, spec, base_config, rows) -> dict[str, object]:
    current_labeled = [dict(row) for row in rows.replay_seed]
    hidden_pool = {row["sequence"]: dict(row) for row in rows.replay_hidden}
    round_metrics: list[dict[str, object]] = []
    selected_sequences: list[dict[str, object]] = []
    candidate_scoring: list[dict[str, object]] = []
    proxy_records: list[dict[str, object]] = []
    proxy_summary: list[dict[str, object]] = []
    calibration_summary: list[dict[str, object]] = []
    terminal_state: dict[str, object] = {}
    ordered_validation_rows = canonical_training_order(rows.validation)
    ordered_holdout_rows = canonical_training_order(rows.holdout)

    for replay_point in range(spec.max_rounds + 1):
        member_seeds = model_seed_schedule(
            options.base_seed,
            spec.outer_fold_id,
            spec.initial_label_count,
            replay_point,
            spec.ensemble_size,
        )
        config = replace(
            base_config,
            random_seed=member_seeds[0],
            ensemble_seeds=member_seeds,
            replay_strategies=[spec.strategies[0]],
        )
        deterministic_settings = enable_phase5_determinism()
        ordered_training_rows = canonical_training_order(current_labeled)
        round_dir = spec.run_dir / "models" / f"replay_point_{replay_point:03d}"
        ensemble = train_ensemble(
            ordered_training_rows,
            ordered_validation_rows,
            round_dir / "ensemble",
            config,
            cache_dir=spec.run_dir / "model_cache",
        )
        validation_metrics, threshold = evaluate_with_validation_threshold(
            ensemble,
            ordered_validation_rows,
            use_calibration=True,
        )
        holdout_metrics = evaluate_with_fixed_threshold(
            ensemble,
            ordered_holdout_rows,
            threshold=threshold,
            use_calibration=True,
        )
        round_metrics.append(
            _metric_row(spec, replay_point, len(current_labeled), "validation", validation_metrics)
        )
        round_metrics.append(
            _metric_row(spec, replay_point, len(current_labeled), "holdout", holdout_metrics)
        )
        calibration_summary.extend(
            calibration_metric_rows(
                spec,
                spec.strategies[0],
                replay_point,
                ensemble,
                ordered_validation_rows,
            )
        )
        if replay_point == spec.max_rounds or not hidden_pool:
            terminal_scores = score_sequences_with_ensemble(
                ensemble,
                [row["sequence"] for row in ordered_holdout_rows],
                use_calibration=True,
                include_raw=True,
            )
            terminal_parameters = np.concatenate(
                [
                    np.asarray(weight, dtype=np.float64).reshape(-1)
                    for weight in ensemble[0].model.get_weights()
                ]
            )
            parameter_path = spec.run_dir / "terminal_parameters.npy"
            np.save(parameter_path, terminal_parameters)
            terminal_state = {
                "outer_fold": spec.outer_fold_id,
                "initial_label_count": spec.initial_label_count,
                "strategy": spec.strategies[0],
                "replay_point": replay_point,
                "labelled_count": len(current_labeled),
                "hidden_count": len(hidden_pool),
                "ordered_row_ids": [
                    int(row["original_dataset_row_id"]) for row in ordered_training_rows
                ],
                "ordered_row_ids_checksum_sha256": ordered_row_id_checksum(
                    ordered_training_rows
                ),
                "ordered_validation_row_ids_checksum_sha256": ordered_row_id_checksum(
                    ordered_validation_rows
                ),
                "model_seed": member_seeds[0],
                "parameter_count": int(len(terminal_parameters)),
                "parameter_vector_path": str(parameter_path),
                "parameter_vector_sha256": hashlib.sha256(
                    terminal_parameters.tobytes()
                ).hexdigest(),
                "holdout_predictions": [
                    float(value) for value in terminal_scores["pred_mean"]
                ],
                "holdout_prediction_checksum_sha256": hashlib.sha256(
                    np.asarray(terminal_scores["pred_mean"], dtype=np.float64).tobytes()
                ).hexdigest(),
                "deterministic_settings": deterministic_settings,
                "numerical_tolerance": TERMINAL_NUMERICAL_TOLERANCE,
            }
            _cleanup_tensorflow_runtime()
            break

        acquisition_step = replay_point
        candidate_rows = list(hidden_pool.values())
        candidate_sequences = [row["sequence"] for row in candidate_rows]
        selection_seed = spec.run_seed + acquisition_step

        # Random selection is completed before predictions or familiarity diagnostics
        # are attached to any candidate record.
        random_selected: list[int] | None = None
        if spec.strategies[0] == "random":
            placeholder = {
                "pred_mean": np.zeros(len(candidate_sequences), dtype=float),
            }
            random_selected, _ = select_batch(
                "random",
                spec.batch_size,
                placeholder,
                np.empty((0, 0), dtype=float),
                config,
                selection_seed,
            )

        scores = score_sequences_with_ensemble(
            ensemble,
            candidate_sequences,
            use_calibration=True,
            include_raw=True,
        )
        combined_sequences = candidate_sequences + [row["sequence"] for row in current_labeled]
        combined_embeddings, embedding_metadata = extract_ap_sp_member_embeddings_strict(
            ensemble,
            combined_sequences,
            expected_width=EMBEDDING_WIDTH,
        )
        candidate_count = len(candidate_sequences)
        candidate_embeddings = [values[:candidate_count] for values in combined_embeddings]
        labelled_embeddings = [values[candidate_count:] for values in combined_embeddings]
        member_distances, mean_distances = memberwise_familiarity_distances(
            candidate_embeddings,
            labelled_embeddings,
        )
        percentiles = stable_difficulty_percentiles(mean_distances)
        if random_selected is not None:
            selected_indices = random_selected
            acquisition_scores = np.zeros(candidate_count, dtype=float)
            eligible = np.ones(candidate_count, dtype=bool)
            eligibility_used = False
            fallback = False
            selected_lambda = 1.0
        else:
            (
                selected_indices,
                acquisition_scores,
                eligible,
                eligibility_used,
                fallback,
                selected_lambda,
            ) = select_phase5_batch(
                spec.strategies[0],
                spec.batch_size,
                scores,
                mean_distances,
                percentiles,
                acquisition_step,
                spec.max_rounds,
                config,
                selection_seed,
            )
        selected_set = set(selected_indices)
        selection_rank = {index: rank for rank, index in enumerate(selected_indices, start=1)}

        _write_json(
            round_dir / "embedding_manifest.json",
            {
                "outer_fold": spec.outer_fold_id,
                "initial_label_count": spec.initial_label_count,
                "strategy": spec.strategies[0],
                "replay_point": replay_point,
                "acquisition_step": acquisition_step,
                "member_seeds": member_seeds,
                "primary_single_model": spec.ensemble_size == 1,
                "selection_seed": selection_seed,
                "ordered_training_row_ids": [
                    int(row["original_dataset_row_id"]) for row in ordered_training_rows
                ],
                "ordered_training_row_ids_checksum_sha256": ordered_row_id_checksum(
                    ordered_training_rows
                ),
                "deterministic_settings": deterministic_settings,
                "labelled_reference_count": len(current_labeled),
                "candidate_count": candidate_count,
                "members": embedding_metadata,
            },
        )

        label_free_rows: list[dict[str, object]] = []
        for index, sequence in enumerate(candidate_sequences):
            payload = {
                "sequence": sequence,
                "outer_fold": spec.outer_fold_id,
                "initial_label_count": spec.initial_label_count,
                "strategy": spec.strategies[0],
                "replay_point": replay_point,
                "acquisition_step": acquisition_step,
                "labelled_count": len(current_labeled),
                "selection_seed": selection_seed,
                "pred_mean": float(scores["pred_mean"][index]),
                "pred_std": (
                    "" if spec.ensemble_size == 1 else float(scores["pred_std"][index])
                ),
                "predictive_entropy": float(scores["pred_entropy"][index]),
                "ensemble_mi": (
                    ""
                    if spec.ensemble_size == 1
                    else float(scores["pred_mutual_information"][index])
                ),
                "ensemble_member_probs": serialize_probabilities(
                    scores["ensemble_member_probs"][index]
                ),
                "embedding_distance_to_labelled": float(mean_distances[index]),
                "difficulty_percentile": float(percentiles[index]),
                "difficulty_quintile": operational_difficulty_quintile(percentiles[index]),
                "pace_lambda": selected_lambda,
                "eligibility_used_for_selection": eligibility_used,
                "eligible": bool(eligible[index]),
                "eligibility_fallback": fallback,
                "acquisition_score": float(acquisition_scores[index]),
                "selection_rank": selection_rank.get(index, ""),
                "selected": index in selected_set,
            }
            for member_index in range(member_distances.shape[1]):
                payload[f"embedding_distance_member_{member_index}"] = float(
                    member_distances[index, member_index]
                )
            label_free_rows.append(payload)
            candidate_scoring.append(payload)

        # Oracle labels enter only after selected_indices and label-free records exist.
        step_proxy_records = _join_post_hoc_labels(label_free_rows, candidate_rows)
        proxy_records.extend(step_proxy_records)
        proxy_summary.extend(_proxy_summary_rows(step_proxy_records))

        for rank, index in enumerate(selected_indices, start=1):
            selected_row = hidden_pool.pop(candidate_sequences[index])
            selected_sequences.append(
                {
                    "outer_fold": spec.outer_fold_id,
                    "initial_label_count": spec.initial_label_count,
                    "strategy": spec.strategies[0],
                    "batch_size": spec.batch_size,
                    "replay_point": replay_point,
                    "acquisition_step": acquisition_step,
                    "selection_rank": rank,
                    "sequence": selected_row["sequence"],
                    "label": selected_row["label"],
                    "label_revealed_after_selection": True,
                    "acquisition_score": float(acquisition_scores[index]),
                    "embedding_distance_to_labelled": float(mean_distances[index]),
                    "difficulty_percentile": float(percentiles[index]),
                }
            )
            current_labeled.append(selected_row)
        _cleanup_tensorflow_runtime()

    return {
        "round_metrics": round_metrics,
        "selected_sequences": selected_sequences,
        "candidate_scoring": candidate_scoring,
        "proxy_records": proxy_records,
        "proxy_summary": proxy_summary,
        "calibration_summary": calibration_summary,
        "terminal_state": terminal_state,
    }


def _join_post_hoc_labels(
    label_free_rows: list[dict[str, object]],
    oracle_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    labels = {row["sequence"]: int(row["label"]) for row in oracle_rows}
    output = []
    for row in label_free_rows:
        sequence = str(row["sequence"])
        if sequence not in labels:
            raise ValueError(f"Missing hidden oracle label for {sequence}")
        truth = labels[sequence]
        probability = float(row["pred_mean"])
        clipped = float(
            np.clip(
                probability,
                POST_HOC_PROBABILITY_EPSILON,
                1.0 - POST_HOC_PROBABILITY_EPSILON,
            )
        )
        payload = dict(row)
        payload.update(
            {
                "true_label": truth,
                "pre_query_pred_mean": probability,
                "pre_query_predictive_entropy": row["predictive_entropy"],
                "pre_query_ensemble_mi": row["ensemble_mi"],
                "post_hoc_pre_query_log_loss": float(
                    -(truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped))
                ),
                "post_hoc_pre_query_error_fixed_0_5": int(
                    int(probability >= 0.5) != truth
                ),
                "post_hoc_pre_query_absolute_probability_error": abs(truth - probability),
            }
        )
        output.append(payload)
    return output


def _proxy_summary_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not records:
        return []
    base = {
        "outer_fold": records[0]["outer_fold"],
        "initial_label_count": records[0]["initial_label_count"],
        "strategy": records[0]["strategy"],
        "acquisition_step": records[0]["acquisition_step"],
        "candidate_count": len(records),
    }
    distances = [float(row["embedding_distance_to_labelled"]) for row in records]
    log_losses = [float(row["post_hoc_pre_query_log_loss"]) for row in records]
    absolute_errors = [
        float(row["post_hoc_pre_query_absolute_probability_error"]) for row in records
    ]
    result = [
        {
            **base,
            "summary_type": "spearman",
            "metric": "post_hoc_pre_query_log_loss",
            "spearman": spearman_with_ties(distances, log_losses),
        },
        {
            **base,
            "summary_type": "spearman",
            "metric": "post_hoc_pre_query_absolute_probability_error",
            "spearman": spearman_with_ties(distances, absolute_errors),
        },
    ]
    for quintile in range(1, 6):
        group = [row for row in records if int(row["difficulty_quintile"]) == quintile]
        result.append(
            {
                **base,
                "summary_type": "difficulty_quintile",
                "metric": "candidate_diagnostics",
                "difficulty_quintile": quintile,
                "quintile_count": len(group),
                "error_rate": _mean(
                    [
                        float(row["post_hoc_pre_query_error_fixed_0_5"])
                        for row in group
                    ]
                ),
                "mean_predictive_entropy": _mean(
                    [float(row["pre_query_predictive_entropy"]) for row in group]
                ),
                "mean_ensemble_mi": _mean(
                    [
                        float(row["pre_query_ensemble_mi"])
                        for row in group
                        if row["pre_query_ensemble_mi"] not in ("", None)
                    ]
                ),
            }
        )
    return result


def average_tied_ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(len(data), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and data[order[end]] == data[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_with_ties(x: Sequence[float], y: Sequence[float]) -> float | str:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return ""
    left_ranks = average_tied_ranks(left)
    right_ranks = average_tied_ranks(right)
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _load_phase5_manifest(output_root: Path) -> dict[str, object]:
    path = output_root / "manifests" / "phase5_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 5 manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_phase5_jobs(manifest: dict[str, object]) -> int:
    return (
        len(manifest["outer_folds"])
        * len(manifest["initial_label_counts"])
        * len(manifest["strategies"])
    )


def _terminal_convergence_rows(
    output_root: Path,
    round_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    terminal_metrics = {
        (
            str(row.get("outer_fold_id", "")),
            str(row.get("initial_label_count", "")),
            str(row.get("strategy", "")),
        ): row
        for row in round_rows
        if row.get("evaluation_dataset") == "holdout"
        and int(float(row.get("labeled_count", 0) or 0)) == 235
    }
    states = {}
    for path in sorted(
        (output_root / "replay").glob("outer_*/initial_*/*/terminal_state.json")
    ):
        state = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(state["outer_fold"]),
            str(state["initial_label_count"]),
            str(state["strategy"]),
        )
        state["_run_dir"] = str(path.parent)
        states[key] = state

    result = []
    folds = sorted({(key[0], key[1]) for key in states})
    for outer, initial in folds:
        reference_key = (outer, initial, "random")
        if reference_key not in states:
            continue
        reference = states[reference_key]
        reference_parameters = np.load(_terminal_parameter_path(reference))
        reference_predictions = np.asarray(
            reference["holdout_predictions"], dtype=float
        )
        reference_metric = terminal_metrics.get(reference_key, {})
        tolerance = float(
            reference.get("numerical_tolerance", TERMINAL_NUMERICAL_TOLERANCE)
        )
        for strategy in PHASE5_STRATEGIES:
            key = (outer, initial, strategy)
            if key not in states:
                continue
            state = states[key]
            parameters = np.load(_terminal_parameter_path(state))
            predictions = np.asarray(state["holdout_predictions"], dtype=float)
            metric = terminal_metrics.get(key, {})
            parameter_difference = (
                float(np.max(np.abs(parameters - reference_parameters)))
                if parameters.shape == reference_parameters.shape
                else float("inf")
            )
            prediction_difference = (
                float(np.max(np.abs(predictions - reference_predictions)))
                if predictions.shape == reference_predictions.shape
                else float("inf")
            )
            f1_difference = (
                abs(float(metric["f1"]) - float(reference_metric["f1"]))
                if metric and reference_metric
                else ""
            )
            pr_auc_difference = (
                abs(float(metric["pr_auc"]) - float(reference_metric["pr_auc"]))
                if metric and reference_metric
                else ""
            )
            result.append(
                {
                    "outer_fold": outer,
                    "initial_label_count": initial,
                    "reference_strategy": "random",
                    "strategy": strategy,
                    "same_235_sequences_and_order": (
                        state["ordered_row_ids_checksum_sha256"]
                        == reference["ordered_row_ids_checksum_sha256"]
                    ),
                    "same_model_seed": state["model_seed"] == reference["model_seed"],
                    "parameter_max_abs_difference": parameter_difference,
                    "holdout_prediction_max_abs_difference": prediction_difference,
                    "holdout_f1_abs_difference": f1_difference,
                    "holdout_pr_auc_abs_difference": pr_auc_difference,
                    "numerical_tolerance": tolerance,
                    "parameters_within_tolerance": parameter_difference <= tolerance,
                    "predictions_within_tolerance": prediction_difference <= tolerance,
                    "metrics_within_tolerance": (
                        f1_difference != ""
                        and pr_auc_difference != ""
                        and float(f1_difference) <= tolerance
                        and float(pr_auc_difference) <= tolerance
                    ),
                }
            )
    return result


def _terminal_parameter_path(state: dict[str, object]) -> Path:
    recorded = Path(str(state["parameter_vector_path"]))
    if recorded.exists():
        return recorded
    local = Path(str(state["_run_dir"])) / recorded.name
    if local.exists():
        return local
    raise FileNotFoundError(
        "Terminal parameter vector is unavailable at both the recorded path "
        f"{recorded} and the imported run-local path {local}"
    )


def _phase1_contextual_rows(
    output_root: Path,
    round_rows: list[dict[str, str]],
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    phase1_root = Path(str(manifest["phase1_root"]).replace("\\", "/"))
    baseline_path = (
        phase1_root
        / "tables"
        / "nested_cv_outer_predictions_AP_SP.csv"
    )
    baseline = {
        str(row["outer_fold_id"]): row
        for row in _read_csv(baseline_path)
        if row.get("model") == "AP_SP" and row.get("threshold_type") == "PR"
    }
    result = []
    for row in round_rows:
        if row.get("evaluation_dataset") != "holdout":
            continue
        if int(float(row.get("labeled_count", 0) or 0)) != 235:
            continue
        outer = str(row["outer_fold_id"])
        if outer not in baseline:
            continue
        reference = baseline[outer]
        phase5_f1 = float(row["f1"])
        phase5_pr_auc = float(row["pr_auc"])
        phase1_f1 = float(reference["F1"])
        phase1_pr_auc = float(reference["PR-AUC"])
        result.append(
            {
                "outer_fold": outer,
                "initial_label_count": row["initial_label_count"],
                "strategy": row["strategy"],
                "phase5_terminal_training_rows": 235,
                "phase5_validation_calibration_rows": 59,
                "phase1_context_training_rows": 294,
                "phase1_context_validation_protocol": (
                    "nested-CV outer model using the complete 294-row development set"
                ),
                "phase5_terminal_f1": phase5_f1,
                "phase1_fold_matched_f1": phase1_f1,
                "absolute_f1_gap_phase5_minus_phase1": phase5_f1 - phase1_f1,
                "phase5_terminal_pr_auc": phase5_pr_auc,
                "phase1_fold_matched_pr_auc": phase1_pr_auc,
                "absolute_pr_auc_gap_phase5_minus_phase1": (
                    phase5_pr_auc - phase1_pr_auc
                ),
                "interpretation": (
                    "contextual_only_not_used_for_strategy_ranking_aulc_or_hypothesis_tests"
                ),
            }
        )
    return result


def aggregate_phase5(output_root: Path) -> dict[str, object]:
    manifest = _load_phase5_manifest(output_root)
    replay_root = output_root / "replay"
    jobs = sorted(replay_root.glob("outer_*/initial_*/**/status.json"))
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in jobs]
    expected = _expected_phase5_jobs(manifest)
    completed = sum(row.get("status") == "completed" for row in statuses)

    round_rows = _collect_job_csv(replay_root, "round_metrics.csv")
    selected_rows = _collect_job_csv(replay_root, "selected_sequences.csv")
    candidate_rows = _collect_job_csv(replay_root, "candidate_scoring.csv")
    proxy_rows = _collect_job_csv(replay_root, "proxy_validity_records.csv")
    proxy_summary = _collect_job_csv(replay_root, "proxy_validity_summary.csv")
    resource_rows = _collect_job_csv(replay_root, "resource_log.csv")

    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "learning_curves.csv", round_rows)
    _write_csv(tables / "selected_sequences.csv", selected_rows)
    _write_csv(tables / "candidate_scoring.csv", candidate_rows)
    _write_csv(tables / "proxy_validity_records.csv", proxy_rows)
    _write_csv(tables / "proxy_validity_summary.csv", proxy_summary)
    _write_csv(tables / "compute_time.csv", resource_rows)

    aulc_rows = _paired_aulc_rows(round_rows)
    aulc_summary = _paired_aulc_summary(aulc_rows)
    target_rows = _labels_to_target_rows(round_rows)
    target_summary = _labels_to_target_summary(target_rows)
    selected_difficulty = _selected_difficulty_rows(candidate_rows)
    eligible_fraction = _eligible_fraction_rows(candidate_rows)
    overlap_rows = _overlap_rows(selected_rows)
    diversity_rows = _diversity_rows(selected_rows)
    yield_rows = _positive_yield_rows(selected_rows)
    terminal_convergence = _terminal_convergence_rows(output_root, round_rows)
    phase1_context = _phase1_contextual_rows(output_root, round_rows, manifest)
    _write_csv(tables / "paired_aulc_differences.csv", aulc_rows)
    _write_csv(tables / "paired_aulc_summary.csv", aulc_summary)
    _write_csv(tables / "labels_to_target.csv", target_rows)
    _write_csv(tables / "labels_to_target_summary.csv", target_summary)
    _write_csv(tables / "selected_difficulty_by_round.csv", selected_difficulty)
    _write_csv(tables / "eligible_pool_fraction.csv", eligible_fraction)
    _write_csv(tables / "selection_overlap.csv", overlap_rows)
    _write_csv(tables / "sequence_diversity.csv", diversity_rows)
    _write_csv(tables / "selected_positive_yield.csv", yield_rows)
    _write_csv(tables / "terminal_convergence_audit.csv", terminal_convergence)
    _write_csv(tables / "phase1_contextual_baseline.csv", phase1_context)
    figures = _write_phase5_figures(output_root)

    payload = {
        "status": "complete" if completed == expected else "incomplete",
        "expected_jobs": expected,
        "completed_jobs": completed,
        "missing_jobs": expected - completed,
        "figures": figures,
        "results_claim_allowed": completed == expected,
        "aggregated_at": _now_iso(),
    }
    _write_json(output_root / "manifests" / "aggregation_status.json", payload)
    return payload


def phase5_status(output_root: Path) -> dict[str, object]:
    manifest = _load_phase5_manifest(output_root)
    expected = _expected_phase5_jobs(manifest)
    statuses = []
    for path in sorted((output_root / "replay").glob("outer_*/initial_*/*/status.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["path"] = str(path)
        statuses.append(payload)
    completed = sum(row.get("status") == "completed" for row in statuses)
    if completed == expected:
        overall_status = "complete"
    elif completed == 0:
        overall_status = "implemented_but_not_yet_executed"
    else:
        overall_status = "in_progress"
    return {
        "status": overall_status,
        "completed_jobs": completed,
        "expected_jobs": expected,
        "ready_for_results_claims": completed == expected,
        "jobs": statuses,
    }


def derive_phase5_walltime(
    phase2_root: Path,
    explicit_override: str | None,
) -> dict[str, object]:
    if explicit_override:
        _parse_walltime(explicit_override)
        return {
            "source": "explicit_override",
            "requested_walltime": explicit_override,
            "source_jobs": [],
            "safety_multiplier": PBS_SAFETY_MULTIPLIER,
            "additional_seconds": PBS_ADDITIONAL_SECONDS,
        }
    records = []
    for path in sorted((phase2_root / "benchmark" / "runs").glob("*/resource_log.csv")):
        for row in _read_csv(path):
            try:
                elapsed = float(row.get("walltime_seconds", ""))
            except (TypeError, ValueError):
                continue
            if row.get("exit_status") != "success" or elapsed <= 0:
                continue
            records.append(
                {
                    "run_id": row.get("run_id", path.parent.name),
                    "resource_log": str(path),
                    "full_benchmark_elapsed_seconds": elapsed,
                    "strategy_count": len(BENCHMARK_STRATEGIES),
                    "estimated_phase5_primary_job_seconds": elapsed
                    * ((PACE_MAX_ROUNDS + 1) / (21 * 5 * len(BENCHMARK_STRATEGIES))),
                }
            )
    if not records:
        raise ValueError(
            "Reliable completed Phase 2 benchmark runtime evidence is unavailable; "
            "provide --supek-walltime HH:MM:SS explicitly."
        )
    comparable_max = max(row["estimated_phase5_primary_job_seconds"] for row in records)
    requested_seconds = comparable_max * PBS_SAFETY_MULTIPLIER + PBS_ADDITIONAL_SECONDS
    requested_hours = max(PBS_MINIMUM_HOURS, int(math.ceil(requested_seconds / 3600.0)))
    return {
        "source": "completed_phase2_benchmark_resource_logs",
        "requested_walltime": f"{requested_hours:02d}:00:00",
        "source_jobs": records,
        "derivation": (
            "max(completed Phase 2 benchmark elapsed * 46 primary Phase 5 fits / "
            "(21 replay points * 5 members * 10 benchmark strategies)) * 2.0 "
            "+ 1800 seconds, rounded upward to an hour, minimum two hours"
        ),
        "safety_multiplier": PBS_SAFETY_MULTIPLIER,
        "additional_seconds": PBS_ADDITIONAL_SECONDS,
    }


def write_phase5_pbs(options: Phase5Options, walltime: str) -> list[Path]:
    pbs_dir = options.output_root / "pbs"
    pbs_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _remote_path_text(options.pbs_repo_root or Path.cwd())
    output_root = _join_remote(repo_root, options.output_root)
    phase1_root = _join_remote(repo_root, options.phase1_root)
    phase2_root = _join_remote(repo_root, options.phase2_root)
    created: list[Path] = []
    job_paths: list[Path] = []
    for outer_fold in options.outer_folds:
        for initial in options.initial_label_counts:
            for strategy in options.strategies:
                stem = f"p5_o{outer_fold}_n{initial}_{strategy}"
                path = pbs_dir / f"{stem}.pbs"
                command = (
                    "python -m active_learning_thesis phase5-self-paced run-job "
                    f"--phase1-root {_quote_text(phase1_root)} "
                    f"--phase2-root {_quote_text(phase2_root)} "
                    f"--output-root {_quote_text(output_root)} "
                    f"--outer-fold {outer_fold} --inner-fold {options.inner_fold} "
                    f"--initial-label-count {initial} --strategy {strategy} "
                    f"--batch-size {options.batch_size} --max-rounds {options.max_rounds} "
                    f"--ensemble-size {options.ensemble_size} --base-seed {options.base_seed}"
                )
                log_root = posixpath.join(output_root, "logs")
                path.write_text(
                    _pbs_text(
                        stem,
                        command,
                        repo_root,
                        posixpath.join(log_root, f"{stem}.out"),
                        posixpath.join(log_root, f"{stem}.err"),
                        walltime,
                        options,
                    ),
                    encoding="utf-8",
                )
                created.append(path)
                job_paths.append(path)
    aggregate = pbs_dir / "p5_aggregate.pbs"
    aggregate.write_text(
        _pbs_text(
            "p5_aggregate",
            (
                "python -m active_learning_thesis phase5-self-paced aggregate "
                f"--output-root {_quote_text(output_root)}"
            ),
            repo_root,
            posixpath.join(output_root, "logs", "p5_aggregate.out"),
            posixpath.join(output_root, "logs", "p5_aggregate.err"),
            "02:00:00",
            replace(options, supek_ngpus=0, supek_ncpus=2, supek_mem="8GB"),
        ),
        encoding="utf-8",
    )
    created.append(aggregate)
    submit = pbs_dir / "submit_phase5_all.sh"
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
        'MAX_ACTIVE_JOBS="${PHASE5_MAX_ACTIVE_JOBS:-6}"',
        "job_ids=()",
        "wait_for_queue_slot() {",
        '  if ! command -v qselect >/dev/null 2>&1; then return 0; fi',
        "  while true; do",
        '    active=$(qselect -u "$USER" 2>/dev/null | wc -l | tr -d " ")',
        '    if [ "${active:-0}" -lt "$MAX_ACTIVE_JOBS" ]; then return 0; fi',
        '    echo "[phase5] $active active jobs; waiting for a scheduler slot..."',
        "    sleep 60",
        "  done",
        "}",
    ]
    for path in job_paths:
        lines.append("wait_for_queue_slot")
        lines.append(f'job_ids+=("$(qsub "$SCRIPT_DIR/{path.name}")")')
    lines.extend(
        [
            'dependency=$(IFS=:; echo "${job_ids[*]}")',
            f'qsub -W depend=afterany:"$dependency" "$SCRIPT_DIR/{aggregate.name}"',
            'printf "submitted_phase5_jobs=%s\\n" "${job_ids[*]}"',
        ]
    )
    submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    created.append(submit)
    return created


def _pbs_text(job_name, command, repo_root, stdout, stderr, walltime, options) -> str:
    gpu = f":ngpus={options.supek_ngpus}" if options.supek_ngpus else ""
    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q {options.supek_queue}
#PBS -l select=1:ncpus={options.supek_ncpus}:mem={options.supek_mem}{gpu}
#PBS -l walltime={walltime}
#PBS -o {stdout}
#PBS -e {stderr}

set -eo pipefail
cd "{repo_root}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
echo "[phase5] start $(date -Is) host=$(hostname)"
status=0
{command} || status=$?
echo "[phase5] end $(date -Is) exit_status=$status"
exit "$status"
"""


def _metric_row(spec, replay_point, labelled_count, dataset, metrics):
    return {
        "mode": "phase5",
        "setup": (
            "single_calibrated" if spec.ensemble_size == 1 else "ensemble_calibrated"
        ),
        "outer_fold_id": spec.outer_fold_id,
        "inner_fold_id": spec.inner_fold_id,
        "initial_label_count": spec.initial_label_count,
        "strategy": spec.strategies[0],
        "round_id": replay_point,
        "replay_point": replay_point,
        "labeled_count": labelled_count,
        "evaluation_dataset": dataset,
        **metrics,
    }


def _write_job_inputs(run_dir, options, spec, rows, config):
    _write_json(
        run_dir / "job_manifest.json",
        {
            "experiment": "SPAL-inspired self-paced active-learning replay",
            "status": "running",
            "outer_fold": spec.outer_fold_id,
            "inner_fold": spec.inner_fold_id,
            "initial_label_count": spec.initial_label_count,
            "strategy": spec.strategies[0],
            "run_seed": spec.run_seed,
            "strategy_independent_model_seed_rule": (
                "base + outer_fold*100000 + initial_label_count*1000 + "
                "replay_point*10 + ensemble_member"
            ),
            "initial_labelled_sequences": [row["sequence"] for row in rows.replay_seed],
            "initial_labelled_ordered_row_ids": [
                int(row["original_dataset_row_id"])
                for row in canonical_training_order(rows.replay_seed)
            ],
            "initial_labelled_ordered_row_ids_checksum_sha256": ordered_row_id_checksum(
                canonical_training_order(rows.replay_seed)
            ),
            "validation_ordered_row_ids_checksum_sha256": ordered_row_id_checksum(
                canonical_training_order(rows.validation)
            ),
            "validation_count": len(rows.validation),
            "holdout_count": len(rows.holdout),
            "train_pool_count": len(rows.train_pool),
            "embedding_contract": ["batch_size", EMBEDDING_WIDTH],
            "config": config.to_dict(),
            "source_checksums": _source_checksums(options),
        },
    )
    _write_csv(
        run_dir / "split_audit.csv",
        [
            {"split": name, "count": len(getattr(rows, name))}
            for name in ("holdout", "validation", "train_pool", "replay_seed", "replay_hidden")
        ],
    )


def _paired_aulc_rows(round_rows):
    holdout = [row for row in round_rows if row.get("evaluation_dataset") == "holdout"]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in holdout:
        key = (
            row.get("outer_fold_id", ""),
            row.get("initial_label_count", ""),
            row.get("strategy", ""),
        )
        groups.setdefault(key, []).append(row)
    values = {}
    for key, points in groups.items():
        initial = int(key[1])
        for endpoint in PREREGISTERED_AULC_ENDPOINTS:
            if endpoint <= initial:
                continue
            value = normalized_aulc_interval(points, "f1", initial, endpoint)
            if value is not None:
                values[(key[0], key[1], key[2], endpoint)] = value
    comparisons = [
        ("self_paced_entropy", "predictive_entropy"),
        ("self_paced_entropy", "static_easy_entropy"),
        ("self_paced_entropy", "random"),
        ("static_easy_entropy", "predictive_entropy"),
    ]
    result = []
    fold_conditions = sorted({(outer, initial) for outer, initial, _strategy in groups})
    for outer, initial in fold_conditions:
        for endpoint in PREREGISTERED_AULC_ENDPOINTS:
            for left, right in comparisons:
                left_value = values.get((outer, initial, left, endpoint))
                right_value = values.get((outer, initial, right, endpoint))
                if left_value is None or right_value is None:
                    continue
                result.append(
                    {
                        "outer_fold": outer,
                        "initial_label_count": initial,
                        "interval_start": initial,
                        "interval_end": endpoint,
                        "aulc_scope": "full" if endpoint == 235 else "partial",
                        "comparison": f"{left} - {right}",
                        "left_aulc_f1": left_value,
                        "right_aulc_f1": right_value,
                        "delta_aulc_f1": left_value - right_value,
                    }
                )
    return result


def normalized_aulc_interval(points, metric: str, start: int, end: int) -> float | None:
    selected = sorted(
        (
            (int(float(row["labeled_count"])), float(row[metric]))
            for row in points
            if start <= int(float(row["labeled_count"])) <= end
        ),
        key=lambda item: item[0],
    )
    if len(selected) < 2 or selected[0][0] != start or selected[-1][0] != end:
        return None
    area = sum(
        ((left_value + right_value) / 2.0) * (right_count - left_count)
        for (left_count, left_value), (right_count, right_value) in zip(
            selected, selected[1:]
        )
    )
    return area / (end - start)


def _labels_to_target_rows(round_rows):
    result = []
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in round_rows:
        if row.get("evaluation_dataset") != "holdout":
            continue
        key = (
            row.get("outer_fold_id", ""),
            row.get("initial_label_count", ""),
            row.get("strategy", ""),
        )
        groups.setdefault(key, []).append(row)
    for (outer, initial, strategy), points in sorted(groups.items()):
        for target in TARGET_F1_VALUES:
            reached = labels_to_target(points, "f1", target)
            result.append(
                {
                    "outer_fold": outer,
                    "initial_label_count": initial,
                    "strategy": strategy,
                    "target_f1": target,
                    "labels_to_target": "" if reached is None else reached,
                    "reached": reached is not None,
                }
            )
    return result


def _paired_aulc_summary(rows):
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["initial_label_count"]),
                str(row["interval_start"]),
                str(row["interval_end"]),
                str(row["comparison"]),
            ),
            [],
        ).append(float(row["delta_aulc_f1"]))
    return [
        {
            "initial_label_count": key[0],
            "interval_start": key[1],
            "interval_end": key[2],
            "aulc_scope": "full" if key[2] == "235" else "partial",
            "comparison": key[3],
            "fold_count": len(values),
            "mean_delta_aulc_f1": _mean(values),
            "std_delta_aulc_f1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum_delta_aulc_f1": min(values),
            "maximum_delta_aulc_f1": max(values),
            "worst_fold_delta_aulc_f1": min(values),
        }
        for key, values in sorted(groups.items())
    ]


def _labels_to_target_summary(rows):
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["initial_label_count"]),
                str(row["strategy"]),
                str(row["target_f1"]),
            ),
            [],
        ).append(row)
    result = []
    for key, group in sorted(groups.items()):
        reached_values = [
            float(row["labels_to_target"])
            for row in group
            if row["labels_to_target"] != ""
        ]
        result.append(
            {
                "initial_label_count": key[0],
                "strategy": key[1],
                "target_f1": key[2],
                "fold_count": len(group),
                "reach_count": len(reached_values),
                "reach_fraction": len(reached_values) / len(group),
                "conditional_mean_labels_to_target": _mean(reached_values),
                "conditional_std_labels_to_target": (
                    float(np.std(reached_values, ddof=1))
                    if len(reached_values) > 1
                    else (0.0 if reached_values else "")
                ),
            }
        )
    return result


def _selected_difficulty_rows(candidate_rows):
    return [
        row
        for row in candidate_rows
        if str(row.get("selected", "")).lower() == "true"
    ]


def _eligible_fraction_rows(candidate_rows):
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in candidate_rows:
        key = (
            row.get("outer_fold", ""),
            row.get("initial_label_count", ""),
            row.get("strategy", ""),
            row.get("acquisition_step", ""),
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "outer_fold": key[0],
            "initial_label_count": key[1],
            "strategy": key[2],
            "acquisition_step": key[3],
            "candidate_count": len(group),
            "eligible_count": sum(
                str(row.get("eligible", "")).lower() == "true" for row in group
            ),
            "eligible_fraction": (
                sum(str(row.get("eligible", "")).lower() == "true" for row in group)
                / len(group)
            ),
        }
        for key, group in sorted(groups.items())
        if group
    ]


def _overlap_rows(selected_rows):
    groups = _selected_rows_by_trajectory(selected_rows)
    result = []
    for outer in sorted({key[0] for key in groups}):
        for initial in sorted({key[1] for key in groups if key[0] == outer}):
            counts = sorted(
                {
                    _labelled_count_after_selection(row)
                    for key, rows in groups.items()
                    if key[0] == outer and key[1] == initial
                    for row in rows
                }
            )
            for labelled_count in counts:
                fixed_budget = labelled_count in FIXED_BUDGET_COUNTS
                terminal = labelled_count == 235
                cumulative = {
                    strategy: {
                        row.get("sequence", "")
                        for row in groups.get((outer, initial, strategy), [])
                        if _labelled_count_after_selection(row) <= labelled_count
                    }
                    for strategy in PHASE5_STRATEGIES
                }
                batch = {
                    strategy: {
                        row.get("sequence", "")
                        for row in groups.get((outer, initial, strategy), [])
                        if _labelled_count_after_selection(row) == labelled_count
                    }
                    for strategy in PHASE5_STRATEGIES
                }
                for scope, sets in (("batch", batch), ("cumulative", cumulative)):
                    for left_index, left in enumerate(PHASE5_STRATEGIES):
                        for right in PHASE5_STRATEGIES[left_index:]:
                            a = sets[left]
                            b = sets[right]
                            union = a | b
                            result.append(
                                {
                                    "outer_fold": outer,
                                    "initial_label_count": initial,
                                    "labelled_count": labelled_count,
                                    "scope": scope,
                                    "fixed_budget_summary": fixed_budget,
                                    "terminal_consistency_only": terminal,
                                    "strategy_a": left,
                                    "strategy_b": right,
                                    "jaccard": len(a & b) / len(union) if union else "",
                                }
                            )
    return result


def _diversity_rows(selected_rows):
    groups = _selected_rows_by_trajectory(selected_rows)
    result = []
    for key, rows in sorted(groups.items()):
        for labelled_count in sorted({_labelled_count_after_selection(row) for row in rows}):
            for scope in ("batch", "cumulative"):
                subset = [
                    row.get("sequence", "")
                    for row in rows
                    if (
                        _labelled_count_after_selection(row) == labelled_count
                        if scope == "batch"
                        else _labelled_count_after_selection(row) <= labelled_count
                    )
                ]
                values = [
                    _normalized_levenshtein(left, right)
                    for index, left in enumerate(subset)
                    for right in subset[index + 1 :]
                ]
                result.append(
                    {
                        "outer_fold": key[0],
                        "initial_label_count": key[1],
                        "strategy": key[2],
                        "labelled_count": labelled_count,
                        "scope": scope,
                        "fixed_budget_summary": labelled_count in FIXED_BUDGET_COUNTS,
                        "terminal_consistency_only": labelled_count == 235,
                        "selected_count": len(subset),
                        "mean_pairwise_normalized_levenshtein": _mean(values),
                        "minimum_pairwise_normalized_levenshtein": (
                            min(values) if values else ""
                        ),
                    }
                )
    return result


def _positive_yield_rows(selected_rows):
    groups = _selected_rows_by_trajectory(selected_rows)
    result = []
    for key, rows in sorted(groups.items()):
        for labelled_count in sorted({_labelled_count_after_selection(row) for row in rows}):
            for scope in ("batch", "cumulative"):
                subset = [
                    int(float(row.get("label", 0) or 0))
                    for row in rows
                    if (
                        _labelled_count_after_selection(row) == labelled_count
                        if scope == "batch"
                        else _labelled_count_after_selection(row) <= labelled_count
                    )
                ]
                result.append(
                    {
                        "outer_fold": key[0],
                        "initial_label_count": key[1],
                        "strategy": key[2],
                        "labelled_count": labelled_count,
                        "scope": scope,
                        "fixed_budget_summary": labelled_count in FIXED_BUDGET_COUNTS,
                        "terminal_consistency_only": labelled_count == 235,
                        "selected_count": len(subset),
                        "positive_count": sum(subset),
                        "positive_yield": sum(subset) / len(subset) if subset else "",
                    }
                )
    return result


def _selected_rows_by_trajectory(selected_rows):
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in selected_rows:
        key = (
            row.get("outer_fold", ""),
            row.get("initial_label_count", ""),
            row.get("strategy", ""),
        )
        groups.setdefault(key, []).append(row)
    return groups


def _labelled_count_after_selection(row) -> int:
    return int(float(row["initial_label_count"])) + (
        int(float(row["acquisition_step"])) + 1
    ) * int(float(row.get("batch_size", 5) or 5))


def _normalized_levenshtein(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def _write_phase5_figures(output_root: Path) -> dict[str, str]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    names = [
        "holdout_f1_vs_labelled_count",
        "paired_aulc_differences",
        "labels_to_target",
        "selected_difficulty_vs_round",
        "eligible_pool_fraction_vs_round",
        "distance_vs_post_hoc_pre_query_log_loss",
        "error_rate_by_difficulty_quintile",
        "selection_overlap_heatmap",
        "sequence_diversity_comparison",
        "compute_time_comparison",
    ]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        outputs = {}
        initial_counts = _load_phase5_manifest(output_root)["initial_label_counts"]
        for initial in initial_counts:
            for name in names:
                path = figure_root / f"{name}_initial_{initial}.svg"
                path.write_text(
                    _simple_svg(
                        f"{name.replace('_', ' ').title()} (n0={initial})",
                        "Matplotlib unavailable; source data are stored in ../tables.",
                    ),
                    encoding="utf-8",
                )
                outputs[path.name] = str(path)
        return outputs

    tables = output_root / "tables"
    learning = _read_csv(tables / "learning_curves.csv")
    paired = _read_csv(tables / "paired_aulc_differences.csv")
    targets = _read_csv(tables / "labels_to_target_summary.csv")
    selected = _read_csv(tables / "selected_difficulty_by_round.csv")
    eligible = _read_csv(tables / "eligible_pool_fraction.csv")
    proxy = _read_csv(tables / "proxy_validity_records.csv")
    proxy_summary = _read_csv(tables / "proxy_validity_summary.csv")
    overlap = _read_csv(tables / "selection_overlap.csv")
    diversity = _read_csv(tables / "sequence_diversity.csv")
    compute = _read_csv(tables / "compute_time.csv")
    initial_counts = _load_phase5_manifest(output_root)["initial_label_counts"]
    outputs = {}

    for initial in initial_counts:
        initial_text = str(initial)
        figure_specs = []

        fig, ax = plt.subplots(figsize=(8, 5))
        filtered = [
            row
            for row in learning
            if row.get("initial_label_count") == initial_text
            and row.get("evaluation_dataset") == "holdout"
        ]
        for strategy in PHASE5_STRATEGIES:
            points: dict[float, list[float]] = {}
            for row in filtered:
                if row.get("strategy") == strategy:
                    points.setdefault(float(row["labeled_count"]), []).append(float(row["f1"]))
            if points:
                xs = sorted(points)
                ax.plot(xs, [_mean(points[x]) for x in xs], marker="o", label=strategy)
        ax.set(xlabel="Labelled count", ylabel="Holdout F1", title=f"Phase 5 holdout F1 (n0={initial})")
        ax.legend(fontsize=8)
        figure_specs.append(("holdout_f1_vs_labelled_count", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        groups: dict[str, list[float]] = {}
        for row in paired:
            if (
                row.get("initial_label_count") == initial_text
                and row.get("interval_end") == "235"
            ):
                groups.setdefault(row["comparison"], []).append(float(row["delta_aulc_f1"]))
        for index, (label, values) in enumerate(sorted(groups.items())):
            ax.scatter([index] * len(values), values, label=label)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(range(len(groups)), [key for key in sorted(groups)], rotation=25, ha="right")
        ax.set(ylabel="Paired delta AULC-F1", title=f"Paired fold differences (n0={initial})")
        figure_specs.append(("paired_aulc_differences", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        target_filtered = [row for row in targets if row.get("initial_label_count") == initial_text]
        labels = [f"{row['strategy']}:{row['target_f1']}" for row in target_filtered]
        values = [float(row["reach_fraction"]) for row in target_filtered]
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
        ax.set(ylabel="Fold reach fraction", title=f"Labels-to-target reach (n0={initial})")
        figure_specs.append(("labels_to_target", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in PHASE5_STRATEGIES:
            groups_by_step: dict[int, list[float]] = {}
            for row in selected:
                if row.get("initial_label_count") == initial_text and row.get("strategy") == strategy:
                    groups_by_step.setdefault(int(row["acquisition_step"]), []).append(float(row["difficulty_percentile"]))
            if groups_by_step:
                xs = sorted(groups_by_step)
                ax.plot(xs, [_mean(groups_by_step[x]) for x in xs], marker="o", label=strategy)
        ax.set(xlabel="Acquisition step", ylabel="Selected difficulty percentile", title=f"Selected familiarity difficulty (n0={initial})")
        ax.legend(fontsize=8)
        figure_specs.append(("selected_difficulty_vs_round", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in PHASE5_STRATEGIES:
            groups_by_step = {}
            for row in eligible:
                if row.get("initial_label_count") == initial_text and row.get("strategy") == strategy:
                    groups_by_step.setdefault(int(row["acquisition_step"]), []).append(float(row["eligible_fraction"]))
            if groups_by_step:
                xs = sorted(groups_by_step)
                ax.plot(xs, [_mean(groups_by_step[x]) for x in xs], marker="o", label=strategy)
        ax.set(xlabel="Acquisition step", ylabel="Eligible fraction", title=f"Eligible hidden pool (n0={initial})")
        ax.legend(fontsize=8)
        figure_specs.append(("eligible_pool_fraction_vs_round", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        proxy_filtered = [row for row in proxy if row.get("initial_label_count") == initial_text]
        ax.scatter(
            [float(row["embedding_distance_to_labelled"]) for row in proxy_filtered],
            [float(row["post_hoc_pre_query_log_loss"]) for row in proxy_filtered],
            s=5,
            alpha=0.25,
        )
        ax.set(xlabel="Calibrated neural-model labelled-manifold distance", ylabel="Post-hoc pre-query log loss", title=f"Familiarity proxy validity (n0={initial})")
        figure_specs.append(("distance_vs_post_hoc_pre_query_log_loss", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        quintile_rows = [
            row
            for row in proxy_summary
            if row.get("initial_label_count") == initial_text
            and row.get("summary_type") == "difficulty_quintile"
            and row.get("error_rate") not in {"", None}
        ]
        quintile_groups: dict[int, list[float]] = {}
        for row in quintile_rows:
            quintile_groups.setdefault(int(row["difficulty_quintile"]), []).append(float(row["error_rate"]))
        xs = sorted(quintile_groups)
        ax.bar(xs, [_mean(quintile_groups[x]) for x in xs])
        ax.set(xlabel="Operational difficulty quintile", ylabel="Fixed-0.5 error rate", title=f"Error by familiarity quintile (n0={initial})")
        figure_specs.append(("error_rate_by_difficulty_quintile", fig))

        fig, ax = plt.subplots(figsize=(6, 5))
        matrix = np.eye(len(PHASE5_STRATEGIES))
        overlap_filtered = [
            row
            for row in overlap
            if row.get("initial_label_count") == initial_text
            and row.get("scope") == "cumulative"
            and row.get("labelled_count") == "160"
        ]
        for row in overlap_filtered:
            if row.get("jaccard") == "":
                continue
            i = PHASE5_STRATEGIES.index(row["strategy_a"])
            j = PHASE5_STRATEGIES.index(row["strategy_b"])
            matrix[i, j] = matrix[j, i] = float(row["jaccard"])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(4), PHASE5_STRATEGIES, rotation=35, ha="right")
        ax.set_yticks(range(4), PHASE5_STRATEGIES)
        ax.set_title(f"Cumulative selection overlap at 160 labels (n0={initial})")
        fig.colorbar(image, ax=ax)
        figure_specs.append(("selection_overlap_heatmap", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        diversity_filtered = [
            row
            for row in diversity
            if row.get("initial_label_count") == initial_text
            and row.get("scope") == "cumulative"
            and row.get("labelled_count") == "160"
        ]
        grouped_diversity: dict[str, list[float]] = {}
        for row in diversity_filtered:
            if row.get("mean_pairwise_normalized_levenshtein") != "":
                grouped_diversity.setdefault(row["strategy"], []).append(float(row["mean_pairwise_normalized_levenshtein"]))
        labels = list(PHASE5_STRATEGIES)
        ax.bar(labels, [_mean(grouped_diversity.get(label, [])) or 0.0 for label in labels])
        ax.set(ylabel="Mean pairwise normalized Levenshtein", title=f"Cumulative selected diversity at 160 labels (n0={initial})")
        ax.tick_params(axis="x", rotation=25)
        figure_specs.append(("sequence_diversity_comparison", fig))

        fig, ax = plt.subplots(figsize=(8, 5))
        compute_values = [
            float(row["walltime_seconds"]) / 3600.0
            for row in compute
            if row.get("initial_label_count") == initial_text
            and row.get("walltime_seconds") not in {"", None}
        ]
        ax.bar(["completed jobs"], [_mean(compute_values) or 0.0])
        ax.set(ylabel="Mean wall time (hours)", title=f"Compute time (n0={initial})")
        figure_specs.append(("compute_time_comparison", fig))

        for name, fig in figure_specs:
            path = figure_root / f"{name}_initial_{initial}.svg"
            fig.tight_layout()
            fig.savefig(path, format="svg")
            plt.close(fig)
            outputs[path.name] = str(path)
    return outputs


def _simple_svg(title: str, subtitle: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
<rect width="960" height="540" fill="#ffffff"/>
<line x1="90" y1="450" x2="900" y2="450" stroke="#333333" stroke-width="2"/>
<line x1="90" y1="80" x2="90" y2="450" stroke="#333333" stroke-width="2"/>
<text x="90" y="45" font-family="Arial" font-size="24" fill="#111111">{title}</text>
<text x="90" y="500" font-family="Arial" font-size="14" fill="#555555">{subtitle}</text>
</svg>
"""


def _source_checksums(options: Phase5Options) -> dict[str, str]:
    paths = [
        options.phase1_root / "frozen_model_config.json",
        Path(__file__),
        Path(__file__).with_name("phase2_replay.py"),
        Path(__file__).with_name("predictive.py"),
    ]
    for outer_fold in options.outer_folds:
        paths.append(
            options.phase1_root
            / "folds"
            / f"replay_manifest_outer_{outer_fold}_inner_{options.inner_fold}.json"
        )
    return {
        str(path): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_dir(output_root: Path, outer_fold: int, initial: int, strategy: str) -> Path:
    return (
        output_root
        / "replay"
        / f"outer_{outer_fold}"
        / f"initial_{initial}"
        / strategy
    )


def _collect_job_csv(root: Path, filename: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(root.glob(f"outer_*/initial_*/*/{filename}")):
        parts = path.relative_to(root).parts
        outer = parts[0].replace("outer_", "")
        initial = parts[1].replace("initial_", "")
        strategy = parts[2]
        for row in _read_csv(path):
            row.setdefault("outer_fold", outer)
            row.setdefault("initial_label_count", initial)
            row.setdefault("strategy", strategy)
            rows.append(row)
    return rows


def _validate_options(options: Phase5Options) -> None:
    if options.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if options.max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if options.ensemble_size <= 0:
        raise ValueError("ensemble_size must be positive")
    unsupported = set(options.strategies) - set(PHASE5_STRATEGIES)
    if unsupported:
        raise ValueError(f"Unsupported Phase 5 strategies: {sorted(unsupported)}")


def _parse_walltime(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Walltime must use HH:MM:SS")
    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("Invalid walltime")
    return hours * 3600 + minutes * 60 + seconds


def _cleanup_tensorflow_runtime() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass


def _quote_text(path: str) -> str:
    return f'"{path}"'


def _remote_path_text(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("/"):
        return text
    return path.resolve().as_posix()


def _join_remote(repo_root: str, path: Path) -> str:
    text = path.as_posix()
    if text.startswith("/"):
        return text
    return posixpath.join(repo_root.rstrip("/"), text)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean(values: Sequence[float]) -> float | str:
    return float(statistics.mean(values)) if values else ""
