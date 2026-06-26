from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import re
import shutil

import numpy as np

from active_learning_thesis.acquisition import (
    ACQUISITION_DIAGNOSTIC_FIELDS,
    ACQUISITION_SELECTION_METADATA_FIELDS,
    acquisition_diagnostics,
    generator_objective_for_strategy,
    generator_objective_requires_embeddings,
    generator_objective_requires_family_models,
    requires_embeddings,
    requires_family_models,
    select_batch,
)
from active_learning_thesis.config import RunConfig
from active_learning_thesis.dataset import (
    build_split_manifest,
    load_split_manifest,
    read_experimental_dataset,
    save_split_manifest,
)
from active_learning_thesis.dependencies import ensure_predictive_runtime
from active_learning_thesis.discovery import (
    discovery_utility_scores,
    mean_pairwise_distance,
    min_distances_to_reference,
)
from active_learning_thesis.generative import generate_candidate_sequences
from active_learning_thesis.ledger import (
    append_rows,
    create_initial_ledger,
    current_real_training_rows,
    empty_row,
    holdout_rows,
    index_by_sequence,
    load_ledger,
    next_real_round_id,
    replay_hidden_rows,
    replay_seed_rows,
    save_ledger,
    serialize_probabilities,
    snapshot_ledger,
    unresolved_proposals,
    validation_rows,
)
from active_learning_thesis.predictive import (
    evaluate_holdout,
    evaluate_rows,
    load_ensemble_from_dir,
    score_sequences_with_ensemble,
    score_sequences_with_family,
    train_ensemble,
    train_family,
)


ROUND_PATTERN = re.compile(r"round_(\d+)")


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "config": run_dir / "config.json",
        "split_manifest": run_dir / "split_manifest.json",
        "ledger": run_dir / "ledger.csv",
        "snapshots": run_dir / "snapshots",
        "metrics": run_dir / "metrics",
        "models": run_dir / "models",
        "replay": run_dir / "replay",
        "discovery": run_dir / "discovery",
        "batches": run_dir / "batches",
        "imports": run_dir / "imports",
        "candidates": run_dir / "candidates",
        "model_cache": run_dir / "models" / "cache",
    }


def _ensure_run_dirs(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key in {"config", "split_manifest", "ledger"}:
            continue
        path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _strategy_paths(base: Path, strategy: str, round_id: int) -> dict[str, Path]:
    round_dir = base / strategy / f"round_{round_id:03d}"
    return {
        "round_dir": round_dir,
        "ensemble": round_dir / "ensemble",
        "family": round_dir / "family",
        "metrics": round_dir / "metrics.json",
        "replay_ledger": round_dir / "ledger.csv",
    }


def _real_round_paths(base: Path, round_id: int, stage: str) -> dict[str, Path]:
    round_dir = base / "real_al" / f"round_{round_id:03d}" / stage
    return {
        "round_dir": round_dir,
        "ensemble": round_dir / "ensemble",
        "family": round_dir / "family",
        "metrics": round_dir / "metrics.json",
    }


def _discovery_strategy_paths(base: Path, strategy: str) -> dict[str, Path]:
    strategy_dir = base / strategy
    return {
        "strategy_dir": strategy_dir,
        "summary": strategy_dir / "summary.json",
        "candidates": strategy_dir / "candidates.csv",
        "top_batch": strategy_dir / "top_batch.csv",
    }


def _rows_to_sequences(rows: list[dict[str, str]]) -> list[str]:
    return [row["sequence"] for row in rows]


def _labeled_embeddings(ensemble, rows: list[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    scored = score_sequences_with_ensemble(
        ensemble,
        _rows_to_sequences(rows),
        include_embeddings=True,
        use_calibration=False,
    )
    return np.asarray(scored["avg_embedding"], dtype=float)


def _round_metrics(
    round_id: int,
    strategy: str,
    labeled_count: int,
    metrics: dict[str, float],
    extra: dict | None = None,
) -> dict:
    payload = {
        "round_id": round_id,
        "strategy": strategy,
        "labeled_count": labeled_count,
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            payload[key] = float(value)
        else:
            payload[key] = value
    if extra:
        payload.update(extra)
    return payload


def _build_replay_snapshot(
    all_train_rows: list[dict[str, str]],
    labeled_sequences: set[str],
    strategy: str,
    round_id: int,
) -> list[dict[str, str]]:
    snapshot_rows = []
    for row in all_train_rows:
        snapshot_rows.append(
            empty_row(
                {
                    "sequence": row["sequence"],
                    "label": row["label"],
                    "label_source": "experimental",
                    "split": row["split"],
                    "mode": "replay",
                    "round_id": str(round_id),
                    "status": "labeled" if row["sequence"] in labeled_sequences else "hidden",
                    "generator_origin": f"replay_{strategy}",
                    "replay_role": row["replay_role"],
                }
            )
        )
    return snapshot_rows


def _export_batch_csv(path: Path, selected_rows: list[dict[str, str]]) -> None:
    fields = [
        "sequence",
        "round_id",
        "acquisition_strategy",
        "pred_mean",
        "pred_std",
        "pred_entropy",
        "pred_mutual_information",
        "raw_pred_mean",
        "raw_pred_std",
        "raw_pred_entropy",
        "raw_pred_mutual_information",
        "acquisition_score",
        "similarity_penalty",
        *ACQUISITION_DIAGNOSTIC_FIELDS,
    ]
    _save_csv(path, fields, selected_rows)


REPLAY_SELECTED_BATCH_FIELDS = [
    "sequence",
    "label",
    "round_id",
    "acquisition_strategy",
    "pred_mean",
    "pred_std",
    "pred_entropy",
    "pred_mutual_information",
    "acquisition_score",
    "similarity_penalty",
    *ACQUISITION_DIAGNOSTIC_FIELDS,
]


def _replay_selected_batch_rows(
    candidate_rows: list[dict[str, str]],
    selected_indices: list[int],
    candidate_scores: dict[str, np.ndarray],
    acquisition_scores: np.ndarray,
    diagnostics: list[dict[str, float | int | str]],
    strategy: str,
    round_id: int,
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for index in selected_indices:
        row = {
            "sequence": candidate_rows[index]["sequence"],
            "label": candidate_rows[index].get("label", ""),
            "round_id": round_id,
            "acquisition_strategy": strategy,
            "pred_mean": float(candidate_scores["pred_mean"][index]),
            "pred_std": float(candidate_scores["pred_std"][index]),
            "pred_entropy": float(candidate_scores["pred_entropy"][index]),
            "pred_mutual_information": float(
                candidate_scores["pred_mutual_information"][index]
            ),
            "acquisition_score": float(acquisition_scores[index]),
        }
        row.update(diagnostics[index])
        rows.append(row)
    return rows


def _selection_metadata_from_diagnostics(
    diagnostics: list[dict[str, float | int | str]],
    selected_indices: list[int],
) -> dict[str, float | int | str]:
    if not selected_indices:
        return {}
    first_selected = diagnostics[selected_indices[0]]
    metadata = {
        field: first_selected.get(field, "")
        for field in ACQUISITION_SELECTION_METADATA_FIELDS
    }
    return {field: value for field, value in metadata.items() if value != ""}


def _cleanup_tensorflow_runtime() -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _round_id_from_dir(round_dir: Path) -> int | None:
    match = ROUND_PATTERN.fullmatch(round_dir.name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_discovery_ensemble_dir(models_root: Path) -> tuple[Path, dict[str, int | str]]:
    real_al_root = models_root / "real_al"
    candidates: list[tuple[int, Path]] = []
    if real_al_root.exists():
        for round_dir in real_al_root.iterdir():
            if not round_dir.is_dir():
                continue
            round_id = _round_id_from_dir(round_dir)
            if round_id is None:
                continue
            ensemble_dir = round_dir / "post_ingest" / "ensemble"
            if (ensemble_dir / "ap_sp_member_00.h5").exists():
                candidates.append((round_id, ensemble_dir))
    if candidates:
        round_id, ensemble_dir = max(candidates, key=lambda item: item[0])
        return ensemble_dir, {"surrogate_stage": "post_ingest", "surrogate_round_id": round_id}

    baseline_dir = real_al_root / "round_000" / "baseline" / "ensemble"
    if (baseline_dir / "ap_sp_member_00.h5").exists():
        return baseline_dir, {"surrogate_stage": "baseline", "surrogate_round_id": 0}

    raise FileNotFoundError(
        "Unable to find a trained discovery ensemble. Expected a latest post_ingest "
        "ensemble or the baseline ensemble under models/real_al/."
    )


def _validate_cgmd_import(
    ledger_rows: list[dict[str, str]],
    import_path: Path,
) -> tuple[int, list[dict[str, str]]]:
    proposed = [row for row in ledger_rows if row["status"] == "proposed"]
    if not proposed:
        raise ValueError("There is no proposed batch waiting for CG-MD labels.")
    round_ids = {row["round_id"] for row in proposed}
    if len(round_ids) != 1:
        raise ValueError("Multiple proposed rounds are pending. Ingest one round at a time.")
    round_id = int(next(iter(round_ids)))
    proposed_by_sequence = {row["sequence"]: row for row in proposed}

    seen: set[str] = set()
    imported_rows: list[dict[str, str]] = []
    with import_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {"sequence", "round_id", "cgmd_label"}
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(
                "CG-MD import CSV must contain exactly the columns "
                "sequence, round_id, cgmd_label"
            )
        for row in reader:
            sequence = row["sequence"].strip()
            if sequence in seen:
                raise ValueError(f"Duplicate sequence in CG-MD import: {sequence}")
            seen.add(sequence)
            if sequence not in proposed_by_sequence:
                raise ValueError(
                    f"Sequence {sequence} was not part of the proposed batch."
                )
            if int(row["round_id"]) != round_id:
                raise ValueError(
                    f"Round mismatch for {sequence}: expected {round_id}, "
                    f"got {row['round_id']}"
                )
            label = row["cgmd_label"].strip()
            if label not in {"0", "1"}:
                raise ValueError(f"Invalid cgmd_label for {sequence}: {label}")
            imported_rows.append(
                {
                    "sequence": sequence,
                    "round_id": round_id,
                    "cgmd_label": label,
                }
            )
    if set(seen) != set(proposed_by_sequence):
        missing = sorted(set(proposed_by_sequence) - set(seen))
        raise ValueError(
            "CG-MD import is missing proposed sequences: " + ", ".join(missing)
        )
    return round_id, imported_rows


def init_run(config: RunConfig, *, train_baseline: bool = True) -> Path:
    run_dir = config.run_dir
    paths = _run_paths(run_dir)
    if paths["config"].exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    if train_baseline:
        ensure_predictive_runtime()
    _ensure_run_dirs(paths)
    records = read_experimental_dataset()
    manifest = build_split_manifest(records, config)
    save_split_manifest(paths["split_manifest"], manifest)
    config.save(paths["config"])

    ledger_rows = create_initial_ledger(records, manifest)
    save_ledger(paths["ledger"], ledger_rows)
    snapshot_ledger(paths["snapshots"], ledger_rows, "ledger_round_000_initial")
    if not train_baseline:
        return run_dir

    training_rows = current_real_training_rows(ledger_rows)
    validation = validation_rows(ledger_rows)
    baseline_paths = _real_round_paths(paths["models"], 0, "baseline")
    ensemble = train_ensemble(
        training_rows,
        validation,
        baseline_paths["ensemble"],
        config,
        cache_dir=paths["model_cache"],
    )
    metrics = _round_metrics(
        round_id=0,
        strategy="baseline",
        labeled_count=len(training_rows),
        metrics=evaluate_rows(
            ensemble,
            validation,
            use_calibration=config.use_calibrated_acquisition,
            threshold_strategy=config.binary_threshold_strategy,
            threshold_source="evaluation_dataset",
        ),
        extra={"stage": "baseline", "evaluation_dataset": "validation"},
    )
    _save_json(paths["metrics"] / "baseline_round_000.json", metrics)

    if config.train_family_for_init or config.real_strategy == "family_qbc":
        train_family(
            training_rows,
            validation,
            baseline_paths["family"],
            config,
            cache_dir=paths["model_cache"],
        )
    return run_dir


def run_replay(run_dir: Path, strategies: list[str] | None = None) -> dict[str, list[dict]]:
    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    train_rows = [row for row in ledger_rows if row["split"] == "train_pool" and row["label"]]
    seed_rows = replay_seed_rows(ledger_rows)
    hidden_rows = replay_hidden_rows(ledger_rows)
    validation = validation_rows(ledger_rows)

    results: dict[str, list[dict]] = {}
    for strategy in strategies or config.replay_strategies:
        current_labeled = [dict(row) for row in seed_rows]
        hidden_pool = {row["sequence"]: dict(row) for row in hidden_rows}
        strategy_metrics: list[dict] = []
        for round_id in range(config.max_rounds + 1):
            round_paths = _strategy_paths(paths["models"] / "replay", strategy, round_id)
            ensemble = train_ensemble(
                current_labeled,
                validation,
                round_paths["ensemble"],
                config,
                cache_dir=paths["model_cache"],
            )
            metrics = evaluate_rows(
                ensemble,
                validation,
                use_calibration=config.use_calibrated_acquisition,
                threshold_strategy=config.binary_threshold_strategy,
                threshold_source="evaluation_dataset",
            )
            strategy_metrics.append(
                _round_metrics(
                    round_id,
                    strategy,
                    len(current_labeled),
                    metrics,
                    extra={
                        "hidden_count": len(hidden_pool),
                        "evaluation_dataset": "validation",
                        "candidate_source": "hidden_experimental_pool",
                        "generator_objective": "none",
                    },
                )
            )
            if config.persist_replay_ledgers:
                replay_rows = _build_replay_snapshot(
                    train_rows,
                    {row["sequence"] for row in current_labeled},
                    strategy,
                    round_id,
                )
                save_ledger(round_paths["replay_ledger"], replay_rows)
            if round_id == config.max_rounds or not hidden_pool:
                break

            candidate_rows = list(hidden_pool.values())
            candidate_sequences = _rows_to_sequences(candidate_rows)
            need_embeddings = requires_embeddings(strategy)
            candidate_scores = score_sequences_with_ensemble(
                ensemble,
                candidate_sequences,
                include_embeddings=need_embeddings,
                use_calibration=config.use_calibrated_acquisition,
            )
            if requires_family_models(strategy):
                family = train_family(
                    current_labeled,
                    validation,
                    round_paths["family"],
                    config,
                    cache_dir=paths["model_cache"],
                )
                candidate_scores.update(
                    score_sequences_with_family(
                        family,
                        candidate_sequences,
                        use_calibration=config.use_calibrated_acquisition,
                    )
                )
            labeled_embeddings = (
                _labeled_embeddings(ensemble, current_labeled)
                if need_embeddings
                else np.empty((0, 0), dtype=float)
            )
            selection_seed = config.random_seed + round_id
            selected_indices, acquisition_scores = select_batch(
                strategy,
                config.batch_size,
                candidate_scores,
                labeled_embeddings,
                config,
                selection_seed,
                candidate_sequences=candidate_sequences,
                reference_sequences=_rows_to_sequences(current_labeled),
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
                reference_sequences=_rows_to_sequences(current_labeled),
            )
            strategy_metrics[-1].update(
                _selection_metadata_from_diagnostics(diagnostics, selected_indices)
            )
            trace_rows = _replay_selected_batch_rows(
                candidate_rows,
                selected_indices,
                candidate_scores,
                acquisition_scores,
                diagnostics,
                strategy,
                round_id,
            )
            _save_csv(
                paths["replay"] / strategy / f"round_{round_id:03d}_selected_batch.csv",
                REPLAY_SELECTED_BATCH_FIELDS,
                trace_rows,
            )
            for index in selected_indices:
                sequence = candidate_sequences[index]
                current_labeled.append(hidden_pool.pop(sequence))

            ensemble = None
            family = None
            candidate_scores = None
            labeled_embeddings = None
            _cleanup_tensorflow_runtime()

        strategy_dir = paths["replay"] / strategy
        _save_json(strategy_dir / "summary.json", strategy_metrics)
        _save_csv(strategy_dir / "learning_curve.csv", list(strategy_metrics[0]), strategy_metrics)
        results[strategy] = strategy_metrics
        _cleanup_tensorflow_runtime()
    return results


def propose_round(run_dir: Path, strategy: str | None = None) -> Path:
    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    chosen_strategy = strategy or config.real_strategy
    if unresolved_proposals(ledger_rows):
        raise ValueError("There is already a proposed batch waiting for CG-MD labels.")

    round_id = next_real_round_id(ledger_rows)
    training_rows = current_real_training_rows(ledger_rows)
    validation = validation_rows(ledger_rows)

    round_paths = _real_round_paths(paths["models"], round_id, "pre_proposal")
    ensemble = train_ensemble(
        training_rows,
        validation,
        round_paths["ensemble"],
        config,
        cache_dir=paths["model_cache"],
    )
    pre_metrics = _round_metrics(
        round_id,
        chosen_strategy,
        len(training_rows),
        evaluate_rows(
            ensemble,
            validation,
            use_calibration=config.use_calibrated_acquisition,
            threshold_strategy=config.binary_threshold_strategy,
            threshold_source="evaluation_dataset",
        ),
        extra={"stage": "pre_proposal", "evaluation_dataset": "validation"},
    )
    _save_json(round_paths["metrics"], pre_metrics)

    family = None
    generator_objective = generator_objective_for_strategy(
        chosen_strategy,
        config.generator_objective_mode,
        config,
    )
    if (
        requires_family_models(chosen_strategy)
        or generator_objective_requires_family_models(generator_objective)
    ):
        family = train_family(
            training_rows,
            validation,
            round_paths["family"],
            config,
            cache_dir=paths["model_cache"],
        )

    existing_sequences = {row["sequence"] for row in ledger_rows}
    generator_labeled_embeddings = (
        _labeled_embeddings(ensemble, training_rows)
        if generator_objective_requires_embeddings(generator_objective)
        else np.empty((0, 0), dtype=float)
    )
    candidate_sequences, generator_metadata = generate_candidate_sequences(
        ensemble,
        existing_sequences,
        config,
        min_unique=config.candidate_pool_min,
        seed_offset=round_id * 100,
        objective=generator_objective,
        family_models=family,
        labeled_embeddings=generator_labeled_embeddings,
        use_similarity_penalty=(
            True
            if chosen_strategy == "similarity_penalized_mean"
            else config.use_similarity_penalty
        ),
        use_length_penalty=config.use_length_penalty,
        return_metadata=True,
    )
    need_embeddings = requires_embeddings(chosen_strategy)
    candidate_scores = score_sequences_with_ensemble(
        ensemble,
        candidate_sequences,
        include_embeddings=need_embeddings,
        use_calibration=config.use_calibrated_acquisition,
        include_raw=True,
    )
    if family is not None:
        candidate_scores.update(
            score_sequences_with_family(
                family,
                candidate_sequences,
                use_calibration=config.use_calibrated_acquisition,
                include_raw=True,
            )
        )

    labeled_embeddings = (
        _labeled_embeddings(ensemble, training_rows)
        if need_embeddings
        else np.empty((0, 0), dtype=float)
    )
    selection_seed = config.random_seed + round_id
    selected_indices, acquisition_scores = select_batch(
        chosen_strategy,
        config.batch_size,
        candidate_scores,
        labeled_embeddings,
        config,
        selection_seed,
        candidate_sequences=candidate_sequences,
        reference_sequences=_rows_to_sequences(training_rows),
        apply_similarity_penalty=False
        if chosen_strategy == "similarity_penalized_mean"
        else True,
    )
    diagnostics = acquisition_diagnostics(
        chosen_strategy,
        selected_indices,
        candidate_scores,
        labeled_embeddings,
        acquisition_scores,
        config,
        selection_seed,
        candidate_sequences=candidate_sequences,
        reference_sequences=_rows_to_sequences(training_rows),
        apply_similarity_penalty=False
        if chosen_strategy == "similarity_penalized_mean"
        else True,
    )

    candidate_rows: list[dict[str, str]] = []
    for index, sequence in enumerate(candidate_sequences):
        generation_meta = generator_metadata.get(sequence, {})
        row = empty_row(
            {
                "sequence": sequence,
                "split": "generated",
                "mode": "real_al",
                "round_id": str(round_id),
                "status": "candidate_scored",
                "pred_mean": candidate_scores["pred_mean"][index],
                "pred_std": candidate_scores["pred_std"][index],
                "pred_entropy": candidate_scores["pred_entropy"][index],
                "pred_expected_entropy": candidate_scores["pred_expected_entropy"][index],
                "pred_mutual_information": candidate_scores["pred_mutual_information"][index],
                "raw_pred_mean": candidate_scores.get("raw_pred_mean", candidate_scores["pred_mean"])[index],
                "raw_pred_std": candidate_scores.get("raw_pred_std", candidate_scores["pred_std"])[index],
                "raw_pred_entropy": candidate_scores.get("raw_pred_entropy", candidate_scores["pred_entropy"])[index],
                "raw_pred_expected_entropy": candidate_scores.get(
                    "raw_pred_expected_entropy",
                    candidate_scores["pred_expected_entropy"],
                )[index],
                "raw_pred_mutual_information": candidate_scores.get(
                    "raw_pred_mutual_information",
                    candidate_scores["pred_mutual_information"],
                )[index],
                "acquisition_strategy": chosen_strategy,
                "acquisition_score": acquisition_scores[index],
                "generator_origin": f"ga_{generator_objective}",
                "candidate_source": "ga_generated_pool",
                "generator_objective": generation_meta.get(
                    "generator_objective",
                    generator_objective,
                ),
                "generator_subpool": generation_meta.get("generator_subpool", ""),
                "subpool_target": generation_meta.get("subpool_target", ""),
                "subpool_unique_count_after_dedup": generation_meta.get(
                    "subpool_unique_count_after_dedup",
                    "",
                ),
                "subpool_fill_count": generation_meta.get("subpool_fill_count", ""),
                "deduplicated_count": generation_meta.get("deduplicated_count", ""),
                "subpool_rank": generation_meta.get("subpool_rank", ""),
                "normalized_mi": generation_meta.get("normalized_mi", ""),
                "embedding_novelty_raw": generation_meta.get(
                    "embedding_novelty_raw",
                    "",
                ),
                "normalized_embedding_novelty": generation_meta.get(
                    "normalized_embedding_novelty",
                    "",
                ),
                "generator_utility_score": generation_meta.get(
                    "generator_utility_score",
                    "",
                ),
                "similarity_penalty": generation_meta.get(
                    "similarity_penalty",
                    "",
                ),
                "length_penalty": generation_meta.get("length_penalty", ""),
                "generator_fitness": generation_meta.get("generator_fitness", ""),
                "ensemble_member_probs": serialize_probabilities(
                    candidate_scores["ensemble_member_probs"][index]
                ),
                "raw_ensemble_member_probs": serialize_probabilities(
                    candidate_scores.get(
                        "raw_ensemble_member_probs",
                        candidate_scores["ensemble_member_probs"],
                    )[index]
                ),
            }
        )
        row.update(diagnostics[index])
        if "family_member_probs" in candidate_scores:
            row["family_member_probs"] = serialize_probabilities(
                candidate_scores["family_member_probs"][index]
            )
            row["raw_family_member_probs"] = serialize_probabilities(
                candidate_scores.get(
                    "raw_family_member_probs",
                    candidate_scores["family_member_probs"],
                )[index]
            )
            row["committee_vote_entropy"] = str(
                candidate_scores["committee_vote_entropy"][index]
            )
            row["committee_prob_std"] = str(
                candidate_scores["committee_prob_std"][index]
            )
        if index in selected_indices:
            row["status"] = "proposed"
        candidate_rows.append(row)

    append_rows(ledger_rows, candidate_rows)
    save_ledger(paths["ledger"], ledger_rows)
    snapshot_ledger(paths["snapshots"], ledger_rows, f"ledger_round_{round_id:03d}_proposed")

    scored_path = paths["candidates"] / f"round_{round_id:03d}_scored.csv"
    save_ledger(scored_path, candidate_rows)
    selected_rows = [candidate_rows[index] for index in selected_indices]
    batch_path = paths["batches"] / f"round_{round_id:03d}_batch.csv"
    _export_batch_csv(batch_path, selected_rows)
    _cleanup_tensorflow_runtime()
    return batch_path


def ingest_round(run_dir: Path, import_csv: Path) -> dict:
    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    round_id, imported_rows = _validate_cgmd_import(ledger_rows, import_csv)
    proposed_rows = [row for row in ledger_rows if row["status"] == "proposed"]
    strategy = proposed_rows[0]["acquisition_strategy"] if proposed_rows else config.real_strategy
    destination = paths["imports"] / f"round_{round_id:03d}_labels.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(import_csv, destination)

    ledger_index = index_by_sequence(ledger_rows)
    for row in imported_rows:
        ledger_row = ledger_index[row["sequence"]]
        ledger_row["label"] = row["cgmd_label"]
        ledger_row["label_source"] = "cgmd"
        ledger_row["status"] = "acquired"

    save_ledger(paths["ledger"], ledger_rows)
    snapshot_ledger(paths["snapshots"], ledger_rows, f"ledger_round_{round_id:03d}_acquired")

    training_rows = current_real_training_rows(ledger_rows)
    validation = validation_rows(ledger_rows)
    round_paths = _real_round_paths(paths["models"], round_id, "post_ingest")
    ensemble = train_ensemble(
        training_rows,
        validation,
        round_paths["ensemble"],
        config,
        cache_dir=paths["model_cache"],
    )
    metrics = _round_metrics(
        round_id,
        strategy,
        len(training_rows),
        evaluate_rows(
            ensemble,
            validation,
            use_calibration=config.use_calibrated_acquisition,
            threshold_strategy=config.binary_threshold_strategy,
            threshold_source="evaluation_dataset",
        ),
        extra={"stage": "post_ingest", "evaluation_dataset": "validation"},
    )
    _save_json(round_paths["metrics"], metrics)
    _cleanup_tensorflow_runtime()
    return metrics


def retrain_after_ingest(run_dir: Path, round_id: int) -> dict:
    """Retrain the final branch models after a fully ingested Real AL round."""

    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    if unresolved_proposals(ledger_rows):
        raise ValueError("Cannot finalize while proposed rows are unresolved.")
    expected_round = next_real_round_id(ledger_rows)
    if expected_round != round_id + 1:
        raise ValueError(
            f"Cannot finalize round {round_id}: branch ledger next round is {expected_round}."
        )

    training_rows = current_real_training_rows(ledger_rows)
    validation = validation_rows(ledger_rows)
    round_paths = _real_round_paths(paths["models"], round_id, "post_ingest")
    ensemble = train_ensemble(
        training_rows,
        validation,
        round_paths["ensemble"],
        config,
        cache_dir=paths["model_cache"],
    )
    metrics = _round_metrics(
        round_id,
        config.real_strategy,
        len(training_rows),
        evaluate_rows(
            ensemble,
            validation,
            use_calibration=config.use_calibrated_acquisition,
            threshold_strategy=config.binary_threshold_strategy,
            threshold_source="evaluation_dataset",
        ),
        extra={
            "stage": "post_ingest",
            "evaluation_dataset": "validation",
            "terminal_retrain": True,
        },
    )
    _save_json(round_paths["metrics"], metrics)

    if requires_family_models(config.real_strategy):
        train_family(
            training_rows,
            validation,
            round_paths["family"],
            config,
            cache_dir=paths["model_cache"],
        )

    _cleanup_tensorflow_runtime()
    return metrics


def evaluate_final(run_dir: Path) -> dict:
    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    training_rows = current_real_training_rows(ledger_rows)
    validation = validation_rows(ledger_rows)
    holdout = holdout_rows(ledger_rows)

    ensemble_dir, source_info = _resolve_discovery_ensemble_dir(paths["models"])
    ensemble = load_ensemble_from_dir(ensemble_dir, config)
    validation_metrics = evaluate_rows(
        ensemble,
        validation,
        use_calibration=config.use_calibrated_acquisition,
        threshold_strategy=config.binary_threshold_strategy,
        threshold_source="validation",
    )
    decision_threshold = float(validation_metrics.get("decision_threshold", 0.5))
    threshold_selection_f1 = float(
        validation_metrics.get("threshold_selection_f1", validation_metrics.get("f1", 0.0))
    )
    metrics = _round_metrics(
        round_id=int(source_info["surrogate_round_id"]),
        strategy="final_evaluation",
        labeled_count=len(training_rows),
        metrics=evaluate_holdout(
            ensemble,
            holdout,
            use_calibration=config.use_calibrated_acquisition,
            threshold_strategy=config.binary_threshold_strategy,
            threshold=decision_threshold,
            threshold_source="validation"
            if config.binary_threshold_strategy == "pr_best_f1"
            else "fixed_0_5",
            threshold_selection_f1=threshold_selection_f1,
        ),
        extra={
            "evaluation_dataset": "holdout",
            "surrogate_stage": source_info["surrogate_stage"],
            "surrogate_round_id": int(source_info["surrogate_round_id"]),
            "validation_decision_threshold": decision_threshold,
            "validation_threshold_selection_f1": threshold_selection_f1,
        },
    )
    _save_json(paths["metrics"] / "final_holdout.json", metrics)
    _cleanup_tensorflow_runtime()
    return metrics


def run_discovery(run_dir: Path, strategies: list[str] | None = None) -> dict[str, dict]:
    ensure_predictive_runtime()
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    ledger_rows = load_ledger(paths["ledger"])
    training_rows = current_real_training_rows(ledger_rows)
    existing_sequences = {row["sequence"] for row in ledger_rows}
    discovery_strategies = strategies or config.discovery_strategies

    ensemble_dir, source_info = _resolve_discovery_ensemble_dir(paths["models"])
    ensemble = load_ensemble_from_dir(ensemble_dir, config)

    labeled_embeddings = np.empty((0, 0), dtype=float)
    incumbent = 0.0
    if training_rows:
        training_scores = score_sequences_with_ensemble(
            ensemble,
            _rows_to_sequences(training_rows),
            include_embeddings=True,
            use_calibration=config.use_calibrated_acquisition,
        )
        labeled_embeddings = np.asarray(training_scores["avg_embedding"], dtype=float)
        incumbent = float(np.max(training_scores["pred_mean"]))

    results: dict[str, dict] = {}
    aggregate_rows: list[dict] = []
    export_count = min(config.discovery_export_count, config.candidate_pool_min)

    for strategy_index, strategy in enumerate(discovery_strategies):
        candidate_sequences, generator_metadata = generate_candidate_sequences(
            ensemble,
            existing_sequences,
            config,
            min_unique=max(config.candidate_pool_min, export_count),
            seed_offset=0,
            objective=strategy,
            incumbent=incumbent,
            minimum_return_count=export_count,
            labeled_embeddings=labeled_embeddings,
            use_similarity_penalty=config.use_similarity_penalty,
            use_length_penalty=config.use_length_penalty,
            return_metadata=True,
        )
        candidate_scores = score_sequences_with_ensemble(
            ensemble,
            candidate_sequences,
            include_embeddings=True,
            use_calibration=config.use_calibrated_acquisition,
            include_raw=True,
        )
        utility_scores = discovery_utility_scores(
            strategy,
            candidate_scores["pred_mean"],
            candidate_scores["pred_std"],
            incumbent,
            config,
            seed=config.random_seed + strategy_index * 1000,
        )
        nearest_labeled = min_distances_to_reference(
            np.asarray(candidate_scores["avg_embedding"], dtype=float),
            labeled_embeddings,
        )
        ordered_indices = list(np.argsort(-utility_scores, kind="mergesort"))

        candidate_rows: list[dict[str, str | float | int]] = []
        for index in ordered_indices:
            sequence = candidate_sequences[index]
            generation_meta = generator_metadata.get(sequence, {})
            candidate_rows.append(
                {
                    "sequence": sequence,
                    "strategy": strategy,
                    "pred_mean": float(candidate_scores["pred_mean"][index]),
                    "pred_std": float(candidate_scores["pred_std"][index]),
                    "raw_pred_mean": float(candidate_scores.get("raw_pred_mean", candidate_scores["pred_mean"])[index]),
                    "raw_pred_std": float(candidate_scores.get("raw_pred_std", candidate_scores["pred_std"])[index]),
                    "utility_score": float(utility_scores[index]),
                    "generator_objective": generation_meta.get(
                        "generator_objective",
                        strategy,
                    ),
                    "generator_utility_score": float(
                        generation_meta.get(
                            "generator_utility_score",
                            utility_scores[index],
                        )
                    ),
                    "similarity_penalty": float(
                        generation_meta.get("similarity_penalty", 0.0)
                    ),
                    "length_penalty": float(
                        generation_meta.get("length_penalty", 0.0)
                    ),
                    "generator_fitness": float(
                        generation_meta.get(
                            "generator_fitness",
                            utility_scores[index],
                        )
                    ),
                    "length": len(sequence),
                    "generator_origin": f"ga_discovery_{strategy}",
                    "nearest_labeled_distance": float(nearest_labeled[index]),
                }
            )

        top_rows = candidate_rows[: min(export_count, len(candidate_rows))]
        top_indices = ordered_indices[: len(top_rows)]
        top_embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)[top_indices]
        top_nearest = nearest_labeled[top_indices] if len(top_indices) else np.empty(0, dtype=float)
        summary = {
            "strategy": strategy,
            "surrogate_stage": source_info["surrogate_stage"],
            "surrogate_round_id": int(source_info["surrogate_round_id"]),
            "incumbent": incumbent,
            "unique_candidate_count": len(candidate_sequences),
            "exported_count": len(top_rows),
            "top_batch_mean_pred_mean": float(np.mean([row["pred_mean"] for row in top_rows])) if top_rows else 0.0,
            "top_batch_mean_pred_std": float(np.mean([row["pred_std"] for row in top_rows])) if top_rows else 0.0,
            "top_batch_mean_utility_score": float(np.mean([row["utility_score"] for row in top_rows])) if top_rows else 0.0,
            "top_batch_mean_generator_fitness": float(np.mean([row["generator_fitness"] for row in top_rows])) if top_rows else 0.0,
            "top_batch_mean_nearest_labeled_distance": float(np.mean(top_nearest)) if len(top_nearest) else 0.0,
            "top_batch_mean_pairwise_distance": mean_pairwise_distance(top_embeddings),
            "top_batch_sequences": [row["sequence"] for row in top_rows],
        }

        strategy_paths = _discovery_strategy_paths(paths["discovery"], strategy)
        candidate_fields = [
            "sequence",
            "strategy",
            "pred_mean",
            "pred_std",
            "raw_pred_mean",
            "raw_pred_std",
            "utility_score",
            "generator_objective",
            "generator_utility_score",
            "similarity_penalty",
            "length_penalty",
            "generator_fitness",
            "length",
            "generator_origin",
            "nearest_labeled_distance",
        ]
        _save_json(strategy_paths["summary"], summary)
        _save_csv(strategy_paths["candidates"], candidate_fields, candidate_rows)
        _save_csv(strategy_paths["top_batch"], candidate_fields, top_rows)
        aggregate_rows.append(summary)
        results[strategy] = summary

    if aggregate_rows:
        aggregate_fields = list(aggregate_rows[0])
        _save_csv(paths["discovery"] / "aggregate_summary.csv", aggregate_fields, aggregate_rows)

    _cleanup_tensorflow_runtime()
    return results


def load_run(run_dir: Path) -> tuple[RunConfig, dict, list[dict[str, str]]]:
    paths = _run_paths(run_dir)
    config = RunConfig.load(paths["config"])
    manifest = load_split_manifest(paths["split_manifest"])
    ledger_rows = load_ledger(paths["ledger"])
    return config, manifest, ledger_rows
