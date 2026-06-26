from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid
from typing import Iterable

import numpy as np

from active_learning_thesis.acquisition import _descending_indices, select_batch
from active_learning_thesis.generative import (
    calculate_amino_acid_frequencies,
    generate_candidate_sequences,
)
from active_learning_thesis.phase3_strategy_selection import normalized_edit_distance
from active_learning_thesis.phase4_bo import (
    CANDIDATE_FIELDS,
    MODEL_GUIDED_POLICIES,
    PHASE4_POLICIES,
    _canonical_sequence,
    _final_candidate_rows,
    _load_verified_ensemble,
    _pbs_repo_root,
    _phase4_utility_callback,
    _predictive_run_config,
    _proposal_exclusions,
    _random_audit_key,
    _read_csv,
    _read_json,
    _sha256_file,
    _target_path,
    _write_csv,
    _write_json,
)


PHASE4D_DEFAULT_SEEDS = {
    "random": 20270417,
    "greedy": 20270517,
    "ucb": 20270617,
    "pi": 20270717,
    "ei": 20270817,
    "mes": 20270917,
}
PHASE4D_SCHEMA_VERSION = 1
PHASE4D_REQUIRED_POOL_SIZE = 50
PHASE4D_MAX_ATTEMPTS = 100

PHASE4D_SELECTION_FIELDS = [
    *CANDIDATE_FIELDS,
    "policy",
    "generation_seed",
    "original_final_acquisition_utility",
    "original_final_utility_rank",
    "phase4d_similarity_aware_applicable",
    "phase4d_final_similarity_penalty",
    "phase4d_selection_score",
    "selection_step",
    "nearest_selected_sequence",
    "pair_similarity_to_nearest_selected",
]

PHASE4D_TRACE_FIELDS = [
    "policy",
    "selection_step",
    "sequence",
    "retained_pool_input_index",
    "selected_set_before_step",
    "original_final_acquisition_utility",
    "original_final_utility_rank",
    "phase4d_final_similarity_penalty",
    "phase4d_selection_score",
    "nearest_selected_sequence",
    "pair_similarity_to_nearest_selected",
    "selected_at_step",
]

_TEMP_SUFFIXES = (".tmp", ".temp", ".partial", ".part")
_LOCK_SUFFIXES = (".lock", ".lck")
_FILESYSTEM_METADATA = {".DS_Store", "Thumbs.db", "desktop.ini"}


def init_phase4_diversity(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    source_round = int(args.source_round)
    run_id = int(args.phase4d_run)
    if source_round != 1:
        raise ValueError("Phase 4-D currently requires completed primary source round 1.")
    if run_id <= 0:
        raise ValueError("phase4d-run must be positive.")

    config = _read_json(output_root / "config.json")
    _verify_frozen_phase4_state(output_root, config)
    run_root = _phase4d_run_root(output_root, run_id)
    if run_root.exists():
        if not bool(getattr(args, "force", False)):
            raise FileExistsError(f"Phase 4-D run already exists: {run_root}")
        _archive_phase4d_run(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    seeds, seed_source = _load_seed_config(getattr(args, "seed_config", None))
    _validate_seed_blocks(seeds)
    initialized_at = _now_iso()
    protected = _primary_artifact_manifest(output_root, initialized_at)
    _write_json(run_root / "primary_phase4_checksums_before.json", protected)
    _write_json(
        run_root / "frozen_phase4_source_checksums.json",
        _frozen_state_manifest(output_root),
    )
    _write_json(
        run_root / "phase4d_seeds_resolved.json",
        {
            "schema_version": PHASE4D_SCHEMA_VERSION,
            "source": seed_source,
            "attempt_index_rule": "0..99",
            "attempt_seed_rule": "policy_base_seed + attempt_index",
            "policy_base_seeds": seeds,
        },
    )

    timing = _derive_phase4d_walltime(
        output_root,
        getattr(args, "phase4d_walltime", None),
    )
    _write_json(run_root / "runtime_audit.json", timing)
    for policy in PHASE4_POLICIES:
        _write_json(
            run_root / "policies" / policy / "status.json",
            {
                "status": "preview_ready",
                "policy": policy,
                "phase4d_run": run_id,
                "policy_base_seed": seeds[policy],
            },
        )

    pbs_path = _write_phase4d_pbs(
        output_root,
        run_id,
        getattr(args, "pbs_repo_root", None),
        timing["pbs_walltime"],
        getattr(args, "seed_config", None),
    )
    manifest = {
        "phase": "phase4d_diversity_aware_generative_replicate",
        "schema_version": PHASE4D_SCHEMA_VERSION,
        "status": "initialized",
        "initialized_at": initialized_at,
        "source_round": source_round,
        "phase4d_run": run_id,
        "primary_phase4_role": "immutable_primary_acquisition_policy_comparison",
        "phase4d_role": "secondary_operational_diversity_aware_selector",
        "policies": list(PHASE4_POLICIES),
        "guided_policies": list(MODEL_GUIDED_POLICIES),
        "candidate_pool_target": int(config["candidate_pool_target"]),
        "classification_threshold_usage": "reporting_only_not_used",
        "holdout_usage": "sequence_identity_only_for_exact_duplicate_exclusion",
        "automatic_submission": False,
        "automatic_simulation": False,
        "pbs_preview": str(pbs_path),
    }
    _write_json(run_root / "phase4d_manifest.json", manifest)
    return {
        "status": "initialized",
        "run_root": str(run_root),
        "policy_base_seeds": seeds,
        "pbs_preview": str(pbs_path),
        "pbs_walltime": timing["pbs_walltime"],
    }


def run_phase4_diversity(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    if int(args.source_round) != 1:
        raise ValueError("Phase 4-D currently requires completed primary source round 1.")
    run_id = int(args.phase4d_run)
    run_root = _phase4d_run_root(output_root, run_id)
    if not run_root.exists():
        raise FileNotFoundError(
            "Phase 4-D is not initialized. Run init-diversity-aware first."
        )
    config = _read_json(output_root / "config.json")
    _verify_frozen_snapshot(
        output_root,
        _read_json(run_root / "frozen_phase4_source_checksums.json"),
    )
    ensemble, model_context = _verify_frozen_phase4_state(output_root, config)
    seed_payload = _read_json(run_root / "phase4d_seeds_resolved.json")
    seeds = {key: int(value) for key, value in seed_payload["policy_base_seeds"].items()}
    if getattr(args, "seed_config", None):
        supplied, _ = _load_seed_config(args.seed_config)
        if supplied != seeds:
            raise ValueError(
                "The supplied Phase 4-D seed config differs from the initialized run."
            )
    _validate_seed_blocks(seeds)

    protected_before = _read_json(run_root / "primary_phase4_checksums_before.json")
    failures: dict[str, str] = {}
    results: dict[str, dict[str, object]] = {}
    for policy in PHASE4_POLICIES:
        policy_dir = run_root / "policies" / policy
        if _policy_output_is_valid(
            policy_dir, policy, seeds[policy], config, output_root
        ):
            results[policy] = {"status": "skipped_valid", "policy": policy}
            continue
        if policy_dir.exists():
            _archive_invalid_policy(run_root, policy_dir, policy)
        try:
            results[policy] = _run_phase4d_policy(
                output_root,
                run_root,
                policy,
                seeds[policy],
                config,
                ensemble,
                model_context,
            )
        except Exception as exc:
            failures[policy] = str(exc)
            _write_json(
                run_root / "policies" / policy / "status.json",
                {
                    "status": "failed",
                    "policy": policy,
                    "phase4d_run": run_id,
                    "error": str(exc),
                },
            )

    _write_run_comparison(run_root)
    protected_after = _verify_primary_artifacts_unchanged(
        output_root,
        protected_before,
    )
    _write_json(run_root / "primary_phase4_checksums_after.json", protected_after)
    manifest_path = run_root / "phase4d_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "status": "failed" if failures else "completed",
            "completed_at": _now_iso(),
            "policy_results": results,
            "policy_failures": failures,
            "primary_artifact_verification": "passed",
        }
    )
    _write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(
            "Phase 4-D incomplete policies: "
            + ", ".join(f"{key}: {value}" for key, value in failures.items())
        )
    return {
        "status": "completed",
        "run_root": str(run_root),
        "policies": {policy: "completed" for policy in PHASE4_POLICIES},
    }


def phase4d_composition_similarity(
    left: str,
    right: str,
    allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
) -> float:
    left_counts = calculate_amino_acid_frequencies(left, allowed_amino_acids)
    right_counts = calculate_amino_acid_frequencies(right, allowed_amino_acids)
    denominator = len(left) + len(right)
    if denominator <= 0:
        return 0.0
    return float(
        0.1
        * (
            1.0
            - np.abs(left_counts - right_counts).sum() / float(denominator)
        )
    )


def phase4d_similarity_aware_selection(
    rows: list[dict[str, object]],
    batch_size: int = 5,
    allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
) -> tuple[list[int], list[dict[str, object]]]:
    if len(rows) < batch_size:
        raise ValueError("The retained pool is smaller than the requested batch.")
    utilities = np.asarray(
        [float(row["final_acquisition_utility"]) for row in rows],
        dtype=float,
    )
    if not np.isfinite(utilities).all():
        raise ValueError("Guided final acquisition utilities must be finite.")
    utility_order = _descending_indices(utilities)
    utility_rank = {index: rank for rank, index in enumerate(utility_order, start=1)}
    selected: list[int] = []
    trace: list[dict[str, object]] = []
    available = set(range(len(rows)))

    for step in range(1, batch_size + 1):
        selected_sequences = [str(rows[index]["sequence"]) for index in selected]
        scored: list[tuple[float, float, float, int, str, float]] = []
        for index in sorted(available):
            sequence = str(rows[index]["sequence"])
            pair_values = [
                (other, phase4d_composition_similarity(
                    sequence,
                    other,
                    allowed_amino_acids,
                ))
                for other in selected_sequences
            ]
            penalty = (
                float(np.mean([value for _, value in pair_values]))
                if pair_values
                else 0.0
            )
            nearest_sequence, nearest_similarity = (
                max(pair_values, key=lambda item: item[1])
                if pair_values
                else ("", 0.0)
            )
            utility = float(utilities[index])
            score = utility - penalty
            scored.append(
                (score, utility, penalty, index, nearest_sequence, nearest_similarity)
            )
        chosen = max(
            scored,
            key=lambda item: (item[0], item[1], -item[2], -item[3]),
        )
        chosen_index = chosen[3]
        selected_before = ";".join(selected_sequences)
        for score, utility, penalty, index, nearest, nearest_similarity in scored:
            trace.append(
                {
                    "selection_step": step,
                    "sequence": rows[index]["sequence"],
                    "retained_pool_input_index": index,
                    "selected_set_before_step": selected_before,
                    "original_final_acquisition_utility": utility,
                    "original_final_utility_rank": utility_rank[index],
                    "phase4d_final_similarity_penalty": penalty,
                    "phase4d_selection_score": score,
                    "nearest_selected_sequence": nearest,
                    "pair_similarity_to_nearest_selected": nearest_similarity,
                    "selected_at_step": index == chosen_index,
                }
            )
        selected.append(chosen_index)
        available.remove(chosen_index)
    return selected, trace


def _run_phase4d_policy(
    output_root: Path,
    run_root: Path,
    policy: str,
    base_seed: int,
    config: dict[str, object],
    ensemble,
    model_context: dict[str, object],
) -> dict[str, object]:
    started_at = _now_iso()
    started_clock = time.monotonic()
    temp_dir = run_root / f".tmp_{policy}_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    attempt_history: list[dict[str, object]] = []
    try:
        run_config = replace(_predictive_run_config(config), random_seed=base_seed)
        exclusions = _proposal_exclusions(output_root)
        utility_callback = _phase4_utility_callback(
            policy,
            None if policy == "random" else ensemble,
            None if policy == "random" else model_context,
            config,
        )
        sequences, metadata = generate_candidate_sequences(
            None if policy == "random" else ensemble,
            exclusions,
            run_config,
            min_unique=int(config["candidate_pool_target"]),
            seed_offset=0,
            objective="broad_pool" if policy == "random" else policy,
            minimum_return_count=int(config["batch_size"]),
            use_similarity_penalty=bool(config["use_similarity_penalty"]),
            use_length_penalty=bool(config["use_length_penalty"]),
            return_metadata=True,
            policy_utility_callback=utility_callback,
            attempt_history=attempt_history,
        )
        candidate_rows = _final_candidate_rows(
            policy,
            sequences,
            metadata,
            None if policy == "random" else ensemble,
            None if policy == "random" else model_context,
            config,
        )
        for row in candidate_rows:
            row["policy"] = policy
            row["generation_seed"] = base_seed
            row["phase4d_similarity_aware_applicable"] = policy != "random"

        _write_csv(temp_dir / "new_candidate_pool.csv", candidate_rows)
        _write_csv(temp_dir / "new_generation_log.csv", _generation_rows(metadata))
        attempt_manifest = _attempt_manifest(
            policy,
            base_seed,
            attempt_history,
            len(candidate_rows),
            started_at,
            started_clock,
        )
        _write_json(temp_dir / "new_ga_manifest.json", attempt_manifest)

        if policy == "random":
            selection_seed = base_seed + 1_000_000
            indices, _ = select_batch(
                "random",
                int(config["batch_size"]),
                {"pred_mean": np.zeros(len(candidate_rows), dtype=float)},
                np.empty((0, 0), dtype=float),
                run_config,
                selection_seed,
            )
            selected = []
            for rank, index in enumerate(indices, start=1):
                row = dict(candidate_rows[index])
                row.update(
                    {
                        "selection_rank": rank,
                        "random_shuffle_audit_key": _random_audit_key(
                            selection_seed, str(row["sequence"])
                        ),
                        "final_acquisition_utility": None,
                        "final_acquisition_utility_applicable": False,
                        "phase4d_similarity_aware_applicable": False,
                    }
                )
                selected.append(row)
            _write_csv(temp_dir / "random_selected_batch.csv", selected)
            attempt_manifest["selection_seed"] = selection_seed
            _write_json(temp_dir / "new_ga_manifest.json", attempt_manifest)
        else:
            utilities = np.asarray(
                [float(row["final_acquisition_utility"]) for row in candidate_rows]
            )
            utility_order = _descending_indices(utilities)
            utility_rank = {
                index: rank for rank, index in enumerate(utility_order, start=1)
            }
            utility_selected = []
            for rank, index in enumerate(utility_order[: int(config["batch_size"])], start=1):
                row = _phase4d_selected_row(
                    candidate_rows[index],
                    policy,
                    base_seed,
                    utility_rank[index],
                    rank,
                    0.0,
                    float(utilities[index]),
                    "",
                    0.0,
                )
                utility_selected.append(row)
            diversity_indices, trace = phase4d_similarity_aware_selection(
                candidate_rows,
                int(config["batch_size"]),
                str(config["allowed_amino_acids"]),
            )
            trace_by_sequence = {
                (int(row["selection_step"]), str(row["sequence"])): row
                for row in trace
                if bool(row["selected_at_step"])
            }
            diversity_selected = []
            for step, index in enumerate(diversity_indices, start=1):
                trace_row = trace_by_sequence[(step, str(candidate_rows[index]["sequence"]))]
                diversity_selected.append(
                    _phase4d_selected_row(
                        candidate_rows[index],
                        policy,
                        base_seed,
                        utility_rank[index],
                        step,
                        float(trace_row["phase4d_final_similarity_penalty"]),
                        float(trace_row["phase4d_selection_score"]),
                        str(trace_row["nearest_selected_sequence"]),
                        float(trace_row["pair_similarity_to_nearest_selected"]),
                    )
                )
            _write_csv(
                temp_dir / "utility_only_selected_batch.csv",
                utility_selected,
                PHASE4D_SELECTION_FIELDS,
            )
            _write_csv(
                temp_dir / "similarity_aware_selected_batch.csv",
                diversity_selected,
                PHASE4D_SELECTION_FIELDS,
            )
            _write_csv(
                temp_dir / "similarity_aware_selection_trace.csv",
                [{"policy": policy, **row} for row in trace],
                PHASE4D_TRACE_FIELDS,
            )
            comparison = _guided_policy_comparison(
                policy,
                utility_selected,
                diversity_selected,
                trace,
            )
            _write_csv(
                temp_dir / "utility_only_vs_similarity_aware_comparison.csv",
                [comparison],
            )

        status = {
            "status": "completed",
            "policy": policy,
            "phase4d_run": int(run_root.name.split("_")[-1]),
            "policy_base_seed": base_seed,
            "started_at": started_at,
            "completed_at": _now_iso(),
            "elapsed_seconds": time.monotonic() - started_clock,
            "retained_pool_size": len(candidate_rows),
        }
        _write_json(temp_dir / "status.json", status)
        _write_json(temp_dir / "checksums.json", _directory_checksums(temp_dir))
        _validate_policy_directory(
            temp_dir,
            policy,
            base_seed,
            config,
            output_root=output_root,
        )
        final_dir = run_root / "policies" / policy
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        return status
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def _phase4d_selected_row(
    source: dict[str, object],
    policy: str,
    generation_seed: int,
    utility_rank: int,
    step: int,
    penalty: float,
    score: float,
    nearest: str,
    nearest_similarity: float,
) -> dict[str, object]:
    row = dict(source)
    row.update(
        {
            "policy": policy,
            "generation_seed": generation_seed,
            "original_final_acquisition_utility": float(
                source["final_acquisition_utility"]
            ),
            "original_final_utility_rank": utility_rank,
            "phase4d_similarity_aware_applicable": True,
            "phase4d_final_similarity_penalty": penalty,
            "phase4d_selection_score": score,
            "selection_step": step,
            "selection_rank": step,
            "nearest_selected_sequence": nearest,
            "pair_similarity_to_nearest_selected": nearest_similarity,
        }
    )
    return row


def _generation_rows(
    metadata: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {"sequence": sequence, **values}
        for sequence, values in metadata.items()
    ]


def _attempt_manifest(
    policy: str,
    base_seed: int,
    history: list[dict[str, object]],
    retained_pool_size: int,
    started_at: str,
    started_clock: float,
) -> dict[str, object]:
    payload = {
        "policy": policy,
        "policy_base_seed": base_seed,
        "attempt_count": len(history),
        "attempt_indices_used": [row["attempt_index"] for row in history],
        "attempt_seeds_used": [row["attempt_seed"] for row in history],
        "candidates_generated_per_attempt": [
            row["candidates_generated"] for row in history
        ],
        "candidates_accepted_per_attempt": [
            row["candidates_accepted"] for row in history
        ],
        "terminal_attempt_index": history[-1]["attempt_index"] if history else None,
        "terminal_attempt_seed": history[-1]["attempt_seed"] if history else None,
        "retained_pool_size": retained_pool_size,
        "attempt_history": history,
        "started_at": started_at,
        "completed_at": _now_iso(),
        "elapsed_seconds": time.monotonic() - started_clock,
    }
    if len(history) == 1 and int(history[0]["candidates_accepted"]) == retained_pool_size:
        payload["successful_generation_seed"] = history[0]["attempt_seed"]
    return payload


def _batch_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    sequences = [str(row["sequence"]) for row in rows]
    pairs = [
        (left, right)
        for index, left in enumerate(sequences)
        for right in sequences[index + 1 :]
    ]
    edit = [normalized_edit_distance(left, right) for left, right in pairs]
    composition = [
        phase4d_composition_similarity(left, right) for left, right in pairs
    ]
    lengths = [len(sequence) for sequence in sequences]
    utilities = [
        float(row["original_final_acquisition_utility"])
        for row in rows
        if row.get("original_final_acquisition_utility") not in ("", None)
    ]
    return {
        "mean_pairwise_normalized_levenshtein_distance": float(np.mean(edit)),
        "minimum_pairwise_normalized_levenshtein_distance": float(np.min(edit)),
        "mean_pairwise_composition_similarity": float(np.mean(composition)),
        "maximum_pairwise_composition_similarity": float(np.max(composition)),
        "mean_sequence_length": float(np.mean(lengths)),
        "minimum_sequence_length": min(lengths),
        "maximum_sequence_length": max(lengths),
        "unique_sequence_lengths": len(set(lengths)),
        "mean_acquisition_utility": float(np.mean(utilities)),
        "minimum_acquisition_utility": float(np.min(utilities)),
        "sum_acquisition_utility": float(np.sum(utilities)),
    }


def _guided_policy_comparison(
    policy: str,
    utility_rows: list[dict[str, object]],
    diversity_rows: list[dict[str, object]],
    trace: list[dict[str, object]],
) -> dict[str, object]:
    original = _batch_metrics(utility_rows)
    diversity = _batch_metrics(diversity_rows)
    original_sequences = {str(row["sequence"]) for row in utility_rows}
    diversity_sequences = {str(row["sequence"]) for row in diversity_rows}
    later_penalties = [
        float(row["phase4d_final_similarity_penalty"])
        for row in diversity_rows
        if int(row["selection_step"]) >= 2
    ]
    all_penalties = [
        float(row["phase4d_final_similarity_penalty"]) for row in trace
    ]
    utilities = [
        float(row["original_final_acquisition_utility"]) for row in trace
    ]
    return {
        "policy": policy,
        **{f"utility_only_{key}": value for key, value in original.items()},
        **{f"similarity_aware_{key}": value for key, value in diversity.items()},
        "delta_mean_pairwise_normalized_levenshtein_distance": (
            diversity["mean_pairwise_normalized_levenshtein_distance"]
            - original["mean_pairwise_normalized_levenshtein_distance"]
        ),
        "delta_mean_acquisition_utility": (
            diversity["mean_acquisition_utility"]
            - original["mean_acquisition_utility"]
        ),
        "jaccard_overlap": len(original_sequences & diversity_sequences)
        / len(original_sequences | diversity_sequences),
        "selected_utility_ranks": ";".join(
            str(row["original_final_utility_rank"]) for row in diversity_rows
        ),
        "first_step_similarity_penalty": float(
            diversity_rows[0]["phase4d_final_similarity_penalty"]
        ),
        "mean_similarity_penalty_steps_2_to_5": float(np.mean(later_penalties)),
        "minimum_similarity_penalty_steps_2_to_5": float(np.min(later_penalties)),
        "maximum_similarity_penalty_steps_2_to_5": float(np.max(later_penalties)),
        "minimum_encountered_final_similarity_penalty": float(np.min(all_penalties)),
        "maximum_encountered_final_similarity_penalty": float(np.max(all_penalties)),
        "minimum_retained_pool_acquisition_utility": float(np.min(utilities)),
        "maximum_retained_pool_acquisition_utility": float(np.max(utilities)),
        "penalty_exceeds_utility_count": sum(
            float(row["phase4d_final_similarity_penalty"])
            > float(row["original_final_acquisition_utility"])
            for row in trace
        ),
    }


def _write_run_comparison(run_root: Path) -> None:
    guided_rows = []
    random_rows = []
    status_rows = []
    for policy in PHASE4_POLICIES:
        policy_dir = run_root / "policies" / policy
        status_path = policy_dir / "status.json"
        status = _read_json(status_path) if status_path.exists() else {
            "status": "missing",
            "policy": policy,
        }
        status_rows.append(status)
        if status.get("status") != "completed":
            continue
        if policy == "random":
            random_rows.extend(_read_csv(policy_dir / "random_selected_batch.csv"))
        else:
            comparison = _read_csv(
                policy_dir / "utility_only_vs_similarity_aware_comparison.csv"
            )
            guided_rows.extend(comparison)
    _write_csv(run_root / "all_policy_tradeoffs.csv", guided_rows)
    _write_csv(run_root / "random_descriptive_summary.csv", random_rows)
    _write_csv(
        run_root / "manual_review_recommendations.csv",
        [
            {
                "policy": policy,
                "recommended_simulation_batch": "manual_review",
                "recommendation_reason": (
                    "diversity--utility trade-off requires scientific review"
                ),
            }
            for policy in PHASE4_POLICIES
        ],
    )
    _write_csv(run_root / "policy_status.csv", status_rows)
    report = [
        "# Phase 4-D diversity-aware generative replicate",
        "",
        "Phase 4-D is a secondary operational analysis. The primary Phase 4 "
        "utility-only comparison remains unchanged.",
        "",
        "Phase 4-D is an operational diversity-aware selector. It intentionally "
        "trades acquisition utility against the inherited amino-acid-composition "
        "similarity penalty and is therefore not a pure acquisition-utility ranking.",
        "",
        "The validation-selected classification threshold was not used for "
        "generation, acquisition, ranking, or final selection. Frozen-holdout "
        "sequence identities were used only for exact duplicate exclusion; no "
        "holdout sequence was scored.",
        "",
        "Utility-only and similarity-aware guided batches use the same regenerated "
        "retained pool and the same frozen final utilities. Comparisons with primary "
        "Phase 4 are descriptive because generation seeds and retained pools differ.",
    ]
    (run_root / "phase4d_report.md").write_text("\n".join(report), encoding="utf-8")
    _write_tradeoff_figures(run_root, guided_rows)


def _write_tradeoff_figures(
    run_root: Path,
    rows: list[dict[str, str]],
) -> None:
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    labels = [row["policy"] for row in rows]
    diversity = [
        float(row["delta_mean_pairwise_normalized_levenshtein_distance"])
        for row in rows
    ]
    utility = [float(row["delta_mean_acquisition_utility"]) for row in rows]
    _simple_svg_bar(
        figures / "phase4d_diversity_change.svg",
        labels,
        diversity,
        "Phase 4-D diversity change",
    )
    _simple_svg_bar(
        figures / "phase4d_utility_tradeoff.svg",
        labels,
        utility,
        "Phase 4-D mean utility change",
    )
    try:
        import matplotlib.pyplot as plt

        for filename, values, title in (
            ("phase4d_diversity_change.png", diversity, "Phase 4-D diversity change"),
            ("phase4d_utility_tradeoff.png", utility, "Phase 4-D mean utility change"),
        ):
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.bar(labels, values)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_title(title)
            figure.tight_layout()
            figure.savefig(figures / filename, dpi=180)
            plt.close(figure)
    except ImportError:
        pass


def _simple_svg_bar(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
) -> None:
    width, height = 720, 420
    margin = 70
    maximum = max(max(abs(value) for value in values), 1e-12)
    zero_y = height // 2
    bar_width = (width - 2 * margin) / max(len(values), 1)
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * bar_width + bar_width * 0.15
        scaled = value / maximum * (height * 0.32)
        y = zero_y - max(scaled, 0)
        bar_height = abs(scaled)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.7:.1f}" '
            f'height="{bar_height:.1f}" fill="#277da1"/>'
        )
        bars.append(
            f'<text x="{x + bar_width * 0.35:.1f}" y="{height - 35}" '
            f'text-anchor="middle" font-size="13">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>'
        f'<line x1="{margin}" x2="{width-margin}" y1="{zero_y}" y2="{zero_y}" '
        f'stroke="black"/>{"".join(bars)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _verify_frozen_phase4_state(
    output_root: Path,
    config: dict[str, object],
):
    ensemble, context = _load_verified_ensemble(output_root, config)
    model_root = output_root / "models" / "ap_sp_fixed_split_ensemble"
    required = [
        model_root / "member_calibrations.json",
        model_root / "training_incumbent.json",
        model_root / "training_predictions.csv",
        model_root / "validation_calibration_report.json",
        output_root / "proposal_exclusions.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen Phase 4 artifacts: " + ", ".join(missing))
    incumbent = context["incumbent"]
    if int(incumbent.get("training_row_count", 0)) != 235:
        raise ValueError("Frozen Phase 4 incumbent is not the approved 235-row incumbent.")
    manifest = _read_json(model_root / "model_manifest.json")
    for source in manifest.get("source_configuration", {}).values():
        path = Path(str(source["path"]))
        if not path.exists() or _sha256_file(path) != source["sha256"]:
            raise ValueError(f"Frozen Phase 4 source checksum mismatch: {path}")
    return ensemble, context


def _frozen_state_manifest(output_root: Path) -> dict[str, object]:
    model_root = output_root / "models" / "ap_sp_fixed_split_ensemble"
    paths = [
        output_root / "config.json",
        output_root / "proposal_exclusions.json",
        output_root / "training_data_manifest.csv",
        model_root / "model_manifest.json",
        model_root / "member_calibrations.json",
        model_root / "training_incumbent.json",
        model_root / "training_predictions.csv",
        model_root / "validation_calibration_report.json",
        *sorted((model_root / "ensemble").glob("ap_sp_member_*.h5")),
        *sorted((model_root / "ensemble").glob("ap_sp_member_*.h5.meta.json")),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Frozen Phase 4 checksum sources are missing: " + ", ".join(missing)
        )
    return {
        "hash_algorithm": "sha256",
        "files": {
            path.relative_to(output_root).as_posix(): _sha256_file(path)
            for path in paths
        },
    }


def _verify_frozen_snapshot(
    output_root: Path,
    snapshot: dict[str, object],
) -> None:
    changed = []
    missing = []
    for relative, expected in snapshot["files"].items():
        path = output_root / relative
        if not path.exists():
            missing.append(relative)
        elif _sha256_file(path) != expected:
            changed.append(relative)
    if missing or changed:
        raise RuntimeError(
            f"Frozen Phase 4 state changed: missing={missing}, changed={changed}"
        )


def _load_seed_config(path_value: str | None) -> tuple[dict[str, int], str]:
    if path_value:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Phase 4-D seed config does not exist: {path}")
        payload = _read_json(path)
        raw = payload.get("policy_base_seeds", payload)
        return {policy: int(raw[policy]) for policy in PHASE4_POLICIES}, str(path)
    return dict(PHASE4D_DEFAULT_SEEDS), "built_in_approved_defaults"


def _validate_seed_blocks(seeds: dict[str, int]) -> None:
    if set(seeds) != set(PHASE4_POLICIES):
        raise ValueError("Phase 4-D seed config must define all six policies.")
    ranges = {
        policy: set(range(seed, seed + PHASE4D_MAX_ATTEMPTS))
        for policy, seed in seeds.items()
    }
    for index, left in enumerate(PHASE4_POLICIES):
        for right in PHASE4_POLICIES[index + 1 :]:
            if ranges[left] & ranges[right]:
                raise ValueError(f"Phase 4-D retry seed blocks overlap: {left}, {right}")


def _phase4d_run_root(output_root: Path, run_id: int) -> Path:
    return output_root / "phase4d" / f"run_{run_id:03d}"


def _archive_phase4d_run(run_root: Path) -> None:
    archive_name = run_root.parent / (
        f"{run_root.name}_archive_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.make_archive(str(archive_name), "zip", root_dir=run_root)
    shutil.rmtree(run_root)


def _archive_invalid_policy(run_root: Path, policy_dir: Path, policy: str) -> None:
    archive_root = run_root / "invalid_policy_outputs"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{policy}_{time.strftime('%Y%m%d_%H%M%S')}"
    if destination.exists():
        destination = archive_root / f"{destination.name}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(policy_dir), str(destination))


def _directory_checksums(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def _policy_output_is_valid(
    policy_dir: Path,
    policy: str,
    seed: int,
    config: dict[str, object],
    output_root: Path | None = None,
) -> bool:
    try:
        _validate_policy_directory(
            policy_dir, policy, seed, config, output_root=output_root
        )
        return True
    except Exception:
        return False


def _validate_policy_directory(
    policy_dir: Path,
    policy: str,
    seed: int,
    config: dict[str, object],
    *,
    output_root: Path | None = None,
) -> None:
    status = _read_json(policy_dir / "status.json")
    if status.get("status") != "completed":
        raise ValueError("Policy status is not completed.")
    if int(status.get("policy_base_seed", -1)) != seed:
        raise ValueError("Policy base seed mismatch.")
    pool = _read_csv(policy_dir / "new_candidate_pool.csv")
    if len(pool) != int(config["candidate_pool_target"]):
        raise ValueError("Retained-pool size mismatch.")
    allowed = set(str(config["allowed_amino_acids"]))
    sequences = [row["sequence"] for row in pool]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Retained pool contains duplicates.")
    if any(not set(sequence) <= allowed or not 3 <= len(sequence) <= 24 for sequence in sequences):
        raise ValueError("Retained pool contains invalid peptide sequences.")
    if output_root is not None:
        excluded = _proposal_exclusions(output_root)
        overlap = sorted(set(sequences) & excluded)
        if overlap:
            raise ValueError(
                "Retained pool violates training/validation/holdout/Phase 3 exclusions: "
                + ", ".join(overlap[:5])
            )
    if policy == "random":
        selected = _read_csv(policy_dir / "random_selected_batch.csv")
        if len(selected) != 5:
            raise ValueError("Random selected batch must contain five peptides.")
        for row in selected:
            if row.get("final_acquisition_utility", "") not in ("", None):
                raise ValueError("Random final utility must be null.")
            if str(row.get("final_acquisition_utility_applicable", "")).lower() != "false":
                raise ValueError("Random final utility applicability must be false.")
            if str(row.get("phase4d_similarity_aware_applicable", "")).lower() != "false":
                raise ValueError("Random similarity-aware applicability must be false.")
    else:
        for filename in (
            "utility_only_selected_batch.csv",
            "similarity_aware_selected_batch.csv",
        ):
            selected = _read_csv(policy_dir / filename)
            if len(selected) != 5:
                raise ValueError(f"{filename} must contain five peptides.")
            if len({row["sequence"] for row in selected}) != 5:
                raise ValueError(f"{filename} contains duplicates.")
        for row in pool:
            numeric_fields = [
                key
                for key in (
                    "final_acquisition_utility",
                    "calibrated_ensemble_mean_probability",
                    "calibrated_ensemble_std_probability",
                    "calibrated_predictive_entropy",
                    "calibrated_expected_member_entropy",
                    "calibrated_mutual_information",
                    *(
                        f"calibrated_member_probability_{index}"
                        for index in range(1, 6)
                    ),
                )
            ]
            if not all(np.isfinite(float(row[key])) for key in numeric_fields):
                raise ValueError("Guided retained-pool predictions are not finite.")
    expected = _read_json(policy_dir / "checksums.json")
    actual = _directory_checksums(policy_dir)
    if expected != actual:
        raise ValueError("Policy output checksum mismatch.")


def _primary_artifact_manifest(
    output_root: Path,
    initialized_at: str,
) -> dict[str, object]:
    files = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if _primary_path_excluded(relative):
            continue
        files[relative.as_posix()] = _sha256_file(path)
    return {
        "initialized_at": initialized_at,
        "hash_algorithm": "sha256",
        "protected_files": files,
        "exclusions": {
            "directory": "phase4d/",
            "temporary_suffixes": list(_TEMP_SUFFIXES),
            "lock_suffixes": list(_LOCK_SUFFIXES),
            "filesystem_metadata": sorted(_FILESYSTEM_METADATA),
            "post_initialization_scheduler_outputs": "logs/supek_runtime/*.out|*.err",
        },
    }


def _primary_path_excluded(relative: Path) -> bool:
    if relative.parts and relative.parts[0] == "phase4d":
        return True
    if relative.name in _FILESYSTEM_METADATA:
        return True
    lower = relative.name.lower()
    return lower.endswith(_TEMP_SUFFIXES) or lower.endswith(_LOCK_SUFFIXES)


def _verify_primary_artifacts_unchanged(
    output_root: Path,
    before: dict[str, object],
) -> dict[str, object]:
    protected = dict(before["protected_files"])
    initialized = datetime.fromisoformat(str(before["initialized_at"]))
    current = {}
    unclassified_additions = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if _primary_path_excluded(relative):
            continue
        relative_text = relative.as_posix()
        if relative_text not in protected and _is_new_scheduler_output(
            path, relative, initialized
        ):
            continue
        current[relative_text] = _sha256_file(path)
        if relative_text not in protected:
            unclassified_additions.append(relative_text)
    missing = sorted(set(protected) - set(current))
    changed = sorted(
        path for path in protected if path in current and protected[path] != current[path]
    )
    if missing or changed or unclassified_additions:
        raise RuntimeError(
            "Primary Phase 4 artifact protection failed: "
            f"missing={missing}, changed={changed}, additions={unclassified_additions}"
        )
    return {
        **before,
        "verified_at": _now_iso(),
        "verification": "passed",
    }


def _is_new_scheduler_output(
    path: Path,
    relative: Path,
    initialized: datetime,
) -> bool:
    if relative.parts[:2] != ("logs", "supek_runtime"):
        return False
    if path.suffix.lower() not in {".out", ".err"}:
        return False
    return datetime.fromtimestamp(path.stat().st_mtime, initialized.tzinfo) >= initialized


def _derive_phase4d_walltime(
    output_root: Path,
    override: str | None,
) -> dict[str, object]:
    if override:
        _parse_walltime(override)
        return {
            "source": "explicit_override",
            "pbs_walltime": override,
            "measured_seconds": None,
            "requested_seconds": None,
        }
    runtime_root = output_root / "logs" / "supek_runtime"
    timings = {}
    for policy in PHASE4_POLICIES:
        path = runtime_root / f"p4_{policy[:7]}.out"
        if not path.exists():
            raise FileNotFoundError(
                "Primary Phase 4 runtime evidence is missing. "
                "Provide --phase4d-walltime explicitly."
            )
        timings[policy] = _runtime_seconds_from_log(path)
    measured = sum(timings.values())
    requested = max(7200, 2 * measured + 1800)
    rounded_hours = int(math.ceil(requested / 3600))
    walltime = f"{rounded_hours:02d}:00:00"
    return {
        "source": "primary_phase4_runtime_logs",
        "source_logs": {
            policy: str(runtime_root / f"p4_{policy[:7]}.out")
            for policy in PHASE4_POLICIES
        },
        "policy_elapsed_seconds": timings,
        "measured_seconds": measured,
        "formula": "max(7200, 2 * measured_seconds + 1800), rounded up to hour",
        "requested_seconds": requested,
        "pbs_walltime": walltime,
    }


def _runtime_seconds_from_log(path: Path) -> float:
    start = end = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[phase4] start "):
            start = datetime.fromisoformat(line.split(" start ", 1)[1].split(" host=", 1)[0])
        elif line.startswith("[phase4] end "):
            end = datetime.fromisoformat(line.split(" end ", 1)[1].split(" exit_status=", 1)[0])
    if start is None or end is None or end < start:
        raise ValueError(f"Cannot derive Phase 4 runtime from {path}")
    return (end - start).total_seconds()


def _parse_walltime(value: str) -> None:
    parts = value.split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("Walltime must use HH:MM:SS.")


def _write_phase4d_pbs(
    output_root: Path,
    run_id: int,
    pbs_repo_root_value: str | None,
    walltime: str,
    seed_config: str | None,
) -> Path:
    run_root = _phase4d_run_root(output_root, run_id)
    pbs_root = run_root / "supek_pbs"
    log_root = run_root / "logs"
    pbs_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(pbs_repo_root=pbs_repo_root_value)
    repo_root = _pbs_repo_root(args)
    target_output = _target_path(output_root, repo_root)
    target_log = _target_path(log_root, repo_root)
    command = (
        "python -m active_learning_thesis phase4-bo run-diversity-aware "
        f"--output-root {target_output.as_posix()} "
        f"--source-round 1 --phase4d-run {run_id}"
    )
    if seed_config:
        seed_path = Path(seed_config)
        target_seed = _target_path(seed_path, repo_root)
        command += f" --seed-config {target_seed.as_posix()}"
    config = _read_json(output_root / "config.json")
    resources = config["supek"]
    text = f"""#!/bin/bash
#PBS -N p4d_run{run_id:03d}
#PBS -q {resources['queue']}
#PBS -l select=1:ncpus={resources['ncpus']}:mem={resources['mem']}:ngpus={resources['ngpus']}
#PBS -l walltime={walltime}
#PBS -o {target_log.as_posix()}/phase4d.out
#PBS -e {target_log.as_posix()}/phase4d.err

set -eo pipefail
cd "{repo_root.as_posix()}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
echo "[phase4d] start $(date -Is) host=$(hostname)"
echo "[phase4d] command: {command}"
status=0
{command} || status=$?
echo "[phase4d] end $(date -Is) exit_status=$status"
exit "$status"
"""
    path = pbs_root / "supek_phase4d_run_001.pbs"
    path.write_text(text, encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
