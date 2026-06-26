from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

from active_learning_thesis.config import RunConfig
from active_learning_thesis.dataset import build_split_manifest, read_experimental_dataset
from active_learning_thesis.acquisition import _descending_indices, select_batch
from active_learning_thesis.generative import generate_candidate_sequences
from active_learning_thesis.metrics import evaluate_binary_classifier, pr_best_f1_threshold
from active_learning_thesis.phase2_replay import load_frozen_model_config
from active_learning_thesis.predictive import (
    load_ensemble_from_dir,
    score_sequences_with_ensemble,
    train_ensemble,
)


PHASE4_POLICIES = ("random", "greedy", "ucb", "pi", "ei", "mes")
MODEL_GUIDED_POLICIES = ("greedy", "ucb", "pi", "ei", "mes")
DEFAULT_OUTPUT_ROOT = Path("thesis_results/04_bayesian_optimization")
DEFAULT_PHASE1_ROOT = Path("thesis_results/01_reproduction")
DEFAULT_PHASE3_ROOT = Path("thesis_results/03_real_al")
STRICT_CONTACT_METRIC = "paper_path_APcontact_last10ns"
LEGACY_CONTACT_METRIC = "paper_APcontact"
PROBABILITY_EPSILON = 1e-6
ZERO_VARIANCE_EPSILON = 1e-8

GENERATION_FIELDS = [
    "sequence",
    "generator_objective",
    "generator_utility_score",
    "generator_utility_scope",
    "similarity_penalty",
    "length_penalty",
    "generator_fitness",
]

CANDIDATE_FIELDS = [
    "sequence",
    "sequence_length",
    "inside_preferred_length_range",
    "valid_sequence",
    "already_labeled",
    "excluded_holdout",
    "excluded_phase3",
    "acquisition_method",
    "acquisition_variant",
    "generator_objective",
    "generator_utility_score",
    "generator_utility_scope",
    "similarity_penalty",
    "length_penalty",
    "generator_fitness",
    "raw_member_probability_1",
    "raw_member_probability_2",
    "raw_member_probability_3",
    "raw_member_probability_4",
    "raw_member_probability_5",
    "calibrated_member_probability_1",
    "calibrated_member_probability_2",
    "calibrated_member_probability_3",
    "calibrated_member_probability_4",
    "calibrated_member_probability_5",
    "raw_ensemble_mean_probability",
    "calibrated_ensemble_mean_probability",
    "calibrated_ensemble_std_probability",
    "calibrated_predictive_entropy",
    "calibrated_expected_member_entropy",
    "calibrated_mutual_information",
    "surrogate_space_incumbent",
    "ucb_kappa",
    "improvement_xi",
    "zero_variance_epsilon",
    "ensemble_sd_ddof",
    "mes_cdf_clip_min",
    "final_acquisition_utility",
    "final_acquisition_utility_applicable",
    "final_acquisition_utility_scope",
    "selection_rank",
    "mes_maximum_value_samples",
    "mes_tie_count",
    "tie_break_seed",
    "random_shuffle_audit_key",
]


def run_phase4_bo(args: argparse.Namespace) -> dict[str, object]:
    action = str(getattr(args, "phase4_bo_action", ""))
    if action == "init":
        return init_phase4(args)
    if action == "train-ensemble":
        return train_phase4_ensemble(args)
    if action == "propose":
        return propose_phase4(args)
    if action == "compare":
        return compare_phase4(args)
    if action == "status":
        return status_phase4(args)
    raise ValueError(f"Unsupported phase4-bo action: {action}")


def init_phase4(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    phase1_root = Path(args.phase1_root)
    phase3_root = Path(args.phase3_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not bool(getattr(args, "force", False)):
            raise FileExistsError(f"Phase 4 output already exists: {output_root}")
        archive = output_root.parent / f"{output_root.name}_archive_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(output_root), str(archive))

    policies = tuple(args.policies or PHASE4_POLICIES)
    unknown = sorted(set(policies) - set(PHASE4_POLICIES))
    if unknown:
        raise ValueError("Unsupported Phase 4 policies: " + ", ".join(unknown))
    if int(args.round) != 1:
        raise ValueError("Phase 4 supports exactly round 1.")

    canonical = identify_canonical_training_data(phase1_root, phase3_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _phase4_config(args, canonical)
    _write_json(output_root / "config.json", config)
    _write_training_manifests(output_root, canonical)
    _write_implementation_audit(output_root, config)
    _write_phase4_scaffold(output_root, policies, config)
    pbs_paths = write_phase4_pbs_previews(output_root, policies, args=args)
    manifest = {
        "phase": "phase4_bayesian_optimization",
        "created_at": _now_iso(),
        "rounds": ["round_001"],
        "policies": list(policies),
        "model_fitting_rows": canonical["model_training_count"],
        "validation_calibration_rows": canonical["validation_count"],
        "development_rows": canonical["development_count"],
        "training_positive_count": canonical["positive_count"],
        "training_negative_count": canonical["negative_count"],
        "holdout_sequence_count": canonical["holdout_count"],
        "holdout_labels_exported": False,
        "phase1_root": str(phase1_root),
        "phase3_root": str(phase3_root),
        "canonical_split_rule": "phase1_dataset_recreated_with_seed_20260317_and_phase3_cross_checked",
        "strict_contact_metric": STRICT_CONTACT_METRIC,
        "legacy_contact_metric": LEGACY_CONTACT_METRIC,
        "automatic_submission": False,
        "automatic_ingestion": False,
        "maximum_round": 1,
        "pbs_paths": pbs_paths,
    }
    _write_json(output_root / "phase4_manifest.json", manifest)
    compare_phase4(argparse.Namespace(output_root=str(output_root), round=1))
    return {
        "status": "initialized",
        "output_root": str(output_root),
        "model_fitting_rows": canonical["model_training_count"],
        "validation_calibration_rows": canonical["validation_count"],
        "development_rows": canonical["development_count"],
        "positive_count": canonical["positive_count"],
        "negative_count": canonical["negative_count"],
        "holdout_count": canonical["holdout_count"],
        "pbs_submit_preview": pbs_paths["submit_all"],
    }


def identify_canonical_training_data(phase1_root: Path, phase3_root: Path) -> dict[str, object]:
    assignments_path = phase1_root / "folds" / "nested_cv_fold_assignments.csv"
    sanity_path = phase1_root / "tables" / "dataset_sanity.json"
    if not assignments_path.exists() or not sanity_path.exists():
        raise FileNotFoundError(
            "Immutable Phase 1 dataset artifacts are required: "
            f"{assignments_path} and {sanity_path}"
        )
    assignments = _read_csv(assignments_path)
    records_by_sequence: dict[str, dict[str, str]] = {}
    for row in assignments:
        sequence = _canonical_sequence(row.get("sequence", ""))
        label = str(row.get("label", "")).strip()
        if not sequence or label not in {"0", "1"}:
            raise ValueError("Invalid Phase 1 fold assignment row.")
        previous = records_by_sequence.get(sequence)
        if previous and previous["label"] != label:
            raise ValueError(f"Conflicting Phase 1 labels for {sequence}.")
        records_by_sequence[sequence] = {
            "sequence": sequence,
            "label": label,
            "label_source": "experimental",
        }
    source_records = read_experimental_dataset()
    source_map = {
        _canonical_sequence(row["sequence"]): str(row["label"])
        for row in source_records
    }
    phase1_map = {
        sequence: str(row["label"])
        for sequence, row in records_by_sequence.items()
    }
    if source_map != phase1_map:
        raise ValueError(
            "The canonical experimental dataset disagrees with the immutable "
            "Phase 1 fold-assignment membership or labels."
        )
    records = [
        {
            "sequence": _canonical_sequence(row["sequence"]),
            "label": str(row["label"]),
            "label_source": "experimental",
        }
        for row in source_records
    ]
    sanity = _read_json(sanity_path)
    if int(sanity.get("total peptides", -1)) != len(records):
        raise ValueError("Phase 1 dataset sanity count does not match fold assignments.")

    split_config = RunConfig(
        random_seed=20260317,
        holdout_fraction=0.20,
        validation_fraction_of_development=0.20,
        replay_seed_size=40,
    )
    split_manifest = build_split_manifest(records, split_config)
    canonical_map = {
        row["sequence"]: {
            "label": row["label"],
            "split": split_manifest["splits"][row["sequence"]],
        }
        for row in records
    }
    _cross_check_phase3_experimental_map(phase3_root, canonical_map)

    model_training_rows = [
        dict(row)
        for row in records
        if canonical_map[row["sequence"]]["split"] == "train_pool"
    ]
    validation_rows = [
        dict(row)
        for row in records
        if canonical_map[row["sequence"]]["split"] == "validation"
    ]
    development_rows = [*model_training_rows, *validation_rows]
    holdout_sequences = sorted(
        sequence
        for sequence, payload in canonical_map.items()
        if payload["split"] == "holdout"
    )
    excluded_phase3 = _phase3_generated_sequences(phase3_root)
    positive_count = sum(row["label"] == "1" for row in development_rows)
    negative_count = sum(row["label"] == "0" for row in development_rows)
    if (
        len(model_training_rows),
        len(validation_rows),
        len(development_rows),
        positive_count,
        negative_count,
        len(holdout_sequences),
    ) != (
        235,
        59,
        294,
        199,
        95,
        74,
    ):
        raise ValueError(
            "Canonical Phase 4 counts differ from the approved 235/59/294/199/95/74 split: "
            f"{len(model_training_rows)}/{len(validation_rows)}/{len(development_rows)}/"
            f"{positive_count}/{negative_count}/{len(holdout_sequences)}"
        )
    checksum_payload = "\n".join(
        f"{row['sequence']},{row['label']},{canonical_map[row['sequence']]['split']}"
        for row in records
    )
    return {
        "records": records,
        "canonical_map": canonical_map,
        "model_training_rows": model_training_rows,
        "validation_rows": validation_rows,
        "development_rows": development_rows,
        "holdout_sequences": holdout_sequences,
        "phase3_excluded_sequences": sorted(excluded_phase3),
        "model_training_count": len(model_training_rows),
        "validation_count": len(validation_rows),
        "development_count": len(development_rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "holdout_count": len(holdout_sequences),
        "dataset_checksum": hashlib.sha256(checksum_payload.encode("utf-8")).hexdigest(),
        "source_paths": [str(assignments_path), str(sanity_path)],
    }


def _cross_check_phase3_experimental_map(
    phase3_root: Path,
    canonical_map: dict[str, dict[str, str]],
) -> None:
    branches_root = phase3_root / "branches"
    if not branches_root.exists():
        raise FileNotFoundError(f"Phase 3 branches are required for cross-checking: {branches_root}")
    checked = 0
    for branch_dir in sorted(path for path in branches_root.iterdir() if path.is_dir()):
        split_path = branch_dir / "split_manifest.json"
        ledger_path = branch_dir / "ledger.csv"
        if not split_path.exists() or not ledger_path.exists():
            continue
        split_manifest = _read_json(split_path)
        ledger_rows = _read_csv(ledger_path)
        experimental = {
            _canonical_sequence(row["sequence"]): {
                "label": str(row["label"]),
                "split": str(row["split"]),
            }
            for row in ledger_rows
            if row.get("label_source") == "experimental"
        }
        split_map = {
            _canonical_sequence(sequence): split
            for sequence, split in dict(split_manifest.get("splits", {})).items()
        }
        if experimental != canonical_map:
            raise ValueError(f"Phase 3 experimental ledger disagrees with canonical data: {ledger_path}")
        if split_map != {sequence: payload["split"] for sequence, payload in canonical_map.items()}:
            raise ValueError(f"Phase 3 split manifest disagrees with canonical data: {split_path}")
        checked += 1
    if checked == 0:
        raise ValueError("No complete Phase 3 branch was available for canonical-data cross-checking.")


def _phase3_generated_sequences(phase3_root: Path) -> set[str]:
    sequences: set[str] = set()
    for ledger_path in phase3_root.rglob("branches/*/ledger.csv"):
        for row in _read_csv(ledger_path):
            if row.get("split") == "generated":
                sequence = _canonical_sequence(row.get("sequence", ""))
                if sequence:
                    sequences.add(sequence)
    for selected_path in phase3_root.rglob("branches/*/rounds/round_*/selected_batch.csv"):
        for row in _read_csv(selected_path):
            sequence = _canonical_sequence(row.get("sequence", ""))
            if sequence:
                sequences.add(sequence)
    for inventory in phase3_root.rglob("md_inventory/md_inventory.csv"):
        for row in _read_csv(inventory):
            sequence = _canonical_sequence(row.get("sequence", ""))
            if sequence:
                sequences.add(sequence)
    return sequences


_PHASE3_GENERATOR_FIELDS = (
    "batch_size",
    "candidate_pool_min",
    "ga_max_attempts",
    "use_similarity_penalty",
    "use_length_penalty",
    "allowed_amino_acids",
    "preferred_length_min",
    "preferred_length_max",
    "min_initial_peptide_length",
    "max_initial_peptide_length",
    "population_size",
    "offspring_count",
    "max_num_generations",
    "tournament_size",
    "mutation_probability",
)


def _resolved_phase3_generator_snapshot(phase3_root: Path) -> dict[str, object]:
    configs = []
    for path in sorted((phase3_root / "branches").glob("*/config.json")):
        payload = _read_json(path)
        configs.append((path, {field: payload.get(field) for field in _PHASE3_GENERATOR_FIELDS}))
    if not configs:
        raise FileNotFoundError("Phase 3 branch configurations are required for Phase 4.")
    reference_path, reference = configs[0]
    disagreements = [
        str(path)
        for path, payload in configs[1:]
        if payload != reference
    ]
    if disagreements:
        raise ValueError(
            "Phase 3 branch generative settings disagree: "
            + ", ".join([str(reference_path), *disagreements])
        )
    return {
        "resolved_values": reference,
        "source_configs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path, _ in configs
        ],
        "seed_rules": {
            "base_random_seed": 20260317,
            "proposal_seed_offset": "round_id * 100",
            "attempt_seed": "base_random_seed + seed_offset + zero_based_attempt",
            "sequence_seed": "sha256(strategy|ordered_sequences) prefix plus base seed modulo 2**32",
            "selection_seed": "base_random_seed + round_id",
        },
        "implementation": {
            "generator": {
                "path": str(Path(__file__).with_name("generative.py")),
                "sha256": _sha256_file(Path(__file__).with_name("generative.py")),
            },
            "acquisition": {
                "path": str(Path(__file__).with_name("acquisition.py")),
                "sha256": _sha256_file(Path(__file__).with_name("acquisition.py")),
            },
            "configuration": {
                "path": str(Path(__file__).with_name("config.py")),
                "sha256": _sha256_file(Path(__file__).with_name("config.py")),
            },
            "ga": {
                "path": str(
                    Path(__file__).resolve().parents[1]
                    / "SA_ML_generative"
                    / "genetic_algorithm_library.py"
                ),
                "sha256": _sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "SA_ML_generative"
                    / "genetic_algorithm_library.py"
                ),
            },
        },
    }


def _phase4_config(args: argparse.Namespace, canonical: dict[str, object]) -> dict[str, object]:
    frozen = load_frozen_model_config(Path(args.phase1_root))
    ap_sp = frozen.get("AP_SP")
    if not ap_sp:
        raise ValueError("Frozen Phase 1 AP_SP configuration is missing.")
    seeds = [int(args.random_seed) + index for index in range(int(args.ensemble_size))]
    if len(seeds) != 5:
        raise ValueError("Phase 4 requires exactly five ensemble members.")
    generator_snapshot = _resolved_phase3_generator_snapshot(Path(args.phase3_root))
    generator = dict(generator_snapshot["resolved_values"])
    return {
        "phase": "phase4_bayesian_optimization",
        "round": 1,
        "policies": list(args.policies or PHASE4_POLICIES),
        "random_seed": int(args.random_seed),
        "ensemble_size": 5,
        "ensemble_seeds": seeds,
        "epochs": int(args.epochs),
        "batch_size": int(generator["batch_size"]),
        "candidate_pool_target": int(generator["candidate_pool_min"]),
        "ga_max_attempts": int(generator["ga_max_attempts"]),
        "population_size": int(generator["population_size"]),
        "offspring_count": int(generator["offspring_count"]),
        "max_num_generations": int(generator["max_num_generations"]),
        "tournament_size": int(generator["tournament_size"]),
        "mutation_probability": float(generator["mutation_probability"]),
        "generator_min_length": int(generator["min_initial_peptide_length"]),
        "generator_max_length": int(generator["max_initial_peptide_length"]),
        "preferred_length_min": int(generator["preferred_length_min"]),
        "preferred_length_max": int(generator["preferred_length_max"]),
        "allowed_amino_acids": str(generator["allowed_amino_acids"]),
        "use_similarity_penalty": bool(generator["use_similarity_penalty"]),
        "use_length_penalty": bool(generator["use_length_penalty"]),
        "phase3_generator_snapshot": generator_snapshot,
        "ucb_kappa": float(args.kappa),
        "improvement_xi": float(args.xi),
        "zero_variance_epsilon": float(args.epsilon),
        "probability_clipping_epsilon": PROBABILITY_EPSILON,
        "calibration_learning_rate": 0.05,
        "calibration_max_iterations": 500,
        "calibration_l2": 1e-3,
        "calibration_regularisation_target": {"coefficient": 1.0, "intercept": 0.0},
        "calibration_low_variation_epsilon": 1e-6,
        "calibration_method": "phase3_memberwise_platt_fixed_validation",
        "ensemble_sd_ddof": 0,
        "mes_cdf_clip_min": 1e-12,
        "model_training_count": canonical["model_training_count"],
        "validation_count": canonical["validation_count"],
        "development_count": canonical["development_count"],
        "positive_count": canonical["positive_count"],
        "negative_count": canonical["negative_count"],
        "holdout_count": canonical["holdout_count"],
        "dataset_checksum": canonical["dataset_checksum"],
        "phase1_root": str(args.phase1_root),
        "phase3_root": str(args.phase3_root),
        "output_root": str(args.output_root),
        "ap_sp_frozen_config": ap_sp,
        "strict_contact_metric": STRICT_CONTACT_METRIC,
        "legacy_contact_metric": LEGACY_CONTACT_METRIC,
        "supek": {
            "queue": args.supek_queue,
            "ncpus": int(args.supek_ncpus),
            "ngpus": int(args.supek_ngpus),
            "mem": args.supek_mem,
        },
    }


def _write_training_manifests(output_root: Path, canonical: dict[str, object]) -> None:
    rows = []
    canonical_map = canonical["canonical_map"]
    for record in canonical["records"]:
        split = canonical_map[record["sequence"]]["split"]
        included_in_development = split in {"train_pool", "validation"}
        rows.append(
            {
                "sequence": record["sequence"],
                "label": record["label"] if included_in_development else "",
                "source_dataset": "phase1_immutable_experimental_dataset",
                "source_split": split,
                "included_in_development": str(included_in_development).lower(),
                "included_in_model_fit": str(split == "train_pool").lower(),
                "included_in_calibration": str(split == "validation").lower(),
                "included_in_threshold_selection": str(split == "validation").lower(),
                "excluded_frozen_holdout": str(split == "holdout").lower(),
                "exclusion_reason": (
                    "" if included_in_development else "frozen_holdout_label_redacted"
                ),
            }
        )
    _write_csv(output_root / "training_data_manifest.csv", rows)
    _write_json(
        output_root / "training_data_manifest.json",
        {
            "total_labeled_rows_found": len(canonical["records"]),
            "model_fitting_rows": canonical["model_training_count"],
            "validation_calibration_rows": canonical["validation_count"],
            "development_rows": canonical["development_count"],
            "positive_count": canonical["positive_count"],
            "negative_count": canonical["negative_count"],
            "excluded_holdout_rows": canonical["holdout_count"],
            "holdout_labels_exported": False,
            "duplicate_handling": "fail_on_duplicate_or_conflicting_sequence",
            "dataset_checksum": canonical["dataset_checksum"],
            "source_paths": canonical["source_paths"],
            "phase3_excluded_sequence_count": len(canonical["phase3_excluded_sequences"]),
            "phase3_cgmd_labels_used_for_fitting": False,
            "rows": rows,
        },
    )
    _write_json(
        output_root / "proposal_exclusions.json",
        {
            "frozen_holdout_sequences": canonical["holdout_sequences"],
            "phase3_generated_or_simulated_sequences": canonical["phase3_excluded_sequences"],
        },
    )


def _write_phase4_scaffold(
    output_root: Path,
    policies: Iterable[str],
    config: dict[str, object],
) -> None:
    for path in [
        output_root / "models" / "ap_sp_fixed_split_ensemble",
        output_root / "comparison" / "round_001",
        output_root / "md_inventory",
        output_root / "evidence",
        output_root / "supek_pbs",
        output_root / "logs" / "supek_runtime",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root
        / "models"
        / "ap_sp_fixed_split_ensemble"
        / "model_manifest.json",
        {
            "status": "not_trained",
            "immutable": True,
            "ensemble_size": 5,
            "model_fitting_rows": config["model_training_count"],
            "validation_calibration_rows": config["validation_count"],
        },
    )
    _write_csv(
        output_root / "md_inventory" / "md_inventory.csv",
        [],
        fieldnames=[
            "sequence",
            "proposing_policies",
            "policy_ranks",
            "cgmd_status",
            "campaign_path",
            "review_status",
            "strict_label",
            "rubric_version",
        ],
    )
    _write_csv(
        output_root / "md_inventory" / "md_inventory_events.csv",
        [],
        fieldnames=["timestamp", "sequence", "policy", "round_id", "event", "notes"],
    )
    for policy in policies:
        branch_dir = output_root / "branches" / policy
        round_dir = branch_dir / "rounds" / "round_001"
        for path in [round_dir / "review", round_dir / "md_campaigns"]:
            path.mkdir(parents=True, exist_ok=True)
        branch_config = dict(config)
        branch_config["policy"] = policy
        branch_config["acquisition_variant"] = _acquisition_variant(policy)
        _write_json(branch_dir / "config.json", branch_config)
        _write_json(branch_dir / "model_manifest.json", {"status": "not_trained", "shared": True})
        _write_json(
            round_dir / "status.json",
            {
                "status": "preview_ready",
                "policy": policy,
                "round": 1,
                "selected_count": 0,
                "model_fitting_rows": config["model_training_count"],
                "validation_calibration_rows": config["validation_count"],
                "objective_mode": "predicted_strict_self_assembly",
            },
        )
        (round_dir / "command_preview.txt").write_text(
            _proposal_command(output_root.resolve(), policy) + "\n",
            encoding="utf-8",
        )


def _write_implementation_audit(output_root: Path, config: dict[str, object]) -> None:
    rows = []
    formulas = {
        "random": "U_random(x)=0 during GA; final selection is inherited seeded shuffle",
        "greedy": "U_greedy(x)=mu(x)",
        "ucb": "U_UCB(x)=mu(x)+1.0*sigma(x)",
        "pi": "I=mu-f*-0.0; z=I/max(sigma,1e-8); PI=Phi(z) with explicit zero-variance branch",
        "ei": "EI=I*Phi(z)+sigma*phi(z) with explicit zero-variance branch",
        "mes": "five calibrated member-function maxima over the active pool; final ranking uses retained-pool maxima",
    }
    for policy in PHASE4_POLICIES:
        rows.append(
            {
                "policy": policy,
                "source_function": f"active_learning_thesis.phase4_bo.phase4_acquisition_scores[{policy}]",
                "formula": formulas[policy],
                "score_space": "none" if policy == "random" else "calibrated_probability",
                "uncertainty_source": "none" if policy in {"random", "greedy"} else "five calibrated member probabilities; population SD ddof=0",
                "target_definition": "training-set surrogate incumbent f*" if policy in {"pi", "ei"} else "none",
                "kappa": config["ucb_kappa"] if policy == "ucb" else "",
                "xi": config["improvement_xi"] if policy in {"pi", "ei"} else "",
                "epsilon": config["zero_variance_epsilon"] if policy in {"pi", "ei", "mes"} else "",
                "zero_variance_behaviour": "explicit epsilon branch" if policy in {"pi", "ei"} else "n/a",
                "probability_clipping": config["probability_clipping_epsilon"],
                "classification_threshold_use": "reporting_only",
                "implementation_action": "Phase 3 generator with Phase 4 utility callback",
                "unit_test_status": "implemented_and_tested",
            }
        )
    audit_root = output_root / "implementation_audit"
    _write_json(
        audit_root / "bo_implementation_audit.json",
        {
            "methods": rows,
            "calibration": {
                "method": "Phase 3 member-wise standardised-logit Platt calibration",
                "clipping_epsilon": PROBABILITY_EPSILON,
                "training_rows": 235,
                "validation_calibration_rows": 59,
                "learning_rate": config["calibration_learning_rate"],
                "maximum_iterations": config["calibration_max_iterations"],
                "l2": config["calibration_l2"],
                "regularisation_target": config["calibration_regularisation_target"],
                "identity_fallbacks": [
                    "single-class validation labels",
                    "validation-logit population SD below 1e-6",
                    "optimiser exception after valid inputs",
                    "non-finite fitted parameters",
                ],
                "identity_output": "clipped raw member probabilities",
            },
            "acquisition_constants": {
                "ucb_kappa": config["ucb_kappa"],
                "pi_ei_xi": config["improvement_xi"],
                "pi_ei_mes_epsilon": config["zero_variance_epsilon"],
                "ensemble_sd_ddof": config["ensemble_sd_ddof"],
                "mes_cdf_clip": [config["mes_cdf_clip_min"], 1.0],
                "incumbent": "maximum calibrated ensemble mean over the fixed 235-row training set",
            },
            "reused_phase3_functions": {
                "candidate_generation": [
                    "active_learning_thesis.generative.generate_candidate_sequences",
                    "active_learning_thesis.generative._generate_candidate_sequences_single",
                ],
                "composition_penalty": [
                    "active_learning_thesis.generative.calculate_similarity_penalty",
                    "active_learning_thesis.generative.calculate_similarity_penalties",
                ],
                "length_penalty": [
                    "active_learning_thesis.generative.calculate_length_penalty",
                    "active_learning_thesis.generative.calculate_length_penalties",
                ],
                "ga_fitness": [
                    "active_learning_thesis.generative.generation_fitness_components",
                    "active_learning_thesis.generative._population_fitness_from_utilities",
                ],
                "ga_operators_and_survival": (
                    "SA_ML_generative.genetic_algorithm_library.GeneticAlgorithm"
                ),
                "guided_selection": "active_learning_thesis.acquisition._descending_indices",
                "random_selection": "active_learning_thesis.acquisition.select_batch[random]",
            },
            "export_schema": {
                "generator_utility_scope": "generation",
                "guided_final_acquisition_utility_scope": "final",
                "random_final_acquisition_utility": None,
                "random_final_acquisition_utility_scope": "not_applicable",
            },
            "generator_reuse": config["phase3_generator_snapshot"],
            "pbs_dependencies": {
                "random": "independent",
                "guided": "afterok on the single shared ensemble-training job",
                "comparison_status": "afterany on all six proposal jobs",
            },
        },
    )
    markdown = [
        "# Phase 4 BO implementation audit",
        "",
        "All guided methods operate on the five member-wise calibrated AP_SP probabilities. "
        "The validation F1-maximising classification threshold is reporting-only.",
        "",
        "Within the shared generative algorithm and retained-pool selection path, "
        "Phase 4 changes only the calculation of `U_policy(x)`; all Phase 3 "
        "penalty functions, genetic-algorithm mechanics, pool construction, and "
        "selector behaviour remain unchanged.",
        "",
        "The surrogate-space incumbent is the maximum calibrated ensemble mean over "
        "the fixed 235-row training set. The frozen holdout is not scored; its sequence "
        "identities are used only for exact duplicate exclusion.",
        "",
        f"- UCB kappa: `{config['ucb_kappa']}`",
        f"- PI/EI xi: `{config['improvement_xi']}`",
        f"- numerical epsilon: `{config['zero_variance_epsilon']}`",
        f"- ensemble SD convention: population SD, `ddof={config['ensemble_sd_ddof']}`",
        f"- MES CDF clipping: `[{config['mes_cdf_clip_min']}, 1.0]`",
        "- calibration: member-wise standardised-logit Platt calibration on the fixed "
        "59-row validation/calibration set",
        "- malformed calibration arrays fail; valid degenerate cases use clipped "
        "raw-probability identity fallback",
        "",
        "## Reused Phase 3 implementation",
        "- generation and pool export: `generate_candidate_sequences`, "
        "`_generate_candidate_sequences_single`",
        "- fitness: `generation_fitness_components`, `_population_fitness_from_utilities`",
        "- composition penalty: `calculate_similarity_penalty`, "
        "`calculate_similarity_penalties`",
        "- length penalty: `calculate_length_penalty`, `calculate_length_penalties`",
        "- tournament, crossover, mutation and survival: `GeneticAlgorithm`",
        "- guided selection: `_descending_indices`",
        "- random selection: `select_batch(\"random\", ...)`",
        "",
        "- random job: independent",
        "- five guided-policy jobs: afterok on the single shared ensemble-training job",
        "- comparison/status job: afterany on all six proposal jobs",
        "",
    ]
    for row in rows:
        markdown.extend(
            [
                f"## {row['policy']}",
                f"- Formula: `{row['formula']}`",
                f"- Score space: `{row['score_space']}`",
                f"- Uncertainty: `{row['uncertainty_source']}`",
                f"- Threshold use: `{row['classification_threshold_use']}`",
                "",
            ]
        )
    (audit_root / "bo_implementation_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (audit_root / "bo_implementation_audit.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


def train_phase4_ensemble(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    config = _read_json(output_root / "config.json")
    model_root = output_root / "models" / "ap_sp_fixed_split_ensemble"
    manifest_path = model_root / "model_manifest.json"
    if manifest_path.exists() and _read_json(manifest_path).get("status") == "completed":
        if not bool(getattr(args, "force", False)):
            raise FileExistsError("The immutable Phase 4 ensemble already exists.")
    training_rows = _phase4_rows_for_split(output_root, "train_pool")
    validation_rows = _phase4_rows_for_split(output_root, "validation")
    if len(training_rows) != 235 or len(validation_rows) != 59:
        raise ValueError(
            f"Phase 4 fixed split is invalid: {len(training_rows)} training, "
            f"{len(validation_rows)} validation."
        )
    run_config = _predictive_run_config(config)
    ensemble_dir = model_root / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    ensemble = train_ensemble(
        training_rows,
        validation_rows,
        ensemble_dir,
        run_config,
        cache_dir=None,
    )

    validation_sequences = [row["sequence"] for row in validation_rows]
    validation_labels = np.asarray([int(row["label"]) for row in validation_rows], dtype=int)
    validation_scores = score_sequences_with_ensemble(
        ensemble,
        validation_sequences,
        use_calibration=True,
        include_raw=True,
    )
    validation_mean = np.asarray(validation_scores["pred_mean"], dtype=float)
    threshold, threshold_f1 = pr_best_f1_threshold(validation_labels, validation_mean)
    validation_report = {
        "partition": "fixed_validation_calibration",
        "row_count": len(validation_rows),
        "decision_threshold": threshold,
        "threshold_selection_f1": threshold_f1,
        "threshold_tie_break": "higher_threshold",
        "metrics": evaluate_binary_classifier(
            validation_labels,
            validation_mean,
            threshold=threshold,
            threshold_strategy="pr_best_f1",
            threshold_source="fixed_validation_calibration",
            threshold_selection_f1=threshold_f1,
        ),
    }
    _write_json(model_root / "validation_calibration_report.json", validation_report)
    _write_csv(
        model_root / "validation_predictions.csv",
        _prediction_rows(validation_rows, validation_scores),
    )

    training_scores = score_sequences_with_ensemble(
        ensemble,
        [row["sequence"] for row in training_rows],
        use_calibration=True,
        include_raw=True,
    )
    incumbent = _training_incumbent(training_rows, training_scores)
    _write_json(model_root / "training_incumbent.json", incumbent)
    _write_csv(
        model_root / "training_predictions.csv",
        _prediction_rows(training_rows, training_scores),
    )

    calibrations = []
    for member_index, member in enumerate(ensemble):
        calibrations.append(
            {
                "member_index": member_index,
                "seed": int(member.seed),
                **dict(member.calibration or {}),
            }
        )
    _write_json(
        model_root / "member_calibrations.json",
        {
            "method": "phase3_memberwise_platt_fixed_validation",
            "probability_clipping_epsilon": PROBABILITY_EPSILON,
            "members": calibrations,
        },
    )
    source_files = _model_source_files(Path(config["phase1_root"]))
    manifest = {
        "status": "completed",
        "immutable": True,
        "completed_at": _now_iso(),
        "architecture": "AP_SP",
        "ensemble_size": 5,
        "model_fitting_rows": len(training_rows),
        "validation_calibration_rows": len(validation_rows),
        "positive_count": sum(row["label"] == "1" for row in training_rows),
        "negative_count": sum(row["label"] == "0" for row in training_rows),
        "epochs": config["epochs"],
        "seeds": config["ensemble_seeds"],
        "dataset_checksum": config["dataset_checksum"],
        "calibration": str(model_root / "member_calibrations.json"),
        "calibration_settings": {
            "probability_clipping_epsilon": config["probability_clipping_epsilon"],
            "learning_rate": config["calibration_learning_rate"],
            "maximum_iterations": config["calibration_max_iterations"],
            "l2": config["calibration_l2"],
            "regularisation_target": config["calibration_regularisation_target"],
            "low_variation_epsilon": config["calibration_low_variation_epsilon"],
        },
        "validation_report": str(model_root / "validation_calibration_report.json"),
        "training_incumbent": str(model_root / "training_incumbent.json"),
        "phase3_generator_snapshot": config["phase3_generator_snapshot"],
        "source_configuration": source_files,
        "member_checksums": {
            path.name: _sha256_file(path)
            for path in sorted(ensemble_dir.glob("ap_sp_member_*.h5"))
        },
    }
    _write_json(manifest_path, manifest)
    for policy in MODEL_GUIDED_POLICIES:
        _write_json(output_root / "branches" / policy / "model_manifest.json", manifest)
    return {"status": "trained", "model_manifest": str(manifest_path), "members": 5}


def _prediction_rows(
    source_rows: list[dict[str, str]],
    scores: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    calibrated = np.asarray(scores["ensemble_member_probs"], dtype=float)
    raw = np.asarray(scores["raw_ensemble_member_probs"], dtype=float)
    rows = []
    for index, source in enumerate(source_rows):
        row: dict[str, object] = {
            "sequence": source["sequence"],
            "label": source["label"],
            "calibrated_ensemble_mean_probability": float(scores["pred_mean"][index]),
            "calibrated_ensemble_std_probability": float(scores["pred_std"][index]),
            "calibrated_predictive_entropy": float(scores["pred_entropy"][index]),
            "calibrated_expected_member_entropy": float(
                scores["pred_expected_entropy"][index]
            ),
            "calibrated_mutual_information": float(
                scores["pred_mutual_information"][index]
            ),
        }
        for member_index in range(calibrated.shape[1]):
            row[f"raw_member_probability_{member_index + 1}"] = float(
                raw[index, member_index]
            )
            row[f"calibrated_member_probability_{member_index + 1}"] = float(
                calibrated[index, member_index]
            )
        rows.append(row)
    return rows


def _training_incumbent(
    training_rows: list[dict[str, str]],
    calibrated_scores: dict[str, np.ndarray],
) -> dict[str, object]:
    if len(training_rows) != 235:
        raise ValueError(
            "The Phase 4 surrogate incumbent requires exactly 235 training rows."
        )
    means = np.asarray(calibrated_scores["pred_mean"], dtype=float)
    members = np.asarray(calibrated_scores["ensemble_member_probs"], dtype=float)
    if means.shape != (235,) or members.shape != (235, 5):
        raise ValueError("Training incumbent inputs must contain calibrated 235x5 predictions.")
    incumbent_index = int(np.argmax(means))
    return {
        "definition": "maximum calibrated ensemble mean over fixed 235-row training set",
        "value": float(means[incumbent_index]),
        "sequence": training_rows[incumbent_index]["sequence"],
        "training_row_count": len(training_rows),
        "calibrated_member_probabilities": [
            float(value) for value in members[incumbent_index]
        ],
    }


def _model_source_files(phase1_root: Path) -> dict[str, dict[str, str]]:
    predictive_code = Path(__file__).resolve().parents[1] / "SA_ML_predictive" / "code"
    candidates = {
        "architecture": phase1_root / "frozen_model_config.json",
        "architecture_implementation": predictive_code / "models.py",
        "preprocessing": predictive_code / "utils.py",
        "preprocessing_adapter": Path(__file__).with_name("predictive.py"),
        "optimizer": predictive_code / "automate_training.py",
        "class_weighting": Path(__file__).with_name("predictive.py"),
        "training_implementation": Path(__file__).with_name("predictive.py"),
        "calibration_implementation": Path(__file__).with_name("predictive.py"),
    }
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in candidates.items()
    }


def _predictive_run_config(config: dict[str, object]) -> RunConfig:
    frozen = config["ap_sp_frozen_config"]
    num_cells = dict(RunConfig().model_num_cells)
    kernel_sizes = dict(RunConfig().model_kernel_size)
    num_cells["AP_SP"] = int(frozen["num_cells"])
    kernel_sizes["AP_SP"] = int(frozen["kernel_size"])
    return RunConfig(
        random_seed=int(config["random_seed"]),
        ensemble_size=5,
        ensemble_seeds=[int(seed) for seed in config["ensemble_seeds"]],
        epochs=int(config["epochs"]),
        model_num_cells=num_cells,
        model_kernel_size=kernel_sizes,
        candidate_pool_min=int(config["candidate_pool_target"]),
        ga_max_attempts=int(config["ga_max_attempts"]),
        batch_size=int(config["batch_size"]),
        use_similarity_penalty=bool(config["use_similarity_penalty"]),
        use_length_penalty=bool(config["use_length_penalty"]),
        allowed_amino_acids=str(config["allowed_amino_acids"]),
        preferred_length_min=int(config["preferred_length_min"]),
        preferred_length_max=int(config["preferred_length_max"]),
        min_initial_peptide_length=int(config["generator_min_length"]),
        max_initial_peptide_length=int(config["generator_max_length"]),
        population_size=int(config["population_size"]),
        offspring_count=int(config["offspring_count"]),
        max_num_generations=int(config["max_num_generations"]),
        tournament_size=int(config["tournament_size"]),
        mutation_probability=float(config["mutation_probability"]),
        discovery_ucb_beta=float(config["ucb_kappa"]),
        discovery_improvement_xi=float(config["improvement_xi"]),
        calibration_learning_rate=float(config["calibration_learning_rate"]),
        calibration_max_iter=int(config["calibration_max_iterations"]),
        calibration_l2=float(config["calibration_l2"]),
        use_calibrated_acquisition=True,
    )


def propose_phase4(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    policy = str(args.branch)
    if policy not in PHASE4_POLICIES:
        raise ValueError(f"Unsupported Phase 4 policy: {policy}")
    if int(args.round) != 1:
        raise ValueError("Phase 4 supports exactly round 1.")
    round_dir = output_root / "branches" / policy / "rounds" / "round_001"
    selected_path = round_dir / "selected_batch.csv"
    if selected_path.exists() and not bool(getattr(args, "force", False)):
        raise FileExistsError(f"Phase 4 proposal already exists: {selected_path}")
    if bool(getattr(args, "dry_run", False)) or bool(getattr(args, "write_supek_pbs", False)):
        pbs = write_phase4_pbs_previews(output_root, PHASE4_POLICIES, args=args)
        return {"status": "preview_ready", "branch": policy, "pbs": pbs.get(policy, "")}

    config = _read_json(output_root / "config.json")
    started = _now_iso()
    _write_branch_status(output_root, policy, "running", started_at=started)
    try:
        run_config = _predictive_run_config(config)
        exclusions = _proposal_exclusions(output_root)
        ensemble = model_context = None
        if policy in MODEL_GUIDED_POLICIES:
            ensemble, model_context = _load_verified_ensemble(output_root, config)

        utility_callback = _phase4_utility_callback(
            policy,
            ensemble,
            model_context,
            config,
        )
        candidate_sequences, generator_metadata = generate_candidate_sequences(
            ensemble,
            exclusions,
            run_config,
            min_unique=int(config["candidate_pool_target"]),
            seed_offset=100,
            objective="broad_pool" if policy == "random" else policy,
            minimum_return_count=int(config["batch_size"]),
            use_similarity_penalty=bool(config["use_similarity_penalty"]),
            use_length_penalty=bool(config["use_length_penalty"]),
            return_metadata=True,
            policy_utility_callback=utility_callback,
        )
        candidates = _final_candidate_rows(
            policy,
            candidate_sequences,
            generator_metadata,
            ensemble,
            model_context,
            config,
        )
        selection_seed = int(config["random_seed"]) + 1
        if policy == "random":
            selected_indices, _ = select_batch(
                "random",
                int(config["batch_size"]),
                {"pred_mean": np.zeros(len(candidates), dtype=float)},
                np.empty((0, 0), dtype=float),
                run_config,
                selection_seed,
            )
        else:
            utilities = np.asarray(
                [float(row["final_acquisition_utility"]) for row in candidates],
                dtype=float,
            )
            selected_indices = _descending_indices(utilities)[: int(config["batch_size"])]
        selected = []
        for rank, index in enumerate(selected_indices, start=1):
            candidates[index]["selection_rank"] = rank
            if policy == "random":
                candidates[index]["random_shuffle_audit_key"] = _random_audit_key(
                    selection_seed, str(candidates[index]["sequence"])
                )
            selected.append(dict(candidates[index]))
        if len(selected) != int(config["batch_size"]):
            raise RuntimeError("A successful Phase 4 branch must select exactly five peptides.")

        generation_rows = [
            {
                "sequence": sequence,
                "generator_objective": metadata.get("generator_objective", ""),
                "generator_utility_score": metadata.get("generator_utility_score", ""),
                "generator_utility_scope": "generation",
                "similarity_penalty": metadata.get("similarity_penalty", ""),
                "length_penalty": metadata.get("length_penalty", ""),
                "generator_fitness": metadata.get("generator_fitness", ""),
            }
            for sequence, metadata in generator_metadata.items()
        ]
        _write_csv(
            round_dir / "generation_log.csv",
            generation_rows,
            fieldnames=GENERATION_FIELDS,
        )
        _write_csv(round_dir / "candidate_pool.csv", candidates, fieldnames=CANDIDATE_FIELDS)
        _write_csv(round_dir / "scored_candidates.csv", candidates, fieldnames=CANDIDATE_FIELDS)
        _write_csv(round_dir / "acquisition_log.csv", candidates, fieldnames=CANDIDATE_FIELDS)
        _write_csv(round_dir / "selected_batch.csv", selected, fieldnames=CANDIDATE_FIELDS)
        _write_json(
            round_dir / "execution_manifest.json",
            {
                "policy": policy,
                "started_at": started,
                "completed_at": _now_iso(),
                "phase3_generator_snapshot": config["phase3_generator_snapshot"],
                "actual_seeds": {
                    "base_random_seed": int(config["random_seed"]),
                    "seed_offset": 100,
                    "attempt_seeds": [
                        int(config["random_seed"]) + 100 + index
                        for index in range(int(config["ga_max_attempts"]))
                    ],
                    "selection_seed": selection_seed,
                    "sequence_seed_rule": config["phase3_generator_snapshot"]["seed_rules"][
                        "sequence_seed"
                    ],
                },
                "acquisition_constants": {
                    "ucb_kappa": config["ucb_kappa"],
                    "improvement_xi": config["improvement_xi"],
                    "zero_variance_epsilon": config["zero_variance_epsilon"],
                    "ensemble_sd_ddof": config["ensemble_sd_ddof"],
                    "mes_cdf_clip": [config["mes_cdf_clip_min"], 1.0],
                },
            },
        )
        if policy in {"pi", "ei"}:
            _write_json(
                round_dir / "acquisition_diagnostics.json",
                _acquisition_diagnostics(candidates, model_context),
            )
        _append_md_inventory(output_root, policy, selected)
        _write_branch_status(
            output_root,
            policy,
            "completed",
            started_at=started,
            completed_at=_now_iso(),
            actual_candidate_count=len(candidates),
            selected_count=len(selected),
        )
        return {
            "status": "completed",
            "branch": policy,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_batch": str(selected_path),
        }
    except Exception as exc:
        _write_branch_status(
            output_root,
            policy,
            "failed",
            started_at=started,
            completed_at=_now_iso(),
            error=str(exc),
        )
        raise


def _phase4_utility_callback(
    policy: str,
    ensemble,
    model_context: dict[str, object] | None,
    config: dict[str, object],
) -> object:
    def callback(sequences: list[str]):
        if policy == "random":
            return {
                "sequences": list(sequences),
                "utilities": np.zeros(len(sequences), dtype=float),
                "metadata": {},
            }
        rows = _score_model_guided_sequences(
            policy,
            ensemble,
            model_context or {},
            sequences,
            config,
        )
        return {
            "sequences": list(sequences),
            "utilities": np.asarray(
                [float(row["final_acquisition_utility"]) for row in rows],
                dtype=float,
            ),
            "metadata": {},
        }

    return callback


def _score_model_guided_sequences(
    policy: str,
    ensemble,
    model_context: dict[str, object],
    sequences: list[str],
    config: dict[str, object],
) -> list[dict[str, object]]:
    scored = score_sequences_with_ensemble(
        ensemble,
        sequences,
        use_calibration=True,
        include_raw=True,
    )
    calibrated = np.asarray(scored["ensemble_member_probs"], dtype=float)
    raw = np.asarray(scored["raw_ensemble_member_probs"], dtype=float)
    mean = np.asarray(scored["pred_mean"], dtype=float)
    std = np.asarray(scored["pred_std"], dtype=float)
    acquisition, mes_samples = phase4_acquisition_scores(
        policy,
        mean,
        std,
        calibrated,
        float(model_context["incumbent"]["value"]),
        float(config["ucb_kappa"]),
        float(config["improvement_xi"]),
        float(config["zero_variance_epsilon"]),
    )
    tie_count = int(len(acquisition) - len(np.unique(acquisition)))
    rows = []
    for index, sequence in enumerate(sequences):
        row = {
            "acquisition_method": policy,
            "acquisition_variant": _acquisition_variant(policy),
            "final_acquisition_utility": float(acquisition[index]),
            "final_acquisition_utility_applicable": True,
            "final_acquisition_utility_scope": "final",
            "raw_ensemble_mean_probability": float(raw[index].mean()),
            "calibrated_ensemble_mean_probability": float(mean[index]),
            "calibrated_ensemble_std_probability": float(std[index]),
            "calibrated_predictive_entropy": float(scored["pred_entropy"][index]),
            "calibrated_expected_member_entropy": float(
                scored["pred_expected_entropy"][index]
            ),
            "calibrated_mutual_information": float(
                scored["pred_mutual_information"][index]
            ),
            "surrogate_space_incumbent": float(model_context["incumbent"]["value"]),
            "ucb_kappa": float(config.get("ucb_kappa", 1.0)),
            "improvement_xi": float(config.get("improvement_xi", 0.0)),
            "zero_variance_epsilon": float(
                config.get("zero_variance_epsilon", ZERO_VARIANCE_EPSILON)
            ),
            "ensemble_sd_ddof": int(config.get("ensemble_sd_ddof", 0)),
            "mes_cdf_clip_min": float(config.get("mes_cdf_clip_min", 1e-12)),
            "mes_maximum_value_samples": (
                json.dumps([float(value) for value in mes_samples])
                if policy == "mes"
                else ""
            ),
            "mes_tie_count": tie_count if policy == "mes" else "",
            "tie_break_seed": int(config["random_seed"]) + 1,
        }
        for member in range(5):
            row[f"raw_member_probability_{member + 1}"] = float(raw[index, member])
            row[f"calibrated_member_probability_{member + 1}"] = float(
                calibrated[index, member]
            )
        rows.append(row)
    return rows


def phase4_acquisition_scores(
    policy: str,
    mean_probability: np.ndarray,
    std_probability: np.ndarray,
    calibrated_member_probabilities: np.ndarray,
    incumbent: float,
    kappa: float,
    xi: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean_probability, dtype=float)
    std = np.asarray(std_probability, dtype=float)
    if policy == "random":
        return np.zeros_like(mean), np.empty(0)
    if policy == "greedy":
        return mean.copy(), np.empty(0)
    if policy == "ucb":
        return mean + kappa * std, np.empty(0)
    improvement = mean - incumbent - xi
    safe_std = np.maximum(std, epsilon)
    z = improvement / safe_std
    if policy == "pi":
        scores = _normal_cdf(z)
        zero = std <= epsilon
        scores[zero] = (improvement[zero] > 0.0).astype(float)
        return scores, np.empty(0)
    if policy == "ei":
        scores = improvement * _normal_cdf(z) + safe_std * _normal_pdf(z)
        zero = std <= epsilon
        scores[zero] = np.maximum(improvement[zero], 0.0)
        return scores, np.empty(0)
    if policy == "mes":
        matrix = np.asarray(calibrated_member_probabilities, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 5:
            raise ValueError("MES requires five coherent ensemble-member function vectors.")
        maximum_samples = matrix.max(axis=0)
        safe = np.maximum(std, epsilon)
        gamma = (maximum_samples[:, None] - mean[None, :]) / safe[None, :]
        cdf = np.clip(_normal_cdf(gamma), 1e-12, 1.0)
        pdf = _normal_pdf(gamma)
        scores = ((gamma * pdf) / (2.0 * cdf) - np.log(cdf)).mean(axis=0)
        scores[std <= epsilon] = 0.0
        return scores, maximum_samples
    raise ValueError(f"Unsupported model-guided policy: {policy}")


def _final_candidate_rows(
    policy: str,
    sequences: list[str],
    generator_metadata: dict[str, dict[str, object]],
    ensemble,
    model_context: dict[str, object] | None,
    config: dict[str, object],
) -> list[dict[str, object]]:
    guided_rows = (
        _score_model_guided_sequences(
            policy,
            ensemble,
            model_context or {},
            sequences,
            config,
        )
        if policy != "random"
        else [{} for _ in sequences]
    )
    rows = []
    for sequence, guided in zip(sequences, guided_rows):
        metadata = generator_metadata.get(sequence, {})
        row = {field: "" for field in CANDIDATE_FIELDS}
        row.update(guided)
        row.update(
            {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "inside_preferred_length_range": 5 <= len(sequence) <= 10,
                "valid_sequence": True,
                "already_labeled": False,
                "excluded_holdout": False,
                "excluded_phase3": False,
                "acquisition_method": policy,
                "acquisition_variant": _acquisition_variant(policy),
                "generator_objective": metadata.get("generator_objective", ""),
                "generator_utility_score": metadata.get("generator_utility_score", ""),
                "generator_utility_scope": "generation",
                "similarity_penalty": metadata.get("similarity_penalty", ""),
                "length_penalty": metadata.get("length_penalty", ""),
                "generator_fitness": metadata.get("generator_fitness", ""),
                "selection_rank": "",
                "ucb_kappa": float(config.get("ucb_kappa", 1.0)),
                "improvement_xi": float(config.get("improvement_xi", 0.0)),
                "zero_variance_epsilon": float(
                    config.get("zero_variance_epsilon", ZERO_VARIANCE_EPSILON)
                ),
                "ensemble_sd_ddof": int(config.get("ensemble_sd_ddof", 0)),
                "mes_cdf_clip_min": float(config.get("mes_cdf_clip_min", 1e-12)),
            }
        )
        if policy == "random":
            row.update(
                {
                    "final_acquisition_utility": None,
                    "final_acquisition_utility_applicable": False,
                    "final_acquisition_utility_scope": "not_applicable",
                }
            )
        rows.append(row)
    return rows


def _proposal_exclusions(
    output_root: Path,
) -> set[str]:
    payload = _read_json(output_root / "proposal_exclusions.json")
    development = _phase4_development_sequences(output_root)
    return {
        *development,
        *(_canonical_sequence(sequence) for sequence in payload["frozen_holdout_sequences"]),
        *(
            _canonical_sequence(sequence)
            for sequence in payload["phase3_generated_or_simulated_sequences"]
        ),
    }


def _random_audit_key(seed: int, sequence: str) -> str:
    return hashlib.sha256(f"{seed}:{_canonical_sequence(sequence)}".encode()).hexdigest()


def _acquisition_diagnostics(
    rows: list[dict[str, object]],
    model_context: dict[str, object] | None,
) -> dict[str, object]:
    values = np.asarray([float(row["final_acquisition_utility"]) for row in rows])
    means = np.asarray(
        [float(row["calibrated_ensemble_mean_probability"]) for row in rows]
    )
    tolerance = 1e-12
    rounded = np.round(values, 12)
    incumbent = float((model_context or {})["incumbent"]["value"])
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "population_sd_ddof_0": float(values.std(ddof=0)),
        "exact_zero_count": int(np.sum(values == 0.0)),
        "exact_zero_proportion": float(np.mean(values == 0.0)),
        "above_1e_12_count": int(np.sum(values > tolerance)),
        "above_1e_12_proportion": float(np.mean(values > tolerance)),
        "raw_unique_count": int(len(np.unique(values))),
        "effective_unique_count_rounded_12dp": int(len(np.unique(rounded))),
        "mean_probability_above_incumbent_count": int(np.sum(means > incumbent)),
        "mean_probability_above_incumbent_proportion": float(np.mean(means > incumbent)),
        "degeneracy_warning": (
            "effective acquisition values are constant at 12 decimal places"
            if len(np.unique(rounded)) == 1
            else ""
        ),
    }


def _load_verified_ensemble(
    output_root: Path,
    config: dict[str, object],
    *,
    required: bool = True,
):
    model_root = output_root / "models" / "ap_sp_fixed_split_ensemble"
    manifest_path = model_root / "model_manifest.json"
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError("Canonical Phase 4 ensemble has not been trained.")
        return None, None
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        if not required:
            return None, None
        raise ValueError("Canonical Phase 4 ensemble has not completed training.")
    if manifest.get("dataset_checksum") != config["dataset_checksum"]:
        raise ValueError("Canonical Phase 4 ensemble manifest is invalid or incompatible.")
    for name, checksum in manifest["member_checksums"].items():
        if _sha256_file(model_root / "ensemble" / name) != checksum:
            raise ValueError(f"Canonical ensemble checksum mismatch: {name}")
    ensemble = load_ensemble_from_dir(
        model_root / "ensemble",
        _predictive_run_config(config),
    )
    context = {
        "incumbent": _read_json(model_root / "training_incumbent.json"),
        "validation_report": _read_json(
            model_root / "validation_calibration_report.json"
        ),
        "member_calibrations": _read_json(model_root / "member_calibrations.json"),
    }
    return ensemble, context


def compare_phase4(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    comparison_root = output_root / "comparison" / "round_001"
    comparison_root.mkdir(parents=True, exist_ok=True)
    model_manifest_path = (
        output_root
        / "models"
        / "ap_sp_fixed_split_ensemble"
        / "model_manifest.json"
    )
    model_status = (
        str(_read_json(model_manifest_path).get("status", "missing"))
        if model_manifest_path.exists()
        else "missing"
    )
    random_status_path = (
        output_root / "branches" / "random" / "rounds" / "round_001" / "status.json"
    )
    random_workflow_status = (
        str(_read_json(random_status_path).get("status", "missing"))
        if random_status_path.exists()
        else "missing"
    )
    workflow_has_run = random_workflow_status in {"completed", "failed", "blocked"}
    all_selected = []
    status_rows = []
    by_policy: dict[str, set[str]] = {}
    for policy in PHASE4_POLICIES:
        round_dir = output_root / "branches" / policy / "rounds" / "round_001"
        status_path = round_dir / "status.json"
        status_payload = _read_json(status_path) if status_path.exists() else {}
        status = str(status_payload.get("status", "missing"))
        inferred_error = ""
        if (
            policy in MODEL_GUIDED_POLICIES
            and status in {"preview_ready", "queued"}
            and workflow_has_run
            and model_status != "completed"
        ):
            status = "blocked"
            inferred_error = (
                "shared ensemble training did not complete; afterok dependency "
                "prevented guided proposal execution"
            )
        if status not in {"completed", "failed", "blocked"}:
            status = "missing"
        selected_path = round_dir / "selected_batch.csv"
        rows = _read_csv(selected_path) if status == "completed" and selected_path.exists() else []
        if status == "completed" and len(rows) != 5:
            status = "failed"
            rows = []
        status_rows.append(
            {
                "policy": policy,
                "status": status,
                "selected_count": len(rows),
                "selected_batch": str(selected_path) if selected_path.exists() else "",
                "error": status_payload.get("error", "") or inferred_error,
            }
        )
        by_policy[policy] = {row["sequence"] for row in rows}
        for row in rows:
            all_selected.append({"policy": policy, **row})
    _write_csv(comparison_root / "branch_status.csv", status_rows)
    _write_csv(comparison_root / "all_selected_peptides.csv", all_selected)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in all_selected:
        grouped.setdefault(row["sequence"], []).append(row)
    unique_rows = [
        {
            "sequence": sequence,
            "policies": ";".join(sorted(row["policy"] for row in rows)),
            "policy_count": len(rows),
        }
        for sequence, rows in sorted(grouped.items())
    ]
    duplicate_rows = [row for row in unique_rows if int(row["policy_count"]) > 1]
    _write_csv(comparison_root / "unique_selected_peptides.csv", unique_rows)
    _write_csv(comparison_root / "cross_policy_duplicates.csv", duplicate_rows)
    overlap_rows = []
    for left in PHASE4_POLICIES:
        row = {"policy": left}
        for right in PHASE4_POLICIES:
            union = by_policy[left] | by_policy[right]
            row[right] = len(by_policy[left] & by_policy[right]) / len(union) if union else ""
        overlap_rows.append(row)
    _write_csv(comparison_root / "policy_overlap_matrix.csv", overlap_rows)
    summary_rows = []
    for policy in PHASE4_POLICIES:
        policy_rows = [row for row in all_selected if row["policy"] == policy]
        summary_rows.append(
            {
                "policy": policy,
                "status": next(row["status"] for row in status_rows if row["policy"] == policy),
                "selected_count": len(policy_rows),
                "mean_final_acquisition_utility": (
                    float(
                        np.mean(
                            [
                                float(row["final_acquisition_utility"])
                                for row in policy_rows
                                if str(row.get("final_acquisition_utility", "")).strip()
                            ]
                        )
                    )
                    if any(
                        str(row.get("final_acquisition_utility", "")).strip()
                        for row in policy_rows
                    )
                    else ""
                ),
            }
        )
    _write_csv(comparison_root / "policy_score_summary.csv", summary_rows)
    markdown = [
        "# Phase 4 Round 1 descriptive comparison",
        "",
        "This report does not rank policies. Only completed five-peptide outputs are compared.",
        "",
        "## Branch status",
        "",
        "| Policy | Status | Selected |",
        "|---|---|---:|",
    ]
    markdown.extend(
        f"| {row['policy']} | {row['status']} | {row['selected_count']} |"
        for row in status_rows
    )
    markdown.extend(
        [
            "",
            f"Available selected slots: {len(all_selected)}",
            f"Unique selected peptides: {len(unique_rows)}",
            f"Cross-policy duplicate peptides: {len(duplicate_rows)}",
            "",
        ]
    )
    (comparison_root / "round_001_comparison.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    return {
        "status": "comparison_written",
        "branch_status": {row["policy"]: row["status"] for row in status_rows},
        "available_selected_count": len(all_selected),
        "unique_selected_count": len(unique_rows),
    }


def status_phase4(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root)
    manifest = _read_json(output_root / "phase4_manifest.json") if (output_root / "phase4_manifest.json").exists() else {}
    branches = {}
    for policy in PHASE4_POLICIES:
        path = output_root / "branches" / policy / "rounds" / "round_001" / "status.json"
        payload = _read_json(path) if path.exists() else {}
        status = str(payload.get("status", "missing"))
        if status not in {"preview_ready", "queued", "running", "completed", "failed", "blocked"}:
            status = "missing"
        branches[policy] = {"status": status, **payload}
    return {
        "status": "ok" if manifest else "not_initialized",
        "manifest": manifest,
        "model_status": (
            _read_json(output_root / "models" / "ap_sp_fixed_split_ensemble" / "model_manifest.json")
            if (output_root / "models" / "ap_sp_fixed_split_ensemble" / "model_manifest.json").exists()
            else {"status": "missing"}
        ),
        "branches": branches,
    }


def write_phase4_pbs_previews(
    output_root: Path,
    policies: Iterable[str],
    *,
    args: argparse.Namespace | None = None,
) -> dict[str, str]:
    config = _read_json(output_root / "config.json")
    pbs_root = output_root / "supek_pbs"
    runtime_root = output_root / "logs" / "supek_runtime"
    pbs_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    repo_root = _pbs_repo_root(args)
    target_output = _target_path(output_root, repo_root)
    target_runtime = _target_path(runtime_root, repo_root)
    target_pbs_root = _target_path(pbs_root, repo_root)
    resources = config["supek"]
    paths = {}
    train_path = pbs_root / "supek_phase4_train_ensemble.pbs"
    train_path.write_text(
        _pbs_text(
            "p4_train",
            _train_command(target_output),
            target_runtime,
            repo_root,
            resources,
            "10:00:00",
        ),
        encoding="utf-8",
    )
    paths["train_ensemble"] = str(train_path)
    walltimes = {
        "random": "02:00:00",
        "greedy": "08:00:00",
        "ucb": "08:00:00",
        "pi": "08:00:00",
        "ei": "08:00:00",
        "mes": "10:00:00",
    }
    for policy in policies:
        path = pbs_root / f"supek_phase4_{policy}_r001.pbs"
        path.write_text(
            _pbs_text(
                f"p4_{policy[:7]}",
                _proposal_command(target_output, policy),
                target_runtime,
                repo_root,
                resources,
                walltimes[policy],
            ),
            encoding="utf-8",
        )
        paths[policy] = str(path)
    compare_path = pbs_root / "supek_phase4_compare_r001.pbs"
    compare_path.write_text(
        _pbs_text(
            "p4_compare",
            _compare_command(target_output),
            target_runtime,
            repo_root,
            {**resources, "ngpus": 0},
            "01:00:00",
        ),
        encoding="utf-8",
    )
    paths["compare"] = str(compare_path)
    submit_path = pbs_root / "supek_phase4_submit_all.sh"
    submit_lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        'STAMP=$(date +%Y%m%d_%H%M%S)',
        f'LOG="{(target_pbs_root / "phase4_job_ids_${STAMP}.log").as_posix()}"',
        f'TRAIN_ID=$(qsub "{(target_pbs_root / train_path.name).as_posix()}")',
        'echo "train_ensemble=$TRAIN_ID" | tee -a "$LOG"',
        f'RANDOM_ID=$(qsub "{(target_pbs_root / Path(paths["random"]).name).as_posix()}")',
        'echo "random=$RANDOM_ID" | tee -a "$LOG"',
    ]
    for policy in MODEL_GUIDED_POLICIES:
        submit_lines.extend(
            [
                f'{policy.upper()}_ID=$(qsub -W depend=afterok:$TRAIN_ID "{(target_pbs_root / Path(paths[policy]).name).as_posix()}")',
                f'echo "{policy}=${{{policy.upper()}_ID}}" | tee -a "$LOG"',
            ]
        )
    dependency_ids = [
        "$RANDOM_ID",
        *[f"${policy.upper()}_ID" for policy in MODEL_GUIDED_POLICIES],
    ]
    submit_lines.extend(
        [
            f'COMPARE_ID=$(qsub -W depend=afterany:{":".join(dependency_ids)} "{(target_pbs_root / compare_path.name).as_posix()}")',
            'echo "compare=$COMPARE_ID dependency=afterany" | tee -a "$LOG"',
            'echo "Submitted Phase 4 preview DAG. Inspect $LOG and qstat manually."',
        ]
    )
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    paths["submit_all"] = str(submit_path)
    return paths


def _pbs_text(
    job_name: str,
    command: str,
    log_dir: Path,
    repo_root: Path,
    resources: dict[str, object],
    walltime: str,
) -> str:
    ngpus = int(resources["ngpus"])
    select = (
        f"select=1:ncpus={resources['ncpus']}:mem={resources['mem']}"
        + (f":ngpus={ngpus}" if ngpus else "")
    )
    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q {resources['queue']}
#PBS -l {select}
#PBS -l walltime={walltime}
#PBS -o {log_dir.as_posix()}/{job_name}.out
#PBS -e {log_dir.as_posix()}/{job_name}.err

set -eo pipefail
cd "{repo_root.as_posix()}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
echo "[phase4] start $(date -Is) host=$(hostname)"
echo "[phase4] python=$(python --version 2>&1)"
echo "[phase4] conda=${{CONDA_DEFAULT_ENV:-unknown}}"
echo "[phase4] git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi || true
echo "[phase4] command: {command}"
status=0
{command} || status=$?
echo "[phase4] end $(date -Is) exit_status=$status"
exit "$status"
"""


def _write_branch_status(
    output_root: Path,
    policy: str,
    status: str,
    **extra,
) -> None:
    config = _read_json(output_root / "config.json")
    payload = {
        "status": status,
        "policy": policy,
        "round": 1,
        "model_fitting_rows": config["model_training_count"],
        "validation_calibration_rows": config["validation_count"],
        "training_positive_count": config["positive_count"],
        "training_negative_count": config["negative_count"],
        "model_architecture": "AP_SP",
        "ensemble_size": 5,
        "training_seeds": config["ensemble_seeds"],
        "calibration_method": config["calibration_method"],
        "objective_mode": "predicted_strict_self_assembly",
        "acquisition_variant": _acquisition_variant(policy),
        "kappa": config["ucb_kappa"] if policy == "ucb" else "",
        "xi": config["improvement_xi"] if policy in {"pi", "ei"} else "",
        "requested_candidate_count": config["candidate_pool_target"],
        "selected_count": 0,
        "hostname": socket.gethostname(),
        "git_commit": _git_commit_hash(),
        "exact_command": _proposal_command(output_root.resolve(), policy),
        **extra,
    }
    _write_json(
        output_root / "branches" / policy / "rounds" / "round_001" / "status.json",
        payload,
    )


def _append_md_inventory(
    output_root: Path,
    policy: str,
    selected: list[dict[str, object]],
) -> None:
    inventory_path = output_root / "md_inventory" / "md_inventory.csv"
    rows = _read_csv(inventory_path)
    by_sequence = {row["sequence"]: row for row in rows}
    for selected_row in selected:
        sequence = str(selected_row["sequence"])
        existing = by_sequence.get(sequence)
        if existing:
            policies = set(filter(None, existing["proposing_policies"].split(";")))
            policies.add(policy)
            ranks = set(filter(None, existing["policy_ranks"].split(";")))
            ranks.add(f"{policy}:{selected_row['selection_rank']}")
            existing["proposing_policies"] = ";".join(sorted(policies))
            existing["policy_ranks"] = ";".join(sorted(ranks))
        else:
            by_sequence[sequence] = {
                "sequence": sequence,
                "proposing_policies": policy,
                "policy_ranks": f"{policy}:{selected_row['selection_rank']}",
                "cgmd_status": "not_submitted",
                "campaign_path": "",
                "review_status": "not_reviewed",
                "strict_label": "",
                "rubric_version": (
                    f"AP_sasa>=1.75 AND {STRICT_CONTACT_METRIC}>=0.5"
                ),
            }
    _write_csv(inventory_path, [by_sequence[key] for key in sorted(by_sequence)])


def _phase4_rows_for_split(
    output_root: Path,
    split: str,
) -> list[dict[str, str]]:
    rows = [
        {
            "sequence": row["sequence"],
            "label": row["label"],
            "label_source": "experimental",
            "split": row["source_split"],
        }
        for row in _read_csv(output_root / "training_data_manifest.csv")
        if row["source_split"] == split
    ]
    return rows


def _phase4_development_sequences(output_root: Path) -> set[str]:
    return {
        _canonical_sequence(row["sequence"])
        for row in _read_csv(output_root / "training_data_manifest.csv")
        if row["included_in_development"] == "true"
    }


def _proposal_command(output_root: Path, policy: str) -> str:
    return (
        "python -m active_learning_thesis phase4-bo propose "
        f"--output-root {shlex.quote(output_root.as_posix())} "
        f"--branch {shlex.quote(policy)} --round 1"
    )


def _train_command(output_root: Path) -> str:
    return (
        "python -m active_learning_thesis phase4-bo train-ensemble "
        f"--output-root {shlex.quote(output_root.as_posix())}"
    )


def _compare_command(output_root: Path) -> str:
    return (
        "python -m active_learning_thesis phase4-bo compare "
        f"--output-root {shlex.quote(output_root.as_posix())} --round 1"
    )


def _acquisition_variant(policy: str) -> str:
    return {
        "random": "deterministic seeded random selection after Phase 3 broad-pool generation",
        "greedy": "calibrated greedy exploitation",
        "ucb": "calibrated upper-confidence bound",
        "pi": "probability-space approximate probability of improvement",
        "ei": "probability-space approximate expected improvement",
        "mes": "calibrated ensemble-based approximate max-value entropy search",
    }[policy]


def _canonical_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.exp(-0.5 * values * values) / math.sqrt(2.0 * math.pi)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def _pbs_repo_root(args: argparse.Namespace | None):
    value = getattr(args, "pbs_repo_root", None) if args is not None else None
    if value and str(value).startswith("/"):
        return PurePosixPath(str(value))
    return Path(value).resolve() if value else Path.cwd().resolve()


def _target_path(path: Path, repo_root):
    if isinstance(repo_root, PurePosixPath):
        return repo_root / path.as_posix()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _git_commit_hash() -> str:
    cwd = Path.cwd().resolve()
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={cwd.as_posix()}", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (fieldnames or list(rows[0].keys())) if rows else (fieldnames or [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
