from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np

from active_learning_thesis.acquisition import (
    candidate_objective_scores,
    embedding_novelty_scores,
    generator_objective_requires_embeddings,
    generator_objective_requires_family_models,
)
from active_learning_thesis.config import RunConfig
from active_learning_thesis.dependencies import load_genetic_algorithm_class
from active_learning_thesis.predictive import (
    score_sequences_with_ensemble,
    score_sequences_with_family,
)


PolicyUtilityCallback = Callable[[list[str]], object]
_FORBIDDEN_UTILITY_METADATA = {
    "similarity_penalty_applied",
    "length_penalty_applied",
    "rank_applied",
    "percentile_applied",
    "normalization_applied",
    "normalisation_applied",
    "postprocessing_applied",
}


def _max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(values.max()) if values.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values / max_value


def calculate_amino_acid_frequencies(
    sequence: str,
    allowed_amino_acids: str,
) -> np.ndarray:
    return np.array([sequence.count(amino_acid) for amino_acid in allowed_amino_acids])


def calculate_similarity_penalty(
    sequence: str,
    population,
    allowed_amino_acids: str,
) -> float:
    penalty = 0.0
    first = True
    frequencies = calculate_amino_acid_frequencies(sequence, allowed_amino_acids)
    for neighbour_peptide in population:
        if neighbour_peptide.sequence == sequence and first:
            first = False
            continue
        neighbour_frequencies = calculate_amino_acid_frequencies(
            neighbour_peptide.sequence,
            allowed_amino_acids,
        )
        denominator = np.sum(frequencies) + np.sum(neighbour_frequencies)
        if denominator == 0:
            continue
        penalty += 0.1 * (
            1
            - np.sum(np.abs(frequencies - neighbour_frequencies)) / denominator
        )
    if len(population) <= 1:
        return 0.0
    return penalty / (len(population) - 1)


def calculate_length_penalty(sequence: str, config: RunConfig) -> float:
    if (
        len(sequence) < config.preferred_length_min
        or len(sequence) > config.preferred_length_max
    ):
        midpoint = (config.preferred_length_min + config.preferred_length_max) / 2
        return min(0.05 * abs(len(sequence) - midpoint), 0.5)
    return 0.0


def calculate_similarity_penalties(
    sequences: list[str],
    allowed_amino_acids: str,
) -> np.ndarray:
    if len(sequences) <= 1:
        return np.zeros(len(sequences), dtype=float)

    frequencies = np.asarray(
        [calculate_amino_acid_frequencies(sequence, allowed_amino_acids) for sequence in sequences],
        dtype=float,
    )
    lengths = frequencies.sum(axis=1)
    pairwise_l1 = np.abs(frequencies[:, None, :] - frequencies[None, :, :]).sum(axis=2)
    denominators = lengths[:, None] + lengths[None, :]
    similarities = np.zeros_like(pairwise_l1, dtype=float)
    valid = denominators > 0
    similarities[valid] = 0.1 * (1 - pairwise_l1[valid] / denominators[valid])
    np.fill_diagonal(similarities, 0.0)
    return similarities.sum(axis=1) / (len(sequences) - 1)


def calculate_length_penalties(
    sequences: list[str],
    config: RunConfig,
) -> np.ndarray:
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=float)
    midpoint = (config.preferred_length_min + config.preferred_length_max) / 2
    penalties = np.zeros(len(sequences), dtype=float)
    outside_range = (
        (lengths < config.preferred_length_min)
        | (lengths > config.preferred_length_max)
    )
    penalties[outside_range] = np.minimum(
        0.05 * np.abs(lengths[outside_range] - midpoint),
        0.5,
    )
    return penalties


def _population_fitness_from_utilities(
    sequences: list[str],
    utility_scores: np.ndarray,
    config: RunConfig,
    use_similarity_penalty: bool = True,
    use_length_penalty: bool = True,
) -> np.ndarray:
    similarity_penalties = (
        calculate_similarity_penalties(sequences, config.allowed_amino_acids)
        if use_similarity_penalty
        else np.zeros(len(sequences), dtype=float)
    )
    length_penalties = (
        calculate_length_penalties(sequences, config)
        if use_length_penalty
        else np.zeros(len(sequences), dtype=float)
    )
    return np.asarray(utility_scores, dtype=float) - similarity_penalties - length_penalties


def _population_fitness_from_probabilities(
    sequences: list[str],
    mean_probabilities: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    return _population_fitness_from_utilities(
        sequences,
        mean_probabilities,
        config,
    )


def _validated_policy_utilities(
    sequences: list[str],
    callback: PolicyUtilityCallback,
) -> np.ndarray:
    result = callback(list(sequences))
    metadata: dict[str, object] = {}
    if isinstance(result, dict):
        returned_sequences = result.get("sequences")
        if returned_sequences is not None and list(returned_sequences) != list(sequences):
            raise ValueError("Policy utility callback returned sequence identifiers out of order.")
        metadata_payload = result.get("metadata", {})
        if metadata_payload is not None:
            if not isinstance(metadata_payload, dict):
                raise ValueError("Policy utility callback metadata must be a mapping.")
            metadata = dict(metadata_payload)
        if "utilities" not in result:
            raise ValueError("Structured policy utility callback result requires 'utilities'.")
        result = result["utilities"]
    forbidden = sorted(
        key for key in _FORBIDDEN_UTILITY_METADATA if bool(metadata.get(key, False))
    )
    if forbidden:
        raise ValueError(
            "Policy utility callback must return raw utility without post-processing: "
            + ", ".join(forbidden)
        )
    utilities = np.asarray(result, dtype=float)
    if utilities.ndim != 1:
        raise ValueError("Policy utility callback must return a one-dimensional array.")
    if len(utilities) != len(sequences):
        raise ValueError(
            "Policy utility callback returned "
            f"{len(utilities)} values for {len(sequences)} sequences."
        )
    if not np.isfinite(utilities).all():
        raise ValueError("Policy utility callback returned non-finite values.")
    return utilities


def generation_fitness_components(
    sequences: list[str],
    candidate_scores: dict[str, np.ndarray],
    objective: str,
    config: RunConfig,
    labeled_embeddings: np.ndarray | None = None,
    seed: int = 0,
    incumbent: float = 0.0,
    use_similarity_penalty: bool | None = None,
    use_length_penalty: bool | None = None,
    policy_utility_callback: PolicyUtilityCallback | None = None,
) -> dict[str, np.ndarray]:
    similarity_enabled = (
        config.use_similarity_penalty
        if use_similarity_penalty is None
        else use_similarity_penalty
    )
    length_enabled = (
        config.use_length_penalty
        if use_length_penalty is None
        else use_length_penalty
    )
    utility_scores = (
        _validated_policy_utilities(sequences, policy_utility_callback)
        if policy_utility_callback is not None
        else candidate_objective_scores(
            objective,
            candidate_scores,
            labeled_embeddings
            if labeled_embeddings is not None
            else np.empty((0, 0), dtype=float),
            config,
            seed,
            incumbent=incumbent,
        )
    )
    similarity_penalties = (
        calculate_similarity_penalties(sequences, config.allowed_amino_acids)
        if similarity_enabled
        else np.zeros(len(sequences), dtype=float)
    )
    length_penalties = (
        calculate_length_penalties(sequences, config)
        if length_enabled
        else np.zeros(len(sequences), dtype=float)
    )
    fitness = np.asarray(utility_scores, dtype=float) - similarity_penalties - length_penalties
    components = {
        "generator_utility_score": np.asarray(utility_scores, dtype=float),
        "similarity_penalty": similarity_penalties,
        "length_penalty": length_penalties,
        "generator_fitness": fitness,
    }
    if objective == "ensemble_mi":
        components["normalized_mi"] = _max_normalize(
            np.asarray(candidate_scores["pred_mutual_information"], dtype=float)
        )
    if objective == "embedding_novelty":
        raw_scores, normalized_scores = embedding_novelty_scores(
            np.asarray(candidate_scores["avg_embedding"], dtype=float),
            labeled_embeddings
            if labeled_embeddings is not None
            else np.empty((0, 0), dtype=float),
        )
        components["embedding_novelty_raw"] = raw_scores
        components["normalized_embedding_novelty"] = normalized_scores
    return components


def _sequence_seed(
    sequences: list[str],
    base_seed: int,
    strategy: str,
) -> int:
    payload = "|".join([strategy, *sequences])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) + base_seed) % (2**32)


def _generate_candidate_sequences_single(
    ensemble,
    existing_sequences: set[str],
    config: RunConfig,
    min_unique: int | None = None,
    seed_offset: int = 0,
    utility_strategy: str | None = None,
    objective: str | None = None,
    incumbent: float = 0.0,
    minimum_return_count: int | None = None,
    family_models=None,
    labeled_embeddings: np.ndarray | None = None,
    use_similarity_penalty: bool | None = None,
    use_length_penalty: bool | None = None,
    return_metadata: bool = False,
    policy_utility_callback: PolicyUtilityCallback | None = None,
    attempt_history: list[dict[str, object]] | None = None,
) -> list[str] | tuple[list[str], dict[str, dict[str, float | str]]]:
    GeneticAlgorithm = load_genetic_algorithm_class()
    generator_objective = objective or utility_strategy or "ensemble_mean"
    required_unique = min_unique or config.candidate_pool_min
    minimum_count = minimum_return_count or required_unique
    discovered: list[str] = []
    seen = set(existing_sequences)
    next_seed = config.random_seed + seed_offset
    score_cache: dict[str, dict[str, np.ndarray | float]] = {}
    metadata_by_sequence: dict[str, dict[str, float | str]] = {}

    def score_many(sequences: list[str]) -> dict[str, np.ndarray]:
        unique_sequences = list(dict.fromkeys(sequences))
        missing_sequences = [
            sequence for sequence in unique_sequences if sequence not in score_cache
        ]
        if missing_sequences:
            summary = score_sequences_with_ensemble(
                ensemble,
                missing_sequences,
                include_embeddings=generator_objective_requires_embeddings(generator_objective),
                use_calibration=config.use_calibrated_acquisition,
                include_raw=True,
            )
            if generator_objective_requires_family_models(generator_objective):
                if family_models is None:
                    raise ValueError(
                        "family_qbc generator objective requires family models"
                    )
                summary.update(
                    score_sequences_with_family(
                        family_models,
                        missing_sequences,
                        use_calibration=config.use_calibrated_acquisition,
                        include_raw=True,
                    )
                )
            for index, sequence in enumerate(missing_sequences):
                score_cache[sequence] = {
                    key: value[index].copy()
                    if isinstance(value[index], np.ndarray)
                    else float(value[index])
                    for key, value in summary.items()
                }
        assembled: dict[str, list] = {}
        for sequence in sequences:
            for key, value in score_cache[sequence].items():
                assembled.setdefault(key, []).append(value)
        return {
            key: np.asarray(values, dtype=float)
            for key, values in assembled.items()
        }

    def predict_mean(sequence: str) -> float:
        if policy_utility_callback is not None:
            return float(_validated_policy_utilities([sequence], policy_utility_callback)[0])
        return float(score_many([sequence])["pred_mean"][0])

    def evaluate_population(population) -> np.ndarray:
        sequences = [peptide.sequence for peptide in population]
        candidate_scores = (
            {"pred_mean": np.zeros(len(sequences), dtype=float)}
            if policy_utility_callback is not None
            else score_many(sequences)
        )
        components = generation_fitness_components(
            sequences,
            candidate_scores,
            generator_objective,
            config,
            labeled_embeddings=labeled_embeddings,
            seed=_sequence_seed(
                sequences,
                config.random_seed + seed_offset,
                generator_objective,
            ),
            incumbent=incumbent,
            use_similarity_penalty=use_similarity_penalty,
            use_length_penalty=use_length_penalty,
            policy_utility_callback=policy_utility_callback,
        )
        return components["generator_fitness"]

    while len(discovered) < required_unique and len(discovered) < required_unique * config.ga_max_attempts:
        attempt_index = next_seed - (config.random_seed + seed_offset)
        attempt_seed = next_seed
        np.random.seed(attempt_seed)
        next_seed += 1
        attempt_events = {"created": 0}

        def record_attempt_event(event: dict[str, object]) -> None:
            if event.get("event") == "created":
                attempt_events["created"] += 1

        ga = GeneticAlgorithm(
            fitness_function=predict_mean,
            similarity_penalty=lambda sequence, population: calculate_similarity_penalty(
                sequence, population, config.allowed_amino_acids
            ),
            length_penalty=lambda sequence: calculate_length_penalty(sequence, config),
            min_initial_peptide_length=config.min_initial_peptide_length,
            max_initial_peptide_length=config.max_initial_peptide_length,
            allowed_amino_acids=config.allowed_amino_acids,
            population_size=config.population_size,
            offspring_count=config.offspring_count,
            max_num_generations=config.max_num_generations,
            tournament_size=config.tournament_size,
            mutation_probability=config.mutation_probability,
            population_fitness_function=evaluate_population,
            event_callback=record_attempt_event if attempt_history is not None else None,
        )
        population = ga.find_peptides()
        unique_sequences: list[str] = []
        for peptide in population:
            sequence = peptide.sequence
            if sequence in seen:
                continue
            unique_sequences.append(sequence)
            seen.add(sequence)
        accepted_before = len(discovered)
        if unique_sequences:
            candidate_scores = (
                {"pred_mean": np.zeros(len(unique_sequences), dtype=float)}
                if policy_utility_callback is not None
                else score_many(unique_sequences)
            )
            components = generation_fitness_components(
                unique_sequences,
                candidate_scores,
                generator_objective,
                config,
                labeled_embeddings=labeled_embeddings,
                seed=_sequence_seed(
                    unique_sequences,
                    config.random_seed + seed_offset,
                    generator_objective,
                ),
                incumbent=incumbent,
                use_similarity_penalty=use_similarity_penalty,
                use_length_penalty=use_length_penalty,
                policy_utility_callback=policy_utility_callback,
            )
            ranking = sorted(
                enumerate(unique_sequences),
                key=lambda item: components["generator_utility_score"][item[0]],
                reverse=True,
            )
            for index, sequence in ranking:
                metadata_by_sequence[sequence] = {
                    "generator_objective": generator_objective,
                    "generator_utility_score": float(
                        components["generator_utility_score"][index]
                    ),
                    "similarity_penalty": float(
                        components["similarity_penalty"][index]
                    ),
                    "length_penalty": float(components["length_penalty"][index]),
                    "generator_fitness": float(
                        components["generator_fitness"][index]
                    ),
                }
                if "normalized_mi" in components:
                    metadata_by_sequence[sequence]["normalized_mi"] = float(
                        components["normalized_mi"][index]
                    )
                if "embedding_novelty_raw" in components:
                    metadata_by_sequence[sequence]["embedding_novelty_raw"] = float(
                        components["embedding_novelty_raw"][index]
                    )
                if "normalized_embedding_novelty" in components:
                    metadata_by_sequence[sequence]["normalized_embedding_novelty"] = float(
                        components["normalized_embedding_novelty"][index]
                    )
                if sequence not in discovered:
                    discovered.append(sequence)
                if len(discovered) >= required_unique:
                    break
        if attempt_history is not None:
            attempt_history.append(
                {
                    "attempt_index": int(attempt_index),
                    "attempt_seed": int(attempt_seed),
                    "candidates_generated": int(attempt_events["created"]),
                    "final_population_size": int(len(population)),
                    "novel_candidates_generated": int(len(unique_sequences)),
                    "candidates_accepted": int(len(discovered) - accepted_before),
                    "retained_pool_size_after_attempt": int(len(discovered)),
                }
            )
        if next_seed - (config.random_seed + seed_offset) >= config.ga_max_attempts:
            break

    if len(discovered) < minimum_count:
        raise RuntimeError(
            f"Unable to generate {minimum_count} unique peptides after "
            f"{config.ga_max_attempts} attempts."
        )
    selected = discovered[: min(required_unique, len(discovered))]
    if return_metadata:
        return selected, {
            sequence: metadata_by_sequence.get(
                sequence,
                {
                    "generator_objective": generator_objective,
                    "generator_utility_score": 0.0,
                    "similarity_penalty": 0.0,
                    "length_penalty": 0.0,
                    "generator_fitness": 0.0,
                },
            )
            for sequence in selected
        }
    return selected


def _hybrid_two_pool_targets(candidate_pool_min: int) -> tuple[int, int]:
    mi_target = int(math.ceil(candidate_pool_min / 2))
    return mi_target, int(candidate_pool_min - mi_target)


def _generate_hybrid_two_pool_sequences(
    ensemble,
    existing_sequences: set[str],
    config: RunConfig,
    min_unique: int | None = None,
    seed_offset: int = 0,
    incumbent: float = 0.0,
    minimum_return_count: int | None = None,
    family_models=None,
    labeled_embeddings: np.ndarray | None = None,
    use_similarity_penalty: bool | None = None,
    use_length_penalty: bool | None = None,
    return_metadata: bool = False,
    policy_utility_callback: PolicyUtilityCallback | None = None,
) -> list[str] | tuple[list[str], dict[str, dict[str, float | str]]]:
    required_unique = min_unique or config.candidate_pool_min
    minimum_count = minimum_return_count or required_unique
    mi_target, novelty_target = _hybrid_two_pool_targets(required_unique)
    merged_sequences: list[str] = []
    metadata_by_sequence: dict[str, dict[str, float | str]] = {}
    deduplicated_count = 0

    def add_sequences(
        sequences: list[str],
        metadata: dict[str, dict[str, float | str]],
        subpool: str,
        subpool_target: int,
        subpool_fill_count: int = 0,
    ) -> None:
        nonlocal deduplicated_count
        for sequence in sequences:
            if sequence in metadata_by_sequence:
                deduplicated_count += 1
                continue
            source_meta = dict(metadata.get(sequence, {}))
            source_meta.update(
                {
                    "generator_objective": "hybrid_two_pool",
                    "generator_subpool": subpool,
                    "subpool_target": subpool_target,
                    "subpool_fill_count": subpool_fill_count,
                }
            )
            metadata_by_sequence[sequence] = source_meta
            merged_sequences.append(sequence)

    def generate_subpool(
        objective: str,
        target: int,
        subpool: str,
        exclusions: set[str],
        offset: int,
        subpool_fill_count: int = 0,
    ) -> None:
        if target <= 0:
            return
        sequences, metadata = _generate_candidate_sequences_single(
            ensemble,
            exclusions,
            config,
            min_unique=target,
            seed_offset=seed_offset + offset,
            objective=objective,
            incumbent=incumbent,
            minimum_return_count=target,
            family_models=family_models,
            labeled_embeddings=labeled_embeddings,
            use_similarity_penalty=use_similarity_penalty,
            use_length_penalty=use_length_penalty,
            return_metadata=True,
            policy_utility_callback=policy_utility_callback,
        )
        add_sequences(
            sequences,
            metadata,
            subpool,
            target,
            subpool_fill_count=subpool_fill_count,
        )

    def family_counts() -> dict[str, int]:
        counts = {"ensemble_mi": 0, "embedding_novelty": 0, "broad_pool": 0}
        for sequence in merged_sequences:
            subpool = str(metadata_by_sequence[sequence].get("generator_subpool", ""))
            if subpool.startswith("ensemble_mi"):
                counts["ensemble_mi"] += 1
            elif subpool.startswith("embedding_novelty"):
                counts["embedding_novelty"] += 1
            elif subpool == "broad_pool_fallback":
                counts["broad_pool"] += 1
        return counts

    # `hybrid_two_pool` enriches the generated candidate menu only. The final
    # `hybrid_mi_diverse` acquisition rule still runs later on the rescored pool.
    generate_subpool("ensemble_mi", mi_target, "ensemble_mi", set(existing_sequences), 0)
    generate_subpool(
        "embedding_novelty",
        novelty_target,
        "embedding_novelty",
        set(existing_sequences),
        10000,
    )

    refill_round = 0
    while len(merged_sequences) < required_unique:
        counts = family_counts()
        mi_missing = max(0, mi_target - counts["ensemble_mi"])
        novelty_missing = max(0, novelty_target - counts["embedding_novelty"])
        if mi_missing <= 0 and novelty_missing <= 0:
            break
        if mi_missing >= novelty_missing and mi_missing > 0:
            objective = "ensemble_mi"
            subpool = "ensemble_mi_fill"
            fill_target = min(mi_missing, required_unique - len(merged_sequences))
        else:
            objective = "embedding_novelty"
            subpool = "embedding_novelty_fill"
            fill_target = min(novelty_missing, required_unique - len(merged_sequences))
        exclusions = set(existing_sequences) | set(merged_sequences)
        try:
            generate_subpool(
                objective,
                fill_target,
                subpool,
                exclusions,
                20000 + refill_round * 1000,
                subpool_fill_count=fill_target,
            )
        except RuntimeError:
            fallback_target = required_unique - len(merged_sequences)
            generate_subpool(
                "broad_pool",
                fallback_target,
                "broad_pool_fallback",
                exclusions,
                30000 + refill_round * 1000,
                subpool_fill_count=fallback_target,
            )
        refill_round += 1
        if refill_round > config.ga_max_attempts:
            break

    selected = merged_sequences[:required_unique]
    if len(selected) < minimum_count:
        raise RuntimeError(
            f"Unable to generate {minimum_count} unique peptides after "
            f"{config.ga_max_attempts} hybrid two-pool attempts."
        )

    counts = family_counts()
    rank_by_subpool: dict[str, int] = {}
    for sequence in selected:
        metadata = metadata_by_sequence[sequence]
        subpool = str(metadata.get("generator_subpool", ""))
        rank_by_subpool[subpool] = rank_by_subpool.get(subpool, 0) + 1
        if subpool.startswith("ensemble_mi"):
            unique_count = counts["ensemble_mi"]
        elif subpool.startswith("embedding_novelty"):
            unique_count = counts["embedding_novelty"]
        else:
            unique_count = counts["broad_pool"]
        metadata["subpool_unique_count_after_dedup"] = unique_count
        metadata["deduplicated_count"] = deduplicated_count
        metadata["subpool_rank"] = rank_by_subpool[subpool]

    if return_metadata:
        return selected, {sequence: metadata_by_sequence[sequence] for sequence in selected}
    return selected


def generate_candidate_sequences(
    ensemble,
    existing_sequences: set[str],
    config: RunConfig,
    min_unique: int | None = None,
    seed_offset: int = 0,
    utility_strategy: str | None = None,
    objective: str | None = None,
    incumbent: float = 0.0,
    minimum_return_count: int | None = None,
    family_models=None,
    labeled_embeddings: np.ndarray | None = None,
    use_similarity_penalty: bool | None = None,
    use_length_penalty: bool | None = None,
    return_metadata: bool = False,
    policy_utility_callback: PolicyUtilityCallback | None = None,
    attempt_history: list[dict[str, object]] | None = None,
) -> list[str] | tuple[list[str], dict[str, dict[str, float | str]]]:
    generator_objective = objective or utility_strategy or "ensemble_mean"
    if generator_objective == "hybrid_two_pool":
        return _generate_hybrid_two_pool_sequences(
            ensemble,
            existing_sequences,
            config,
            min_unique=min_unique,
            seed_offset=seed_offset,
            incumbent=incumbent,
            minimum_return_count=minimum_return_count,
            family_models=family_models,
            labeled_embeddings=labeled_embeddings,
            use_similarity_penalty=use_similarity_penalty,
            use_length_penalty=use_length_penalty,
            return_metadata=return_metadata,
            policy_utility_callback=policy_utility_callback,
        )
    return _generate_candidate_sequences_single(
        ensemble,
        existing_sequences,
        config,
        min_unique=min_unique,
        seed_offset=seed_offset,
        utility_strategy=utility_strategy,
        objective=objective,
        incumbent=incumbent,
        minimum_return_count=minimum_return_count,
        family_models=family_models,
        labeled_embeddings=labeled_embeddings,
        use_similarity_penalty=use_similarity_penalty,
        use_length_penalty=use_length_penalty,
        return_metadata=return_metadata,
        policy_utility_callback=policy_utility_callback,
        attempt_history=attempt_history,
    )
