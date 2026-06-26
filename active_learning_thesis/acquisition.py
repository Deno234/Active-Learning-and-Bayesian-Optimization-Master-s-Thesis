from __future__ import annotations

import math
import random

import numpy as np

from active_learning_thesis.config import RunConfig
from active_learning_thesis.discovery import discovery_utility_scores


def requires_family_models(strategy: str) -> bool:
    return strategy == "family_qbc"


def requires_embeddings(strategy: str) -> bool:
    return strategy in {
        "cluster_representative",
        "cluster_diverse_representative",
        "embedding_farthest_first",
        "oed_logdet",
        "hybrid_mi_diverse",
    }


POINTWISE_GENERATOR_OBJECTIVES = {
    "ensemble_mean",
    "predictive_entropy",
    "ensemble_mi",
    "embedding_novelty",
    "family_qbc",
    "ucb",
    "ei",
    "pi",
    "mes",
}

ACQUISITION_SELECTION_METADATA_FIELDS = [
    "requested_batch_size",
    "candidate_count",
    "requested_cluster_count",
    "non_empty_cluster_count",
    "selected_cluster_count",
    "fallback_fill_count",
]

ACQUISITION_DIAGNOSTIC_FIELDS = [
    "selection_rank",
    "pointwise_score",
    "selection_score",
    "cluster_id",
    "distance_to_centroid",
    "distance_to_labeled",
    "oed_gain",
    "diversity_rank",
    *ACQUISITION_SELECTION_METADATA_FIELDS,
]

POINTWISE_SELECTION_STRATEGIES = {
    "ensemble_mean",
    "predictive_entropy",
    "ensemble_mi",
    "ucb",
    "family_qbc",
}


def generator_objective_for_strategy(
    strategy: str,
    mode: str,
    config: RunConfig,
) -> str:
    if mode == "fixed_mean":
        return "ensemble_mean"
    if mode == "broad_pool":
        return "broad_pool"
    if mode == "bo_utility":
        return (config.discovery_strategies or ["ucb"])[0]
    if mode != "match_acquisition":
        raise ValueError(f"Unsupported generator objective mode: {mode}")

    if strategy in {
        "random",
        "cluster_representative",
        "embedding_farthest_first",
    }:
        return "broad_pool"
    if strategy == "hybrid_mi_diverse":
        return "hybrid_two_pool"
    if strategy == "cluster_diverse_representative":
        return "embedding_novelty"
    if strategy == "similarity_penalized_mean":
        return "ensemble_mean"
    if strategy in {
        "ensemble_mean",
        "predictive_entropy",
        "ensemble_mi",
        "family_qbc",
        "oed_logdet",
        "ucb",
        "ei",
        "pi",
        "mes",
    }:
        return strategy
    raise ValueError(f"Unsupported acquisition strategy for generation: {strategy}")


def generator_objective_requires_family_models(objective: str) -> bool:
    return objective == "family_qbc"


def generator_objective_requires_embeddings(objective: str) -> bool:
    return objective in {"embedding_novelty", "hybrid_two_pool", "oed_logdet"}


def embedding_novelty_scores(
    candidate_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_embeddings = np.asarray(candidate_embeddings, dtype=float)
    labeled_embeddings = np.asarray(labeled_embeddings, dtype=float)
    if candidate_embeddings.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(-1, 1)
    if labeled_embeddings.size and labeled_embeddings.ndim == 1:
        labeled_embeddings = labeled_embeddings.reshape(-1, candidate_embeddings.shape[1])

    if labeled_embeddings.size:
        distances = np.linalg.norm(
            candidate_embeddings[:, None, :] - labeled_embeddings[None, :, :],
            axis=2,
        )
        raw_scores = distances.min(axis=1)
    else:
        # Edge-case fallback: when no labeled embeddings exist, encourage a broad
        # candidate cloud by scoring distance from the candidate centroid.
        centroid = candidate_embeddings.mean(axis=0)
        raw_scores = np.linalg.norm(candidate_embeddings - centroid, axis=1)

    raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    max_score = float(raw_scores.max()) if raw_scores.size else 0.0
    if max_score > 0.0:
        normalized_scores = raw_scores / max_score
    else:
        normalized_scores = np.zeros_like(raw_scores, dtype=float)
    return raw_scores, normalized_scores


def single_candidate_oed_logdet_scores(
    candidate_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    regularization: float,
) -> np.ndarray:
    if candidate_embeddings.size == 0:
        return np.zeros(0, dtype=float)
    embedding_dim = candidate_embeddings.shape[1]
    base_matrix = regularization * np.eye(embedding_dim)
    if labeled_embeddings.size:
        base_matrix = base_matrix + labeled_embeddings.T @ labeled_embeddings
    base_logdet = _safe_logdet(base_matrix)
    scores = np.asarray(
        [
            _safe_logdet(base_matrix + np.outer(embedding, embedding)) - base_logdet
            for embedding in candidate_embeddings
        ],
        dtype=float,
    )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def candidate_objective_scores(
    objective: str,
    candidate_scores: dict[str, np.ndarray],
    labeled_embeddings: np.ndarray,
    config: RunConfig,
    seed: int,
    incumbent: float = 0.0,
) -> np.ndarray:
    candidate_count = len(candidate_scores["pred_mean"])
    if objective == "broad_pool":
        return np.zeros(candidate_count, dtype=float)
    if objective == "ensemble_mean":
        return np.asarray(candidate_scores["pred_mean"], dtype=float)
    if objective == "predictive_entropy":
        return np.asarray(candidate_scores["pred_entropy"], dtype=float)
    if objective == "ensemble_mi":
        return np.asarray(candidate_scores["pred_mutual_information"], dtype=float)
    if objective == "family_qbc":
        return np.asarray(candidate_scores["committee_vote_entropy"], dtype=float)
    if objective == "embedding_novelty":
        _, normalized_scores = embedding_novelty_scores(
            np.asarray(candidate_scores["avg_embedding"], dtype=float),
            np.asarray(labeled_embeddings, dtype=float),
        )
        return normalized_scores
    if objective == "oed_logdet":
        return single_candidate_oed_logdet_scores(
            np.asarray(candidate_scores["avg_embedding"], dtype=float),
            np.asarray(labeled_embeddings, dtype=float),
            config.oed_regularization,
        )
    if objective in {"ucb", "ei", "pi", "mes"}:
        return discovery_utility_scores(
            objective,
            candidate_scores["pred_mean"],
            candidate_scores["pred_std"],
            incumbent,
            config,
            seed=seed,
        )
    raise ValueError(f"Unsupported generator objective: {objective}")


def _descending_indices(values: np.ndarray, tie_breaker: np.ndarray | None = None) -> list[int]:
    if tie_breaker is None:
        return list(np.argsort(-values, kind="mergesort"))
    structured = np.lexsort(( -tie_breaker, -values ))
    return list(structured)


def _amino_acid_frequencies(sequence: str, allowed_amino_acids: str) -> np.ndarray:
    return np.asarray(
        [sequence.count(amino_acid) for amino_acid in allowed_amino_acids],
        dtype=float,
    )


def _similarity_penalty_to_references(
    sequence: str,
    reference_sequences: list[str],
    allowed_amino_acids: str,
) -> float:
    if not reference_sequences:
        return 0.0
    frequencies = _amino_acid_frequencies(sequence, allowed_amino_acids)
    penalties: list[float] = []
    for reference_sequence in reference_sequences:
        reference_frequencies = _amino_acid_frequencies(
            reference_sequence,
            allowed_amino_acids,
        )
        denominator = float(np.sum(frequencies) + np.sum(reference_frequencies))
        if denominator == 0:
            continue
        penalties.append(
            0.1
            * (
                1
                - float(np.sum(np.abs(frequencies - reference_frequencies)))
                / denominator
            )
        )
    if not penalties:
        return 0.0
    return float(np.mean(penalties))


def _validate_sequence_context(
    strategy: str,
    candidate_sequences: list[str] | None,
    reference_sequences: list[str] | None,
    candidate_count: int,
) -> tuple[list[str], list[str]]:
    if candidate_sequences is None or reference_sequences is None:
        raise ValueError(
            f"{strategy} requires candidate_sequences and reference_sequences "
            "for sequence-composition similarity scoring."
        )
    if len(candidate_sequences) != candidate_count:
        raise ValueError(
            f"{strategy} received {len(candidate_sequences)} candidate_sequences "
            f"for {candidate_count} scored candidates."
        )
    return list(candidate_sequences), list(reference_sequences)


def _similarity_penalized_mean_selection(
    candidate_scores: dict[str, np.ndarray],
    batch_size: int,
    config: RunConfig,
    candidate_sequences: list[str] | None,
    reference_sequences: list[str] | None,
    apply_similarity_penalty: bool,
) -> dict[str, object]:
    mean_scores = np.asarray(candidate_scores["pred_mean"], dtype=float)
    candidate_count = len(mean_scores)
    sequences, references = _validate_sequence_context(
        "similarity_penalized_mean",
        candidate_sequences,
        reference_sequences,
        candidate_count,
    )
    acquisition_scores = mean_scores.copy()
    similarity_penalties = np.zeros(candidate_count, dtype=float)
    if candidate_count == 0 or batch_size <= 0:
        return {
            "selected_indices": [],
            "acquisition_scores": acquisition_scores,
            "similarity_penalties": similarity_penalties,
        }

    if not apply_similarity_penalty:
        ordered = list(np.argsort(-mean_scores, kind="mergesort")[:batch_size])
        return {
            "selected_indices": ordered,
            "acquisition_scores": acquisition_scores,
            "similarity_penalties": similarity_penalties,
        }

    selected: list[int] = []
    remaining = set(range(candidate_count))
    current_references = list(references)
    while remaining and len(selected) < min(batch_size, candidate_count):
        best_index = None
        best_key: tuple[float, float, int] | None = None
        for index in sorted(remaining):
            penalty = _similarity_penalty_to_references(
                sequences[index],
                current_references,
                config.allowed_amino_acids,
            )
            score = float(mean_scores[index]) - penalty
            acquisition_scores[index] = score
            similarity_penalties[index] = penalty
            key = (score, float(mean_scores[index]), -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        current_references.append(sequences[best_index])

    return {
        "selected_indices": selected,
        "acquisition_scores": acquisition_scores,
        "similarity_penalties": similarity_penalties,
    }


def _kmeans(
    embeddings: np.ndarray,
    cluster_count: int,
    seed: int,
    max_iters: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) < cluster_count:
        raise ValueError("cluster_count cannot exceed the number of embeddings")
    rng = np.random.default_rng(seed)
    centroids = [embeddings[rng.integers(len(embeddings))]]
    while len(centroids) < cluster_count:
        distances = np.min(
            np.linalg.norm(
                embeddings[:, None, :] - np.asarray(centroids)[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        next_index = int(np.argmax(distances))
        centroids.append(embeddings[next_index])
    centroids = np.asarray(centroids, dtype=float)

    assignments = np.zeros(len(embeddings), dtype=int)
    for _ in range(max_iters):
        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_index in range(cluster_count):
            members = embeddings[assignments == cluster_index]
            if len(members) == 0:
                centroids[cluster_index] = embeddings[rng.integers(len(embeddings))]
            else:
                centroids[cluster_index] = members.mean(axis=0)
    return centroids, assignments


def _cluster_representative_indices(
    embeddings: np.ndarray,
    batch_size: int,
    seed: int,
) -> list[int]:
    centroids, assignments = _kmeans(embeddings, batch_size, seed)
    selected: list[int] = []
    available = set(range(len(embeddings)))
    for cluster_index in range(batch_size):
        cluster_members = np.where(assignments == cluster_index)[0]
        if len(cluster_members) == 0:
            continue
        distances = np.linalg.norm(
            embeddings[cluster_members] - centroids[cluster_index],
            axis=1,
        )
        ordered_members = cluster_members[np.argsort(distances)]
        chosen = None
        for member_index in ordered_members:
            member_index = int(member_index)
            if member_index in available:
                chosen = member_index
                break
        if chosen is not None:
            selected.append(chosen)
            available.remove(chosen)
    if len(selected) < batch_size:
        remaining = sorted(available)
        selected.extend(remaining[: batch_size - len(selected)])
    return selected


def _cluster_diverse_representative_selection(
    candidate_scores: dict[str, np.ndarray],
    labeled_embeddings: np.ndarray,
    batch_size: int,
    config: RunConfig,
    seed: int,
) -> dict[str, object]:
    embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)
    candidate_count = len(embeddings)
    acquisition_scores = np.zeros(candidate_count, dtype=float)
    requested_batch_size = min(batch_size, candidate_count)
    metadata = {
        "requested_batch_size": requested_batch_size,
        "candidate_count": candidate_count,
        "requested_cluster_count": 0,
        "non_empty_cluster_count": 0,
        "selected_cluster_count": 0,
        "fallback_fill_count": 0,
    }
    empty_selection = {
        "selected_indices": [],
        "acquisition_scores": acquisition_scores,
        "metadata": metadata,
        "assignments": np.zeros(candidate_count, dtype=int),
        "distance_to_centroid": np.zeros(candidate_count, dtype=float),
        "cluster_novelty_by_index": acquisition_scores.copy(),
    }
    if candidate_count == 0 or requested_batch_size <= 0:
        return empty_selection

    requested_cluster_count = min(
        candidate_count,
        max(
            requested_batch_size,
            math.ceil(config.diversity_prefilter_multiplier * requested_batch_size),
        ),
    )
    metadata["requested_cluster_count"] = requested_cluster_count
    centroids, assignments = _kmeans(embeddings, requested_cluster_count, seed)

    labeled = np.asarray(labeled_embeddings, dtype=float)
    global_centroid = embeddings.mean(axis=0)
    distance_to_centroid = np.zeros(candidate_count, dtype=float)
    cluster_rows: list[dict[str, float | int]] = []
    cluster_novelty: dict[int, float] = {}

    for cluster_index in range(requested_cluster_count):
        members = np.where(assignments == cluster_index)[0]
        if len(members) == 0:
            continue
        member_distances = np.linalg.norm(
            embeddings[members] - centroids[cluster_index],
            axis=1,
        )
        for member_index, distance in zip(members, member_distances):
            distance_to_centroid[int(member_index)] = float(distance)
        representative_order = sorted(
            (float(distance), int(member_index))
            for member_index, distance in zip(members, member_distances)
        )
        representative_distance, representative_index = representative_order[0]
        if labeled.size:
            novelty = float(
                _min_distances(
                    centroids[cluster_index : cluster_index + 1],
                    labeled,
                )[0]
            )
        else:
            novelty = float(np.linalg.norm(centroids[cluster_index] - global_centroid))
        cluster_novelty[cluster_index] = novelty
        cluster_rows.append(
            {
                "cluster_id": int(cluster_index),
                "novelty": novelty,
                "representative_distance": representative_distance,
                "representative_index": representative_index,
            }
        )

    metadata["non_empty_cluster_count"] = len(cluster_rows)
    for index, assignment in enumerate(assignments):
        acquisition_scores[index] = cluster_novelty.get(int(assignment), 0.0)

    ordered_clusters = sorted(
        cluster_rows,
        key=lambda row: (
            -float(row["novelty"]),
            float(row["representative_distance"]),
            int(row["representative_index"]),
        ),
    )
    selected: list[int] = []
    seen: set[int] = set()
    for row in ordered_clusters:
        if len(selected) >= requested_batch_size:
            break
        representative_index = int(row["representative_index"])
        if representative_index in seen:
            continue
        selected.append(representative_index)
        seen.add(representative_index)

    metadata["selected_cluster_count"] = len(selected)
    if len(selected) < requested_batch_size:
        fill_candidates = sorted(
            (
                float(distance_to_centroid[index]),
                int(index),
            )
            for index in range(candidate_count)
            if index not in seen
        )
        for _, index in fill_candidates:
            if len(selected) >= requested_batch_size:
                break
            selected.append(index)
            seen.add(index)
    metadata["fallback_fill_count"] = len(selected) - int(metadata["selected_cluster_count"])

    return {
        "selected_indices": selected,
        "acquisition_scores": acquisition_scores,
        "metadata": metadata,
        "assignments": assignments,
        "distance_to_centroid": distance_to_centroid,
        "cluster_novelty_by_index": acquisition_scores.copy(),
    }


def _min_distances_to_labeled(
    candidate_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
) -> np.ndarray | None:
    candidates = np.asarray(candidate_embeddings, dtype=float)
    labeled = np.asarray(labeled_embeddings, dtype=float)
    if candidates.size == 0 or labeled.size == 0:
        return None
    distances = np.linalg.norm(
        candidates[:, None, :] - labeled[None, :, :],
        axis=2,
    )
    return distances.min(axis=1)


def _min_distances(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        candidate_embeddings[:, None, :] - reference_embeddings[None, :, :],
        axis=2,
    )
    return distances.min(axis=1)


def _embedding_farthest_first_indices(
    candidate_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    batch_size: int,
) -> tuple[list[int], np.ndarray]:
    """Greedy diversity-only traversal with per-selection dynamic distances."""
    candidates = np.asarray(candidate_embeddings, dtype=float)
    reference = np.asarray(labeled_embeddings, dtype=float)
    scores = np.zeros(len(candidates), dtype=float)
    if len(candidates) == 0 or batch_size <= 0:
        return [], scores

    select_count = min(batch_size, len(candidates))
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    if reference.size == 0:
        center = candidates.mean(axis=0, keepdims=True)
        first_distances = np.linalg.norm(candidates - center, axis=1)
    else:
        first_distances = _min_distances(candidates, reference)
    first_index = int(np.argmax(first_distances))
    selected.append(first_index)
    remaining.remove(first_index)
    scores[first_index] = float(first_distances[first_index])

    while remaining and len(selected) < select_count:
        reference_block = candidates[selected]
        if reference.size:
            reference_block = np.vstack([reference_block, reference])
        remaining_indices = list(remaining)
        distances = _min_distances(candidates[remaining_indices], reference_block)
        best_local = int(np.argmax(distances))
        best_index = remaining_indices[best_local]
        selected.append(best_index)
        remaining.remove(best_index)
        scores[best_index] = float(distances[best_local])

    return selected, scores


def _hybrid_mi_diverse_indices(
    candidate_scores: dict[str, np.ndarray],
    labeled_embeddings: np.ndarray,
    batch_size: int,
) -> tuple[list[int], np.ndarray]:
    mi_scores = np.asarray(candidate_scores["pred_mutual_information"], dtype=float)
    embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)
    acquisition_scores = mi_scores.copy()
    if len(mi_scores) == 0 or batch_size <= 0:
        return [], acquisition_scores

    uncertainty_count = min(len(mi_scores), math.ceil(batch_size / 2))
    uncertainty_indices = list(
        np.argsort(-mi_scores, kind="mergesort")[:uncertainty_count]
    )
    diversity_count = min(
        batch_size - uncertainty_count,
        len(mi_scores) - len(uncertainty_indices),
    )
    if diversity_count <= 0:
        return uncertainty_indices, acquisition_scores

    remaining_indices = [
        index for index in range(len(mi_scores)) if index not in set(uncertainty_indices)
    ]
    reference = np.asarray(labeled_embeddings, dtype=float)
    uncertainty_embeddings = embeddings[uncertainty_indices]
    if reference.size:
        reference = np.vstack([reference, uncertainty_embeddings])
    else:
        reference = uncertainty_embeddings
    diversity_local, _ = _embedding_farthest_first_indices(
        embeddings[remaining_indices],
        reference,
        diversity_count,
    )
    diversity_indices = [int(remaining_indices[index]) for index in diversity_local]
    return uncertainty_indices + diversity_indices, acquisition_scores


def _safe_logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    return float(value) if sign > 0 else -np.inf


def _oed_indices(
    candidate_embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
    batch_size: int,
    regularization: float,
) -> tuple[list[int], np.ndarray]:
    embedding_dim = candidate_embeddings.shape[1]
    base_matrix = regularization * np.eye(embedding_dim)
    if labeled_embeddings.size:
        base_matrix = base_matrix + labeled_embeddings.T @ labeled_embeddings
    selected: list[int] = []
    base_logdet = _safe_logdet(base_matrix)
    scores = np.asarray(
        [
            _safe_logdet(base_matrix + np.outer(embedding, embedding)) - base_logdet
            for embedding in candidate_embeddings
        ],
        dtype=float,
    )
    chosen = set()
    current_matrix = base_matrix.copy()
    for _ in range(min(batch_size, len(candidate_embeddings))):
        current_logdet = _safe_logdet(current_matrix)
        best_index = None
        best_gain = -np.inf
        for index, embedding in enumerate(candidate_embeddings):
            if index in chosen:
                continue
            proposal_matrix = current_matrix + np.outer(embedding, embedding)
            proposal_logdet = _safe_logdet(proposal_matrix)
            gain = proposal_logdet - current_logdet
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        chosen.add(best_index)
        current_matrix = current_matrix + np.outer(
            candidate_embeddings[best_index],
            candidate_embeddings[best_index],
        )
        scores[best_index] = best_gain
    return selected, scores


def acquisition_diagnostics(
    strategy: str,
    selected_indices: list[int],
    candidate_scores: dict[str, np.ndarray],
    labeled_embeddings: np.ndarray,
    acquisition_scores: np.ndarray,
    config: RunConfig,
    seed: int,
    *,
    candidate_sequences: list[str] | None = None,
    reference_sequences: list[str] | None = None,
    apply_similarity_penalty: bool = True,
) -> list[dict[str, float | int | str]]:
    """Return reporting-only diagnostics without changing selection decisions.

    For hybrid_mi_diverse, selection_score intentionally remains the pointwise
    MI score. The diversity effect is represented by diversity_rank and, when
    embeddings are available, distance_to_labeled.
    """
    candidate_count = len(candidate_scores["pred_mean"])
    diagnostics: list[dict[str, float | int | str]] = [
        {field: "" for field in ACQUISITION_DIAGNOSTIC_FIELDS}
        for _ in range(candidate_count)
    ]
    for rank, index in enumerate(selected_indices, start=1):
        diagnostics[index]["selection_rank"] = rank

    scores = np.asarray(acquisition_scores, dtype=float)
    if strategy in POINTWISE_SELECTION_STRATEGIES:
        for index in range(candidate_count):
            diagnostics[index]["pointwise_score"] = float(scores[index])
            diagnostics[index]["selection_score"] = float(scores[index])
        return diagnostics

    if strategy == "random":
        return diagnostics

    if strategy == "similarity_penalized_mean":
        selection = _similarity_penalized_mean_selection(
            candidate_scores,
            len(selected_indices),
            config,
            candidate_sequences,
            reference_sequences,
            apply_similarity_penalty,
        )
        mean_scores = np.asarray(candidate_scores["pred_mean"], dtype=float)
        selection_scores = np.asarray(selection["acquisition_scores"], dtype=float)
        similarity_penalties = np.asarray(
            selection["similarity_penalties"],
            dtype=float,
        )
        for index in range(candidate_count):
            diagnostics[index]["pointwise_score"] = float(mean_scores[index])
            diagnostics[index]["selection_score"] = float(selection_scores[index])
            if apply_similarity_penalty:
                diagnostics[index]["similarity_penalty"] = float(
                    similarity_penalties[index]
                )
        return diagnostics

    embeddings = np.asarray(candidate_scores.get("avg_embedding", []), dtype=float)
    if embeddings.size:
        distances_to_labeled = _min_distances_to_labeled(embeddings, labeled_embeddings)
        if distances_to_labeled is not None:
            for index, distance in enumerate(distances_to_labeled):
                diagnostics[index]["distance_to_labeled"] = float(distance)

    if strategy == "cluster_diverse_representative":
        if embeddings.size:
            selection = _cluster_diverse_representative_selection(
                candidate_scores,
                labeled_embeddings,
                len(selected_indices),
                config,
                seed,
            )
            assignments = np.asarray(selection["assignments"], dtype=int)
            distance_to_centroid = np.asarray(
                selection["distance_to_centroid"],
                dtype=float,
            )
            cluster_novelty_by_index = np.asarray(
                selection["cluster_novelty_by_index"],
                dtype=float,
            )
            metadata = dict(selection["metadata"])
            for index in range(candidate_count):
                diagnostics[index]["pointwise_score"] = float(
                    candidate_scores["pred_mean"][index]
                )
                diagnostics[index]["selection_score"] = float(
                    cluster_novelty_by_index[index]
                )
                diagnostics[index]["cluster_id"] = int(assignments[index])
                diagnostics[index]["distance_to_centroid"] = float(
                    distance_to_centroid[index]
                )
                for field, value in metadata.items():
                    diagnostics[index][field] = value
        return diagnostics

    if strategy == "cluster_representative":
        if embeddings.size:
            cluster_count = min(len(embeddings), max(1, len(selected_indices)))
            centroids, assignments = _kmeans(embeddings, cluster_count, seed)
            for index, assignment in enumerate(assignments):
                distance = float(np.linalg.norm(embeddings[index] - centroids[assignment]))
                diagnostics[index]["pointwise_score"] = float(candidate_scores["pred_mean"][index])
                diagnostics[index]["cluster_id"] = int(assignment)
                diagnostics[index]["distance_to_centroid"] = distance
                diagnostics[index]["selection_score"] = -distance
        return diagnostics

    if strategy == "oed_logdet":
        for index in selected_indices:
            gain = float(scores[index])
            diagnostics[index]["oed_gain"] = gain
            diagnostics[index]["selection_score"] = gain
        return diagnostics

    if strategy == "embedding_farthest_first":
        for rank, index in enumerate(selected_indices, start=1):
            diagnostics[index]["selection_score"] = float(scores[index])
            diagnostics[index]["diversity_rank"] = rank
        return diagnostics

    if strategy == "hybrid_mi_diverse":
        mi_scores = np.asarray(candidate_scores["pred_mutual_information"], dtype=float)
        for index in range(candidate_count):
            diagnostics[index]["pointwise_score"] = float(mi_scores[index])
            diagnostics[index]["selection_score"] = float(mi_scores[index])
        uncertainty_count = min(len(selected_indices), math.ceil(len(selected_indices) / 2))
        uncertainty_indices = selected_indices[:uncertainty_count]
        diversity_indices = selected_indices[uncertainty_count:]
        if diversity_indices and embeddings.size:
            remaining_indices = [
                index
                for index in range(candidate_count)
                if index not in set(uncertainty_indices)
            ]
            reference = np.asarray(labeled_embeddings, dtype=float)
            uncertainty_embeddings = embeddings[uncertainty_indices]
            if reference.size:
                reference = np.vstack([reference, uncertainty_embeddings])
            else:
                reference = uncertainty_embeddings
            diversity_local, diversity_scores = _embedding_farthest_first_indices(
                embeddings[remaining_indices],
                reference,
                len(diversity_indices),
            )
            dynamic_scores = {
                int(remaining_indices[local_index]): float(diversity_scores[local_index])
                for local_index in diversity_local
            }
            for rank, index in enumerate(diversity_indices, start=1):
                diagnostics[index]["diversity_rank"] = rank
                if index in dynamic_scores:
                    diagnostics[index]["selection_score"] = dynamic_scores[index]
        return diagnostics

    return diagnostics


def select_batch(
    strategy: str,
    batch_size: int,
    candidate_scores: dict[str, np.ndarray],
    labeled_embeddings: np.ndarray,
    config: RunConfig,
    seed: int,
    *,
    candidate_sequences: list[str] | None = None,
    reference_sequences: list[str] | None = None,
    apply_similarity_penalty: bool = True,
) -> tuple[list[int], np.ndarray]:
    candidate_count = len(candidate_scores["pred_mean"])
    if batch_size > candidate_count:
        batch_size = candidate_count
    acquisition_scores = np.zeros(candidate_count, dtype=float)

    if strategy == "random":
        rng = random.Random(seed)
        indices = list(range(candidate_count))
        rng.shuffle(indices)
        return indices[:batch_size], acquisition_scores

    if strategy == "ensemble_mean":
        acquisition_scores = np.asarray(candidate_scores["pred_mean"], dtype=float)
        indices = list(np.argsort(-acquisition_scores, kind="mergesort")[:batch_size])
        return indices, acquisition_scores

    if strategy == "predictive_entropy":
        acquisition_scores = np.asarray(candidate_scores["pred_entropy"], dtype=float)
        indices = list(np.argsort(-acquisition_scores, kind="mergesort")[:batch_size])
        return indices, acquisition_scores

    if strategy == "ensemble_mi":
        acquisition_scores = np.asarray(
            candidate_scores["pred_mutual_information"],
            dtype=float,
        )
        indices = list(np.argsort(-acquisition_scores, kind="mergesort")[:batch_size])
        return indices, acquisition_scores

    if strategy == "ucb":
        mean_scores = np.asarray(candidate_scores["pred_mean"], dtype=float)
        std_scores = np.asarray(candidate_scores["pred_std"], dtype=float)
        acquisition_scores = mean_scores + config.discovery_ucb_beta * std_scores
        indices = list(np.argsort(-acquisition_scores, kind="mergesort")[:batch_size])
        return indices, acquisition_scores

    if strategy == "family_qbc":
        vote_scores = np.asarray(candidate_scores["committee_vote_entropy"], dtype=float)
        std_scores = np.asarray(candidate_scores["committee_prob_std"], dtype=float)
        acquisition_scores = vote_scores
        ordered = _descending_indices(vote_scores, std_scores)
        return ordered[:batch_size], acquisition_scores

    if strategy == "similarity_penalized_mean":
        selection = _similarity_penalized_mean_selection(
            candidate_scores,
            batch_size,
            config,
            candidate_sequences,
            reference_sequences,
            apply_similarity_penalty,
        )
        return (
            list(selection["selected_indices"]),
            np.asarray(selection["acquisition_scores"], dtype=float),
        )

    if strategy == "cluster_representative":
        acquisition_scores = np.asarray(candidate_scores["pred_mean"], dtype=float)
        embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)
        return _cluster_representative_indices(embeddings, batch_size, seed), acquisition_scores

    if strategy == "cluster_diverse_representative":
        selection = _cluster_diverse_representative_selection(
            candidate_scores,
            labeled_embeddings,
            batch_size,
            config,
            seed,
        )
        return (
            list(selection["selected_indices"]),
            np.asarray(selection["acquisition_scores"], dtype=float),
        )

    if strategy == "oed_logdet":
        embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)
        labeled = np.asarray(labeled_embeddings, dtype=float)
        indices, acquisition_scores = _oed_indices(
            embeddings,
            labeled,
            batch_size,
            config.oed_regularization,
        )
        return indices, acquisition_scores

    if strategy == "embedding_farthest_first":
        embeddings = np.asarray(candidate_scores["avg_embedding"], dtype=float)
        labeled = np.asarray(labeled_embeddings, dtype=float)
        return _embedding_farthest_first_indices(embeddings, labeled, batch_size)

    if strategy == "hybrid_mi_diverse":
        return _hybrid_mi_diverse_indices(
            candidate_scores,
            labeled_embeddings,
            batch_size,
        )

    raise ValueError(f"Unsupported acquisition strategy: {strategy}")
