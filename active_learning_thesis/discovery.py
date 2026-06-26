from __future__ import annotations

import math

import numpy as np

from active_learning_thesis.config import RunConfig


DISCOVERY_STRATEGIES = {
    "ensemble_mean",
    "ucb",
    "ei",
    "pi",
    "mes",
}


def discovery_utility_scores(
    strategy: str,
    pred_mean,
    pred_std,
    incumbent: float,
    config: RunConfig,
    seed: int = 0,
) -> np.ndarray:
    mean = np.asarray(pred_mean, dtype=float)
    std = np.asarray(pred_std, dtype=float)
    safe_std = np.maximum(std, 1e-8)

    if strategy == "ensemble_mean":
        return mean.copy()

    if strategy == "ucb":
        return mean + config.discovery_ucb_beta * std

    improvement = mean - float(incumbent) - config.discovery_improvement_xi
    z_scores = improvement / safe_std

    if strategy == "pi":
        scores = _normal_cdf(z_scores)
        zero_mask = std <= 1e-8
        if np.any(zero_mask):
            scores = np.asarray(scores, dtype=float)
            scores[zero_mask] = (improvement[zero_mask] > 0).astype(float)
        return scores

    if strategy == "ei":
        scores = improvement * _normal_cdf(z_scores) + safe_std * _normal_pdf(z_scores)
        zero_mask = std <= 1e-8
        if np.any(zero_mask):
            scores = np.asarray(scores, dtype=float)
            scores[zero_mask] = np.maximum(improvement[zero_mask], 0.0)
        return scores

    if strategy == "mes":
        return _mes_scores(mean, std, sample_count=config.discovery_mes_samples, seed=seed)

    raise ValueError(f"Unsupported discovery strategy: {strategy}")


def min_distances_to_reference(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    candidates = np.asarray(candidate_embeddings, dtype=float)
    reference = np.asarray(reference_embeddings, dtype=float)
    if len(candidates) == 0:
        return np.empty(0, dtype=float)
    if reference.size == 0:
        return np.zeros(len(candidates), dtype=float)
    distances = np.linalg.norm(
        candidates[:, None, :] - reference[None, :, :],
        axis=2,
    )
    return distances.min(axis=1)


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    vectors = np.asarray(embeddings, dtype=float)
    if len(vectors) < 2:
        return 0.0
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    upper = distances[np.triu_indices(len(vectors), k=1)]
    if len(upper) == 0:
        return 0.0
    return float(upper.mean())


def _mes_scores(
    mean: np.ndarray,
    std: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    if len(mean) == 0:
        return np.empty(0, dtype=float)
    safe_std = np.maximum(std, 1e-8)
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mean, scale=safe_std, size=(sample_count, len(mean)))
    sampled_maxima = draws.max(axis=1)
    gamma = (sampled_maxima[:, None] - mean[None, :]) / safe_std[None, :]
    cdf = np.clip(_normal_cdf(gamma), 1e-12, 1.0)
    pdf = _normal_pdf(gamma)
    scores = ((gamma * pdf) / (2.0 * cdf) - np.log(cdf)).mean(axis=0)
    scores = np.asarray(scores, dtype=float)
    scores[std <= 1e-8] = 0.0
    return scores


def _normal_pdf(values) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(values) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    vectorized_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + vectorized_erf(z / math.sqrt(2.0)))