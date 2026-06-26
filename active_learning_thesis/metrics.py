from __future__ import annotations

import math
import numpy as np


BINARY_THRESHOLD_STRATEGIES = ["pr_best_f1", "fixed_0_5"]
BINARY_CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "gmean",
]


def binary_entropy(probabilities, eps: float = 1e-8):
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))


def summarize_ensemble(probability_matrix) -> dict[str, np.ndarray]:
    probs = np.asarray(probability_matrix, dtype=float)
    mean_prob = probs.mean(axis=1)
    predictive_entropy = binary_entropy(mean_prob)
    expected_entropy = binary_entropy(probs).mean(axis=1)
    return {
        "pred_mean": mean_prob,
        "pred_std": probs.std(axis=1),
        "pred_entropy": predictive_entropy,
        "pred_expected_entropy": expected_entropy,
        "pred_mutual_information": predictive_entropy - expected_entropy,
    }


def vote_entropy(probability_matrix, threshold: float = 0.5) -> np.ndarray:
    probs = np.asarray(probability_matrix, dtype=float)
    positive_vote_fraction = (probs >= threshold).mean(axis=1)
    return binary_entropy(positive_vote_fraction)


def probability_std(probability_matrix) -> np.ndarray:
    probs = np.asarray(probability_matrix, dtype=float)
    return probs.std(axis=1)


def farthest_first_indices(
    candidate_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    select_count: int,
) -> list[int]:
    if select_count <= 0 or len(candidate_embeddings) == 0:
        return []
    selected: list[int] = []
    remaining = set(range(len(candidate_embeddings)))
    reference = np.asarray(reference_embeddings, dtype=float)
    candidates = np.asarray(candidate_embeddings, dtype=float)
    if reference.size == 0:
        center = candidates.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(candidates - center, axis=1)
        first_index = int(np.argmax(distances))
    else:
        distances = _min_distances(candidates, reference)
        first_index = int(np.argmax(distances))
    selected.append(first_index)
    remaining.remove(first_index)
    while remaining and len(selected) < select_count:
        reference_block = candidates[selected]
        if reference.size:
            reference_block = np.vstack([reference_block, reference])
        distances = _min_distances(candidates[list(remaining)], reference_block)
        best_local = int(np.argmax(distances))
        best_index = list(remaining)[best_local]
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def _min_distances(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        candidates[:, None, :] - reference[None, :, :],
        axis=2,
    )
    return distances.min(axis=1)


def _binary_metrics_at_threshold(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, len(truth))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2
    gmean = math.sqrt(max(recall * specificity, 0.0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "gmean": gmean,
    }


def pr_best_f1_threshold(y_true, y_scores) -> tuple[float, float]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    if len(truth) == 0 or len(scores) == 0:
        return 0.5, 0.0
    thresholds = np.unique(np.clip(scores, 0.0, 1.0))
    if len(thresholds) == 0:
        return 0.5, 0.0
    best_threshold = float(thresholds[-1])
    best_f1 = -1.0
    # Higher-threshold tie break keeps positive calls as specific as possible.
    for threshold in sorted((float(value) for value in thresholds), reverse=True):
        f1 = _binary_metrics_at_threshold(truth, scores, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, float(max(best_f1, 0.0))


def evaluate_binary_classifier(
    y_true,
    y_scores,
    threshold: float | None = None,
    threshold_strategy: str = "fixed_0_5",
    threshold_source: str = "evaluation_dataset",
    threshold_selection_f1: float | None = None,
) -> dict[str, float | str]:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    if threshold_strategy not in BINARY_THRESHOLD_STRATEGIES:
        allowed = ", ".join(BINARY_THRESHOLD_STRATEGIES)
        raise ValueError(f"threshold_strategy must be one of: {allowed}")

    fixed_metrics = _binary_metrics_at_threshold(truth, scores, 0.5)
    if threshold is not None:
        decision_threshold = float(threshold)
        selected_f1 = (
            float(threshold_selection_f1)
            if threshold_selection_f1 is not None
            else _binary_metrics_at_threshold(truth, scores, decision_threshold)["f1"]
        )
    elif threshold_strategy == "pr_best_f1":
        decision_threshold, selected_f1 = pr_best_f1_threshold(truth, scores)
    else:
        decision_threshold = 0.5
        selected_f1 = fixed_metrics["f1"]

    primary_metrics = _binary_metrics_at_threshold(
        truth,
        scores,
        decision_threshold,
    )

    metrics: dict[str, float | str] = {
        **primary_metrics,
        "roc_auc": roc_auc_score(truth, scores),
        "pr_auc": pr_auc_score(truth, scores),
        "brier_score": brier_score(truth, scores),
        "log_loss": log_loss_score(truth, scores),
        "ece_10": expected_calibration_error(truth, scores, bin_count=10),
        "mce_10": maximum_calibration_error(truth, scores, bin_count=10),
        "decision_threshold": float(decision_threshold),
        "threshold_strategy": threshold_strategy,
        "threshold_selection_f1": float(selected_f1),
        "threshold_source": threshold_source,
    }
    for key in BINARY_CLASSIFICATION_METRICS:
        metrics[f"{key}_fixed_0_5"] = fixed_metrics[key]
    return metrics


def roc_auc_score(y_true, y_scores) -> float:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    positives = scores[truth == 1]
    negatives = scores[truth == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    pairwise = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(pairwise + 0.5 * ties)


def pr_auc_score(y_true, y_scores) -> float:
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    if len(truth) == 0 or np.sum(truth) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[order]
    tp = np.cumsum(sorted_truth == 1)
    fp = np.cumsum(sorted_truth == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    widths = recall[1:] - recall[:-1]
    heights = (precision[1:] + precision[:-1]) / 2
    return float(np.sum(widths * heights))


def brier_score(y_true, y_scores) -> float:
    truth = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_scores, dtype=float)
    if len(truth) == 0:
        return 0.0
    return float(np.mean((scores - truth) ** 2))


def log_loss_score(y_true, y_scores, eps: float = 1e-8) -> float:
    truth = np.asarray(y_true, dtype=float)
    scores = np.clip(np.asarray(y_scores, dtype=float), eps, 1 - eps)
    if len(truth) == 0:
        return 0.0
    losses = -(truth * np.log(scores) + (1 - truth) * np.log(1 - scores))
    return float(np.mean(losses))


def expected_calibration_error(y_true, y_scores, bin_count: int = 10) -> float:
    bin_errors, bin_weights = _calibration_bins(y_true, y_scores, bin_count)
    if len(bin_errors) == 0:
        return 0.0
    return float(np.sum(bin_errors * bin_weights))


def maximum_calibration_error(y_true, y_scores, bin_count: int = 10) -> float:
    bin_errors, _ = _calibration_bins(y_true, y_scores, bin_count)
    if len(bin_errors) == 0:
        return 0.0
    return float(np.max(bin_errors))


def _calibration_bins(y_true, y_scores, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    scores = np.clip(np.asarray(y_scores, dtype=float), 0.0, 1.0)
    if len(truth) == 0 or bin_count <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    errors = []
    weights = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for bin_index in range(bin_count):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == bin_count - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(truth[mask]))
        bin_confidence = float(np.mean(scores[mask]))
        errors.append(abs(bin_accuracy - bin_confidence))
        weights.append(float(np.mean(mask)))
    return np.asarray(errors, dtype=float), np.asarray(weights, dtype=float)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
