from __future__ import annotations

import csv
import json
from pathlib import Path


LEDGER_FIELDS = [
    "sequence",
    "label",
    "label_source",
    "split",
    "mode",
    "round_id",
    "status",
    "pred_mean",
    "pred_std",
    "pred_entropy",
    "pred_expected_entropy",
    "pred_mutual_information",
    "raw_pred_mean",
    "raw_pred_std",
    "raw_pred_entropy",
    "raw_pred_expected_entropy",
    "raw_pred_mutual_information",
    "acquisition_strategy",
    "acquisition_score",
    "selection_rank",
    "pointwise_score",
    "selection_score",
    "cluster_id",
    "distance_to_centroid",
    "distance_to_labeled",
    "oed_gain",
    "diversity_rank",
    "requested_batch_size",
    "candidate_count",
    "requested_cluster_count",
    "non_empty_cluster_count",
    "selected_cluster_count",
    "fallback_fill_count",
    "generator_origin",
    "candidate_source",
    "generator_objective",
    "generator_subpool",
    "subpool_target",
    "subpool_unique_count_after_dedup",
    "subpool_fill_count",
    "deduplicated_count",
    "subpool_rank",
    "normalized_mi",
    "embedding_novelty_raw",
    "normalized_embedding_novelty",
    "generator_utility_score",
    "similarity_penalty",
    "length_penalty",
    "generator_fitness",
    "replay_role",
    "ensemble_member_probs",
    "raw_ensemble_member_probs",
    "family_member_probs",
    "raw_family_member_probs",
    "committee_vote_entropy",
    "committee_prob_std",
]


def create_initial_ledger(records: list[dict[str, str]], manifest: dict) -> list[dict[str, str]]:
    ledger: list[dict[str, str]] = []
    for record in records:
        sequence = record["sequence"]
        split = manifest["splits"][sequence]
        ledger.append(
            empty_row(
                {
                    "sequence": sequence,
                    "label": record["label"],
                    "label_source": "experimental",
                    "split": split,
                    "mode": "experimental",
                    "round_id": "0",
                    "status": split,
                    "generator_origin": "experimental_dataset",
                    "replay_role": manifest["replay_roles"].get(sequence, "none"),
                }
            )
        )
    return ledger


def empty_row(values: dict | None = None) -> dict[str, str]:
    row = {field: "" for field in LEDGER_FIELDS}
    if values:
        for key, value in values.items():
            if key not in row:
                row[key] = value
            else:
                row[key] = str(value)
    return row


def save_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = empty_row(row)
            writer.writerow(normalized)


def load_ledger(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [empty_row(row) for row in reader]


def snapshot_ledger(snapshot_dir: Path, rows: list[dict[str, str]], name: str) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{name}.csv"
    save_ledger(path, rows)
    return path


def index_by_sequence(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sequence"]: row for row in rows}


def append_rows(rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> None:
    seen = {row["sequence"] for row in rows}
    for row in new_rows:
        if row["sequence"] in seen:
            raise ValueError(f"Sequence already exists in ledger: {row['sequence']}")
        rows.append(empty_row(row))
        seen.add(row["sequence"])


def current_real_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    training: list[dict[str, str]] = []
    for row in rows:
        if not row["label"]:
            continue
        if row["split"] == "train_pool" and row["label_source"] == "experimental":
            training.append(row)
        if row["split"] == "generated" and row["status"] == "acquired":
            training.append(row)
    return training


def validation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == "validation" and row["label"]]


def holdout_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == "holdout" and row["label"]]


def replay_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["split"] == "train_pool" and row["replay_role"] == "seed" and row["label"]
    ]


def replay_hidden_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["split"] == "train_pool" and row["replay_role"] == "hidden" and row["label"]
    ]


def unresolved_proposals(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "proposed"]


def next_real_round_id(rows: list[dict[str, str]]) -> int:
    generated_rounds = [
        int(row["round_id"])
        for row in rows
        if row["split"] == "generated" and row["round_id"]
    ]
    return max(generated_rounds, default=0) + 1


def serialize_probabilities(values) -> str:
    return json.dumps([float(value) for value in values])


def deserialize_probabilities(value: str) -> list[float]:
    if not value:
        return []
    return [float(item) for item in json.loads(value)]
