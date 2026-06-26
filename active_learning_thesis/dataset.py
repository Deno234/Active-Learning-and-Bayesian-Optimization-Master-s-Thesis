from __future__ import annotations

import csv
import json
from pathlib import Path
import random

from .config import RunConfig
from .paths import DATASET_PATH


def read_experimental_dataset(dataset_path: Path | None = None) -> list[dict[str, str]]:
    path = dataset_path or DATASET_PATH
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            sequence = row["peptide_sequence"].strip()
            label = row["peptide_label"].strip()
            if label not in {"0", "1"}:
                continue
            records.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "label_source": "experimental",
                }
            )
    duplicates = _duplicate_sequences(records)
    if duplicates:
        raise ValueError(f"Dataset contains duplicate sequences: {duplicates[:5]}")
    return records


def _duplicate_sequences(records: list[dict[str, str]]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        sequence = record["sequence"]
        if sequence in seen:
            duplicates.append(sequence)
        seen.add(sequence)
    return duplicates


def _group_by_label(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {"0": [], "1": []}
    for record in records:
        grouped[record["label"]].append(dict(record))
    return grouped


def _allocate_counts(group_sizes: dict[str, int], total_count: int) -> dict[str, int]:
    total_records = sum(group_sizes.values())
    if total_count > total_records:
        raise ValueError("Requested more samples than records available")
    allocations = {label: 0 for label in group_sizes}
    remainders: list[tuple[float, str]] = []
    for label, size in group_sizes.items():
        if total_records == 0:
            exact = 0.0
        else:
            exact = total_count * size / total_records
        allocations[label] = min(size, int(exact))
        remainders.append((exact - allocations[label], label))
    remaining = total_count - sum(allocations.values())
    remainders.sort(reverse=True)
    for _, label in remainders:
        if remaining == 0:
            break
        if allocations[label] < group_sizes[label]:
            allocations[label] += 1
            remaining -= 1
    return allocations


def _take_stratified(
    records: list[dict[str, str]],
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped = _group_by_label(records)
    for label in grouped:
        rng.shuffle(grouped[label])
    allocations = _allocate_counts(
        {label: len(grouped[label]) for label in grouped}, count
    )
    selected: list[dict[str, str]] = []
    remaining: list[dict[str, str]] = []
    for label, items in grouped.items():
        cutoff = allocations[label]
        selected.extend(items[:cutoff])
        remaining.extend(items[cutoff:])
    return selected, remaining


def _take_stratified_fraction(
    records: list[dict[str, str]],
    fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    target_count = int(round(len(records) * fraction))
    return _take_stratified(records, target_count, rng)


def build_split_manifest(
    records: list[dict[str, str]],
    config: RunConfig,
) -> dict:
    rng = random.Random(config.random_seed)
    holdout_records, development_records = _take_stratified_fraction(
        records, config.holdout_fraction, rng
    )
    validation_records, training_pool_records = _take_stratified_fraction(
        development_records, config.validation_fraction_of_development, rng
    )
    replay_seed_records, replay_hidden_records = _take_stratified(
        training_pool_records, config.replay_seed_size, rng
    )

    split_map = {}
    replay_role_map = {}
    for record in holdout_records:
        split_map[record["sequence"]] = "holdout"
        replay_role_map[record["sequence"]] = "none"
    for record in validation_records:
        split_map[record["sequence"]] = "validation"
        replay_role_map[record["sequence"]] = "none"
    for record in training_pool_records:
        split_map[record["sequence"]] = "train_pool"
        replay_role_map[record["sequence"]] = "hidden"
    for record in replay_seed_records:
        replay_role_map[record["sequence"]] = "seed"
    for record in replay_hidden_records:
        replay_role_map.setdefault(record["sequence"], "hidden")

    manifest = {
        "dataset_path": str(DATASET_PATH),
        "random_seed": config.random_seed,
        "holdout_fraction": config.holdout_fraction,
        "validation_fraction_of_development": config.validation_fraction_of_development,
        "replay_seed_size": config.replay_seed_size,
        "splits": split_map,
        "replay_roles": replay_role_map,
        "counts": {
            "holdout": len(holdout_records),
            "validation": len(validation_records),
            "train_pool": len(training_pool_records),
            "replay_seed": len(replay_seed_records),
            "replay_hidden": len(replay_hidden_records),
        },
    }
    return manifest


def save_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records_for_split(
    records: list[dict[str, str]],
    manifest: dict,
    split: str,
    replay_role: str | None = None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for record in records:
        sequence = record["sequence"]
        if manifest["splits"].get(sequence) != split:
            continue
        if replay_role is not None and manifest["replay_roles"].get(sequence) != replay_role:
            continue
        selected.append(dict(record))
    return selected
