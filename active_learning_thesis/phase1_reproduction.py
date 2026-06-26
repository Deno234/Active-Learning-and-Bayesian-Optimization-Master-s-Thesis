from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shlex
import shutil
import socket
import sys
import time
from typing import Sequence

import numpy as np

from active_learning_thesis.config import RunConfig
from active_learning_thesis.dataset import read_experimental_dataset
from active_learning_thesis.metrics import evaluate_binary_classifier, pr_best_f1_threshold
from active_learning_thesis.paths import DATASET_PATH, PREDICTIVE_MODEL_DIR


PHASE1_MODELS = ["AP", "SP", "AP_SP", "TSNE_SP", "TSNE_AP_SP"]
NUM_CELLS_GRID = [32, 48, 64]
KERNEL_SIZE_GRID = [4, 6, 8]
KNOWN_PEPTIDES = ["FMGIIF", "IMGIIA", "IMCIEW", "FATAAGGNMF", "FGDAAGGNTT"]
DEFAULT_OUTPUT_ROOT = Path("thesis_results") / "01_reproduction"
DEFAULT_SEED = 20260317
PHASE1_GENERATION_TARGET_UNIQUE = 20
PHASE1_GENERATION_MINIMUM_RETURN_COUNT = 10
PHASE1_GENERATION_GA_MAX_ATTEMPTS = 20

PHASE1_DIRS = [
    "tables",
    "models",
    "backups",
    "generated",
    "cgmd",
    "folds",
    "logs",
]

STATUS_ARTIFACTS = {
    "dataset_sanity": [
        "tables/dataset_sanity.csv",
        "tables/dataset_sanity.json",
    ],
    "preprocessing": ["tables/preprocessing_shapes.csv"],
    "folds": [
        "folds/nested_cv_fold_assignments.csv",
        "folds/nested_cv_fold_assignments.json",
    ],
    "nested_cv": [
        "frozen_model_config.json",
        "tables/hyperparameter_summary.csv",
        "tables/hyperparameter_summary.json",
    ],
    "thresholds": [
        "tables/threshold_summary.csv",
        "tables/threshold_summary.json",
        "tables/reproduced_predictive_performance.csv",
    ],
    "final_baseline": ["models/AP_SP.h5"],
    "known_predictions": ["tables/known_peptide_prediction_sanity.csv"],
    "generation": [
        "generated/generated_candidates.csv",
        "generated/raw_generator_output.csv",
        "generated/generated_similarity_summary.csv",
        "generated/generated_similarity_summary.json",
        "generated/generation_settings.json",
    ],
    "cgmd_template": ["cgmd/cgmd_sanity_template.csv"],
    "checklist": ["reproduction_checklist.md"],
}


@dataclass(frozen=True)
class Phase1Options:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    dataset_path: Path = DATASET_PATH
    mode: str | None = None
    step: str | None = None
    status: bool = False
    dry_run: bool = False
    force: bool = False
    skip_heavy: bool = False
    write_supek_pbs: bool = False
    pbs_repo_root: Path | None = None
    models: tuple[str, ...] = tuple(PHASE1_MODELS)
    seed: int = DEFAULT_SEED
    epochs: int = 70
    generation_target_unique: int = PHASE1_GENERATION_TARGET_UNIQUE
    generation_minimum_return_count: int = PHASE1_GENERATION_MINIMUM_RETURN_COUNT
    generation_ga_max_attempts: int = PHASE1_GENERATION_GA_MAX_ATTEMPTS


@dataclass(frozen=True)
class ResourceLogRow:
    step: str
    model: str
    start_timestamp: str
    end_timestamp: str
    walltime_seconds: float
    hostname: str
    command: str
    exit_status: str
    output_artifacts: str


def normalize_models(
    models: Sequence[str] | None = None,
    model: str | None = None,
) -> tuple[str, ...]:
    selected: list[str] = []
    if models:
        selected.extend(str(item) for item in models)
    if model:
        selected.append(str(model))
    if not selected:
        selected = list(PHASE1_MODELS)

    normalized: list[str] = []
    invalid: list[str] = []
    for item in selected:
        name = item.strip()
        if name not in PHASE1_MODELS:
            invalid.append(name)
            continue
        if name not in normalized:
            normalized.append(name)
    if invalid:
        allowed = ", ".join(PHASE1_MODELS)
        raise ValueError(f"Invalid Phase 1 model(s): {', '.join(invalid)}. Allowed: {allowed}")
    return tuple(normalized)


def options_from_args(args) -> Phase1Options:
    return Phase1Options(
        output_root=Path(args.output_root),
        dataset_path=Path(args.dataset_path),
        mode=args.mode,
        step=args.step,
        status=bool(args.status),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        skip_heavy=bool(args.skip_heavy),
        write_supek_pbs=bool(args.write_supek_pbs),
        pbs_repo_root=Path(args.pbs_repo_root) if getattr(args, "pbs_repo_root", None) else None,
        models=normalize_models(args.models, args.model),
        seed=int(args.seed),
        epochs=int(args.epochs),
        generation_target_unique=_positive_int(
            getattr(args, "generation_target_unique", PHASE1_GENERATION_TARGET_UNIQUE),
            "--generation-target-unique",
        ),
        generation_minimum_return_count=_positive_int(
            getattr(args, "generation_minimum_return_count", PHASE1_GENERATION_MINIMUM_RETURN_COUNT),
            "--generation-minimum-return-count",
        ),
        generation_ga_max_attempts=_positive_int(
            getattr(args, "generation_ga_max_attempts", PHASE1_GENERATION_GA_MAX_ATTEMPTS),
            "--generation-ga-max-attempts",
        ),
    )


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def run_phase1_reproduce(args_or_options) -> dict:
    options = (
        args_or_options
        if isinstance(args_or_options, Phase1Options)
        else options_from_args(args_or_options)
    )
    if options.status:
        return phase1_status(options.output_root, options.models)
    if not options.dry_run or options.write_supek_pbs:
        ensure_phase1_dirs(options.output_root)
    if options.write_supek_pbs:
        write_supek_pbs_scripts(options)

    steps = _steps_for_options(options)
    summary = {"output_root": str(options.output_root), "steps": []}
    for step in steps:
        if step == "sanity":
            summary["steps"].append(run_sanity(options))
        elif step == "folds":
            summary["steps"].append(write_nested_cv_folds(options))
        elif step == "nested-cv":
            summary["steps"].append(run_nested_cv(options))
        elif step == "aggregate-nested-cv":
            summary["steps"].append(run_aggregate_nested_cv(options))
        elif step == "thresholds":
            summary["steps"].append(run_thresholds(options))
        elif step == "train-final":
            summary["steps"].append(run_train_final(options))
        elif step == "generate":
            summary["steps"].append(run_generate(options))
        elif step == "cgmd-template":
            summary["steps"].append(write_cgmd_template(options))
        elif step == "checklist":
            summary["steps"].append(write_checklist(options))
        else:
            raise ValueError(f"Unknown Phase 1 step: {step}")
    summary["status"] = phase1_status(options.output_root, options.models)
    return summary


def _steps_for_options(options: Phase1Options) -> list[str]:
    if options.step:
        return [options.step]
    if options.mode in {None, "full"}:
        return [
            "sanity",
            "folds",
            "nested-cv",
            "thresholds",
            "train-final",
            "generate",
            "cgmd-template",
            "checklist",
        ]
    raise ValueError(f"Unsupported Phase 1 mode: {options.mode}")


def ensure_phase1_dirs(output_root: Path) -> None:
    for dirname in PHASE1_DIRS:
        (output_root / dirname).mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _needleman_wunsch_identity_percent(left: str, right: str) -> float:
    """Global-alignment identity percent used for Phase 1 Simtrain/Simgen reports."""

    left = str(left or "").strip().upper()
    right = str(right or "").strip().upper()
    if not left and not right:
        return 0.0
    rows = len(left) + 1
    cols = len(right) + 1
    scores = [[0 for _ in range(cols)] for _ in range(rows)]
    pointers: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        pointers[i][0] = "U"
    for j in range(1, cols):
        pointers[0][j] = "L"
    for i, left_aa in enumerate(left, start=1):
        for j, right_aa in enumerate(right, start=1):
            diagonal = scores[i - 1][j - 1] + (1 if left_aa == right_aa else 0)
            up = scores[i - 1][j]
            left_score = scores[i][j - 1]
            best = max(diagonal, up, left_score)
            scores[i][j] = best
            pointers[i][j] = "D" if diagonal == best else ("U" if up == best else "L")
    i = len(left)
    j = len(right)
    matches = 0
    alignment_length = 0
    while i > 0 or j > 0:
        pointer = pointers[i][j]
        if pointer == "D":
            matches += 1 if left[i - 1] == right[j - 1] else 0
            i -= 1
            j -= 1
        elif pointer == "U":
            i -= 1
        else:
            j -= 1
        alignment_length += 1
    return 100.0 * matches / alignment_length if alignment_length else 0.0


def generated_similarity_summary_rows(
    generated_sequences: Sequence[str],
    training_sequences: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    generated = [str(sequence).strip().upper() for sequence in generated_sequences if str(sequence).strip()]
    training = [str(sequence).strip().upper() for sequence in training_sequences if str(sequence).strip()]
    for sequence in generated:
        train_similarities = [
            _needleman_wunsch_identity_percent(sequence, training_sequence)
            for training_sequence in training
        ]
        generated_similarities = [
            _needleman_wunsch_identity_percent(sequence, other_sequence)
            for other_sequence in generated
            if other_sequence != sequence
        ]
        nearest_train_index = int(np.argmax(train_similarities)) if train_similarities else -1
        nearest_generated_candidates = [
            (other_sequence, _needleman_wunsch_identity_percent(sequence, other_sequence))
            for other_sequence in generated
            if other_sequence != sequence
        ]
        nearest_generated = max(
            nearest_generated_candidates,
            key=lambda item: item[1],
            default=("", 0.0),
        )
        rows.append(
            {
                "Sequence": sequence,
                "Simtrain_avg_percent": round(float(np.mean(train_similarities)), 4)
                if train_similarities
                else 0.0,
                "Simtrain_max_percent": round(float(max(train_similarities)), 4)
                if train_similarities
                else 0.0,
                "Nearest_training_sequence": training[nearest_train_index]
                if nearest_train_index >= 0
                else "",
                "Simgen_avg_percent": round(float(np.mean(generated_similarities)), 4)
                if generated_similarities
                else 0.0,
                "Simgen_max_percent": round(float(nearest_generated[1]), 4),
                "Nearest_generated_sequence": nearest_generated[0],
            }
        )
    return rows


def write_generated_similarity_summary(
    output_root: Path,
    generated_sequences: Sequence[str],
    training_sequences: Sequence[str],
) -> tuple[Path, Path]:
    rows = generated_similarity_summary_rows(generated_sequences, training_sequences)
    csv_path = output_root / "generated" / "generated_similarity_summary.csv"
    json_path = output_root / "generated" / "generated_similarity_summary.json"
    overall = {
        "method": "Needleman-Wunsch global identity percent",
        "generated_count": len([sequence for sequence in generated_sequences if str(sequence).strip()]),
        "training_count": len([sequence for sequence in training_sequences if str(sequence).strip()]),
        "Simtrain_avg_percent": round(
            float(np.mean([float(row["Simtrain_avg_percent"]) for row in rows])), 4
        )
        if rows
        else 0.0,
        "Simgen_avg_percent": round(
            float(np.mean([float(row["Simgen_avg_percent"]) for row in rows])), 4
        )
        if rows
        else 0.0,
    }
    _write_csv(csv_path, rows)
    _write_json(json_path, {"overall": overall, "rows": rows})
    return csv_path, json_path


def _timestamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def resource_logger(options: Phase1Options, step: str, model: str = ""):
    start = time.perf_counter()
    start_ts = _timestamp()
    status = "success"
    artifacts: list[str] = []
    try:
        yield artifacts
    except Exception:
        status = "failed"
        raise
    finally:
        row = ResourceLogRow(
            step=step,
            model=model,
            start_timestamp=start_ts,
            end_timestamp=_timestamp(),
            walltime_seconds=round(time.perf_counter() - start, 3),
            hostname=socket.gethostname(),
            command=" ".join(sys.argv),
            exit_status=status,
            output_artifacts=";".join(artifacts),
        )
        _append_resource_log(options.output_root, row)


def _append_resource_log(output_root: Path, row: ResourceLogRow) -> None:
    path = output_root / "logs" / "phase1_resource_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row).keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))


def run_sanity(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "sanity", "status": "dry-run", "outputs": []}
    rows = read_experimental_dataset(options.dataset_path)
    dataset_payload = dataset_sanity_payload(rows)
    dataset_csv = options.output_root / "tables" / "dataset_sanity.csv"
    dataset_json = options.output_root / "tables" / "dataset_sanity.json"
    _write_csv(dataset_csv, [dataset_payload])
    _write_json(dataset_json, dataset_payload)
    preprocessing_csv = write_preprocessing_shapes(options, rows)
    return {
        "step": "sanity",
        "status": "complete",
        "outputs": [str(dataset_csv), str(dataset_json), str(preprocessing_csv)],
    }


def dataset_sanity_payload(rows: list[dict[str, str]]) -> dict[str, int | float]:
    sequences = [row["sequence"] for row in rows]
    labels = [str(row["label"]) for row in rows]
    total = len(rows)
    sa = labels.count("1")
    nsa = labels.count("0")
    lengths = [len(sequence) for sequence in sequences] or [0]
    return {
        "total peptides": total,
        "SA": sa,
        "NSA": nsa,
        "max length": max(lengths),
        "min length": min(lengths),
        "duplicate sequences": len(sequences) - len(set(sequences)),
        "SA percentage": round(100.0 * sa / total, 6) if total else 0.0,
        "NSA percentage": round(100.0 * nsa / total, 6) if total else 0.0,
    }


def write_preprocessing_shapes(options: Phase1Options, rows: list[dict[str, str]]) -> Path:
    path = options.output_root / "tables" / "preprocessing_shapes.csv"
    output_rows: list[dict[str, str | int]] = []
    for model_name in options.models:
        payload: dict[str, str | int] = {
            "model name": model_name,
            "number of input branches": "not evaluated",
            "shape of each input branch": "not evaluated",
            "number of labels": len(rows),
            "positive labels": sum(1 for row in rows if str(row["label"]) == "1"),
            "negative labels": sum(1 for row in rows if str(row["label"]) == "0"),
            "status": "skipped heavy preprocessing runtime",
        }
        if not options.skip_heavy:
            try:
                from active_learning_thesis import predictive

                inputs, labels = predictive._prepare_training_tensors(model_name, rows)
                branch_shapes = _input_shapes(inputs)
                labels_array = np.asarray(labels).reshape(-1)
                payload.update(
                    {
                        "number of input branches": len(branch_shapes),
                        "shape of each input branch": json.dumps(branch_shapes),
                        "number of labels": int(len(labels_array)),
                        "positive labels": int(np.sum(labels_array == 1)),
                        "negative labels": int(np.sum(labels_array == 0)),
                        "status": "complete",
                    }
                )
            except Exception as exc:
                payload["status"] = f"runtime unavailable: {type(exc).__name__}: {exc}"
        output_rows.append(payload)
    _write_csv(path, output_rows)
    return path


def _input_shapes(inputs) -> list[list[int]]:
    if isinstance(inputs, (list, tuple)):
        return [list(np.asarray(item).shape) for item in inputs]
    return [list(np.asarray(inputs).shape)]


def write_nested_cv_folds(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "folds", "status": "dry-run", "outputs": []}
    rows = read_experimental_dataset(options.dataset_path)
    assignments, manifests = build_nested_cv_assignments(rows, seed=options.seed)
    assignments_csv = options.output_root / "folds" / "nested_cv_fold_assignments.csv"
    assignments_json = options.output_root / "folds" / "nested_cv_fold_assignments.json"
    _write_csv(
        assignments_csv,
        assignments,
        ["sequence", "label", "outer_fold_id", "inner_fold_id", "role_for_this_outer_inner_pair"],
    )
    _write_json(assignments_json, assignments)
    manifest_paths = []
    for (outer_id, inner_id), manifest_rows in manifests.items():
        path = options.output_root / "folds" / f"replay_manifest_outer_{outer_id}_inner_{inner_id}.json"
        _write_json(
            path,
            {
                "split_mode": "paper_nested_cv_replay",
                "outer_fold_id": outer_id,
                "inner_fold_id": inner_id,
                "rows": manifest_rows,
            },
        )
        manifest_paths.append(path)
    return {
        "step": "folds",
        "status": "complete",
        "outputs": [str(assignments_csv), str(assignments_json), *map(str, manifest_paths)],
    }


def build_nested_cv_assignments(rows: list[dict[str, str]], seed: int = DEFAULT_SEED):
    sequences = np.asarray([row["sequence"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    records_by_sequence = {row["sequence"]: row for row in rows}
    assignments: list[dict[str, str | int]] = []
    manifests: dict[tuple[int, int], list[dict[str, str | int]]] = {}
    for outer_fold_id, (dev_idx, test_idx) in enumerate(_stratified_kfold_indices(labels, n_splits=5, seed=seed), start=1):
        dev_sequences = sequences[dev_idx]
        dev_labels = labels[dev_idx]
        for inner_fold_id, (train_local_idx, val_local_idx) in enumerate(
            _stratified_kfold_indices(dev_labels, n_splits=5, seed=seed + outer_fold_id),
            start=1,
        ):
            train_indices = dev_idx[train_local_idx]
            val_indices = dev_idx[val_local_idx]
            split_rows: list[dict[str, str | int]] = []
            for index in train_indices:
                split_rows.append(_fold_row(records_by_sequence[str(sequences[index])], outer_fold_id, inner_fold_id, "train"))
            for index in val_indices:
                split_rows.append(_fold_row(records_by_sequence[str(sequences[index])], outer_fold_id, inner_fold_id, "validation"))
            for index in test_idx:
                split_rows.append(_fold_row(records_by_sequence[str(sequences[index])], outer_fold_id, inner_fold_id, "test"))
            assignments.extend(split_rows)
            manifests[(outer_fold_id, inner_fold_id)] = [
                {
                    "split_mode": "paper_nested_cv_replay",
                    "split": _manifest_split(str(row["role_for_this_outer_inner_pair"])),
                    "replay_role": "none",
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_fold_id,
                    "sequence": row["sequence"],
                    "label": row["label"],
                }
                for row in split_rows
            ]
    return assignments, manifests


def _stratified_kfold_indices(labels, n_splits: int, seed: int):
    labels_array = np.asarray(labels)
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        dummy = np.zeros(len(labels_array), dtype=int)
        return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(dummy, labels_array)]
    except ModuleNotFoundError:
        rng = np.random.default_rng(seed)
        per_label: dict[int, list[int]] = {}
        for index, label in enumerate(labels_array):
            per_label.setdefault(int(label), []).append(index)
        folds: list[list[int]] = [[] for _ in range(n_splits)]
        for indices in per_label.values():
            shuffled = np.asarray(indices, dtype=int)
            rng.shuffle(shuffled)
            for local_index, item in enumerate(shuffled):
                folds[local_index % n_splits].append(int(item))
        pairs = []
        all_indices = np.arange(len(labels_array), dtype=int)
        for fold in folds:
            test_idx = np.asarray(sorted(fold), dtype=int)
            test_set = set(int(index) for index in test_idx)
            train_idx = np.asarray([int(index) for index in all_indices if int(index) not in test_set], dtype=int)
            pairs.append((train_idx, test_idx))
        return pairs


def _fold_row(row: dict[str, str], outer_fold_id: int, inner_fold_id: int, role: str) -> dict[str, str | int]:
    return {
        "sequence": row["sequence"],
        "label": row["label"],
        "outer_fold_id": outer_fold_id,
        "inner_fold_id": inner_fold_id,
        "role_for_this_outer_inner_pair": role,
    }


def _manifest_split(role: str) -> str:
    if role == "test":
        return "holdout"
    if role == "validation":
        return "validation"
    return "train_pool"


def run_nested_cv(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "nested-cv", "status": "dry-run", "outputs": []}
    if options.skip_heavy:
        pbs_outputs = write_supek_pbs_scripts(options)
        return {"step": "nested-cv", "status": "skipped-heavy", "outputs": list(map(str, pbs_outputs))}
    with resource_logger(options, "nested-cv") as artifacts:
        rows = read_experimental_dataset(options.dataset_path)
        inner_rows, outer_eval_rows = train_nested_cv_models(options, rows)
        artifacts.extend(str(path) for path in write_per_model_nested_cv_outputs(options, inner_rows, outer_eval_rows))
        # Per-model PBS jobs run concurrently, so shared thesis summaries are
        # intentionally written only by an explicit aggregate step or by
        # thresholds after all model-specific evidence exists.
        if len(options.models) > 1:
            artifacts.extend(str(path) for path in aggregate_nested_cv_outputs(options.output_root, options.models))
    return {"step": "nested-cv", "status": "complete", "outputs": artifacts}


def write_per_model_nested_cv_outputs(
    options: Phase1Options,
    inner_rows: list[dict],
    outer_eval_rows: list[dict],
) -> list[Path]:
    outputs: list[Path] = []
    for model_name in options.models:
        model_inner_rows = [row for row in inner_rows if row.get("model") == model_name]
        model_outer_rows = [row for row in outer_eval_rows if row.get("model") == model_name]
        inner_path = _per_model_inner_path(options.output_root, model_name)
        outer_path = _per_model_outer_path(options.output_root, model_name)
        _write_csv(inner_path, model_inner_rows)
        _write_csv(outer_path, model_outer_rows)
        frozen_config, hyper_rows = summarize_hyperparameters(model_inner_rows, (model_name,))
        frozen_path = _per_model_frozen_path(options.output_root, model_name)
        hyper_csv = _per_model_hyper_csv_path(options.output_root, model_name)
        hyper_json = _per_model_hyper_json_path(options.output_root, model_name)
        _write_json(frozen_path, frozen_config)
        _write_csv(hyper_csv, hyper_rows)
        _write_json(hyper_json, hyper_rows)
        outputs.extend([inner_path, outer_path, frozen_path, hyper_csv, hyper_json])
    return outputs


def run_aggregate_nested_cv(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "aggregate-nested-cv", "status": "dry-run", "outputs": []}
    outputs = aggregate_nested_cv_outputs(options.output_root, options.models)
    return {"step": "aggregate-nested-cv", "status": "complete", "outputs": [str(path) for path in outputs]}


def aggregate_nested_cv_outputs(output_root: Path, models: Sequence[str]) -> list[Path]:
    missing: list[str] = []
    inner_rows: list[dict] = []
    outer_rows: list[dict] = []
    for model_name in models:
        inner_path = _per_model_inner_path(output_root, model_name)
        outer_path = _per_model_outer_path(output_root, model_name)
        if not inner_path.exists():
            missing.append(str(inner_path))
        else:
            inner_rows.extend(_read_csv(inner_path))
        if not outer_path.exists():
            missing.append(str(outer_path))
        else:
            outer_rows.extend(_read_csv(outer_path))
    if missing:
        raise FileNotFoundError("Missing per-model nested-CV evidence: " + "; ".join(missing))

    inner_path = output_root / "tables" / "nested_cv_inner_results.csv"
    outer_path = output_root / "tables" / "nested_cv_outer_predictions.csv"
    frozen_path = output_root / "frozen_model_config.json"
    hyper_csv = output_root / "tables" / "hyperparameter_summary.csv"
    hyper_json = output_root / "tables" / "hyperparameter_summary.json"
    _write_csv(inner_path, inner_rows)
    _write_csv(outer_path, outer_rows)
    frozen_config, hyper_rows = summarize_hyperparameters(inner_rows, models)
    _write_json(frozen_path, frozen_config)
    _write_csv(hyper_csv, hyper_rows)
    _write_json(hyper_json, hyper_rows)
    return [inner_path, outer_path, frozen_path, hyper_csv, hyper_json]


def _per_model_inner_path(output_root: Path, model_name: str) -> Path:
    return output_root / "tables" / f"nested_cv_inner_results_{model_name}.csv"


def _per_model_outer_path(output_root: Path, model_name: str) -> Path:
    return output_root / "tables" / f"nested_cv_outer_predictions_{model_name}.csv"


def _per_model_frozen_path(output_root: Path, model_name: str) -> Path:
    return output_root / "tables" / f"frozen_model_config_{model_name}.json"


def _per_model_hyper_csv_path(output_root: Path, model_name: str) -> Path:
    return output_root / "tables" / f"hyperparameter_summary_{model_name}.csv"


def _per_model_hyper_json_path(output_root: Path, model_name: str) -> Path:
    return output_root / "tables" / f"hyperparameter_summary_{model_name}.json"


def train_nested_cv_models(options: Phase1Options, rows: list[dict[str, str]]):
    from active_learning_thesis.predictive import train_model

    sequences = np.asarray([row["sequence"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    inner_results: list[dict] = []
    outer_predictions: list[dict] = []
    cache_dir = options.output_root / "models" / ".phase1_cache"
    for model_name in options.models:
        for outer_fold_id, (dev_idx, test_idx) in enumerate(_stratified_kfold_indices(labels, n_splits=5, seed=options.seed), start=1):
            dev_rows = [rows[int(i)] for i in dev_idx]
            test_rows = [rows[int(i)] for i in test_idx]
            dev_sequences = sequences[dev_idx]
            dev_labels = labels[dev_idx]
            for num_cells in NUM_CELLS_GRID:
                kernel_values = [None] if model_name == "AP" else KERNEL_SIZE_GRID
                for kernel_size in kernel_values:
                    for inner_fold_id, (train_local_idx, val_local_idx) in enumerate(
                        _stratified_kfold_indices(dev_labels, n_splits=5, seed=options.seed + outer_fold_id),
                        start=1,
                    ):
                        train_rows = [dev_rows[int(i)] for i in train_local_idx]
                        val_rows = [dev_rows[int(i)] for i in val_local_idx]
                        config = _training_config(options, model_name, num_cells, kernel_size)
                        model_path = (
                            options.output_root
                            / "models"
                            / "nested_cv"
                            / model_name
                            / f"outer_{outer_fold_id:02d}_inner_{inner_fold_id:02d}_cells_{num_cells}_kernel_{_kernel_display(model_name, kernel_size)}.h5"
                        )
                        managed = train_model(
                            model_name,
                            train_rows,
                            val_rows,
                            seed=options.seed + outer_fold_id * 100 + inner_fold_id,
                            output_path=model_path,
                            config=config,
                            cache_dir=cache_dir,
                        )
                        val_loss, val_acc, val_probs, val_truth = _evaluate_managed_model(managed, model_name, val_rows)
                        val_metrics = evaluate_binary_classifier(val_truth, val_probs, threshold=0.5)
                        inner_results.append(
                            {
                                "model": model_name,
                                "outer_fold_id": outer_fold_id,
                                "inner_fold_id": inner_fold_id,
                                "num_cells": num_cells,
                                "kernel_size": _kernel_display(model_name, kernel_size),
                                "validation_loss": val_loss,
                                "validation_accuracy": val_acc,
                                "validation_f1_fixed_0_5": val_metrics["f1"],
                                "validation_predictions": json.dumps(list(map(float, val_probs))),
                                "validation_labels": json.dumps(list(map(int, val_truth))),
                            }
                        )
            selected = _select_outer_hyperparameters(inner_results, model_name, outer_fold_id)
            config = _training_config(
                options,
                model_name,
                int(selected["num_cells"]),
                _kernel_from_summary(model_name, selected["kernel_size"]),
            )
            outer_model_path = options.output_root / "models" / "nested_cv" / model_name / f"outer_{outer_fold_id:02d}_selected.h5"
            managed_outer = train_model(
                model_name,
                dev_rows,
                [],
                seed=options.seed + outer_fold_id,
                output_path=outer_model_path,
                config=config,
                cache_dir=cache_dir,
            )
            _, _, test_probs, test_truth = _evaluate_managed_model(managed_outer, model_name, test_rows)
            validation_probs, validation_truth = _validation_predictions_for_setting(inner_results, model_name, outer_fold_id, selected)
            thresholds = select_thresholds_from_validation(validation_truth, validation_probs)
            for threshold_type, threshold_value in thresholds.items():
                metrics = evaluate_binary_classifier(
                    test_truth,
                    test_probs,
                    threshold=threshold_value,
                    threshold_source="inner_validation",
                )
                outer_predictions.append(
                    {
                        "model": model_name,
                        "outer_fold_id": outer_fold_id,
                        "threshold_type": threshold_type,
                        "threshold_value": threshold_value,
                        "num_cells": selected["num_cells"],
                        "kernel_size": selected["kernel_size"],
                        "test_labels": json.dumps(list(map(int, test_truth))),
                        "test_predictions": json.dumps(list(map(float, test_probs))),
                        **_metrics_for_summary(metrics),
                    }
                )
    return inner_results, outer_predictions


def _training_config(options: Phase1Options, model_name: str, num_cells: int, kernel_size: int | None) -> RunConfig:
    config = RunConfig(random_seed=options.seed, epochs=options.epochs)
    config.model_num_cells[model_name] = int(num_cells)
    if kernel_size is not None:
        config.model_kernel_size[model_name] = int(kernel_size)
    return config


def _kernel_display(model_name: str, kernel_size: int | None) -> str:
    return "n/a" if model_name == "AP" or kernel_size is None else str(kernel_size)


def _kernel_from_summary(model_name: str, value: str | int | None) -> int | None:
    if model_name == "AP" or str(value) == "n/a":
        return None
    return int(value)


def _evaluate_managed_model(managed, model_name: str, rows: list[dict[str, str]]):
    from active_learning_thesis import predictive

    inputs, labels = predictive._prepare_training_tensors(model_name, rows)
    truth = np.asarray(labels).reshape(-1).astype(int)
    probs = predictive._predict_probabilities_from_inputs(managed, inputs)
    loss = _binary_cross_entropy(truth, probs)
    accuracy = float(np.mean((probs >= 0.5).astype(int) == truth)) if len(truth) else 0.0
    return loss, accuracy, probs, truth


def _binary_cross_entropy(truth, probs) -> float:
    truth = np.asarray(truth, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1 - 1e-8)
    if len(truth) == 0:
        return 0.0
    return float(np.mean(-(truth * np.log(probs) + (1 - truth) * np.log(1 - probs))))


def _select_outer_hyperparameters(inner_results: list[dict], model_name: str, outer_fold_id: int) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in inner_results:
        if row["model"] == model_name and int(row["outer_fold_id"]) == outer_fold_id:
            grouped.setdefault((str(row["num_cells"]), str(row["kernel_size"])), []).append(row)
    if not grouped:
        raise ValueError(f"No inner CV rows for {model_name} outer fold {outer_fold_id}")
    best_key = min(
        grouped,
        key=lambda key: (float(np.mean([float(row["validation_loss"]) for row in grouped[key]])), int(key[0]), key[1]),
    )
    return {"num_cells": int(best_key[0]), "kernel_size": best_key[1]}


def _validation_predictions_for_setting(inner_results: list[dict], model_name: str, outer_fold_id: int, selected: dict):
    probs: list[float] = []
    labels: list[int] = []
    for row in inner_results:
        if row["model"] != model_name or int(row["outer_fold_id"]) != outer_fold_id:
            continue
        if int(row["num_cells"]) != int(selected["num_cells"]):
            continue
        if str(row["kernel_size"]) != str(selected["kernel_size"]):
            continue
        probs.extend(float(value) for value in json.loads(row["validation_predictions"]))
        labels.extend(int(value) for value in json.loads(row["validation_labels"]))
    return np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)


def summarize_hyperparameters(inner_results: list[dict], models: Sequence[str]) -> tuple[dict, list[dict]]:
    frozen: dict[str, dict] = {}
    rows: list[dict] = []
    for model_name in models:
        model_rows = [row for row in inner_results if row["model"] == model_name]
        grouped: dict[tuple[str, str], list[dict]] = {}
        outer_selections: list[str] = []
        for row in model_rows:
            grouped.setdefault((str(row["num_cells"]), str(row["kernel_size"])), []).append(row)
        if not grouped:
            frozen[model_name] = {"status": "missing", "selection_rule": "lowest average inner-validation loss"}
            rows.append(
                {
                    "Model": model_name,
                    "Best num-cells": "missing",
                    "Best kernel-size": "missing",
                    "Selection rule": "lowest average inner-validation loss",
                    "Outer-fold selections": "missing",
                    "Notes": "nested CV evidence missing",
                }
            )
            continue
        best_key = min(
            grouped,
            key=lambda key: (float(np.mean([float(row["validation_loss"]) for row in grouped[key]])), int(key[0]), key[1]),
        )
        for outer_fold_id in sorted({int(row["outer_fold_id"]) for row in model_rows}):
            selected = _select_outer_hyperparameters(model_rows, model_name, outer_fold_id)
            outer_selections.append(f"outer_{outer_fold_id}:cells={selected['num_cells']},kernel={selected['kernel_size']}")
        best_val_loss = float(np.mean([float(row["validation_loss"]) for row in grouped[best_key]]))
        best_val_acc = float(np.mean([float(row["validation_accuracy"]) for row in grouped[best_key]]))
        best_val_f1 = float(np.mean([float(row["validation_f1_fixed_0_5"]) for row in grouped[best_key]]))
        frozen[model_name] = {
            "model": model_name,
            "num_cells": int(best_key[0]),
            "kernel_size": best_key[1],
            "selection_rule": "lowest average inner-validation loss across inner folds",
            "average_validation_loss": best_val_loss,
            "average_validation_accuracy": best_val_acc,
            "average_validation_f1_fixed_0_5": best_val_f1,
            "outer_fold_selections": outer_selections,
        }
        rows.append(
            {
                "Model": model_name,
                "Best num-cells": int(best_key[0]),
                "Best kernel-size": best_key[1],
                "Selection rule": "lowest average inner-validation loss across inner validation folds only",
                "Outer-fold selections": "; ".join(outer_selections),
                "Notes": f"avg_val_loss={best_val_loss:.6f}; avg_val_acc={best_val_acc:.6f}; avg_val_f1={best_val_f1:.6f}",
            }
        )
    return frozen, rows


def select_thresholds_from_validation(validation_truth, validation_probs) -> dict[str, float]:
    truth = np.asarray(validation_truth, dtype=int)
    probs = np.asarray(validation_probs, dtype=float)
    pr_threshold, _ = pr_best_f1_threshold(truth, probs)
    return {
        "fixed_0_5": 0.5,
        "PR": float(pr_threshold),
        "ROC": float(_roc_best_gmean_threshold(truth, probs)),
    }


def _roc_best_gmean_threshold(truth: np.ndarray, probs: np.ndarray) -> float:
    if len(truth) == 0 or len(probs) == 0:
        return 0.5
    thresholds = sorted((float(value) for value in np.unique(np.clip(probs, 0.0, 1.0))), reverse=True)
    if not thresholds:
        return 0.5
    best_threshold = thresholds[0]
    best_gmean = -1.0
    for threshold in thresholds:
        metrics = evaluate_binary_classifier(truth, probs, threshold=threshold)
        gmean = float(metrics["gmean"])
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = threshold
    return float(best_threshold)


def _metrics_for_summary(metrics: dict) -> dict[str, float]:
    return {
        "Accuracy": float(metrics["accuracy"]),
        "F1": float(metrics["f1"]),
        "ROC-AUC": float(metrics["roc_auc"]),
        "PR-AUC": float(metrics["pr_auc"]),
        "gmean": float(metrics["gmean"]),
        "Brier": float(metrics["brier_score"]),
        "ECE-10": float(metrics["ece_10"]),
        "MCE-10": float(metrics["mce_10"]),
        "decision threshold": float(metrics["decision_threshold"]),
    }


def run_thresholds(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "thresholds", "status": "dry-run", "outputs": []}
    if options.skip_heavy:
        return {"step": "thresholds", "status": "skipped-heavy", "outputs": []}
    aggregate_outputs = aggregate_nested_cv_outputs(options.output_root, options.models)
    outer_path = options.output_root / "tables" / "nested_cv_outer_predictions.csv"
    outer_rows = _read_csv(outer_path)
    summary_rows = aggregate_threshold_rows(outer_rows, options.models)
    threshold_csv = options.output_root / "tables" / "threshold_summary.csv"
    threshold_json = options.output_root / "tables" / "threshold_summary.json"
    performance_csv = options.output_root / "tables" / "reproduced_predictive_performance.csv"
    _write_csv(threshold_csv, summary_rows)
    _write_json(threshold_json, summary_rows)
    _write_csv(performance_csv, reproduced_performance_rows(summary_rows))
    return {
        "step": "thresholds",
        "status": "complete",
        "outputs": [*(str(path) for path in aggregate_outputs), str(threshold_csv), str(threshold_json), str(performance_csv)],
    }


def aggregate_threshold_rows(outer_rows: list[dict], models: Sequence[str]) -> list[dict]:
    summaries: list[dict] = []
    metric_keys = ["Accuracy", "F1", "ROC-AUC", "PR-AUC", "gmean", "Brier", "ECE-10", "MCE-10", "decision threshold"]
    for model_name in models:
        for threshold_type in ["fixed_0_5", "PR", "ROC"]:
            selected = [row for row in outer_rows if row.get("model") == model_name and row.get("threshold_type") == threshold_type]
            if not selected:
                continue
            summary = {"Model": model_name, "Threshold type": threshold_type}
            for key in metric_keys:
                values = [float(row[key]) for row in selected if row.get(key) not in {None, ""}]
                summary["Threshold value" if key == "decision threshold" else key] = float(np.mean(values)) if values else 0.0
            summaries.append(summary)
    return summaries


def reproduced_performance_rows(summary_rows: list[dict]) -> list[dict]:
    rows = []
    for row in summary_rows:
        if row.get("Threshold type") != "PR":
            continue
        note = ""
        if row.get("Model") == "AP_SP":
            note = "AP_SP paper target: approximately 81.9% accuracy and 0.865 F1"
        rows.append({**row, "Comparison note": note})
    return rows


def run_train_final(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "train-final", "status": "dry-run", "outputs": []}
    if options.skip_heavy:
        return {"step": "train-final", "status": "skipped-heavy", "outputs": []}
    with resource_logger(options, "train-final") as artifacts:
        rows = read_experimental_dataset(options.dataset_path)
        frozen = _load_frozen_config(options.output_root)
        backup_existing_models(options.output_root)
        from active_learning_thesis.predictive import train_model

        defaults = RunConfig()
        for index, model_name in enumerate(options.models):
            model_config = frozen.get(model_name, {})
            config = _training_config(
                options,
                model_name,
                int(model_config.get("num_cells", defaults.model_num_cells.get(model_name, 32))),
                _kernel_from_summary(model_name, model_config.get("kernel_size", defaults.model_kernel_size.get(model_name, 4))),
            )
            output_path = options.output_root / "models" / f"{model_name}.h5"
            train_model(
                model_name,
                rows,
                [],
                seed=options.seed + index,
                output_path=output_path,
                config=config,
                cache_dir=options.output_root / "models" / ".phase1_cache",
            )
            artifacts.append(str(output_path))
        if "AP_SP" in options.models:
            known_path = write_known_peptide_predictions(options)
            artifacts.append(str(known_path))
    return {"step": "train-final", "status": "complete", "outputs": artifacts}


def _load_frozen_config(output_root: Path) -> dict:
    path = output_root / "frozen_model_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def backup_existing_models(output_root: Path) -> list[Path]:
    backup_dir = output_root / "backups" / "models_before_phase1"
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in sorted(PREDICTIVE_MODEL_DIR.glob("*.h5")):
        destination = backup_dir / source.name
        if destination.exists():
            continue
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def write_known_peptide_predictions(options: Phase1Options) -> Path:
    model = _load_final_model(options.output_root, "AP_SP")
    from active_learning_thesis.predictive import score_sequences_with_ensemble

    scores = score_sequences_with_ensemble([model], KNOWN_PEPTIDES, use_calibration=False)
    probs = np.asarray(scores["pred_mean"], dtype=float)
    rows = [
        {
            "Sequence": sequence,
            "Paper/generated candidate": "paper/generated sanity candidate",
            "AP_SP probability": float(prob),
        }
        for sequence, prob in zip(KNOWN_PEPTIDES, probs)
    ]
    path = options.output_root / "tables" / "known_peptide_prediction_sanity.csv"
    _write_csv(path, rows)
    return path


def _load_final_model(output_root: Path, model_name: str):
    from active_learning_thesis.predictive import _load_saved_model

    model_path = output_root / "models" / f"{model_name}.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 final model: {model_path}")
    return _load_saved_model(model_name, DEFAULT_SEED, model_path)


def run_generate(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "generate", "status": "dry-run", "outputs": []}
    if options.skip_heavy:
        settings_path = write_generation_settings(options, skipped=True)
        return {"step": "generate", "status": "skipped-heavy", "outputs": [str(settings_path)]}
    with resource_logger(options, "generate", "AP_SP") as artifacts:
        rows = read_experimental_dataset(options.dataset_path)
        existing_sequences = {row["sequence"] for row in rows}
        model = _load_final_model(options.output_root, "AP_SP")
        from active_learning_thesis.generative import generate_candidate_sequences
        from active_learning_thesis.predictive import score_sequences_with_ensemble

        target_unique = _positive_int(options.generation_target_unique, "generation_target_unique")
        minimum_return_count = _positive_int(
            options.generation_minimum_return_count,
            "generation_minimum_return_count",
        )
        ga_max_attempts = _positive_int(options.generation_ga_max_attempts, "generation_ga_max_attempts")

        config = RunConfig(random_seed=options.seed, candidate_pool_min=target_unique)
        config.preferred_length_min = 5
        config.preferred_length_max = 10
        config.ga_max_attempts = ga_max_attempts
        config.use_similarity_penalty = True
        config.use_length_penalty = True

        sequences, metadata = generate_candidate_sequences(
            [model],
            existing_sequences,
            config,
            min_unique=target_unique,
            minimum_return_count=minimum_return_count,
            objective="ensemble_mean",
            use_similarity_penalty=True,
            use_length_penalty=True,
            return_metadata=True,
        )
        scores = score_sequences_with_ensemble([model], sequences, use_calibration=False)
        probs = np.asarray(scores["pred_mean"], dtype=float)
        candidate_rows = []
        raw_rows = []
        for rank, (sequence, probability) in enumerate(zip(sequences, probs), start=1):
            base = {
                "Rank": rank,
                "Sequence": sequence,
                "Length": len(sequence),
                "AP_SP probability": float(probability),
                "Novel?": sequence not in existing_sequences,
                "In training set?": sequence in existing_sequences,
            }
            candidate_rows.append(base)
            raw_rows.append({**base, **{f"metadata_{key}": value for key, value in metadata.get(sequence, {}).items()}})
        candidates_path = options.output_root / "generated" / "generated_candidates.csv"
        raw_path = options.output_root / "generated" / "raw_generator_output.csv"
        similarity_csv, similarity_json = write_generated_similarity_summary(
            options.output_root,
            sequences,
            sorted(existing_sequences),
        )
        settings_path = write_generation_settings(options, skipped=False)
        _write_csv(candidates_path, candidate_rows)
        _write_csv(raw_path, raw_rows)
        artifacts.extend(
            [
                str(candidates_path),
                str(raw_path),
                str(similarity_csv),
                str(similarity_json),
                str(settings_path),
            ]
        )
    return {"step": "generate", "status": "complete", "outputs": artifacts}


def write_generation_settings(options: Phase1Options, skipped: bool) -> Path:
    path = options.output_root / "generated" / "generation_settings.json"
    payload = {
        "model": "AP_SP",
        "canonical_amino_acids": RunConfig().allowed_amino_acids,
        "preferred_length_range": [5, 10],
        "objective": "ensemble_mean",
        "target_unique": options.generation_target_unique,
        "minimum_return_count": options.generation_minimum_return_count,
        "ga_max_attempts": options.generation_ga_max_attempts,
        "use_similarity_penalty": True,
        "use_length_penalty": True,
        "similarity_report_method": "Needleman-Wunsch global identity percent",
        "exclude_training_duplicates": True,
        "skipped_heavy": skipped,
        "note": (
            "Phase 1 generation is for baseline sanity only; replay retrains from scratch. "
            "The reduced return count keeps the baseline package resumable on local hardware."
        ),
    }
    _write_json(path, payload)
    return path


def write_cgmd_template(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "cgmd-template", "status": "dry-run", "outputs": []}
    generated_path = options.output_root / "generated" / "generated_candidates.csv"
    sequences = []
    if generated_path.exists():
        sequences = [row["Sequence"] for row in _read_csv(generated_path)]
    if not sequences:
        sequences = KNOWN_PEPTIDES
    rows = [
        {
            "Sequence": sequence,
            "APSASA": "",
            "APcontact": "",
            "Visual aggregate": "",
            "Strict label": "",
            "Practical label": "",
            "Notes": (
                "strict positive: APSASA >= 1.75 and APcontact >= 0.5; "
                "practical positive: APSASA >= 1.75 and clear visual aggregation; "
                "negative: APSASA < 1.75 and no clear aggregate; "
                "borderline: conflicting metrics; do not use as AL label until resolved"
            ),
        }
        for sequence in sequences
    ]
    path = options.output_root / "cgmd" / "cgmd_sanity_template.csv"
    _write_csv(path, rows)
    return {"step": "cgmd-template", "status": "complete", "outputs": [str(path)]}


def write_checklist(options: Phase1Options) -> dict:
    if options.dry_run:
        return {"step": "checklist", "status": "dry-run", "outputs": []}
    path = options.output_root / "reproduction_checklist.md"
    content = """# Phase 1 Reproduction Checklist

- [ ] Dataset size and class ratio match the paper approximately.
- [ ] AP/SP/AP_SP/TSNE_SP/TSNE_AP_SP preprocessing works.
- [ ] Hyperparameter selection completes for all five models.
- [ ] Threshold evaluation completes for all five models.
- [ ] AP_SP has F1 around 0.84-0.87, or at least clearly strong.
- [ ] AP_SP beats AP-only.
- [ ] Single-peptide prediction works.
- [ ] GA generator produces high-probability 5-10mer candidates.
- [ ] Generated candidates are mostly novel and not duplicates from training.
- [ ] CG-MD sanity template is ready.
- [ ] Existing models/artifacts were not accidentally overwritten.

Note: Phase 1 final full-data weights are baseline sanity artifacts only. Phase 2 active-learning replay must retrain from scratch per replay seed and round.
"""
    path.write_text(content, encoding="utf-8")
    return {"step": "checklist", "status": "complete", "outputs": [str(path)]}


def write_supek_pbs_scripts(options: Phase1Options) -> list[Path]:
    pbs_dir = options.output_root / "logs" / "supek_pbs"
    pbs_dir.mkdir(parents=True, exist_ok=True)
    supek_log_dir = options.output_root / "logs" / "supek_runtime"
    supek_log_dir.mkdir(parents=True, exist_ok=True)
    pbs_repo_root = _pbs_repo_root(options)
    absolute_output_root = _pbs_target_path(options.output_root, options.pbs_repo_root)
    absolute_dataset_path = _pbs_target_path(options.dataset_path, options.pbs_repo_root)
    absolute_supek_log_dir = absolute_output_root / "logs" / "supek_runtime"
    created: list[Path] = []
    for model_name in options.models:
        for step, walltime in [("nested-cv", "24:00:00"), ("train-final", "06:00:00")]:
            path = pbs_dir / f"phase1_{step.replace('-', '_')}_{model_name}.pbs"
            job_name = f"p1_{step[:3]}_{model_name}"
            command = (
                f"python -m active_learning_thesis phase1-reproduce "
                f"--output-root {_shell_quote(absolute_output_root.as_posix())} "
                f"--dataset-path {_shell_quote(absolute_dataset_path.as_posix())} "
                f"--step {step} --models {model_name} --seed {options.seed} --epochs {options.epochs}"
            )
            path.write_text(
                _pbs_text(
                    job_name=job_name,
                    walltime=walltime,
                    command=command,
                    log_dir=absolute_supek_log_dir,
                    repo_root=pbs_repo_root,
                ),
                encoding="utf-8",
            )
            created.append(path)
    aggregate_path = pbs_dir / "phase1_lightweight_artifacts.pbs"
    model_args = " ".join(options.models)
    aggregate_command = (
        f"python -m active_learning_thesis phase1-reproduce --output-root {_shell_quote(absolute_output_root.as_posix())} "
        f"--dataset-path {_shell_quote(absolute_dataset_path.as_posix())} --step sanity --models {model_args} && "
        f"python -m active_learning_thesis phase1-reproduce --output-root {_shell_quote(absolute_output_root.as_posix())} "
        f"--dataset-path {_shell_quote(absolute_dataset_path.as_posix())} --step folds --models {model_args} && "
        f"python -m active_learning_thesis phase1-reproduce --output-root {_shell_quote(absolute_output_root.as_posix())} "
        f"--step checklist --models {model_args}"
    )
    aggregate_path.write_text(
        _pbs_text(
            job_name="p1_light",
            walltime="02:00:00",
            command=aggregate_command,
            log_dir=absolute_supek_log_dir,
            repo_root=pbs_repo_root,
        ),
        encoding="utf-8",
    )
    created.append(aggregate_path)
    return created


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _pbs_target_path(path: Path, pbs_repo_root: Path | None) -> Path:
    if path.is_absolute():
        return path
    if pbs_repo_root is not None:
        return pbs_repo_root / path
    return (Path.cwd() / path).resolve()


def _pbs_repo_root(options: Phase1Options) -> Path:
    if options.pbs_repo_root is not None:
        return options.pbs_repo_root
    return Path.cwd().resolve()


def _pbs_text(job_name: str, walltime: str, command: str, log_dir: Path, repo_root: Path) -> str:
    log_dir_text = log_dir.as_posix()
    repo_root_text = repo_root.as_posix()
    return f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=40GB
#PBS -l walltime={walltime}
#PBS -o {log_dir_text}/{job_name}.out
#PBS -e {log_dir_text}/{job_name}.err

set -euo pipefail
cd "{repo_root_text}"
mkdir -p "{log_dir_text}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ml_peptide_self_assembly

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"

mkdir -p /lustre/scratch/$USER/ml_peptide_self_assembly_runs

echo "[phase1] start $(date -Is) host=$(hostname)"
echo "[phase1] workdir=$(pwd)"
echo "[phase1] CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
echo "[phase1] command: {command}"
nvidia-smi || true
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
{command}
echo "[phase1] end $(date -Is)"
"""


def phase1_status(output_root: Path, models: Sequence[str] | None = None) -> dict:
    models = tuple(models or PHASE1_MODELS)
    checks: dict[str, dict] = {}
    blockers: list[str] = []

    def add_blocker(path: str) -> None:
        if path not in blockers:
            blockers.append(path)

    for step, relative_paths in STATUS_ARTIFACTS.items():
        present = []
        missing = []
        for relative in relative_paths:
            path = output_root / relative
            if path.exists():
                present.append(str(path))
            else:
                missing.append(str(path))
        checks[step] = {"present": present, "missing": missing, "complete": not missing}
        for path in missing:
            add_blocker(path)
    model_check = {"present": [], "missing": [], "complete": True}
    for model_name in models:
        model_path = output_root / "models" / f"{model_name}.h5"
        if model_path.exists():
            model_check["present"].append(str(model_path))
        else:
            model_check["missing"].append(str(model_path))
            add_blocker(str(model_path))
    model_check["complete"] = not model_check["missing"]
    checks["selected_models"] = model_check
    ready = not blockers
    return {
        "output_root": str(output_root),
        "checks": checks,
        "missing_blockers": blockers,
        "ready_for_phase2": ready,
        "verdict": f"Ready for Phase 2: {'yes' if ready else 'no'}",
    }


def format_status_report(status: dict) -> str:
    lines = [f"Phase 1 status for {status['output_root']}"]
    for step, check in status.get("checks", {}).items():
        state = "complete" if check.get("complete") else "missing"
        lines.append(f"- {step}: {state}")
        for missing in check.get("missing", []):
            lines.append(f"  missing: {missing}")
    lines.append(status.get("verdict", "Ready for Phase 2: no"))
    blockers = status.get("missing_blockers", [])
    if blockers:
        lines.append("Phase 2 blockers:")
        for blocker in blockers:
            lines.append(f"- {blocker}")
    return "\n".join(lines)
