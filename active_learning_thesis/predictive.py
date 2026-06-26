from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

from active_learning_thesis.config import RunConfig
from active_learning_thesis.dependencies import load_predictive_modules
from active_learning_thesis.metrics import (
    evaluate_binary_classifier,
    probability_std,
    summarize_ensemble,
    vote_entropy,
)
from active_learning_thesis.paths import PREDICTIVE_DATA_DIR


FINGERPRINT_TRAINING_DEFAULTS = {
    "batch_size": 600,
    "learning_rate": 0.01,
    "dropout": 0.5,
    "lstm": 5,
    "conv": 5,
    "lambda": 0.0,
}


class _InMemoryBestWeights:
    def __init__(self, tf_module, monitor: str):
        self._tf = tf_module
        self.monitor = monitor
        self.best_value = np.inf
        self.best_epoch = -1
        self.best_weights = None

    def callback(self):
        tracker = self

        class BestWeightsCallback(tracker._tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(tracker.monitor)
                if current is None:
                    return
                current = float(current)
                if current < tracker.best_value:
                    tracker.best_value = current
                    tracker.best_epoch = epoch
                    tracker.best_weights = self.model.get_weights()

        return BestWeightsCallback()

    def restore(self, model) -> None:
        if self.best_weights is not None:
            model.set_weights(self.best_weights)


@dataclass
class ManagedModel:
    model_name: str
    seed: int
    path: Path | None
    model: object
    calibration: dict[str, float | int | str] | None = None
    embedding_model: object | None = None


@lru_cache(maxsize=1)
def _predictive_modules():
    return load_predictive_modules()


def _properties_mask() -> tuple[np.ndarray, int, int]:
    properties = np.ones(95)
    properties[0] = 0
    offset = 1
    mask_value = 2
    return properties, offset, mask_value


def _rows_to_sequences_and_labels(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    sequences = [row["sequence"] for row in rows]
    labels = [str(row["label"]) for row in rows]
    return sequences, labels


@lru_cache(maxsize=None)
def _load_ap_feature_maps(offset: int):
    utils, _, _ = _predictive_modules()
    return utils.load_data_AP(offset)


@lru_cache(maxsize=1)
def _load_tsne_feature_maps():
    return (
        np.load(
            PREDICTIVE_DATA_DIR / "TSNE_SP_1_component.npy",
            allow_pickle=True,
        ).item(),
        np.load(
            PREDICTIVE_DATA_DIR / "TSNE_SP_2_components.npy",
            allow_pickle=True,
        ).item(),
        np.load(
            PREDICTIVE_DATA_DIR / "TSNE_SP_3_components.npy",
            allow_pickle=True,
        ).item(),
    )


def _sp_fixed_length(model_name: str, max_len: int) -> int:
    if "SP" in model_name and "TSNE" not in model_name:
        return max_len + 1
    return max_len


def _masked_sp_property(
    property_values: np.ndarray,
    sequence_length: int,
    fixed_length: int,
    mask_value: int,
) -> np.ndarray:
    masked = np.full(fixed_length, mask_value, dtype=float)
    copy_length = min(len(property_values), fixed_length)
    masked[:copy_length] = property_values[:copy_length]
    masked[min(sequence_length, fixed_length) :] = mask_value
    return masked


def _build_feature_bins(
    model_name: str,
    sequences: list[str],
    labels: list[str],
):
    from seqprops import SequentialPropertiesEncoder
    from sklearn.preprocessing import MinMaxScaler

    utils, _, _ = _predictive_modules()
    if not sequences:
        return [], []

    properties, offset, mask_value = _properties_mask()
    max_len = utils.MAX_LEN

    encoded_sequences = None
    if "SP" in model_name and "TSNE" not in model_name:
        encoder = SequentialPropertiesEncoder(
            scaler=MinMaxScaler(feature_range=(-offset, offset))
        )
        encoded_sequences = np.asarray(encoder.encode(sequences), dtype=float)

    amino_acids_ap = dipeptides_ap = tripeptides_ap = None
    if "AP" in model_name:
        amino_acids_ap, dipeptides_ap, tripeptides_ap = _load_ap_feature_maps(offset)

    tsne_feature_maps = None
    if "SP" in model_name and "TSNE" in model_name:
        tsne_feature_maps = _load_tsne_feature_maps()

    sa_rows = []
    nsa_rows = []
    for index, sequence in enumerate(sequences):
        row_features = []

        if "AP" in model_name:
            ap_length = max_len + int("SP" in model_name and "TSNE" not in model_name)
            row_features.append(
                utils.padding(
                    utils.split_amino_acids(sequence, amino_acids_ap),
                    ap_length,
                    mask_value,
                )
            )
            row_features.append(
                utils.padding(
                    utils.split_dipeptides(sequence, dipeptides_ap),
                    ap_length,
                    mask_value,
                )
            )
            row_features.append(
                utils.padding(
                    utils.split_tripeptides(sequence, tripeptides_ap),
                    ap_length,
                    mask_value,
                )
            )

        if "SP" in model_name and "TSNE" not in model_name:
            other_props = np.transpose(encoded_sequences[index])
            fixed_length = _sp_fixed_length(model_name, max_len)
            selected_props = []
            for prop_index, include in enumerate(properties):
                if include != 1 or prop_index >= len(other_props):
                    continue
                selected_props.append(
                    _masked_sp_property(
                        np.asarray(other_props[prop_index], dtype=float),
                        len(sequence),
                        fixed_length,
                        mask_value,
                    )
                )
            if "AP" in model_name:
                row_features.extend(selected_props)
            else:
                row_features = np.transpose(np.asarray(selected_props, dtype=float))

        if "SP" in model_name and "TSNE" in model_name:
            for feature_map in tsne_feature_maps:
                row_features.append(
                    utils.padding(
                        utils.split_amino_acids(sequence, feature_map),
                        max_len,
                        mask_value,
                    )
                )
            if "AP" not in model_name:
                row_features = np.asarray(row_features, dtype=float)

        if str(labels[index]) == "1":
            sa_rows.append(row_features)
        elif str(labels[index]) == "0":
            nsa_rows.append(row_features)

    return sa_rows, nsa_rows


@lru_cache(maxsize=128)
def _prepare_training_tensors_cached(
    model_name: str,
    sequences: tuple[str, ...],
    labels: tuple[str, ...],
):
    utils, _, _ = _predictive_modules()
    sa_data, nsa_data = _build_feature_bins(model_name, list(sequences), list(labels))
    all_data, all_labels = utils.merge_data(sa_data, nsa_data)
    return utils.reshape_for_model(model_name, all_data, all_labels)


def _prepare_training_tensors(model_name: str, rows: list[dict[str, str]]):
    sequences, labels = _rows_to_sequences_and_labels(rows)
    return _prepare_training_tensors_cached(
        model_name,
        tuple(sequences),
        tuple(labels),
    )


@lru_cache(maxsize=256)
def _prepare_inference_tensors_cached(
    model_name: str,
    sequences: tuple[str, ...],
):
    utils, _, _ = _predictive_modules()
    dummy_labels = ["1" for _ in sequences]
    sa_data, nsa_data = _build_feature_bins(model_name, list(sequences), dummy_labels)
    all_data, all_labels = utils.merge_data(sa_data, nsa_data)
    model_inputs, _ = utils.reshape_for_model(model_name, all_data, all_labels)
    return model_inputs


def _prepare_inference_tensors(model_name: str, sequences: list[str]):
    return _prepare_inference_tensors_cached(model_name, tuple(sequences))


def _build_model(model_name: str, model_inputs, config: RunConfig):
    _, models_module, automate_training = _predictive_modules()
    num_cells = config.model_num_cells[model_name]
    kernel_size = config.model_kernel_size[model_name]
    _, _, mask_value = _properties_mask()

    if "AP" in model_name and "SP" in model_name:
        return models_module.amino_di_tri_model(
            input_shape=np.shape(model_inputs[3][0]),
            conv=automate_training.CONV,
            numcells=num_cells,
            kernel_size=kernel_size,
            lstm1=automate_training.LSTM,
            lstm2=automate_training.LSTM,
            dense=2 * num_cells,
            dropout=automate_training.DROPOUT,
            lambda2=automate_training.LAMBDA,
            mask_value=mask_value,
        )
    if "AP" in model_name and "SP" not in model_name:
        return models_module.only_amino_di_tri_model(
            lstm1=automate_training.LSTM,
            lstm2=automate_training.LSTM,
            dense=2 * num_cells,
            dropout=automate_training.DROPOUT,
            lambda2=automate_training.LAMBDA,
            mask_value=mask_value,
        )
    return models_module.create_seq_model(
        input_shape=np.shape(model_inputs[0]),
        conv1_filters=automate_training.CONV,
        conv2_filters=automate_training.CONV,
        conv_kernel_size=kernel_size,
        num_cells=num_cells,
        dropout=automate_training.DROPOUT,
        mask_value=mask_value,
    )


def _class_weight(rows: list[dict[str, str]]) -> dict[int, float]:
    positives = sum(1 for row in rows if str(row["label"]) == "1")
    negatives = sum(1 for row in rows if str(row["label"]) == "0")
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    return {0: positives / negatives, 1: 1.0}


class CalibrationInputError(ValueError):
    """Raised when calibration inputs are structurally invalid."""


def _identity_calibration(
    *,
    method: str = "identity",
    reason: str = "",
    sample_count: int = 0,
    positive_count: int = 0,
    logit_std: float | None = None,
    fitting_attempted: bool = False,
) -> dict[str, float | int | str | bool | None]:
    return {
        "method": method,
        "coef": 1.0,
        "intercept": 0.0,
        "center": 0.0,
        "scale": 1.0,
        "sample_count": int(sample_count),
        "positive_count": int(positive_count),
        "fallback_used": True,
        "fallback_reason": reason,
        "validation_logit_std": logit_std,
        "fitting_attempted": bool(fitting_attempted),
        "probability_clipping_epsilon": 1e-6,
    }


def _sigmoid(values) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _logit(probabilities, eps: float = 1e-6) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return np.log(probs / (1.0 - probs))


def _fit_platt_calibration(
    probabilities,
    labels,
    config: RunConfig,
) -> dict[str, float | int | str]:
    if probabilities is None or labels is None:
        raise CalibrationInputError("Calibration predictions and labels are required.")
    raw_probs = np.asarray(probabilities, dtype=float)
    raw_labels = np.asarray(labels, dtype=float)
    if raw_probs.ndim != 1 or raw_labels.ndim != 1:
        raise CalibrationInputError("Calibration predictions and labels must be one-dimensional.")
    if raw_probs.size == 0 or raw_labels.size == 0:
        raise CalibrationInputError("Calibration predictions and labels must not be empty.")
    if raw_probs.size != raw_labels.size:
        raise CalibrationInputError(
            "Calibration predictions and labels must contain the same number of values."
        )
    if not np.isfinite(raw_probs).all() or not np.isfinite(raw_labels).all():
        raise CalibrationInputError("Calibration predictions and labels must be finite.")
    if not np.isin(raw_labels, [0.0, 1.0]).all():
        raise CalibrationInputError("Calibration labels must be binary values 0 or 1.")

    labels_array = raw_labels.reshape(-1)
    probs = np.clip(raw_probs.reshape(-1), 1e-6, 1 - 1e-6)
    positives = int(np.sum(labels_array == 1))
    negatives = int(np.sum(labels_array == 0))
    if positives == 0 or negatives == 0:
        return _identity_calibration(
            method="identity_single_class",
            reason="validation_labels_single_class",
            sample_count=len(labels_array),
            positive_count=positives,
            fitting_attempted=False,
        )

    logits = _logit(probs)
    center = float(np.mean(logits))
    scale = float(np.std(logits))
    if scale < 1e-6:
        return _identity_calibration(
            method="identity_low_logit_variation",
            reason="validation_logit_sd_below_1e-6",
            sample_count=len(labels_array),
            positive_count=positives,
            logit_std=scale,
            fitting_attempted=False,
        )
    features = (logits - center) / scale

    coef = 1.0
    intercept = 0.0
    learning_rate = float(config.calibration_learning_rate)
    l2 = float(config.calibration_l2)
    try:
        for iteration in range(1, int(config.calibration_max_iter) + 1):
            predictions = _sigmoid(coef * features + intercept)
            residual = predictions - labels_array
            grad_coef = float(np.mean(residual * features) + l2 * (coef - 1.0))
            grad_intercept = float(np.mean(residual) + l2 * intercept)
            coef -= learning_rate * grad_coef
            intercept -= learning_rate * grad_intercept
            if abs(grad_coef) + abs(grad_intercept) < 1e-8:
                break
    except Exception as exc:
        return _identity_calibration(
            method="identity_calibration_failed",
            reason=f"{type(exc).__name__}: {exc}",
            sample_count=len(labels_array),
            positive_count=positives,
            logit_std=scale,
            fitting_attempted=True,
        )
    if not np.isfinite([coef, intercept]).all():
        return _identity_calibration(
            method="identity_non_finite_parameters",
            reason="fitted_platt_parameters_non_finite",
            sample_count=len(labels_array),
            positive_count=positives,
            logit_std=scale,
            fitting_attempted=True,
        )

    return {
        "method": "platt_logit",
        "coef": float(coef),
        "intercept": float(intercept),
        "center": center,
        "scale": scale,
        "sample_count": int(len(labels_array)),
        "positive_count": positives,
        "fallback_used": False,
        "fallback_reason": "",
        "validation_logit_std": scale,
        "fitting_attempted": True,
        "iterations": iteration,
        "learning_rate": learning_rate,
        "l2": l2,
        "max_iterations": int(config.calibration_max_iter),
        "probability_clipping_epsilon": 1e-6,
    }


def _apply_calibration(probabilities, calibration: dict | None) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if not calibration or str(calibration.get("method", "identity")).startswith("identity"):
        return probs
    logits = _logit(probs)
    center = float(calibration.get("center", 0.0) or 0.0)
    scale = float(calibration.get("scale", 1.0) or 1.0)
    if abs(scale) < 1e-12:
        scale = 1.0
    coef_value = calibration.get("coef", 1.0)
    intercept_value = calibration.get("intercept", 0.0)
    coef = 1.0 if coef_value is None else float(coef_value)
    intercept = 0.0 if intercept_value is None else float(intercept_value)
    return _sigmoid(coef * ((logits - center) / scale) + intercept)


def _row_signature(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    return sorted(
        (
            row["sequence"],
            str(row["label"]),
            row.get("label_source", ""),
        )
        for row in rows
    )


def _training_fingerprint(
    model_name: str,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    seed: int,
    config: RunConfig,
) -> str:
    training_defaults = dict(FINGERPRINT_TRAINING_DEFAULTS)
    try:
        _, _, automate_training = _predictive_modules()
        training_defaults.update(
            {
                "batch_size": automate_training.BATCH_SIZE,
                "learning_rate": automate_training.LEARNING_RATE_SET,
                "dropout": automate_training.DROPOUT,
                "lstm": automate_training.LSTM,
                "conv": automate_training.CONV,
                "lambda": automate_training.LAMBDA,
            }
        )
    except RuntimeError:
        pass
    payload = {
        "model_name": model_name,
        "seed": seed,
        "train_rows": _row_signature(train_rows),
        "validation_rows": _row_signature(validation_rows),
        "epochs": config.epochs,
        "num_cells": config.model_num_cells[model_name],
        "kernel_size": config.model_kernel_size[model_name],
        "batch_size": training_defaults["batch_size"],
        "learning_rate": training_defaults["learning_rate"],
        "dropout": training_defaults["dropout"],
        "lstm": training_defaults["lstm"],
        "conv": training_defaults["conv"],
        "lambda": training_defaults["lambda"],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _read_metadata(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_artifact(src_model: Path, src_meta: Path, dst_model: Path, dst_meta: Path) -> None:
    if src_model.resolve() != dst_model.resolve():
        dst_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_model, dst_model)
    if src_meta.exists() and src_meta.resolve() != dst_meta.resolve():
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_meta, dst_meta)


def _cache_artifact_paths(
    cache_dir: Path,
    model_name: str,
    fingerprint: str,
) -> tuple[Path, Path]:
    model_dir = cache_dir / model_name
    return model_dir / f"{fingerprint}.h5", model_dir / f"{fingerprint}.meta.json"


def _load_saved_model(
    model_name: str,
    seed: int,
    model_path: Path,
) -> ManagedModel:
    import tensorflow as tf
    metadata = _read_metadata(_metadata_path(model_path)) or {}

    return ManagedModel(
        model_name=model_name,
        seed=seed,
        path=model_path,
        model=tf.keras.models.load_model(str(model_path)),
        calibration=metadata.get("calibration") if isinstance(metadata.get("calibration"), dict) else None,
    )


def _try_load_reusable_model(
    model_name: str,
    seed: int,
    output_path: Path | None,
    cache_dir: Path | None,
    fingerprint: str,
) -> ManagedModel | None:
    if output_path is not None and output_path.exists():
        metadata = _read_metadata(_metadata_path(output_path))
        if metadata and metadata.get("fingerprint") == fingerprint:
            return _load_saved_model(model_name, seed, output_path)

    if cache_dir is None:
        return None

    cache_model_path, cache_meta_path = _cache_artifact_paths(
        cache_dir,
        model_name,
        fingerprint,
    )
    metadata = _read_metadata(cache_meta_path)
    if not cache_model_path.exists() or not metadata:
        return None
    if metadata.get("fingerprint") != fingerprint:
        return None

    load_path = cache_model_path
    if output_path is not None:
        _copy_artifact(
            cache_model_path,
            cache_meta_path,
            output_path,
            _metadata_path(output_path),
        )
        load_path = output_path
    return _load_saved_model(model_name, seed, load_path)


def _persist_cached_model(
    output_path: Path | None,
    cache_dir: Path | None,
    model_name: str,
    fingerprint: str,
    metadata: dict,
    model,
) -> None:
    if output_path is not None:
        _write_metadata(_metadata_path(output_path), metadata)

    if cache_dir is None:
        return

    cache_model_path, cache_meta_path = _cache_artifact_paths(
        cache_dir,
        model_name,
        fingerprint,
    )
    cache_model_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is not None and output_path.exists():
        _copy_artifact(
            output_path,
            _metadata_path(output_path),
            cache_model_path,
            cache_meta_path,
        )
    elif not cache_model_path.exists():
        model.save(str(cache_model_path))
        _write_metadata(cache_meta_path, metadata)


def train_model(
    model_name: str,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    seed: int,
    output_path: Path | None,
    config: RunConfig,
    cache_dir: Path | None = None,
) -> ManagedModel:
    import tensorflow as tf

    _, _, automate_training = _predictive_modules()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    fingerprint = _training_fingerprint(
        model_name,
        train_rows,
        validation_rows,
        seed,
        config,
    )
    cached_model = _try_load_reusable_model(
        model_name,
        seed,
        output_path,
        cache_dir,
        fingerprint,
    )
    if cached_model is not None:
        return cached_model

    train_inputs, train_labels = _prepare_training_tensors(model_name, train_rows)
    validation_inputs, validation_labels = (None, None)
    if validation_rows:
        validation_inputs, validation_labels = _prepare_training_tensors(
            model_name, validation_rows
        )

    model = _build_model(model_name, train_inputs, config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=automate_training.LEARNING_RATE_SET
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(automate_training.scheduler)
    ]
    monitor = "loss"
    fit_kwargs = {}
    if validation_rows:
        monitor = "val_loss"
        fit_kwargs["validation_data"] = (validation_inputs, validation_labels)
    best_weights = _InMemoryBestWeights(tf, monitor)
    callbacks.insert(0, best_weights.callback())
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_inputs,
        train_labels,
        epochs=config.epochs,
        batch_size=automate_training.BATCH_SIZE,
        class_weight=_class_weight(train_rows),
        callbacks=callbacks,
        verbose=0,
        **fit_kwargs,
    )

    best_weights.restore(model)
    calibration = _identity_calibration()
    if validation_rows and validation_inputs is not None and validation_labels is not None:
        calibration_labels = np.asarray(validation_labels).reshape(-1)
        if len(np.unique(calibration_labels)) < 2:
            # The shared calibration protocol immediately uses identity fallback
            # for a single-class validation set, so a model forward pass is neither
            # necessary nor informative in this degenerate case.
            raw_validation_predictions = np.full(
                calibration_labels.shape,
                0.5,
                dtype=float,
            )
        else:
            raw_validation_predictions = _batched_forward_pass(
                model,
                validation_inputs,
                automate_training.BATCH_SIZE,
            ).reshape(-1)
        calibration = _fit_platt_calibration(
            raw_validation_predictions,
            calibration_labels,
            config,
        )
    if output_path is not None:
        model.save(str(output_path))

    metadata = {
        "fingerprint": fingerprint,
        "model_name": model_name,
        "seed": seed,
        "calibration": calibration,
    }
    _persist_cached_model(
        output_path,
        cache_dir,
        model_name,
        fingerprint,
        metadata,
        model,
    )

    return ManagedModel(
        model_name=model_name,
        seed=seed,
        path=output_path,
        model=model,
        calibration=calibration,
    )


def train_ensemble(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    model_dir: Path,
    config: RunConfig,
    cache_dir: Path | None = None,
) -> list[ManagedModel]:
    members: list[ManagedModel] = []
    for member_index in range(config.ensemble_size):
        seed = config.ensemble_seeds[member_index]
        output_path = model_dir / f"ap_sp_member_{member_index:02d}.h5"
        members.append(
            train_model(
                "AP_SP",
                train_rows,
                validation_rows,
                seed,
                output_path,
                config,
                cache_dir=cache_dir,
            )
        )
    return members


def load_ensemble_from_dir(
    model_dir: Path,
    config: RunConfig,
) -> list[ManagedModel]:
    members: list[ManagedModel] = []
    for member_index in range(config.ensemble_size):
        model_path = model_dir / f"ap_sp_member_{member_index:02d}.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing ensemble member: {model_path}")
        members.append(
            _load_saved_model(
                "AP_SP",
                config.ensemble_seeds[member_index],
                model_path,
            )
        )
    return members

def train_family(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    model_dir: Path,
    config: RunConfig,
    cache_dir: Path | None = None,
) -> list[ManagedModel]:
    family_models = ["AP", "SP", "AP_SP", "TSNE_SP", "TSNE_AP_SP"]
    members: list[ManagedModel] = []
    for index, model_name in enumerate(family_models):
        output_path = model_dir / f"{model_name}.h5"
        members.append(
            train_model(
                model_name,
                train_rows,
                validation_rows,
                seed=config.random_seed + index,
                output_path=output_path,
                config=config,
                cache_dir=cache_dir,
            )
        )
    return members


def _input_length(model_inputs) -> int:
    if isinstance(model_inputs, (list, tuple)):
        return len(model_inputs[0]) if model_inputs else 0
    return len(model_inputs)


def _slice_model_inputs(model_inputs, start: int, end: int):
    if isinstance(model_inputs, tuple):
        return tuple(batch[start:end] for batch in model_inputs)
    if isinstance(model_inputs, list):
        return [batch[start:end] for batch in model_inputs]
    return model_inputs[start:end]


def _batched_forward_pass(callable_model, model_inputs, batch_size: int) -> np.ndarray:
    total = _input_length(model_inputs)
    if total == 0:
        return np.empty((0,), dtype=float)
    outputs = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_inputs = _slice_model_inputs(model_inputs, start, end)
        batch_output = callable_model(batch_inputs, training=False)
        outputs.append(np.asarray(batch_output, dtype=float))
    return np.concatenate(outputs, axis=0)


def _predict_probabilities_from_inputs(model: ManagedModel, model_inputs) -> np.ndarray:
    _, _, automate_training = _predictive_modules()
    predictions = _batched_forward_pass(
        model.model,
        model_inputs,
        automate_training.BATCH_SIZE,
    )
    return np.asarray(predictions, dtype=float).reshape(-1)


def _embedding_submodel(model: ManagedModel):
    import tensorflow as tf

    if model.embedding_model is None:
        model.embedding_model = tf.keras.Model(
            inputs=model.model.inputs,
            outputs=model.model.layers[-2].output,
        )
    return model.embedding_model


def _predict_embeddings_from_inputs(model: ManagedModel, model_inputs) -> np.ndarray:
    _, _, automate_training = _predictive_modules()
    embeddings = _batched_forward_pass(
        _embedding_submodel(model),
        model_inputs,
        automate_training.BATCH_SIZE,
    )
    return np.asarray(embeddings, dtype=float)


def extract_ap_sp_member_embeddings_strict(
    ensemble: list[ManagedModel],
    sequences: list[str],
    *,
    expected_width: int = 384,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Extract the explicit penultimate AP_SP concatenation representation."""
    if not ensemble:
        raise ValueError("AP_SP embedding extraction requires at least one ensemble member")
    member_embeddings: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    extractor_checksum = hashlib.sha256(
        inspect.getsource(extract_ap_sp_member_embeddings_strict).encode("utf-8")
    ).hexdigest()
    resolved_layers = []
    for member_index, member in enumerate(ensemble):
        if member.model_name != "AP_SP":
            raise ValueError(
                f"Embedding member {member_index} is {member.model_name}, expected AP_SP"
            )
        layers = list(getattr(member.model, "layers", []))
        if len(layers) < 2:
            raise ValueError(f"AP_SP member {member_index} has no penultimate layer")
        layer = layers[-2]
        if layer.__class__.__name__ != "Concatenate":
            raise ValueError(
                f"AP_SP member {member_index} penultimate layer {getattr(layer, 'name', '')!r} "
                "is not Concatenate"
            )
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None:
            shape_values = tuple(output_shape)
            if len(shape_values) != 2 or int(shape_values[-1]) != int(expected_width):
                raise ValueError(
                    f"AP_SP member {member_index} configured embedding shape is "
                    f"{shape_values}, expected (batch_size, {expected_width})"
                )
        resolved_layers.append(layer)

    prepared_inputs = _prepare_inference_tensors("AP_SP", sequences)
    for member_index, (member, layer) in enumerate(zip(ensemble, resolved_layers)):
        import tensorflow as tf

        extractor = tf.keras.Model(inputs=member.model.inputs, outputs=layer.output)
        embeddings = np.asarray(
            _batched_forward_pass(
                extractor,
                prepared_inputs,
                _predictive_modules()[2].BATCH_SIZE,
            ),
            dtype=float,
        )
        expected_rows = len(sequences)
        if embeddings.ndim != 2:
            raise ValueError(
                f"AP_SP member {member_index} embedding output rank is "
                f"{embeddings.ndim}, expected 2"
            )
        if embeddings.shape != (expected_rows, int(expected_width)):
            raise ValueError(
                f"AP_SP member {member_index} embedding output shape is "
                f"{embeddings.shape}, expected ({expected_rows}, {expected_width})"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"AP_SP member {member_index} embeddings contain non-finite values")

        model_checksum = ""
        if member.path is not None and Path(member.path).is_file():
            model_checksum = hashlib.sha256(Path(member.path).read_bytes()).hexdigest()
        member_embeddings.append(embeddings)
        metadata.append(
            {
                "member_index": member_index,
                "member_seed": int(member.seed),
                "model_name": member.model_name,
                "model_path": str(member.path) if member.path is not None else "",
                "model_checksum_sha256": model_checksum,
                "embedding_layer_name": str(getattr(layer, "name", "")),
                "embedding_layer_class": layer.__class__.__name__,
                "runtime_shape": [int(value) for value in embeddings.shape],
                "embedding_width": int(expected_width),
                "embedding_extractor_checksum_sha256": extractor_checksum,
            }
        )
    return member_embeddings, metadata


def score_sequences_with_ensemble(
    ensemble: list[ManagedModel],
    sequences: list[str],
    include_embeddings: bool = False,
    use_calibration: bool = True,
    include_raw: bool = False,
) -> dict[str, np.ndarray]:
    if not sequences:
        summary = summarize_ensemble(np.empty((0, len(ensemble)), dtype=float))
        summary["ensemble_member_probs"] = np.empty((0, len(ensemble)), dtype=float)
        summary["raw_ensemble_member_probs"] = np.empty((0, len(ensemble)), dtype=float)
        for key, value in list(summary.items()):
            if key.startswith("pred_"):
                summary[f"raw_{key}"] = value
        if include_embeddings:
            summary["avg_embedding"] = np.empty((0, 0), dtype=float)
        return summary

    prepared_inputs = _prepare_inference_tensors(ensemble[0].model_name, sequences)
    raw_probability_matrix = np.column_stack(
        [
            _predict_probabilities_from_inputs(member, prepared_inputs)
            for member in ensemble
        ]
    )
    probability_matrix = raw_probability_matrix
    if use_calibration:
        probability_matrix = np.column_stack(
            [
                _apply_calibration(raw_probability_matrix[:, index], member.calibration)
                for index, member in enumerate(ensemble)
            ]
        )
    summary = summarize_ensemble(probability_matrix)
    summary["ensemble_member_probs"] = probability_matrix
    if include_raw or use_calibration:
        raw_summary = summarize_ensemble(raw_probability_matrix)
        for key, value in raw_summary.items():
            summary[f"raw_{key}"] = value
        summary["raw_ensemble_member_probs"] = raw_probability_matrix
    if include_embeddings:
        embedding_stack = np.stack(
            [
                _predict_embeddings_from_inputs(member, prepared_inputs)
                for member in ensemble
            ],
            axis=0,
        )
        summary["avg_embedding"] = embedding_stack.mean(axis=0)
    return summary


def score_sequences_with_family(
    family_models: list[ManagedModel],
    sequences: list[str],
    use_calibration: bool = True,
    include_raw: bool = False,
) -> dict[str, np.ndarray]:
    prepared_inputs = {
        model_name: _prepare_inference_tensors(model_name, sequences)
        for model_name in {member.model_name for member in family_models}
    }
    raw_probability_matrix = np.column_stack(
        [
            _predict_probabilities_from_inputs(
                member,
                prepared_inputs[member.model_name],
            )
            for member in family_models
        ]
    )
    probability_matrix = raw_probability_matrix
    if use_calibration:
        probability_matrix = np.column_stack(
            [
                _apply_calibration(raw_probability_matrix[:, index], member.calibration)
                for index, member in enumerate(family_models)
            ]
        )
    summary = {
        "family_member_probs": probability_matrix,
        "committee_vote_entropy": vote_entropy(probability_matrix),
        "committee_prob_std": probability_std(probability_matrix),
    }
    if include_raw or use_calibration:
        summary["raw_family_member_probs"] = raw_probability_matrix
    return summary


def evaluate_rows(
    ensemble: list[ManagedModel],
    rows: list[dict[str, str]],
    use_calibration: bool = True,
    threshold_strategy: str = "fixed_0_5",
    threshold: float | None = None,
    threshold_source: str = "evaluation_dataset",
    threshold_selection_f1: float | None = None,
) -> dict[str, float | str]:
    sequences, labels = _rows_to_sequences_and_labels(rows)
    scored = score_sequences_with_ensemble(
        ensemble,
        sequences,
        include_embeddings=False,
        use_calibration=use_calibration,
    )
    return evaluate_binary_classifier(
        labels,
        scored["pred_mean"],
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        threshold_source=threshold_source,
        threshold_selection_f1=threshold_selection_f1,
    )


def evaluate_holdout(
    ensemble: list[ManagedModel],
    holdout_rows: list[dict[str, str]],
    use_calibration: bool = True,
    threshold_strategy: str = "fixed_0_5",
    threshold: float | None = None,
    threshold_source: str = "holdout",
    threshold_selection_f1: float | None = None,
) -> dict[str, float | str]:
    return evaluate_rows(
        ensemble,
        holdout_rows,
        use_calibration=use_calibration,
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        threshold_source=threshold_source,
        threshold_selection_f1=threshold_selection_f1,
    )
