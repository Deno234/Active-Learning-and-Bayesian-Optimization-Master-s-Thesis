import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from active_learning_thesis.dependencies import ensure_predictive_runtime

ensure_predictive_runtime()

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from automate_training import BATCH_SIZE, hyperparameter_kernel_size, hyperparameter_numcells
from utils import TMP_MODELS_PATH, convert_list, data_and_labels_from_indices, reshape_for_model

PREDS_PATH = Path(__file__).resolve().parents[1] / "predictions"


def predict_for_validation(model_name, test_number, train_and_validation_data, train_and_validation_labels, kfold_second):
    params_nr = 0

    tmp_model_dir = Path(TMP_MODELS_PATH) / model_name
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(
        train_and_validation_data, train_and_validation_labels
    ):
        indices.append([train_data_indices, validation_data_indices])

    prediction_dir = PREDS_PATH / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for numcells in hyperparameter_numcells:
        for kernel in hyperparameter_kernel_size:
            params_nr += 1
            for i, pair in enumerate(indices):
                model_path = (
                    tmp_model_dir
                    / f"{model_name}_test_{test_number}_fold_{i + 1}_params_{params_nr}_num_cells_{numcells}_kernel_size_{kernel}.h5"
                )
                model_used = tf.keras.models.load_model(str(model_path))

                validation_data_indices = pair[1]
                val_data, val_labels = data_and_labels_from_indices(
                    train_and_validation_data,
                    train_and_validation_labels,
                    validation_data_indices,
                )

                val_data, val_labels = reshape_for_model(model_name, val_data, val_labels)
                model_predictions = model_used.predict(val_data, batch_size=BATCH_SIZE)
                model_predictions = convert_list(model_predictions)

                file_predictions = (
                    prediction_dir
                    / f"predictions_{model_name}_test_{test_number}_fold_{i + 1}_params_{params_nr}_num_cells_{numcells}_kernel_size_{kernel}.txt"
                )
                file_predictions.write_text(str(model_predictions), encoding="utf-8")

                file_labels = (
                    prediction_dir
                    / f"labels_{model_name}_test_{test_number}_fold_{i + 1}_params_{params_nr}_num_cells_{numcells}_kernel_size_{kernel}.txt"
                )
                file_labels.write_text(str(list(val_labels)), encoding="utf-8")


def weird_division(n, d):
    return n / d if d else 0


def convert_to_binary(model_predictions, threshold=0.5):
    model_predictions_binary = []

    for x in model_predictions:
        if x >= threshold:
            model_predictions_binary.append(1.0)
        else:
            model_predictions_binary.append(0.0)

    return model_predictions_binary


def return_GMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1

    return np.sqrt(tp / apo * tn / ane)


def my_accuracy_calculate(test_labels, model_predictions, threshold=0.5):
    score = 0

    model_predictions = convert_to_binary(model_predictions, threshold)

    for i in range(len(test_labels)):
        if model_predictions[i] == test_labels[i]:
            score += 1

    return score / len(test_labels) * 100


def read_PR_ROC(model_name, numcells, kernel):
    test_labels = []
    model_predictions = []
    print("%s num_cells: %d kernel_size: %d" % (model_name, numcells, kernel))
    model_dir = PREDS_PATH / model_name
    for model_predictions_path in model_dir.iterdir():
        model_predictions_file = model_predictions_path.name
        if (
            "predictions_" in model_predictions_file
            and f"_num_cells_{numcells}_kernel_size_{kernel}" in model_predictions_file
        ):
            model_predictions_one_part = ast.literal_eval(
                model_predictions_path.read_text(encoding="utf-8").strip()
            )
            model_predictions.extend(model_predictions_one_part)
            labels_path = model_dir / model_predictions_file.replace("predictions", "labels")
            model_labels_one_part = ast.literal_eval(
                labels_path.read_text(encoding="utf-8").strip()
            )
            for l in model_labels_one_part:
                test_labels.append(int(l))

    precision, recall, thresholdsPR = precision_recall_curve(test_labels, model_predictions)
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

    fscore = []
    for i in range(len(precision)):
        fscore.append(weird_division(2 * precision[i] * recall[i], precision[i] + recall[i]))

    ixPR = np.argmax(fscore)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ixROC = np.argmax(gmeans)

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, thresholdsPR[ixPR])
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, thresholdsROC[ixROC])
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    print("PR thr", thresholdsPR[ixPR])
    print("PR AUC", auc(recall, precision))

    print("ROC thr", thresholdsROC[ixROC])
    print("ROC AUC", roc_auc_score(test_labels, model_predictions))

    print("F1 (0.5)", f1_score(test_labels, model_predictions_binary))
    print("F1 (PR thr)", f1_score(test_labels, model_predictions_binary_thrPR_new))
    print("F1 (ROC thr)", f1_score(test_labels, model_predictions_binary_thrROC_new))

    print("gmean (0.5)", return_GMEAN(test_labels, model_predictions_binary))
    print("gmean (PR thr)", return_GMEAN(test_labels, model_predictions_binary_thrPR_new))
    print("gmean (ROC thr)", return_GMEAN(test_labels, model_predictions_binary_thrROC_new))

    print("Accuracy (0.5)", my_accuracy_calculate(test_labels, model_predictions, 0.5))
    print("Accuracy (PR thr)", my_accuracy_calculate(test_labels, model_predictions, thresholdsPR[ixPR]))
    print("Accuracy (ROC thr)", my_accuracy_calculate(test_labels, model_predictions, thresholdsROC[ixROC]))
