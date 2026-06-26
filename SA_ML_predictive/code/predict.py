import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the self-assembly probability for a single peptide.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Peptide sequence to score.",
    )
    parser.add_argument(
        "--ml-model",
        default="AP_SP",
        help="Predictive model to use. Defaults to AP_SP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from active_learning_thesis.dependencies import ensure_predictive_runtime

    ensure_predictive_runtime()

    import numpy as np
    import tensorflow as tf

    from automate_training import BATCH_SIZE
    from utils import (
        ALL_MODELS,
        MAX_LEN,
        MODELS_PATH,
        convert_list,
        load_data,
        merge_data,
        reshape_for_model,
    )

    sequence = args.sequence.strip()
    if not sequence:
        print("No peptide")
        return 1
    if len(sequence) > MAX_LEN:
        print(
            f"Peptide {sequence} is too long, the maximum peptide lenght for the model is {MAX_LEN}"
        )
        return 1
    if args.ml_model not in ALL_MODELS:
        parser.error(
            "argument --ml-model: invalid choice: "
            f"{args.ml_model!r} (choose from {', '.join(ALL_MODELS)})"
        )

    pep_list = [sequence, "A" * MAX_LEN]
    pep_labels = ["1", "1"]
    offset = 1
    properties = np.ones(95)
    properties[0] = 0
    mask_value = 2

    sa_rows, nsa_rows = load_data(
        args.ml_model,
        [pep_list, pep_labels],
        offset,
        properties,
        mask_value,
    )
    all_data, all_labels = merge_data(sa_rows, nsa_rows)

    best_model = tf.keras.models.load_model(MODELS_PATH + args.ml_model + ".h5")
    test_data, test_labels = reshape_for_model(args.ml_model, all_data, all_labels)
    model_predictions = best_model.predict(test_data, batch_size=BATCH_SIZE)
    model_predictions = convert_list(model_predictions)
    print(model_predictions[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
