from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATIVE_DIR = Path(__file__).resolve().parent
PREDICTIVE_CODE_DIR = PROJECT_ROOT / "SA_ML_predictive" / "code"
for candidate in (PROJECT_ROOT, GENERATIVE_DIR, PREDICTIVE_CODE_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)


###############################################################################
RANDOM_SEED = 9879

ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PREFERRED_LENGTH_RANGE = [5, 10]

MIN_INITIAL_PEPTIDE_LENGTH = 3
MAX_INITIAL_PEPTIDE_LENGTH = 24
POPULATION_SIZE = 50
OFFSPRING_COUNT = 30
MAX_NUM_GENERATIONS = 30
TOURNAMENT_SIZE = 3
MUTATION_PROBABILITY = 0.05
###############################################################################

PRETRAINED_MODELS = ["AP", "AP_SP", "SP", "TSNE_AP_SP", "TSNE_SP"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate novel peptide suggestions with the GA-guided generative model.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ml-model",
        choices=PRETRAINED_MODELS,
        help="Name of a pre-trained model in SA_ML_predictive/models.",
    )
    group.add_argument(
        "--ml-model-path",
        help="Path to a saved .h5 predictive model.",
    )
    return parser


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if args.ml_model:
        model_path = PROJECT_ROOT / "SA_ML_predictive" / "models" / f"{args.ml_model}.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                "Can't find the pre-trained model in the 'SA_ML_predictive/models' folder. "
                "Try running the code from the project's root directory ('ml_peptide_self_assembly')."
            )
        return model_path

    model_path = Path(args.ml_model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"The model at {model_path} does not exist.")
    return model_path


def _model_type_from_path(model_path: Path) -> str:
    model_type = model_path.stem
    if model_type not in PRETRAINED_MODELS:
        raise ValueError(
            "Unable to infer the predictive model type from the model filename. "
            "Expected the filename stem to be one of: " + ", ".join(PRETRAINED_MODELS)
        )
    return model_type


def predict_SA_probability(sequence, model_path: Path, sa_ml_model):
    import numpy as np

    from SA_ML_predictive.code import automate_training, utils

    if len(sequence) > utils.MAX_LEN:
        raise Exception("Peptide is too long.")
    if len(sequence) < 1:
        raise Exception("No peptide for prediction.")

    pep_list = [sequence, "A" * utils.MAX_LEN]
    pep_labels = ["1", "1"]

    offset = 1
    properties = np.ones(95)
    properties[0] = 0
    mask_value = 2
    model_type = _model_type_from_path(model_path)

    sa_rows, nsa_rows = utils.load_data(
        model_type,
        [pep_list, pep_labels],
        offset,
        properties,
        mask_value,
    )
    all_data, all_labels = utils.merge_data(sa_rows, nsa_rows)
    test_data, test_labels = utils.reshape_for_model(model_type, all_data, all_labels)
    model_predictions = sa_ml_model.predict(test_data, batch_size=automate_training.BATCH_SIZE)
    model_predictions = utils.convert_list(model_predictions)
    return model_predictions[0]


def calculate_amino_acid_frequencies(sequence):
    import numpy as np

    frequencies = []

    for amino_acid in ALLOWED_AMINO_ACIDS:
        frequencies.append(sequence.count(amino_acid))

    return np.array(frequencies)


def calculate_similarity_penalty(sequence, population):
    import numpy as np

    if len(population) <= 1:
        return 0

    penalty = 0
    first = True
    frequencies = calculate_amino_acid_frequencies(sequence)

    for neighbour_peptide in population:
        if neighbour_peptide.sequence == sequence and first:
            first = False
            continue

        neighbour_frequencies = calculate_amino_acid_frequencies(neighbour_peptide.sequence)
        denominator = np.sum(frequencies) + np.sum(neighbour_frequencies)
        if denominator == 0:
            continue
        penalty += 0.1 * (1 - np.sum(np.abs(frequencies - neighbour_frequencies)) / denominator)

    return penalty / (len(population) - 1)


def calculate_length_penalty(sequence):
    if len(sequence) < PREFERRED_LENGTH_RANGE[0] or len(sequence) > PREFERRED_LENGTH_RANGE[1]:
        return min(0.05 * abs(len(sequence) - (PREFERRED_LENGTH_RANGE[0] + PREFERRED_LENGTH_RANGE[1]) / 2), 0.5)
    return 0


def main(argv: list[str] | None = None) -> int:
    from active_learning_thesis.dependencies import ensure_predictive_runtime

    ensure_predictive_runtime()

    import numpy as np
    import tensorflow as tf

    from genetic_algorithm_library import GeneticAlgorithm

    parser = _build_parser()
    args = parser.parse_args(argv)

    np.random.seed(RANDOM_SEED)
    model_path = _resolve_model_path(args)
    sa_ml_model = tf.keras.models.load_model(str(model_path))

    ga = GeneticAlgorithm(
        fitness_function=lambda sequence: predict_SA_probability(sequence, model_path, sa_ml_model),
        similarity_penalty=calculate_similarity_penalty,
        length_penalty=calculate_length_penalty,
        min_initial_peptide_length=MIN_INITIAL_PEPTIDE_LENGTH,
        max_initial_peptide_length=MAX_INITIAL_PEPTIDE_LENGTH,
        allowed_amino_acids=ALLOWED_AMINO_ACIDS,
        population_size=POPULATION_SIZE,
        offspring_count=OFFSPRING_COUNT,
        max_num_generations=MAX_NUM_GENERATIONS,
        tournament_size=TOURNAMENT_SIZE,
        mutation_probability=MUTATION_PROBABILITY,
    )

    final_population = ga.find_peptides()
    best_sequences = list(set([peptide.sequence for peptide in final_population]))
    suggested_sequences = []

    for sequence in best_sequences:
        sa_probability = predict_SA_probability(sequence, model_path, sa_ml_model)
        suggested_sequences.append([sequence, sa_probability])

    suggested_sequences = sorted(
        suggested_sequences, key=lambda list_member: list_member[1], reverse=True
    )

    print_list = ["Peptide,Self-assembly probability [%]\n"]
    print_list += [
        f"{sequence},{np.round(sa_probability * 100, decimals=1)}%\n"
        for sequence, sa_probability in suggested_sequences
    ]

    output_path = Path(
        f"suggested_SA_peptides_{PREFERRED_LENGTH_RANGE[0]}_{PREFERRED_LENGTH_RANGE[1]}.csv"
    )
    with output_path.open("w", encoding="utf-8") as file:
        file.writelines(print_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
