# Exact Runtime Tensor Shapes for the Reproduced Model Families

The shapes below were verified on 20 June 2026 by executing the actual preprocessing and model-construction pipeline with the pinned scientific environment:

- Python 3.10 environment: `F:\Anaconda\envs\ml_peptide_self_assembly\python.exe`
- TensorFlow 2.10.1
- `seqprops` from the same environment
- eight real rows from `SA_ML_predictive/data/data_SA.csv`

Three different shape levels must be distinguished:

1. **per-peptide storage shape** created by `_build_feature_bins`;
2. **batched NumPy shape** returned by `_prepare_training_tensors` and passed to `model.fit`/inference;
3. **effective Keras input shape** after Keras standardizes inputs against the declared `Input` tensors.

## A. Thesis-ready table

| Model family | Input branch and order | Description | Runtime shape excluding batch | Example batch shape (`B=8`) | Padding length | Channels | Notes |
|---|---|---|---:|---:|---:|---:|---|
| AP | 1. amino-acid AP | One AP scalar per residue | `(24, 1)` | NumPy supplied as `(8, 24)`; effective Keras tensor `(8, 24, 1)` | 24 | 1 | Keras automatically adds the singleton channel required by `Input((None, 1))`. |
| AP | 2. dipeptide AP | One AP scalar per overlapping dipeptide, then padding | `(24, 1)` | `(8, 24)` -> `(8, 24, 1)` | 24 | 1 | Input order is fixed after the amino-acid branch. |
| AP | 3. tripeptide AP | One AP scalar per overlapping tripeptide, then padding | `(24, 1)` | `(8, 24)` -> `(8, 24, 1)` | 24 | 1 | Third and final AP-only input. |
| SP | 1. physicochemical tensor | 94 selected sequential-property channels | `(25, 94)` | `(8, 25, 94)` | 25 | 94 | Stored and passed as `(length, channels)`; no transpose occurs after batching. |
| AP_SP | 1. amino-acid AP | AP scalar sequence | `(25, 1)` | `(8, 25)` -> `(8, 25, 1)` | 25 | 1 | AP padding is 25 here because non-TSNE SP is also present. |
| AP_SP | 2. dipeptide AP | Dipeptide AP scalar sequence | `(25, 1)` | `(8, 25)` -> `(8, 25, 1)` | 25 | 1 | Same AP recurrent submodel as AP, but one extra padded position. |
| AP_SP | 3. tripeptide AP | Tripeptide AP scalar sequence | `(25, 1)` | `(8, 25)` -> `(8, 25, 1)` | 25 | 1 | Third AP branch. |
| AP_SP | 4. physicochemical tensor | 94 selected sequential-property channels | `(25, 94)` | `(8, 25, 94)` | 25 | 94 | Stored as 94 property arrays per peptide, then transposed during `reshape_for_model` to `(length, channels)`. |
| TSNE_SP | 1. t-SNE tensor | Three precomputed t-SNE-derived arrays, each padded to 24 | `(3, 24)` | `(8, 3, 24)` | 24 within each component | 24 as interpreted by Conv1D | **No transpose occurs.** Keras therefore interprets dimension 3 as sequence length/timesteps and dimension 24 as channels. |
| TSNE_AP_SP | 1. amino-acid AP | AP scalar sequence | `(24, 1)` | `(8, 24)` -> `(8, 24, 1)` | 24 | 1 | TSNE models do not trigger the AP length-25 rule. |
| TSNE_AP_SP | 2. dipeptide AP | Dipeptide AP scalar sequence | `(24, 1)` | `(8, 24)` -> `(8, 24, 1)` | 24 | 1 | Second input. |
| TSNE_AP_SP | 3. tripeptide AP | Tripeptide AP scalar sequence | `(24, 1)` | `(8, 24)` -> `(8, 24, 1)` | 24 | 1 | Third input. |
| TSNE_AP_SP | 4. t-SNE tensor | Three t-SNE-derived values per residue | `(24, 3)` | `(8, 24, 3)` | 24 | 3 | The three stored `(24,)` components are explicitly transposed during `reshape_for_model` to `(length, channels)`. |

All five families use masking value `2`. AP, physicochemical, and t-SNE feature values are scaled or stored within the model feature range, while `2` is reserved for padding.

## Exact input order

### AP

```text
[
  amino_acid_AP,
  dipeptide_AP,
  tripeptide_AP
]
```

### SP

```text
physicochemical_sequence_tensor
```

### AP_SP

```text
[
  amino_acid_AP,
  dipeptide_AP,
  tripeptide_AP,
  physicochemical_sequence_tensor
]
```

### TSNE_SP

```text
tsne_tensor_stored_as_(3_components, 24_positions)
```

### TSNE_AP_SP

```text
[
  amino_acid_AP,
  dipeptide_AP,
  tripeptide_AP,
  tsne_tensor_transposed_to_(24_positions, 3_components)
]
```

## Verified per-peptide and batched storage

| Model | Per-peptide storage created before merge/reshape | Batched object passed to the model |
|---|---|---|
| AP | `(3, 24)` | three arrays `(B, 24)` |
| SP | `(25, 94)` | one array `(B, 25, 94)` |
| AP_SP | `(97, 25)` = 3 AP arrays + 94 property arrays | three arrays `(B, 25)` and one `(B, 25, 94)` |
| TSNE_SP | `(3, 24)` | one array `(B, 3, 24)` |
| TSNE_AP_SP | `(6, 24)` = 3 AP arrays + 3 t-SNE arrays | three arrays `(B, 24)` and one `(B, 24, 3)` |

The AP NumPy arrays are rank 2, but the Keras inputs are declared as `(None, 1)`. A runtime probe model showed that TensorFlow/Keras 2.10.1 standardizes:

```text
(B, 24) -> (B, 24, 1)
(B, 25) -> (B, 25, 1)
```

before the masking/LSTM branch processes them.

## Phase 1 versus adaptive training

There is **no preprocessing-shape difference** between Phase 1 reproduction and later adaptive AP_SP ensemble training:

- Phase 1 `train_nested_cv_models` and `run_train_final` call `predictive.train_model`.
- Phase 2 and Phase 3 call `predictive.train_ensemble`.
- `train_ensemble` calls the same `predictive.train_model`.
- `train_model` always calls the same `_prepare_training_tensors`, `_build_feature_bins`, `utils.reshape_for_model`, and `_build_model`.

Therefore every adaptive AP_SP member receives:

```text
[
  (B, 25) -> effective (B, 25, 1),
  (B, 25) -> effective (B, 25, 1),
  (B, 25) -> effective (B, 25, 1),
  (B, 25, 94)
]
```

Only the batch size, training examples, seed, and frozen architecture parameters change.

## Assessment of the previous summary statement

Original statement:

> AP uses three branches approximately shaped `(24, 1)`, SP uses `(25, 94)`, AP_SP uses three AP branches plus one SP tensor, TSNE_SP uses a three-component t-SNE representation, and TSNE_AP_SP uses three AP branches plus the t-SNE-derived tensor.

This is directionally correct but incomplete and should be replaced with:

> The AP model receives three ordered scalar sequence branches - amino-acid, dipeptide, and tripeptide aggregation-propensity channels - each padded to 24 positions and processed as `(24, 1)`. The SP model receives a 94-channel physicochemical tensor shaped `(25, 94)`. AP_SP receives three AP branches padded to 25 positions, each processed as `(25, 1)`, followed by an SP tensor `(25, 94)`. TSNE_SP receives a single tensor shaped `(3, 24)` in the implemented runtime, so Conv1D interprets three timesteps with 24 channels. TSNE_AP_SP receives three AP branches `(24, 1)` followed by a separately transposed t-SNE tensor `(24, 3)`.

The phrase "three-component t-SNE representation" is conceptually true for both t-SNE families, but it hides the important orientation difference between `TSNE_SP` and `TSNE_AP_SP`.

## B. Appendix / code-reproducibility details

### Files and functions inspected

- `active_learning_thesis/predictive.py`
  - `_properties_mask`
  - `_sp_fixed_length`
  - `_masked_sp_property`
  - `_build_feature_bins`
  - `_prepare_training_tensors`
  - `_prepare_inference_tensors`
  - `_build_model`
  - `train_model`
  - `train_ensemble`
- `SA_ML_predictive/code/utils.py`
  - `MAX_LEN`
  - `padding`
  - `load_data_AP`
  - `load_data`
  - `reshape_for_model`
- `SA_ML_predictive/code/models.py`
  - `_one_prop_model`
  - `_create_seq_model`
  - `only_amino_di_tri_model`
  - `create_seq_model`
  - `amino_di_tri_model`
- `active_learning_thesis/phase1_reproduction.py`
  - `train_nested_cv_models`
  - `run_train_final`
  - `write_preprocessing_shapes`
- `thesis_results/01_reproduction/tables/preprocessing_shapes.csv`
  - present, but its heavy runtime preprocessing was skipped, so it did not contain usable shapes.

No retained `.h5` model files or saved Keras summaries were present in the local Phase 1/Phase 3 result copy. Shapes were therefore verified by constructing the exact models and tensors using the pinned TensorFlow 2.10.1 environment.

### Reproduction snippet

Run from the repository root:

```python
import numpy as np
import tensorflow as tf

from active_learning_thesis.dataset import read_experimental_dataset
from active_learning_thesis.config import RunConfig
from active_learning_thesis import predictive

rows = read_experimental_dataset()[:8]
config = RunConfig()

for name in ["AP", "SP", "AP_SP", "TSNE_SP", "TSNE_AP_SP"]:
    inputs, labels = predictive._prepare_training_tensors(name, rows)
    model = predictive._build_model(name, inputs, config)
    arrays = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]

    print("\n", name)
    print("NumPy:", [np.asarray(x).shape for x in arrays])
    print("Keras declared:", model.input_shape)

    # Probe the effective tensors after Keras input standardization.
    probe = tf.keras.Model(inputs=model.inputs, outputs=model.inputs)
    standardized = probe(inputs, training=False)
    standardized = standardized if isinstance(standardized, list) else [standardized]
    print("Keras effective:", [tuple(x.shape) for x in standardized])
```

For the project environment on the inspected workstation:

```powershell
$code | F:\Anaconda\envs\ml_peptide_self_assembly\python.exe -
```

### Confirmed mismatch

`TSNE_SP` and the t-SNE branch of `TSNE_AP_SP` do not share the same orientation:

```text
TSNE_SP:       (B, 3, 24)
TSNE_AP_SP:    (B, 24, 3)
```

This is not an inference; it was observed from the actual preprocessing output and Keras input signatures. Whether the standalone `TSNE_SP` orientation was scientifically intended by the original authors is **needs confirmation**. The thesis should describe the implemented orientation factually and avoid silently rewriting it as `(24, 3)`.
